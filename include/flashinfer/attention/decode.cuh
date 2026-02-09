/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz) {
  float m_prev = st.m; // max(qk*sm_scale) = max(qk/sqrt(head_dim))
#pragma unroll
  // tile_size = bdy * tile_size_per_bdx = group_size * tile_size_per_bdx
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      // 一个 thread 一共 load group_size * tile_size_per_bdx 个 vec，对应了 float* s 的长度
      // 同一组 (ty, tz) 的 bdx 个线程加载的是同一个 token 的 k（的不同 vec）
      // 同样 tx 的线程，如果 tz 不同，加载的就是 k_smem 上不同段的数据，对应了不同的 token，compute_qk() 传入的 smem 参数本身自带 offset，这个 offset 只和 tz 相关
      // 一个 threadblock 一共加载 (group_size * tile_size_per_bdx) * bdy * bdz 个 token
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i]; // QK^T[i][j] = ith row vec of Q * jth row vec of K
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      // parallell reduction，warp 中 lane 0（即 threadIdx.x % 32 == 0 的线程）的 s[j] 包含整个 warp 32 个线程的 s[j] 的累加和
      // bdx 是2的幂，并且不超过 warp size，所以实际上做的就是把一个 bdx 上的 s[j] 加到 threadIdx.x = 0 的线程的 s[j] 上，形成一个完整的 qk 向量点积结果
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx); // alibi / logits cap
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2; // qk result * sm scale * log2(e), log2(e) 是为了换底数
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx); // custom mask / sliding window, 如果都没有则为 true
    // iter_base: 当前 tile (bdy*bdz*tile_size_per_bdx) 在全局 k/v data 的 token index，对整个 block 都是相等的
    // + tz * tile_size + j = tz * (bdy*tile_size_per_bdx) + j: token 在 tile 内的 offset，因为一个 block 是 bdy * tile_size_per_bdx 个 token 所以 bdz 个 线程 一起处理一个 tile (bdy*bdz*tile_size_per_bdx) 的 token
    // token index 只和 tz 有关，因为相同 (tx, ty) 线程在这个函数里分到的 smem 是同一个指针
    // TODO：为什么要一个线程去循环 group_size * tile_size_per_bdx 呢？这样不是浪费并行度吗？
    // iter_bound: kv seq len (if no split kv)
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf; // causal mask (?)
    st.m = max(st.m, s[j]); // max pre-softmax val
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__device__ __forceinline__ void sync_state(AttentionVariant variant, state_t<vec_size>& st,
                                           float* smem, float* smem_md, const uint32_t tx,
                                           const uint32_t ty, const uint32_t tz) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param head_dim A integer indicates the head dimension
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void SingleDecodeWithKVCacheKernel(const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const DTypeQ* q = params.q;
  const DTypeKV* k = params.k;
  const DTypeKV* v = params.v;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t kv_stride_n = params.kv_stride_n;
  const uint32_t kv_stride_h = params.kv_stride_h;
  DTypeO* o = params.o;
  float* lse = params.lse;
  uint32_t kv_chunk_size = params.kv_chunk_size;

  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t num_qo_heads = params.num_qo_heads;

  extern __shared__ uint8_t smem[];
  AttentionVariant variant(params, /*batch_idx=*/0, smem);
  const uint32_t seq_len = variant.kv_len;
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeKV));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }

    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + qo_head_idx * q_stride_h, freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);
  }
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, /*batch_idx=*/0,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, qo_head_idx,
        kv_head_idx, s, st_local, tx, ty, tz);
    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local, tx);
    block.sync();

    // load v
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_local, reinterpret_cast<float*>(smem), smem_md,
                                      tx, ty, tz);
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st_local.o[i] = variant.OutputTransform(params, st_local.o[i], /*batch_idx=*/0, /*qo_idx=*/0,
                                            qo_head_idx, st_local.m, st_local.d, /*scale=*/1.0f);
  }

  st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  if (lse != nullptr) {
    lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchDecodeWithPagedKVCacheDevice(const Params& params, uint8_t smem[],
                                                             const uint32_t bx = blockIdx.x,
                                                             const uint32_t by = blockIdx.y,
                                                             const uint32_t tx = threadIdx.x,
                                                             const uint32_t ty = threadIdx.y,
                                                             const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size; // constexpr uint32_t bdx = HEAD_DIM / vec_size;
  const uint32_t batch_idx = params.request_indices[bx]; // request index in current batch, if !split_kv, batch_idx == bx
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx]; // if !split_kv, kv_tile_idx == bx
  const uint32_t kv_head_idx = by; // constexpr uint32_t bdy = GROUP_SIZE;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty; // kv_head_idx * bdy calculates the qo head's group, ty means qo head's offset in 1 group
  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr); // kv_chunk_size_ptr_h[0] = kv_chunk_size_in_pages * page_size; 就一个元素
  const uint32_t kv_len = paged_kv.get_length(batch_idx); // current request's kv len (# tokens)
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len; // partition_kv == split_kv (?)
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start; // !split_kv, chunk_size = kv_len of current request

  AttentionVariant variant(params, batch_idx, smem);
  // smem is shared by all threads in threadblock
  DTypeKV* k_smem = (DTypeKV*)smem; 
  // k_smem[stage][bdz][bdy][tile_size_per_bdx][head_dim] 
  // tile: 在 seq_len 维度上的一个逻辑分块，大小为 = tile_size_per_bdx × bdy × bdz 个 token
  //       每个 token 对应 head_dim = bdx * vec_size 个元素，一个 threadblock 加载一个 tile
  // tile_size_per_bdx: 每个 x-dimension 线程负责的 token 数量
  // tile_size_per_bdx: 每个 (bdz, bdy) 组额外处理的 token 倍数（用于优化单 query 场景）
  // bdy = group size: 一个 block 负责一个 group 的计算，所以对每个 token 都要加载 group_size 个 kv
  // bdz: 补齐线程数，为了让 warp 有更大的 token 维度并行度，而不是串行取数据 (?)
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV)); // smem for m (max logit) and d (sum of exp(pre-softmax logits - m))，地址和 kv_offset_smem 一致，同一块 shared mem 在不同 phase 被复用

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll // this unrolls a for loop into code like freq[0] = ..., freq[1] = ..., ..., freq[vec_size-1] = ...
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // load current qo head's q tensor of current request, only load vec_size (out of head_dim) elements
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size]; // total number of pages that this batch's kv cache uses，即规定 page table 的边界，用于 protective_get_kv_offset 防止越界访问

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start; // 当前 req 在全局 KV 序列中的起始 token 位置（逻辑索引），paged_kv.indptr[batch_idx] 代表当前 req 前面的 req 一共占据了几个 page，chunk start 表示从第几个 token 开始
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r; // 页索引 q (商), 页内偏移 r (余数)
    // n = packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx 代表在 page table 中的全局逻辑 token index，一共取 tile_size_per_bdx * bdz * bdy * bdx 个 token
    // 一次循环中 block 内每个 thread 读1个 token，这里的索引只是为了方便并行读数据，和后面的实际 attention 计算没有映射关系
    // j: 当前线程在 tile 内负责的第 j 个 token
    // (tz, ty) pair 决定在 tile 内处理哪个 token，tx 决定处理该 token 的 head_dim 的哪一部分
    // (j * bdz + tz) * bdy + ty: tile 内 token 偏移，值域是 [0, num tokens per tile = tile_size_per_bdx * bdy * bdz)，这么算是为了保证 warp 内访问的地址连续，比如 j=1时，所有线程访问 token 24~27
    // q = n / page_size = page index
    // r = n % page_size = token's local index inside a page
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
        //                               page_iter=q, head_idx=kv_head_idx, entry_idx=r, feat_idx=0, last_indptr=last_indptr
  }
  block.sync(); // 等价于 __syncthreads() (see https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#sync)

  size_t kv_offset[tile_size_per_bdx]; // variables defined like this are thread-local variables and reside in register (or local memory)
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    // 每个线程读取 num_stages_smem * tile_size_per_bdx 个 kv vec 在 kv data 中的起始地址，存入 kv_offset，作为计算前的预热
    // 对于一个 threadblock，取的是 num_stages_smem * bdz * bdy * tile_size_per_bdx 个 token 的 bdx * vec_size = head_dim 长度的 kv vec（也就是完整的 head_dim 的向量）
    // 在这里 tx = threadIdx.x 用来算 head_dim 特征维度的 offset，不参与 token index 的计算了，也就是说，(threadIdx.y, threadIdx.z) 相等的所有线程搬运的都是同一个 token 的数据，threadIdx.x 用来区分搬运的具体是 head_dim 上的哪一段 vec_size 个特征维度元素
    // 由于之前 static_assert(num_stages_smem <= bdx); 所以 num_stages_smem * bdz * bdy * tile_size_per_bdx < tile_size_per_bdx * bdz * bdy * bdx 
    // 读出来的 kv_offset 肯定在上面 protective_get_kv_offset 写入 kv_offset_smem 的范围
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      // ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j: token index, iter 是 stages 的 iterator
      // kv_offset_smem 映射 token -> element offset of token (整个 head_dim 向量的起始位置，单位是 DTypeKV 不是 token) in kv data (physical offset)，+ tx * vec_size 是为了算特征维度的偏移
      // token index = (iter*bdz*bdy + tz*bdy + ty) * tile + j，说明 (tz, ty) 相等的一组 thread，读的是同一个 token 的 某一个 kv head 的 长度为 head_dim 的 embedding 向量
      // 而一整个 block 在一个 iter 内又读的是这个 block 负责的 kv head 的 (tile_size_per_bdx*bdy*bdz) 个连续 token 的 kv offset
      // thread index = blockId*bdx*bdy*bdz + tz*(bdx*bdy) + ty*bdx + tx= block offset + (tz*bdy+ty)*bdx + tx
      // 另外由于之前 assert bdx <= 32，所以一个 block 内，相同 (threadIdx.y, threadIdx.z) 的线程的 thread index 是连续的，它们就肯定在同一个 warp 内
      // 实际上 CUDA 不保证 ty tz 不同的 thread 一定在不同 warp，但是根据 flashinfer 对 head_dim, group_size 的约束，vec_dize \in {8, 16}, bdx \in {4, 8, 16, 32}，所以人为使得一组 (ty, tz) 的线程数肯定整除 warp 大小 (32)，所以不同 ty/tz 值的线程肯定分到不同 warp 中
      // 实际上 CUDA 不保证 ty tz 不同的 thread 一定在不同 warp，但是根据 flashinfer 对 head_dim, group_size 的约束，vec_dize \in {8, 16}, bdx \in {4, 8, 16, 32}，所以人为使得一组 (ty, tz) 的线程数肯定整除 warp 大小 (32)，所以相同 (ty, tz) 的线程肯定在一个 warp 内而不会被分到不同 warp，一个 warp 可能包含若干个 (ty, tz) 组
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      // kv_offset[j]: 当前线程读取的第 j 个 token 的 vec 的全局内存地址
      // stage_idx: 当前计算 attention 需要用的数据在 smem 的哪个 stage buffer
      // token index = ((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j
      // 一个线程读 tile_size_per_bdx 个 token 的 vec，一个 threadblock 读一个 tile 的 token 的 head_dim data
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size, // shared mem ptr (dest)
          paged_kv.k_data + kv_offset[j], // global mem ptr (source)
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size); // predicate，判断是否访问越界，越界了就跳过 load 指令，因为 kNoFill 所以也不会往 smem 填0（不填比填快）
    }
    cp_async::commit_group(); // cp.async.commit_group instruction creates a new cp.async-group per thread and batches all prior cp.async instructions initiated by the executing thread but not committed to any cp.async-group into the new cp.async-group. (see https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group)
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) { // load v data 同上
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>( // 这里 predicate = false 时会填0
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem; // 在这个 for(iter) 的循环里，stage_idx == iter
  }

  state_t<vec_size> st;
  // s: thread-local result of qk computation
  float s[bdy * tile_size_per_bdx]; // group size * tile_size_per_bdx

#pragma unroll 2 // embed 2 iterations of the loop in a single cycle
  // 1 iteration handles a tile = (tile_size_per_bdx * bdy * bdz) tokens
  // kv_offset_smem 存储了 bdx * tile 个 token 的 kv element offset（从上面可以看出一次加载就是这么大的量）
  // so every `bdx` iterations, we refresh kv offset smem
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      // 之前的 kv 全都计算完毕，需要取新一轮的 kv offset
      // 读 tile_size_per_bdx * bdz * bdy * bdx 个 kv token offset 填入 kv_offset_smem
      // 注意后面每次 pred_load 到 k/v smem 的 kv tensor data 是现在读 kv offset 对应的 token set 的子集，因为 num_stages_smem <= bdx
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz + // 读取这一轮的 offset 的起点，跳过之前已经读完的 token: (iter+num_stages_smem) = n * bdx, n * bdx * tile 就是 n 轮，n = 1, 2, 3, ...
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx), // 同 preload k/v tiles
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    // cp.async.wait_group instruction will cause executing thread to wait till only N or fewer of the most recent cp.async-groups are pending and all the prior cp.async-groups committed by the executing threads are complete. (see https://docs.nvidia.com/cuda/parallel-thread-execution/)
    cp_async::wait_group<2 * num_stages_smem - 1>(); // An executing thread can wait for the completion of all cp.async operations in a cp.async-group using cp.async.wait_group.
    block.sync();
    compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, // 传入的 smem 起始位置只和 tz 相关，所以 bdz 个线程负责一个 tile，每个线程分配到 tile_size_per_bdx * bdy 个 token 的 qk 计算
        q_vec, freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]) +
            chunk_start + iter * tile_size_per_bdx * bdy * bdz, // 无 rope 无 split kv 的情况下，kv_idx_base = iter * tile_size_per_bdx * bdy * bdz，表示当前 tile 在全局 k/v data 的 token index（k, v data 是两个并列的 tensor，所以 k 和 v 内部的同一个位置对应的肯定是相同 token），不过这里只需要 k
        iter * tile_size_per_bdx * bdy * bdz, // 无 rope 无 split kv 时，iter_base 同 kv_idx_base
        chunk_size, // iter_bound
        qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz);
    block.sync();

#pragma unroll
    // 一个 block 从 kv offset smem 读取 bdz * bdy * tile_size_per_bdx 个 token 的 kv offset
    // (iter + num_stages_smem) % bdx 是因为 kv_offset_smem 包含 bdx * (bdz * bdy * tile_size_per_bdx) 个 token 的 offset，
    // 这里只需要读它的 1/bdx 长度的一段，(iter + num_stages_smem) % bdx) 就是这个“段”的 offset
    // 因为每次 % bdx == 0 的时候会加载新的 kv offset smem，所以 % bdx == 0 的时候就读起点，% bdx == 1 的时候就往后读一段，以此类推
    // iter + num_stages_smem < bdx 的时候，kv offset smem 没有更新过，用的是第一次读的数据（就是 preload k/v tiles 那段代码，进入当前 for(iter) 大循环真正开始做 attention 计算之前，读上来的数据
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles（k 用完了，加载下个 tile = bdy * bdz * tile_size_per_bdx 个 token）
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      // ((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j 
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    // 计算当前 tile 的 softmax(qk) * v
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

    // load v tiles（v 用完了，加载下个 tile，同上 load k tiles）
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>(); // The executing thread waits on all the prior cp.async-groups to complete.
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st.o[i] = variant.OutputTransform(params, st.o[i], bx, /*qo_idx=*/0, qo_head_idx, st.m, st.d,
                                      /*scale=*/1.0f);
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem);
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single request
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 *   head_dim] for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param pos_encoding_mode The positional encoding mode
 * \param rope_scale The scaling factor used in RoPE Interpolation
 * \param rope_theta The theta used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t seq_len = params.kv_len;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  auto compute_capacity = GetCudaComputeCapability();
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeKV)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM * sizeof(DTypeKV) +
          2U * bdy * bdz * sizeof(float);
      auto kernel =
          SingleDecodeWithKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                        vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      if (seq_len <= 256 || tmp == nullptr) {
        // no need to use partition-kv kernel
        dim3 nblks = dim3(1, num_kv_heads);
        dim3 nthrs = dim3(bdx, bdy, bdz);
        params.kv_chunk_size = seq_len;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // use partition-kv kernel
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        int dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
        uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
        uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
        uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
        dim3 nblks = dim3(num_chunks, num_kv_heads);
        if (nblks.x == 0 || nblks.y == 0) {
          std::ostringstream err_msg;
          err_msg << "Invalid kernel configuration: nblks=(" << nblks.x << "," << nblks.y << ")";
          FLASHINFER_ERROR(err_msg.str());
        }
        dim3 nthrs = dim3(bdx, bdy, bdz);
        float* tmp_lse = (float*)(tmp + num_chunks * num_qo_heads * HEAD_DIM);
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp;
        params.lse = tmp_lse;
        params.kv_chunk_size = kv_chunk_size;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(
              MergeStates(tmp, tmp_lse, o, lse, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        } else {
          FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy); // max num threads per block = max bdx * max bdy = 32 * 8 = 256 (8 warps)
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      auto kernel =
          BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                            vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 nblks(padded_batch_size, num_kv_heads); // gridDim，一个 
      dim3 nthrs(bdx, bdy, bdz);

      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk_and_update_local_stat_mla(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* ckv_smem,
    const vec_t<float, vec_size_ckv>& q_nope_vec, const T* kpe_smem,
    const vec_t<float, vec_size_kpe>& q_pe_vec, const vec_t<float, vec_size_kpe>& freq,
    uint32_t kv_idx_base, uint32_t iter_base, uint32_t iter_bound, state_t<vec_size_ckv>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  float s[tile_size];
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> ckv_vec;
    ckv_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);

    vec_t<float, vec_size_kpe> kpe_vec;
    kpe_vec.cast_load(kpe_smem + j * head_dim_kpe + tx * vec_size_kpe);

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      s[j] += q_nope_vec[i] * ckv_vec[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size_kpe; ++i) {
      s[j] += q_pe_vec[i] * kpe_vec[i];
    }
    s[j] *= params.sm_scale;
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size_ckv; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> v_vec;
    v_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q_nope = params.q_nope;
  const DTypeQ* q_pe = params.q_pe;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto& paged_kv = params.paged_kv;
  const IdType* q_rope_offset = params.q_rope_offset;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const float rope_rcp_scale = params.rope_rcp_scale;
  const float rope_rcp_theta = params.rope_rcp_theta;
  const bool partition_kv = params.partition_kv;
  params.sm_scale *= math::log2e;

  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const uint32_t t_offset = dim3_offset(bdy, bdx, tz, ty, tx);

  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[batch_idx]) return;
  const uint32_t mapped_batch_idx = params.request_indices[batch_idx];

  const uint32_t orig_seq_len = paged_kv.get_length(mapped_batch_idx);
  int32_t q_rope_offset_val =
      q_rope_offset == nullptr ? (orig_seq_len - 1) : q_rope_offset[mapped_batch_idx];

  const uint32_t kv_chunk_idx_in_orig_mapped_batch = params.kv_tile_indices[batch_idx];
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t cur_chunk_start =
      partition_kv ? kv_chunk_idx_in_orig_mapped_batch * kv_chunk_size : 0;
  const uint32_t cur_chunk_end =
      partition_kv ? min((kv_chunk_idx_in_orig_mapped_batch + 1) * kv_chunk_size, orig_seq_len)
                   : orig_seq_len;
  const uint32_t cur_chunk_len = cur_chunk_end - cur_chunk_start;

  uint32_t packed_page_iter_base =
      paged_kv.indptr[mapped_batch_idx] * paged_kv.page_size + cur_chunk_start;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  constexpr uint32_t kv_iter_len = bdy * bdz;
  constexpr uint32_t compute_qk_tile = bdy;

  extern __attribute__((shared)) uint8_t smem[];
  DTypeKV* ckv_smem = (DTypeKV*)smem;
  DTypeKV* kpe_smem = (DTypeKV*)((uint8_t*)ckv_smem +
                                 num_stages_smem * kv_iter_len * head_dim_ckv * sizeof(DTypeKV));
  size_t* ckv_offset_smem = (size_t*)((uint8_t*)kpe_smem + num_stages_smem * kv_iter_len *
                                                               head_dim_kpe * sizeof(DTypeKV));
  size_t* kpe_offset_smem = (size_t*)((uint8_t*)ckv_offset_smem + bdx * bdy * bdz * sizeof(size_t));
  float* smem_md = (float*)ckv_offset_smem;

  AttentionVariant variant(params, batch_idx, smem);

  vec_t<float, vec_size_ckv> q_nope_vec[tile_size_qo_heads];
  vec_t<float, vec_size_kpe> q_pe_vec[tile_size_qo_heads];
  state_t<vec_size_ckv> st[tile_size_qo_heads];
  uint32_t qo_head_idx[tile_size_qo_heads];

  vec_t<float, vec_size_kpe> freq;

#pragma unroll
  for (uint32_t i = 0; i < vec_size_kpe; ++i) {
    freq[i] = rope_rcp_scale * __powf(rope_rcp_theta, float(2 * ((tx * vec_size_kpe + i) / 2)) /
                                                          float(head_dim_kpe));
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // load q_nope and q_pe tile
#pragma unroll
  for (int i = 0; i < tile_size_qo_heads; ++i) {
    qo_head_idx[i] = dim3_offset(bdy, tile_size_qo_heads, blockIdx.y, threadIdx.y, i);
    if (qo_head_idx[i] < num_qo_heads) {
      q_nope_vec[i].cast_load(q_nope +
                              (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                              tx * vec_size_ckv);
      q_pe_vec[i].cast_load(q_pe +
                            (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_kpe +
                            tx * vec_size_kpe);
    }
  }

  // init paged-cache read offset to be used
  uint32_t q, r;
  paged_kv.page_size.divmod(packed_page_iter_base + t_offset, q, r);
  ckv_offset_smem[t_offset] = paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
  kpe_offset_smem[t_offset] = paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
  block.sync();

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size_ckv * 8;
  constexpr uint32_t tx_fold = vec_size_ckv / vec_size_kpe;
  static_assert(num_stages_smem <= bdx);
  size_t offset_bytes;
  bool is_valid_range;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    is_valid_range = (iter * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;

    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes =
        kpe_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);

    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll
  for (uint32_t iter = 0; iter < ceil_div(cur_chunk_len, kv_iter_len); ++iter) {
    cp_async::wait_group<1 * num_stages_smem - 1>();
    block.sync();
    const int32_t kv_idx_base =
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[mapped_batch_idx]) +
        cur_chunk_start + iter * kv_iter_len;
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      compute_qk_and_update_local_stat_mla<vec_size_ckv, vec_size_kpe, bdx, compute_qk_tile>(
          params, variant, mapped_batch_idx,
          ckv_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_ckv, q_nope_vec[i],
          kpe_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_kpe, q_pe_vec[i],
          freq, kv_idx_base,
          /*iter_base*/ iter * kv_iter_len, /*iter_bound*/ cur_chunk_len, st[i]);
    }

    if ((iter + num_stages_smem) % bdx == 0) {
      uint32_t q, r;
      paged_kv.page_size.divmod(
          packed_page_iter_base + (iter + num_stages_smem) * kv_iter_len + t_offset, q, r);
      ckv_offset_smem[t_offset] =
          paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
      kpe_offset_smem[t_offset] =
          paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
    }
    block.sync();

    is_valid_range =
        ((iter + num_stages_smem) * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;
    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes = kpe_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  if (bdz != 1) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads)
        sync_state<vec_size_ckv, bdx, bdy, bdz>(variant, st[i], (float*)smem, smem_md, tx, ty, tz);
    }
  }

  if (tz == 0) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads) {
#pragma unroll
        for (size_t j = 0; j < vec_size_ckv; ++j) {
          st[i].o[j] = variant.OutputTransform(params, st[i].o[j], batch_idx, /*qo_idx=*/0,
                                               qo_head_idx[i], st[i].m, st[i].d, /*scale=*/1.0f);
        }
        st[i].o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                           tx * vec_size_ckv);

        if (lse != nullptr) {
          lse[batch_idx * num_qo_heads + qo_head_idx[i]] = st[i].get_lse();
        }
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA(Params params, typename Params::DTypeO* tmp_v,
                                                     float* tmp_s, bool enable_pdl,
                                                     cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
  constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
  constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

  constexpr uint32_t bdy = 8;
  constexpr uint32_t tile_size_qo_heads = 2;
  constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  const uint32_t gdy = ceil_div(num_qo_heads, qo_heads_per_block);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nblks(padded_batch_size, gdy);
    dim3 nthrs(bdx, bdy, bdz);

    // PDL launch config
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config;
    if (enable_pdl) {
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
    }

    if (tmp_v == nullptr) {
      // do not use partition-kv kernel
      params.partition_kv = false;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
    } else {
      // use partition-kv kernel
      params.partition_kv = true;
      auto o = params.o;
      auto lse = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
      FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
          tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr, num_qo_heads,
          HEAD_DIM_CKV, enable_pdl, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
