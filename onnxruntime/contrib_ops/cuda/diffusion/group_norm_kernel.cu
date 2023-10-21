// The implementation below is based on OneFlow's GroupNorm kernel at
// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/group_norm_kernel.cu
/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modification: gamma and beta are stored in compute type; pass device property to layernorm kernel
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/diffusion/layer_norm.cuh"
#include <cub/cub.cuh>
#include <cutlass/fast_math.h>
#include <complex>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

enum class UnaryOp {
  kIdentity,
  kSilu,
};

template <UnaryOp unary_op, typename T, typename U>
struct UnaryFunctor;

template <typename T, typename U>
struct UnaryFunctor<UnaryOp::kIdentity, T, U> {
  __device__ __host__ __forceinline__ UnaryFunctor() {}

  __device__ __host__ __forceinline__ T operator()(U x) const { return static_cast<T>(x); }
};

template <typename T, typename U>
struct UnaryFunctor<UnaryOp::kSilu, T, U> {
  __device__ __host__ __forceinline__ UnaryFunctor() {}

  __device__ __host__ __forceinline__ T operator()(U x) const {
    return static_cast<T>(x / (static_cast<U>(1) + expf(-x)));
  }
};

}  // namespace

// Affine means whether there is gamma and beta.
template <typename T, typename U, UnaryOp activation, bool affine>
struct AffineStore {
  AffineStore(T* y, int64_t row_size, int64_t channel_size, int64_t spatial_size, const U* gamma, const U* beta)
      : y(y),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size),
        gamma(gamma),
        beta(beta),
        act() {}

  template <int PackSize>
  __device__ void store(const U* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<T, PackSize> y_pack;
    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_beta_offset = (offset / spatial_size) % channel_size;
    U gamma_val = 1.0;
    U beta_val = 0.0;
    if (affine) {
      gamma_val = gamma[gamma_beta_offset];
      beta_val = beta[gamma_beta_offset];
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      U normalized_i = static_cast<U>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_val + beta_val);
      } else {
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<T, PackSize>*>(y) + packed_offset) = y_pack.storage;
  }

  bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }

  T* y;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
  const U* gamma;
  const U* beta;
  UnaryFunctor<activation, T, U> act;
};

template <typename T, typename U, bool affine>
struct ScaleLoad {
  using LoadType = U;

  ScaleLoad(const T* src, const U* gamma, int64_t row_size, int64_t channel_size, int64_t spatial_size)
      : src(src),
        gamma(gamma),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size) {}

  template <int PackSize>
  __device__ void load(U* dst, int64_t row, int64_t col) const {
    cuda::layer_norm::Pack<T, PackSize> src_pack;
    cuda::layer_norm::Pack<U, PackSize> gamma_pack;

    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_offset = (offset / spatial_size) % channel_size;

    src_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<T, PackSize>*>(src) + packed_offset);

    U gamma_val = static_cast<U>(1.0);
    if (affine) {
      gamma_val = gamma[gamma_offset];
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      dst[i] = static_cast<U>(src_pack.elem[i] * gamma_val);
    }
  }

  bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }
  const T* src;
  const U* gamma;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
};

#ifdef USE_CUTLASS
template <typename T, typename U, UnaryOp activation, bool affine>
struct ChannelsLastStore {
  ChannelsLastStore(T* y, const U* gamma, const U* beta, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups),
        act() {}

  template <int PackSize>
  __device__ void store(const U* src, int32_t row, int32_t col) {
    cuda::layer_norm::Pack<T, PackSize> y_pack;
    cuda::layer_norm::Pack<U, PackSize> gamma_pack;
    cuda::layer_norm::Pack<U, PackSize> beta_pack;
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    const int32_t y_offset =
        (batch_idx * c0.divisor * c1.divisor * spatial_size + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx) / PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1.divisor + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<U, PackSize>*>(gamma) + gamma_beta_offset);
      beta_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<U, PackSize>*>(beta) + gamma_beta_offset);
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      U normalized_i = static_cast<U>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
      } else {
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<T, PackSize>*>(y) + y_offset) = y_pack.storage;
  }

  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }

  T* y;
  const U* gamma;
  const U* beta;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
  UnaryFunctor<activation, T, U> act;
};

template <typename T, typename U>
struct ChannelsLastLoad {
  using LoadType = U;
  ChannelsLastLoad(const T* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}

  template <int N>
  __device__ void load(U* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    cuda::layer_norm::Pack<T, N> pack;
    const int32_t offset = (batch_idx * c0.divisor * c1.divisor * spatial_size + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx) / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<T, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<U>(pack.elem[i]);
    }
  }

  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }

  const T* src;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};

#else

template <typename T, typename U, UnaryOp activation, bool affine>
struct ChannelsLastStore {
  ChannelsLastStore(T* y, const U* gamma, const U* beta, int64_t spatial_size,
                    int64_t channel_size, int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups),
        act() {}

  template <int PackSize>
  __device__ void store(const U* src, int32_t row, int32_t col) {
    cuda::layer_norm::Pack<T, PackSize> y_pack;
    cuda::layer_norm::Pack<U, PackSize> gamma_pack;
    cuda::layer_norm::Pack<U, PackSize> beta_pack;
    int32_t spatial_idx = col / c1;
    int32_t c1_idx = col - spatial_idx * c1;
    int32_t batch_idx = row / c0;
    int32_t c0_idx = row - batch_idx * c0;
    const int32_t y_offset =
        (batch_idx * c0 * c1 * spatial_size + spatial_idx * c0 * c1 + c0_idx * c1 + c1_idx) / PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1 + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<U, PackSize>*>(gamma) + gamma_beta_offset);
      beta_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<U, PackSize>*>(beta) + gamma_beta_offset);
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      U normalized_i = static_cast<U>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
      } else {
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<T, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (c1 % pack_size) == 0; }
  T* y;
  const U* gamma;
  const U* beta;
  int32_t spatial_size;
  int32_t c0;
  int32_t c1;
  UnaryFunctor<activation, T, U> act;
};

template <typename T, typename U>
struct ChannelsLastLoad {
  using LoadType = U;
  ChannelsLastLoad(const T* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
  template <int N>

  __device__ void load(U* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx = col / c1;
    int32_t c1_idx = col - spatial_idx * c1;
    int32_t batch_idx = row / c0;
    int32_t c0_idx = row - batch_idx * c0;
    cuda::layer_norm::Pack<T, N> pack;
    const int32_t offset =
        (batch_idx * c0 * c1 * spatial_size + spatial_idx * c0 * c1 + c0_idx * c1 + c1_idx) / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<T, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<U>(pack.elem[i]);
    }
  }

  bool CanPackAs(size_t pack_size) { return (c1 % pack_size) == 0; }

  const T* src;
  int32_t spatial_size;
  int32_t c0;
  int32_t c1;
};
#endif

template <typename T, typename U, UnaryOp activation, bool affine>
cudaError_t GroupNormForwardGpu(cudaStream_t stream, const int64_t num_instances, const int64_t norm_size,
                         const int64_t channel_size, const int64_t spatial_size, const double epsilon,
                         const T* x, const U* gamma, const U* beta,
                         T* y, U* mean, U* inv_variance,
                         bool channels_first, const int sm_count) {
  if (channels_first) {
    cuda::layer_norm::DirectLoad<T, U> load(x, norm_size);
    AffineStore<T, U, activation, affine> store(y, norm_size, channel_size, spatial_size, gamma, beta);

    return cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), U>(
        stream, load, store, num_instances, norm_size, epsilon, mean, inv_variance, sm_count);
  } else {
    ChannelsLastLoad<T, U> load(x, spatial_size, channel_size, channel_size / (norm_size / spatial_size));
    ChannelsLastStore<T, U, activation, affine> store(y, gamma, beta, spatial_size, channel_size,
                                                      channel_size / (norm_size / spatial_size));

    return cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), U>(
        stream, load, store, num_instances, norm_size, epsilon, mean, inv_variance, sm_count);
  }
}

template <typename T, typename U, UnaryOp activation>
cudaError_t DispatchGroupNormAffine(cudaStream_t stream, const int64_t num_instances, const int64_t norm_size,
                             const int64_t channel_size, const int64_t spatial_size, const double epsilon,
                             const T* x, const U* gamma, const U* beta,
                             T* y, U* mean, U* inv_variance,
                             bool channels_first,
                             const int sm_count) {
  if (gamma != nullptr && beta != nullptr) {
    return GroupNormForwardGpu<T, U, activation, true>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon,
        x, gamma, beta, y, mean, inv_variance, channels_first, sm_count);
  } else {
    return GroupNormForwardGpu<T, U, activation, false>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon,
        x, gamma, beta, y, mean, inv_variance, channels_first, sm_count);
  }
}

template <typename T, typename U>
cudaError_t DispatchGroupNormKernel(cudaStream_t stream, const int64_t num_instances,
                             const int64_t norm_size, const int64_t channel_size,
                             const int64_t spatial_size, const double epsilon, const T* x,
                             const U* gamma, const U* beta, T* y,
                             U* mean, U* inv_variance,
                             bool channels_first,
                             bool has_silu_activation,
                             const int sm_count) {
  if (has_silu_activation) {
    return DispatchGroupNormAffine<T, U, UnaryOp::kSilu>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon,
        x, gamma, beta, y, mean, inv_variance, channels_first, sm_count);
  } else {
    return DispatchGroupNormAffine<T, U, UnaryOp::kIdentity>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon,
        x, gamma, beta, y, mean, inv_variance, channels_first, sm_count);
  }
}

template cudaError_t DispatchGroupNormKernel<float, float>(cudaStream_t stream, const int64_t num_instances,
                                                    const int64_t norm_size, const int64_t channel_size,
                                                    const int64_t spatial_size, const double epsilon,
                                                    const float* x, const float* gamma, const float* beta,
                                                    float* y, float* mean, float* inv_variance,
                                                    bool channels_first, bool has_silu_activation,
                                                    const int sm_count);

template cudaError_t DispatchGroupNormKernel<half, float>(cudaStream_t stream, const int64_t num_instances,
                                                   const int64_t norm_size, const int64_t channel_size,
                                                   const int64_t spatial_size, const double epsilon,
                                                   const half* x, const float* gamma, const float* beta,
                                                   half* y, float* mean, float* inv_variance,
                                                   bool channels_first, bool has_silu_activation,
                                                   const int sm_count);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
