/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "orttraining/training_ops/cuda/fp8/activation/activation.h"

#include "orttraining/training_ops/cuda/fp8/util/math.h"
#include "orttraining/training_ops/cuda/fp8/util/vectorized_pointwise.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

template <typename IType, typename OType>
void Gelu(cudaStream_t stream, const IType* input_data, OType* output_data, const float* scale, float* amax,
          const size_t num_elements) {
  constexpr int nvec = 32 / sizeof(IType);
  VectorizedUnaryKernelLauncher<nvec, Empty, gelu<float, float>>(input_data, output_data, scale, amax, num_elements,
                                                                 Empty(), stream);
}

template <typename IType, typename OType>
void DGelu(cudaStream_t stream, const IType* grad_data, const IType* input_data, OType* output_data, const float* scale,
           float* amax, const size_t num_elements) {
  constexpr int nvec = 32 / sizeof(IType);
  VectorizedUnaryGradKernelLauncher<nvec, Empty, dgelu<float, float>>(grad_data, input_data, output_data, scale, amax,
                                                                      num_elements, {}, stream);
}

#define SPECIALIZED_GELU_IMPL(IType, OType)                                                          \
  template void Gelu<IType, OType>(cudaStream_t stream, const IType* input_data, OType* output_data, \
                                   const float* scale, float* amax, const size_t num_elements);

SPECIALIZED_GELU_IMPL(fp16, fp8e4m3)

#undef SPECIALIZED_GELU_IMPL

#define SPECIALIZED_DGELU_IMPL(IType, OType)                                                              \
  template void DGelu<IType, OType>(cudaStream_t stream, const IType* grad_data, const IType* input_data, \
                                    OType* output_data, const float* scale, float* amax, const size_t num_elements);

SPECIALIZED_DGELU_IMPL(fp8e5m2, fp16)

#undef SPECIALIZED_DGELU_IMPL

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
