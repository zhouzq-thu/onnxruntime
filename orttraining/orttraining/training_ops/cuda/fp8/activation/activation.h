// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
namespace fp8 {

template <typename IType, typename OType>
void Gelu(cudaStream_t stream, const IType* input_data, OType* output_data, const float* scale, float* amax,
          const size_t num_elements);

template <typename IType, typename OType>
void DGelu(cudaStream_t stream, const IType* grad_data, const IType* input_data, OType* output_data, const float* scale,
           float* amax, const size_t num_elements);

}  // fp8
}  // cuda
}  // onnxruntime
