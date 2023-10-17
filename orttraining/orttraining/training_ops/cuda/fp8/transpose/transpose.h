// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
namespace fp8 {

template <typename InputType, typename OutputType>
void CastTranspose(cudaStream_t stream, const InputType* input_data, OutputType* cast_output_data,
                   OutputType* transposed_output_data, const fp32* scale, fp32* amax, const size_t row_length,
                   const size_t num_rows);

template <typename OutputType>
size_t GetCastTransposeBiasWorkspaceSize(const size_t row_length, const size_t num_rows);

template <typename InputType, typename OutputType>
void CastTransposeBias(cudaStream_t stream, const InputType* input_data, OutputType* cast_output_data,
                       OutputType* transposed_output_data, InputType* dbias_data, fp32* workspace, const fp32* scale,
                       fp32* amax, const size_t row_length, const size_t num_rows);

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
