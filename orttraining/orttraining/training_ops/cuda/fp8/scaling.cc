// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/fp8/scaling.h"

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/tensor/pad_impl.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

Scaling::Scaling(size_t count, size_t length) : count(count), length(length) {
  // TODO: need to reset data to 0 or 1.
  CUDA_CALL_THROW(cudaMalloc(&scale, sizeof(fp32) * count * 2));
  CUDA_CALL_THROW(cudaMalloc(&scale_inv, sizeof(fp32) * count));
  CUDA_CALL_THROW(cudaMalloc(&amax, sizeof(fp32) * count));
  CUDA_CALL_THROW(cudaMalloc(&amax_history, sizeof(fp32) * length * count * 2));
}

Scaling::~Scaling() {
  CUDA_CALL_THROW(cudaFree(scale));
  CUDA_CALL_THROW(cudaFree(scale_inv));
  CUDA_CALL_THROW(cudaFree(amax));
  CUDA_CALL_THROW(cudaFree(amax_history));
}

Status Scaling::Update(const CudaKernel* kernel, OpKernelContext* context) {
  // TODO: update.
  // Reduce to amax.
  cudaStream_t stream = kernel->Stream(context);
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(amax, 0, sizeof(fp32) * count, stream));
  size_t workspace_bytes = 0;
  size_t indices_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;
  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  ORT_RETURN_IF_ERROR(reduce_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set({static_cast<int64_t>(length), static_cast<int64_t>(count)}, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set({static_cast<int64_t>(count)}, cudnn_type_X));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(kernel->GetCudnnHandle(context), reduce_desc, input_tensor,
                                                       output_tensor, &workspace_bytes));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(kernel->GetCudnnHandle(context), reduce_desc, input_tensor,
                                                     output_tensor, &indices_bytes));
  IAllocatorUniquePtr<void> workspace_cuda =
      workspace_bytes == 0 ? nullptr : kernel->GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
  IAllocatorUniquePtr<void> indices_cuda =
      indices_bytes == 0 ? nullptr : kernel->GetScratchBuffer<void>(indices_bytes, context->GetComputeStream());
  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(kernel->GetCudnnHandle(context), reduce_desc, indices_cuda.get(),
                                          indices_bytes, workspace_cuda.get(), workspace_bytes, &one, input_tensor,
                                          AmaxHistory(), &zero, output_tensor, amax));
  // Roll amax_history.
  TArray<int64_t> input_dims({static_cast<int64_t>(length - 1), static_cast<int64_t>(count)});
  TArray<int64_t> input_strides({static_cast<int64_t>(count), static_cast<int64_t>(1)});
  TArray<int64_t> lower_pads({static_cast<int64_t>(1), static_cast<int64_t>(0)});
  TArray<fast_divmod> fdm_output_strides;
  fdm_output_strides.SetSize(2);
  fdm_output_strides[0] = fast_divmod(static_cast<int>(count));
  fdm_output_strides[1] = fast_divmod(1);
  PadImpl<fp32>(stream, 2, input_dims, input_strides, lower_pads, 0.0f, Mode::Constant, AmaxHistory() + count,
                fdm_output_strides, NextAmaxHistory(), length * count);
  // Compute scale and scale_inv.
  ComputeDefaultScalingFactor(stream, amax, Scale(), NextScale(), ScaleInv(), count);
  // Update amax_history_idx.
  amax_history_idx = 1 - amax_history_idx;
  return Status::OK();
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
