/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "orttraining/training_ops/cuda/fp8/common.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

void ComputeDefaultScalingFactor(cudaStream_t stream, const fp32* amax, const fp32* scale_in, fp32* scale_out,
                                 fp32* scale_inv, size_t count);

class Scaling {
 public:
  Scaling(size_t count, size_t length);
  ~Scaling();
  Status Update(const CudaKernel* kernel, OpKernelContext* context);

  fp32* Scale() { return scale + (amax_history_idx * count); }
  fp32* ScaleInv() { return scale_inv; }
  fp32* AmaxHistory() { return amax_history + (amax_history_idx * count * length); }

 private:
  fp32* NextScale() { return scale + ((1 - amax_history_idx) * count); }
  fp32* NextAmaxHistory() { return amax_history + ((1 - amax_history_idx) * count * length); }

  size_t count = 0;
  size_t length = 0;
  fp32* scale = nullptr;
  fp32* scale_inv = nullptr;
  fp32* amax = nullptr;
  fp32* amax_history = nullptr;
  size_t amax_history_idx = 0;
};

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
