// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cuda/fp8/scaling.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

class FP8Linear : public CudaKernel {
 public:
  FP8Linear(const OpKernelInfo& info) : CudaKernel(info), scaling(3, 1024) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Scaling scaling;
};

class FP8LinearGrad : public CudaKernel {
 public:
  FP8LinearGrad(const OpKernelInfo& info) : CudaKernel(info), scaling(2, 1024) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Scaling scaling;
};

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
