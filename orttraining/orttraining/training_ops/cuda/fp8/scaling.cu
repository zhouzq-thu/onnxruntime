// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/fp8/scaling.h"

#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

struct OpDefaultScalingFactor {
  OpDefaultScalingFactor(const fp32* amax, const fp32* scale) : amax_(amax), scale_(scale) {}

  __device__ __inline__ fp32 operator()(TIndex idx) const {
    // TODO: real compute.
    return 1.0f;
  }

  const fp32* amax_;
  const fp32* scale_;
};

struct OpScalingFactorInverse {
  OpScalingFactorInverse(const fp32* scale) : scale_(scale) {}

  __device__ __inline__ fp32 operator()(TIndex idx) const { return 1.0f / scale_[idx]; }

  const fp32* scale_;
};

void ComputeDefaultScalingFactor(cudaStream_t stream, const fp32* amax, const fp32* scale_in, fp32* scale_out,
                                 fp32* scale_inv, size_t count) {
  OpDefaultScalingFactor op_scale(amax, scale_in);
  LaunchElementwiseKernel<fp32, OpDefaultScalingFactor, size_t>(stream, scale_out, op_scale, count);
  OpScalingFactorInverse op_scale_inv(scale_out);
  LaunchElementwiseKernel<fp32, OpScalingFactorInverse, size_t>(stream, scale_inv, op_scale_inv, count);
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
