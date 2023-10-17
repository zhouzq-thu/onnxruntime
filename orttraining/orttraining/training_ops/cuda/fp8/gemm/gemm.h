// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "orttraining/training_ops/cuda/fp8/common.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

struct SimpleTensor {
  void* data_ptr;
  std::vector<int64_t> shape;
  DType dtype;

  SimpleTensor(void* data_ptr, const std::vector<int64_t>& shape, DType dtype)
      : data_ptr(data_ptr), shape(shape), dtype(dtype) {}
  SimpleTensor() : SimpleTensor(nullptr, {}, DType::kFloat32) {}
};

class FP8GemmWorkspace {
 public:
  static FP8GemmWorkspace& Instance() {
    static FP8GemmWorkspace instance;
    return instance;
  }

  size_t SizeInBytes() const { return workspace_size_bytes; }

  void* GetWorkspace() const { return workspace; }

 private:
  FP8GemmWorkspace();
  ~FP8GemmWorkspace();

  static constexpr size_t workspace_size_bytes = 33554432;
  void* workspace = nullptr;
};

Status FP8Gemm(cudaStream_t stream, const SimpleTensor input_a, const SimpleTensor input_b,
               const SimpleTensor input_bias, SimpleTensor output_d, SimpleTensor pre_gelu_out, void* a_scale_inv,
               void* b_scale_inv, void* scale, void* amax, bool trans_a, bool trans_b, bool grad,
               bool use_split_accumulator, int math_sm_count);

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
