// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RelPosAttnBiasBase : public OpKernel {
 public:
  explicit RelPosAttnBiasBase(const OpKernelInfo& op_kernel_info);

 private:
  int max_distance_;
  bool is_bidirectional_;
};

template <typename T>
class RelativeAttnBias : public RelPosAttnBiasBase {
 public:
  explicit RelativeAttnBias(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
