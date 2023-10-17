// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm.h"
#include "embed_layer_norm_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"

#include <atomic>

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      RelativePositionBias,                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RelativeAttnBias<T>);

REGISTER_KERNEL_TYPED(float)

RelPosAttnBiasBase::RelPosAttnBiasBase(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  is_bidirectional_ = info.GetAttrOrDefault<int64_t>("is_bidirectional", 0) == 1;

  int64_t max_distance = 0;
  ORT_ENFORCE(info.GetAttr("max_distance", &max_distance).IsOK() && max_distance > 0);
  max_distance_ = static_cast<int>(max_distance);
}

template <typename T>
RelativeAttnBias<T>::RelativeAttnBias(const OpKernelInfo& op_kernel_info)
    : RelPosAttnBiasBase(op_kernel_info) {
}

template <typename T>
Status RelativeAttnBias<T>::Compute(OpKernelContext* context) const {
  const Tensor* bias_table = context->Input<Tensor>(0);
  const Tensor* query_length = context->Input<Tensor>(1);
  const Tensor* key_length = context->Input<Tensor>(2);

  const auto& bias_table_dims = bias_table->Shape().GetDims();
  const int64_t num_buckets = bias_table_dims[0];
  const int64_t num_heads = bias_table_dims[1];

  const int64_t query_len = *query_length->Data<int64_t>();
  const int64_t key_len = *key_length->Data<int64_t>();

  if (query_len != key_len) {
    ORT_THROW("Relative position bias currently only support query length equal to key length in Self Attention.");
  }

  Tensor* output = context->Output(0, {1, num_heads, query_len, key_len});
  T* relative_attention_bias = output->template MutableData<T>();
  const T* relative_attention_bias_table = bias_table->template Data<T>();
  const int head_id = blockIdx.x;
  for (int head_id = 0; head_id < head_num; ++head_id) {
    for (int seq_id = 0; seq_id < seq_len * seq_len; ++seq_id) {
      int row_id = seq_id / seq_len;
      int col_id = seq_id % seq_len;

      int relative_position = col_id - row_id;

      int relative_buckets = 0;
      int tmp_num_bucket = num_bucket;

      if (is_bidirectional) {
        tmp_num_bucket /= 2;
        if (relative_position > 0) {
          relative_buckets += tmp_num_bucket;
        } else {
          relative_position *= -1;
        }
      } else {
        if (relative_position > 0) {
          relative_position = 0;
        } else {
          relative_position *= -1;
        }
      }

      int max_exact = tmp_num_bucket / 2;
      bool is_small = relative_position < max_exact;

      int relative_position_if_large =
          max_exact + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact) * (tmp_num_bucket - max_exact));

      relative_position_if_large = min(relative_position_if_large, tmp_num_bucket - 1);

      relative_buckets += is_small ? relative_position : relative_position_if_large;

      relative_attention_bias[head_id * seq_len * seq_len + seq_id] =
          relative_attention_bias_table[head_id * num_bucket + relative_buckets];
    }
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
