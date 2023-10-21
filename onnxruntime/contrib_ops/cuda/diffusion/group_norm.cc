// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/diffusion/group_norm.h"
#include "contrib_ops/cuda/diffusion/group_norm_impl.h"
#include "contrib_ops/cuda/diffusion/group_norm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define GROUP_NORM_TYPES float, MLFloat16

ONNX_OPERATOR_KERNEL_EX(
    GroupNorm, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<GROUP_NORM_TYPES>()), GroupNorm);

using namespace ONNX_NAMESPACE;

template <typename T>
struct DispatchGroupNorm {
  Status operator()(cudaStream_t stream,
                    Tensor* output,
                    const Tensor* input,
                    const Tensor* gamma,
                    const Tensor* beta,
                    void* workspace,
                    float epsilon,
                    int batch_size,
                    int num_channels,
                    int height,
                    int width,
                    int num_groups,
                    bool use_swish_activation,
                    bool channels_last,
                    const cudaDeviceProp& device_prop) {
    typedef typename ToCudaType<T>::MappedType CudaT;

    // Use TensorRT NHWC GroupNorm fast kernel whenever possible.
    if (channels_last && num_groups == static_cast<int>(kGroupNormNumberOfGroups) &&
        batch_size <= static_cast<int>(kMaxGroupNormBatchSize) &&
        (num_channels == 320 || num_channels == 640 || num_channels == 960 || num_channels == 1280 || num_channels == 1920 || num_channels == 2560 || num_channels == 128 || num_channels == 256 || num_channels == 512)) {
      return LaunchGroupNormKernel<CudaT>(
          stream,
          reinterpret_cast<CudaT*>(output->MutableData<T>()),
          reinterpret_cast<const CudaT*>(input->Data<T>()),
          gamma->Data<float>(),
          beta->Data<float>(),
          workspace,
          epsilon,
          batch_size,
          num_channels,
          height,
          width,
          num_groups,
          use_swish_activation);
    } else {
      const int64_t num_instances = static_cast<int64_t>(batch_size) * num_groups;
      const int64_t norm_size = static_cast<int64_t>(height) * width * num_channels / num_groups;
      const int64_t spatial_size = static_cast<int64_t>(height) * width;
      const bool channels_first = !channels_last;

      float* mean = reinterpret_cast<float*>(workspace);
      float* inv_variance = mean + kMaxGroupNormBatchSize * kGroupNormNumberOfGroups;
      const int sm_count = device_prop.multiProcessorCount;
      auto result = DispatchGroupNormKernel<CudaT, float>(
          stream, num_instances, norm_size, num_channels, spatial_size,
          static_cast<double>(epsilon),
          reinterpret_cast<const CudaT*>(input->Data<T>()),
          gamma->Data<float>(),
          beta->Data<float>(),
          reinterpret_cast<CudaT*>(output->MutableData<T>()),
          mean,
          inv_variance,
          channels_first,
          use_swish_activation,
          sm_count);
        if (result != cudaSuccess) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to run DispatchGroupNormKernel. Error:", cudaGetErrorString(result));
        }
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    }
    return Status::OK();
  }
};

GroupNorm::GroupNorm(const OpKernelInfo& op_info) : CudaKernel(op_info) {
  epsilon_ = op_info.GetAttrOrDefault<float>("epsilon", 1e-5f);
  ORT_ENFORCE(epsilon_ >= 0);

  int64_t num_groups;
  ORT_ENFORCE(op_info.GetAttr("groups", &num_groups).IsOK());
  ORT_ENFORCE(num_groups >= 0);
  num_groups_ = static_cast<int>(num_groups);

  int64_t activation;
  ORT_ENFORCE(op_info.GetAttr("activation", &activation).IsOK());
  ORT_ENFORCE(activation == 0 || activation == 1);  // 0 is None, 1 is Swish
  use_swish_activation_ = (activation == 1);

  channels_last_ = (op_info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(1)) != 0);
}

Status GroupNorm::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* gamma = context->Input<Tensor>(1);
  const Tensor* beta = context->Input<Tensor>(2);
  Tensor* output = context->Output(0, input->Shape());

  // if (!channels_last_) {
  //   return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
  //                          "only the channels_last layout is supported");
  // }

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 4 dimensions, got ", input_dims.size());
  }

  // Input and output format is either NHWC or NCHW
  int batch_size = static_cast<int>(input_dims[0]);
  int num_channels = channels_last_ ? static_cast<int>(input_dims[3]) : static_cast<int>(input_dims[1]);
  int height = channels_last_ ? static_cast<int>(input_dims[1]) : static_cast<int>(input_dims[2]);
  int width = channels_last_ ? static_cast<int>(input_dims[2]) : static_cast<int>(input_dims[3]);

  if (batch_size * num_groups_ > static_cast<int>(kMaxGroupNormBatchSize * kGroupNormNumberOfGroups)){
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "batch_size * num_groups shall be <= ",  kMaxGroupNormBatchSize * kGroupNormNumberOfGroups);
  }

  if (num_channels % num_groups_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "number of channels should be divisiable by num_groups");
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of channels in gamma and input does not match");
  }

  const auto& beta_dims = beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimension, got ", beta_dims.size());
  }
  if (beta_dims[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of channels in beta and input does not match");
  }

  if (context->GetUseDeterministicCompute()) {
    static std::once_flag log_warning;
    std::call_once(log_warning, []() {
      LOGS_DEFAULT(WARNING) << "GroupNorm has no deterministic CUDA kernel, its outputs may still be nondeterministic.";
    });
  }

  auto workspace = GetScratchBuffer<void>(GetGroupNormWorkspaceSizeInBytes(), context->GetComputeStream());

  auto& device_prop = GetDeviceProp();
  utils::MLTypeCallDispatcher<GROUP_NORM_TYPES> dispatcher(input->GetElementType());
  return dispatcher.InvokeRet<Status, DispatchGroupNorm>(Stream(context), output, input, gamma, beta, workspace.get(),
                                                         epsilon_,
                                                         batch_size,
                                                         num_channels,
                                                         height,
                                                         width,
                                                         num_groups_,
                                                         use_swish_activation_,
                                                         channels_last_,
                                                         device_prop
                                                         );
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
