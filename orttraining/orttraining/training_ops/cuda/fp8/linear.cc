// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/fp8/linear.h"

#include "orttraining/training_ops/cuda/fp8/gemm/gemm.h"
#include "orttraining/training_ops/cuda/fp8/transpose/transpose.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    FP8Linear, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>()), fp8::FP8Linear);

ONNX_OPERATOR_KERNEL_EX(
    FP8LinearGrad, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>()),
    fp8::FP8LinearGrad);

namespace fp8 {

namespace {

template <typename T>
struct DispatchFP8LinearImpl {
  Status operator()(const FP8Linear* fp8_linear, OpKernelContext* context, Tensor* input, Tensor* weight, Tensor* bias,
                    Tensor* output, Tensor* trans_input, Tensor* trans_weight, Tensor* scale_inv_fwd, float* scale,
                    float* scale_inv, float* amax_history) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    cudaStream_t stream = fp8_linear->Stream(context);
    const CudaT* input_data = reinterpret_cast<CudaT*>(input->Data<T>());
    const CudaT* weight_data = reinterpret_cast<CudaT*>(weight->Data<T>());
    const CudaT* bias_data = bias ? reinterpret_cast<CudaT*>(bias->Data<T>()) : nullptr;
    CudaT* output_data = reinterpret_cast<CudaT*>(output->MutableData<T>());
    Float8E4M3FN* trans_input_data = trans_input->MutableData<Float8E4M3FN>();
    Float8E4M3FN* trans_weight_data = trans_weight->MutableData<Float8E4M3FN>();
    float* scale_inv_fwd_data = scale_inv_fwd->MutableData<float>();
    const TensorShape& input_shape = input->Shape();
    const TensorShape& weight_shape = weight->Shape();
    IAllocatorUniquePtr<Float8E4M3FN> fp8_input_data = fp8_linear->GetScratchBuffer<Float8E4M3FN>(
        static_cast<size_t>(input_shape.Size()), context->GetComputeStream());
    IAllocatorUniquePtr<Float8E4M3FN> fp8_weight_data = fp8_linear->GetScratchBuffer<Float8E4M3FN>(
        static_cast<size_t>(weight_shape.Size()), context->GetComputeStream());
    CastTranspose(stream, input_data, fp8_input_data.get(), trans_input_data, scale, amax_history,
                  static_cast<size_t>(input_shape[1]), static_cast<size_t>(input_shape[0]));
    CastTranspose(stream, weight_data, fp8_weight_data.get(), trans_weight_data, scale, amax_history,
                  static_cast<size_t>(weight_shape[1]), static_cast<size_t>(weight_shape[0]));
    SimpleTensor weight_tensor(fp8_weight_data, {}, DType::kFloat8E4M3);
    SimpleTensor input_tensor(fp8_input_data, {}, DType::kFloat8E4M3);
    SimpleTensor bias_tensor(bias_data, {}, DType::kFloat16);
    SimpleTensor output_tensor(output_data, {}, DType::kFloat16);
    SimpleTensor pre_gelu_tensor(nullptr, {}, DType::kFloat16);
    return FP8Gemm(stream, weight_tensor, input_tensor, bias_tensor, output_tensor, pre_gelu_tensor, scale_inv,
                   scale_inv, scale, amax_history, true, false, false, false, 0);
  }
};

template <typename T>
struct DispatchFP8LinearGradImpl {
  Status operator()(const FP8LinearGrad* fp8_linear_grad, OpKernelContext* context, Tensor* grad_output,
                    Tensor* trans_input, Tensor* trans_weight, Tensor* scale_inv_fwd, Tensor* grad_input,
                    Tensor* grad_weight, Tensor* grad_bias, float* scale, float* scale_inv, float* amax_history) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    cudaStream_t stream = fp8_linear_grad->Stream(context);
    const CudaT* grad_output_data = reinterpret_cast<CudaT*>(grad_output->Data<T>());
    const Float8E4M3FN* trans_input_data = trans_input->Data<Float8E4M3FN>();
    const Float8E4M3FN* trans_weight_data = trans_weight->Data<Float8E4M3FN>();
    const float* scale_inv_fwd_data = scale_inv_fwd->Data<float>();

    CudaT* grad_input_data = grad_input ? reinterpret_cast<CudaT*>(grad_input->MutableData<T>()) : nullptr;
    CudaT* grad_weight_data = grad_weight ? reinterpret_cast<CudaT*>(grad_weight->MutableData<T>()) : nullptr;
    CudaT* grad_bias_data = grad_bias ? reinterpret_cast<CudaT*>(grad_bias->MutableData<T>()) : nullptr;
    return Status::OK();
  }
};

}  // namespace

Status FP8Linear::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();
  size_t input_rank = input_shape.NumDimensions();

  TensorShapeVector output_shape_vec = input_shape.AsShapeVector();
  output_shape_vec[input_rank - 1] = weight_shape[0];
  TensorShapeVector input_t_shape_vec = TensorShapeVector(2);
  input_t_shape_vec[0] = input_shape[input_rank - 1];
  input_t_shape_vec[1] = input_shape.SizeToDimension(input_rank - 1);
  TensorShapeVector weight_t_shape_vec = TensorShapeVector(2);
  weight_t_shape_vec[0] = weight_shape[1];
  weight_t_shape_vec[1] = weight_shape[0];
  TensorShapeVector scale_inv_fwd_shape_vec = TensorShapeVector(1);
  scale_inv_fwd_shape_vec[0] = 3;
  Tensor* output = context->Output(0, TensorShape(output_shape_vec));
  Tensor* trans_input = context->Output(1, TensorShape(input_t_shape_vec));
  Tensor* trans_weight = context->Output(2, TensorShape(weight_t_shape_vec));
  Tensor* scale_inv_fwd = context->Output(3, TensorShape(scale_inv_fwd_shape_vec));

  utils::MLTypeCallDispatcher<float, MLFloat16> t_disp(intput->GetElementType());
  return t_disp.InvokeRet<Status, DispatchFP8LinearImpl>(this, context, input, weight, bias, output, trans_input,
                                                         trans_weight, scale_inv_fwd, scaling.Scale(),
                                                         scaling.ScaleInv(), scaling.AmaxHistory());
}

Status FP8LinearGrad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* grad_output = context->Input<Tensor>(0);
  const Tensor* trans_input = context->Input<Tensor>(1);
  const Tensor* trans_weight = context->Input<Tensor>(2);
  const Tensor* scale_inv_fwd = context->Input<Tensor>(3);
  const TensorShape& grad_output_shape = grad_output->Shape();
  const TensorShape& trans_weight_shape = trans_weight->Shape();
  size_t grad_output_rank = grad_output_shape.NumDimensions();

  TensorShapeVector grad_input_shape_vec = grad_output_shape.AsShapeVector();
  grad_input_shape_vec[grad_output_rank - 1] = trans_weight_shape[0];
  TensorShapeVector grad_weight_shape_vec = TensorShapeVector(2);
  grad_weight_shape_vec[0] = trans_weight_shape[1];
  grad_weight_shape_vec[1] = trans_weight_shape[0];
  TensorShapeVector grad_bias_shape_vec = TensorShapeVector(1);
  grad_bias_shape_vec[0] = trans_weight_shape[1];
  Tensor* grad_input = context->Output(0, TensorShape(grad_input_shape_vec));
  Tensor* grad_weight = context->Output(1, TensorShape(grad_weight_shape_vec));
  Tensor* grad_bias = context->Output(2, TensorShape(grad_bias_shape_vec));

  utils::MLTypeCallDispatcher<float, MLFloat16> t_disp(intput->GetElementType());
  return t_disp.InvokeRet<Status, DispatchFP8LinearGradImpl>(
      this, context, grad_output, trans_input, trans_weight, scale_inv_fwd, grad_input, grad_weight, grad_bias,
      scaling.Scale(), scaling.ScaleInv(), scaling.AmaxHistory());
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
