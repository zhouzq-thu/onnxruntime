// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/float16.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/rocm/bert/gemm_ck_fp8.cuh"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;
using namespace onnxruntime::rocm::tunable::blas;

class Fp8MatMul final : public RocmKernel {
 public:
  Fp8MatMul(const OpKernelInfo& info) : RocmKernel(info) {
    info.GetAttrOrDefault<float>("scale_a", &scale_a_, 1.0f);
    info.GetAttrOrDefault<float>("scale_b", &scale_b_, 1.0f);

    tunable_op_fp8_fp16_fp16_ = std::make_unique<decltype(tunable_op_fp8_fp16_fp16_)::element_type>();
    tunable_op_fp16_fp8_fp16_ = std::make_unique<decltype(tunable_op_fp16_fp8_fp16_)::element_type>();
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status ComputeFp8Fp16Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* B, Tensor* C) const;
  Status ComputeFp16Fp8Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* B, Tensor* C) const;

  std::unique_ptr<F8GemmTunableOp<Float8E4M3FNUZ, MLFloat16, MLFloat16, internal::Row, internal::Row>> tunable_op_fp8_fp16_fp16_;
  std::unique_ptr<F8GemmTunableOp<MLFloat16, Float8E4M3FNUZ, MLFloat16, internal::Row, internal::Row>> tunable_op_fp16_fp8_fp16_;
  float scale_a_;
  float scale_b_;
};

Status Fp8MatMul::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  auto a_shape = A->Shape();
  auto b_shape = B->Shape();
  ORT_ENFORCE(a_shape.NumDimensions() >= 2 && b_shape.NumDimensions() == 2);  // is in form of input @ weight
  ORT_ENFORCE(a_shape[a_shape.NumDimensions() - 1] == b_shape[0]);            // k is compatiable

  TensorShapeVector output_shape = a_shape.AsShapeVector();
  output_shape[output_shape.size() - 1] = b_shape[b_shape.NumDimensions() - 1];
  Tensor* Y = ctx->Output(0, output_shape);

  if (A->IsDataType<Float8E4M3FNUZ>()) {
    return ComputeFp8Fp16Fp16(ctx, A, B, Y);
  } else if (B->IsDataType<Float8E4M3FNUZ>()) {
    return ComputeFp16Fp8Fp16(ctx, A, B, Y);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unhandled type combination of F8 GEMM");
}

Status Fp8MatMul::ComputeFp8Fp16Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* B, Tensor* C) const {
  ORT_ENFORCE(A->IsDataType<Float8E4M3FNUZ>() && B->IsDataType<MLFloat16>());
  onnxruntime::rocm::tunable::blas::FP8GemmParams<Float8E4M3FNUZ, MLFloat16, MLFloat16> params{};

  ORT_ENFORCE(false, "A @ B for weight @ input is not implemented");
}

Status Fp8MatMul::ComputeFp16Fp8Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* B, Tensor* C) const {
  ORT_ENFORCE(A->IsDataType<MLFloat16>() && B->IsDataType<Float8E4M3FNUZ>());

  auto a_shape = A->Shape();
  auto b_shape = B->Shape();

  auto m = a_shape.Slice(0, a_shape.NumDimensions() - 1).Size();
  auto k = a_shape[a_shape.NumDimensions() - 1];
  auto n = b_shape[b_shape.NumDimensions() - 1];

  onnxruntime::rocm::tunable::blas::FP8GemmParams<MLFloat16, Float8E4M3FNUZ, MLFloat16> params{};
  params.tuning_ctx = GetTuningContext();
  params.stream = ctx->GetComputeStream();
  params.handle = GetRocblasHandle(ctx);
  params.opa = tunable::blas::BlasOp::NonTrans;
  params.opb = tunable::blas::BlasOp::NonTrans;

  params.m = m;
  params.n = n;
  params.k = k;

  params.a = static_cast<const MLFloat16*>(A->DataRaw());
  params.lda = k;
  params.scale_a = 1.0f;  // NOTE: not used

  params.b = static_cast<const Float8E4M3FNUZ*>(B->DataRaw());
  params.ldb = n;
  params.scale_b = scale_b_;

  params.c = static_cast<MLFloat16*>(C->MutableDataRaw());
  params.ldc = n;
  params.scale_c = 1.0f;  // NOTE: not used

  return (*tunable_op_fp16_fp8_fp16_)(&params);
}

ONNX_OPERATOR_KERNEL_EX(
    Fp8MatMul,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, Float8E4M3FNUZ>())
        .TypeConstraint("T2", BuildKernelDefConstraints<MLFloat16, Float8E4M3FNUZ>())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16>()),
    Fp8MatMul);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
