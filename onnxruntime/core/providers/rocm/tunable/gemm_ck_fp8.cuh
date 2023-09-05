// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif

#include "core/framework/float8.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

struct Scale {
  explicit Scale(const float* dev_scale_ptr) : dev_scale_ptr{dev_scale_ptr} {}
  explicit Scale(float host_scale_value) : dev_scale_ptr{nullptr}, scale_value{host_scale_value} {}

  __forceinline__ __device__ void operator()(ck::half_t& y, const ck::f8_t& x) const {
    float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    y = ck::type_convert<ck::half_t>(scale * ck::type_convert<float>(x));
  }

  const float* dev_scale_ptr;
  float scale_value;
};

namespace blas {

template <typename TA, typename TB, typename TC>
struct FP8GemmParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
  }

  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  float scale_a{};
  const float* scale_a_dev{};
  const TA* a;
  int64_t lda;
  float scale_b{};
  const float* scale_b_dev{};
  const TB* b;
  int64_t ldb;
  TC* c;
  float scale_c{};
  const float* scale_c_dev{};
  int64_t ldc;
};

namespace internal {

#ifdef USE_COMPOSABLE_KERNEL

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, ck::f8_t, ck::half_t, ck::half_t, Scale, Nop, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, ck::half_t, ck::f8_t, ck::half_t, Nop, Scale, Nop>>>& instances);

template <typename CKT>
auto CreateOp(float scale, const float* dev_scale) {
  if constexpr (std::is_same_v<CKT, ck::f8_t>) {
    if (dev_scale != nullptr) {
      return Scale(dev_scale);
    } else {
      return Scale(scale);
    }
  } else {
    return Nop{};
  }
}

template <typename TA, typename TB, typename TC, typename ALayout, typename BLayout>
auto GetCKF8SplitKGemmTypeStringAndOps() {
  using CKTA = typename CKDataTypeAdaptor<TA>::type;
  using CKTB = typename CKDataTypeAdaptor<TB>::type;
  using CKTC = typename CKDataTypeAdaptor<TC>::type;

  using OpA = std::conditional_t<std::is_same_v<CKTA, ck::f8_t>, Scale, Nop>;
  using OpB = std::conditional_t<std::is_same_v<CKTB, ck::f8_t>, Scale, Nop>;
  using OpC = std::conditional_t<std::is_same_v<CKTC, ck::f8_t>, Scale, Nop>;

  using DeviceGemm = ck::tensor_operation::device::DeviceGemmSplitK<
      ALayout, BLayout, Row,
      CKTA, CKTB, CKTC,
      OpA, OpB, OpC>;

  std::vector<std::pair<std::string, Op<FP8GemmParams<TA, TB, TC>>>> ret;
  std::vector<std::unique_ptr<DeviceGemm>> instances{};
  // FIXME: only supports fp8_fp16_fp16_row_row_row and fp16_fp8_fp16_row_row_row now.
  if constexpr (std::is_same_v<CKTA, ck::f8_t> && std::is_same_v<CKTB, ck::half_t> && std::is_same_v<CKTC, ck::half_t>) {
    add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances(instances);
  } else if constexpr (std::is_same_v<CKTA, ck::half_t> && std::is_same_v<CKTB, ck::f8_t> && std::is_same_v<CKTC, ck::half_t>) {
    add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances(instances);
  } else {
    // static_assert(false, "no instances");
  }
  for (auto&& impl : instances) {
    auto type_string = impl->GetTypeString();
    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemm_op = [impl = std::move(impl), invoker = std::move(invoker)](const FP8GemmParams<TA, TB, TC>* params) -> Status {
      OpA op_a = CreateOp<CKTA>(params->scale_a, params->scale_a_dev);
      OpB op_b = CreateOp<CKTB>(params->scale_b, params->scale_b_dev);
      OpC op_c = CreateOp<CKTC>(params->scale_c, params->scale_c_dev);

      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           op_a, op_b, op_c, 4);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                impl->GetTypeString(), " does not support ", params->Signature());
      invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
  }
  return ret;
}

#endif  // USE_COMPOSABLE_KERNEL

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
