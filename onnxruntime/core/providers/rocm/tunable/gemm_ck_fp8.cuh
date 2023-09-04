// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif

#include "core/framework/float8.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

template <typename TA, typename TB, typename TC>
struct F8GemmParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
  }

  rocblas_handle handle;
  BlasOp opa;
  BlasOp opb;
  int64_t m;
  int64_t n;
  int64_t k;
  float a_scale{};
  const float* a_scale_dev{};
  const TA* a;
  int64_t lda;
  float b_scale{};
  const float* b_scale_dev{};
  const TB* b;
  int64_t ldb;
  TC* c;
  float c_scale{};
  const float* c_scale_dev{};
  int64_t ldc;
};

namespace internal {

struct Scale {
  explicit Scale(const float* dev_ptr) : dev_ptr{dev_ptr} {
  }
  explicit Scale(float host_value) : dev_ptr{nullptr}, value{host_value} {
  }

  __forceinline__ __device__ void operator()(float& y, const ck::f8_t& x) const {
    float scale;
    if (dev_ptr) {
      scale = ck::type_convert<float>(*dev_ptr);
    } else {
      scale = ck::type_convert<float>(value);
    }
    y = scale * ck::type_convert<float>(x);
  }

  const float* dev_ptr;
  float value;
};

#ifdef USE_COMPOSABLE_KERNEL

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Nop = ck::tensor_operation::element_wise::PassThrough;

template <typename CKT>
auto CreateOp(float scale, const float* scale_dev) {
  if constexpr (std::is_same_v<CKT, ck::f8_t>) {
    if (scale_dev != nullptr) {
      return Scale(scale_dev);
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

  using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
      ALayout, BLayout, Row,
      CKTA, CKTB, CKTC,
      OpA, OpB, OpC>;
  using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;

  std::vector<std::pair<std::string, Op<F8GemmParams<TA, TB, TC>>>> ret;
  for (auto&& impl : InstanceFactory::GetInstances()) {
    auto type_string = impl->GetTypeString();
    auto invoker = impl->MakeInvokerPointer();
    auto ck_gemm_op = [impl = std::move(impl), invoker = std::move(invoker)](const F8GemmParams<TA, TB, TC>* params) -> Status {
      OpA op_a = CreateOp<CKTA>(params->a_scale, params->a_scale_dev);
      OpA op_b = CreateOp<CKTB>(params->b_scale, params->b_scale_dev);
      OpA op_c = CreateOp<CKTC>(params->c_scale, params->c_scale_dev);

      auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                           params->m, params->n, params->k,
                                           params->lda, params->ldb, params->ldc,
                                           op_a, op_b, op_c);
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
