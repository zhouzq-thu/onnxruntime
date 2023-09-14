// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/utility/functional3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk_hgy.hpp"
// #include "ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#endif

#include "core/framework/float8.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

template <typename T>
constexpr bool always_false = false;

template <typename Y, typename X>
inline __host__ __device__ Y fast_type_convert(X x) {
  static_assert(always_false<X>, "not implemented");
  (void)x;
}

template <>
inline __host__ __device__ ck::half_t fast_type_convert<ck::half_t, ck::f8_t>(ck::f8_t x) {
  // https://github.com/ROCmSoftwarePlatform/triton/blob/0cc3f8b84a16892396f6e08a04991034d67e32b1/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L220-L233
  constexpr const uint16_t mask = 0x7fff;
  constexpr const uint16_t sign_mask = 0x8000;
  // constexpr const uint16_t exp_compensate = 0x2000;  // for float8_e4m3fn
  constexpr const uint16_t exp_compensate = 0x1c00;  // for float8_e4m3fnuz

  uint8_t x_u8 = reinterpret_cast<uint8_t&>(x);
  uint16_t x_u16 = static_cast<uint16_t>(x_u8) << 8;
  uint16_t exp = (x_u16 & mask) >> 1;
  uint16_t y = (x_u16 & sign_mask) | (exp + exp_compensate);
  return reinterpret_cast<ck::half_t&>(y);
}

#define PRINTF(...) if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0&& threadIdx.y == 0&& threadIdx.z == 0) printf(__VA_ARGS__)

struct Scale {
  explicit Scale(const float* dev_scale_ptr) : dev_scale_ptr{dev_scale_ptr} {}
  explicit Scale(float host_scale_value) : dev_scale_ptr{nullptr}, scale_value{host_scale_value} {}

  __forceinline__ __host__ __device__ void operator()(ck::half_t& y, const ck::f8_t& x) const {
    float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    // y = ck::type_convert<ck::half_t>(scale * ck::type_convert<ck::half_t>(x));
    y = ck::type_convert<ck::half_t>(scale * fast_type_convert<ck::half_t>(x));
  }

  __forceinline__ __host__ __device__ void operator()(ck::half_t& y0, ck::half_t& y1,
                                                      const ck::f8_t& x0, const ck::f8_t& x1) const {
    float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    constexpr const uint32_t mask = 0x7fff7fff;
    constexpr const uint32_t sign_mask = 0x80008000;
    // constexpr const uint32_t exp_compensate = 0x20002000;  // for float8_e4m3fn
    constexpr const uint32_t exp_compensate = 0x1c001c00;  // for float8_e4m3fnuz

    uchar4 x{0, x0, 0, x1};
    // PRINTF("%x %x %x %x\n", 0, x0, 0, x1);
    uint32_t x_u32 = reinterpret_cast<uint32_t&>(x);
    // PRINTF("%x\n", x_u32);
    uint32_t exp = (x_u32 & mask) >> 1;
    // PRINTF("%x\n", exp);
    uint32_t v = (x_u32 & sign_mask) | (exp + exp_compensate);
    // PRINTF("%x\n", v);
    half2 u = scale * reinterpret_cast<half2&>(v);
    // NOTE: don't use u.x, u.y ...
    y0 = u.data[0];
    y1 = u.data[1];
  }

#if 0
  __forceinline__ __host__ __device__ void operator()(ck::half_t& y0, ck::half_t& y1, ck::half_t& y2, ck::half_t& y3,
                                                      const ck::f8_t& x0, const ck::f8_t& x1, const ck::f8_t& x2, const ck::f8_t& x3) const {
#if 0
    float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    y0 = scale * fast_type_convert<ck::half_t>(x0);
    y1 = scale * fast_type_convert<ck::half_t>(x1);
    y2 = scale * fast_type_convert<ck::half_t>(x2);
    y3 = scale * fast_type_convert<ck::half_t>(x3);
#elif 0
    // NOTE: no improvement
    float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    constexpr const uint32_t mask = 0x7fff7fff;
    constexpr const uint32_t sign_mask = 0x80008000;
    // constexpr const uint32_t exp_compensate = 0x20002000;  // for float8_e4m3fn
    constexpr const uint32_t exp_compensate = 0x1c001c00;  // for float8_e4m3fnuz

    uchar4 x_0{0, x0, 0, x1};
    uchar4 x_1{0, x2, 0, x3};
    uint32_t x_u32_0 = reinterpret_cast<uint32_t&>(x_0);
    uint32_t x_u32_1 = reinterpret_cast<uint32_t&>(x_1);
    uint32_t exp_0 = (x_u32_0 & mask) >> 1;
    uint32_t exp_1 = (x_u32_1 & mask) >> 1;
    uint32_t v_0 = (x_u32_0 & sign_mask) | (exp_0 + exp_compensate);
    uint32_t v_1 = (x_u32_1 & sign_mask) | (exp_1 + exp_compensate);
    half2 u_0 = scale * reinterpret_cast<half2&>(v_0);
    half2 u_1 = scale * reinterpret_cast<half2&>(v_1);
    // NOTE: don't use u.x, u.y ...
    y0 = u_0.data[0];
    y1 = u_0.data[1];
    y2 = u_1.data[0];
    y3 = u_1.data[1];
#elif 1
    // NOTE: no improvement
    float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    constexpr const uint64_t mask = 0x7fff7fff7fff7fff;
    constexpr const uint64_t sign_mask = 0x8000800080008000;
    // constexpr const uint64_t exp_compensate = 0x2000200020002000;  // for float8_e4m3fn
    constexpr const uint64_t exp_compensate = 0x1c001c001c001c00;  // for float8_e4m3fnuz

    uchar4 x[2]{{0, x0, 0, x1}, {0, x2, 0, x3}};
    uint64_t x_u64 = reinterpret_cast<uint64_t&>(x);
    uint64_t exp = (x_u64 & mask) >> 1;
    uint64_t v = (x_u64 & sign_mask) | (exp + exp_compensate);
    half2* u = reinterpret_cast<half2*>(&v);
    half2 w0 = scale * u[0];
    half2 w1 = scale * u[1];
    // NOTE: don't use u.x, u.y ...
    y0 = w0.data[0];
    y1 = w0.data[1];
    y2 = w1.data[0];
    y3 = w1.data[1];
#endif
  }
#endif

  const float* dev_scale_ptr;
  float scale_value;
};

static_assert(std::is_invocable_r_v<void, Scale,
                                    ck::half_t&,
                                    const ck::f8_t&>);

static_assert(std::is_invocable_r_v<void, Scale,
                                    ck::half_t&, ck::half_t&,
                                    const ck::f8_t&, const ck::f8_t&>);

static_assert(!std::is_invocable_r_v<void, Scale,
                                    ck::half_t&, ck::half_t&, ck::half_t&, ck::half_t&,
                                    const ck::f8_t&, const ck::f8_t&, const ck::f8_t&, const ck::f8_t&>);

struct MacPassThrough {
  __forceinline__ __device__ void operator()(ck::half_t& y, const ck::f8_t& x) const {
    // store at low 8 bits! and pass through and then convert in MAC loop right before tensor core
    uint8_t x_uint8 = *reinterpret_cast<const uint8_t*>(&x);
    uint16_t x_uint16 = static_cast<uint16_t>(x_uint8);
    // x_uint16 |= x_uint16 << 8;
    y = *reinterpret_cast<ck::half_t*>(&x_uint16);
  }
};

struct MacScale {
  explicit MacScale(const float* dev_scale_ptr) : dev_scale_ptr{dev_scale_ptr} {}
  explicit MacScale(float host_scale_value) : dev_scale_ptr{nullptr}, scale_value{host_scale_value} {}

  __forceinline__ __device__ void operator()(ck::half_t& y, const ck::half_t& x) const {
    // load scale
    // float scale = nullptr == dev_scale_ptr ? scale_value : *dev_scale_ptr;
    float scale = scale_value;

    // extract low 8 bits as fp8
    uint16_t x_uint16 = *reinterpret_cast<const uint16_t*>(&x);
    uint8_t x_uint8 = static_cast<uint8_t>(x_uint16);
    ck::f8_t x_f8 = *reinterpret_cast<ck::f8_t*>(&x_uint8);

    // do the conversion and scale
    float x_actually_converted = ck::type_convert<float>(x_f8);
    float x_scaled = scale * x_actually_converted;
    y = ck::type_convert<ck::half_t>(x_scaled);
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

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_hgy(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitKHgy<
        Row, Row, Row, ck::f8_t, ck::half_t, ck::half_t, MacPassThrough, Nop, Nop, MacScale, Nop>>>& instances);

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_hgy(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitKHgy<
        Row, Row, Row, ck::half_t, ck::f8_t, ck::half_t, Nop, MacPassThrough, Nop, Nop, MacScale>>>&
        instances);

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

  for (auto num_split : {1, 4, 16, 64}) {
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
      auto type_string = impl->GetTypeString() + "_SplitK" + std::to_string(num_split);
      auto invoker = impl->MakeInvokerPointer();
      auto ck_gemm_op = [num_split, impl = std::move(impl), invoker = std::move(invoker)](const FP8GemmParams<TA, TB, TC>* params) -> Status {
        OpA op_a = CreateOp<CKTA>(params->scale_a, params->scale_a_dev);
        OpB op_b = CreateOp<CKTB>(params->scale_b, params->scale_b_dev);
        OpC op_c = CreateOp<CKTC>(params->scale_c, params->scale_c_dev);

        auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                             params->m, params->n, params->k,
                                             params->lda, params->ldb, params->ldc,
                                             op_a, op_b, op_c, num_split);
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                  impl->GetTypeString(), " does not support ", params->Signature());
        invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
        return Status::OK();
      };
      ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
    }
  }
  return ret;
}

template <typename CKT>
auto CreateOpAndMacOp(float scale, const float* dev_scale) {
  if constexpr (std::is_same_v<CKT, ck::f8_t>) {
    if (dev_scale != nullptr) {
      return std::tuple<MacPassThrough, MacScale>{MacPassThrough{}, MacScale(dev_scale)};
    } else {
      return std::tuple<MacPassThrough, MacScale>{MacPassThrough{}, MacScale(scale)};
    }
  } else {
    return std::tuple<Nop, Nop>{Nop{}, Nop{}};
  }
}

template <typename TA, typename TB, typename TC, typename ALayout, typename BLayout>
auto GetHgyCKF8SplitKGemmTypeStringAndOps() {
  using CKTA = typename CKDataTypeAdaptor<TA>::type;
  using CKTB = typename CKDataTypeAdaptor<TB>::type;
  using CKTC = typename CKDataTypeAdaptor<TC>::type;

  using OpA = std::conditional_t<std::is_same_v<CKTA, ck::f8_t>, MacPassThrough, Nop>;
  using OpB = std::conditional_t<std::is_same_v<CKTB, ck::f8_t>, MacPassThrough, Nop>;
  using OpC = std::conditional_t<std::is_same_v<CKTC, ck::f8_t>, Scale, Nop>;
  using MacOpA = std::conditional_t<std::is_same_v<CKTA, ck::f8_t>, MacScale, Nop>;
  using MacOpB = std::conditional_t<std::is_same_v<CKTB, ck::f8_t>, MacScale, Nop>;

  using DeviceGemm = ck::tensor_operation::device::DeviceGemmSplitKHgy<
      ALayout, BLayout, Row,
      CKTA, CKTB, CKTC,
      OpA, OpB, OpC,
      MacOpA, MacOpB>;

  std::vector<std::pair<std::string, Op<FP8GemmParams<TA, TB, TC>>>> ret;

  for (auto num_split : {1, 4, 16, 64}) {
    std::vector<std::unique_ptr<DeviceGemm>> instances{};
    // FIXME: only supports fp8_fp16_fp16_row_row_row and fp16_fp8_fp16_row_row_row now.
    if constexpr (std::is_same_v<CKTA, ck::f8_t> && std::is_same_v<CKTB, ck::half_t> && std::is_same_v<CKTC, ck::half_t>) {
      add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_hgy(instances);
    } else if constexpr (std::is_same_v<CKTA, ck::half_t> && std::is_same_v<CKTB, ck::f8_t> && std::is_same_v<CKTC, ck::half_t>) {
      add_device_gemm_xdl_splitk_f16_f8_f16_mk_kn_mn_instances_hgy(instances);
    } else {
      // static_assert(false, "no instances");
    }
    for (auto&& impl : instances) {
      auto type_string = impl->GetTypeString() + "_SplitK" + std::to_string(num_split);
      auto invoker = impl->MakeInvokerPointer();
      auto ck_gemm_op = [num_split, impl = std::move(impl), invoker = std::move(invoker)](const FP8GemmParams<TA, TB, TC>* params) -> Status {
        auto [op_a, mac_op_a] = CreateOpAndMacOp<CKTA>(params->scale_a, params->scale_a_dev);
        auto [op_b, mac_op_b] = CreateOpAndMacOp<CKTB>(params->scale_b, params->scale_b_dev);
        OpC op_c = CreateOp<CKTC>(params->scale_c, params->scale_c_dev);

        auto arg = impl->MakeArgumentPointer(params->a, params->b, params->c,
                                             params->m, params->n, params->k,
                                             params->lda, params->ldb, params->ldc,
                                             op_a, op_b, op_c, num_split, mac_op_a, mac_op_b);
        TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg.get()),
                                                  impl->GetTypeString(), " does not support ", params->Signature());
        invoker->Run(arg.get(), StreamConfig{params->StreamHandle()});
        return Status::OK();
      };
      ret.emplace_back(std::make_pair(std::move(type_string), std::move(ck_gemm_op)));
    }
  }
  return ret;
}

#endif  // USE_COMPOSABLE_KERNEL

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
