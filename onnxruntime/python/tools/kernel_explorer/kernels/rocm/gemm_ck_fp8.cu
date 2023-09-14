// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/gemm_ck_fp8.cuh"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using namespace onnxruntime::rocm::tunable::blas;
using namespace onnxruntime::rocm::tunable::blas::internal;

namespace py = pybind11;

namespace onnxruntime {

#ifdef USE_COMPOSABLE_KERNEL
template <typename TA, typename TB, typename TC, typename ALayout, typename BLayout>
class CKFP8Gemm : public IKernelExplorer {
 public:
  CKFP8Gemm(BlasOp opa, BlasOp opb,
            int64_t m, int64_t n, int64_t k,
            DeviceArray& a, int64_t lda, float scale_a,
            DeviceArray& b, int64_t ldb, float scale_b,
            DeviceArray& c, int64_t ldc, float scale_c)
      : params_{} {
    auto supports_a = opa == BlasOp::N ? std::is_same_v<ALayout, Row> : std::is_same_v<ALayout, Col>;
    auto supports_b = opb == BlasOp::N ? std::is_same_v<BLayout, Row> : std::is_same_v<BLayout, Col>;
    ORT_ENFORCE(supports_a && supports_b);

    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    // rocblas handle is not used for ck
    params_.handle = nullptr;
    params_.opa = opa;
    params_.opb = opb;
    params_.m = m;
    params_.n = n;
    params_.k = k;
    params_.a = static_cast<TA*>(a.ptr());
    params_.lda = lda;
    if constexpr (std::is_same_v<TA, Float8E4M3FNUZ>) {
      params_.scale_a = scale_a;
    }
    params_.b = static_cast<TB*>(b.ptr());
    params_.ldb = ldb;
    if constexpr (std::is_same_v<TB, Float8E4M3FNUZ>) {
      params_.scale_b = scale_b;
    }
    params_.c = static_cast<TC*>(c.ptr());
    params_.ldc = ldc;
    if constexpr (std::is_same_v<TC, Float8E4M3FNUZ>) {
      params_.scale_c = scale_c;
    }

    for (auto&& [type_string, op] : GetCKF8SplitKGemmTypeStringAndOps<TA, TB, TC, ALayout, BLayout>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
    for (auto&& [type_string, op] : GetHgyCKF8SplitKGemmTypeStringAndOps<TA, TB, TC, ALayout, BLayout>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
    ORT_ENFORCE(!ops_.empty());
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = FP8GemmParams<TA, TB, TC>;
  using OpT = Op<ParamsT>;
  ParamsT params_{};
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};

#define REGISTER_OP_COMMON(registered_name, tpl, dta, dtb, dtc, alayout, blayout) \
  py::class_<tpl<dta, dtb, dtc, alayout, blayout>>(m, registered_name)            \
      .def("SetRepeats", &tpl<dta, dtb, dtc, alayout, blayout>::SetRepeats)       \
      .def("Profile", &tpl<dta, dtb, dtc, alayout, blayout>::Profile)             \
      .def("Run", &tpl<dta, dtb, dtc, alayout, blayout>::Run)                     \
      .def("ListOps", &tpl<dta, dtb, dtc, alayout, blayout>::ListOps)             \
      .def("SelectOp", &tpl<dta, dtb, dtc, alayout, blayout>::SelectOp)

#define REGISTER_CKFP8GEMM(registered_name, dta, dtb, dtc, alayout, blayout)      \
  REGISTER_OP_COMMON(registered_name, CKFP8Gemm, dta, dtb, dtc, alayout, blayout) \
      .def(py::init<BlasOp, BlasOp, int64_t, int64_t, int64_t,                    \
                    DeviceArray&, int64_t, float,                                 \
                    DeviceArray&, int64_t, float,                                 \
                    DeviceArray&, int64_t, float>());

KE_REGISTER(m) {
  REGISTER_CKFP8GEMM("CKFP8Gemm_f8_half_half_NN", Float8E4M3FNUZ, half, half, Row, Row);
  REGISTER_CKFP8GEMM("CKFP8Gemm_half_f8_half_NN", half, Float8E4M3FNUZ, half, Row, Row);
}
#endif  // USE_COMPOSABLE_KERNEL

}  // namespace onnxruntime
