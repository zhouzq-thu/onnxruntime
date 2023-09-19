// SPDX-License-Identifier: MIT
// Modifications Copyright (c) Microsoft.
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

#include "core/providers/rocm/tunable/gemm_ck_fp8.cuh"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

using F8 = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

#define DeviceGemmXdlSplitKCShuffle ck::tensor_operation::device::DeviceGemmXdlSplitKCShuffle


// The derived version is not simply double ABlockTransferSrcScalarPerVector, as it is already large enough.
// Instead, I am trying to eliminate the L2 cacheline wasting: A is K contiguous, KPerBlock=32, MI250x L2 is 64bytes however.
// Trying to fix the 50% L2 cache wasting (hypothetically). But not going well...

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_derived = std::tuple<
    // clang-format off
           //#########################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer| Compute|
           //#########################| Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Type|
           //#########################|     |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|        |
           //#########################|     |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |        |
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,   128,   128,     4, 16,   32,   32,  4/2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,    64,   256,     4, 16,   32,   32,  2/2,    4,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    64,   128,     4, 16,   32,   32,  4/2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,    32,   192,     4, 16,   32,   32,  1/2,    3,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 48, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,    96,    64,     4, 16,   32,   32,  3/2,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,    64,   128,     4, 16,   32,   32,  2/2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    64,    64,     4, 16,   32,   32,  2/2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    32,   128,     4, 16,   32,   32,  2/2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,    64,    64,     4, 16,   32,   32,  2/2,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              1,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   256,    32,   128,     4, 16,   32,   32,  1/2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    16,   192,     4, 16,   32,   32,  1/2,    3,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 24, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    96,    32,     4, 16,   32,   32,  3/2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              1,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    16,    64,     4, 16,   32,   32,  1/2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              2,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    32,    32,     4, 16,   32,   32,  1/2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              1,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>,
        //    DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    16,   128,     4, 16,   32,   32,  1/2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              4,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8,    F16>,
           DeviceGemmXdlSplitKCShuffle<   F8,   F16,   F16,     F32,     Row,      Row,    Row,       Scale, PassThrough, PassThrough, GemmMNPadding,   128,    64,    32,     4, 16,   32,   32,  2/2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,             16,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             2,              1,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8,    F16>
    // clang-format on
    >;

void add_device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_derived(
    std::vector<std::unique_ptr<ck::tensor_operation::device::DeviceGemmSplitK<
        Row, Row, Row, F8, F16, F16, Scale, PassThrough, PassThrough>>>&
        instances) {
// FIXME: the derivation does not going well due to the memory layout and hardware constraint
//   ck::tensor_operation::device::instance::add_device_operation_instances(
//       instances,
//       device_gemm_xdl_splitk_f8_f16_f16_mk_kn_mn_instances_derived{});
}

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
