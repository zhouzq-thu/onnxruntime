/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace onnxruntime {
namespace cuda {
namespace fp8 {

template <typename T>
constexpr T DIVUP(const T& x, const T& y) {
  return (((x) + ((y)-1)) / (y));
}

using byte = uint8_t;
using int32 = int32_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

inline size_t product(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (const auto& elem : shape) {
    ret *= elem;
  }
  return ret;
}

inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

enum class DType {
  kByte = 0,
  kInt32 = 1,
  kInt64 = 2,
  kFloat32 = 3,
  kFloat16 = 4,
  kBFloat16 = 5,
  kFloat8E4M3 = 6,
  kFloat8E5M2 = 7,
  kNumTypes
};

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
