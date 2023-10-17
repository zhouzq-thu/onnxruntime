/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

namespace onnxruntime {
namespace cuda {
namespace fp8 {

struct Empty {};

template <typename OType, typename IType>
__device__ inline OType gelu(const IType val, const Empty&) {
    const float cval = val;
    return cval * (0.5F + 0.5F * tanhf(cval * (0.79788456F + 0.03567741F * cval * cval)));
}

template <typename OType, typename IType>
__device__ inline OType dgelu(const IType val, const Empty&) {
    const float cval = val;
    const float tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
    return 0.5f * cval * ((1.f - tanh_out * tanh_out) *
                          (0.79788456f + 0.1070322243f * cval * cval)) +
           0.5f * (1.f + tanh_out);
}

template <typename OType, typename IType>
__device__ inline OType sigmoid(const IType val, const Empty&) {
    const float cval = val;
    return 1.f / (1.f + expf(-cval));
}

template <typename OType, typename IType>
__device__ inline OType dsigmoid(const IType val, const Empty& e) {
    const float cval = val;
    const float s = sigmoid<float, float>(cval, e);
    return s * (1.f - s);
}

template <typename OType, typename IType>
__device__ inline OType swish(const IType val, const Empty& e) {
    const float cval = val;
    return cval * sigmoid<float, float>(cval, e);
}

template <typename OType, typename IType>
__device__ inline OType dswish(const IType val, const Empty& e) {
    const float cval = val;
    return cval * dsigmoid<float, float>(cval, e) + sigmoid<float, float>(cval, e);
}

template <typename OType, typename IType>
__device__ inline OType relu(IType value, const Empty &) {
    return fmaxf(value, 0.f);
}

template <typename OType, typename IType>
__device__ inline OType drelu(IType value, const Empty &) {
    return value > 0.f ? 1.f : 0.f;
}


}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
