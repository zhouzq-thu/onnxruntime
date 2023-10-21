#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// T is output type, U means compute type in this file.
// The gamma and beta are stored in compute type U in this op.
template <typename T, typename U>
cudaError_t DispatchGroupNormKernel(cudaStream_t stream, const int64_t num_instances,
                                 const int64_t norm_size, const int64_t channel_size,
                                 const int64_t spatial_size, const double epsilon, const T* x,
                                 const U* gamma, const U* beta, T* y, U* mean, U* inv_variance,
                                 bool channels_first, bool has_silu_activation,
                                 const int sm_count);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
