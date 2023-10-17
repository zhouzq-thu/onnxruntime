/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "orttraining/training_ops/cuda/fp8/gemm/gemm.h"

#include <cublasLt.h>
#include <cublas_v2.h>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

namespace {

bool IsFP8Type(cudaDataType_t dtype) { return dtype == CUDA_R_8F_E4M3 || dtype == CUDA_R_8F_E5M2; }

cudaDataType_t GetCudaDType(const DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kFloat32:
      return CUDA_R_32F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    default:
      throw std::runtime_error("Invalid type");
  }
}

}  // namespace

FP8GemmWorkspace::FP8GemmWorkspace() { CUDA_CALL_THROW(cudaMalloc(&workspace, workspace_size_bytes)); }

FP8GemmWorkspace::~FP8GemmWorkspace() { CUDA_CALL_THROW(cudaFree(workspace)); }

Status CublasGemm(cudaStream_t stream, const void* a_data, const void* b_data, const void* bias_data, void* output_data,
                  void* pre_gelu_out_data, cudaDataType_t a_type, cudaDataType_t b_type, cudaDataType_t bias_type,
                  cudaDataType_t output_type, cudaDataType_t aux_type, void* a_scale_inv, void* b_scale_inv,
                  void* scale, void* amax, int m, int n, int k, int lda, int ldb, int ldd, cublasOperation_t transa,
                  cublasOperation_t transb, bool grad, void* workspace, size_t workspaceSize, bool accumulate,
                  bool use_split_accumulator, int math_sm_count) {
  void* c_data = output_data;
  void* d_data = output_data;
  const bool bias = bias_data != nullptr;
  const bool gelu = pre_gelu_out_data != nullptr;
  const bool use_fp8 = IsFP8Type(a_type) || IsFP8Type(b_type);

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  cublasLtHandle_t handle;
  CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t)ldd;

  // Use TF32 only for pure FP32 GEMM.
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (a_type == CUDA_R_32F && b_type == CUDA_R_32F && output_type == CUDA_R_32F) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Adesc, a_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&Bdesc, b_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Ddesc, output_type, m, n, ldd));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  // Set math SM count
  if (math_sm_count != 0) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                          &math_sm_count, sizeof(math_sm_count)));
  }

  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode,
                                                          sizeof(fastAccuMode)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                          &a_scale_inv, sizeof(a_scale_inv)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                          &b_scale_inv, sizeof(b_scale_inv)));
    if (IsFP8Type(output_type)) {
      // Accumulation mode not supported for FP8 output
      c_data = nullptr;
      CUBLAS_RETURN_IF_ERROR(
          cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scale, sizeof(scale)));
      CUBLAS_RETURN_IF_ERROR(
          cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax, sizeof(amax)));
      // For FP8 output, cuBLAS requires C_type to be same as bias_type
      CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, bias_type, m, n, ldd));
    } else {
      CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, output_type, m, n, ldd));
    }
    if (bias) {
      CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                            &bias_type, sizeof(bias_type)));
    }
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&Cdesc, output_type, m, n, ldd));
  }

  if (bias && gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_data,
                                                          sizeof(bias_data)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                          &pre_gelu_out_data, sizeof(pre_gelu_out_data)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                          &ld_gelumat, sizeof(ld_gelumat)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                                          &aux_type, sizeof(aux_type)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = CUBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_data,
                                                          sizeof(bias_data)));
  } else if (gelu) {
    if (grad) {
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = CUBLASLT_EPILOGUE_GELU_AUX;
    }
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                          &pre_gelu_out_data, sizeof(pre_gelu_out_data)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                          &ld_gelumat, sizeof(ld_gelumat)));
  }

  CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspaceSize, sizeof(workspaceSize)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                                                        1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(handle, operationDesc, static_cast<const void*>(&one), /* alpha */
                                        a_data,                                                /* A */
                                        Adesc, b_data,                                         /* B */
                                        Bdesc, static_cast<const void*>(&beta),                /* beta */
                                        c_data,                                                /* C */
                                        Cdesc, d_data,                                         /* D */
                                        Ddesc, &heuristicResult.algo,                          /* algo */
                                        workspace,                                             /* workspace */
                                        workspaceSize, stream));                               /* stream */

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Ddesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescDestroy(operationDesc));
  return Status::OK();
}

Status FP8Gemm(cudaStream_t stream, const SimpleTensor input_a, const SimpleTensor input_b,
               const SimpleTensor input_bias, SimpleTensor output_d, SimpleTensor pre_gelu_out, void* a_scale_inv,
               void* b_scale_inv, void* scale, void* amax, bool trans_a, bool trans_b, bool grad,
               bool use_split_accumulator, int math_sm_count) {
  const int m = trans_a ? input_a.shape[0] : input_a.shape[1];
  const int k = trans_a ? input_a.shape[1] : input_a.shape[0];
  const int n = trans_b ? input_b.shape[1] : input_b.shape[0];
  int lda, ldb, ldd;
  if (trans_a && !trans_b) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!trans_a && !trans_b) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!trans_a && trans_b) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    ORT_THROW("TT layout not allowed.");
  }

  return CublasGemm(stream, input_a.data_ptr, input_b.data_ptr, input_bias.data_ptr, output_d.data_ptr,
                    pre_gelu_out.data_ptr, GetCudaDType(input_a.dtype), GetCudaDType(input_b.dtype),
                    GetCudaDType(input_bias.dtype), GetCudaDType(output_d.dtype), GetCudaDType(pre_gelu_out.dtype),
                    a_scale_inv, b_scale_inv, scale, amax, m, n, k, lda, ldb, ldd, trans_a, trans_b, grad,
                    FP8GemmWorkspace::Instance().GetWorkspace(), FP8GemmWorkspace::Instance().SizeInBytes(), false,
                    use_split_accumulator, math_sm_count);
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
