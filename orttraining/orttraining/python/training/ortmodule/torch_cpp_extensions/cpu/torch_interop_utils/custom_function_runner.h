// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/DLConvertor.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/autograd/python_cpp_function.h>
// #include <aten/src/ATen/dlpack.h>

#include <chrono>
#include <iostream>

void clear_grad_fns_for_next_edges(at::Tensor& target,
                                   std::vector<at::Tensor>& saved_tensors);

void register_grad_fn_and_remove_from_autograd(size_t ctx_address, at::Tensor target);

void unregister_grad_fn(py::object ctx);

// Supposed to be cleared on python program exit to resolve the following issue:
// When training program exits, PyNodeSharedPointerPool destructor is called, if grad_fns_ is not empty,
// PyNode::release_variables() will be called.
// (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/python_function.cpp#L168)
// On The other hand, there is a known issue when acquiring GIL in pybind11 destructors, there will be
// probably a deadlock issue. (https://github.com/pybind/pybind11/issues/1446)
// The resolution here, we remove all maintained states before the program exits.

// A known existing issue: when forward functions are called repeatedly without corresponding backward calls,
// grad functions keep accumulating without releasing, there might be memory (bound to those gradient functions) leaks.
// Ideally this usually won't happen in real training cases, so it should be fine.

// We CANNOT explicitly clear grad functions before each forward pass to mitigate the known issue above.
// For example:
//     loss1 = forward_run(inputs1)
//     loss2 = forward_run(inputs2)
//     loss = loss1 + loss2
//     loss.backward()
// If we clear grad functions at the beginning of the second `forward_run`, when `loss.backward()` runs,
// the backward path of `loss1` will fail to run PythonOpGrad ops (if there is any).
void clear_all_grad_fns();

bool get_materialize_grads(at::Tensor target);

std::vector<bool> are_tensors_marked_as_dirty(at::Tensor& target, std::vector<at::Tensor>& tensors_to_check);

class CustomFuncOpKernelInfo {
 public:
  CustomFuncOpKernelInfo(const std::string& invoke_id, bool safe_run) {
    kernel_invoke_id = invoke_id;
    safe_run_enabled = safe_run;
  }

  // kernel_invoke_id is a string contains session thread id, op kernel creation time stamp in ms, a random int,
  // and address of op_kernel pointer. This can guarantee the uniqueness of the key in case of multiple
  // instances of a same named PythonOp/PythonOpGrad in one session, or multiple sessions.
  std::string kernel_invoke_id;

  // For the tensors generated from ORT backend, there is special handling here:
  // 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
  //    all such tensors will be cloned in case they are saved in context (but ORT backend is not aware of the
  //    reference, may release the content of the tensor before it is needed in backward). Once
  //    `autograd.Function.apply` completes, by checking the existence of the tensor in the saved_tensors,
  //    `_GlobalOpKernelInfoMap` is updated to save the input indices that are saved in context.
  // 2. For the subsequent runs, if the input index is in `tensor_input_indices_to_save_in_ctx`, the tensor
  //    will be cloned before fed into `autograd.Function.apply` as input.
  std::unordered_map<int, bool> tensor_input_indices_to_save_in_ctx;

  // To align with PyTorch `ctx.set_materialize_grads(False|True)`, default to be true.
  // materialize_grads_config is a map from output index to (device, dtype, shape) of the output tensor, used
  // for materializing the gradient of the output tensor in backward.
  bool materialize_grads{true};
  // key: output index, value: (shape, tensor options including device, layerout, data types, etc)
  std::unordered_map<size_t, std::tuple<std::vector<int64_t>, c10::TensorOptions>> materialize_grads_config;

  // For the tensors generated from ORT backend, there is special handling here:
  // 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
  //    all such tensors will be cloned (with gradient) in case they are marked as dirty (if not cloned, but marked
  //    as dirty, PyTorch will complain the tensor is a leaf, should not be used for inplace update). Once
  //    `autograd.Function.apply` completes, by checking the existence of the tensor in the dirty_tensors,
  //    `_GlobalOpKernelInfoMap` is updated to save the input indices that are marked as dirty.
  // 2. For the subsequent runs, if the input index is in `tensor_input_indices_for_mark_dirty`, the tensor
  //    will be cloned (with gradient) before fed into `autograd.Function.apply` as input.
  std::unordered_map<int, bool> tensor_input_indices_for_mark_dirty;

  // A list of output indices that needs to be clone before returned, due to inplace update analysis.
  std::vector<size_t> output_indices_for_clone;

  bool is_first_run{true};
  bool safe_run_enabled{false};
};

std::vector<PyObject*> custom_function_forward_runner(const char* func_name_char,
                                                      void* callback,
                                                      const std::vector<int64_t>& requires_grad_flags,
                                                      const std::vector<int64_t>& tensor_type_flags,
                                                      const bool is_training_mode,
                                                      const std::vector<int64_t>& inplace_map,
                                                      const char* kernel_invoke_id_char,
                                                      const std::vector<PyObject*>& tensor_args);

std::vector<PyObject*> custom_function_backward_runner(const char* func_name_char,
                                                       void* callback,
                                                       const std::vector<int64_t>& requires_grad_flags,
                                                       const std::vector<int64_t>& tensor_type_flags,
                                                       const bool is_training_mode,
                                                       const std::vector<int64_t>& inplace_map,
                                                       const char* kernel_invoke_id_char,
                                                       const std::vector<PyObject*>& args);
