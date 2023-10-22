// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_function_runner.h"
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

size_t get_custom_function_forward_runner() { return reinterpret_cast<size_t>(&custom_function_forward_runner); }
size_t get_custom_function_backward_runner() { return reinterpret_cast<size_t>(&custom_function_backward_runner); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("register_grad_fn_and_remove_from_autograd", &register_grad_fn_and_remove_from_autograd,
  //       "Increase grad_fn shared pointer reference.");
  // m.def("unregister_grad_fn", &unregister_grad_fn, "Release grad_fn shared pointer reference.");
  m.def("clear_all_grad_fns", &clear_all_grad_fns, "Clear all grad_fn shared pointer references.");
  // m.def("clear_grad_fns_for_next_edges", &clear_grad_fns_for_next_edges,
  //       "Remove reference on next edges' gradient functions.");
  // m.def("get_materialize_grads", &get_materialize_grads, "Return whether materialize_grads is enabled or not.");
  // m.def("are_tensors_marked_as_dirty", &are_tensors_marked_as_dirty, "Return whether the tensors are marked dirty or not.");
  // m.def("forward_runner", &forward_runner, "Forward runner.");
  // m.def("_finalize_training_mode_forward", &_finalize_training_mode_forward, "Finalize training mode forward.");
  // m.def("complete_forward_runner", &complete_forward_runner, "Complete forward runner.");
  // m.def("backward_runner", &backward_runner, "Backward runner.");
  m.def("get_custom_function_forward_runner", &get_custom_function_forward_runner, "Get custom function forward runner.");
  m.def("get_custom_function_backward_runner", &get_custom_function_backward_runner, "Get custom function backward runner.");
}
