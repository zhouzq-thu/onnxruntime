// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ctx_pool.h"
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

void register_grad_fn_and_remove_from_autograd(size_t ctx_address, at::Tensor target) {
  // //std::cout << "register_grad_fn_and_remove_from_autograd ctx_address: " << ctx_address << std::endl;
  // py::gil_scoped_release release;
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  PyNodeSharedPointerPool::GetInstance().RegisterGradFuncAndRemoveFromAutoGrad(ctx_address, autograd_meta);
}

void clear_grad_fns_for_next_edges(at::Tensor& target, std::vector<at::Tensor>& saved_tensors) {
  // For leaf tensor, there will be a AccumulateGrad (gradient function) created, which owns a
  // reference to the tensor.
  // For any user saved tensors (with save_for_backward), if the tensor is leaf, we put the map
  // {AccumulateGrad*, Tensor*} into grad_fn_to_tensor_map.
  std::unordered_map<torch::autograd::Node*, at::Tensor*> grad_fn_to_tensor_map;
  for (auto& t : saved_tensors) {
    auto grad_fn = t.grad_fn();
    if (!grad_fn) {
      grad_fn = torch::autograd::impl::try_get_grad_accumulator(t);
      if (grad_fn) {
        TORCH_CHECK(grad_fn_to_tensor_map.find(grad_fn.get()) == grad_fn_to_tensor_map.end(),
                    "found AccumulateGrad* is used by more than one tensors.");
        grad_fn_to_tensor_map.insert({grad_fn.get(), &t});
      }
    }
  }

  const auto& gradient_func_sptr = target.grad_fn();
  for (auto& edge : gradient_func_sptr->next_edges()) {
    torch::autograd::Node* node_func = edge.function.get();
    // If we find the next gradient function is AccumulateGrad, we will check whether its owned
    // tensors is in ctx.save_tensors or not. If yes, we skip it; otherwise, we clean the edge, which
    // will release the AccumulateGrad function.
    if (dynamic_cast<torch::autograd::AccumulateGrad*>(node_func)) {
      if (grad_fn_to_tensor_map.find(node_func) != grad_fn_to_tensor_map.end()) {
        // skip the edges that connect to saved_tensors. Because when unpack ctx.saved_tensors using
        // following code in backward:
        //     input, = ctx.saved_tensors
        // there is such a check: if the saved tensor is a leaf and requires grad, it should have grad accumulator.
        // If we clean the edge, then an exception "RuntimeError: No grad accumulator for a saved leaf!" will be thrown
        continue;
      } else {
        edge.function.reset();
      }
    }
  }
}

void unregister_grad_fn(py::object ctx) {
  // py::gil_scoped_release release;
  uint32_t y = reinterpret_cast<uintptr_t>(ctx.ptr());
  size_t ctx_address = static_cast<size_t>(y);
  PyNodeSharedPointerPool::GetInstance().UnRegisterGradFunc(ctx_address);
}

void clear_all_grad_fns() {
  // py::gil_scoped_release release;
  PyNodeSharedPointerPool::GetInstance().ClearAll();
}

bool get_materialize_grads(at::Tensor target) {
  // py::gil_scoped_release release;
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  const auto& grad_fn = autograd_meta->grad_fn_;
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(grad_fn.get());
  TORCH_CHECK(py_node_fn != nullptr, "grad_fn is not PyNode type.");
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  return py_fn->materialize_grads;
}

std::vector<bool> are_tensors_marked_as_dirty(at::Tensor& target, std::vector<at::Tensor>& tensors_to_check) {
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  const auto& grad_fn = autograd_meta->grad_fn_;
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(grad_fn.get());
  TORCH_CHECK(py_node_fn != nullptr, "grad_fn is not PyNode type.");
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  std::vector<bool> are_tensors_marked_dirty(tensors_to_check.size(), false);
  if (!py_fn->dirty_tensors)
    return are_tensors_marked_dirty;

  Py_ssize_t num_dirty = PyTuple_GET_SIZE(py_fn->dirty_tensors);
  for (const auto j : c10::irange(tensors_to_check.size())) {
    bool is_tensor_marked_dirty = false;
    for (const auto i : c10::irange(num_dirty)) {
      PyObject* obj = PyTuple_GET_ITEM(py_fn->dirty_tensors, i);
      const auto& tensor = THPVariable_Unpack(obj);
      if (tensor.is_same(tensors_to_check[j])) {
        is_tensor_marked_dirty = true;
        break;
      }
    }

    are_tensors_marked_dirty[j] = is_tensor_marked_dirty;
  }

  return are_tensors_marked_dirty;
}

static std::unordered_map<std::string, CustomFuncOpKernelInfo>& GetKernelInfoMap() {
  static std::unordered_map<std::string, CustomFuncOpKernelInfo> _kernel_info_map;
  return _kernel_info_map;
}

std::optional<at::Tensor> _try_to_get_tensor_owning_context(const py::tuple& forward_output_tensors) {
  py::object ctx = py::none();
  std::optional<at::Tensor> first_tensor_output;

  for (size_t i = 0; i < forward_output_tensors.size(); ++i) {
    PyObject* obj = forward_output_tensors[i].ptr();
    at::Tensor t;
    {
      // pybind11::gil_scoped_acquire gil;
      if (!THPVariable_Check(obj)) {
        continue;
      }

      t = THPVariable_Unpack(obj);
      if (!t.grad_fn()) {
        continue;
      }
    }

    // Be noted, in Python, we need additional check as below.
    // For the following case, it is possible grad_fn exists, but its value is None,
    // so we need to continue to search for the first tensor having a non-None grad_fn.
    //
    //  >>> w = torch.randn(5, 6)
    //  >>> hasattr(w, "grad_fn")
    //  True
    //  >>> w.grad_fn is None
    //  True
    //  >>> w, ... = CustomFunc.apply(w) # where CustomFunc forward just return w and other tensors.
    //
    //  Then hasattr(w, "grad_fn") is True, but w.grad_fn is None.

    first_tensor_output = t;
    break;
  }

  return first_tensor_output;
}

py::object _finalize_training_mode_forward(
    const std::unordered_map<int, at::Tensor>& input_tensors_used_for_fw_run,
    const py::tuple& forward_output_tensors,
    CustomFuncOpKernelInfo& kernel_info) {
  std::optional<at::Tensor> tensor_owning_ctx = _try_to_get_tensor_owning_context(forward_output_tensors);
  if (!tensor_owning_ctx.has_value()) {
    // ctx being None in training mode means the forward function is not differentiable, so backward is not needed.
    return py::none();
  }

  py::object ret = py::reinterpret_steal<py::object>(torch::autograd::functionToPyObject(tensor_owning_ctx.value().grad_fn()));

  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(tensor_owning_ctx.value());
  const auto& grad_fn = autograd_meta->grad_fn_;
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(grad_fn.get());
  TORCH_CHECK(py_node_fn != nullptr, "grad_fn is not PyNode type.");
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  TORCH_CHECK(py_fn != nullptr, "grad_fn is not THPFunction type.");

  std::vector<at::Tensor> saved_tensors;
  if (py_fn->saved_for_forward) {
    Py_ssize_t num_saved_for_forward = PyTuple_GET_SIZE(py_fn->saved_for_forward);
    saved_tensors.reserve(num_saved_for_forward);
    for (const auto i : c10::irange(num_saved_for_forward)) {
      PyObject* obj = PyTuple_GET_ITEM(py_fn->saved_for_forward, i);
      if (THPVariable_Check(obj)) {
        const auto& tensor = THPVariable_Unpack(obj);
        saved_tensors.push_back(tensor);
      }
    }
  }

  if (kernel_info.is_first_run) {
    kernel_info.materialize_grads = py_fn->materialize_grads;
    if (kernel_info.materialize_grads) {
      for (size_t i = 0; i < forward_output_tensors.size(); ++i) {
        PyObject* obj = forward_output_tensors[i].ptr();
        if (!THPVariable_Check(obj)) {
          continue;
        }
        at::Tensor t = THPVariable_Unpack(obj);
        kernel_info.materialize_grads_config.insert({i, {t.sizes().vec(), t.options()}});
      }
    }

    if (kernel_info.safe_run_enabled) {
      for (auto& pair : input_tensors_used_for_fw_run) {
        auto& tensor = pair.second;
        bool found = false;
        for (auto& t : saved_tensors) {
          if (t.is_same(tensor)) {
            found = true;
            break;
          }
        }
        kernel_info.tensor_input_indices_to_save_in_ctx[pair.first] = found;
      }

      // Check tensors generated by ORT are marked as dirty(for inplace update) or not .
      // If yes, save the input index of the tensor in the GetKernelInfoMap().
      std::vector<at::Tensor> tensors_to_check;
      tensors_to_check.reserve(input_tensors_used_for_fw_run.size());
      for (auto& pair : input_tensors_used_for_fw_run) {
        tensors_to_check.push_back(pair.second);
      }

      std::vector<bool> are_dirty = are_tensors_marked_as_dirty(tensor_owning_ctx.value(), tensors_to_check);
      size_t index = 0;
      for (auto& pair : input_tensors_used_for_fw_run) {
        kernel_info.tensor_input_indices_for_mark_dirty[pair.first] = are_dirty[index];
        index += 1;
      }
    }
  }

  // #FORWARD BACKWARD FUNCTION CONNECTIONS
  // #input_1(leaf, constructed by from_dlpack) < -- --reference-- --AccumulateGrad gradient function
  // #             ↓                                                                 ↑
  // #autograd.Function apply()-- -- -- -- -- --> autograd.Function backward()
  // #             ↓ |                            ↑
  // #output_1, output_2-- - shared_ptr < PyNode> -- -                            ↑
  // #             ↓ previous gradient function

  // #We remove the edges starting between current autograd.Function's gradient function and
  // #it 's input' s gradient function(e.g.AccumulateGrad gradient function), then
  // #AccumulateGrad gradient function will be destroyed, releasing the reference to input_1
  // #(https: //github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/functions/accumulate_grad.cpp#L21).
  // #The next edges are stored in Node, with which we can get next gradient function.
  // #https:  // github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/function.h#L527

  clear_grad_fns_for_next_edges(tensor_owning_ctx.value(), saved_tensors);

  // This is mainly to hold grad_fn references by registering it into our PyNodeSharedPointerPool.
  size_t y = reinterpret_cast<uintptr_t>(ret.ptr());
  register_grad_fn_and_remove_from_autograd(y, tensor_owning_ctx.value());

  return ret;
}

void detect_memory_reuse_once(
    CustomFuncOpKernelInfo& kernel_info,
    const std::unordered_map<size_t, int>& input_tensor_address_to_tensor_input_index_map,
    const std::vector<py::object>& all_outputs_of_kernel_run,
    const std::vector<int64_t>& all_outputs_to_tensor_inputs_reuse_map,
    const std::unordered_map<int, at::Tensor>& raw_input_tensors_used_inplace,
    const std::string& log_prefix) {
  // Procedure 1: Detect all outputs to tensor inputs reuse mapping, according to `all_outputs_of_kernel_run` and
  // `input_tensors_of_kernel_run`.

  TORCH_CHECK(all_outputs_to_tensor_inputs_reuse_map.size() == all_outputs_of_kernel_run.size(),
              log_prefix +
                  "all_outputs_to_tensor_inputs_reuse_map and kernel run outputs sizes not expected:" +
                  std::to_string(all_outputs_to_tensor_inputs_reuse_map.size()) + " vs " +
                  std::to_string(all_outputs_of_kernel_run.size()));

  // Detect all outputs to tensor inputs reuse mapping.
  std::vector<int> detected_reuse_map(all_outputs_of_kernel_run.size() + 1, -1);
  for (size_t output_index = 0; output_index < all_outputs_of_kernel_run.size(); ++output_index) {
    py::object arg = all_outputs_of_kernel_run[output_index];
    if (!THPVariable_Check(arg.ptr())) {
      continue;
    }
    at::Tensor t = THPVariable_Unpack(arg.ptr());
    size_t t_data_address = static_cast<size_t>(reinterpret_cast<uintptr_t>(t.data_ptr()));
    if (input_tensor_address_to_tensor_input_index_map.find(t_data_address) != input_tensor_address_to_tensor_input_index_map.end()) {
      int tensor_input_index = input_tensor_address_to_tensor_input_index_map.at(t_data_address);
      TORCH_CHECK(tensor_input_index != -1, "Reused tensor input index should not be -1");
      detected_reuse_map[output_index] = tensor_input_index;
    }
  }

  // Procedure 2: Validate the detected inplace_map with the registered inplace_map in ORT.
  // collect the output indices that need to be cloned before returned in case 2.1.2.
  for (size_t output_index = 0; output_index < all_outputs_of_kernel_run.size(); ++output_index) {
    int detected_inplace_index = detected_reuse_map[output_index];
    int inplace_index = all_outputs_to_tensor_inputs_reuse_map[output_index];
    if (inplace_index == detected_inplace_index) {
      continue;
    }

    if (raw_input_tensors_used_inplace.count(inplace_index) &&
        !raw_input_tensors_used_inplace.at(inplace_index).defined()) {
      // Use specified inplace input index, but the input tensor is None, which means the input is not
      // a tensor, so we don't do further checks.
      continue;
    }

    // If users register inplace_map (alloc planner will do buffer reuse),
    // but detected inplace_map indicates it is NO inplace reusing, we raise an error.
    if (inplace_index != -1 && detected_inplace_index == -1) {
      throw std::runtime_error(
          log_prefix + "Fatal: ONNX Op attribute 'tensor_reuse_map' indicates " +
          std::to_string(output_index) + "-th output is reusing input " +
          std::to_string(inplace_index) + ", but detected inplace_map indicates it is NOT reusing any input. " +
          "Please update inplace_map explicitly to make it consistent " +
          "to avoid undefined behavior due to ORT's memory reuse plan. " +
          +"detected reused input index: " + std::to_string(detected_inplace_index));
    }

    if (inplace_index == -1 && detected_inplace_index != -1) {
      std::cout << log_prefix << "ONNX Op attribute "
                << "'tensor_reuse_map' doesn't indicate " << std::to_string(output_index)
                << "-th output is reusing any input, "
                << "but detected inplace_map indicates it is reusing input index "
                << std::to_string(detected_inplace_index)
                << ". A clone will be done before returning to ORT, to align with ORT's NO Buffer reuse plan. "
                << "Please update inplace_map explicitly to avoid such a copy." << std::endl;

      kernel_info.output_indices_for_clone.push_back(output_index);
      continue;
    }

    throw std::runtime_error(
        log_prefix + "Fatal: ONNX Op attribute 'tensor_reuse_map' indicates " +
        std::to_string(output_index) + "-th output is reusing input " + std::to_string(inplace_index) +
        " but detected inplace_map indicates it is reusing input index " +
        std::to_string(detected_inplace_index) +
        ". Please update inplace_map explicitly to avoid undefined behavior due to memory reuse.");
  }
}

void _process_inplace_outputs(
    const CustomFuncOpKernelInfo& kernel_info,
    const std::string& func_name,
    const std::unordered_map<int, at::Tensor>& input_tensors_used_for_fw_run,
    const std::vector<int64_t>& all_outputs_to_tensor_inputs_reuse_map,
    const std::unordered_map<int, at::Tensor>& raw_input_tensors_used_inplace,
    bool is_backward,
    const std::string& log_prefix,
    std::vector<py::object>& all_outputs_of_kernel_run) {
  // Procedure 3: Do copies for 2.1.2 cases.
  for (const size_t& output_index : kernel_info.output_indices_for_clone) {
    std::cout << "_process_inplace_outputs " << output_index << " 1111" << std::endl;
    at::Tensor t = THPVariable_Unpack(all_outputs_of_kernel_run[output_index].ptr());
    std::cout << "_process_inplace_outputs 2222 " << all_outputs_of_kernel_run.size() << std::endl;
    at::Tensor t_clone = t.clone();
    std::cout << "_process_inplace_outputs 3333" << std::endl;
    auto pp = py::reinterpret_steal<py::object>(THPVariable_Wrap(t_clone));
    std::cout << "_process_inplace_outputs 4444" << std::endl;
    all_outputs_of_kernel_run[output_index] = pp;
    std::cout << "_process_inplace_outputs 5555" << std::endl;
  }

  // Procedure 4: Do copies for 2.0.2 cases.
  if (!is_backward && kernel_info.safe_run_enabled) {
    for (auto& pair : raw_input_tensors_used_inplace) {
      auto raw_tensor_input_index = pair.first;
      auto raw_input_tensor = pair.second;
      // raw_input_tensor can be None for backward run, but backward won't go here.
      if (!raw_input_tensor.defined()) {
        continue;
      }

      if (!kernel_info.tensor_input_indices_to_save_in_ctx.at(raw_tensor_input_index) &&
          !kernel_info.tensor_input_indices_for_mark_dirty.at(raw_tensor_input_index)) {
        continue;
      }

      // We did not do the check with tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty
      // because even for those tensor indices not in
      // tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty, we still need to do the
      // copy for the first-time run.
      if (raw_input_tensor.data_ptr() == input_tensors_used_for_fw_run.at(raw_tensor_input_index).data_ptr()) {
        // If the raw input tensor is not copied, we don't need this handling.
        continue;
      }

      // for each tensor, we don't do the copy once.
      bool copied = false;
      std::vector<size_t> output_indices_reusing_current_raw_input;
      for (size_t output_index = 0; output_index < all_outputs_to_tensor_inputs_reuse_map.size(); ++output_index) {
        if (all_outputs_to_tensor_inputs_reuse_map[output_index] == raw_tensor_input_index) {
          output_indices_reusing_current_raw_input.push_back(output_index);
        }
      }

      auto output_tensor_address = THPVariable_Unpack(all_outputs_of_kernel_run[output_indices_reusing_current_raw_input[0]].ptr()).data_ptr();
      for (size_t& output_index : output_indices_reusing_current_raw_input) {
        auto t = THPVariable_Unpack(all_outputs_of_kernel_run[output_index].ptr());
        TORCH_CHECK(output_tensor_address == t.data_ptr(),
                    "Outputs reusing the same input tensor should have the same address.");

        if (!copied) {
          // Only need a copy once.
          // Inplace copy only happens for non-leaf variables, so we have to set requires_grad to False.
          raw_input_tensor.requires_grad_(false);
          raw_input_tensor.copy_(t);
          std::cout << "Copy output tensor " << output_index << " to raw input tensor " << raw_tensor_input_index << "."
                    << (!kernel_info.is_first_run
                            ? "Provide output to input reuse mapping to avoid the copy overhead."
                            : "")
                    << std::endl;
          copied = true;
        }

        all_outputs_of_kernel_run[output_index] = py::reinterpret_steal<py::object>(THPVariable_Wrap(raw_input_tensor));
      }
    }
  }
}

void DLPack_Capsule_Destructor(PyObject* data) {
  if (!PyCapsule_IsValid(data, "dltensor")) {
    // early out, see DLPack spec: if a consuming library sets the capsule
    // name to something else, they own it and we don't need to do anything
    return;
  }

  // Causes overheads for validity checks again, but this case is rare
  // since consuming libraries should rename the capsule according to spec.
  // Note that this cannot set a python error (we checked validity above),
  // so we don't need to handle python error state here.
  DLManagedTensor* dlMTensor =
      (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  // the dlMTensor has not been consumed, call deleter ourselves.
  // DLPack spec mentions that deleter may be NULL, but deleter from
  // `at::toDLPack` is never NULL, so no need for an additional check here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
}

std::vector<PyObject*> custom_function_forward_runner(const char* func_name_char,
                                                      void* callback,
                                                      const std::vector<int64_t>& requires_grad_flags,
                                                      const std::vector<int64_t>& tensor_type_flags,
                                                      const bool is_training_mode,
                                                      const std::vector<int64_t>& inplace_map,
                                                      const char* kernel_invoke_id_char,
                                                      const std::vector<PyObject*>& args) {
  try {
    pybind11::gil_scoped_acquire gil;

    std::string func_name(func_name_char);
    std::string kernel_invoke_id(kernel_invoke_id_char);
    auto it = GetKernelInfoMap().find(kernel_invoke_id);
    if (it == GetKernelInfoMap().end()) {
      bool safe_run = true;
      GetKernelInfoMap().emplace(kernel_invoke_id, CustomFuncOpKernelInfo(kernel_invoke_id, safe_run));
    }

    CustomFuncOpKernelInfo& kernel_info = GetKernelInfoMap().at(kernel_invoke_id);

    std::unordered_map<int, at::Tensor> raw_input_tensors_used_inplace;
    std::unordered_map<int, at::Tensor> input_tensors_used_for_fw_run;

    int tensor_input_index = 0;
    std::vector<py::object> raii_call_args;
    raii_call_args.reserve(args.size());
    for (size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
      bool is_tensor = tensor_type_flags[arg_index] == 1;
      if (!is_tensor) {
        raii_call_args.push_back(py::reinterpret_borrow<py::object>(args[arg_index]));
        continue;
      }

      // Assume it's a DLPack tensor and convert it to PyTorch tensor.
      TORCH_CHECK(PyCapsule_IsValid(args[arg_index], "dltensor") != 0, "found invalid pycapsule");
      at::Tensor tensor = torch::utils::tensor_fromDLPack(args[arg_index]);
      bool requires_grad = requires_grad_flags[arg_index] && is_training_mode;
      tensor.requires_grad_(requires_grad);

      if (kernel_info.safe_run_enabled) {
        bool is_input_used_inplace = (std::find(inplace_map.begin(), inplace_map.end(), tensor_input_index) !=
                                      inplace_map.end());
        if (is_input_used_inplace) {
          raw_input_tensors_used_inplace[tensor_input_index] = tensor;
        }

        if (kernel_info.is_first_run) {
          at::Tensor tensor_clone;
          if (is_training_mode) {
            at::AutoGradMode enable_grad(true);
            tensor_clone = tensor.clone();
            tensor_clone.requires_grad_(requires_grad);
          } else {
            tensor_clone = tensor;
          }

          raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor_clone)));
          input_tensors_used_for_fw_run[tensor_input_index] = tensor_clone;
        } else {
          // Saving tensor for backward only affect the training.
          bool is_input_index_saved_in_ctx =
              is_training_mode && kernel_info.tensor_input_indices_to_save_in_ctx.at(tensor_input_index);

          bool is_input_index_marked_dirty =
              kernel_info.tensor_input_indices_for_mark_dirty.at(tensor_input_index);

          if (is_input_index_saved_in_ctx || is_input_index_marked_dirty) {
            at::AutoGradMode enable_grad(is_input_index_marked_dirty);
            auto wrapped_arg = tensor.clone();
            wrapped_arg.requires_grad_(requires_grad);
            raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(wrapped_arg)));
            input_tensors_used_for_fw_run[tensor_input_index] = wrapped_arg;
          } else {
            raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
            input_tensors_used_for_fw_run[tensor_input_index] = tensor;
          }
        }
      } else {
        raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
      }

      tensor_input_index++;
    }

    if (kernel_info.safe_run_enabled && kernel_info.is_first_run) {
      for (const auto i : c10::irange(input_tensors_used_for_fw_run.size())) {
        kernel_info.tensor_input_indices_to_save_in_ctx.insert({{i, false}});
        kernel_info.tensor_input_indices_for_mark_dirty.insert({{i, false}});
      }
    }

    py::tuple call_args = py::cast(raii_call_args);
    PyObject* result_pyobj;
    {
      at::AutoGradMode enable_grad(is_training_mode);
      result_pyobj = PyObject_CallObject(reinterpret_cast<PyObject*>(callback), call_args.ptr());
    }

    if (!result_pyobj) {
      throw std::runtime_error("Get null result");
    }

    py::object ret = py::reinterpret_steal<py::object>(result_pyobj);

    if (PyErr_Occurred()) {
      PyErr_Print();
      throw std::runtime_error("Python function execution fails with the above information.");
    }

    py::tuple forward_outputs;
    if (THPVariable_Check(ret.ptr())) {  // Don't check be tensor?
      forward_outputs = py::make_tuple(ret);
    } else {
      TORCH_CHECK(PyTuple_Check(ret.ptr()), "Python function must return a tuple.");
      forward_outputs = ret.cast<py::tuple>();
    }

    py::object ctx;
    if (is_training_mode) {
      ctx = _finalize_training_mode_forward(input_tensors_used_for_fw_run, forward_outputs, kernel_info);
      if (!ctx.is_none()) {
        PyObject_SetAttrString(ctx.ptr(), "fw_kernel_invoke_id", py::cast(kernel_invoke_id).ptr());
      }
    } else {
      ctx = py::none();
    }

    std::vector<py::object> all_outputs_of_kernel_run;
    all_outputs_of_kernel_run.reserve(forward_outputs.size() + 1);
    all_outputs_of_kernel_run.push_back(ctx);
    for (size_t i = 0; i < forward_outputs.size(); ++i) {
      all_outputs_of_kernel_run.push_back(forward_outputs[i]);
    }

    if (kernel_info.safe_run_enabled) {
      bool is_backward = false;
      std::string log_prefix = func_name + " -> " + (is_backward ? "Backward " : "Forward ");
      if (kernel_info.is_first_run) {
        // key: tensor data address;
        // value: if the tensor is defined it records the tensor input index, otherwise, -1.
        std::unordered_map<size_t, int> input_tensor_address_to_tensor_input_index_map;
        input_tensor_address_to_tensor_input_index_map.reserve(input_tensors_used_for_fw_run.size());
        for (auto& input : input_tensors_used_for_fw_run) {
          if (input.second.defined()) {
            input_tensor_address_to_tensor_input_index_map.insert(
                {{static_cast<size_t>(reinterpret_cast<uintptr_t>(input.second.data_ptr())), input.first}});
          }
        }

        detect_memory_reuse_once(kernel_info,
                                 input_tensor_address_to_tensor_input_index_map,
                                 all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/,
                                 inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                                 raw_input_tensors_used_inplace,
                                 log_prefix);
      }

      _process_inplace_outputs(kernel_info,
                               func_name,
                               input_tensors_used_for_fw_run,
                               inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                               raw_input_tensors_used_inplace,
                               false /*is_backward*/,
                               log_prefix,
                               all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/);
    }

    std::vector<PyObject*> rets;
    rets.reserve(all_outputs_of_kernel_run.size());
    for (auto& py_obj : all_outputs_of_kernel_run) {
      PyObject* obj = py_obj.ptr();

      if (!THPVariable_Check(obj)) {
        Py_INCREF(obj);
        rets.push_back(obj);
        continue;
      }

      DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(obj));
      rets.push_back(PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor));
    }

    if (kernel_info.is_first_run) {
      kernel_info.is_first_run = false;
    }

    // std::cout << "custom_function_forward_runner>>> completed" << std::endl;
    return rets;
  } catch (const std::exception& e) {  // NOLINT
    std::cerr << e.what() << std::endl;
    throw;
  }
}

std::vector<PyObject*> custom_function_backward_runner(const char* func_name_char,
                                                       void* callback,
                                                       const std::vector<int64_t>& requires_grad_flags,
                                                       const std::vector<int64_t>& tensor_type_flags,
                                                       const bool is_training_mode,
                                                       const std::vector<int64_t>& inplace_map,
                                                       const char* kernel_invoke_id_char,
                                                       const std::vector<PyObject*>& args) {
  // py::gil_scoped_release release;
  std::cout << "custom_function_backward_runner<<<enter" << std::endl;
  pybind11::gil_scoped_acquire gil;
  // GilGuard gil;
  try {
    std::string func_name(func_name_char);
    std::string kernel_invoke_id(kernel_invoke_id_char);
    at::AutoGradMode enable_grad(false);

    // auto t0 = std::chrono::high_resolution_clock::now();
    auto it = GetKernelInfoMap().find(kernel_invoke_id);
    if (it == GetKernelInfoMap().end()) {
      bool safe_run = true;
      GetKernelInfoMap().emplace(kernel_invoke_id, CustomFuncOpKernelInfo(kernel_invoke_id, safe_run));
    }

    CustomFuncOpKernelInfo& kernel_info = GetKernelInfoMap().at(kernel_invoke_id);

    std::unordered_map<int, at::Tensor> raw_input_tensors_used_inplace;
    std::unordered_map<int, at::Tensor> input_tensors_used_for_bw_run;

    int tensor_input_index = 0;
    std::vector<py::object> raii_call_args;
    raii_call_args.reserve(args.size());
    py::object ctx = py::reinterpret_borrow<py::object>(args[0]);
    // pybind11::gil_scoped_acquire gil;
    raii_call_args.push_back(ctx);
    for (size_t arg_index = 1; arg_index < args.size(); ++arg_index) {
      if (tensor_type_flags[arg_index] != 1) {
        raii_call_args.push_back(py::reinterpret_borrow<py::object>(args[arg_index]));
        continue;
      }

      at::Tensor tensor;
      // Assume it's a DLPack tensor and convert it to PyTorch tensor.
      bool is_dlpack = PyCapsule_IsValid(args[arg_index], "dltensor") != 0;
      if (is_dlpack) {
        tensor = torch::utils::tensor_fromDLPack(args[arg_index]);
      } else {
        TORCH_CHECK(args[arg_index] == Py_None, "Only None is supported for non-tensor input.");
        PyObject* fw_kernel_invoke_id = PyObject_GetAttrString(ctx.ptr(), "fw_kernel_invoke_id");
        std::string fw_kernel_invoke_id_str = py::cast<std::string>(py::reinterpret_borrow<py::object>(fw_kernel_invoke_id));
        CustomFuncOpKernelInfo& fw_kernel_info = GetKernelInfoMap().at(fw_kernel_invoke_id_str);
        if (fw_kernel_info.materialize_grads) {
          auto& config = fw_kernel_info.materialize_grads_config.at(arg_index - 1);
          tensor = at::zeros(std::get<0>(config), std::get<1>(config));  // shift by 1 to skip context input.
        }
      }

      if (kernel_info.safe_run_enabled) {
        bool is_input_used_inplace = std::find(inplace_map.begin(), inplace_map.end(), arg_index) !=
                                     inplace_map.end();
        if (is_input_used_inplace) {
          raw_input_tensors_used_inplace[tensor_input_index] = tensor;
        }
        input_tensors_used_for_bw_run[tensor_input_index] = tensor;
      }

      if (tensor.defined()) {
        raii_call_args.push_back(py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
      } else {
        raii_call_args.push_back(py::none());
      }

      tensor_input_index++;
    }

    std::cout << "custom_function_backward_runner<<<before call" << std::endl;
    py::tuple call_args = py::cast(raii_call_args);
    py::object ret;
    {
      at::AutoGradMode enable_grad(false);
      ret = py::reinterpret_steal<py::object>(PyObject_CallObject(reinterpret_cast<PyObject*>(callback), call_args.ptr()));
    }
    if (PyErr_Occurred()) {
      PyErr_Print();
      throw std::runtime_error("Python function execution fails with the above information.");
    }
    std::cout << "custom_function_backward_runner<<<after call" << std::endl;

    std::vector<py::object> all_outputs_of_kernel_run;
    if (THPVariable_Check(ret.ptr())) {
      all_outputs_of_kernel_run.push_back(ret);
    } else {
      TORCH_CHECK(PyTuple_Check(ret.ptr()), "Python function must return a tuple.");
      all_outputs_of_kernel_run = ret.cast<std::vector<py::object>>();
    }

    if (kernel_info.safe_run_enabled) {
      std::cout << "custom_function_backward_runner<<<safe_run_enabled" << std::endl;
      bool is_backward = true;
      std::string log_prefix = func_name + " -> " + (is_backward ? "Backward " : "Forward ");
      if (kernel_info.is_first_run) {
        std::cout << "custom_function_backward_runner<<<detect_memory_reuse_once start" << std::endl;

        // key: tensor data address;
        // value: if the tensor is defined it records the tensor input index, otherwise, -1.
        std::unordered_map<size_t, int> input_tensor_address_to_tensor_input_index_map;
        input_tensor_address_to_tensor_input_index_map.reserve(input_tensors_used_for_bw_run.size());
        for (auto& input : input_tensors_used_for_bw_run) {
          input_tensor_address_to_tensor_input_index_map.insert(
              {{static_cast<size_t>(reinterpret_cast<uintptr_t>(input.second.data_ptr())),
                input.second.defined() ? (input.first + 1) : -1}}); /* skip the ctx input*/
        }

        detect_memory_reuse_once(kernel_info,
                                 input_tensor_address_to_tensor_input_index_map,
                                 all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/,
                                 inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                                 raw_input_tensors_used_inplace,
                                 log_prefix);

        std::cout << "custom_function_backward_runner<<<detect_memory_reuse_once done" << std::endl;
      }

      _process_inplace_outputs(kernel_info,
                               func_name,
                               input_tensors_used_for_bw_run,
                               inplace_map /*all_outputs_to_tensor_inputs_reuse_map*/,
                               raw_input_tensors_used_inplace,
                               is_backward /*is_backward*/,
                               log_prefix,
                               all_outputs_of_kernel_run /*all_outputs_of_kernel_run*/);
      std::cout << "custom_function_backward_runner<<<_process_inplace_outputs done" << std::endl;
    }

    std::vector<PyObject*> rets;
    for (auto& py_obj : all_outputs_of_kernel_run) {
      PyObject* obj = py_obj.ptr();

      if (!THPVariable_Check(obj)) {
        Py_INCREF(obj);
        rets.push_back(obj);
        continue;
      }

      DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(obj));
      rets.push_back(PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor));
    }

    if (kernel_info.is_first_run) {
      kernel_info.is_first_run = false;
    }
    return rets;
  } catch (const std::exception& e) {  // NOLINT
    std::cerr << e.what() << std::endl;
    throw;
  }
}
