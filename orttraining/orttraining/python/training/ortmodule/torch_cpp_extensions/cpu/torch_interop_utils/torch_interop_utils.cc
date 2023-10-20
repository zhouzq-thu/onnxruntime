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

// In PyTorch forward run (e.g. THPFunction_apply), ctx of type THPFunction* (which is also a PyObject*)
// is created (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/python_function.cpp#L673).
// The ctx is used to run user-defined forward function and backward function as the first
// parameter. The same time, a cdata of type std::shared_ptr<PyNode> is created
// (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/python_function.cpp#L677),
// cdata is owned by:
//    a). forward run output tensors as grad_fn_ property. (The full hierarchy is: Tensor owns
//        shared_pointer<TensorImpl>; TensorImpl owns std::unique_ptr<AutogradMeta>; AutogradMeta
//        manages grad_/grad_fn_/grad_accumulator_. Among them, grad_fn_ is std::shared_ptr<PyNode>,
//        e.g, the so called gradient function.)
//        https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/variable.h#L194
//    b). the consumer operator of forward run outputs, will let its own PyNode/Node (gradient function)
//        owns the grad_fn_ (of type std::shared_ptr<PyNode>) of all inputs that require grad.
//        https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/function.h#L263
// BUT, if we run torch computation within PythonOp, b) is lost. So for some cases, where forward outputs
// are not used and freed before backward function runs, the grad_fn_ (std::shared_ptr<PyNode>) references
// in a) will be released. Without b)'s reference, grad_fn_ release PyNode as reference count reach 0;
// Then when PythonOpGrad runs, segment fault.
//
// So we add b)'s reference in this Pool when forward run returns; dereference from this Pool when backward
// completes, then ~PyNode() is called, which subsequently calls ~THPFunction() destroying ctx.
class PyNodeSharedPointerPool {
 public:
  static PyNodeSharedPointerPool& GetInstance() {
    static PyNodeSharedPointerPool pool;
    return pool;
  };

  void RegisterGradFuncAndRemoveFromAutoGrad(const size_t& ctx_address,
                                             torch::autograd::AutogradMeta* autograd_meta) {
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it == grad_fns_.end(), "should not register grad_fn twice for ctx ", ctx_address);

    // Add new entry if key hasn't been registered.
    // After this, the grad_fn_ is removed from torch autograd.
    grad_fns_.emplace(ctx_address, std::move(autograd_meta->grad_fn_));
    TORCH_CHECK(autograd_meta->grad_fn_ == nullptr, "fail to remove grad_fn_ from torch autograd for ctx ",
                ctx_address);
  };

  void UnRegisterGradFunc(const size_t& ctx_address) {
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it != grad_fns_.end(), "fail to find grad_fn for ctx ", ctx_address);

    grad_fns_.erase(ctx_address);
  };

  void ClearAll() {
    grad_fns_.clear();
  }

 private:
  PyNodeSharedPointerPool(){};
  ~PyNodeSharedPointerPool(){};

  PyNodeSharedPointerPool(const PyNodeSharedPointerPool&) = delete;
  PyNodeSharedPointerPool& operator=(const PyNodeSharedPointerPool&) = delete;
  PyNodeSharedPointerPool(PyNodeSharedPointerPool&&) = delete;
  PyNodeSharedPointerPool& operator=(PyNodeSharedPointerPool&&) = delete;

  std::unordered_map<size_t, std::shared_ptr<torch::autograd::Node>> grad_fns_;
};

void clear_grad_fns_for_next_edges(at::Tensor target,
                                   std::vector<at::Tensor>& saved_tensors) {
  py::gil_scoped_release release;

  // For leaf tensor, there will be a AccumulateGrad (gradient function) created, which owns a
  // reference to the tensor.
  // For any user saved tensors (with save_for_backward), if the tensor is leaf, we put the map
  // {AccumulateGrad*, Tensor*} into grad_fn_to_tensor_map.
  std::unordered_map<torch::autograd::Node*, at::Tensor*> grad_fn_to_tensor_map;
  if (saved_tensors.size() > 0) {
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

void register_grad_fn_and_remove_from_autograd(size_t ctx_address, at::Tensor target) {
  // std::cout << "register_grad_fn_and_remove_from_autograd ctx_address: " << ctx_address << std::endl;
  py::gil_scoped_release release;
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  PyNodeSharedPointerPool::GetInstance().RegisterGradFuncAndRemoveFromAutoGrad(ctx_address, autograd_meta);
}

void unregister_grad_fn(py::object ctx) {
  py::gil_scoped_release release;
  uint32_t y = reinterpret_cast<uintptr_t>(ctx.ptr());
  size_t ctx_address = static_cast<size_t>(y);
  PyNodeSharedPointerPool::GetInstance().UnRegisterGradFunc(ctx_address);
}

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
void clear_all_grad_fns() {
  py::gil_scoped_release release;
  PyNodeSharedPointerPool::GetInstance().ClearAll();
}

bool get_materialize_grads(at::Tensor target) {
  py::gil_scoped_release release;
  torch::autograd::AutogradMeta* autograd_meta = torch::autograd::impl::get_autograd_meta(target);
  const auto& grad_fn = autograd_meta->grad_fn_;
  auto py_node_fn = dynamic_cast<torch::autograd::PyNode*>(grad_fn.get());
  TORCH_CHECK(py_node_fn != nullptr, "grad_fn is not PyNode type.");
  THPFunction* py_fn = (THPFunction*)py_node_fn->obj;
  return py_fn->materialize_grads;
}

std::vector<bool> are_tensors_marked_as_dirty(at::Tensor target, std::vector<at::Tensor> tensors_to_check) {
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

class CustomFuncOpKernelInfo {
 public:
  CustomFuncOpKernelInfo(const std::string& invoke_id, bool safe_run) {
    // # kernel_invoke_id is a string contains session thread id, op kernel creation time stamp in ms, a random int,
    // # and address of op_kernel pointer. This can guarantee the uniqueness of the key in case of multiple
    // # instances of a same named PythonOp/PythonOpGrad in one session, or multiple sessions.
    kernel_invoke_id = invoke_id;

    safe_run_enabled = safe_run;

    // position_to_tensor_index_map: Optional[Tuple[Tuple[int, ...], ...]] = None

    // # For the tensors generated from ORT backend, there is special handling here:
    // # 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
    // # all such tensors will be cloned in case they are saved in context (but ORT backend is not aware of the
    // # reference, may release the content of the tensor before it is needed in backward). Once
    // # `autograd.Function.apply` completes, by checking the existence of the tensor in the saved_tensors,
    // # `_GlobalOpKernelInfoMap` is updated to save the input indices that are saved in context.
    // # 2. For the subsequent runs, if the input index is in `tensor_input_indices_to_save_in_ctx`, the tensor
    // # will be cloned before fed into `autograd.Function.apply` as input.
    // self.tensor_input_indices_to_save_in_ctx: Optional[Tuple[int, ...]] = None

    // # To align with PyTorch `ctx.set_materialize_grads(False|True)``
    // # materialize_grads_config is a map from output index to (device, dtype, shape) of the output tensor, used
    // # for materializing the gradient of the output tensor in backward.
    // self.materialize_grads: bool = False
    // self.materialize_grads_config: Optional[Dict[int, Tuple[torch.device, torch.dtype, torch.shape]]] = None

    // # For the tensors generated from ORT backend, there is special handling here:
    // # 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
    // # all such tensors will be cloned (with gradient) in case they are marked as dirty (if not cloned, but marked
    // # as dirty, PyTorch will complain the tensor is a leaf, should not be used for inplace update). Once
    // # `autograd.Function.apply` completes, by checking the existence of the tensor in the dirty_tensors,
    // # `_GlobalOpKernelInfoMap` is updated to save the input indices that are marked as dirty.
    // # 2. For the subsequent runs, if the input index is in `tensor_input_indices_for_mark_dirty`, the tensor
    // # will be cloned (with gradient) before fed into `autograd.Function.apply` as input.
    // self.tensor_input_indices_for_mark_dirty: Optional[Tuple[int, ...]] = None

    // # A list of output indices that needs to be clone before returned, due to inplace update analysis.
    // self.output_indices_for_clone: Optional[List[int]] = None

    // self.tensor_input_states = OrderedDict()  # key: tensor input index, value: TensorInputState.
  }

  // std::tuple<int, int> check_with_input_index(int tensor_input_index){
  //     if tensor_input_index not in self.tensor_input_states:
  //         is_input_index_saved_in_ctx = tensor_input_index in self.tensor_input_indices_to_save_in_ctx
  //         is_input_index_marked_dirty = tensor_input_index in self.tensor_input_indices_for_mark_dirty
  //         self.tensor_input_states[tensor_input_index] = [is_input_index_saved_in_ctx, is_input_index_marked_dirty]
  //     return self.tensor_input_states[tensor_input_index]
  // }

  std::string kernel_invoke_id;
  std::unordered_map<int, int> input_global_index_to_tensor_index_map;
  std::optional<std::unordered_map<int, int>> tensor_input_indices_to_save_in_ctx;
  bool materialize_grads;
  // std::unordered_map<int, std::tuple<c10::Device, c10::ScalarType, torch::Shape>> materialize_grads_config;

  std::optional<std::unordered_map<int, int>> tensor_input_indices_for_mark_dirty;
  std::vector<int> output_indices_for_clone;
  bool is_first_run{true};
  bool safe_run_enabled{false};
};

std::unordered_map<std::string, CustomFuncOpKernelInfo> _GlobalOpKernelInfoMap;

py::list forward_runner(
    // std::function forward_function,
    const std::vector<bool>& requires_grad_flags,
    const std::vector<int>& tensor_type_flags,
    bool is_training_mode,
    const std::vector<int>& inplace_map,
    const std::string& kernel_invoke_id,
    const std::string& func_name,
    py::tuple args) {
  py::gil_scoped_release release;

  // auto t0 = std::chrono::high_resolution_clock::now();
  auto it = _GlobalOpKernelInfoMap.find(kernel_invoke_id);
  if (it == _GlobalOpKernelInfoMap.end()) {
    bool safe_run = false;
    _GlobalOpKernelInfoMap.emplace(kernel_invoke_id, CustomFuncOpKernelInfo(kernel_invoke_id, safe_run));
  }

  CustomFuncOpKernelInfo& kernel_info = _GlobalOpKernelInfoMap.at(kernel_invoke_id);
  // std::unordered_map<int, at::Tensor> raw_input_tensors_used_inplace;
  // std::unordered_map<int, at::Tensor> input_tensors_used_for_fw_run;
  int tensor_input_index = 0;
  std::vector<std::variant<py::object, at::Tensor>> wrapped_args;
  wrapped_args.reserve(args.size());
  {
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> fs = t1 - t0;
    // std::cout << "ckpt 1 latency(ms): " << fs.count() * 1000 << ", kernel_info.is_first_run: " << kernel_info.is_first_run << std::endl;
  }
  for (size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
    bool is_tensor = tensor_type_flags[arg_index] == 1;
    bool requires_grad = requires_grad_flags[arg_index] && is_training_mode;
    if (!is_tensor) {
      wrapped_args.push_back(args[arg_index]);
      continue;
    }

    at::Tensor tensor;
    {
      // auto t0 = std::chrono::high_resolution_clock::now();
      pybind11::gil_scoped_acquire gil;
      // Assume it's a DLPack tensor and convert it to PyTorch tensor.
      TORCH_CHECK(PyCapsule_IsValid(args[arg_index].ptr(), "dltensor") != 0, "found invalid pycapsule");
      tensor = torch::utils::tensor_fromDLPack(args[arg_index].ptr());
      // auto t1 = std::chrono::high_resolution_clock::now();
      // std::chrono::duration<float> fs = t1 - t0;
      // std::cout << "dlpack latency(ms): " << fs.count() * 1000 << ", kernel_info.is_first_run: " << kernel_info.is_first_run << std::endl;
    }

    // bool is_input_used_inplace = std::find(inplace_map.begin(), inplace_map.end(), tensor_input_index) != inplace_map.end();
    // if (is_input_used_inplace) {
    //   // raw_input_tensors_used_inplace[tensor_input_index] = tensor;
    // }

    tensor.requires_grad_(requires_grad);

    if (is_training_mode && kernel_info.safe_run_enabled) {
      if (kernel_info.is_first_run) {
        at::AutoGradMode enable_grad(true);
        auto wrapped_arg = tensor.clone();
        wrapped_args.push_back(wrapped_arg);
      } else {
        bool is_input_index_saved_in_ctx = kernel_info.tensor_input_indices_to_save_in_ctx.value().find(tensor_input_index) !=
                                           kernel_info.tensor_input_indices_to_save_in_ctx.value().end();
        // std::find(kernel_info.tensor_input_indices_to_save_in_ctx.value().begin(),
        //                                              kernel_info.tensor_input_indices_to_save_in_ctx.value().end(),
        //                                              tensor_input_index) !=
        //                                    kernel_info.tensor_input_indices_to_save_in_ctx.value().end();

        bool is_input_index_marked_dirty = kernel_info.tensor_input_indices_for_mark_dirty.value().find(tensor_input_index) !=
                                           kernel_info.tensor_input_indices_for_mark_dirty.value().end();
        // std::find(kernel_info.tensor_input_indices_for_mark_dirty.value().begin(),
        //                                              kernel_info.tensor_input_indices_for_mark_dirty.value().end(),
        //                                              tensor_input_index) !=
        //                                    kernel_info.tensor_input_indices_for_mark_dirty.value().end();

        if (is_input_index_saved_in_ctx || is_input_index_marked_dirty) {
          at::AutoGradMode enable_grad(is_input_index_marked_dirty);
          auto wrapped_arg = tensor.clone();
          // with torch.set_grad_enabled(is_input_index_marked_dirty):
          //     wrapped_arg = wrapped_arg.clone()

          // input_tensors_used_for_fw_run[tensor_input_index] = wrapped_arg
          // wrapped_args[input_position] = wrapped_arg
          wrapped_arg.requires_grad_(requires_grad);
          wrapped_args.push_back(wrapped_arg);
        } else {
          wrapped_args.push_back(tensor);
        }
      }
    } else {
      wrapped_args.push_back(tensor);
    }

    // input_tensors_used_for_fw_run[tensor_input_index] = wrapped_args.back();
    tensor_input_index++;
    {
      // auto t1 = std::chrono::high_resolution_clock::now();
      // std::chrono::duration<float> fs = t1 - t0;
      // // std::chrono::milliseconds d = std::chrono::duration_cast<ms>(fs);
      // std::cout << "ckpt 2 latency(ms): " << fs.count() * 1000 << ", kernel_info.is_first_run: " << kernel_info.is_first_run << std::endl;
    }
  }
  {
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> fs = t1 - t0;
    // // std::chrono::milliseconds d = std::chrono::duration_cast<ms>(fs);
    // std::cout << "runner e2e latency(ms): " << fs.count() * 1000 << ", kernel_info.is_first_run: " << kernel_info.is_first_run << std::endl;
  }

  if (kernel_info.is_first_run) {
    kernel_info.is_first_run = false;
  }
  return py::cast(wrapped_args);
}

std::optional<at::Tensor> _get_context(py::list forward_output_tensors) {
  py::object ctx = py::none();
  std::optional<at::Tensor> first_tensor_output;

  for (size_t i = 0; i < forward_output_tensors.size(); ++i) {
    PyObject* obj = forward_output_tensors[i].ptr();
    at::Tensor t;
    {
      pybind11::gil_scoped_acquire gil;
      if (!THPVariable_Check(obj)) {
        continue;
      }

      t = THPVariable_Unpack(obj);
      if (!t.grad_fn()) {
        continue;
      }
    }

    // // # For the following case, it is possible grad_fn exists, but its value is None,
    // // # so we need to continue to search for the first tensor having a non-None grad_fn.
    // // #
    // // # >>> w = torch.randn(5, 6)
    // // # >>> hasattr(w, "grad_fn")
    // // # True
    // // # >>> w.grad_fn is None
    // // # True
    // // # >>> w, ... = CustomFunc.apply(w) # where CustomFunc forward just return w and other tensors.
    // // #
    // // # Then hasattr(w, "grad_fn") is True, but w.grad_fn is None.
    // continue

    first_tensor_output = t;
    break;
  }

  // if (first_tensor_output.has_value()) {
  //   // # Use the first context we see because all of arg's share the same one.
  //   ctx = first_tensor_output.value().grad_fn();
  // }
  return first_tensor_output;
}

py::object _finalize_training_mode_forward(
    std::string kernel_invoke_id,
    std::string func_name,
    py::list forward_output_tensors) {
  std::optional<at::Tensor> tensor_owning_ctx;
  tensor_owning_ctx = _get_context(forward_output_tensors);
  CustomFuncOpKernelInfo& kernel_info = _GlobalOpKernelInfoMap.at(kernel_invoke_id);

  py::object ret;
  {
    pybind11::gil_scoped_acquire gil;
    if (tensor_owning_ctx.has_value()) {
      ret = py::reinterpret_steal<py::object>(torch::autograd::functionToPyObject(tensor_owning_ctx.value().grad_fn()));
    } else {
      ret = py::none();
    }
  }

  // #ctx being None in training mode means the forward function is not differentiable, so backward is not needed.
  if (!tensor_owning_ctx.has_value()) {
    // #If this is the first time run, collect kernel - specific information.
    if (!kernel_info.tensor_input_indices_to_save_in_ctx.has_value()) {
      kernel_info.tensor_input_indices_to_save_in_ctx = std::unordered_map<int, int>{};
    }
    if (!kernel_info.tensor_input_indices_for_mark_dirty.has_value()) {
      kernel_info.tensor_input_indices_for_mark_dirty = std::unordered_map<int, int>{};
    }

    return ret;
  }

  //  std::unordered_map<int, at::Tensor> input_tensors_used_for_fw_run,

  // #Filter out the None in the saved_tensors.
  // saved_tensors = [t for t in ctx.saved_tensors if t is not None];

  // ctx.fw_kernel_invoke_id = kernel_invoke_id

  // #If this is the first time run, collect kernel - specific information.
  //    if (!kernel_info.tensor_input_indices_to_save_in_ctx.has_value()) {
  //     auto saved_tensors = [t for t in ctx.saved_tensors if t is not None];
  //         if len(saved_tensors):
  // // #Check tensors generated by ORT are in the saved_tensors or not .
  // // #If yes, save the input index of the tensor in the _GlobalOpKernelInfoMap.
  //             kernel_info.tensor_input_indices_to_save_in_ctx = tuple([
  //                 tensor_input_index
  //                 for tensor_input_index, tensor in input_tensors_used_for_fw_run.items()
  //                 if any(tensor is saved_tensor for saved_tensor in saved_tensors)
  //             ])
  //             _log_warning(
  //                 f"{func_name}: Add input index to _GlobalOpKernelInfoMap, to avoid extra copy in every iteration."
  //             )
  //         else:
  //             kernel_info.tensor_input_indices_to_save_in_ctx = ()
  //         kernel_info.materialize_grads = torch_interop_utils.get_materialize_grads(tensor_owning_ctx)
  //         kernel_info.materialize_grads_config = OrderedDict()
  //         if kernel_info.materialize_grads:
  //             for output_index, tensor in enumerate(forward_output_tensors):
  //                 if isinstance(tensor, torch.Tensor):
  //                     kernel_info.materialize_grads_config[output_index] = (
  //                         tensor.device,
  //                         tensor.dtype,
  //                         tensor.shape,
  //                     )
  //    }

  //     if kernel_info.tensor_input_indices_for_mark_dirty is None:
  // // #Check tensors generated by ORT are marked as dirty(for inplace update) or not .
  // // #If yes, save the input index of the tensor in the _GlobalOpKernelInfoMap.
  //         are_tensors_marked_as_dirty = torch_interop_utils.are_tensors_marked_as_dirty(
  //             tensor_owning_ctx, [t for t in input_tensors_used_for_fw_run.values()]
  //         )
  //         kernel_info.tensor_input_indices_for_mark_dirty = tuple([
  //             tensor_input_index
  //             for is_dirty, (tensor_input_index, tensor) in zip(
  //                 are_tensors_marked_as_dirty, input_tensors_used_for_fw_run.items()
  //             )
  //             if is_dirty is True
  //         ])
  //         _log_warning(f"{func_name}: Add input index to _GlobalOpKernelInfoMap, to support leaf node do inplace update.")

  //     torch_nvtx_range_push(f"{func_name}.clear_grad")
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

  std::vector<at::Tensor> saved_tensors;  // todo(pengwa)
  clear_grad_fns_for_next_edges(tensor_owning_ctx.value(), saved_tensors);

  // #This is mainly to hold grad_fn references by registering it into our PyNodeSharedPointerPool.
  // torch_nvtx_range_push(f "{func_name}.rg_grad_fn")

  // const std::shared_ptr<torch::autograd::Node>& cdata_ptr = tensor_owning_ctx.value().grad_fn();
  uint32_t y = reinterpret_cast<uintptr_t>(ret.ptr());
  register_grad_fn_and_remove_from_autograd(static_cast<size_t>(y), tensor_owning_ctx.value());

  return ret;
}

void _process_inplace_outputs(
    std::string kernel_invoke_id,
    std::string func_name,
    py::list forward_output_tensors,
    std::vector<at::Tensor> input_tensors_used_for_fw_run,
    py::list all_outputs_of_kernel_run,
    std::vector<int> all_outputs_to_tensor_inputs_reuse_map,
    std::unordered_map<int, std::optional<at::Tensor>> raw_input_tensors_used_inplace,
    bool is_backward) {
  // kernel_info: CustomFuncOpKernelInfo,
  // func_name: str,
  // input_tensors_of_kernel_run: Dict[int, Union[torch.Tensor, None]],
  // all_outputs_of_kernel_run: List[Union[torch.Tensor, any]],
  // all_outputs_to_tensor_inputs_reuse_map: List[int],
  // raw_input_tensors_used_inplace: Dict[int, Union[torch.Tensor, None]],
  // is_backward=False,

  // log_prefix = f"{func_name}->{'Backward' if is_backward else 'Forward'}: "
  // input_tensor_address_list = [
  //     t.data_ptr() if isinstance(t, torch.Tensor) else -1 for t in input_tensors_of_kernel_run.values()
  // ]
  // if is_backward:
  //     input_tensor_address_list = [-1, *input_tensor_address_list]  # skip the context input

  // is_first_time_init = kernel_info.output_indices_for_clone is None
  // # If this is the first time run, collect runtime tensor reuse mapping.
  // if is_first_time_init:
  //     # Procedure 1: Detect all outputs to tensor inputs reuse mapping, according to `all_outputs_of_kernel_run` and
  //     # `input_tensors_of_kernel_run`.
  //     assert len(all_outputs_to_tensor_inputs_reuse_map) == len(all_outputs_of_kernel_run), (
  //         f"{log_prefix}all_outputs_to_tensor_inputs_reuse_map and kernel run outputs should have the same length."
  //         f"all_outputs_to_tensor_inputs_reuse_map: {all_outputs_to_tensor_inputs_reuse_map}, "
  //         f"kernel run outputs: {all_outputs_of_kernel_run}"
  //     )

  //     # Detect all outputs to tensor inputs reuse mapping.
  //     detected_reuse_map = [-1] * (len(all_outputs_of_kernel_run))
  //     for output_index, arg in enumerate(all_outputs_of_kernel_run):
  //         if not isinstance(arg, torch.Tensor):
  //             continue
  //         if arg.data_ptr() in input_tensor_address_list:
  //             input_index = input_tensor_address_list.index(arg.data_ptr())
  //             detected_reuse_map[output_index] = input_index

  //     # Procedure 2: Validate the detected inplace_map with the registered inplace_map in ORT.
  //     output_indices_for_clone = (
  //         []
  //     )  # collect the output indices that need to be cloned before returned in case 2.1.2.
  //     for output_index, (detected_inplace_index, inplace_index) in enumerate(
  //         zip(detected_reuse_map, all_outputs_to_tensor_inputs_reuse_map)
  //     ):
  //         if inplace_index == detected_inplace_index:
  //             continue

  //         if (
  //             inplace_index in raw_input_tensors_used_inplace
  //             and raw_input_tensors_used_inplace[inplace_index] is None
  //         ):
  //             # Use specified inplace input index, but the input tensor is None, which means the input is not
  //             # a tensor, so we don't do further checks.
  //             continue

  //         # If users register inplace_map (alloc planner will do buffer reuse),
  //         # but detected inplace_map indicates it is NO inplace reusing, we raise an error.
  //         if inplace_index != -1 and detected_inplace_index == -1:
  //             raise RuntimeError(
  //                 f"{log_prefix}Fatal: "
  //                 f"ONNX Op attribute 'tensor_reuse_map' indicates {output_index}-th output is reusing input "
  //                 f"{inplace_index}, but detected inplace_map indicates it is NOT reusing any input. "
  //                 "Please update inplace_map explicitly to make it consistent "
  //                 f"to avoid undefined behavior due to ORT's memory reuse plan. "
  //                 f"inplace_map: {all_outputs_to_tensor_inputs_reuse_map}, "
  //                 f"detected inplace_map: {detected_reuse_map}"
  //             )

  //         if inplace_index == -1 and detected_inplace_index != -1:
  //             output_indices_for_clone.append(output_index)
  //             continue

  //         raise RuntimeError(
  //             f"{log_prefix}Fatal: "
  //             f"ONNX Op attribute 'inplace_map' indicates {inplace_index}-th output is reusing "
  //             f"input index {detected_inplace_index}, but detected inplace_map indicates it is reusing "
  //             f"input index {inplace_index}. Please update inplace_map explicitly to avoid undefined behavior "
  //             f"due to memory reuse. inplace_map: {all_outputs_to_tensor_inputs_reuse_map}, "
  //             f"detected inplace_map: {detected_reuse_map}"
  //         )

  //     kernel_info.output_indices_for_clone = output_indices_for_clone

  // assert kernel_info.output_indices_for_clone is not None

  // # Procedure 3: Do copies for 2.1.2 cases.
  // for output_index in kernel_info.output_indices_for_clone:
  //     _log_warning(
  //         f"{log_prefix}ONNX Op attribute "
  //         f"'tensor_reuse_map' doesn't indicate {output_index}-th output is reusing any input, "
  //         f"but detected inplace_map indicates it is reusing some input index. "
  //         "A clone will be done before returning to ORT, to align with ORT's NO Buffer reuse plan. "
  //         "Please update inplace_map explicitly to avoid such a copy."
  //     )
  //     all_outputs_of_kernel_run[output_index] = all_outputs_of_kernel_run[output_index].detach().clone()

  // # Procedure 4: Do copies for 2.0.2 cases.
  // if is_backward is False and (
  //     is_first_time_init
  //     or kernel_info.tensor_input_indices_to_save_in_ctx
  //     or kernel_info.tensor_input_indices_for_mark_dirty
  // ):
  //     for raw_tensor_input_index, raw_input_tensor in raw_input_tensors_used_inplace.items():
  //         # raw_input_tensor can be None for backward run, but backward won't go here.
  //         if not isinstance(raw_input_tensor, torch.Tensor):
  //             continue

  //         # We did not do the check with tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty
  //         # because even for those tensor indices not in
  //         # tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty, we still need to do the
  //         # copy for the first-time run.
  //         if raw_input_tensor.data_ptr() == input_tensor_address_list[raw_tensor_input_index]:
  //             # If the raw input tensor is not copied, we don't need this handling.
  //             continue

  //         copied = False  # for each tensor, we don't do the copy once.
  //         output_indices_reusing_current_raw_input = [
  //             output_index
  //             for output_index, input_index in enumerate(all_outputs_to_tensor_inputs_reuse_map)
  //             if input_index == raw_tensor_input_index
  //         ]
  //         output_tensor_address = all_outputs_of_kernel_run[output_indices_reusing_current_raw_input[0]].data_ptr()
  //         for output_index in output_indices_reusing_current_raw_input:
  //             assert (
  //                 output_tensor_address == all_outputs_of_kernel_run[output_index].data_ptr()
  //             ), "Outputs reusing the same input tensor should have the same address."

  //             if not copied:
  //                 # Only need a copy once.
  //                 # Inplace copy only happens for non-leaf variables, so we have to set requires_grad to False.
  //                 raw_input_tensor.requires_grad = False
  //                 raw_input_tensor.copy_(all_outputs_of_kernel_run[output_index])
  //                 _log_warning(
  //                     f"{log_prefix}Copy output tensor {output_index} to raw input tensor {raw_tensor_input_index}. "
  //                     f"{'Provide output to input reuse mapping to avoid the copy overhead.' if not is_first_time_init else ''}"
  //                 )
  //                 copied = True

  //             all_outputs_of_kernel_run[output_index] = raw_input_tensor
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

py::tuple complete_forward_runner(
    bool is_training_mode,
    std::string kernel_invoke_id,
    std::string func_name,
    py::list forward_output_tensors) {
  py::object ctx;
  if (is_training_mode) {
    ctx = _finalize_training_mode_forward(kernel_invoke_id, func_name, forward_output_tensors);
    // if ctx is not None:
    //     ctx.fw_kernel_invoke_id = kernel_invoke_id
  }

  std::vector<py::object> rets;
  rets.push_back(ctx);
  pybind11::gil_scoped_acquire gil;
  for (auto& py_obj : forward_output_tensors) {
    PyObject* obj = py_obj.ptr();
    at::Tensor t;

    if (!THPVariable_Check(obj)) {
      rets.push_back(py::reinterpret_borrow<py::object>(py_obj));
      continue;
    }

    DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(obj));
    rets.push_back(py::reinterpret_steal<py::object>(PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor)));
  }

  // # _process_inplace_outputs(
  // #     kernel_info,
  // #     func_name,
  // #     input_tensors_used_for_fw_run,
  // #     final_rets,
  // #     inplace_map,
  // #     raw_input_tensors_used_inplace,
  // # )

  // dlpacks = [final_rets[0]]
  // torch_nvtx_range_push(f"{func_name}.post")
  // def _wrap_dlpack(value):
  //     return to_dlpack(value) if value is not None else None

  // dlpacks.extend(list(map(_wrap_dlpack, final_rets[1:])))
  // torch_nvtx_range_pop()

  return py::cast(rets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_grad_fn_and_remove_from_autograd", &register_grad_fn_and_remove_from_autograd,
        "Increase grad_fn shared pointer reference.");
  m.def("unregister_grad_fn", &unregister_grad_fn, "Release grad_fn shared pointer reference.");
  m.def("clear_all_grad_fns", &clear_all_grad_fns, "Clear all grad_fn shared pointer references.");
  m.def("clear_grad_fns_for_next_edges", &clear_grad_fns_for_next_edges,
        "Remove reference on next edges' gradient functions.");
  m.def("get_materialize_grads", &get_materialize_grads, "Return whether materialize_grads is enabled or not.");
  m.def("are_tensors_marked_as_dirty", &are_tensors_marked_as_dirty, "Return whether the tensors are marked dirty or not.");
  m.def("forward_runner", &forward_runner, "Forward runner.");
  m.def("_finalize_training_mode_forward", &_finalize_training_mode_forward, "Finalize training mode forward.");
  m.def("complete_forward_runner", &complete_forward_runner, "Complete forward runner.");
}
