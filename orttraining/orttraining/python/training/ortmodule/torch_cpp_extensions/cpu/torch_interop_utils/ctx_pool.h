

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
  }

  void RegisterGradFuncAndRemoveFromAutoGrad(const size_t& ctx_address,
                                             torch::autograd::AutogradMeta* autograd_meta) {
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it == grad_fns_.end(), "should not register grad_fn twice for ctx ", ctx_address);

    // Add new entry if key hasn't been registered.
    // After this, the grad_fn_ is removed from torch autograd.
    grad_fns_.emplace(ctx_address, std::move(autograd_meta->grad_fn_));
    TORCH_CHECK(autograd_meta->grad_fn_ == nullptr, "fail to remove grad_fn_ from torch autograd for ctx ",
                ctx_address);
  }

  void UnRegisterGradFunc(const size_t& ctx_address) {
    auto it = grad_fns_.find(ctx_address);
    TORCH_CHECK(it != grad_fns_.end(), "fail to find grad_fn for ctx ", ctx_address);

    grad_fns_.erase(ctx_address);
  }

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
