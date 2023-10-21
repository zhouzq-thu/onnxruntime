# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import sys
import warnings
from collections import OrderedDict
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils
from onnxruntime.training.utils import nvtx_function_decorator, torch_nvtx_range_pop, torch_nvtx_range_push

from ._fallback import ORTModuleFallbackException, ORTModuleIOError, _FallbackManager, wrap_exception  # noqa: F401
from ._utils import get_rank


def _log_warning(message: str):
    """Configure the logger for PythonOp runner according to following rules.
    1. If multiple processes are used, the rank will be appended
       to the logger name.
    2. The logger will be disabled for non-zero ranks.
    """
    if get_rank() == 0:
        warnings.warn(f"[rank-{get_rank()}] {message}")


@nvtx_function_decorator
def call_python_forward_function(
    forward_function: Callable,
    requires_grad_flags: List[bool],
    tensor_type_flags: List[int],
    is_training_mode: bool,
    inplace_map: List[int],
    kernel_invoke_id: str,
    func_name: Union[bytes, str],
    *args,
):
    """
    This function bridges the gap between ORT variables and autograd.Function.apply.
    It conducts basic casting from ORT to PyTorch (before calling "forward_function") and from PyTorch to ORT
    (after calling "forward_function"). It also enable autograd in PyTorch. It formats returned outputs,
    for example, dropping None's from forward_function's output list.

    The major difference between call_python_forward_function and call_python_backward_function is that
    in the forward one, we have extra code to process autograd context from PyTorch.

    Args:
        forward_function: pointer to autograd.Function.apply (e.g., MyReLU.apply).
        requires_grad_flags: requires_grad_flags[i] indicates if the i-th arg needs gradient.
        tensor_type_flags: tensor_type_flags[i] indicates the type of the i-th arg, 0 - non-tensor, 1 - tensor.
        is_training_mode: indicates if this model is running under training mode.
        inplace_map: a list of the same length of kernel outputs, each element represents which input index
          it is reusing. If there is no reuse, the value is -1.
        args: inputs to "backward_function".
    """

    try:
        func_name = func_name.decode("utf-8") if isinstance(func_name, bytes) else func_name
        kernel_invoke_id = kernel_invoke_id.decode("utf-8") if isinstance(kernel_invoke_id, bytes) else kernel_invoke_id
        wrapped_args = torch_interop_utils.forward_runner(requires_grad_flags, tensor_type_flags,
                                                          is_training_mode, inplace_map,
                                                          kernel_invoke_id, func_name, args)

        with torch.set_grad_enabled(is_training_mode):
            # Run autograd.Function.apply(...).
            # TODO(pengwa): looks like we are assuming all outputs will be either Tensor or None.
            # We should revisit if it is possible to support other types of output, for example int, or, etc.
            # But that might also require some work in backend.
            torch_nvtx_range_push(f"{func_name}.fw")
            result = forward_function(*wrapped_args)
            torch_nvtx_range_pop()

            results = []
            if isinstance(result, torch.Tensor):
                results = [result]
            elif isinstance(result, (tuple, list)):
                results = [r for r in result]
            else:
                raise wrap_exception(
                    ORTModuleIOError,
                    TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                )

            torch_nvtx_range_push(f"{func_name}.post")
            rets = torch_interop_utils.complete_forward_runner(is_training_mode, kernel_invoke_id, func_name, tuple(results))
            torch_nvtx_range_pop()
            return tuple(rets)
    except Exception as e:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        print("Exception happens when running ", forward_function)
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904


@nvtx_function_decorator
def call_python_backward_function(
    backward_function: Callable,
    requires_grad_flags: List[bool],
    tensor_type_flags: List[int],
    is_training_mode: bool,
    inplace_map: List[int],
    kernel_invoke_id: str,
    func_name: Union[bytes, str],
    *args,
):
    """
    This function bridges the gap between ORT variables and autograd.Function.backward.
    It conducts basic casting from ORT to PyTorch (before calling "backward_function")
    and from PyTorch to ORT (after calling "backward_function").  It formats returned
    outputs, example, dropping None's from backward_function's output list.

    Args:
        backward_function: pointer to autograd.Function.backward (e.g., MyReLU.backward).
        requires_grad_flags: requires_grad_flags[i] indicates if the i-th arg needs gradient.
        tensor_type_flags: tensor_type_flags[i] indicates the type of the i-th arg.
        is_training_mode: indicates if this model is running under training mode.
        inplace_map: a list of the same length of kernel outputs, each element represents which input index
          it is reusing. If there is no reuse, the value is -1.
        args: inputs to "backward_function".
    """

    try:

        func_name = func_name.decode("utf-8") if isinstance(func_name, bytes) else func_name
        kernel_invoke_id = kernel_invoke_id.decode("utf-8") if isinstance(kernel_invoke_id, bytes) else kernel_invoke_id

        wrapped_args = torch_interop_utils.backward_runner(requires_grad_flags, tensor_type_flags,
                                                           is_training_mode, inplace_map,
                                                           kernel_invoke_id, func_name, args)
        ctx = args[0]

        with torch.no_grad():
            # Call Python function.
            torch_nvtx_range_push(f"{func_name}.bw")
            result = backward_function(*wrapped_args)
            torch_nvtx_range_pop()

            # Extract results as DLPack tensor list.
            if isinstance(result, torch.Tensor):
                result = [result]
            elif isinstance(result, (tuple, list)):
                result = list(result)
            else:
                raise wrap_exception(
                    ORTModuleIOError,
                    TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                )

            # _process_inplace_outputs(
            #     kernel_info,
            #     func_name,
            #     input_tensors_used_for_bw_run,
            #     result,
            #     inplace_map,
            #     raw_input_tensors_used_inplace,
            #     is_backward=True,
            # )
            def wrap_all_outputs(result):
                if isinstance(result, torch.Tensor):
                    return [to_dlpack(result)]
                elif isinstance(result, (tuple, list)):
                    return [to_dlpack(value) if value is not None else None for value in result]
                else:
                    raise wrap_exception(
                        ORTModuleIOError,
                        TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                    )
            wrapped_returned_args = wrap_all_outputs(result)
            torch_interop_utils.unregister_grad_fn(ctx)
            return tuple(wrapped_returned_args)
    except Exception as e:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        print("Exception happens when running ", backward_function)
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904
