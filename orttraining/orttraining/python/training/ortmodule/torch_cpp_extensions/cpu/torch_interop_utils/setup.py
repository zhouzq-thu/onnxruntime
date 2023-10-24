# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

from setuptools import Extension, setup  # noqa: F401
from torch.utils import cpp_extension

source_filenames = ["torch_interop_utils.cc", "custom_function_runner.cc"]
header_filenames = ["ctx_pool.h", "custom_function_runner.h"]

extra_compile_args = {"cxx": ["-O3"]}
setup(
    name="torch_interop_utils",
    ext_modules=[
        cpp_extension.CppExtension(
            name="torch_interop_utils",
            sources=[os.path.join(os.path.dirname(__file__), filename) for filename in source_filenames],
            extra_compile_args=extra_compile_args,
            include_dirs=[os.path.join(os.path.dirname(__file__), filename) for filename in header_filenames] + ["/usr/local/cuda-11.8/include/", ],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
