# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import ctypes
from dataclasses import dataclass
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
from ml_dtypes import float8_e4m3fnuz, finfo
from utils import dtype_to_suffix, get_gemm_basic_sizes, get_gemm_bert_sizes, get_gemm_bound, matmul, transab_to_suffix


def create_device_array(a):
    # ptr = ctypes.c_void_p(a.__array_interface__["data"][0])
    ptr = a.__array_interface__["data"][0]
    size = a.size
    itemsize = finfo(a.dtype).bits // 8
    return ke.DeviceArray(ptr, size, itemsize)


def compute_scaling_factor(a: np.ndarray, fp8_max: float, margin: int) -> np.ndarray:
    amax = np.abs(a).max()
    scale = (fp8_max - margin) / amax  # fallback scale
    exp = np.floor(np.log2(fp8_max / amax)) - margin
    sf = np.round(np.power(2, np.abs(exp)))
    sf = np.where(amax > 0.0, sf, scale)
    sf = np.where(np.isfinite(amax), sf, scale)
    sf = np.where(exp < 0, 1 / sf, sf)

    return sf


def cast_and_scale(a, dtype: str):
    if dtype == "float16":
        return a.astype(dtype), 1.0
    elif dtype == "float8_e4m3fnuz":
        sf = compute_scaling_factor(a, fp8_max=finfo(float8_e4m3fnuz).max, margin=4)
        return (a * sf).astype(float8_e4m3fnuz), sf
    else:
        raise ValueError(dtype)


def _test_gemm(func, dta: str, dtb: str, dtc: str, transa: bool, transb: bool, m: int, n: int, k: int):
    assert dta in ["float16", "float8_e4m3fnuz"]
    assert dtb in ["float16", "float8_e4m3fnuz"]
    assert dtc in ["float16"]

    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)

    a, scale_a = cast_and_scale(np.random.rand(*a_shape) + 0.5, dta)
    b, scale_b = cast_and_scale(np.random.rand(*b_shape) + 0.5, dtb)

    print(scale_a, scale_b)

    ref_c = matmul((a / scale_a).astype("float64"), (b / scale_b).astype("float64"), transa, transb)
    scale_c = float("nan")

    bound = 0.01  # FIXME: how to derive the bound for fp8?

    my_c = np.ones((m, n), dtype=dtc)
    dev_a = create_device_array(a)
    dev_b = create_device_array(b)
    dev_c = create_device_array(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    my_gemm = func(opa, opb, m, n, k, dev_a, lda, 1 / scale_a, dev_b, ldb, 1 / scale_b, dev_c, n, 1 / scale_c)

    failures = {}
    print(f"{dta} {dtb} {dtc} {transab_to_suffix((transa, transb))} m={m:<5} n={n:<5} k={k:<5} bound: {bound}")

    for impl in my_gemm.ListOps():
        if not my_gemm.SelectOp(impl):
            continue
        # Restore C Array
        my_c.fill(1.0)
        dev_c.UpdateDeviceArray()
        my_gemm.Run()
        dev_c.UpdateHostNumpyArray()

        try:
            np.testing.assert_allclose(my_c, ref_c, rtol=bound)
        except Exception as err:
            header = "*" * 30 + impl + "*" * 30
            print(header)
            print(err)
            print("*" * len(header))
            failures[impl] = str(err)

    if failures:
        raise Exception(failures)


dtypes = [("float8_e4m3fnuz", "float16", "float16"), ("float16", "float8_e4m3fnuz", "float16")]
all_transabs = [(False, False)]


@pytest.mark.skipif(not ke.is_composable_kernel_available(), reason="ck is not enabled")
@pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
# @pytest.mark.parametrize("m, n, k", [(1, 8192, 28672)])
# @pytest.mark.parametrize("m, n, k", [(64, 256, 16)])
@pytest.mark.parametrize("transa, transb", all_transabs)
@pytest.mark.parametrize("dta, dtb, dtc", dtypes)
def test_ck_gemm_bert_cases(dta, dtb, dtc, transa, transb, m, n, k):
    wrapper_name = f"CKFP8Gemm_{dtype_to_suffix(dta)}_{dtype_to_suffix(dtb)}_{dtype_to_suffix(dtc)}_{transab_to_suffix((transa, transb))}"
    _test_gemm(getattr(ke, wrapper_name), dta, dtb, dtc, transa, transb, m, n, k)


# Tunable is basically wrapped around of rocblas and ck gemm, so no need for full tests
# @pytest.mark.parametrize("m, n, k", get_gemm_basic_sizes(full=False) + get_gemm_bert_sizes(full=False))
# @pytest.mark.parametrize("transa, transb", all_transabs)
# @pytest.mark.parametrize("dtype", dtypes)
# def test_gemm_tunable_bert_cases(dtype, transa, transb, m, n, k):
#     wrapper_name = f"GemmTunable_{dtype_to_suffix(dtype)}_{transab_to_suffix((transa, transb))}"
#     _test_gemm(getattr(ke, wrapper_name), dtype, transa, transb, m, n, k)


@dataclass
class GemmMetric(ke.ComputeMetric):
    transa: bool
    transb: bool
    m: int
    n: int
    k: int

    def report(self):
        common = (
            f"{self.dtype} {transab_to_suffix((self.transa, self.transb))} "
            f"m={self.m:<4} n={self.n:<4} k={self.k:<4} {self.name}"
        )
        if self.duration <= 0:
            return "not supported          " + common

        return f"{self.duration:>6.2f} us {self.tflops:>5.2f} tflops " + common


def profile_gemm_func(func, dta: str, dtb: str, dtc: str, transa: bool, transb: bool, m: int, n: int, k: int):
    a_shape = (k, m) if transa else (m, k)
    b_shape = (n, k) if transb else (k, n)

    np.random.seed(0)
    a, scale_a = cast_and_scale(np.random.rand(*a_shape) + 0.5, dta)
    b, scale_b = cast_and_scale(np.random.rand(*b_shape) + 0.5, dtb)
    my_c = np.ones((m, n), dtype=dtc)
    scale_c = 1.0

    dev_a = create_device_array(a)
    dev_b = create_device_array(b)
    dev_c = create_device_array(my_c)

    opa = ke.blas_op.T if transa else ke.blas_op.N
    opb = ke.blas_op.T if transb else ke.blas_op.N
    lda = a_shape[1]
    ldb = b_shape[1]
    my_gemm = func(opa, opb, m, n, k, dev_a, lda, scale_a, dev_b, ldb, scale_b, dev_c, n, scale_c)

    for impl in my_gemm.ListOps():
        duration_ms = -1
        if my_gemm.SelectOp(impl):
            duration_ms = my_gemm.Profile()
        FLOPs = m * k * n * 2  # noqa: N806

        ke.report(GemmMetric(impl, f"{dta}_{dtb}_{dtc}", duration_ms, FLOPs, transa, transb, m, n, k))


def profile_with_args(dta, dtb, dtc, transa, transb, m, n, k, sort):
    dtype_suffix = "_" + dtype_to_suffix(dta) + "_" + dtype_to_suffix(dtb) + "_" + dtype_to_suffix(dtc)
    transab_suffix = "_" + transab_to_suffix((transa, transb))
    with ke.benchmark(sort):
        profile_gemm_func(
            getattr(ke, "CKFP8Gemm" + dtype_suffix + transab_suffix), dta, dtb, dtc, transa, transb, m, n, k
        )
        # profile_gemm_func(getattr(ke, "GemmTunable" + dtype_suffix + transab_suffix), dtype, transa, transb, m, n, k)
    print()


def profile():
    for dta, dtb, dtc in dtypes:
        for m, n, k in get_gemm_bert_sizes(full=True):
            profile_with_args(dta, dtb, dtc, False, False, m, n, k, True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("dta", choices=["float8_e4m3fnuz", "float16"])
    group.add_argument("dtb", choices=["float8_e4m3fnuz", "float16"])
    group.add_argument("dtc", choices=["float8_e4m3fnuz", "float16"])
    group.add_argument("transa", choices="NT")
    group.add_argument("transb", choices="NT")
    group.add_argument("m", type=int)
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(
            args.dta, args.dtb, args.dtc, args.transa == "T", args.transb == "T", args.m, args.n, args.k, args.sort
        )
