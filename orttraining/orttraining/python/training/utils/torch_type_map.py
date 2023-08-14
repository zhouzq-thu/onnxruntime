# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import torch


# Mapping from pytorch scalar type to onnx scalar type.
_CAST_PYTORCH_TO_ONNX = {
    "Byte": [torch.onnx.TensorProtoDataType.UINT8, torch.uint8],
    "Char": [torch.onnx.TensorProtoDataType.INT8, torch.int8],
    "Double": [torch.onnx.TensorProtoDataType.DOUBLE, torch.double],
    "Float": [torch.onnx.TensorProtoDataType.FLOAT, torch.float],
    "Half": [torch.onnx.TensorProtoDataType.FLOAT16, torch.half],
    "Int": [torch.onnx.TensorProtoDataType.INT32, torch.int],
    "Long": [torch.onnx.TensorProtoDataType.INT64, torch.int64],
    "Short": [torch.onnx.TensorProtoDataType.INT16, torch.short],
    "Bool": [torch.onnx.TensorProtoDataType.BOOL, torch.bool],
    "ComplexFloat": [torch.onnx.TensorProtoDataType.COMPLEX64, torch.complex64],
    "ComplexDouble": [torch.onnx.TensorProtoDataType.COMPLEX128, torch.complex128],
    "BFloat16": [torch.onnx.TensorProtoDataType.BFLOAT16, torch.bfloat16],
    # Not yet defined in torch.
    # "Float8E4M3FN": torch.onnx.TensorProtoDataType.FLOAT8E4M3FN,
    # "Float8E4M3FNUZ": torch.onnx.TensorProtoDataType.FLOAT8E4M3FNUZ,
    # "Float8E5M2": torch.onnx.TensorProtoDataType.FLOAT8E5M2,
    # "Float8E5M2FNUZ": torch.onnx.TensorProtoDataType.FLOAT8E5M2FNUZ,
    "Undefined": [torch.onnx.TensorProtoDataType.UNDEFINED, None],
}

# _SCALAR_TYPE_TO_DTYPE = {
#     JitScalarType.BOOL: torch.bool,
#     JitScalarType.UINT8: torch.uint8,
#     JitScalarType.INT8: torch.int8,
#     JitScalarType.INT16: torch.short,
#     JitScalarType.INT: torch.int,
#     JitScalarType.INT64: torch.int64,
#     JitScalarType.HALF: torch.half,
#     JitScalarType.FLOAT: torch.float,
#     JitScalarType.DOUBLE: torch.double,
#     JitScalarType.COMPLEX32: torch.complex32,
#     JitScalarType.COMPLEX64: torch.complex64,
#     JitScalarType.COMPLEX128: torch.complex128,
#     JitScalarType.QINT8: torch.qint8,
#     JitScalarType.QUINT8: torch.quint8,
#     JitScalarType.QINT32: torch.qint32,
#     JitScalarType.BFLOAT16: torch.bfloat16,
# }

# _SCALAR_TYPE_TO_NAME: Dict[JitScalarType, ScalarName] = {
#     JitScalarType.BOOL: "Bool",
#     JitScalarType.UINT8: "Byte",
#     JitScalarType.INT8: "Char",
#     JitScalarType.INT16: "Short",
#     JitScalarType.INT: "Int",
#     JitScalarType.INT64: "Long",
#     JitScalarType.HALF: "Half",
#     JitScalarType.FLOAT: "Float",
#     JitScalarType.DOUBLE: "Double",
#     JitScalarType.COMPLEX32: "ComplexHalf",
#     JitScalarType.COMPLEX64: "ComplexFloat",
#     JitScalarType.COMPLEX128: "ComplexDouble",
#     JitScalarType.QINT8: "QInt8",
#     JitScalarType.QUINT8: "QUInt8",
#     JitScalarType.QINT32: "QInt32",
#     JitScalarType.BFLOAT16: "BFloat16",
#     JitScalarType.UNDEFINED: "Undefined",
# }

_DTYPE_TO_ONNX = {torch_dtype: onnx_dtype for k, (onnx_dtype, torch_dtype) in _CAST_PYTORCH_TO_ONNX.items()}

def pytorch_dtype_to_onnx(dtype: torch.dtype) -> torch.onnx.TensorProtoDataType:
    return _DTYPE_TO_ONNX[dtype]


def pytorch_scalar_type_str_to_onnx(scalar_type: str) -> torch.onnx.TensorProtoDataType:
    try:
        return torch.onnx.JitScalarType.from_name(scalar_type).onnx_type()
    except AttributeError:
        return _CAST_PYTORCH_TO_ONNX[scalar_type][0]
