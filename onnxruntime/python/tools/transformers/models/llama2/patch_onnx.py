import argparse
import ctypes
import os

import numpy as np
from ml_dtypes import finfo, float8_e4m3fnuz
import onnx
import onnx.helper as helper

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input of the model path")
parser.add_argument(
    "--output",
    "-o",
    required=True,
    type=str,
    help="output model path, e.g., output.onnx",
)

args = parser.parse_args()

model = onnx.load(args.input)
onnx.load_external_data_for_model(model, base_dir=os.path.dirname(args.input))


def compute_scaling_factor(a: np.ndarray, fp8_max: float, margin: int) -> float:
    amax = np.abs(a).max()
    scale = (fp8_max - margin) / amax  # fallback scale
    exp = np.floor(np.log2(fp8_max / amax)) - margin
    sf = np.round(np.power(2, np.abs(exp)))
    sf = np.where(amax > 0.0, sf, scale)
    sf = np.where(np.isfinite(amax), sf, scale)
    sf = np.where(exp < 0, 1 / sf, sf)

    return sf.item()


def cast_and_scale(a, dtype: str):
    if dtype == "float16":
        return a.astype(dtype), 1.0
    elif dtype == "float8_e4m3fnuz":
        sf = compute_scaling_factor(a, fp8_max=finfo(float8_e4m3fnuz).max, margin=4)
        return (a * sf).astype(float8_e4m3fnuz), sf
    else:
        raise ValueError(dtype)


class ReplaceMatMulAsFp8MatMul:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.node_index: dict[str, int] = {node.name: idx for idx, node in enumerate(model.graph.node)}
        self.initializer_index: dict[str, int] = {init.name: idx for idx, init in enumerate(model.graph.initializer)}
        self.initializer_scale: dict[str, float] = {}
        self.initializer_consumers: dict[str, list[str]] = {}
        for node in model.graph.node:
            for i in node.input:
                if i in self.initializer_index:  # i is an initializer
                    if i not in self.initializer_consumers:
                        self.initializer_consumers[i] = [node.name]
                    else:
                        self.initializer_consumers[i].append(node.name)
        self.patched_nodes = set()

    def _is_input_an_initializer_and_fp16(self, input_name):
        if input_name not in self.initializer_index:
            return False
        init = model.graph.initializer[self.initializer_index[input_name]]
        return init.data_type == onnx.TensorProto.FLOAT16

    def _should_patch_matmul(self, node: onnx.NodeProto):
        a, b = node.input
        return self._is_input_an_initializer_and_fp16(b)

    def _try_patch_matmul(self, node: onnx.NodeProto):
        assert len(node.input) == 2
        if self._should_patch_matmul(node):
            print(f"---- patching node MatMul[{node.name}]")
            a, b = node.input
            scale = self._try_overwrite_initializer(b)
            fp8_matmul = onnx.helper.make_node(
                "Fp8MatMul", inputs=[a, b], outputs=list(node.output), domain="com.microsoft", scale_b=1.0 / scale
            )
            self.model.graph.node.pop(self.node_index[node.name])
            self.model.graph.node.insert(self.node_index[node.name], fp8_matmul)

    def _try_overwrite_initializer(self, initializer_name: str):
        if initializer_name in self.initializer_scale:
            print(f"  -- skip overwriting initializer[{initializer_name}]")
            return self.initializer_scale[initializer_name]

        print(f"  -- overwriting initializer[{initializer_name}]")
        idx = self.initializer_index[initializer_name]
        weight_fp16 = np.frombuffer(self.model.graph.initializer[idx].raw_data, dtype=np.float16)
        scale = compute_scaling_factor(weight_fp16, fp8_max=finfo(float8_e4m3fnuz).max, margin=4)
        weight_fp8 = (weight_fp16 * scale).astype(float8_e4m3fnuz)
        self.model.graph.initializer[idx].raw_data = bytes(
            (ctypes.c_byte * weight_fp8.size).from_address(weight_fp8.__array_interface__["data"][0])
        )
        self.model.graph.initializer[idx].data_type = onnx.TensorProto.FLOAT8E4M3FNUZ
        self.initializer_scale[initializer_name] = scale
        return scale

    def patch(self):
        for n in self.model.graph.node:
            if n.op_type == "MatMul":
                self._try_patch_matmul(n)
        return self.model


model = ReplaceMatMulAsFp8MatMul(model).patch()
onnx.save(model, args.output, save_as_external_data=True, location=os.path.basename(args.output) + ".data")
