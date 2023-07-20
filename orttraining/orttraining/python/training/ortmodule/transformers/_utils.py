# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from onnx import GraphProto, NodeProto, TensorProto, helper, numpy_helper


# Get attribute value from node by attribute key.
def _get_attribute(node: NodeProto, attr_name: str, default_value: Any = None) -> Any:
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value


# Convert Constant node or TensorProto to Python value.
def _to_numpy_array(node: Any) -> np.ndarray:
    tensor = node
    if isinstance(node, NodeProto):
        tensor = _get_attribute(node, "value")
    assert isinstance(tensor, TensorProto)
    return numpy_helper.to_array(tensor).tolist()


class GraphMatcher:
    def __init__(self, graph: GraphProto):
        self._graph: GraphProto = graph
        self._op_type_to_nodes: Dict[str, List[NodeProto]] = {}
        for node in graph.node:
            if node.op_type not in self._op_type_to_nodes:
                self._op_type_to_nodes[node.op_type] = []
            self._op_type_to_nodes[node.op_type].append(node)

    def get_producer(self, arg: str, op_type: str):
        if op_type not in self._op_type_to_nodes:
            return []
        return [node for node in self._op_type_to_nodes[op_type] if arg in node.output]

    def get_consumer(self, arg: str, op_type: str):
        if op_type not in self._op_type_to_nodes:
            return []
        return [node for node in self._op_type_to_nodes[op_type] if arg in node.input]

    def get_constant_value(self, arg: str):
        node_or_initializer = None
        if "Constant" in self._op_type_to_nodes:
            for node in self._op_type_to_nodes["Constant"]:
                if arg in node.output:
                    node_or_initializer = node
                    break
        if node_or_initializer is None:
            for initializer in self._graph.initializer:
                if arg == initializer.name:
                    node_or_initializer = initializer
                    break
        if node_or_initializer is None:
            return None
        return _to_numpy_array(node_or_initializer)

    def get_shape(self, arg: str):
        value_infos = [
            value_info
            for value_info in itertools.chain(self._graph.input, self._graph.value_info)
            if value_info.name == arg
        ]
        if len(value_infos) > 0 and value_infos[0].type.tensor_type.HasField("shape"):
            shape = []
            for dim in value_infos[0].type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            return shape
        initializers = [initializer for initializer in self._graph.initializer if initializer.name == arg]
        if len(initializers) > 0:
            return initializers[0].dims
        return None

    def _match_pattern(self, node: NodeProto, pattern: List[Tuple[str, bool, List[Tuple[int, int, int]]]]):
        nodes = [node]
        for i in range(1, len(pattern)):
            next_op_type = pattern[i][0]
            is_producer = pattern[i][1]
            node_idx, output_idx, input_idx = pattern[i][2][0]
            next_nodes = (
                self.get_producer(nodes[node_idx].input[input_idx], next_op_type)
                if is_producer
                else self.get_consumer(nodes[node_idx].output[output_idx], next_op_type)
            )
            if len(next_nodes) != 1:
                return []
            next_node = next_nodes[0]
            for j in range(len(pattern[i][2])):
                node_idx, output_idx, input_idx = pattern[i][2][j]
                if (not is_producer and nodes[node_idx].output[output_idx] != next_node.input[input_idx]) or (
                    is_producer and next_node.output[output_idx] != nodes[node_idx].input[input_idx]
                ):
                    return []
            nodes.append(next_node)
        return nodes

    def match_pattern(self, pattern: List[Tuple[str, bool, List[Tuple[int, int, int]]]]):
        if pattern[0][0] not in self._op_type_to_nodes:
            return iter(())
        for node in self._op_type_to_nodes[pattern[0][0]]:
            result = self._match_pattern(node, pattern)
            if len(result) == len(pattern):
                yield result


def check_attribute_value(node: NodeProto, attr_name: str, expected_value: Any):
    value = _get_attribute(node, attr_name)
    return value == expected_value


def make_constant_node(name: str, dtype: TensorProto.DataType, dims: Sequence[int], vals: Any):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=dtype, dims=dims, vals=vals),
    )


def update_attribute(node: NodeProto, attr_name: str, attr_value: Any):
    attrs = [attr for attr in node.attribute if attr.name != attr_name]
    attrs.append(helper.make_attribute(attr_name, attr_value))
    node.ClearField("attribute")
    node.attribute.extend(attrs)


def update_graph(
    graph: GraphProto,
    nodes_to_remove: List[NodeProto],
    nodes_to_add: List[NodeProto],
    new_value_infos: List[TensorProto] = [],
):
    nodes = [node for node in graph.node if node not in nodes_to_remove]
    nodes.extend(nodes_to_add)
    graph.ClearField("node")
    graph.node.extend(nodes)
    if len(new_value_infos) > 0:
        graph.value_info.extend(new_value_infos)
