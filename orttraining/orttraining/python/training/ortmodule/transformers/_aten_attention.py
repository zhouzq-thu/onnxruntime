# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import List, Tuple

from onnx import GraphProto, NodeProto, TensorProto, helper

from ..graph_transformer_registry import register_graph_transformer
from ._utils import GraphMatcher, check_attribute_value, make_constant_node, update_attribute, update_graph


def _make_efficient_attention_nodes(
    idx: int,
    q: str,
    k: str,
    v: str,
    y: str,
    dy: str,
    dq: str,
    dk: str,
    dv: str,
    bias: str,
    expand_bias: bool,
    scale: float,
    dropout_ratio: float,
):
    nodes_to_add = []
    scale_node = make_constant_node("scale_" + str(idx), TensorProto.FLOAT, [], [scale])
    dropout_ratio_node = make_constant_node("dropout_ratio_" + str(idx), TensorProto.FLOAT, [], [dropout_ratio])
    int_zero_node = make_constant_node("int_zero_" + str(idx), TensorProto.INT64, [], [0])
    true_node = make_constant_node("true_" + str(idx), TensorProto.BOOL, [], [True])
    logsumexp = helper.make_tensor_value_info("logsumexp" + str(idx), TensorProto.FLOAT, [])
    seed = helper.make_tensor_value_info("seed" + str(idx), TensorProto.INT64, [])
    offset = helper.make_tensor_value_info("offset" + str(idx), TensorProto.INT64, [])
    new_value_infos = [logsumexp, seed, offset]
    if expand_bias:
        shape_0 = helper.make_node("Shape", [q], ["shape_0_" + str(idx)], start=0, end=1)
        shape_1 = helper.make_node("Shape", [q], ["shape_1_" + str(idx)], start=2, end=3)
        shape_2 = helper.make_node("Shape", [q], ["shape_2_" + str(idx)], start=1, end=2)
        shape_3 = helper.make_node("Shape", [k], ["shape_3_" + str(idx)], start=1, end=2)
        concat = helper.make_node(
            "Concat",
            ["shape_0_" + str(idx), "shape_1_" + str(idx), "shape_2_" + str(idx), "shape_3_" + str(idx)],
            ["concated_shape_" + str(idx)],
            axis=0,
        )
        expand = helper.make_node("Expand", [bias, "concated_shape_" + str(idx)], ["expanded_bias_" + str(idx)])
        nodes_to_add.extend([shape_0, shape_1, shape_2, shape_3, concat, expand])
        bias = "expanded_bias_" + str(idx)
    fwd_node = helper.make_node(
        "ATen",
        [
            q,
            k,
            v,
            bias,
            "",
            "",
            "",
            dropout_ratio_node.output[0],
            int_zero_node.output[0],
            true_node.output[0],
            scale_node.output[0],
            "",
            "",
        ],
        [y, logsumexp.name, seed.name, offset.name],
        "efficient_attention_forward_" + str(idx),
        None,
        "org.pytorch.aten",
        operator="_efficient_attention_forward",
    )
    bwd_node = helper.make_node(
        "ATen",
        [
            dy,
            q,
            k,
            v,
            bias,
            y,
            "",
            "",
            int_zero_node.output[0],
            int_zero_node.output[0],
            logsumexp.name,
            dropout_ratio_node.output[0],
            seed.name,
            offset.name,
            int_zero_node.output[0],
            scale_node.output[0],
            "",
        ],
        [dq, dk, dv, ""],
        "efficient_attention_backward_" + str(idx),
        None,
        "org.pytorch.aten",
        operator="_efficient_attention_backward",
    )
    nodes_to_add.extend([scale_node, dropout_ratio_node, int_zero_node, true_node, fwd_node, bwd_node])
    return nodes_to_add, new_value_infos


PATTERN_0: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("Split", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Squeeze", False, [(0, 0, 0)]),  # 2
    ("Squeeze", False, [(0, 1, 0)]),  # 3
    ("Squeeze", False, [(0, 2, 0)]),  # 4
    ("Mul", False, [(2, 0, 0)]),  # 5
    ("Transpose", False, [(3, 0, 0)]),  # 6
    ("MatMul", False, [(5, 0, 0), (6, 0, 1)]),  # 7
    ("Softmax", False, [(7, 0, 0)]),  # 8
    ("MatMul", False, [(8, 0, 0), (4, 0, 1)]),  # 9
    ("Transpose", False, [(9, 0, 0)]),  # 10
    ("FusedMatMul", False, [(4, 0, 1)]),  # 11
    ("SoftmaxGrad_13", False, [(11, 0, 0), (8, 0, 1)]),  # 12
    ("FusedMatMul", False, [(6, 0, 1), (12, 0, 0)]),  # 13
    ("Mul", False, [(13, 0, 0)]),  # 14
    ("Identity", False, [(14, 0, 0)]),  # 15
    ("FusedMatMul", False, [(5, 0, 0), (12, 0, 1)]),  # 16
    ("Transpose", False, [(16, 0, 0)]),  # 17
    ("FusedMatMul", False, [(8, 0, 0)]),  # 18
    ("Transpose", True, [(11, 0, 0), (18, 0, 1)]),  # 19
    ("Unsqueeze", False, [(15, 0, 0)]),  # 20
    ("Unsqueeze", False, [(17, 0, 0)]),  # 21
    ("Unsqueeze", False, [(18, 0, 0)]),  # 22
    ("Concat", False, [(20, 0, 0), (21, 0, 1), (22, 0, 2)]),  # 23
    ("Transpose", False, [(23, 0, 0)]),  # 24
]


def _apply_transform_for_pattern_0(matcher: GraphProto, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[5].input[1])
    if not (
        check_attribute_value(nodes[0], "axis", 0)
        and check_attribute_value(nodes[1], "perm", [2, 0, 3, 1, 4])
        and matcher.get_constant_value(nodes[2].input[1]) == [0]
        and matcher.get_constant_value(nodes[3].input[1]) == [0]
        and matcher.get_constant_value(nodes[4].input[1]) == [0]
        and scale_value is not None
        and check_attribute_value(nodes[6], "perm", [0, 1, 3, 2])
        and check_attribute_value(nodes[8], "axis", -1)
        and check_attribute_value(nodes[10], "perm", [0, 2, 1, 3])
    ):
        return [], [], []

    update_attribute(nodes[1], "perm", [2, 0, 1, 3, 4])
    update_attribute(nodes[24], "perm", [1, 2, 0, 3, 4])
    nodes_to_remove = nodes[5:20]
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[2].output[0],
        nodes[3].output[0],
        nodes[4].output[0],
        nodes[10].output[0],
        nodes[19].input[0],
        nodes[20].input[0],
        nodes[21].input[0],
        nodes[22].input[0],
        "",
        False,
        float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        0.0,
    )
    return nodes_to_remove, nodes_to_add, new_value_infos


PATTERN_1: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("Split", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Squeeze", False, [(0, 0, 0)]),  # 2
    ("Squeeze", False, [(0, 1, 0)]),  # 3
    ("Squeeze", False, [(0, 2, 0)]),  # 4
    ("Mul", False, [(2, 0, 0)]),  # 5
    ("Transpose", False, [(5, 0, 0)]),  # 6
    ("MatMul", False, [(6, 0, 0), (3, 0, 1)]),  # 7
    ("Softmax", False, [(7, 0, 0)]),  # 8
    ("Transpose", False, [(4, 0, 0)]),  # 9
    ("MatMul", False, [(8, 0, 0), (9, 0, 1)]),  # 10
    ("Transpose", False, [(10, 0, 0)]),  # 11
    ("FusedMatMul", False, [(9, 0, 1)]),  # 12
    ("SoftmaxGrad_13", False, [(12, 0, 0), (8, 0, 1)]),  # 13
    ("FusedMatMul", False, [(3, 0, 1), (13, 0, 0)]),  # 14
    ("Transpose", False, [(14, 0, 0)]),  # 15
    ("Mul", False, [(15, 0, 0)]),  # 16
    ("Identity", False, [(16, 0, 0)]),  # 17
    ("FusedMatMul", False, [(6, 0, 0), (13, 0, 1)]),  # 18
    ("FusedMatMul", False, [(8, 0, 0)]),  # 19
    ("Transpose", False, [(19, 0, 0)]),  # 20
    ("Transpose", True, [(19, 0, 1), (12, 0, 0)]),  # 21
    ("Unsqueeze", False, [(17, 0, 0)]),  # 22
    ("Unsqueeze", False, [(18, 0, 0)]),  # 23
    ("Unsqueeze", False, [(20, 0, 0)]),  # 24
    ("Concat", False, [(22, 0, 0), (23, 0, 1), (24, 0, 2)]),  # 25
    ("Transpose", False, [(25, 0, 0)]),  # 26
]


def _apply_transform_for_pattern_1(matcher: GraphProto, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[5].input[1])
    if not (
        check_attribute_value(nodes[0], "axis", 0)
        and check_attribute_value(nodes[1], "perm", [2, 0, 3, 1, 4])
        and matcher.get_constant_value(nodes[2].input[1]) == [0]
        and matcher.get_constant_value(nodes[3].input[1]) == [0]
        and matcher.get_constant_value(nodes[4].input[1]) == [0]
        and scale_value is not None
        and check_attribute_value(nodes[6], "perm", [0, 1, 3, 2])
        and check_attribute_value(nodes[8], "axis", -1)
        and check_attribute_value(nodes[9], "perm", [0, 1, 3, 2])
        and check_attribute_value(nodes[11], "perm", [0, 3, 1, 2])
    ):
        return [], [], []

    update_attribute(nodes[1], "perm", [2, 0, 4, 3, 1])
    update_attribute(nodes[11], "perm", [0, 3, 2, 1])
    update_attribute(nodes[21], "perm", [0, 3, 2, 1])
    update_attribute(nodes[26], "perm", [1, 4, 0, 3, 2])
    nodes_to_remove = nodes[5:11]
    nodes_to_remove.extend(nodes[12:21])
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[2].output[0],
        nodes[3].output[0],
        nodes[4].output[0],
        nodes[10].output[0],
        nodes[21].output[0],
        nodes[22].input[0],
        nodes[23].input[0],
        nodes[24].input[0],
        "",
        False,
        float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        0.0,
    )
    return nodes_to_remove, nodes_to_add, new_value_infos


PATTERN_2: List[Tuple[str, bool, List[Tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Transpose", True, [(0, 0, 1)]),  # 2
    ("Div", False, [(0, 0, 0)]),  # 3
    ("Add", False, [(3, 0, 0)]),  # 4
    ("Softmax", False, [(4, 0, 0)]),  # 5
    ("Dropout", False, [(5, 0, 0)]),  # 6
    ("MatMul", False, [(6, 0, 0)]),  # 7
    ("Transpose", True, [(7, 0, 1)]),  # 8
    ("Transpose", False, [(7, 0, 0)]),  # 9
    ("FusedMatMul", False, [(8, 0, 1)]),  # 10
    ("DropoutGrad", False, [(10, 0, 0), (6, 1, 1)]),  # 11
    ("SoftmaxGrad_13", False, [(11, 0, 0), (5, 0, 1)]),  # 12
    ("Identity", False, [(12, 0, 0)]),  # 13
    ("Div", False, [(13, 0, 0)]),  # 14"
    ("Identity", False, [(14, 0, 0)]),  # 15
    ("FusedMatMul", False, [(2, 0, 1), (15, 0, 0)]),  # 16
    ("FusedMatMul", False, [(1, 0, 0), (15, 0, 1)]),  # 17
    ("FusedMatMul", False, [(6, 0, 0)]),  # 18
    ("Transpose", True, [(18, 0, 1)]),  # 19
    ("Transpose", False, [(16, 0, 0)]),  # 20
    ("Transpose", False, [(17, 0, 0)]),  # 21
    ("Transpose", False, [(18, 0, 0)]),  # 22
]


def _apply_transform_for_pattern_2(matcher: GraphProto, idx: int, nodes: List[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[3].input[1])
    ratio_value = matcher.get_constant_value(nodes[6].input[1])
    if not (
        check_attribute_value(nodes[1], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[2], "perm", [0, 2, 3, 1])
        and scale_value is not None
        and ratio_value is not None
        and check_attribute_value(nodes[8], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[9], "perm", [0, 2, 1, 3])
    ):
        return [], [], []

    add_input_shape_0 = matcher.get_shape(nodes[4].input[0])
    add_input_shape_1 = matcher.get_shape(nodes[4].input[1])
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[1].input[0],
        nodes[2].input[0],
        nodes[8].input[0],
        nodes[9].output[0],
        nodes[19].input[0],
        nodes[20].output[0],
        nodes[21].output[0],
        nodes[22].output[0],
        nodes[4].input[1],
        add_input_shape_0 != add_input_shape_1,
        1 / float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        ratio_value,
    )
    return nodes, nodes_to_add, new_value_infos


@register_graph_transformer(devices="cuda")
def transform_aten_efficient_attention(graph: GraphProto):
    nodes_to_remove = []
    nodes_to_add = []
    new_value_infos = []
    matcher = GraphMatcher(graph)
    idx = 0
    for nodes in matcher.match_pattern(PATTERN_0):
        remove_nodes, add_nodes, add_value_infos = _apply_transform_for_pattern_0(matcher, idx, nodes)
        if len(add_nodes) > 0:
            nodes_to_remove.extend(remove_nodes)
            nodes_to_add.extend(add_nodes)
            new_value_infos.extend(add_value_infos)
            idx += 1
    for nodes in matcher.match_pattern(PATTERN_1):
        remove_nodes, add_nodes, add_value_infos = _apply_transform_for_pattern_1(matcher, idx, nodes)
        if len(add_nodes) > 0:
            nodes_to_remove.extend(remove_nodes)
            nodes_to_add.extend(add_nodes)
            new_value_infos.extend(add_value_infos)
            idx += 1
    for nodes in matcher.match_pattern(PATTERN_2):
        remove_nodes, add_nodes, add_value_infos = _apply_transform_for_pattern_2(matcher, idx, nodes)
        if len(add_nodes) > 0:
            nodes_to_remove.extend(remove_nodes)
            nodes_to_add.extend(add_nodes)
            new_value_infos.extend(add_value_infos)
            idx += 1
    update_graph(graph, nodes_to_remove, nodes_to_add, new_value_infos)
