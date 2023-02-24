# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from fusion_base import Fusion
from fusion_skiplayernorm import FusionSkipLayerNormalization
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)

# TODO: Support decoder self/cross attention fusion and encoder self attention fusion
class FusionT5Attention(FusionAttention):
    """
    Fuse T5 Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        use_multi_head_attention = True
        super().__init__(model, hidden_size, num_heads, attention_mask, use_multi_head_attention, ["SkipSimplifiedLayerNormalization"])

    def create_attention_node(
        self,
        query: str,
        key: str,
        value: str,
        mask_index: str,
        past_key: str,
        past_value: str,
        num_heads: int,
        hidden_size: int,
        output: str,
    ) -> Union[NodeProto, None]:

        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        attention_node_name = self.model.create_node_name("MultiHeadAttention")
        attention_inputs = [
            query,
            "" if key is None else key , # key
            "" if value is None else value, # value
            "", # bias
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")
        if past_key is not None:
            assert(past_value is not None)
            attention_inputs.append(past_key)
            attention_inputs.append(past_value)

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )

        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend([helper.make_attribute("scale", 1.0)])
        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node


    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if normalize_node.op_type != "SkipSimplifiedLayerNormalization":
            return

        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 0, 0, 0],
        )
        if qkv_nodes is None:
            return

        _, reshape_qkv, transpose_qkv, matmul_qkv = qkv_nodes

        qkv_shape_nodes = self.model.match_parent_path(
            reshape_qkv,
            ["Concat", "Unsqueeze", "Gather", "Shape"],
            [1, 0, 0, 0],
        )
        input_shape_node = qkv_shape_nodes[-1]

        past_value = matmul_qkv.input[1]
        if past_value in output_name_to_node:
            return

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "MatMul"],
            [0, 0, 0],
        )
        if qk_nodes is None:
            return
        _, add_qk, matmul_qk = qk_nodes

        mask_nodes = self.model.match_parent_path(
            add_qk,
            ["Add", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
            [1, 1, 0, 1, 0, 0],
        )
        if mask_nodes is None:
            return

        mul_node = mask_nodes[1]
        if mask_nodes[1].op_type != "Mul":
            return

        _, mul_val = self.model.get_constant_input(mul_node)
        if mul_val != -10000:
            self.mask_filter_value = mul_val

        mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

        v_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose"],
            [1],
        )
        if v_nodes is None:
            return

        transpose_v = v_nodes[0]

        past_key = transpose_v.input[0]
        if past_key in output_name_to_node:
            return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [0, 0, 0],
        )
        if q_nodes is None:
            return

        transpose_q, reshape_q, matmul_q = q_nodes

        if matmul_q.input[0] != input_shape_node.input[0]:
            return
        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
        new_node = self.create_attention_node(
            matmul_q.output[0],
            None,
            None,
            mask_index,
            past_key,
            past_value,
            q_num_heads,
            q_hidden_size,
            reshape_qkv.output[0],
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend(qkv_nodes[1:])
        self.nodes_to_remove.extend(qk_nodes)
        self.nodes_to_remove.extend(v_nodes)
        self.nodes_to_remove.extend(q_nodes[:-1])

        self.prune_graph = True







class FusionRelativePositionBiasBlock(Fusion):
    def __init__(self, model: OnnxModel, max_distance: int):
        super().__init__(model, "RelativePositionBias", ["Add", "Slice"])
        self.max_distance = max_distance
        # bidirectional=(not self.is_decoder)
        self.is_bidirectional = False

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # TODO: Optimization opportunity: only last dimension of relative_position_bias is used in decoder.
        # Cuda kernel can be optimized to only compute last dimension.
        if node.op_type != "Add" and node.op_type != "Slice":
            return

        compute_bias_nodes = self.model.match_parent_path(
            node, ["Unsqueeze", "Transpose", "Gather", "Where"], [0, 0, 0, 1]
        )
        if compute_bias_nodes is None:
            compute_bias_nodes = self.model.match_parent_path(
                node, ["Unsqueeze", "Transpose", "Gather", "Add", "Where"], [0, 0, 0, 1, 1]
            )
            if compute_bias_nodes is None:
                return

        gather = compute_bias_nodes[2]
        where = compute_bias_nodes[-1]
        unsqueeze = compute_bias_nodes[0]

        compute_buckets_nodes = self.model.match_parent_path(
            where,
            ["Min", "ConstantOfShape", "Shape", "Add", "Cast", "Mul", "Div", "Log", "Div"],
            [2, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        if compute_buckets_nodes is None:
            return

        div = compute_buckets_nodes[-1]

        range_nodes = self.model.match_parent_path(
            div,
            ["Cast", "Neg", "Min", "ConstantOfShape", "Shape", "Sub", "Unsqueeze", "Range"],
            [0, 0, 0, 1, 0, 0, 0, 0],
        )
        if range_nodes is None:
            range_nodes = self.model.match_parent_path(
                div, ["Cast", "Abs", "Sub", "Unsqueeze", "Range"], [0, 0, 0, 0, 0]
            )
            self.is_bidirectional = True
            if range_nodes is None:
                return

        range_node = range_nodes[-1]

        self.nodes_to_remove.extend(compute_bias_nodes)
        self.nodes_to_remove.extend(compute_buckets_nodes)
        self.nodes_to_remove.extend(range_nodes)

        node_name_prefix = "encoder" if self.is_bidirectional else "decoder"

        table_weight_i = self.model.get_initializer(gather.input[0])
        table_weight = NumpyHelper.to_array(table_weight_i)
        table_weight_t = np.transpose(table_weight)
        bias_table = helper.make_tensor(
            name=self.model.create_node_name("bias_table_weight", name_prefix=node_name_prefix),
            data_type=TensorProto.FLOAT,
            dims=[np.shape(table_weight)[0], np.shape(table_weight)[1]],
            vals=table_weight_t.flatten().tolist(),
        )

        self.model.add_initializer(bias_table, self.this_graph_name)
        inputs = [bias_table.name, range_node.input[1], range_node.input[1]]
        outputs = [unsqueeze.output[0]]
        rpb_node = helper.make_node(
            "RelativePositionBias",
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name("RelativePositionBias", name_prefix=node_name_prefix),
        )
        rpb_node.domain = "com.microsoft"
        rpb_node.attribute.extend([helper.make_attribute("max_distance", self.max_distance)])
        rpb_node.attribute.extend([helper.make_attribute("is_bidirectional", self.is_bidirectional)])

        self.nodes_to_add.append(rpb_node)
        self.node_name_to_graph_name[rpb_node.name] = self.this_graph_name


class FusionSkipSimplifiedLayerNormalization(FusionSkipLayerNormalization):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipSimplifiedLayerNormalization", "SimplifiedLayerNormalization")
        self.shape_infer_helper = self.model.infer_runtime_shape(
            {"batch_size": 2, "seq_len": 1, "encode_sequence_length": 8, "past_decode_sequence_length": 4}, update=True
        )

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        super().fuse(node, input_name_to_nodes, output_name_to_node)


class T5OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionT5Attention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.skip_layer_norm_fusion = FusionSkipSimplifiedLayerNormalization(self)
        # TODO: consider retrive max_distance from model.
        # math.log(max_distance / (num_buckets // 2))
        self.rpb_fusion = FusionRelativePositionBiasBlock(self, 128)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_skip_layer_norm(self):
        self.skip_layer_norm_fusion.apply()

    # Remove get_extended_attention_mask() since it generates all zeros.
    def remove_extended_mask_decoder_init(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["RelativePositionBias"], [0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def remove_extended_mask_decoder(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Concat",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["Slice", "RelativePositionBias"], [0, 0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def postprocess(self):
        self.rpb_fusion.apply()
        # remove get_extended_attention_mask() since it generates all zeros.
        self.remove_extended_mask_decoder_init()
        self.remove_extended_mask_decoder()

        self.prune_graph()
