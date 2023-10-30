# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSkipGroupNorm(Fusion):
    """
    Fuse Add + GroupNorm into one node: SkipGroupNorm
    """

    def __init__(
        self,
        model: OnnxModel,
        fused_op_type: str = "SkipGroupNorm",
        search_op_types: str = "GroupNorm",
    ):
        super().__init__(model, fused_op_type, search_op_types)
        # Update shape inference is needed since other fusions might add new edge which does not have shape info yet.
        # TODO: support optimum model symbolic shape
        self.shape_infer_helper = self.model.infer_runtime_shape({"B": 3, "2B": 6, "H": 10, "W": 14}, update=True)

        if self.shape_infer_helper is None:
            # TODO(tianleiwu): support subgraph in shape inference or add broadcasting in SkipGroupNorm op.
            logger.warning("symbolic shape inference disabled or failed.")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        add = self.model.get_parent(node, 0, output_name_to_node)

        if add is None or add.op_type != "Add":
            return

        # The number of inputs of add should be 2
        assert len(add.input) == 2

        # In some models there is input_ids->gather->add->LayerNorm and one of input of the
        # add node is initializer with fixed shape which should not be fused into SkipLayerNorm
        for add_input in add.input:
            if self.model.get_initializer(add_input) is not None:
                return

        # To avoid an Add node have two children of GroupNorm, we shall only fuse one SkipGroupNorm
        if add in self.nodes_to_remove:
            return

        skip = -1
        if self.shape_infer_helper is not None:
            shape_a = self.shape_infer_helper.get_edge_shape(add.input[0])
            shape_b = self.shape_infer_helper.get_edge_shape(add.input[1])
            assert shape_a is not None and shape_b is not None
            if len(shape_a) != 4 or len(shape_b) != 4:
                return
            if shape_a == shape_b:
                skip = 1
            elif shape_a[0] == shape_b[0] and shape_a[3] == shape_b[3]:
                if shape_b[1] == 1 and shape_b[2] == 1:
                    skip = 1
                elif shape_a[1] == 1 and shape_a[2] == 1:
                    skip = 0
            if skip < 0:
                logger.debug(
                    "skip SkipGroupNorm fusion since shape of Add inputs (%s, %s) are not expected",
                    add.input[0],
                    add.input[1],
                )
                return
        else:
            logger.debug("skip SkipGroupNorm fusion since symbolic shape inference failed")
            return

        # This means that the residual Add before the GroupNorm produces an output
        # that is consumed by some other nodes or graph output other than the GroupNorm itself
        # We can still go ahead with the SkipGroupNorm fusion but we need to
        # preserve the output of Add and that needs to be produced by SkipGroupNorm.
        add_has_graph_output = self.model.find_graph_output(add.output[0]) is not None
        residual_add_has_multiple_consumers = (
            add_has_graph_output or len(self.model.get_children(add, input_name_to_nodes)) > 1
        )

        outputs_to_keep = node.output

        if residual_add_has_multiple_consumers:
            outputs_to_keep.extend([add.output[0]])

        outputs = [node.output[0]]

        # Skip the other optional outputs of SkipGroupNorm before adding the Add's output
        if residual_add_has_multiple_consumers:
            outputs.extend([add.output[0]])

        if self.model.is_safe_to_fuse_nodes([add, node], outputs_to_keep, input_name_to_nodes, output_name_to_node):
            self.nodes_to_remove.extend([add, node])

            inputs = [add.input[1 - skip], node.input[1], node.input[2], add.input[skip]]
            skip_group_norm = helper.make_node(
                self.fused_op_type,
                inputs=inputs,
                outputs=outputs,
                name=self.model.create_node_name(self.fused_op_type, name_prefix="SkipLayerNorm"),
            )
            skip_group_norm.domain = "com.microsoft"

            # Pass attributes from GroupNorm node to SkipGroupNorm
            for att in node.attribute:
                skip_group_norm.attribute.extend([att])

            self.nodes_to_add.append(skip_group_norm)
            self.node_name_to_graph_name[skip_group_norm.name] = self.this_graph_name


class FusionBiasGroupNorm(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipGroupNorm", "SkipGroupNorm", "add bias")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 4:
            return

        return_indice = []
        (_reshape, add, _matmul) = self.model.match_parent_path(
            node, ["Reshape", "Add", "MatMul"], [None, 0, None], output_name_to_node, return_indice
        )

        # Add shall have only one children, and it shall not be graph output.
        if (add.output[0] in input_name_to_nodes and len(input_name_to_nodes[add.output[0]]) > 1) or add.output[
            0
        ] in self.model.graph.output:
            return

        if return_indice[0] not in [0, 3]:
            return

        matmul_output = add.input[return_indice[2]]
        bias_input = add.input[1 - return_indice[2]]
        skip_input = node.input[1 - return_indice[0]]

        # bias should be one dimension
        initializer = self.model.get_initializer(bias_input)
        if initializer is None:
            return
        bias_weight = NumpyHelper.to_array(initializer)
        if bias_weight is None:
            logger.debug("Bias weight not found")
            return
        if len(bias_weight.shape) != 1:
            logger.debug("Bias weight is not 1D")
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, node.output, input_name_to_nodes, output_name_to_node):
            logger.debug("Skip fusing SkipGroupNorm with Bias since it is not safe")
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        inputs = [
            matmul_output,
            node.input[1],
            node.input[2],
            skip_input,
            bias_input,
        ]
        new_node = helper.make_node(
            "SkipGroupNorm",
            inputs=inputs,
            outputs=node.output,
            name=self.model.create_node_name("SkipGroupNorm", "SkipLayerNorm_AddBias_"),
        )
        new_node.domain = "com.microsoft"

        # Pass attributes from SkipGroupNorm node to SkipGroupNorm
        for att in node.attribute:
            new_node.attribute.extend([att])

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
