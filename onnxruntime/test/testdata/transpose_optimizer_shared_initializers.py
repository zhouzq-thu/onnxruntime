import numpy as np  # noqa: F401
import onnx
from onnx import TensorProto, helper

# Create model with shared initializers and no broadcasting.
# Transpose optimizer should be able to 'see through' the DQ to update the initializer layout and push the Transpose
# through the Add nodes, and update the fp32 initializers.
# The first initializer update in either case should be a copy.
# The second time we should see the same modification is being done, use the copy, and remove the now unused original
# initializer.
#
# If we are not broadcasting, TransposeInput in onnx_transpose_optimizer.cc is used to transpose from 1, 2, 3 to
# 3, 1, 2 when pushing through the first Transpose.
# If we are broadcasting, UnsqueezeInput is also used to adjust the initializer from [2, 3] to [1, 2, 3] followed by
# transpose to 3, 1, 2.
# TODO: Is this a 2 step with UnqueezeInput happening first to go to [3, 1, 4] followed by TransposeInput to go to
# [4, 1, 3] and we need to track that the shared initializer is updated twice
def create_model(broadcast_weights: bool):
    if broadcast_weights:
        bias_shape = [2, 2]
        bias_values = np.random.randn(2, 2)
    else:
        # we need a shape that has a 1 in it make the cost check happy
        bias_shape = [1, 3, 2, 2]
        bias_values = np.random.randn(1, 3, 2, 2)

    graph = helper.make_graph(
        name="graph",
        inputs=[
            helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 2, 2, 3]),
        ],
        initializer=[
            helper.make_tensor("bias_quant", TensorProto.UINT8, bias_shape, bias_values.astype(np.uint8)),
            helper.make_tensor("bias_fp32", TensorProto.FLOAT, bias_shape, bias_values.astype(np.float32)),
            helper.make_tensor("dq_scale0", TensorProto.FLOAT, [], [1.5]),
            helper.make_tensor("dq_zp0", TensorProto.UINT8, [], [5]),
            helper.make_tensor("dq_scale1", TensorProto.FLOAT, [], [0.5]),
        ],
        nodes=[
            # Transpose input from channels last to channels first
            helper.make_node("Transpose", ["input0"], ["input_T"], perm=[0, 3, 1, 2]),
            helper.make_node("DequantizeLinear", ["bias_quant", "dq_scale0", "dq_zp0"], ["DQ0"], "DQ0"),
            helper.make_node("Add", ["input_T", "DQ0"], ["A0"], "A0"),
            helper.make_node("DequantizeLinear", ["bias_quant", "dq_scale1"], ["DQ1"], "DQ1"),
            helper.make_node("Add", ["A0", "DQ1"], ["A1"], "A1"),
            helper.make_node("DequantizeLinear", ["bias_quant", "dq_scale0"], ["DQ2"], "DQ2"),
            helper.make_node("Add", ["A1", "DQ2"], ["A2"], "A2"),
            helper.make_node("Add", ["A2", "bias_fp32"], ["A3"], "A3"),
            helper.make_node("Add", ["A3", "bias_fp32"], ["A4"], "A4"),
            # NCHW to NHWC
            helper.make_node("Transpose", ["A4"], ["output0"], perm=[0, 2, 3, 1]),
        ],
        outputs=[
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 2, 2, 3]),
        ],
    )

    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    return model


# def create_model(broadcast_weights: bool):
#     if broadcast_weights:
#         bias_shape = [2, 2]
#         bias_values = np.random.randn(2, 2)
#     else:
#         # we need a shape that has a 1 in it make the cost check happy
#         bias_shape = [1, 3, 2, 2]
#         bias_values = np.random.randn(1, 3, 2, 2)
#
#     graph = helper.make_graph(
#         name="graph",
#         inputs=[
#             helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 2, 2, 3]),
#         ],
#         initializer=[
#             helper.make_tensor("bias0_fp32", TensorProto.FLOAT, bias_shape, bias_values.astype(np.float32)),
#             helper.make_tensor("bias1_fp32", TensorProto.FLOAT, bias_shape, bias_values.astype(np.float32)),
#         ],
#         nodes=[
#             # Transpose input from channels last to channels first
#             helper.make_node("Transpose", ["input0"], ["input_T"], perm=[0, 3, 1, 2]),
#             helper.make_node("Add", ["input_T", "bias0_fp32"], ["A0"], "A0"),
#             helper.make_node("Add", ["A0", "bias0_fp32"], ["A1"], "A1"),
#             helper.make_node("Add", ["A1", "bias1_fp32"], ["A2"], "A2"),
#             helper.make_node("Add", ["A2", "bias1_fp32"], ["A3"], "A3"),
#             helper.make_node("Add", ["A3", "bias1_fp32"], ["A4"], "A4"),
#             # NCHW to NHWC
#             helper.make_node("Transpose", ["A4"], ["output0"], perm=[0, 2, 3, 1]),
#         ],
#         outputs=[
#             helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 2, 2, 3]),
#         ],
#     )
#
#     model = helper.make_model(graph)
#     onnx.checker.check_model(model, full_check=True)
#     return model


if __name__ == "__main__":
    model = create_model(broadcast_weights=False)
    onnx.save(model, "transpose_optimizer_shared_initializers.onnx")
    model = create_model(broadcast_weights=True)
    onnx.save(model, "transpose_optimizer_shared_initializers_broadcast.onnx")

    # model = create_model_dq(broadcast_weights=False)
    # onnx.save(model, "transpose_optimizer_shared_initializers_dq.onnx")
    # model = create_model_dq(broadcast_weights=True)
    # onnx.save(model, "transpose_optimizer_shared_initializers_broadcast_dq.onnx")
