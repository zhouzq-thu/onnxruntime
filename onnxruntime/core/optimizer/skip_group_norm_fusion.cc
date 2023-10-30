// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// LayerNorm supports limited data types.
static constexpr std::array supported_data_types{
    "tensor(float16)", "tensor(float)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

static bool CheckFirstAdd(Node& add, ProviderType provider_type, bool broadcast_skip) {
  if (provider_type != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) /*|| add.GetOutputEdgesCount() != 1*/) {
    return false;
  }

  // Check the input dimensions of the "Add" node.
  const TensorShapeProto* add_input1_shape = add.MutableInputDefs()[0]->Shape();
  const TensorShapeProto* add_input2_shape = add.MutableInputDefs()[1]->Shape();

  if (add_input1_shape == nullptr || add_input2_shape == nullptr) {
    return false;
  }
  // "Add" inputs have to be 4d.
  if (add_input1_shape->dim_size() != 4 || add_input2_shape->dim_size() != 4) {
    return false;
  }

  // "Add" inputs have to be of same dimensions.
  bool is_valid_input = true;
  for (int i = 0; i < 4; i++) {
    // The second and third dimension supports broadcasting.
    // If one dimension has broadcasting, the other one shall also broadcast. And we will check it later.
    if (broadcast) {
          if ((i == 1 || i == 2) &&
            utils::HasDimValue(add_input2_shape->dim(i) &&
            add_input2_shape->dim(i).dim_value() == 1)
        {
        continue;
        }
        else {
        is_valid_input = false;
        break;
        }
    }

    if (!utils::HasDimValue(add_input1_shape->dim(i)) ||
        !utils::HasDimValue(add_input2_shape->dim(i)) ||
        add_input1_shape->dim(i).dim_value() != add_input2_shape->dim(i).dim_value()) {
          // Allow dimension only has dim_param.
          if (!utils::HasDimParam(add_input1_shape->dim(i)) ||
              !utils::HasDimParam(add_input2_shape->dim(i)) ||
              add_input1_shape->dim(i).dim_param() != add_input2_shape->dim(i).dim_param()) {
            is_valid_input = false;
            break;
          }
    }
  }

  return is_valid_input;
}

// Add2 is the 2nd add of the to be fused sub-graph
// One input (bias) should be a 1D initializer
// The other input should be a 3D tensor
static bool CheckSecondAdd(Graph& graph, Node& add, ProviderType provider_type, int& bias) {
  if (provider_type != add.GetExecutionProviderType() ||
      !IsSupportedDataType(add) ||
      add.GetOutputEdgesCount() != 1) {
    return false;
  }

  bias = -1;
  for (int i = 0; i < 2; i++) {
    if (graph_utils::NodeArgIsConstant(graph, *(add.MutableInputDefs()[i]))) {
          bias = i;
          break;
    }
  }

  if (bias < 0) {
    return false;
  }

  // Check the input dimensions of the "Add" node.
  const TensorShapeProto* add_input_shape = add.MutableInputDefs()[1 - bias]->Shape();
  const TensorShapeProto* bias_shape = add.MutableInputDefs()[bias]->Shape();

  if (add_input_shape == nullptr || bias_shape == nullptr) {
    return false;
  }

  return add_input_shape->dim_size() == 3 &&
         bias_shape->dim_size() == 1 &&
         utils::HasDimValue(add_input_shape->dim(2)) &&
         utils::HasDimValue(bias_shape->dim(0)) &&
         add_input_shape->dim(2).dim_value() == bias_shape->dim(0).dim_value();
}

// Add a Cast to convert input from float16 to float when input type is different fromm output type
static NodeArg* CastToFloat(Graph& graph, NodeArg* input, int32_t output_data_type, ProviderType provider_type) {
  if (nullptr == input->Type() ||
      input->TypeAsProto()->tensor_type().elem_type() == output_data_type ||
      output_data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return input;
  }

  auto input_shape = input->Shape();
  TypeProto input_float;
  input_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  for (auto i = 0; i < input_shape->dim_size(); ++i) {
    auto dim = input_float.mutable_tensor_type()->mutable_shape()->add_dim();
    *dim = input_shape->dim(i);
  }
  auto& cast_float = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(input->Name() + "_Float"), &input_float);

  auto& node = graph.AddNode(graph.GenerateNodeName(input->Name() + "_Cast"),
                             "Cast",
                             "Cast Input to float",
                             std::array{input},
                             std::array{&cast_float},
                             nullptr,
                             kOnnxDomain);

  node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
  node.SetExecutionProviderType(provider_type);
  return &cast_float;
}

/**
Skip Layer Normalization will fuse Add + LayerNormalization into one node, and another Add if applicable

Format 1:
  [Sub1]  Bias [Sub2] [Skip]
        \  /  /      /
        Add2 /      /
          \ /      /
        Reshape   /
           \     /
            Add1
            /   \
               GroupNorm
                  |

  After fusion:
    [Sub1]    [Sub2] [Skip]
        \     /      /
         \   /      /
          \ /      /
        Reshape   /    Bias
            \    /    /
          SkipGroupNorm
             /  |

Format 2:
      [x]  [Skip]
        \     /
         Add1
         /   \
             GroupNorm
               |

   After fusion:
      [x]  [Skip]
        \    /
      SkipGroupNorm
         /  |

Format 3 (skip has broadcast):
      [x]          [Skip]
       \              |
        \       UnSqueeze(1)
         \            |
          \     UnSqueeze(2)
           \    /
            Add1
             |
         GroupNorm
             |

   After fusion:
       [x]     [Skip]
         \      /
          \    /
      SkipGroupNorm
            |
*/

Status SkipLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    Node* p_group_norm = graph.GetNode(node_index);
    if (p_group_norm == nullptr)
          continue;  // node was removed in an earlier fusion.

    Node& gn_node = *p_group_norm;
    ORT_RETURN_IF_ERROR(Recurse(gn_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(gn_node, "GroupNorm", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(gn_node, GetCompatibleExecutionProviders()) ||
        !IsSupportedDataType(gn_node)) {
          continue;
    }

    enum class Format : int8_t {
      Format1,
      Format2,
      Format3,
      None
    };

    Node* p_add1 = nullptr;
    Node* p_add2 = nullptr;
    int bias = -1;
    Format matched_format = Format::None;

    // Format 1
    std::vector<graph_utils::EdgeEndToMatch> format1_parent_path{
        {0, 0, "Add", {7, 13, 14}, kOnnxDomain},
        {0, 0, "Reshape", {5, 13, 14, 19}, kOnnxDomain},
        {0, 0, "Add", {7, 13, 14}, kOnnxDomain}};

    std::vector<const Node::EdgeEnd*> edges;
    if (graph_utils::FindPath(gn_node, true, format1_parent_path, edges, logger)) {
          p_add1 = const_cast<Node*>(&edges[0]->GetNode());
          p_reshape = const_cast<Node*>(&edges[1]->GetNode());
          p_add2 = const_cast<Node*>(&edges[2]->GetNode());

          if (CheckFirstAdd(*p_add1, gn_node.GetExecutionProviderType(), false) &&
              CheckSecondAdd(graph, *p_add2, gn_node.GetExecutionProviderType(), bias) &&
              !graph.NodeProducesGraphOutput(*p_add1) &&
              !graph.NodeProducesGraphOutput(*p_add2)) {
            matched_format = Format::Format1;
          }
    }

    if (matched_format == Format::None) {
          // Format 2
          std::vector<graph_utils::EdgeEndToMatch> format2_parent_path{
              {0, 0, "Add", {7, 13, 14}, kOnnxDomain}};

          if (graph_utils::FindPath(gn_node, true, format2_parent_path, edges, logger)) {
            p_add1 = const_cast<Node*>(&edges[0]->GetNode());

            if (CheckFirstAdd(*p_add1, gn_node.GetExecutionProviderType(), false) &&
                !graph.NodeProducesGraphOutput(*p_add1)) {
              matched_format = Format::Format2;
            }
          }
    }

    if (matched_format == Format::None) {
          // Format 3
          std::vector<graph_utils::EdgeEndToMatch> format3_parent_path{
              {0, 0, "Add", {7, 13, 14}, kOnnxDomain},
              {0, 1, "Unsqueeze", {7, 13, 14}, kOnnxDomain},
              {0, 0, "Unsqueeze", {7, 13, 14}, kOnnxDomain},
          };

          if (graph_utils::FindPath(gn_node, true, format3_parent_path, edges, logger)) {
            p_add1 = const_cast<Node*>(&edges[0]->GetNode());

            if (CheckFirstAdd(*p_add1, gn_node.GetExecutionProviderType(), true) &&
                !graph.NodeProducesGraphOutput(*p_add1)) {
              matched_format = Format::Format3;
            }
          }
    }

    if (matched_format == Format::None) {
          continue;
    }

    NodeArg beta_place_holder("", nullptr);

    // Get the inputs for the new SkipLayerNormalization node.
    InlinedVector<NodeArg*> skip_group_norm_input_defs{
        p_add1->MutableInputDefs()[0],
        gn_node.MutableInputDefs()[1],
        gn_node.MutableInputDefs()[2],
        p_add1->MutableInputDefs()[1]  // skip
    };

    if (matched_format == Format::Format1) {
          skip_group_norm_input_defs.push_back(p_add2->MutableInputDefs()[bias]);
          nodes_to_remove.push_back(*p_add2);
    } else if (matched_format == Format::Format2) {
          skip_group_norm_input_defs[1] = p_add2->MutableInputDefs()[0];
          skip_group_norm_input_defs.push_back(p_add2->MutableInputDefs()[1]);
          nodes_to_remove.push_back(*p_add2);
    }

    nodes_to_remove.push_back(*p_add1);
    nodes_to_remove.push_back(gn_node);

    // If input types are different than output type and output type is float, insert cast node after inputs.
    for (auto& input_def : skip_group_norm_input_defs) {
          input_def = CastToFloat(graph,
                                  input_def,
                                  gn_node.MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
                                  gn_node.GetExecutionProviderType());
    }

    Node& skip_group_norm_node = graph.AddNode(graph.GenerateNodeName("SkipGroupNorm"),
                                               "SkipGroupNorm",
                                               "fused SkipGroupNorm subgraphs",
                                               skip_group_norm_input_defs,
                                               gn_node.MutableOutputDefs(), {}, kMSDomain);

    NodeAttributes attrs = gn_node.GetAttributes();
    for (auto kv : attrs) {
          skip_group_norm_node.AddAttributeProto(kv.second);
    }

    // Assign provider to this new node. Provider should be same as the provider for old node.
    skip_group_norm_node.SetExecutionProviderType(gn_node.GetExecutionProviderType());
  }

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  modified = true;

  return Status::OK();
}
}  // namespace onnxruntime
