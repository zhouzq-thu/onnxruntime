// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pre_inliner.h"

namespace onnxruntime {

namespace {

const InlinedHashSet<std::string> kDomains = {"pkg.onnxscript.torch_lib"};
const InlinedHashSet<std::string> kOpTypes = {"CastLike"};

static Status InlineGraph(Graph& graph, bool& modified_graph) {
  // recurse into nested graphs first so we process from bottom up
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      ORT_RETURN_IF_ERROR(InlineGraph(*subgraph, modified_graph));
    }
  }

  // See if the node with no provider can be inlined. If one such nodes can be
  // successfully inlined, we re-run the partitioner on the modified graph.
  // NOTE: Inlining the function will change the nodes in the Graph instance, so we can't do that while iterating
  // using graph.Nodes().
  InlinedVector<Node*> nodes_to_inline;
  for (auto& node : graph.Nodes()) {
    if ((kDomains.find(node.Domain()) != kDomains.end() || kOpTypes.find(node.OpType()) != kOpTypes.end()) &&
        node.CanBeInlined()) {
      nodes_to_inline.push_back(&node);
    }
  }

  for (auto* node : nodes_to_inline) {
    ORT_RETURN_IF_ERROR(graph.InlineFunction(*node));
    modified_graph = true;
  }

  return Status::OK();
}

}  // namespace

Status PreInliner::ApplyImpl(Graph& graph, bool& modified, int, const logging::Logger&) const {
  return InlineGraph(graph, modified);
}

}  // namespace onnxruntime
