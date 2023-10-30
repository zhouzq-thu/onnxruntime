// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class SkipGroupNormFusion

Rewrite graph fusing Add + GroupNorm subgraph to a single SkipGroupNorm node.

*/
class SkipGroupNormFusion : public GraphTransformer {
 public:
  explicit SkipGroupNormFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("SkipGroupNormFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
