// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_model.h"

#include <iostream>
#include <fstream>

#include "QnnOpDef.h"
#include "HTP/QnnHtpGraph.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

bool QnnModel::GetGraphInfoFromModel(QnnModelWrapper& model_wrapper) {
  bool rt = true;

  graph_info_ = std::make_unique<GraphInfo>(model_wrapper.GetQnnGraph(),
                                            model_wrapper.GetQnnGraphName(),
                                            std::move(model_wrapper.GetGraphInputTensorWrappers()),
                                            std::move(model_wrapper.GetGraphOutputTensorWrappers()));
  if (graph_info_ == nullptr) {
    LOGS(logger_, ERROR) << "GetGraphInfoFromModel() failed to allocate GraphInfo.";
    return false;
  }

  return rt;
}

Status QnnModel::SetGraphInputOutputInfo(const GraphViewer& graph_viewer,
                                         const onnxruntime::Node& fused_node) {
  auto graph_initializers = graph_viewer.GetAllInitializedTensors();
  for (auto graph_ini : graph_initializers) {
    initializer_inputs_.emplace(graph_ini.first);
  }
  auto input_defs = fused_node.InputDefs();
  ORT_RETURN_IF_ERROR(ParseGraphInputOrOutput(input_defs, input_names_, inputs_info_, model_input_index_map_, true));

  auto output_defs = fused_node.OutputDefs();
  ORT_RETURN_IF_ERROR(ParseGraphInputOrOutput(output_defs, output_names_, outputs_info_, model_output_index_map_));

  return Status::OK();
}

Status QnnModel::ParseGraphInputOrOutput(ConstPointerContainer<std::vector<NodeArg*>>& input_output_defs,
                                         std::vector<std::string>& input_output_names,
                                         std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                         std::unordered_map<std::string, size_t>& input_output_index_map,
                                         bool is_input) {
  for (size_t i = 0, end = input_output_defs.size(), index = 0; i < end; ++i) {
    const auto& name = input_output_defs[i]->Name();
    if (is_input) {
      if (IsGraphInitializerInput(name)) {
        continue;  // exclude initializer inputs
      }
    }
    // Validate input/output shape
    LOGS(logger_, VERBOSE) << (is_input ? "input " : "output ") << i << " " << name;
    input_output_index_map.emplace(name, index++);
    const auto* shape_proto = input_output_defs[i]->Shape();  // consider use qnn_model_wrapper.GetOnnxShape
    ORT_RETURN_IF(shape_proto == nullptr, "shape_proto cannot be null for output: ", name);

    const auto& dims = shape_proto->dim();
    std::vector<int64_t> shape;
    shape.reserve(dims.size());
    for (const auto& dim : dims) {
      ORT_RETURN_IF_NOT(dim.has_dim_value(), "Dynamic shape is not supported yet, for output: ", name);
      shape.push_back(dim.dim_value());
    }
    const auto* type_proto = input_output_defs[i]->TypeAsProto();
    int32_t data_type = type_proto->tensor_type().elem_type();
    // use index i so that for graph input, it has initializers included
    input_output_info_table.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(i, data_type, std::move(shape)));
    input_output_names.push_back(name);
  }

  return Status::OK();
}

const NodeUnit& QnnModel::GetNodeUnit(const Node* node,
                                      const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map) const {
  const auto node_unit_it = node_unit_map.find(node);
  ORT_ENFORCE(node_unit_it != node_unit_map.end(), "Node does not have corresponding NodeUnit.");
  return *node_unit_it->second;
}

Status QnnModel::ComposeGraph(const GraphViewer& graph_viewer,
                              const onnxruntime::Node& fused_node,
                              const std::string& debug_json_graph_path) {
  LOGS(logger_, VERBOSE) << "ComposeGraph Graph name: " << graph_viewer.Name();

  // Holder for the NodeUnits in the graph, this will guarantee the NodeUnits is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph_viewer);

  const auto& graph_name = graph_viewer.Name();
  ORT_RETURN_IF_ERROR(SetGraphInputOutputInfo(graph_viewer, fused_node));

  qnn::QnnBackendType backend_type = qnn_backend_manager_->GetQnnBackendType();

  QnnModelWrapper qnn_model_wrapper = QnnModelWrapper(graph_viewer, logger_,
                                                      qnn_backend_manager_->GetQnnInterface(),
                                                      qnn_backend_manager_->GetQnnBackendHandle(),
                                                      model_input_index_map_,
                                                      model_output_index_map_,
                                                      initializer_inputs_,
                                                      qnn_backend_manager_->GetQnnBackendType());

  // TODO: Refactor. This is test code that turns on graph optimizations!
  if (backend_type == qnn::QnnBackendType::HTP) {
#if 1
    QnnHtpGraph_CustomConfig_t htp_graph_opt_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    htp_graph_opt_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    htp_graph_opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    htp_graph_opt_config.optimizationOption.floatValue = 3;

    QnnGraph_Config_t graph_opt_config = QNN_GRAPH_CONFIG_INIT;
    graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_opt_config.customConfig = &htp_graph_opt_config;

    const QnnGraph_Config_t* graph_configs[] = {&graph_opt_config, nullptr};
#else
    const QnnGraph_Config_t** graph_configs = nullptr;
#endif
    bool rt = qnn_model_wrapper.CreateQnnGraph(qnn_backend_manager_->GetQnnContext(), graph_name, graph_configs);
    LOGS(logger_, WARNING) << "CREATED GRAPH WITH OPTs";
    if (!rt) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to initialize qnn_model_wrapper.");
    }
  } else {
    bool rt = qnn_model_wrapper.CreateQnnGraph(qnn_backend_manager_->GetQnnContext(), graph_name);
    if (!rt) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to initialize qnn_model_wrapper.");
    }
  }

  auto get_const_initializer = [&graph_viewer](const std::string& initializer_name) {
    return graph_viewer.GetConstantInitializer(initializer_name, true);
  };

  std::unordered_set<const NodeUnit*> handled_node_units;

  auto handle_dq_q_sequence = [&](const NodeUnit& node_unit) -> bool {
    // Looking for a standalone DQ to start the sequence.
    if (node_unit.OpType() != QDQ::DQOpName || node_unit.UnitType() != NodeUnit::Type::SingleNode) {
      return false;
    }

    const Node& dq_node = node_unit.GetNode();

    // Must have a single Q child.
    auto children = graph_utils::FindChildrenByType(dq_node, QDQ::QOpName);
    if (children.size() != 1 || dq_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(dq_node)) {
      return false;
    }

    const Node& q_node = *children[0];
    const NodeUnit& q_node_unit = GetNodeUnit(&q_node, node_unit_map);

    // Q child must not already be part of a QDQ NodeUnit (i.e., be standalone).
    if (q_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
      return false;
    }

    assert(handled_node_units.count(&q_node_unit) == 0);

    // DQ and Q must have equal zero-point/scale, which must also be constant and scalar.
    if (!QDQ::IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph_viewer.ModelPath())) {
      return false;
    }

    const NodeUnitIODef& dq_input = node_unit.Inputs()[0];
    const auto& dq_input_name = qnn_model_wrapper.GetTensorName(dq_input.node_arg.Name());

    const NodeUnitIODef& q_output = q_node_unit.Outputs()[0];
    const std::string& q_output_name = q_output.node_arg.Name();

    const bool from_graph_input = qnn_model_wrapper.IsGraphInput(dq_input_name);
    const bool to_graph_output = qnn_model_wrapper.IsGraphOutput(q_output_name);

    // Don't simplify if it requires shorting the input to the output. Ex: input -> DQ -> Q -> output
    if (from_graph_input && to_graph_output) {
      return false;
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(dq_input_name)) {
      // Add the DQ input to the model wrapper.
      OnnxInputInfo dq_input_info = {};
      auto status = qnn_model_wrapper.GetOnnxInputInfo(dq_input, dq_input_info);

      ORT_ENFORCE(status.IsOK());
      ORT_ENFORCE(!dq_input_info.is_initializer);

      Qnn_TensorType_t tensor_type = from_graph_input ? QNN_TENSOR_TYPE_APP_WRITE
                                                      : (to_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE);
      QnnTensorWrapper input_tensorwrapper(dq_input_name, tensor_type, dq_input_info.qnn_data_type, dq_input_info.quant_param,
                                           std::move(dq_input_info.shape), {});
      bool added_tensor = qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper));
      ORT_ENFORCE(added_tensor);
    } else if (to_graph_output) {
      bool success = qnn_model_wrapper.OverrideTensorType(dq_input_name, QNN_TENSOR_TYPE_APP_READ);
      ORT_ENFORCE(success);

      const auto& orig_output_info = outputs_info_.at(q_output_name);
      outputs_info_.emplace(std::piecewise_construct, std::forward_as_tuple(dq_input_name),
                            std::forward_as_tuple(orig_output_info.index_, orig_output_info.data_type_,
                                                  std::vector<int64_t>(orig_output_info.shape_)));
      outputs_info_.erase(q_output_name);
    }

    // Alias the Q output to the DQ input.
    LOGS(logger_, WARNING) << "QNN EP will remove the DQ -> Q sequence with DQ node " << dq_node.Name()
                           << " and Q node " << q_node.Name() << ". Input: " << dq_input_name
                           << ", Output: " << q_output_name;
    bool alias_result = qnn_model_wrapper.AddTensorAlias(dq_input_name, q_output_name);
    ORT_ENFORCE(alias_result);

    // Add DQ and Q to the handled set.
    handled_node_units.insert(&node_unit);
    handled_node_units.insert(&q_node_unit);

    return true;
  };

  // Op builer
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);  // Inputs first
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer.GetNode(node_indices[i]));

    // Check whether it's part of NodeUnit
    const NodeUnit& node_unit = GetNodeUnit(node, node_unit_map);
    // Q, DQ nodes in the node unit only carry the quantization parameters
    // Add the QNN node when it is the target node (It's a normal node or a singel Q/DQ node)
    const std::string& op_type = node_unit.OpType();
    if (node != &node_unit.GetNode()) {
      continue;
    }

    if (handled_node_units.count(&node_unit) != 0) {
      continue;  // Already handled.
    }

    // Optimize away DQ -> Q sequences.
    if (handle_dq_q_sequence(node_unit)) {
      continue;
    }

    LOGS(logger_, WARNING) << " node name: [" << node->Name()
                           << "] node optype: [" << op_type
                           << "] as part of the NodeUnit type: [" << node_unit.OpType()
                           << "] name: [" << node_unit.Name()
                           << "]";

    if (const auto* op_builder = GetOpBuilder(op_type)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(qnn_model_wrapper, node_unit, logger_));
    }

    handled_node_units.insert(&node_unit);
  }

  const bool build_debug_json_graph = !debug_json_graph_path.empty();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ComposeQnnGraph(build_debug_json_graph), "Failed to compose Qnn graph.");

  if (build_debug_json_graph) {
    const nlohmann::json& json_graph = qnn_model_wrapper.GetQnnJSONGraph();
    std::ofstream ofs(debug_json_graph_path);

    if (ofs.is_open()) {
      ofs << json_graph.dump();
      ofs.close();
    } else {
      LOGS(logger_, WARNING) << "Could not open JSON graph file: " << debug_json_graph_path;
    }
  }

  bool rt = GetGraphInfoFromModel(qnn_model_wrapper);
  if (!rt) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetGraphInfoFromModel failed.");
  }
  LOGS(logger_, VERBOSE) << "GetGraphInfoFromModel completed.";
  return Status::OK();
}

Status QnnModel::FinalizeGraphs() {
  LOGS(logger_, VERBOSE) << "FinalizeGraphs started.";
  Qnn_ErrorHandle_t status = qnn_backend_manager_->GetQnnInterface().graphFinalize(graph_info_->Graph(),
                                                                                   qnn_backend_manager_->GetQnnProfileHandle(),
                                                                                   nullptr);
  if (QNN_GRAPH_NO_ERROR != status) {
    LOGS(logger_, ERROR) << "Failed to finalize QNN graph.";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to finalize QNN graph.");
  }

  ORT_RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo());

  LOGS(logger_, VERBOSE) << "FinalizeGraphs completed.";
  return Status::OK();
}

Status QnnModel::SetupQnnInputOutput() {
  LOGS(logger_, VERBOSE) << "Setting up QNN input/output for graph: " << graph_info_->Name();

  auto result = SetupTensors(qnn_inputs_, graph_info_->InputTensors());

  if (Status::OK() != result) {
    LOGS(logger_, ERROR) << "Failed to setup QNN input output tensors for graph: " << graph_info_->Name();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to setup QNN input tensors!");
  }

  result = SetupTensors(qnn_outputs_, graph_info_->OutputTensors(), false);
  if (Status::OK() != result) {
    LOGS(logger_, ERROR) << "Failed to setup QNN input output tensors for graph: " << graph_info_->Name();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to setup QNN output tensors!");
  }

  return Status::OK();
}

Status QnnModel::ExecuteGraph(const Ort::KernelContext& context) {
  LOGS(logger_, VERBOSE) << "QnnModel::ExecuteGraphs";
  const size_t num_inputs = context.GetInputCount();
  const size_t num_outputs = context.GetOutputCount();
  ORT_RETURN_IF_NOT(qnn_inputs_.size() <= num_inputs, "Inconsistent input sizes");
  ORT_RETURN_IF_NOT(qnn_outputs_.size() == num_outputs, "Inconsistent output sizes");

  using namespace qnn::utils;
  auto TensorDataSize = [&](auto ort_tensor) -> size_t {
    auto tensor_type_and_shape = ort_tensor.GetTensorTypeAndShapeInfo();
    size_t length = tensor_type_and_shape.GetElementCount();
    ONNXTensorElementDataType element_type = tensor_type_and_shape.GetElementType();
    size_t element_size = GetElementSizeByType(element_type);
    return element_size * length;
  };

  for (auto& qnn_input_tensor : qnn_inputs_) {
    const std::string& model_input_name(GetQnnTensorName(qnn_input_tensor));
    auto index = GetOrtInputIndex(model_input_name);
    LOGS(logger_, VERBOSE) << "model_input = " << model_input_name << " index = " << index;
    auto ort_input_tensor = context.GetInput(index);
    auto qnn_tensor_size = GetQnnTensorClientBuf(qnn_input_tensor).dataSize;
    auto ort_tensor_size = TensorDataSize(ort_input_tensor);
    LOGS(logger_, VERBOSE) << "Qnn tensor size: " << qnn_tensor_size << "Ort tensor size: " << ort_tensor_size;
    ORT_ENFORCE(qnn_tensor_size == ort_tensor_size,
                "ORT Tensor data size does not match QNN tensor data size.");
    SetQnnTensorClientBufData(qnn_input_tensor,
                              const_cast<void*>(ort_input_tensor.GetTensorData<void>()));
  }

  for (auto& qnn_output_tensor : qnn_outputs_) {
    const std::string& model_output_name(GetQnnTensorName(qnn_output_tensor));
    auto index = GetOutputIndex(model_output_name);
    LOGS(logger_, VERBOSE) << "model_output = " << model_output_name << " index = " << index;
    const auto& output_info = GetOutputInfo(model_output_name);
    const std::vector<int64_t>& output_shape = output_info->shape_;
    auto output_tensor = context.GetOutput(index, output_shape.data(), output_shape.size());
    auto qnn_tensor_size = GetQnnTensorClientBuf(qnn_output_tensor).dataSize;
    auto ort_tensor_size = TensorDataSize(output_tensor);
    LOGS(logger_, VERBOSE) << "Qnn tensor size: " << qnn_tensor_size << "Ort tensor size: " << ort_tensor_size;
    ORT_ENFORCE(qnn_tensor_size == ort_tensor_size,
                "ORT Tensor data size does not match QNN tensor data size");
    SetQnnTensorClientBufData(qnn_output_tensor,
                              const_cast<void*>(output_tensor.GetTensorData<void>()));
  }

  LOGS(logger_, VERBOSE) << "Start execute QNN graph:" << graph_info_->Name();
  auto qnn_interface = qnn_backend_manager_->GetQnnInterface();
  auto profile_backend_handle = qnn_backend_manager_->GetQnnProfileHandle();
  Qnn_ErrorHandle_t execute_status = QNN_GRAPH_NO_ERROR;
  execute_status = qnn_interface.graphExecute(graph_info_->Graph(),
                                              qnn_inputs_.data(),
                                              static_cast<uint32_t>(qnn_inputs_.size()),
                                              qnn_outputs_.data(),
                                              static_cast<uint32_t>(qnn_outputs_.size()),
                                              profile_backend_handle,
                                              nullptr);

  ORT_RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo());
  if (QNN_GRAPH_NO_ERROR != execute_status) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN graph execute error. Error code: ", execute_status);
  }

  return Status::OK();
}

Status QnnModel::GetQnnTensorDataLength(const std::vector<uint32_t>& dims,
                                        Qnn_DataType_t data_type,
                                        size_t& data_length) const {
  ORT_RETURN_IF(dims.empty(), "Tensor dimensions is nullptr");

  data_length = utils::GetElementSizeByType(data_type);

  for (size_t r = 0; r < dims.size(); r++) {
    data_length *= dims[r];
  }

  return Status::OK();
}

// Setup details for Qnn_Tensor_t for execution
// based on information in QnnTensorWrapper
Status QnnModel::SetupTensors(std::vector<Qnn_Tensor_t>& qnn_tensors,
                              const std::vector<QnnTensorWrapper>& tensor_wrappers,
                              bool is_input) {
  size_t tensor_count = tensor_wrappers.size();
  ORT_RETURN_IF(0 == tensor_count, "Zero tensor size!");
  qnn_tensors.resize(tensor_count);

  for (auto& tensor_wrapper : tensor_wrappers) {
    size_t length = 0;
    using namespace qnn::utils;
    ORT_RETURN_IF_ERROR(GetQnnTensorDataLength(tensor_wrapper.GetTensorDims(),
                                               tensor_wrapper.GetTensorDataType(),
                                               length));
    auto tensor_name = tensor_wrapper.GetName();
    auto index = is_input ? GetGraphInputIndex(tensor_name) : GetOutputIndex(tensor_name);
    qnn_tensors[index] = tensor_wrapper.GetQnnTensor();
    SetQnnTensorClientBufSize(qnn_tensors[index], static_cast<uint32_t>(length));
  }
  return Status::OK();
}

Status QnnModel::DeserializeGraphInfoFromBinaryInfo(const QnnSystemContext_GraphInfo_t& qnn_sys_ctx_graph_info) {
  std::vector<QnnTensorWrapper> input_tensor_wrappers;
  std::vector<QnnTensorWrapper> output_tensor_wrappers;

  std::string graph_name;
  if (qnn_sys_ctx_graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    graph_name.assign(qnn_sys_ctx_graph_info.graphInfoV1.graphName);
    auto graph_input_num = qnn_sys_ctx_graph_info.graphInfoV1.numGraphInputs;
    auto graph_output_num = qnn_sys_ctx_graph_info.graphInfoV1.numGraphOutputs;
    ORT_RETURN_IF(nullptr == qnn_sys_ctx_graph_info.graphInfoV1.graphInputs, "Graph from cached context doesn't have any inputs.");
    ORT_RETURN_IF(nullptr == qnn_sys_ctx_graph_info.graphInfoV1.graphOutputs, "Graph from cached context doesn't have any outputs.");

    // Copy graph input
    Qnn_Tensor_t* input_tensors = qnn_sys_ctx_graph_info.graphInfoV1.graphInputs;
    for (size_t i = 0; i < graph_input_num; ++i) {
      QnnTensorWrapper tensorwrapper(input_tensors[i]);
      input_tensor_wrappers.push_back(std::move(tensorwrapper));
    }

    // Copy graph output
    Qnn_Tensor_t* output_tensors = qnn_sys_ctx_graph_info.graphInfoV1.graphOutputs;
    for (size_t i = 0; i < graph_output_num; ++i) {
      QnnTensorWrapper tensorwrapper(output_tensors[i]);
      output_tensor_wrappers.push_back(std::move(tensorwrapper));
    }
  }
  Qnn_GraphHandle_t graph;
  auto qnn_interface = qnn_backend_manager_->GetQnnInterface();
  qnn_interface.graphRetrieve(qnn_backend_manager_->GetQnnContext(),
                              graph_name.c_str(), &graph);

  graph_info_ = std::make_unique<GraphInfo>(graph,
                                            graph_name,
                                            std::move(input_tensor_wrappers),
                                            std::move(output_tensor_wrappers));
  ORT_RETURN_IF(graph_info_ == nullptr, "Failed to allocate GraphInfo");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
