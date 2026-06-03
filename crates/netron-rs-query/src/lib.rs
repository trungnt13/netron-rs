use std::{
  collections::{BTreeMap, BTreeSet, VecDeque},
  path::{Path, PathBuf},
  sync::atomic::{AtomicU64, Ordering},
};

use netron_rs_core::{
  Attribute, AttributeValue, Dimension, DimensionValue, Function, FunctionNode, FunctionValue,
  Model, ModelError, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
  TypeInfo, Value,
};
use netron_rs_layout::{
  LayoutBounds, LayoutEdge, LayoutGraph, LayoutNode, LayoutNodeKind, LayoutOptions, LayoutPoint,
  LayoutStats,
};
use serde::Serialize;

pub const SESSION_API_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchKind {
  Graph,
  Node,
  Value,
  Tensor,
  Function,
  Metadata,
  OperatorSet,
  Module,
  Operation,
  Region,
  Block,
  Symbol,
  Dialect,
  Attribute,
  Resource,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchEntry {
  pub kind: SearchKind,
  pub handle: EntityHandle,
  pub graph: Option<usize>,
  pub id: usize,
  pub name: Option<String>,
  pub operator: Option<String>,
  pub origin: Option<&'static str>,
  #[serde(skip)]
  searchable: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EntityHandle {
  Graph {
    graph: usize,
  },
  Node {
    graph: usize,
    node: usize,
  },
  Value {
    graph: usize,
    value: usize,
  },
  Tensor {
    tensor: usize,
  },
  Function {
    function: usize,
  },
  Metadata {
    owner: String,
    key: String,
  },
  OperatorSet {
    domain: Option<String>,
    version: i64,
  },
  Diagnostic {
    diagnostic: usize,
  },
  MlirModule {
    module: usize,
  },
  MlirFunction {
    function: usize,
  },
  MlirOperation {
    scope: String,
    operation: usize,
  },
  MlirValue {
    scope: String,
    value: usize,
  },
  MlirRegion {
    scope: String,
    region: usize,
  },
  MlirBlock {
    scope: String,
    block: usize,
  },
  MlirSymbol {
    symbol: usize,
  },
  MlirDialect {
    dialect: String,
  },
  MlirAttribute {
    scope: String,
    attribute: usize,
  },
  MlirResource {
    resource: usize,
  },
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatKind {
  Onnx,
  Mlir,
  Unknown,
}

#[derive(Debug, Clone)]
pub enum FormatIndex {
  Onnx(OnnxIndex),
  Mlir(MlirIndex),
  Unknown(ModelIndex),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSourceKind {
  File {
    path: PathBuf,
    base_dir: PathBuf,
  },
  Memory {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
  },
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSource {
  pub kind: ModelSourceKind,
  pub byte_len: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub content_identity: Option<String>,
  pub allow_unsafe_paths: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SessionLimits {
  pub export: usize,
  pub search: usize,
  pub slice: usize,
  pub layout: usize,
  pub preview: usize,
  pub detail: usize,
  pub diagnostics: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
  pub api_version: u32,
  pub session_id: u64,
  pub source: ModelSource,
  pub format: FormatKind,
  pub source_format_name: String,
  pub byte_len: usize,
  pub graphs: usize,
  pub functions: usize,
  pub nodes: usize,
  pub values: usize,
  pub tensors: usize,
  pub initializers: usize,
  pub subgraphs: usize,
  pub sparse_tensors: usize,
  pub metadata: usize,
  pub opsets: usize,
  pub external_data: usize,
  pub mlir_resources: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub onnx: Option<OnnxSummary>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub mlir: Option<MlirSummary>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticKind {
  Info,
  Warning,
  Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub handle: Option<EntityHandle>,
  pub kind: DiagnosticKind,
  pub code: &'static str,
  pub message: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OnnxSummary {
  pub producer: Option<String>,
  pub producer_version: Option<String>,
  pub model_domain: Option<String>,
  pub model_version: Option<i64>,
  pub description: Option<String>,
  pub graph_count: usize,
  pub function_count: usize,
  pub node_count: usize,
  pub value_count: usize,
  pub tensor_count: usize,
  pub initializer_count: usize,
  pub subgraph_count: usize,
  pub sparse_tensor_count: usize,
  pub opsets: Vec<OnnxOperatorSet>,
  pub metadata_keys: Vec<String>,
  pub graph_summaries: Vec<OnnxGraphSummary>,
  pub histograms: OnnxHistograms,
}

#[derive(Debug, Clone, Serialize)]
pub struct OnnxOperatorSet {
  pub domain: Option<String>,
  pub version: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OnnxGraphSummary {
  pub handle: EntityHandle,
  pub name: Option<String>,
  pub node_count: usize,
  pub value_count: usize,
  pub tensor_count: usize,
  pub input_count: usize,
  pub output_count: usize,
  pub initializer_count: usize,
  pub subgraph_count: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OnnxHistograms {
  pub graph_node_counts: Vec<HistogramEntry>,
  pub graph_value_counts: Vec<HistogramEntry>,
  pub graph_tensor_counts: Vec<HistogramEntry>,
  pub operator_types: Vec<HistogramEntry>,
  pub domains: Vec<HistogramEntry>,
  pub dtypes: Vec<HistogramEntry>,
  pub storage_kinds: Vec<HistogramEntry>,
  pub shape_ranks: Vec<HistogramEntry>,
  pub fan_in: Vec<HistogramEntry>,
  pub fan_out: Vec<HistogramEntry>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct HistogramEntry {
  pub key: String,
  pub count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityDetail {
  pub handle: EntityHandle,
  pub title: String,
  pub fields: BTreeMap<String, String>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub related: Vec<EntityHandle>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub format: FormatKind,
  pub scope: EntityHandle,
  pub cache_key: String,
  pub limit_used: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub warnings: Vec<String>,
  pub entities: Vec<SliceEntity>,
  pub edges: Vec<SliceEdge>,
  pub boundaries: Vec<SliceBoundary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceEntity {
  pub id: String,
  pub kind: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub handle: Option<EntityHandle>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub label: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub operator: Option<String>,
  pub boundary: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceEdge {
  pub from: String,
  pub to: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub value: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceBoundary {
  pub id: String,
  pub reason: String,
  pub omitted_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub format: FormatKind,
  pub scope: EntityHandle,
  pub cache_key: String,
  pub limit_used: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub warnings: Vec<String>,
  pub graph: LayoutGraph,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirSymbolEntry {
  pub handle: EntityHandle,
  pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirSummary {
  pub module_count: usize,
  pub function_count: usize,
  pub operation_count: usize,
  pub value_count: usize,
  pub block_argument_count: usize,
  pub region_count: usize,
  pub block_count: usize,
  pub symbol_count: usize,
  pub dialect_count: usize,
  pub attribute_count: usize,
  pub resource_count: usize,
  pub diagnostic_count: usize,
  pub modules: Vec<MlirScopeSummary>,
  pub functions: Vec<MlirScopeSummary>,
  pub regions: Vec<MlirRegionSummary>,
  pub blocks: Vec<MlirBlockSummary>,
  pub dialects: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirScopeSummary {
  pub handle: EntityHandle,
  pub scope_id: String,
  pub name: Option<String>,
  pub operation_count: usize,
  pub value_count: usize,
  pub block_argument_count: usize,
  pub attribute_count: usize,
  pub region_count: usize,
  pub block_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirRegionSummary {
  pub handle: EntityHandle,
  pub scope_id: String,
  pub block_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirBlockSummary {
  pub handle: EntityHandle,
  pub scope_id: String,
  pub operation_count: usize,
  pub value_count: usize,
  pub block_argument_count: usize,
}

#[derive(Debug, Clone)]
struct MlirAttributeDetail {
  scope: String,
  name: String,
  values: Vec<String>,
}

#[derive(Debug, Clone)]
struct MlirResourceDetail {
  scope: String,
  name: String,
  kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorMetadata {
  pub handle: EntityHandle,
  pub name: Option<String>,
  pub element_type: String,
  pub shape: Vec<String>,
  pub storage: TensorStorageKind,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub byte_len: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub element_count: Option<usize>,
  #[serde(skip_serializing_if = "BTreeMap::is_empty")]
  pub external_data: BTreeMap<String, String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub sparse_values: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub sparse_indices: Option<usize>,
  #[serde(skip_serializing_if = "BTreeMap::is_empty")]
  pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorStorageKind {
  Absent,
  InlineBytes,
  ElementList,
  External,
  Sparse,
}

#[derive(Debug, Clone)]
pub struct OnnxIndex {
  entries: Vec<SearchEntry>,
  tensor_metadata: Vec<TensorMetadata>,
  graph_summaries: Vec<OnnxGraphSummary>,
  histograms: OnnxHistograms,
  graph_count: usize,
  function_count: usize,
  node_count: usize,
  value_count: usize,
  tensor_count: usize,
  initializer_count: usize,
  subgraph_count: usize,
  sparse_tensor_count: usize,
  metadata_keys: Vec<String>,
  opsets: Vec<OnnxOperatorSet>,
  diagnostics: Vec<Diagnostic>,
}

impl OnnxIndex {
  pub fn build(model: &Model) -> Self {
    let mut entries = Vec::new();
    let metadata_keys = model
      .metadata
      .properties
      .keys()
      .cloned()
      .collect::<Vec<_>>();
    for (index, (key, value)) in model.metadata.properties.iter().enumerate() {
      entries.push(SearchEntry::new_with_handle_and_search_terms(
        SearchKind::Metadata,
        EntityHandle::Metadata {
          owner: "model".to_owned(),
          key: key.clone(),
        },
        None,
        index,
        Some(key.clone()),
        None,
        Some("onnx_metadata"),
        [value.clone()],
      ));
    }
    for (index, opset) in model.metadata.opsets.iter().enumerate() {
      let domain = opset.domain.clone();
      entries.push(SearchEntry::new_with_handle_and_search_terms(
        SearchKind::OperatorSet,
        EntityHandle::OperatorSet {
          domain: domain.clone(),
          version: opset.version,
        },
        None,
        index,
        domain.clone().or_else(|| Some("default".to_owned())),
        None,
        Some("onnx_opset"),
        [domain.unwrap_or_default(), opset.version.to_string()],
      ));
    }

    let mut node_count = 0;
    let mut value_count = 0;
    let tensor_count = model.tensors.len();
    let mut subgraph_count = 0;
    let mut diagnostics = Vec::new();

    for (graph_index, graph) in model.graphs.iter().enumerate() {
      let graph_name = graph.name.map(|id| model.strings.get(id).to_owned());
      let mut graph_terms = searchable_terms(graph_name.clone(), &graph.metadata);
      graph_terms.extend(model.metadata.properties.keys().cloned());
      graph_terms.extend(model.metadata.properties.values().cloned());
      entries.push(SearchEntry::new_with_search_terms(
        SearchKind::Graph,
        Some(graph_index),
        graph_index,
        graph_name,
        None,
        None,
        graph_terms,
      ));

      subgraph_count += graph.subgraphs.len();

      for node in &graph.nodes {
        node_count += 1;

        let name = node.name.map(|id| model.strings.get(id).to_owned());
        let operator = Some(model.strings.get(node.operator.name).to_owned());
        let mut search_terms = searchable_terms(name.clone(), &node.metadata);

        if let Some(domain) = node.operator.domain {
          search_terms.push(model.strings.get(domain).to_owned());
        }

        if node
          .outputs
          .iter()
          .flatten()
          .any(|value| graph.values[value.index()].initializer.is_some())
        {
          search_terms.push("initializer".to_string());
        }

        for attribute in &node.attributes {
          if let Some(diagnostic) = attribute_diagnostic(
            "graph node",
            graph_index,
            node.id.index(),
            Some(model.strings.get(attribute.name)),
            &attribute.value,
          ) {
            diagnostics.push(diagnostic);
          }
        }

        entries.push(SearchEntry::new_with_search_terms(
          SearchKind::Node,
          Some(graph_index),
          node.id.index(),
          name,
          operator,
          Some(node.operator.origin),
          search_terms,
        ));
      }

      for value in &graph.values {
        value_count += 1;

        let mut search_terms = searchable_terms(
          Some(model.strings.get(value.name).to_owned()),
          &value.metadata,
        );
        if value.initializer.is_some() {
          search_terms.push("initializer".to_string());
        }
        entries.push(SearchEntry::new_with_search_terms(
          SearchKind::Value,
          Some(graph_index),
          value.id.index(),
          Some(model.strings.get(value.name).to_owned()),
          None,
          None,
          search_terms,
        ));
      }
    }

    for tensor in &model.tensors {
      let mut search_terms = searchable_terms(
        tensor.name.map(|id| model.strings.get(id).to_owned()),
        &tensor.metadata,
      );

      match &tensor.storage {
        netron_rs_core::TensorStorage::External { entries } => {
          for (key, value) in entries {
            search_terms.push(key.to_owned());
            search_terms.push(value.to_owned());
          }
          search_terms.push("external".to_owned());
          search_terms.push("external_data".to_owned());
        }
        netron_rs_core::TensorStorage::Sparse { .. } => {
          search_terms.push("sparse".to_owned());
        }
        _ => {}
      }

      entries.push(SearchEntry::new_with_search_terms(
        SearchKind::Tensor,
        None,
        tensor.id.index(),
        tensor.name.map(|id| model.strings.get(id).to_owned()),
        None,
        None,
        search_terms,
      ));
    }

    for (index, function) in model.functions.iter().enumerate() {
      let mut search_terms = vec![model.strings.get(function.name).to_owned()];
      if let Some(domain) = function.domain {
        search_terms.push(model.strings.get(domain).to_owned());
      }
      if let Some(overload) = function.overload {
        search_terms.push(model.strings.get(overload).to_owned());
      }
      search_terms.extend(function.metadata.keys().cloned());
      search_terms.extend(function.metadata.values().cloned());
      search_terms.extend(
        function
          .outputs
          .iter()
          .chain(&function.inputs)
          .chain(&function.attributes)
          .map(|name| model.strings.get(*name).to_owned()),
      );

      for value in &function.values {
        if value.initializer.is_some() {
          search_terms.push("initializer".to_owned());
        }
      }

      for (node_index, node) in function.nodes.iter().enumerate() {
        search_terms.extend(node.metadata.keys().cloned());
        search_terms.extend(node.metadata.values().cloned());
        search_terms.push(model.strings.get(node.operator.name).to_owned());
        if let Some(domain) = node.operator.domain {
          search_terms.push(model.strings.get(domain).to_owned());
        }
        if let Some(overload) = node.operator.overload {
          search_terms.push(model.strings.get(overload).to_owned());
        }
        for attribute in &node.attributes {
          if let Some(diagnostic) = attribute_diagnostic(
            "function node",
            index,
            node_index,
            Some(model.strings.get(attribute.name)),
            &attribute.value,
          ) {
            diagnostics.push(diagnostic);
          }
        }
      }

      entries.push(SearchEntry::new_with_search_terms(
        SearchKind::Function,
        None,
        index,
        Some(model.strings.get(function.name).to_owned()),
        None,
        None,
        search_terms,
      ));
    }

    let graph_count = model.graphs.len();
    let function_count = model.functions.len();
    let initializer_count = initializer_count(model);
    let sparse_tensor_count = sparse_tensor_count(model);
    let tensor_metadata = tensor_metadata(model);
    let graph_summaries = graph_summaries(model);
    let histograms = onnx_histograms(model);
    let opsets = model
      .metadata
      .opsets
      .iter()
      .map(|entry| OnnxOperatorSet {
        domain: entry.domain.clone(),
        version: entry.version,
      })
      .collect();
    assign_diagnostic_handles(&mut diagnostics);

    Self {
      entries,
      tensor_metadata,
      graph_summaries,
      histograms,
      graph_count,
      function_count,
      node_count,
      value_count,
      tensor_count,
      initializer_count,
      subgraph_count,
      sparse_tensor_count,
      metadata_keys,
      opsets,
      diagnostics,
    }
  }

  pub fn search(&self, query: &str, limit: usize) -> Vec<SearchEntry> {
    search_index_entries(&self.entries, query, limit)
  }

  pub fn tensor_metadata(&self, limit: usize) -> Vec<TensorMetadata> {
    self.tensor_metadata.iter().take(limit).cloned().collect()
  }

  pub fn detail(&self, model: &Model, handle: &EntityHandle, limit: usize) -> Option<EntityDetail> {
    onnx_detail(model, self, handle, limit)
  }
}

#[derive(Debug, Clone)]
pub struct MlirIndex {
  entries: Vec<SearchEntry>,
  summary: MlirSummary,
  symbols: Vec<String>,
  attributes: Vec<MlirAttributeDetail>,
  resources: Vec<MlirResourceDetail>,
  diagnostics: Vec<Diagnostic>,
}

impl MlirIndex {
  pub fn build(model: &Model, source_text: Option<&str>) -> Self {
    let mut entries = Vec::new();
    let mut modules = Vec::new();
    let mut functions = Vec::new();
    let mut regions = Vec::new();
    let mut blocks = Vec::new();
    let mut dialects = BTreeSet::new();
    let mut symbol_set = BTreeSet::new();
    let mut symbols = Vec::new();
    let mut attribute_count = 0;
    let mut resource_count = 0;
    let mut attributes = Vec::new();
    let mut resources = Vec::new();
    let mut diagnostics = Vec::new();

    for (key, value) in &model.metadata.properties {
      entries.push(mlir_attribute_entry(
        "model",
        attribute_count,
        key,
        [value.clone()],
      ));
      attributes.push(MlirAttributeDetail {
        scope: "model".to_owned(),
        name: key.to_owned(),
        values: vec![value.clone()],
      });
      attribute_count += 1;
      if key.starts_with("bytecode.diagnostic.") {
        diagnostics.push(Diagnostic {
          handle: None,
          kind: DiagnosticKind::Warning,
          code: "mlir.bytecode",
          message: value.clone(),
          source: Some("bytecode".to_owned()),
        });
      }
    }

    for (module_index, graph) in model.graphs.iter().enumerate() {
      let scope = mlir_module_scope(module_index);
      let name = graph.name.map(|id| model.strings.get(id).to_owned());
      let block_arguments = 0;
      modules.push(MlirScopeSummary {
        handle: EntityHandle::MlirModule {
          module: module_index,
        },
        scope_id: scope.clone(),
        name: name.clone(),
        operation_count: graph.nodes.len(),
        value_count: graph.values.len(),
        block_argument_count: block_arguments,
        attribute_count: graph.metadata.len(),
        region_count: 1,
        block_count: 1,
      });
      push_mlir_region_block_summaries(
        &mut entries,
        &mut regions,
        &mut blocks,
        &scope,
        graph.nodes.len(),
        graph.values.len(),
        block_arguments,
      );

      let mut terms = searchable_terms(name.clone(), &graph.metadata);
      terms.push(scope.clone());
      entries.push(SearchEntry::new_with_handle_and_search_terms(
        SearchKind::Module,
        EntityHandle::MlirModule {
          module: module_index,
        },
        Some(module_index),
        module_index,
        name,
        None,
        Some("MLIR"),
        terms,
      ));
      if let Some(name) = graph.name {
        insert_mlir_symbol(
          &mut entries,
          &mut symbol_set,
          &mut symbols,
          &scope,
          model.strings.get(name),
        );
      }
      index_metadata_attributes(
        &mut entries,
        &mut attributes,
        &mut attribute_count,
        &scope,
        &graph.metadata,
      );

      for node in &graph.nodes {
        index_mlir_operation(
          model,
          &mut entries,
          &mut dialects,
          &mut symbol_set,
          &mut symbols,
          &mut attributes,
          &mut attribute_count,
          &mut resource_count,
          &mut resources,
          &scope,
          node.id.index(),
          None,
          Some(module_index),
          &node.operator,
          &node.metadata,
          &node.attributes,
          node.inputs.iter().flatten().count(),
          node.outputs.iter().flatten().count(),
        );
      }
      for value in &graph.values {
        let terms = mlir_value_terms(model, model.strings.get(value.name), &value.type_info);
        entries.push(SearchEntry::new_with_handle_and_search_terms(
          SearchKind::Value,
          EntityHandle::MlirValue {
            scope: scope.clone(),
            value: value.id.index(),
          },
          Some(module_index),
          value.id.index(),
          Some(model.strings.get(value.name).to_owned()),
          None,
          Some("MLIR"),
          terms,
        ));
      }
    }

    for (function_index, function) in model.functions.iter().enumerate() {
      let scope = mlir_function_scope(function_index);
      let name = model.strings.get(function.name).to_owned();
      let block_arguments = mlir_function_block_argument_count(model, function);
      functions.push(MlirScopeSummary {
        handle: EntityHandle::MlirFunction {
          function: function_index,
        },
        scope_id: scope.clone(),
        name: Some(name.clone()),
        operation_count: function.nodes.len(),
        value_count: function.values.len(),
        block_argument_count: block_arguments,
        attribute_count: function.metadata.len() + function.attributes.len(),
        region_count: 1,
        block_count: 1,
      });
      push_mlir_region_block_summaries(
        &mut entries,
        &mut regions,
        &mut blocks,
        &scope,
        function.nodes.len(),
        function.values.len(),
        block_arguments,
      );

      let mut terms = searchable_terms(Some(name.clone()), &function.metadata);
      terms.push(scope.clone());
      terms.extend(
        function
          .inputs
          .iter()
          .map(|id| model.strings.get(*id).to_owned()),
      );
      terms.extend(
        function
          .outputs
          .iter()
          .map(|id| model.strings.get(*id).to_owned()),
      );
      terms.extend(
        function
          .attributes
          .iter()
          .map(|id| model.strings.get(*id).to_owned()),
      );
      entries.push(SearchEntry::new_with_handle_and_search_terms(
        SearchKind::Function,
        EntityHandle::MlirFunction {
          function: function_index,
        },
        None,
        function_index,
        Some(name.clone()),
        None,
        Some("MLIR"),
        terms,
      ));
      insert_mlir_symbol(&mut entries, &mut symbol_set, &mut symbols, &scope, &name);
      index_metadata_attributes(
        &mut entries,
        &mut attributes,
        &mut attribute_count,
        &scope,
        &function.metadata,
      );

      let block_arg_names = mlir_function_block_argument_names(model, function);
      for (value_index, value) in function.values.iter().enumerate() {
        let value_name = model.strings.get(value.name);
        let mut terms = mlir_value_terms(model, value_name, &value.type_info);
        if block_arg_names.contains(value_name) {
          terms.push("block_argument".to_owned());
        }
        if value.initializer.is_some() {
          terms.extend(["dense".to_owned(), "constant".to_owned()]);
        }
        entries.push(SearchEntry::new_with_handle_and_search_terms(
          SearchKind::Value,
          EntityHandle::MlirValue {
            scope: scope.clone(),
            value: value_index,
          },
          None,
          value_index,
          Some(value_name.to_owned()),
          None,
          Some("MLIR"),
          terms,
        ));
      }
      for (operation_index, node) in function.nodes.iter().enumerate() {
        index_mlir_operation(
          model,
          &mut entries,
          &mut dialects,
          &mut symbol_set,
          &mut symbols,
          &mut attributes,
          &mut attribute_count,
          &mut resource_count,
          &mut resources,
          &scope,
          operation_index,
          Some(function_index),
          None,
          &node.operator,
          &node.metadata,
          &node.attributes,
          node.inputs.iter().flatten().count(),
          node.outputs.iter().flatten().count(),
        );
      }
    }
    if let Some(source_text) = source_text {
      index_mlir_source_terms(
        &mut entries,
        &mut blocks,
        &mut attributes,
        &mut attribute_count,
        &mut resource_count,
        &mut resources,
        source_text,
      );
    }

    if is_mlir_bytecode_model(model) {
      for resource in mlir_bytecode_resources(model) {
        entries.push(SearchEntry::new_with_handle_and_search_terms(
          SearchKind::Resource,
          EntityHandle::MlirResource {
            resource: resource_count,
          },
          None,
          resource_count,
          Some(resource.name.clone()),
          None,
          Some("MLIR"),
          [
            resource.scope.clone(),
            resource.kind.clone(),
            "mlirbc".to_owned(),
          ],
        ));
        resources.push(resource);
        resource_count += 1;
      }
    }

    let mut operation_count = modules
      .iter()
      .map(|module| module.operation_count)
      .sum::<usize>()
      + functions
        .iter()
        .map(|function| function.operation_count)
        .sum::<usize>();
    let mut value_count = model
      .graphs
      .iter()
      .map(|graph| graph.values.len())
      .sum::<usize>()
      + model
        .functions
        .iter()
        .map(|function| function.values.len())
        .sum::<usize>();
    let mut block_argument_count = functions
      .iter()
      .map(|function| function.block_argument_count)
      .sum();
    let mut module_count = modules.len();
    let mut function_count = functions.len();
    let mut region_count = regions.len();
    let mut block_count = blocks.len();
    let mut summary_attribute_count = attribute_count;
    if is_mlir_bytecode_model(model) {
      module_count = mlir_bytecode_count(model, "bytecode.ir_module_count").unwrap_or(module_count);
      function_count =
        mlir_bytecode_count(model, "bytecode.ir_function_count").unwrap_or(function_count);
      operation_count =
        mlir_bytecode_count(model, "bytecode.ir_operation_count").unwrap_or(operation_count);
      value_count = mlir_bytecode_count(model, "bytecode.ir_value_count").unwrap_or(value_count);
      block_argument_count = mlir_bytecode_count(model, "bytecode.ir_block_argument_count")
        .unwrap_or(block_argument_count);
      region_count = mlir_bytecode_count(model, "bytecode.ir_region_count").unwrap_or(region_count);
      block_count = mlir_bytecode_count(model, "bytecode.ir_block_count").unwrap_or(block_count);
      summary_attribute_count =
        mlir_bytecode_count(model, "bytecode.attribute_count").unwrap_or(summary_attribute_count);
      if let Some(module) = modules.first_mut() {
        module.operation_count = operation_count;
        module.value_count = value_count;
        module.block_argument_count = block_argument_count;
        module.attribute_count = summary_attribute_count;
        module.region_count = region_count;
        module.block_count = block_count;
      }
      if let Some(region) = regions.first_mut() {
        region.block_count = block_count;
      }
      if let Some(block) = blocks.first_mut() {
        block.operation_count = operation_count;
        block.value_count = value_count;
        block.block_argument_count = block_argument_count;
      }
    }
    let summary = MlirSummary {
      module_count,
      function_count,
      operation_count,
      value_count,
      block_argument_count,
      region_count,
      block_count,
      symbol_count: symbol_set.len(),
      dialect_count: dialects.len(),
      attribute_count: summary_attribute_count,
      resource_count,
      diagnostic_count: diagnostics.len(),
      modules,
      functions,
      regions,
      blocks,
      dialects: dialects.into_iter().collect(),
    };
    assign_diagnostic_handles(&mut diagnostics);

    Self {
      entries,
      summary,
      symbols,
      attributes,
      resources,
      diagnostics,
    }
  }

  pub fn search(&self, query: &str, limit: usize) -> Vec<SearchEntry> {
    search_index_entries(&self.entries, query, limit)
  }

  pub fn detail(&self, model: &Model, handle: &EntityHandle, limit: usize) -> Option<EntityDetail> {
    mlir_detail(model, self, handle, limit)
  }
}

fn is_mlir_bytecode_model(model: &Model) -> bool {
  model.metadata.properties.contains_key("bytecode.version")
}

fn mlir_bytecode_count(model: &Model, key: &str) -> Option<usize> {
  model.metadata.properties.get(key)?.parse().ok()
}

fn mlir_bytecode_resources(model: &Model) -> Vec<MlirResourceDetail> {
  let count = mlir_bytecode_count(model, "bytecode.resource_count").unwrap_or_default();
  (0..count)
    .filter_map(|index| {
      Some(MlirResourceDetail {
        scope: model
          .metadata
          .properties
          .get(&format!("bytecode.resource.{index}.scope"))?
          .clone(),
        name: model
          .metadata
          .properties
          .get(&format!("bytecode.resource.{index}.name"))?
          .clone(),
        kind: model
          .metadata
          .properties
          .get(&format!("bytecode.resource.{index}.kind"))?
          .clone(),
      })
    })
    .collect()
}

fn push_mlir_region_block_summaries(
  entries: &mut Vec<SearchEntry>,
  regions: &mut Vec<MlirRegionSummary>,
  blocks: &mut Vec<MlirBlockSummary>,
  scope: &str,
  operation_count: usize,
  value_count: usize,
  block_argument_count: usize,
) {
  let region = MlirRegionSummary {
    handle: EntityHandle::MlirRegion {
      scope: scope.to_owned(),
      region: 0,
    },
    scope_id: scope.to_owned(),
    block_count: 1,
  };
  let block = MlirBlockSummary {
    handle: EntityHandle::MlirBlock {
      scope: scope.to_owned(),
      block: 0,
    },
    scope_id: scope.to_owned(),
    operation_count,
    value_count,
    block_argument_count,
  };
  entries.push(SearchEntry::new_with_handle_and_search_terms(
    SearchKind::Region,
    region.handle.clone(),
    None,
    regions.len(),
    Some(format!("{scope} region 0")),
    None,
    Some("MLIR"),
    [scope.to_owned(), "region".to_owned()],
  ));
  entries.push(SearchEntry::new_with_handle_and_search_terms(
    SearchKind::Block,
    block.handle.clone(),
    None,
    blocks.len(),
    Some(format!("{scope} block 0")),
    None,
    Some("MLIR"),
    [
      scope.to_owned(),
      "block".to_owned(),
      "entry".to_owned(),
      "block_argument".to_owned(),
    ],
  ));
  regions.push(region);
  blocks.push(block);
}

fn index_metadata_attributes(
  entries: &mut Vec<SearchEntry>,
  attribute_details: &mut Vec<MlirAttributeDetail>,
  attribute_count: &mut usize,
  scope: &str,
  metadata: &BTreeMap<String, String>,
) {
  for (key, value) in metadata {
    entries.push(mlir_attribute_entry(
      scope,
      *attribute_count,
      key,
      [value.clone()],
    ));
    attribute_details.push(MlirAttributeDetail {
      scope: scope.to_owned(),
      name: key.to_owned(),
      values: vec![value.clone()],
    });
    *attribute_count += 1;
  }
}

#[allow(clippy::too_many_arguments)]
fn index_mlir_operation(
  model: &Model,
  entries: &mut Vec<SearchEntry>,
  dialects: &mut BTreeSet<String>,
  symbols: &mut BTreeSet<String>,
  symbol_names: &mut Vec<String>,
  attribute_details: &mut Vec<MlirAttributeDetail>,
  attribute_count: &mut usize,
  resource_count: &mut usize,
  resource_details: &mut Vec<MlirResourceDetail>,
  scope: &str,
  operation_index: usize,
  function: Option<usize>,
  module: Option<usize>,
  operator: &Operator,
  metadata: &BTreeMap<String, String>,
  attributes: &[Attribute],
  fan_in: usize,
  fan_out: usize,
) {
  let operator_name = model.strings.get(operator.name).to_owned();
  let dialect = mlir_dialect(&operator_name);
  if dialects.insert(dialect.clone()) {
    entries.push(SearchEntry::new_with_handle_and_search_terms(
      SearchKind::Dialect,
      EntityHandle::MlirDialect {
        dialect: dialect.clone(),
      },
      None,
      dialects.len() - 1,
      Some(dialect.clone()),
      None,
      Some("MLIR"),
      [operator_name.clone()],
    ));
  }

  let mut terms = searchable_terms(None, metadata);
  terms.extend([
    scope.to_owned(),
    dialect,
    format!("fan_in:{fan_in}"),
    format!("fan_out:{fan_out}"),
  ]);
  for attribute in attributes {
    let name = model.strings.get(attribute.name).to_owned();
    let value_terms = attribute_value_terms(model, &attribute.value);
    terms.push(name.clone());
    terms.extend(value_terms.clone());
    entries.push(mlir_attribute_entry(
      scope,
      *attribute_count,
      &name,
      value_terms.clone(),
    ));
    attribute_details.push(MlirAttributeDetail {
      scope: scope.to_owned(),
      name: name.to_owned(),
      values: value_terms.clone(),
    });
    *attribute_count += 1;
    for symbol in mlir_symbol_terms(&name, &value_terms) {
      insert_mlir_symbol(entries, symbols, symbol_names, scope, &symbol);
    }
    if name == "rodata" {
      for value in value_terms {
        entries.push(SearchEntry::new_with_handle_and_search_terms(
          SearchKind::Resource,
          EntityHandle::MlirResource {
            resource: *resource_count,
          },
          None,
          *resource_count,
          Some(value.clone()),
          None,
          Some("MLIR"),
          [scope.to_owned(), "rodata".to_owned()],
        ));
        resource_details.push(MlirResourceDetail {
          scope: scope.to_owned(),
          name: value.clone(),
          kind: "rodata".to_owned(),
        });
        *resource_count += 1;
      }
    }
  }
  for (key, value) in metadata {
    if key == "location" {
      terms.push(value.clone());
    }
  }

  let (graph, handle) = if let Some(function) = function {
    (
      None,
      EntityHandle::MlirOperation {
        scope: mlir_function_scope(function),
        operation: operation_index,
      },
    )
  } else {
    (
      module,
      EntityHandle::MlirOperation {
        scope: scope.to_owned(),
        operation: operation_index,
      },
    )
  };
  entries.push(SearchEntry::new_with_handle_and_search_terms(
    SearchKind::Operation,
    handle,
    graph,
    operation_index,
    None,
    Some(operator_name),
    Some("MLIR"),
    terms,
  ));
}

fn mlir_attribute_entry<I: IntoIterator<Item = String>>(
  scope: &str,
  id: usize,
  name: &str,
  terms: I,
) -> SearchEntry {
  let mut terms = terms.into_iter().collect::<Vec<_>>();
  terms.push(scope.to_owned());
  SearchEntry::new_with_handle_and_search_terms(
    SearchKind::Attribute,
    EntityHandle::MlirAttribute {
      scope: scope.to_owned(),
      attribute: id,
    },
    None,
    id,
    Some(name.to_owned()),
    None,
    Some("MLIR"),
    terms,
  )
}

fn insert_mlir_symbol(
  entries: &mut Vec<SearchEntry>,
  symbols: &mut BTreeSet<String>,
  symbol_names: &mut Vec<String>,
  scope: &str,
  symbol: &str,
) {
  let symbol = symbol.trim_start_matches('@');
  if symbol.is_empty() || !symbols.insert(symbol.to_owned()) {
    return;
  }
  let id = symbol_names.len();
  symbol_names.push(symbol.to_owned());
  entries.push(SearchEntry::new_with_handle_and_search_terms(
    SearchKind::Symbol,
    EntityHandle::MlirSymbol { symbol: id },
    None,
    id,
    Some(symbol.to_owned()),
    None,
    Some("MLIR"),
    [scope.to_owned()],
  ));
}

fn index_mlir_source_terms(
  entries: &mut Vec<SearchEntry>,
  blocks: &mut Vec<MlirBlockSummary>,
  attribute_details: &mut Vec<MlirAttributeDetail>,
  attribute_count: &mut usize,
  resource_count: &mut usize,
  resource_details: &mut Vec<MlirResourceDetail>,
  source_text: &str,
) {
  for label in mlir_source_block_labels(source_text) {
    let block = blocks.len();
    entries.push(SearchEntry::new_with_handle_and_search_terms(
      SearchKind::Block,
      EntityHandle::MlirBlock {
        scope: "source".to_owned(),
        block,
      },
      None,
      block,
      Some(label.clone()),
      None,
      Some("MLIR"),
      [
        label.trim_start_matches('^').to_owned(),
        "block_label".to_owned(),
      ],
    ));
    blocks.push(MlirBlockSummary {
      handle: EntityHandle::MlirBlock {
        scope: "source".to_owned(),
        block,
      },
      scope_id: "source".to_owned(),
      operation_count: 0,
      value_count: 0,
      block_argument_count: 0,
    });
  }
  for type_text in mlir_source_type_literals(source_text) {
    entries.push(mlir_attribute_entry(
      "source",
      *attribute_count,
      "type",
      [type_text.clone()],
    ));
    attribute_details.push(MlirAttributeDetail {
      scope: "source".to_owned(),
      name: "type".to_owned(),
      values: vec![type_text.clone()],
    });
    *attribute_count += 1;
  }
  for resource in mlir_source_dense_resources(source_text) {
    entries.push(SearchEntry::new_with_handle_and_search_terms(
      SearchKind::Resource,
      EntityHandle::MlirResource {
        resource: *resource_count,
      },
      None,
      *resource_count,
      Some(resource.clone()),
      None,
      Some("MLIR"),
      ["dense_resource".to_owned()],
    ));
    resource_details.push(MlirResourceDetail {
      scope: "source".to_owned(),
      name: resource,
      kind: "dense_resource".to_owned(),
    });
    *resource_count += 1;
  }
}

fn mlir_source_block_labels(text: &str) -> Vec<String> {
  text
    .lines()
    .filter_map(|line| {
      let line = line.trim_start();
      let rest = line.strip_prefix('^')?;
      let end = rest
        .find(|ch: char| ch.is_whitespace() || matches!(ch, '(' | ':'))
        .unwrap_or(rest.len());
      (end > 0).then(|| format!("^{}", &rest[..end]))
    })
    .collect()
}

fn mlir_source_type_literals(text: &str) -> Vec<String> {
  ["tensor<", "memref<", "vector<"]
    .into_iter()
    .flat_map(|prefix| mlir_angle_literals(text, prefix))
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect()
}

fn mlir_source_dense_resources(text: &str) -> Vec<String> {
  mlir_angle_literals(text, "dense_resource<")
    .into_iter()
    .map(|literal| {
      literal
        .trim_start_matches("dense_resource<")
        .trim_end_matches('>')
        .to_owned()
    })
    .filter(|value| !value.is_empty())
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect()
}

fn mlir_angle_literals(text: &str, prefix: &str) -> Vec<String> {
  let mut literals = Vec::new();
  let mut offset = 0;
  while let Some(start) = text[offset..].find(prefix).map(|index| offset + index) {
    let mut depth = 0usize;
    for (relative, ch) in text[start..].char_indices() {
      match ch {
        '<' => depth += 1,
        '>' => {
          depth = depth.saturating_sub(1);
          if depth == 0 {
            let end = start + relative + 1;
            literals.push(text[start..end].to_owned());
            offset = end;
            break;
          }
        }
        _ => {}
      }
    }
    if offset <= start {
      break;
    }
  }
  literals
}

fn mlir_value_terms(model: &Model, name: &str, type_info: &Option<TypeInfo>) -> Vec<String> {
  let mut terms = vec![name.to_owned()];
  if let Some(type_info) = type_info {
    add_type_terms(
      model,
      &mut terms,
      type_info.element_type.as_ref(),
      &type_info.shape,
    );
  }
  terms
}

fn add_type_terms(
  model: &Model,
  terms: &mut Vec<String>,
  element_type: Option<&TensorElementType>,
  shape: &[Dimension],
) {
  terms.push("type".to_owned());
  terms.push(format!("rank:{}", shape.len()));
  if let Some(element_type) = element_type {
    terms.push(element_type_name(element_type));
  }
  if !shape.is_empty() {
    let dims = shape
      .iter()
      .map(|dimension| dimension_label(model, &dimension.value))
      .collect::<Vec<_>>();
    terms.extend(dims.iter().cloned());
    terms.push(format!("tensor<{}>", dims.join("x")));
  }
}

fn attribute_value_terms(model: &Model, value: &AttributeValue) -> Vec<String> {
  match value {
    AttributeValue::String(id) | AttributeValue::Reference(id) => {
      vec![model.strings.get(*id).to_owned()]
    }
    AttributeValue::Tensor(id) => vec![format!("tensor:{}", id.index()), "dense".to_owned()],
    AttributeValue::Strings(values) => values
      .iter()
      .map(|id| model.strings.get(*id).to_owned())
      .collect(),
    AttributeValue::Tensors(values) => values
      .iter()
      .map(|id| format!("tensor:{}", id.index()))
      .collect(),
    AttributeValue::Ints(values) => values.iter().map(i64::to_string).collect(),
    AttributeValue::Floats(values) => values.iter().map(f32::to_string).collect(),
    AttributeValue::Bool(value) => vec![value.to_string()],
    AttributeValue::Float(value) => vec![value.to_string()],
    AttributeValue::Int(value) => vec![value.to_string()],
    AttributeValue::Type(value) => vec![value.clone()],
    AttributeValue::TypeList(values) => values.clone(),
    AttributeValue::Bytes { byte_len } => vec![format!("bytes:{byte_len}")],
    AttributeValue::Graph(id) => vec![format!("graph:{}", id.index())],
    AttributeValue::Graphs(values) => values
      .iter()
      .map(|id| format!("graph:{}", id.index()))
      .collect(),
    AttributeValue::Null | AttributeValue::Unsupported(_) => Vec::new(),
  }
}

fn mlir_symbol_terms(name: &str, values: &[String]) -> Vec<String> {
  if matches!(
    name,
    "sym_name" | "callee" | "global" | "variable" | "entry_points" | "rodata"
  ) {
    values
      .iter()
      .map(|value| value.trim_start_matches('@').to_owned())
      .filter(|value| !value.is_empty())
      .collect()
  } else {
    Vec::new()
  }
}

fn mlir_dialect(operator: &str) -> String {
  operator
    .split_once('.')
    .map(|(dialect, _)| dialect)
    .unwrap_or("builtin")
    .to_owned()
}

fn mlir_module_scope(index: usize) -> String {
  format!("module:{index}")
}

fn mlir_function_scope(index: usize) -> String {
  format!("function:{index}")
}

fn mlir_function_block_argument_count(model: &Model, function: &Function) -> usize {
  mlir_function_block_argument_names(model, function).len()
}

fn mlir_function_block_argument_names(model: &Model, function: &Function) -> BTreeSet<String> {
  let declared = function
    .inputs
    .iter()
    .chain(&function.outputs)
    .map(|id| model.strings.get(*id))
    .collect::<BTreeSet<_>>();
  let produced = function
    .nodes
    .iter()
    .flat_map(|node| node.outputs.iter().flatten())
    .map(|id| model.strings.get(*id))
    .collect::<BTreeSet<_>>();
  function
    .values
    .iter()
    .filter(|value| value.initializer.is_none())
    .map(|value| model.strings.get(value.name))
    .filter(|name| !declared.contains(name) && !produced.contains(name))
    .map(str::to_owned)
    .collect()
}

fn attribute_diagnostic(
  scope: &str,
  graph: usize,
  node: usize,
  name: Option<&str>,
  value: &AttributeValue,
) -> Option<Diagnostic> {
  match value {
    AttributeValue::Unsupported(message) => Some(Diagnostic {
      handle: None,
      kind: DiagnosticKind::Warning,
      code: "onnx.unsupported_attribute",
      message: format!(
        "Unsupported attribute value for {scope} {graph}/{node} {name:?}: {message}"
      ),
      source: None,
    }),
    _ => None,
  }
}

fn assign_diagnostic_handles(diagnostics: &mut [Diagnostic]) {
  for (index, diagnostic) in diagnostics.iter_mut().enumerate() {
    diagnostic.handle = Some(EntityHandle::Diagnostic { diagnostic: index });
  }
}

fn searchable_terms(name: Option<String>, metadata: &BTreeMap<String, String>) -> Vec<String> {
  let mut search = Vec::new();
  if let Some(name) = name {
    search.push(name);
  }
  search.extend(metadata.keys().cloned());
  search.extend(metadata.values().cloned());
  search
}

fn graph_summaries(model: &Model) -> Vec<OnnxGraphSummary> {
  model
    .graphs
    .iter()
    .enumerate()
    .map(|(index, graph)| {
      let initializer_count = graph
        .values
        .iter()
        .filter(|value| value.initializer.is_some())
        .count();
      OnnxGraphSummary {
        handle: EntityHandle::Graph { graph: index },
        name: graph.name.map(|id| model.strings.get(id).to_owned()),
        node_count: graph.nodes.len(),
        value_count: graph.values.len(),
        tensor_count: initializer_count,
        input_count: graph.inputs.len(),
        output_count: graph.outputs.len(),
        initializer_count,
        subgraph_count: graph.subgraphs.len(),
      }
    })
    .collect()
}

fn onnx_histograms(model: &Model) -> OnnxHistograms {
  let mut graph_node_counts = BTreeMap::new();
  let mut graph_value_counts = BTreeMap::new();
  let mut graph_tensor_counts = BTreeMap::new();
  let mut operator_histograms = OperatorHistogramMaps::default();
  let mut dtypes = BTreeMap::new();
  let mut storage_kinds = BTreeMap::new();
  let mut shape_ranks = BTreeMap::new();

  for graph in &model.graphs {
    let graph_tensor_count = graph
      .values
      .iter()
      .filter(|value| value.initializer.is_some())
      .count();
    increment(&mut graph_node_counts, graph.nodes.len().to_string());
    increment(&mut graph_value_counts, graph.values.len().to_string());
    increment(&mut graph_tensor_counts, graph_tensor_count.to_string());

    for node in &graph.nodes {
      operator_histograms.record(
        model,
        &node.operator,
        node.inputs.iter().flatten().count(),
        node.outputs.iter().flatten().count(),
      );
    }
    for value in &graph.values {
      if let Some(type_info) = &value.type_info {
        if let Some(element_type) = &type_info.element_type {
          increment(&mut dtypes, element_type_name(element_type));
        }
        increment(&mut shape_ranks, type_info.shape.len().to_string());
      }
    }
  }

  for function in &model.functions {
    for node in &function.nodes {
      operator_histograms.record(
        model,
        &node.operator,
        node.inputs.iter().flatten().count(),
        node.outputs.iter().flatten().count(),
      );
    }
    for value in &function.values {
      if let Some(type_info) = &value.type_info {
        if let Some(element_type) = &type_info.element_type {
          increment(&mut dtypes, element_type_name(element_type));
        }
        increment(&mut shape_ranks, type_info.shape.len().to_string());
      }
    }
  }

  for tensor in &model.tensors {
    increment(&mut dtypes, element_type_name(&tensor.element_type));
    increment(
      &mut storage_kinds,
      storage_kind_name(&tensor.storage).to_owned(),
    );
    increment(&mut shape_ranks, tensor.shape.len().to_string());
  }

  OnnxHistograms {
    graph_node_counts: histogram_entries(graph_node_counts),
    graph_value_counts: histogram_entries(graph_value_counts),
    graph_tensor_counts: histogram_entries(graph_tensor_counts),
    operator_types: histogram_entries(operator_histograms.operator_types),
    domains: histogram_entries(operator_histograms.domains),
    dtypes: histogram_entries(dtypes),
    storage_kinds: histogram_entries(storage_kinds),
    shape_ranks: histogram_entries(shape_ranks),
    fan_in: histogram_entries(operator_histograms.fan_in),
    fan_out: histogram_entries(operator_histograms.fan_out),
  }
}

#[derive(Default)]
struct OperatorHistogramMaps {
  operator_types: BTreeMap<String, usize>,
  domains: BTreeMap<String, usize>,
  fan_in: BTreeMap<String, usize>,
  fan_out: BTreeMap<String, usize>,
}

impl OperatorHistogramMaps {
  fn record(&mut self, model: &Model, operator: &Operator, inputs: usize, outputs: usize) {
    increment(
      &mut self.operator_types,
      model.strings.get(operator.name).to_owned(),
    );
    increment(&mut self.domains, operator_domain(model, operator));
    increment(&mut self.fan_in, inputs.to_string());
    increment(&mut self.fan_out, outputs.to_string());
  }
}

fn increment(map: &mut BTreeMap<String, usize>, key: String) {
  *map.entry(key).or_default() += 1;
}

fn histogram_entries(map: BTreeMap<String, usize>) -> Vec<HistogramEntry> {
  let mut entries = map
    .into_iter()
    .map(|(key, count)| HistogramEntry { key, count })
    .collect::<Vec<_>>();
  entries.sort_by(|left, right| right.count.cmp(&left.count).then(left.key.cmp(&right.key)));
  entries
}

fn limit_histograms(histograms: &OnnxHistograms, limit: usize) -> OnnxHistograms {
  OnnxHistograms {
    graph_node_counts: limit_histogram_entries(&histograms.graph_node_counts, limit),
    graph_value_counts: limit_histogram_entries(&histograms.graph_value_counts, limit),
    graph_tensor_counts: limit_histogram_entries(&histograms.graph_tensor_counts, limit),
    operator_types: limit_histogram_entries(&histograms.operator_types, limit),
    domains: limit_histogram_entries(&histograms.domains, limit),
    dtypes: limit_histogram_entries(&histograms.dtypes, limit),
    storage_kinds: limit_histogram_entries(&histograms.storage_kinds, limit),
    shape_ranks: limit_histogram_entries(&histograms.shape_ranks, limit),
    fan_in: limit_histogram_entries(&histograms.fan_in, limit),
    fan_out: limit_histogram_entries(&histograms.fan_out, limit),
  }
}

fn limit_histogram_entries(entries: &[HistogramEntry], limit: usize) -> Vec<HistogramEntry> {
  entries.iter().take(limit).cloned().collect()
}

fn limit_mlir_summary(summary: &MlirSummary, limit: usize) -> MlirSummary {
  MlirSummary {
    module_count: summary.module_count,
    function_count: summary.function_count,
    operation_count: summary.operation_count,
    value_count: summary.value_count,
    block_argument_count: summary.block_argument_count,
    region_count: summary.region_count,
    block_count: summary.block_count,
    symbol_count: summary.symbol_count,
    dialect_count: summary.dialect_count,
    attribute_count: summary.attribute_count,
    resource_count: summary.resource_count,
    diagnostic_count: summary.diagnostic_count,
    modules: summary.modules.iter().take(limit).cloned().collect(),
    functions: summary.functions.iter().take(limit).cloned().collect(),
    regions: summary.regions.iter().take(limit).cloned().collect(),
    blocks: summary.blocks.iter().take(limit).cloned().collect(),
    dialects: summary.dialects.iter().take(limit).cloned().collect(),
  }
}

fn tensor_metadata(model: &Model) -> Vec<TensorMetadata> {
  model
    .tensors
    .iter()
    .map(|tensor| tensor_metadata_entry(model, tensor))
    .collect()
}

fn tensor_metadata_entry(model: &Model, tensor: &Tensor) -> TensorMetadata {
  let (storage, byte_len, element_count, external_data, sparse_values, sparse_indices) =
    match &tensor.storage {
      TensorStorage::Absent => (
        TensorStorageKind::Absent,
        None,
        None,
        BTreeMap::new(),
        None,
        None,
      ),
      TensorStorage::InlineBytes { byte_len } => (
        TensorStorageKind::InlineBytes,
        Some(*byte_len),
        None,
        BTreeMap::new(),
        None,
        None,
      ),
      TensorStorage::ElementList { len } => (
        TensorStorageKind::ElementList,
        None,
        Some(*len),
        BTreeMap::new(),
        None,
        None,
      ),
      TensorStorage::External { entries } => (
        TensorStorageKind::External,
        None,
        None,
        entries.clone(),
        None,
        None,
      ),
      TensorStorage::Sparse { values, indices } => (
        TensorStorageKind::Sparse,
        None,
        None,
        BTreeMap::new(),
        Some(values.index()),
        Some(indices.index()),
      ),
    };

  TensorMetadata {
    handle: EntityHandle::Tensor {
      tensor: tensor.id.index(),
    },
    name: tensor.name.map(|id| model.strings.get(id).to_owned()),
    element_type: element_type_name(&tensor.element_type),
    shape: tensor
      .shape
      .iter()
      .map(|dimension| dimension_label(model, &dimension.value))
      .collect(),
    storage,
    byte_len,
    element_count,
    external_data,
    sparse_values,
    sparse_indices,
    metadata: tensor.metadata.clone(),
  }
}

fn element_type_name(element_type: &TensorElementType) -> String {
  match element_type {
    TensorElementType::Other(value) => value.clone(),
    _ => format!("{element_type:?}").to_ascii_lowercase(),
  }
}

fn operator_domain(model: &Model, operator: &Operator) -> String {
  operator
    .domain
    .map(|domain| model.strings.get(domain).to_owned())
    .unwrap_or_else(|| "default".to_owned())
}

fn storage_kind_name(storage: &TensorStorage) -> &'static str {
  match storage {
    TensorStorage::Absent => "absent",
    TensorStorage::InlineBytes { .. } => "inline_bytes",
    TensorStorage::ElementList { .. } => "element_list",
    TensorStorage::External { .. } => "external",
    TensorStorage::Sparse { .. } => "sparse",
  }
}

fn dimension_label(model: &Model, value: &DimensionValue) -> String {
  match value {
    DimensionValue::Known(value) => value.to_string(),
    DimensionValue::Symbolic(value) => model.strings.get(*value).to_owned(),
    DimensionValue::Unknown => "?".to_owned(),
  }
}

fn initializer_count(model: &Model) -> usize {
  model
    .graphs
    .iter()
    .map(|graph| {
      graph
        .values
        .iter()
        .filter(|value| value.initializer.is_some())
        .count()
    })
    .sum::<usize>()
    + model
      .functions
      .iter()
      .map(|function| {
        function
          .values
          .iter()
          .filter(|value| value.initializer.is_some())
          .count()
      })
      .sum::<usize>()
}

fn subgraph_count(model: &Model) -> usize {
  model.graphs.iter().map(|graph| graph.subgraphs.len()).sum()
}

fn sparse_tensor_count(model: &Model) -> usize {
  model
    .tensors
    .iter()
    .filter(|tensor| matches!(tensor.storage, TensorStorage::Sparse { .. }))
    .count()
}

fn metadata_count(model: &Model) -> usize {
  model.metadata.properties.len()
    + model
      .graphs
      .iter()
      .map(|graph| {
        graph.metadata.len()
          + graph
            .nodes
            .iter()
            .map(|node| node.metadata.len())
            .sum::<usize>()
          + graph
            .values
            .iter()
            .map(|value| value.metadata.len())
            .sum::<usize>()
      })
      .sum::<usize>()
    + model
      .functions
      .iter()
      .map(|function| {
        function.metadata.len()
          + function
            .nodes
            .iter()
            .map(|node| node.metadata.len())
            .sum::<usize>()
      })
      .sum::<usize>()
    + model
      .tensors
      .iter()
      .map(|tensor| tensor.metadata.len())
      .sum::<usize>()
}

fn opset_count(model: &Model) -> usize {
  model.metadata.opsets.len()
    + model
      .functions
      .iter()
      .map(|function| function.opsets.len())
      .sum::<usize>()
}

fn onnx_detail(
  model: &Model,
  index: &OnnxIndex,
  handle: &EntityHandle,
  limit: usize,
) -> Option<EntityDetail> {
  match handle {
    EntityHandle::Graph { graph } => graph_detail(model, *graph, limit),
    EntityHandle::Node { graph, node } => node_detail(model, *graph, *node, limit),
    EntityHandle::Value { graph, value } => value_detail(model, *graph, *value, limit),
    EntityHandle::Tensor { tensor } => tensor_detail(index, *tensor, limit),
    EntityHandle::Function { function } => function_detail(model, *function),
    EntityHandle::Metadata { owner, key } => metadata_detail(model, owner, key),
    EntityHandle::OperatorSet { domain, version } => {
      opset_detail(model, domain.as_deref(), *version)
    }
    EntityHandle::Diagnostic { diagnostic } => diagnostic_detail(&index.diagnostics, *diagnostic),
    EntityHandle::MlirModule { .. }
    | EntityHandle::MlirFunction { .. }
    | EntityHandle::MlirOperation { .. }
    | EntityHandle::MlirValue { .. }
    | EntityHandle::MlirRegion { .. }
    | EntityHandle::MlirBlock { .. }
    | EntityHandle::MlirSymbol { .. }
    | EntityHandle::MlirDialect { .. }
    | EntityHandle::MlirAttribute { .. }
    | EntityHandle::MlirResource { .. } => None,
  }
}

fn mlir_detail(
  model: &Model,
  index: &MlirIndex,
  handle: &EntityHandle,
  limit: usize,
) -> Option<EntityDetail> {
  match handle {
    EntityHandle::MlirModule { module } => mlir_module_detail(model, *module, limit),
    EntityHandle::MlirFunction { function } => mlir_function_detail(model, *function, limit),
    EntityHandle::MlirOperation { scope, operation } => {
      mlir_operation_detail(model, scope, *operation, limit)
    }
    EntityHandle::MlirValue { scope, value } => mlir_value_detail(model, scope, *value, limit),
    EntityHandle::MlirRegion { scope, region } => mlir_region_detail(index, scope, *region, limit),
    EntityHandle::MlirBlock { scope, block } => mlir_block_detail(index, scope, *block, limit),
    EntityHandle::MlirSymbol { symbol } => mlir_symbol_detail(index, *symbol),
    EntityHandle::MlirDialect { dialect } => mlir_dialect_detail(model, index, dialect, limit),
    EntityHandle::MlirAttribute { scope, attribute } => {
      mlir_attribute_detail(index, scope, *attribute)
    }
    EntityHandle::MlirResource { resource } => mlir_resource_detail(index, *resource),
    EntityHandle::Diagnostic { diagnostic } => diagnostic_detail(&index.diagnostics, *diagnostic),
    EntityHandle::Graph { .. }
    | EntityHandle::Node { .. }
    | EntityHandle::Value { .. }
    | EntityHandle::Tensor { .. }
    | EntityHandle::Function { .. }
    | EntityHandle::Metadata { .. }
    | EntityHandle::OperatorSet { .. } => None,
  }
}

fn mlir_module_detail(model: &Model, module_index: usize, limit: usize) -> Option<EntityDetail> {
  let graph = model.graphs.get(module_index)?;
  let scope = mlir_module_scope(module_index);
  let mut fields = mlir_scope_fields(
    graph.nodes.len(),
    graph.values.len(),
    0,
    graph.metadata.len(),
  );
  fields.insert("scope".to_owned(), scope.clone());
  if let Some(name) = graph.name {
    fields.insert("name".to_owned(), model.strings.get(name).to_owned());
  }
  add_metadata_fields(&mut fields, &graph.metadata);

  let mut related = Vec::new();
  push_related(
    &mut related,
    limit,
    EntityHandle::MlirRegion {
      scope: scope.clone(),
      region: 0,
    },
  );
  push_related(
    &mut related,
    limit,
    EntityHandle::MlirBlock {
      scope: scope.clone(),
      block: 0,
    },
  );
  for node in &graph.nodes {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirOperation {
        scope: scope.clone(),
        operation: node.id.index(),
      },
    );
  }

  Some(EntityDetail {
    handle: EntityHandle::MlirModule {
      module: module_index,
    },
    title: graph
      .name
      .map(|id| model.strings.get(id).to_owned())
      .unwrap_or_else(|| format!("module {module_index}")),
    fields,
    related,
  })
}

fn mlir_function_detail(
  model: &Model,
  function_index: usize,
  limit: usize,
) -> Option<EntityDetail> {
  let function = model.functions.get(function_index)?;
  let scope = mlir_function_scope(function_index);
  let mut fields = mlir_scope_fields(
    function.nodes.len(),
    function.values.len(),
    mlir_function_block_argument_count(model, function),
    function.metadata.len() + function.attributes.len(),
  );
  fields.insert("scope".to_owned(), scope.clone());
  fields.insert(
    "name".to_owned(),
    model.strings.get(function.name).to_owned(),
  );
  fields.insert("input_count".to_owned(), function.inputs.len().to_string());
  fields.insert(
    "output_count".to_owned(),
    function.outputs.len().to_string(),
  );
  if let Some(domain) = function.domain {
    fields.insert("domain".to_owned(), model.strings.get(domain).to_owned());
  }
  add_metadata_fields(&mut fields, &function.metadata);

  let mut related = Vec::new();
  push_related(
    &mut related,
    limit,
    EntityHandle::MlirRegion {
      scope: scope.clone(),
      region: 0,
    },
  );
  push_related(
    &mut related,
    limit,
    EntityHandle::MlirBlock {
      scope: scope.clone(),
      block: 0,
    },
  );
  for (operation, _) in function.nodes.iter().enumerate() {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirOperation {
        scope: scope.clone(),
        operation,
      },
    );
  }
  for (value, _) in function.values.iter().enumerate() {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirValue {
        scope: scope.clone(),
        value,
      },
    );
  }

  Some(EntityDetail {
    handle: EntityHandle::MlirFunction {
      function: function_index,
    },
    title: model.strings.get(function.name).to_owned(),
    fields,
    related,
  })
}

fn mlir_operation_detail(
  model: &Model,
  scope: &str,
  operation_index: usize,
  limit: usize,
) -> Option<EntityDetail> {
  if let Some(module) = mlir_scope_index(scope, "module:") {
    let graph = model.graphs.get(module)?;
    let node = graph.nodes.get(operation_index)?;
    return Some(mlir_graph_operation_detail(
      model,
      scope,
      operation_index,
      node,
      limit,
    ));
  }
  let function_index = mlir_scope_index(scope, "function:")?;
  let function = model.functions.get(function_index)?;
  let node = function.nodes.get(operation_index)?;
  Some(mlir_function_operation_detail(
    model,
    scope,
    operation_index,
    function,
    node,
    limit,
  ))
}

fn mlir_graph_operation_detail(
  model: &Model,
  scope: &str,
  operation_index: usize,
  node: &Node,
  limit: usize,
) -> EntityDetail {
  let mut fields = mlir_operation_fields(
    model,
    &node.operator,
    &node.metadata,
    node.attributes.len(),
    node.inputs.iter().flatten().count(),
    node.outputs.iter().flatten().count(),
  );
  if let Some(name) = node.name {
    fields.insert("name".to_owned(), model.strings.get(name).to_owned());
  }
  let mut related = Vec::new();
  for value in node.inputs.iter().chain(&node.outputs).flatten() {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirValue {
        scope: scope.to_owned(),
        value: value.index(),
      },
    );
  }
  EntityDetail {
    handle: EntityHandle::MlirOperation {
      scope: scope.to_owned(),
      operation: operation_index,
    },
    title: mlir_node_title(model, node.name, &node.operator),
    fields,
    related,
  }
}

fn mlir_function_operation_detail(
  model: &Model,
  scope: &str,
  operation_index: usize,
  function: &Function,
  node: &FunctionNode,
  limit: usize,
) -> EntityDetail {
  let mut fields = mlir_operation_fields(
    model,
    &node.operator,
    &node.metadata,
    node.attributes.len(),
    node.inputs.iter().flatten().count(),
    node.outputs.iter().flatten().count(),
  );
  if let Some(name) = node.name {
    fields.insert("name".to_owned(), model.strings.get(name).to_owned());
  }
  let mut related = Vec::new();
  for value in node.inputs.iter().chain(&node.outputs).flatten() {
    if let Some(value) = mlir_function_value_index(function, *value) {
      push_related(
        &mut related,
        limit,
        EntityHandle::MlirValue {
          scope: scope.to_owned(),
          value,
        },
      );
    }
  }
  EntityDetail {
    handle: EntityHandle::MlirOperation {
      scope: scope.to_owned(),
      operation: operation_index,
    },
    title: mlir_node_title(model, node.name, &node.operator),
    fields,
    related,
  }
}

fn mlir_value_detail(
  model: &Model,
  scope: &str,
  value_index: usize,
  limit: usize,
) -> Option<EntityDetail> {
  if let Some(module) = mlir_scope_index(scope, "module:") {
    let graph = model.graphs.get(module)?;
    let value = graph.values.get(value_index)?;
    return Some(mlir_graph_value_detail(
      model,
      scope,
      value_index,
      value,
      limit,
    ));
  }
  let function_index = mlir_scope_index(scope, "function:")?;
  let function = model.functions.get(function_index)?;
  let value = function.values.get(value_index)?;
  Some(mlir_function_value_detail(
    model,
    scope,
    value_index,
    function,
    value,
    limit,
  ))
}

fn mlir_graph_value_detail(
  model: &Model,
  scope: &str,
  value_index: usize,
  value: &Value,
  limit: usize,
) -> EntityDetail {
  let mut fields = mlir_value_fields(model, model.strings.get(value.name), &value.type_info);
  fields.insert(
    "consumer_count".to_owned(),
    value.consumers.len().to_string(),
  );
  fields.insert(
    "is_graph_input".to_owned(),
    value.is_graph_input.to_string(),
  );
  fields.insert(
    "is_graph_output".to_owned(),
    value.is_graph_output.to_string(),
  );
  add_metadata_fields(&mut fields, &value.metadata);
  let mut related = Vec::new();
  if let Some(producer) = value.producer {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirOperation {
        scope: scope.to_owned(),
        operation: producer.index(),
      },
    );
  }
  for consumer in &value.consumers {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirOperation {
        scope: scope.to_owned(),
        operation: consumer.index(),
      },
    );
  }
  if let Some(tensor) = value.initializer {
    fields.insert("initializer".to_owned(), tensor.index().to_string());
    push_related(
      &mut related,
      limit,
      EntityHandle::Tensor {
        tensor: tensor.index(),
      },
    );
  }
  EntityDetail {
    handle: EntityHandle::MlirValue {
      scope: scope.to_owned(),
      value: value_index,
    },
    title: model.strings.get(value.name).to_owned(),
    fields,
    related,
  }
}

fn mlir_function_value_detail(
  model: &Model,
  scope: &str,
  value_index: usize,
  function: &Function,
  value: &FunctionValue,
  limit: usize,
) -> EntityDetail {
  let name = model.strings.get(value.name);
  let mut fields = mlir_value_fields(model, name, &value.type_info);
  fields.insert(
    "is_block_argument".to_owned(),
    mlir_function_block_argument_names(model, function)
      .contains(name)
      .to_string(),
  );
  let mut related = Vec::new();
  for (operation, node) in function.nodes.iter().enumerate() {
    if node
      .inputs
      .iter()
      .chain(&node.outputs)
      .flatten()
      .any(|id| *id == value.name)
    {
      push_related(
        &mut related,
        limit,
        EntityHandle::MlirOperation {
          scope: scope.to_owned(),
          operation,
        },
      );
    }
  }
  if let Some(tensor) = value.initializer {
    fields.insert("initializer".to_owned(), tensor.index().to_string());
    push_related(
      &mut related,
      limit,
      EntityHandle::Tensor {
        tensor: tensor.index(),
      },
    );
  }
  EntityDetail {
    handle: EntityHandle::MlirValue {
      scope: scope.to_owned(),
      value: value_index,
    },
    title: name.to_owned(),
    fields,
    related,
  }
}

fn mlir_region_detail(
  index: &MlirIndex,
  scope: &str,
  region: usize,
  limit: usize,
) -> Option<EntityDetail> {
  let handle = EntityHandle::MlirRegion {
    scope: scope.to_owned(),
    region,
  };
  let summary = index
    .summary
    .regions
    .iter()
    .find(|summary| summary.handle == handle)?;
  let mut fields = BTreeMap::new();
  fields.insert("scope".to_owned(), summary.scope_id.clone());
  fields.insert("region".to_owned(), region.to_string());
  fields.insert("block_count".to_owned(), summary.block_count.to_string());
  let mut related = Vec::new();
  for block in index
    .summary
    .blocks
    .iter()
    .filter(|block| block.scope_id == summary.scope_id)
  {
    push_related(&mut related, limit, block.handle.clone());
  }
  Some(EntityDetail {
    handle,
    title: format!("{scope} region {region}"),
    fields,
    related,
  })
}

fn mlir_block_detail(
  index: &MlirIndex,
  scope: &str,
  block: usize,
  limit: usize,
) -> Option<EntityDetail> {
  let handle = EntityHandle::MlirBlock {
    scope: scope.to_owned(),
    block,
  };
  let summary = index
    .summary
    .blocks
    .iter()
    .find(|summary| summary.handle == handle)?;
  let mut fields = BTreeMap::new();
  fields.insert("scope".to_owned(), summary.scope_id.clone());
  fields.insert("block".to_owned(), block.to_string());
  fields.insert(
    "operation_count".to_owned(),
    summary.operation_count.to_string(),
  );
  fields.insert("value_count".to_owned(), summary.value_count.to_string());
  fields.insert(
    "block_argument_count".to_owned(),
    summary.block_argument_count.to_string(),
  );
  let mut related = Vec::new();
  for region in index
    .summary
    .regions
    .iter()
    .filter(|region| region.scope_id == summary.scope_id)
  {
    push_related(&mut related, limit, region.handle.clone());
  }
  Some(EntityDetail {
    handle,
    title: format!("{scope} block {block}"),
    fields,
    related,
  })
}

fn mlir_symbol_detail(index: &MlirIndex, symbol: usize) -> Option<EntityDetail> {
  let name = index.symbols.get(symbol)?;
  let mut fields = BTreeMap::new();
  fields.insert("name".to_owned(), name.clone());
  fields.insert("index".to_owned(), symbol.to_string());
  fields.insert(
    "symbol_count".to_owned(),
    index.summary.symbol_count.to_string(),
  );

  Some(EntityDetail {
    handle: EntityHandle::MlirSymbol { symbol },
    title: name.clone(),
    fields,
    related: Vec::new(),
  })
}

fn mlir_dialect_detail(
  model: &Model,
  index: &MlirIndex,
  dialect: &str,
  limit: usize,
) -> Option<EntityDetail> {
  let mut fields = BTreeMap::new();
  fields.insert("name".to_owned(), dialect.to_owned());

  let mut related = Vec::new();
  let mut operation_count = 0usize;
  for (module, graph) in model.graphs.iter().enumerate() {
    let scope = mlir_module_scope(module);
    for (operation, node) in graph.nodes.iter().enumerate() {
      if mlir_dialect(model.strings.get(node.operator.name)) == dialect {
        operation_count += 1;
        push_related(
          &mut related,
          limit,
          EntityHandle::MlirOperation {
            scope: scope.clone(),
            operation,
          },
        );
      }
    }
  }
  for (function_index, function) in model.functions.iter().enumerate() {
    let scope = mlir_function_scope(function_index);
    for (operation, node) in function.nodes.iter().enumerate() {
      if mlir_dialect(model.strings.get(node.operator.name)) == dialect {
        operation_count += 1;
        push_related(
          &mut related,
          limit,
          EntityHandle::MlirOperation {
            scope: scope.clone(),
            operation,
          },
        );
      }
    }
  }
  if operation_count == 0 && !index.summary.dialects.iter().any(|value| value == dialect) {
    return None;
  }
  fields.insert("operation_count".to_owned(), operation_count.to_string());

  Some(EntityDetail {
    handle: EntityHandle::MlirDialect {
      dialect: dialect.to_owned(),
    },
    title: dialect.to_owned(),
    fields,
    related,
  })
}

fn mlir_attribute_detail(index: &MlirIndex, scope: &str, attribute: usize) -> Option<EntityDetail> {
  let detail = index.attributes.get(attribute)?;
  if detail.scope != scope {
    return None;
  }
  let mut fields = BTreeMap::new();
  fields.insert("name".to_owned(), detail.name.clone());
  fields.insert("scope".to_owned(), detail.scope.clone());
  fields.insert("index".to_owned(), attribute.to_string());
  fields.insert("value_count".to_owned(), detail.values.len().to_string());
  for (index, value) in detail.values.iter().enumerate() {
    fields.insert(format!("value_{index}"), value.clone());
  }

  let mut related = Vec::new();
  if let Some(module) = mlir_scope_index(&detail.scope, "module:") {
    push_related(&mut related, 1, EntityHandle::MlirModule { module });
  }
  if let Some(function) = mlir_scope_index(&detail.scope, "function:") {
    push_related(&mut related, 1, EntityHandle::MlirFunction { function });
  }

  Some(EntityDetail {
    handle: EntityHandle::MlirAttribute {
      scope: detail.scope.clone(),
      attribute,
    },
    title: format!("{}::{}", detail.scope, detail.name),
    fields,
    related,
  })
}

fn mlir_resource_detail(index: &MlirIndex, resource: usize) -> Option<EntityDetail> {
  let detail = index.resources.get(resource)?;
  let mut fields = BTreeMap::new();
  fields.insert("name".to_owned(), detail.name.clone());
  fields.insert("kind".to_owned(), detail.kind.clone());
  fields.insert("scope".to_owned(), detail.scope.clone());
  fields.insert("index".to_owned(), resource.to_string());

  Some(EntityDetail {
    handle: EntityHandle::MlirResource { resource },
    title: detail.name.clone(),
    fields,
    related: Vec::new(),
  })
}

fn mlir_scope_fields(
  operation_count: usize,
  value_count: usize,
  block_argument_count: usize,
  attribute_count: usize,
) -> BTreeMap<String, String> {
  BTreeMap::from([
    ("operation_count".to_owned(), operation_count.to_string()),
    ("value_count".to_owned(), value_count.to_string()),
    (
      "block_argument_count".to_owned(),
      block_argument_count.to_string(),
    ),
    ("attribute_count".to_owned(), attribute_count.to_string()),
    ("region_count".to_owned(), "1".to_owned()),
    ("block_count".to_owned(), "1".to_owned()),
  ])
}

fn mlir_operation_fields(
  model: &Model,
  operator: &Operator,
  metadata: &BTreeMap<String, String>,
  attribute_count: usize,
  input_count: usize,
  output_count: usize,
) -> BTreeMap<String, String> {
  let operator_name = model.strings.get(operator.name).to_owned();
  let mut fields = BTreeMap::new();
  fields.insert("operator".to_owned(), operator_name.clone());
  fields.insert("dialect".to_owned(), mlir_dialect(&operator_name));
  fields.insert("origin".to_owned(), operator.origin.to_owned());
  fields.insert("input_count".to_owned(), input_count.to_string());
  fields.insert("output_count".to_owned(), output_count.to_string());
  fields.insert("attribute_count".to_owned(), attribute_count.to_string());
  add_metadata_fields(&mut fields, metadata);
  fields
}

fn mlir_value_fields(
  model: &Model,
  name: &str,
  type_info: &Option<TypeInfo>,
) -> BTreeMap<String, String> {
  let mut fields = BTreeMap::from([("name".to_owned(), name.to_owned())]);
  if let Some(type_info) = type_info {
    add_type_fields(
      model,
      &mut fields,
      type_info.element_type.as_ref(),
      &type_info.shape,
    );
  }
  fields
}

fn mlir_node_title(
  model: &Model,
  name: Option<netron_rs_core::StringId>,
  operator: &Operator,
) -> String {
  name
    .map(|id| model.strings.get(id).to_owned())
    .unwrap_or_else(|| model.strings.get(operator.name).to_owned())
}

fn mlir_function_value_index(function: &Function, name: netron_rs_core::StringId) -> Option<usize> {
  function.values.iter().position(|value| value.name == name)
}

fn mlir_scope_index(scope: &str, prefix: &str) -> Option<usize> {
  scope.strip_prefix(prefix)?.parse().ok()
}

fn graph_detail(model: &Model, graph_index: usize, limit: usize) -> Option<EntityDetail> {
  let graph = model.graphs.get(graph_index)?;
  let mut fields = BTreeMap::new();
  fields.insert("node_count".to_owned(), graph.nodes.len().to_string());
  fields.insert("value_count".to_owned(), graph.values.len().to_string());
  fields.insert("input_count".to_owned(), graph.inputs.len().to_string());
  fields.insert("output_count".to_owned(), graph.outputs.len().to_string());
  fields.insert(
    "subgraph_count".to_owned(),
    graph.subgraphs.len().to_string(),
  );
  let tensor_count = graph
    .values
    .iter()
    .filter(|value| value.initializer.is_some())
    .count();
  fields.insert("tensor_count".to_owned(), tensor_count.to_string());
  fields.insert("initializer_count".to_owned(), tensor_count.to_string());
  add_metadata_fields(&mut fields, &graph.metadata);

  let mut related = Vec::new();
  for node in &graph.nodes {
    push_related(
      &mut related,
      limit,
      EntityHandle::Node {
        graph: graph_index,
        node: node.id.index(),
      },
    );
  }
  for subgraph in &graph.subgraphs {
    push_related(
      &mut related,
      limit,
      EntityHandle::Graph {
        graph: subgraph.index(),
      },
    );
  }

  Some(EntityDetail {
    handle: EntityHandle::Graph { graph: graph_index },
    title: graph
      .name
      .map(|id| model.strings.get(id).to_owned())
      .unwrap_or_else(|| format!("graph {graph_index}")),
    fields,
    related,
  })
}

fn node_detail(
  model: &Model,
  graph_index: usize,
  node_index: usize,
  limit: usize,
) -> Option<EntityDetail> {
  let graph = model.graphs.get(graph_index)?;
  let node = graph.nodes.get(node_index)?;
  let mut fields = BTreeMap::new();
  if let Some(name) = node.name {
    fields.insert("name".to_owned(), model.strings.get(name).to_owned());
  }
  fields.insert(
    "operator".to_owned(),
    model.strings.get(node.operator.name).to_owned(),
  );
  fields.insert("domain".to_owned(), operator_domain(model, &node.operator));
  fields.insert("origin".to_owned(), node.operator.origin.to_owned());
  if let Some(version) = node.operator.version {
    fields.insert("version".to_owned(), version.to_string());
  }
  fields.insert(
    "input_count".to_owned(),
    node.inputs.iter().flatten().count().to_string(),
  );
  fields.insert(
    "output_count".to_owned(),
    node.outputs.iter().flatten().count().to_string(),
  );
  fields.insert(
    "attribute_count".to_owned(),
    node.attributes.len().to_string(),
  );
  add_metadata_fields(&mut fields, &node.metadata);

  let mut related = Vec::new();
  for value in node.inputs.iter().chain(&node.outputs).flatten() {
    push_related(
      &mut related,
      limit,
      EntityHandle::Value {
        graph: graph_index,
        value: value.index(),
      },
    );
  }

  Some(EntityDetail {
    handle: EntityHandle::Node {
      graph: graph_index,
      node: node_index,
    },
    title: node
      .name
      .map(|id| model.strings.get(id).to_owned())
      .unwrap_or_else(|| model.strings.get(node.operator.name).to_owned()),
    fields,
    related,
  })
}

fn value_detail(
  model: &Model,
  graph_index: usize,
  value_index: usize,
  limit: usize,
) -> Option<EntityDetail> {
  let graph = model.graphs.get(graph_index)?;
  let value = graph.values.get(value_index)?;
  let mut fields = BTreeMap::new();
  fields.insert("name".to_owned(), model.strings.get(value.name).to_owned());
  fields.insert(
    "consumer_count".to_owned(),
    value.consumers.len().to_string(),
  );
  fields.insert(
    "is_graph_input".to_owned(),
    value.is_graph_input.to_string(),
  );
  fields.insert(
    "is_graph_output".to_owned(),
    value.is_graph_output.to_string(),
  );
  if let Some(type_info) = &value.type_info {
    add_type_fields(
      model,
      &mut fields,
      type_info.element_type.as_ref(),
      &type_info.shape,
    );
  }
  add_metadata_fields(&mut fields, &value.metadata);

  let mut related = Vec::new();
  if let Some(producer) = value.producer {
    push_related(
      &mut related,
      limit,
      EntityHandle::Node {
        graph: graph_index,
        node: producer.index(),
      },
    );
  }
  for consumer in &value.consumers {
    push_related(
      &mut related,
      limit,
      EntityHandle::Node {
        graph: graph_index,
        node: consumer.index(),
      },
    );
  }
  if let Some(tensor) = value.initializer {
    fields.insert("initializer".to_owned(), tensor.index().to_string());
    push_related(
      &mut related,
      limit,
      EntityHandle::Tensor {
        tensor: tensor.index(),
      },
    );
  }

  Some(EntityDetail {
    handle: EntityHandle::Value {
      graph: graph_index,
      value: value_index,
    },
    title: model.strings.get(value.name).to_owned(),
    fields,
    related,
  })
}

fn tensor_detail(index: &OnnxIndex, tensor_index: usize, limit: usize) -> Option<EntityDetail> {
  let tensor = index.tensor_metadata.get(tensor_index)?;
  let mut fields = BTreeMap::new();
  if let Some(name) = &tensor.name {
    fields.insert("name".to_owned(), name.clone());
  }
  fields.insert("element_type".to_owned(), tensor.element_type.clone());
  fields.insert("rank".to_owned(), tensor.shape.len().to_string());
  fields.insert("shape".to_owned(), tensor.shape.join(","));
  fields.insert(
    "storage".to_owned(),
    tensor_storage_kind_name(tensor.storage).to_owned(),
  );
  if let Some(byte_len) = tensor.byte_len {
    fields.insert("byte_len".to_owned(), byte_len.to_string());
  }
  if let Some(element_count) = tensor.element_count {
    fields.insert("element_count".to_owned(), element_count.to_string());
  }
  for (key, value) in &tensor.external_data {
    fields.insert(format!("external.{key}"), value.clone());
  }
  add_metadata_fields(&mut fields, &tensor.metadata);

  let mut related = Vec::new();
  if let Some(values) = tensor.sparse_values {
    push_related(&mut related, limit, EntityHandle::Tensor { tensor: values });
  }
  if let Some(indices) = tensor.sparse_indices {
    push_related(
      &mut related,
      limit,
      EntityHandle::Tensor { tensor: indices },
    );
  }

  Some(EntityDetail {
    handle: tensor.handle.clone(),
    title: tensor
      .name
      .clone()
      .unwrap_or_else(|| format!("tensor {tensor_index}")),
    fields,
    related,
  })
}

fn function_detail(model: &Model, function_index: usize) -> Option<EntityDetail> {
  let function = model.functions.get(function_index)?;
  let mut fields = BTreeMap::new();
  fields.insert(
    "name".to_owned(),
    model.strings.get(function.name).to_owned(),
  );
  if let Some(domain) = function.domain {
    fields.insert("domain".to_owned(), model.strings.get(domain).to_owned());
  }
  if let Some(overload) = function.overload {
    fields.insert(
      "overload".to_owned(),
      model.strings.get(overload).to_owned(),
    );
  }
  fields.insert("input_count".to_owned(), function.inputs.len().to_string());
  fields.insert(
    "output_count".to_owned(),
    function.outputs.len().to_string(),
  );
  fields.insert(
    "attribute_count".to_owned(),
    function.attributes.len().to_string(),
  );
  fields.insert("value_count".to_owned(), function.values.len().to_string());
  fields.insert("node_count".to_owned(), function.nodes.len().to_string());
  fields.insert("opset_count".to_owned(), function.opsets.len().to_string());
  add_metadata_fields(&mut fields, &function.metadata);

  Some(EntityDetail {
    handle: EntityHandle::Function {
      function: function_index,
    },
    title: model.strings.get(function.name).to_owned(),
    fields,
    related: Vec::new(),
  })
}

fn metadata_detail(model: &Model, owner: &str, key: &str) -> Option<EntityDetail> {
  if owner != "model" {
    return None;
  }
  let value = model.metadata.properties.get(key)?;
  let mut fields = BTreeMap::new();
  fields.insert("owner".to_owned(), owner.to_owned());
  fields.insert("key".to_owned(), key.to_owned());
  fields.insert("value".to_owned(), value.clone());
  Some(EntityDetail {
    handle: EntityHandle::Metadata {
      owner: owner.to_owned(),
      key: key.to_owned(),
    },
    title: key.to_owned(),
    fields,
    related: Vec::new(),
  })
}

fn opset_detail(model: &Model, domain: Option<&str>, version: i64) -> Option<EntityDetail> {
  model
    .metadata
    .opsets
    .iter()
    .find(|opset| opset.version == version && opset.domain.as_deref() == domain)?;
  let mut fields = BTreeMap::new();
  fields.insert("domain".to_owned(), domain.unwrap_or("default").to_owned());
  fields.insert("version".to_owned(), version.to_string());
  Some(EntityDetail {
    handle: EntityHandle::OperatorSet {
      domain: domain.map(str::to_owned),
      version,
    },
    title: domain.unwrap_or("default").to_owned(),
    fields,
    related: Vec::new(),
  })
}

fn diagnostic_detail(diagnostics: &[Diagnostic], diagnostic_index: usize) -> Option<EntityDetail> {
  let diagnostic = diagnostics.get(diagnostic_index)?;
  let handle = EntityHandle::Diagnostic {
    diagnostic: diagnostic_index,
  };
  let mut fields = BTreeMap::new();
  fields.insert("code".to_owned(), diagnostic.code.to_owned());
  fields.insert(
    "kind".to_owned(),
    diagnostic_kind_name(diagnostic.kind).to_owned(),
  );
  fields.insert("message".to_owned(), diagnostic.message.clone());
  if let Some(source) = &diagnostic.source {
    fields.insert("source".to_owned(), source.clone());
  }
  Some(EntityDetail {
    handle,
    title: diagnostic.code.to_owned(),
    fields,
    related: Vec::new(),
  })
}

fn diagnostic_kind_name(kind: DiagnosticKind) -> &'static str {
  match kind {
    DiagnosticKind::Info => "info",
    DiagnosticKind::Warning => "warning",
    DiagnosticKind::Error => "error",
  }
}

fn session_slice(
  model: &Model,
  index: &FormatIndex,
  session_id: u64,
  scope: &EntityHandle,
  limits: SessionLimits,
) -> Option<SliceResponse> {
  let mut builder = SliceBuilder::new(limits.slice);
  match scope {
    EntityHandle::Graph { graph } => onnx_graph_slice(model, *graph, &mut builder)?,
    EntityHandle::Node { graph, node } => onnx_node_slice(model, *graph, *node, &mut builder)?,
    EntityHandle::Value { graph, value } => onnx_value_slice(model, *graph, *value, &mut builder)?,
    EntityHandle::Function { function } => onnx_function_slice(model, *function, &mut builder)?,
    EntityHandle::MlirFunction { function } => mlir_function_slice(model, *function, &mut builder)?,
    EntityHandle::MlirOperation { scope, operation } => {
      mlir_operation_slice(model, scope, *operation, &mut builder)?
    }
    EntityHandle::MlirRegion { scope, region } => {
      mlir_region_slice(model, mlir_index(index)?, scope, *region, &mut builder)?
    }
    EntityHandle::MlirBlock { scope, block } => {
      mlir_block_slice(model, mlir_index(index)?, scope, *block, &mut builder)?
    }
    EntityHandle::MlirSymbol { symbol } => {
      mlir_symbol_slice(model, mlir_index(index)?, *symbol, &mut builder)?
    }
    EntityHandle::MlirDialect { dialect } => mlir_dialect_slice(model, dialect, &mut builder)?,
    EntityHandle::MlirModule { module } => mlir_module_slice(model, *module, &mut builder)?,
    EntityHandle::Tensor { .. }
    | EntityHandle::Metadata { .. }
    | EntityHandle::OperatorSet { .. }
    | EntityHandle::Diagnostic { .. }
    | EntityHandle::MlirValue { .. }
    | EntityHandle::MlirAttribute { .. }
    | EntityHandle::MlirResource { .. } => return None,
  }
  let truncated = builder.omitted_count > 0;
  let mut warnings = Vec::new();
  if truncated {
    warnings.push(format!(
      "slice truncated to {} visible entities",
      limits.slice
    ));
  }
  Some(SliceResponse {
    api_version: SESSION_API_VERSION,
    session_id,
    format: index.kind(),
    scope: scope.clone(),
    cache_key: projection_cache_key(session_id, index.kind(), "slice", scope, limits.slice),
    limit_used: limits.slice,
    truncated,
    omitted_count: builder.omitted_count,
    warnings,
    entities: builder.entities,
    edges: builder.edges,
    boundaries: builder.boundaries,
  })
}

fn session_layout(
  model: &Model,
  index: &FormatIndex,
  session_id: u64,
  scope: &EntityHandle,
  limits: SessionLimits,
) -> Option<LayoutResponse> {
  let graph = match scope {
    EntityHandle::Graph { graph } => netron_rs_layout::layout_graph(
      model,
      &LayoutOptions {
        graph: *graph,
        max_nodes: Some(limits.layout),
        ..LayoutOptions::default()
      },
    )
    .ok()?,
    EntityHandle::MlirFunction { function } => {
      mlir_function_layout(model, *function, limits.layout)?
    }
    EntityHandle::MlirModule { module } => mlir_module_layout(model, *module, limits.layout)?,
    EntityHandle::MlirRegion { scope, .. } | EntityHandle::MlirBlock { scope, .. } => {
      if let Some(function) = mlir_scope_index(scope, "function:") {
        mlir_function_layout(model, function, limits.layout)?
      } else {
        mlir_module_layout(model, mlir_scope_index(scope, "module:")?, limits.layout)?
      }
    }
    _ => return None,
  };
  let omitted_count =
    graph.stats.omitted_nodes + graph.stats.omitted_edges + graph.stats.omitted_initializers;
  let truncated = omitted_count > 0;
  let mut warnings = Vec::new();
  if truncated {
    warnings.push(format!("layout truncated under limit {}", limits.layout));
  }
  Some(LayoutResponse {
    api_version: SESSION_API_VERSION,
    session_id,
    format: index.kind(),
    scope: scope.clone(),
    cache_key: projection_cache_key(session_id, index.kind(), "layout", scope, limits.layout),
    limit_used: limits.layout,
    truncated,
    omitted_count,
    warnings,
    graph,
  })
}

fn mlir_index(index: &FormatIndex) -> Option<&MlirIndex> {
  match index {
    FormatIndex::Mlir(index) => Some(index),
    _ => None,
  }
}

struct SliceBuilder {
  limit: usize,
  entities: Vec<SliceEntity>,
  edges: Vec<SliceEdge>,
  boundaries: Vec<SliceBoundary>,
  seen_entities: BTreeSet<String>,
  omitted_entities: BTreeSet<String>,
  seen_edges: BTreeSet<String>,
  omitted_edges: BTreeSet<String>,
  omitted_count: usize,
}

impl SliceBuilder {
  fn new(limit: usize) -> Self {
    Self {
      limit,
      entities: Vec::new(),
      edges: Vec::new(),
      boundaries: Vec::new(),
      seen_entities: BTreeSet::new(),
      omitted_entities: BTreeSet::new(),
      seen_edges: BTreeSet::new(),
      omitted_edges: BTreeSet::new(),
      omitted_count: 0,
    }
  }

  fn handle(
    &mut self,
    handle: EntityHandle,
    kind: &str,
    label: Option<String>,
    operator: Option<String>,
    boundary: bool,
  ) {
    self.entity(
      handle_key(&handle),
      kind,
      Some(handle),
      label,
      operator,
      boundary,
    );
  }

  fn entity(
    &mut self,
    id: String,
    kind: &str,
    handle: Option<EntityHandle>,
    label: Option<String>,
    operator: Option<String>,
    boundary: bool,
  ) {
    if self.seen_entities.contains(&id) {
      return;
    }
    if self.entities.len() >= self.limit {
      if self.omitted_entities.insert(id.clone()) {
        self.omitted_count += 1;
        self.boundary(id, "entity_limit", 1);
      }
      return;
    }
    self.seen_entities.insert(id.clone());
    self.entities.push(SliceEntity {
      id,
      kind: kind.to_owned(),
      handle,
      label,
      operator,
      boundary,
    });
  }

  fn edge(&mut self, from: String, to: String, value: Option<String>, label: Option<String>) {
    let id = format!("{from}->{to}:{}", value.as_deref().unwrap_or(""));
    if self.seen_edges.contains(&id) {
      return;
    }
    if !self.seen_entities.contains(&from) || !self.seen_entities.contains(&to) {
      if self.omitted_edges.insert(id) {
        self.omitted_count += 1;
        self.boundary(format!("{from}->{to}"), "edge_endpoint_limit", 1);
      }
      return;
    }
    if self.edges.len() >= self.limit {
      if self.omitted_edges.insert(id) {
        self.omitted_count += 1;
        self.boundary(format!("{from}->{to}"), "edge_limit", 1);
      }
      return;
    }
    self.seen_edges.insert(id);
    self.edges.push(SliceEdge {
      from,
      to,
      value,
      label,
    });
  }

  fn boundary(&mut self, id: String, reason: &str, omitted_count: usize) {
    if let Some(boundary) = self
      .boundaries
      .iter_mut()
      .find(|boundary| boundary.reason == reason)
    {
      boundary.omitted_count += omitted_count;
      return;
    }
    self.boundaries.push(SliceBoundary {
      id,
      reason: reason.to_owned(),
      omitted_count,
    });
  }

  fn omit_boundary(&mut self, id: String, reason: &str, omitted_count: usize) {
    if omitted_count == 0 {
      return;
    }
    self.omitted_count += omitted_count;
    self.boundary(id, reason, omitted_count);
  }
}

fn onnx_graph_slice(model: &Model, graph_index: usize, builder: &mut SliceBuilder) -> Option<()> {
  let graph = model.graphs.get(graph_index)?;
  builder.handle(
    EntityHandle::Graph { graph: graph_index },
    "graph",
    graph.name.map(|name| model.strings.get(name).to_owned()),
    None,
    true,
  );
  for value in graph.inputs.iter().chain(&graph.outputs) {
    let value = &graph.values[value.index()];
    push_onnx_value_entity(model, graph_index, value, builder);
  }
  push_onnx_boundary_path(model, graph_index, builder);
  for node in &graph.nodes {
    let node_id = handle_key(&EntityHandle::Node {
      graph: graph_index,
      node: node.id.index(),
    });
    if builder.entities.len() >= builder.limit && !builder.seen_entities.contains(&node_id) {
      builder.omit_boundary(
        handle_key(&EntityHandle::Graph { graph: graph_index }),
        "graph_node_limit",
        graph.nodes.len().saturating_sub(node.id.index()),
      );
      break;
    }
    push_onnx_node_entity(model, graph_index, node, builder);
  }
  push_onnx_edges(model, graph_index, builder);
  Some(())
}

fn push_onnx_boundary_path(model: &Model, graph_index: usize, builder: &mut SliceBuilder) {
  let Some(graph) = model.graphs.get(graph_index) else {
    return;
  };
  let mut queue = graph.outputs.iter().copied().collect::<VecDeque<_>>();
  let mut seen_values = BTreeSet::new();
  let mut seen_nodes = BTreeSet::new();
  let step_limit = builder.limit.saturating_mul(4).max(1);
  let mut steps = 0usize;
  while let Some(value_id) = queue.pop_front() {
    if steps >= step_limit {
      builder.omit_boundary(
        format!("graph:{graph_index}:boundary_path"),
        "path_limit",
        queue.len() + 1,
      );
      break;
    }
    steps += 1;
    if !seen_values.insert(value_id.index()) {
      continue;
    }
    let value = &graph.values[value_id.index()];
    push_onnx_value_entity(model, graph_index, value, builder);
    if let Some(producer) = value.producer
      && seen_nodes.insert(producer.index())
    {
      let node = &graph.nodes[producer.index()];
      push_onnx_node_entity(model, graph_index, node, builder);
      for input in node.inputs.iter().flatten() {
        queue.push_back(*input);
      }
    }
  }
}

fn onnx_node_slice(
  model: &Model,
  graph_index: usize,
  node_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let graph = model.graphs.get(graph_index)?;
  let node = graph.nodes.get(node_index)?;
  push_onnx_node_entity(model, graph_index, node, builder);
  for value in node.inputs.iter().chain(&node.outputs).flatten() {
    let value = &graph.values[value.index()];
    push_onnx_value_entity(model, graph_index, value, builder);
    if let Some(producer) = value.producer {
      push_onnx_node_entity(model, graph_index, &graph.nodes[producer.index()], builder);
    }
    for consumer in &value.consumers {
      push_onnx_node_entity(model, graph_index, &graph.nodes[consumer.index()], builder);
    }
  }
  push_onnx_edges(model, graph_index, builder);
  Some(())
}

fn onnx_value_slice(
  model: &Model,
  graph_index: usize,
  value_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let graph = model.graphs.get(graph_index)?;
  let value = graph.values.get(value_index)?;
  push_onnx_value_entity(model, graph_index, value, builder);
  if let Some(producer) = value.producer {
    push_onnx_node_entity(model, graph_index, &graph.nodes[producer.index()], builder);
  }
  for consumer in &value.consumers {
    push_onnx_node_entity(model, graph_index, &graph.nodes[consumer.index()], builder);
  }
  push_onnx_edges(model, graph_index, builder);
  Some(())
}

fn onnx_function_slice(
  model: &Model,
  function_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let function = model.functions.get(function_index)?;
  let function_id = format!("function:{function_index}");
  builder.handle(
    EntityHandle::Function {
      function: function_index,
    },
    "function",
    Some(model.strings.get(function.name).to_owned()),
    None,
    true,
  );
  for (index, node) in function.nodes.iter().enumerate() {
    builder.entity(
      format!("{function_id}:operation:{index}"),
      "function_operation",
      None,
      node.name.map(|name| model.strings.get(name).to_owned()),
      Some(model.strings.get(node.operator.name).to_owned()),
      false,
    );
    for input in node.inputs.iter().flatten() {
      let value_id = format!("{function_id}:value:{}", model.strings.get(*input));
      builder.entity(
        value_id.clone(),
        "function_value",
        None,
        Some(model.strings.get(*input).to_owned()),
        None,
        false,
      );
      builder.edge(
        value_id,
        format!("{function_id}:operation:{index}"),
        None,
        Some(model.strings.get(*input).to_owned()),
      );
    }
    for output in node.outputs.iter().flatten() {
      let value_id = format!("{function_id}:value:{}", model.strings.get(*output));
      builder.entity(
        value_id.clone(),
        "function_value",
        None,
        Some(model.strings.get(*output).to_owned()),
        None,
        false,
      );
      builder.edge(
        format!("{function_id}:operation:{index}"),
        value_id,
        None,
        Some(model.strings.get(*output).to_owned()),
      );
    }
  }
  Some(())
}

fn push_onnx_node_entity(
  model: &Model,
  graph_index: usize,
  node: &Node,
  builder: &mut SliceBuilder,
) {
  builder.handle(
    EntityHandle::Node {
      graph: graph_index,
      node: node.id.index(),
    },
    "node",
    node.name.map(|name| model.strings.get(name).to_owned()),
    Some(model.strings.get(node.operator.name).to_owned()),
    false,
  );
}

fn push_onnx_value_entity(
  model: &Model,
  graph_index: usize,
  value: &Value,
  builder: &mut SliceBuilder,
) {
  builder.handle(
    EntityHandle::Value {
      graph: graph_index,
      value: value.id.index(),
    },
    "value",
    Some(model.strings.get(value.name).to_owned()),
    None,
    value.is_graph_input || value.is_graph_output || value.initializer.is_some(),
  );
}

fn push_onnx_edges(model: &Model, graph_index: usize, builder: &mut SliceBuilder) {
  let Some(graph) = model.graphs.get(graph_index) else {
    return;
  };
  for value in &graph.values {
    let value_id = handle_key(&EntityHandle::Value {
      graph: graph_index,
      value: value.id.index(),
    });
    if !builder.seen_entities.contains(&value_id) {
      continue;
    }
    if let Some(producer) = value.producer {
      let node_id = handle_key(&EntityHandle::Node {
        graph: graph_index,
        node: producer.index(),
      });
      if builder.seen_entities.contains(&node_id) {
        builder.edge(
          node_id,
          value_id.clone(),
          None,
          Some(model.strings.get(value.name).to_owned()),
        );
      }
    }
    for consumer in &value.consumers {
      let node_id = handle_key(&EntityHandle::Node {
        graph: graph_index,
        node: consumer.index(),
      });
      if builder.seen_entities.contains(&node_id) {
        builder.edge(
          value_id.clone(),
          node_id,
          None,
          Some(model.strings.get(value.name).to_owned()),
        );
      }
    }
  }
}

fn mlir_module_slice(model: &Model, module: usize, builder: &mut SliceBuilder) -> Option<()> {
  let graph = model.graphs.get(module)?;
  let scope = mlir_module_scope(module);
  builder.handle(
    EntityHandle::MlirModule { module },
    "mlir_module",
    graph.name.map(|name| model.strings.get(name).to_owned()),
    None,
    true,
  );
  builder.handle(
    EntityHandle::MlirRegion {
      scope: scope.clone(),
      region: 0,
    },
    "mlir_region",
    Some(format!("{scope} region 0")),
    None,
    true,
  );
  builder.handle(
    EntityHandle::MlirBlock {
      scope: scope.clone(),
      block: 0,
    },
    "mlir_block",
    Some(format!("{scope} block 0")),
    None,
    true,
  );
  let module_id = handle_key(&EntityHandle::MlirModule { module });
  for (function, item) in model.functions.iter().enumerate() {
    if mlir_function_belongs_to_module(model, item, graph) {
      let function_handle = EntityHandle::MlirFunction { function };
      let function_id = handle_key(&function_handle);
      builder.handle(
        function_handle,
        "mlir_function",
        Some(model.strings.get(item.name).to_owned()),
        None,
        true,
      );
      builder.edge(
        module_id.clone(),
        function_id,
        None,
        Some("contains".to_owned()),
      );
    }
  }
  for node in &graph.nodes {
    builder.handle(
      EntityHandle::MlirOperation {
        scope: scope.clone(),
        operation: node.id.index(),
      },
      "mlir_operation",
      node.name.map(|name| model.strings.get(name).to_owned()),
      Some(model.strings.get(node.operator.name).to_owned()),
      false,
    );
  }
  Some(())
}

fn mlir_function_slice(
  model: &Model,
  function_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let function = model.functions.get(function_index)?;
  let scope = mlir_function_scope(function_index);
  builder.handle(
    EntityHandle::MlirFunction {
      function: function_index,
    },
    "mlir_function",
    Some(model.strings.get(function.name).to_owned()),
    None,
    true,
  );
  builder.handle(
    EntityHandle::MlirRegion {
      scope: scope.clone(),
      region: 0,
    },
    "mlir_region",
    Some(format!("{scope} region 0")),
    None,
    true,
  );
  builder.handle(
    EntityHandle::MlirBlock {
      scope: scope.clone(),
      block: 0,
    },
    "mlir_block",
    Some(format!("{scope} block 0")),
    None,
    true,
  );
  let block_args = mlir_function_block_argument_names(model, function);
  for (value, item) in function.values.iter().enumerate() {
    let name = model.strings.get(item.name);
    builder.handle(
      EntityHandle::MlirValue {
        scope: scope.clone(),
        value,
      },
      "mlir_value",
      Some(name.to_owned()),
      None,
      block_args.contains(name)
        || function
          .outputs
          .iter()
          .any(|id| model.strings.get(*id) == name),
    );
  }
  for (operation, node) in function.nodes.iter().enumerate() {
    builder.handle(
      EntityHandle::MlirOperation {
        scope: scope.clone(),
        operation,
      },
      "mlir_operation",
      node.name.map(|name| model.strings.get(name).to_owned()),
      Some(model.strings.get(node.operator.name).to_owned()),
      false,
    );
    for input in node.inputs.iter().flatten() {
      if let Some(value) = mlir_function_value_index(function, *input) {
        builder.edge(
          handle_key(&EntityHandle::MlirValue {
            scope: scope.clone(),
            value,
          }),
          handle_key(&EntityHandle::MlirOperation {
            scope: scope.clone(),
            operation,
          }),
          None,
          Some(model.strings.get(*input).to_owned()),
        );
      }
    }
    for output in node.outputs.iter().flatten() {
      if let Some(value) = mlir_function_value_index(function, *output) {
        builder.edge(
          handle_key(&EntityHandle::MlirOperation {
            scope: scope.clone(),
            operation,
          }),
          handle_key(&EntityHandle::MlirValue {
            scope: scope.clone(),
            value,
          }),
          None,
          Some(model.strings.get(*output).to_owned()),
        );
      }
    }
  }
  Some(())
}

fn mlir_operation_slice(
  model: &Model,
  scope: &str,
  operation: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  if let Some(function) = mlir_scope_index(scope, "function:") {
    let function = model.functions.get(function)?;
    let node = function.nodes.get(operation)?;
    builder.handle(
      EntityHandle::MlirOperation {
        scope: scope.to_owned(),
        operation,
      },
      "mlir_operation",
      node.name.map(|name| model.strings.get(name).to_owned()),
      Some(model.strings.get(node.operator.name).to_owned()),
      false,
    );
    for value in node.inputs.iter().chain(&node.outputs).flatten() {
      if let Some(value) = mlir_function_value_index(function, *value) {
        builder.handle(
          EntityHandle::MlirValue {
            scope: scope.to_owned(),
            value,
          },
          "mlir_value",
          Some(model.strings.get(function.values[value].name).to_owned()),
          None,
          false,
        );
      }
    }
    for input in node.inputs.iter().flatten() {
      if let Some(value) = mlir_function_value_index(function, *input) {
        builder.edge(
          handle_key(&EntityHandle::MlirValue {
            scope: scope.to_owned(),
            value,
          }),
          handle_key(&EntityHandle::MlirOperation {
            scope: scope.to_owned(),
            operation,
          }),
          None,
          Some(model.strings.get(*input).to_owned()),
        );
      }
    }
    for output in node.outputs.iter().flatten() {
      if let Some(value) = mlir_function_value_index(function, *output) {
        builder.edge(
          handle_key(&EntityHandle::MlirOperation {
            scope: scope.to_owned(),
            operation,
          }),
          handle_key(&EntityHandle::MlirValue {
            scope: scope.to_owned(),
            value,
          }),
          None,
          Some(model.strings.get(*output).to_owned()),
        );
      }
    }
    return Some(());
  }
  let module = mlir_scope_index(scope, "module:")?;
  let graph = model.graphs.get(module)?;
  let node = graph.nodes.get(operation)?;
  builder.handle(
    EntityHandle::MlirOperation {
      scope: scope.to_owned(),
      operation,
    },
    "mlir_operation",
    node.name.map(|name| model.strings.get(name).to_owned()),
    Some(model.strings.get(node.operator.name).to_owned()),
    false,
  );
  Some(())
}

fn mlir_region_slice(
  model: &Model,
  index: &MlirIndex,
  scope: &str,
  region: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let handle = EntityHandle::MlirRegion {
    scope: scope.to_owned(),
    region,
  };
  index
    .summary
    .regions
    .iter()
    .any(|item| item.scope_id == scope && item.handle == handle)
    .then_some(())?;
  builder.handle(
    handle,
    "mlir_region",
    Some(format!("{scope} region {region}")),
    None,
    true,
  );
  mlir_scope_slice(model, scope, builder)
}

fn mlir_block_slice(
  model: &Model,
  index: &MlirIndex,
  scope: &str,
  block: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let handle = EntityHandle::MlirBlock {
    scope: scope.to_owned(),
    block,
  };
  index
    .summary
    .blocks
    .iter()
    .any(|item| item.scope_id == scope && item.handle == handle)
    .then_some(())?;
  builder.handle(
    handle,
    "mlir_block",
    Some(format!("{scope} block {block}")),
    None,
    true,
  );
  mlir_scope_slice(model, scope, builder)
}

fn mlir_symbol_slice(
  model: &Model,
  index: &MlirIndex,
  symbol: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let symbol_name = index.symbols.get(symbol)?;
  let symbol_id = handle_key(&EntityHandle::MlirSymbol { symbol });
  builder.handle(
    EntityHandle::MlirSymbol { symbol },
    "mlir_symbol",
    Some(format!("@{symbol_name}")),
    None,
    true,
  );

  for (module, graph) in model.graphs.iter().enumerate() {
    let scope = mlir_module_scope(module);
    if graph
      .name
      .is_some_and(|name| mlir_symbol_matches(model.strings.get(name), symbol_name))
    {
      let module_handle = EntityHandle::MlirModule { module };
      let module_id = handle_key(&module_handle);
      builder.handle(
        module_handle,
        "mlir_module",
        graph.name.map(|name| model.strings.get(name).to_owned()),
        None,
        true,
      );
      builder.edge(
        symbol_id.clone(),
        module_id,
        None,
        Some("defines".to_owned()),
      );
    }
    for (operation, node) in graph.nodes.iter().enumerate() {
      if mlir_node_mentions_symbol(model, node, symbol_name) {
        let operation_handle = EntityHandle::MlirOperation {
          scope: scope.clone(),
          operation,
        };
        let operation_id = handle_key(&operation_handle);
        builder.handle(
          operation_handle,
          "mlir_operation",
          node.name.map(|name| model.strings.get(name).to_owned()),
          Some(model.strings.get(node.operator.name).to_owned()),
          false,
        );
        builder.edge(
          symbol_id.clone(),
          operation_id,
          None,
          Some("references".to_owned()),
        );
      }
    }
  }

  for (function, item) in model.functions.iter().enumerate() {
    let scope = mlir_function_scope(function);
    if mlir_symbol_matches(model.strings.get(item.name), symbol_name) {
      let function_handle = EntityHandle::MlirFunction { function };
      let function_id = handle_key(&function_handle);
      builder.handle(
        function_handle,
        "mlir_function",
        Some(model.strings.get(item.name).to_owned()),
        None,
        true,
      );
      builder.edge(
        symbol_id.clone(),
        function_id,
        None,
        Some("defines".to_owned()),
      );
    }
    for (operation, node) in item.nodes.iter().enumerate() {
      if mlir_function_node_mentions_symbol(model, node, symbol_name) {
        let operation_handle = EntityHandle::MlirOperation {
          scope: scope.clone(),
          operation,
        };
        let operation_id = handle_key(&operation_handle);
        builder.handle(
          operation_handle,
          "mlir_operation",
          node.name.map(|name| model.strings.get(name).to_owned()),
          Some(model.strings.get(node.operator.name).to_owned()),
          false,
        );
        builder.edge(
          symbol_id.clone(),
          operation_id,
          None,
          Some("references".to_owned()),
        );
      }
    }
  }
  Some(())
}

fn mlir_function_belongs_to_module(
  model: &Model,
  function: &Function,
  graph: &netron_rs_core::Graph,
) -> bool {
  let Some(module_name) = graph.name.map(|name| model.strings.get(name)) else {
    return model.graphs.len() == 1;
  };
  let module_name = module_name.trim_start_matches('@');
  let function_name = model.strings.get(function.name).trim_start_matches('@');
  function_name.starts_with(&format!("{module_name}::"))
    || (model.graphs.len() == 1 && !function_name.contains("::"))
}

fn mlir_node_mentions_symbol(model: &Model, node: &Node, symbol: &str) -> bool {
  node
    .name
    .is_some_and(|name| mlir_symbol_matches(model.strings.get(name), symbol))
    || node
      .attributes
      .iter()
      .any(|attribute| mlir_attribute_mentions_symbol(model, attribute, symbol))
}

fn mlir_function_node_mentions_symbol(model: &Model, node: &FunctionNode, symbol: &str) -> bool {
  node
    .name
    .is_some_and(|name| mlir_symbol_matches(model.strings.get(name), symbol))
    || node
      .attributes
      .iter()
      .any(|attribute| mlir_attribute_mentions_symbol(model, attribute, symbol))
}

fn mlir_attribute_mentions_symbol(model: &Model, attribute: &Attribute, symbol: &str) -> bool {
  mlir_symbol_matches(model.strings.get(attribute.name), symbol)
    || attribute_value_terms(model, &attribute.value)
      .iter()
      .any(|term| mlir_symbol_matches(term, symbol))
}

fn mlir_symbol_matches(value: &str, symbol: &str) -> bool {
  let value = value.trim_start_matches('@');
  value == symbol
    || value.ends_with(&format!("::{symbol}"))
    || value.ends_with(&format!("::@{symbol}"))
}

fn mlir_scope_slice(model: &Model, scope: &str, builder: &mut SliceBuilder) -> Option<()> {
  if let Some(function) = mlir_scope_index(scope, "function:") {
    return mlir_function_slice(model, function, builder);
  }
  mlir_module_slice(model, mlir_scope_index(scope, "module:")?, builder)
}

fn mlir_dialect_slice(model: &Model, dialect: &str, builder: &mut SliceBuilder) -> Option<()> {
  builder.handle(
    EntityHandle::MlirDialect {
      dialect: dialect.to_owned(),
    },
    "mlir_dialect",
    Some(dialect.to_owned()),
    None,
    true,
  );
  let mut matched = false;
  for (module, graph) in model.graphs.iter().enumerate() {
    let scope = mlir_module_scope(module);
    for (operation, node) in graph.nodes.iter().enumerate() {
      if mlir_dialect(model.strings.get(node.operator.name)) == dialect {
        matched = true;
        builder.handle(
          EntityHandle::MlirOperation {
            scope: scope.clone(),
            operation,
          },
          "mlir_operation",
          node.name.map(|name| model.strings.get(name).to_owned()),
          Some(model.strings.get(node.operator.name).to_owned()),
          false,
        );
      }
    }
  }
  for (function, item) in model.functions.iter().enumerate() {
    let scope = mlir_function_scope(function);
    for (operation, node) in item.nodes.iter().enumerate() {
      if mlir_dialect(model.strings.get(node.operator.name)) == dialect {
        matched = true;
        builder.handle(
          EntityHandle::MlirOperation {
            scope: scope.clone(),
            operation,
          },
          "mlir_operation",
          node.name.map(|name| model.strings.get(name).to_owned()),
          Some(model.strings.get(node.operator.name).to_owned()),
          false,
        );
      }
    }
  }
  matched.then_some(())
}

fn mlir_function_layout(model: &Model, function_index: usize, limit: usize) -> Option<LayoutGraph> {
  let function = model.functions.get(function_index)?;
  let scope = mlir_function_scope(function_index);
  let visible_ops = function.nodes.len().min(limit);
  let mut nodes = Vec::new();
  let mut value_nodes = BTreeMap::new();
  let mut op_nodes = Vec::new();
  let mut output_nodes = Vec::new();
  let block_args = mlir_function_block_argument_names(model, function);
  for (value, item) in function.values.iter().enumerate() {
    let name = model.strings.get(item.name);
    if block_args.contains(name) {
      value_nodes.insert(item.name, nodes.len());
      nodes.push(layout_node(MlirLayoutNodeSpec {
        id: format!("mlir_value:{scope}:{value}"),
        graph: function_index,
        kind: LayoutNodeKind::GraphInput,
        rank: 0,
        value: Some(value),
        name: Some(name.to_owned()),
        operator: None,
        order: 0,
        node: None,
      }));
    }
  }
  for (operation, node) in function.nodes.iter().take(visible_ops).enumerate() {
    op_nodes.push(nodes.len());
    nodes.push(layout_node(MlirLayoutNodeSpec {
      id: format!("mlir_operation:{scope}:{operation}"),
      graph: function_index,
      kind: LayoutNodeKind::Operator,
      rank: operation + 1,
      value: None,
      name: node.name.map(|name| model.strings.get(name).to_owned()),
      operator: Some(model.strings.get(node.operator.name).to_owned()),
      order: operation + 1,
      node: Some(operation),
    }));
    for output in node.outputs.iter().flatten() {
      value_nodes.insert(*output, op_nodes[operation]);
    }
  }
  for output in &function.outputs {
    let name = model.strings.get(*output);
    let node_index = nodes.len();
    output_nodes.push((*output, node_index));
    nodes.push(layout_node(MlirLayoutNodeSpec {
      id: format!("mlir_output:{scope}:{node_index}"),
      graph: function_index,
      kind: LayoutNodeKind::GraphOutput,
      rank: visible_ops + 1,
      value: None,
      name: Some(name.to_owned()),
      operator: None,
      order: visible_ops + 1,
      node: None,
    }));
  }
  position_layout_nodes(&mut nodes);

  let mut edges = Vec::new();
  for (operation, node) in function.nodes.iter().take(visible_ops).enumerate() {
    for input in node.inputs.iter().flatten() {
      if let Some(source) = value_nodes.get(input).copied() {
        edges.push(layout_edge(
          &nodes,
          source,
          op_nodes[operation],
          edges.len(),
          model.strings.get(*input),
        ));
      }
    }
  }
  for (output, target) in output_nodes {
    if let Some(source) = value_nodes.get(&output).copied() {
      edges.push(layout_edge(
        &nodes,
        source,
        target,
        edges.len(),
        model.strings.get(output),
      ));
    }
  }
  let stats = LayoutStats {
    nodes_used: nodes.len(),
    edges_used: edges.len(),
    omitted_nodes: function.nodes.len().saturating_sub(visible_ops),
    ..LayoutStats::default()
  };
  Some(LayoutGraph {
    graph: function_index,
    parent: None,
    name: Some(model.strings.get(function.name).to_owned()),
    subgraphs: Vec::new(),
    bounds: layout_bounds(&nodes),
    nodes,
    edges,
    stats,
  })
}

fn mlir_module_layout(model: &Model, module_index: usize, limit: usize) -> Option<LayoutGraph> {
  netron_rs_layout::layout_graph(
    model,
    &LayoutOptions {
      graph: module_index,
      max_nodes: Some(limit),
      ..LayoutOptions::default()
    },
  )
  .ok()
}

struct MlirLayoutNodeSpec {
  id: String,
  graph: usize,
  kind: LayoutNodeKind,
  rank: usize,
  value: Option<usize>,
  name: Option<String>,
  operator: Option<String>,
  order: usize,
  node: Option<usize>,
}

fn layout_node(spec: MlirLayoutNodeSpec) -> LayoutNode {
  let MlirLayoutNodeSpec {
    id,
    graph,
    kind,
    rank,
    value,
    name,
    operator,
    order,
    node,
  } = spec;
  LayoutNode {
    id,
    kind,
    graph,
    node,
    value,
    name,
    operator,
    origin: (kind == LayoutNodeKind::Operator).then_some("MLIR"),
    rank,
    order,
    x: 0.0,
    y: 0.0,
    width: 190.0,
    height: 64.0,
    hidden_initializers: 0,
  }
}

fn position_layout_nodes(nodes: &mut [LayoutNode]) {
  let mut by_rank: BTreeMap<usize, usize> = BTreeMap::new();
  for node in nodes {
    let order = by_rank.entry(node.rank).or_default();
    node.order = *order;
    node.x = node.rank as f32 * 260.0;
    node.y = *order as f32 * 96.0;
    *order += 1;
  }
}

fn layout_edge(
  nodes: &[LayoutNode],
  source: usize,
  target: usize,
  index: usize,
  label: &str,
) -> LayoutEdge {
  let from = nodes[source].id.clone();
  let to = nodes[target].id.clone();
  LayoutEdge {
    id: format!("mlir_edge:{index}"),
    value: index,
    name: label.to_owned(),
    from,
    to,
    points: vec![
      LayoutPoint {
        x: nodes[source].x + nodes[source].width,
        y: nodes[source].y + nodes[source].height / 2.0,
      },
      LayoutPoint {
        x: nodes[target].x,
        y: nodes[target].y + nodes[target].height / 2.0,
      },
    ],
  }
}

fn layout_bounds(nodes: &[LayoutNode]) -> LayoutBounds {
  LayoutBounds {
    width: nodes
      .iter()
      .map(|node| node.x + node.width)
      .fold(0.0, f32::max),
    height: nodes
      .iter()
      .map(|node| node.y + node.height)
      .fold(0.0, f32::max),
  }
}

fn projection_cache_key(
  session_id: u64,
  format: FormatKind,
  kind: &str,
  scope: &EntityHandle,
  limit: usize,
) -> String {
  format!(
    "session:{session_id}:format:{format:?}:scope:{}:collapse:none:{kind}:{limit}",
    handle_key(scope)
  )
}

fn handle_key(handle: &EntityHandle) -> String {
  match handle {
    EntityHandle::Graph { graph } => format!("graph:{graph}"),
    EntityHandle::Node { graph, node } => format!("node:{graph}:{node}"),
    EntityHandle::Value { graph, value } => format!("value:{graph}:{value}"),
    EntityHandle::Tensor { tensor } => format!("tensor:{tensor}"),
    EntityHandle::Function { function } => format!("function:{function}"),
    EntityHandle::Metadata { owner, key } => format!("metadata:{owner}:{key}"),
    EntityHandle::OperatorSet { domain, version } => {
      format!("opset:{}:{version}", domain.as_deref().unwrap_or("default"))
    }
    EntityHandle::Diagnostic { diagnostic } => format!("diagnostic:{diagnostic}"),
    EntityHandle::MlirModule { module } => format!("mlir_module:{module}"),
    EntityHandle::MlirFunction { function } => format!("mlir_function:{function}"),
    EntityHandle::MlirOperation { scope, operation } => {
      format!("mlir_operation:{scope}:{operation}")
    }
    EntityHandle::MlirValue { scope, value } => format!("mlir_value:{scope}:{value}"),
    EntityHandle::MlirRegion { scope, region } => format!("mlir_region:{scope}:{region}"),
    EntityHandle::MlirBlock { scope, block } => format!("mlir_block:{scope}:{block}"),
    EntityHandle::MlirSymbol { symbol } => format!("mlir_symbol:{symbol}"),
    EntityHandle::MlirDialect { dialect } => format!("mlir_dialect:{dialect}"),
    EntityHandle::MlirAttribute { scope, attribute } => {
      format!("mlir_attribute:{scope}:{attribute}")
    }
    EntityHandle::MlirResource { resource } => format!("mlir_resource:{resource}"),
  }
}

fn add_type_fields(
  model: &Model,
  fields: &mut BTreeMap<String, String>,
  element_type: Option<&TensorElementType>,
  shape: &[Dimension],
) {
  if let Some(element_type) = element_type {
    fields.insert("element_type".to_owned(), element_type_name(element_type));
  }
  fields.insert("rank".to_owned(), shape.len().to_string());
  fields.insert(
    "shape".to_owned(),
    shape
      .iter()
      .map(|dimension| dimension_label(model, &dimension.value))
      .collect::<Vec<_>>()
      .join(","),
  );
}

fn add_metadata_fields(fields: &mut BTreeMap<String, String>, metadata: &BTreeMap<String, String>) {
  for (key, value) in metadata {
    fields.insert(format!("metadata.{key}"), value.clone());
  }
}

fn push_related(related: &mut Vec<EntityHandle>, limit: usize, handle: EntityHandle) {
  if related.len() < limit {
    related.push(handle);
  }
}

fn tensor_storage_kind_name(kind: TensorStorageKind) -> &'static str {
  match kind {
    TensorStorageKind::Absent => "absent",
    TensorStorageKind::InlineBytes => "inline_bytes",
    TensorStorageKind::ElementList => "element_list",
    TensorStorageKind::External => "external",
    TensorStorageKind::Sparse => "sparse",
  }
}

fn search_index_entries(entries: &[SearchEntry], query: &str, limit: usize) -> Vec<SearchEntry> {
  let needle = query.trim().to_lowercase();
  if needle.is_empty() || limit == 0 {
    return Vec::new();
  }

  let mut matches = entries
    .iter()
    .enumerate()
    .filter_map(|(index, entry)| entry.score(&needle).map(|score| (score, index)))
    .collect::<Vec<_>>();
  matches.sort_unstable();
  matches.truncate(limit);
  matches
    .into_iter()
    .map(|(_, index)| entries[index].clone())
    .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticsResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub diagnostics: Vec<Diagnostic>,
  pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct ModelSession {
  id: u64,
  source: ModelSource,
  index: FormatIndex,
  model: Model,
}

impl ModelSession {
  pub fn open(data: &[u8], mut source: ModelSource) -> Result<Self, ModelError> {
    source.byte_len = data.len();
    source.content_identity = Some(content_identity(data));
    let model = netron_rs_formats::parse(source.input(data))?;
    let index = FormatIndex::build(&model, Some(data));
    Ok(Self {
      id: next_session_id(),
      source,
      index,
      model,
    })
  }

  pub fn id(&self) -> u64 {
    self.id
  }

  pub fn source(&self) -> &ModelSource {
    &self.source
  }

  pub fn format(&self) -> &FormatIndex {
    &self.index
  }

  pub fn summary(&self, limits: &SessionLimits) -> SessionSummary {
    let detail_limit = limits.clamp().detail;
    let initializers = initializer_count(&self.model);
    let subgraphs = subgraph_count(&self.model);
    let sparse_tensors = sparse_tensor_count(&self.model);
    let metadata = metadata_count(&self.model);
    let opsets = opset_count(&self.model);
    let onnx = match &self.index {
      FormatIndex::Onnx(index) => Some(OnnxSummary {
        producer: self.model.metadata.producer.clone(),
        producer_version: self.model.metadata.producer_version.clone(),
        model_domain: self.model.metadata.domain.clone(),
        model_version: self.model.metadata.model_version,
        description: self.model.metadata.description.clone(),
        graph_count: index.graph_count,
        function_count: index.function_count,
        node_count: index.node_count,
        value_count: index.value_count,
        tensor_count: index.tensor_count,
        initializer_count: index.initializer_count,
        subgraph_count: index.subgraph_count,
        sparse_tensor_count: index.sparse_tensor_count,
        opsets: index
          .opsets
          .iter()
          .take(detail_limit)
          .map(|opset| OnnxOperatorSet {
            domain: opset.domain.clone(),
            version: opset.version,
          })
          .collect(),
        metadata_keys: index
          .metadata_keys
          .iter()
          .take(detail_limit)
          .cloned()
          .collect(),
        graph_summaries: index
          .graph_summaries
          .iter()
          .take(detail_limit)
          .cloned()
          .collect(),
        histograms: limit_histograms(&index.histograms, detail_limit),
      }),
      _ => None,
    };
    let mlir = match &self.index {
      FormatIndex::Mlir(index) => Some(limit_mlir_summary(&index.summary, detail_limit)),
      _ => None,
    };
    let graphs = match &self.index {
      FormatIndex::Mlir(index) => index.summary.module_count,
      _ => self.model.graphs.len(),
    };
    let functions = match &self.index {
      FormatIndex::Mlir(index) => index.summary.function_count,
      _ => self.model.functions.len(),
    };
    let nodes = match &self.index {
      FormatIndex::Mlir(index) => index.summary.operation_count,
      _ => self
        .model
        .graphs
        .iter()
        .map(|graph| graph.nodes.len())
        .sum(),
    };
    let values = match &self.index {
      FormatIndex::Mlir(index) => index.summary.value_count,
      _ => self
        .model
        .graphs
        .iter()
        .map(|graph| graph.values.len())
        .sum(),
    };
    let mlir_resources = match &self.index {
      FormatIndex::Mlir(index) => index.summary.resource_count,
      _ => mlir_resource_count(&self.model),
    };

    SessionSummary {
      api_version: SESSION_API_VERSION,
      session_id: self.id,
      source: self.source.clone(),
      format: self.index.kind(),
      source_format_name: self.model.format.name.to_owned(),
      byte_len: self.source.byte_len,
      graphs,
      functions,
      nodes,
      values,
      tensors: self.model.tensors.len(),
      initializers,
      subgraphs,
      sparse_tensors,
      metadata,
      opsets,
      external_data: external_data_count(&self.model),
      mlir_resources,
      onnx,
      mlir,
    }
  }

  pub fn diagnostics(&self, limits: &SessionLimits) -> DiagnosticsResponse {
    let limit = limits.clamp().diagnostics;
    let mut diagnostics = self.index.diagnostics().to_vec();
    let truncated = diagnostics.len() > limit;
    diagnostics.truncate(limit);
    DiagnosticsResponse {
      api_version: SESSION_API_VERSION,
      session_id: self.id,
      diagnostics,
      truncated,
    }
  }

  pub fn search(&self, query: &str, limits: &SessionLimits) -> Vec<SearchEntry> {
    self.index.search(query, limits.clamp().search)
  }

  pub fn tensor_metadata(&self, limits: &SessionLimits) -> Vec<TensorMetadata> {
    self.index.tensor_metadata(limits.clamp().detail)
  }

  pub fn detail(&self, handle: &EntityHandle, limits: &SessionLimits) -> Option<EntityDetail> {
    self
      .index
      .detail(&self.model, handle, limits.clamp().detail)
  }

  pub fn slice(&self, scope: &EntityHandle, limits: &SessionLimits) -> Option<SliceResponse> {
    session_slice(&self.model, &self.index, self.id, scope, limits.clamp())
  }

  pub fn layout(&self, scope: &EntityHandle, limits: &SessionLimits) -> Option<LayoutResponse> {
    session_layout(&self.model, &self.index, self.id, scope, limits.clamp())
  }

  pub fn mlir_symbols(&self, limits: &SessionLimits) -> Option<Vec<MlirSymbolEntry>> {
    let FormatIndex::Mlir(index) = &self.index else {
      return None;
    };
    Some(
      index
        .symbols
        .iter()
        .take(limits.clamp().detail)
        .enumerate()
        .map(|(symbol, name)| MlirSymbolEntry {
          handle: EntityHandle::MlirSymbol { symbol },
          name: name.clone(),
        })
        .collect(),
    )
  }
}

impl FormatIndex {
  fn build(model: &Model, data: Option<&[u8]>) -> Self {
    match model.format.name {
      "ONNX" | "ONNX Tensor" => Self::Onnx(OnnxIndex::build(model)),
      "MLIR" => Self::Mlir(MlirIndex::build(
        model,
        data.and_then(|data| std::str::from_utf8(data).ok()),
      )),
      _ => Self::Unknown(ModelIndex::build(model)),
    }
  }

  fn diagnostics(&self) -> &[Diagnostic] {
    match self {
      Self::Onnx(index) => &index.diagnostics,
      Self::Mlir(index) => &index.diagnostics,
      Self::Unknown(index) => &index.diagnostics,
    }
  }

  pub fn kind(&self) -> FormatKind {
    match self {
      Self::Onnx(_) => FormatKind::Onnx,
      Self::Mlir(_) => FormatKind::Mlir,
      Self::Unknown(_) => FormatKind::Unknown,
    }
  }

  pub fn search(&self, query: &str, limit: usize) -> Vec<SearchEntry> {
    match self {
      Self::Onnx(index) => index.search(query, limit),
      Self::Mlir(index) => index.search(query, limit),
      Self::Unknown(index) => index.search(query, limit),
    }
  }

  pub fn tensor_metadata(&self, limit: usize) -> Vec<TensorMetadata> {
    match self {
      Self::Onnx(index) => index.tensor_metadata(limit),
      Self::Mlir(_) | Self::Unknown(_) => Vec::new(),
    }
  }

  pub fn detail(&self, model: &Model, handle: &EntityHandle, limit: usize) -> Option<EntityDetail> {
    match self {
      Self::Onnx(index) => index.detail(model, handle, limit),
      Self::Mlir(index) => index.detail(model, handle, limit),
      Self::Unknown(_) => None,
    }
  }
}

impl ModelSource {
  pub fn from_file(path: PathBuf, byte_len: usize) -> Self {
    let base_dir = path
      .parent()
      .filter(|path| !path.as_os_str().is_empty())
      .unwrap_or_else(|| Path::new("."))
      .to_path_buf();
    Self {
      kind: ModelSourceKind::File { path, base_dir },
      byte_len,
      content_identity: None,
      allow_unsafe_paths: false,
    }
  }

  pub fn from_memory(name: Option<String>, byte_len: usize) -> Self {
    Self {
      kind: ModelSourceKind::Memory { name },
      byte_len,
      content_identity: None,
      allow_unsafe_paths: false,
    }
  }

  pub fn with_allow_unsafe_paths(mut self, allow_unsafe_paths: bool) -> Self {
    self.allow_unsafe_paths = allow_unsafe_paths;
    self
  }

  fn input<'a>(&'a self, data: &'a [u8]) -> ModelInput<'a> {
    ModelInput {
      data,
      path: self.path(),
      allow_unsafe_paths: self.allow_unsafe_paths,
    }
  }

  fn path(&self) -> Option<&Path> {
    match &self.kind {
      ModelSourceKind::File { path, .. } => Some(path.as_path()),
      ModelSourceKind::Memory { name } => name.as_deref().map(Path::new),
    }
  }
}

impl Default for SessionLimits {
  fn default() -> Self {
    Self {
      export: 1_000,
      search: 25,
      slice: 500,
      layout: 500,
      preview: 200,
      detail: 100,
      diagnostics: 64,
    }
  }
}

impl SessionLimits {
  pub const HARD_MAX: Self = Self {
    export: 10_000,
    search: 500,
    slice: 5_000,
    layout: 5_000,
    preview: 5_000,
    detail: 1_000,
    diagnostics: 500,
  };

  pub fn clamp(self) -> Self {
    let default = Self::default();
    Self {
      export: clamp_limit(self.export, default.export, Self::HARD_MAX.export),
      search: clamp_limit(self.search, default.search, Self::HARD_MAX.search),
      slice: clamp_limit(self.slice, default.slice, Self::HARD_MAX.slice),
      layout: clamp_limit(self.layout, default.layout, Self::HARD_MAX.layout),
      preview: clamp_limit(self.preview, default.preview, Self::HARD_MAX.preview),
      detail: clamp_limit(self.detail, default.detail, Self::HARD_MAX.detail),
      diagnostics: clamp_limit(
        self.diagnostics,
        default.diagnostics,
        Self::HARD_MAX.diagnostics,
      ),
    }
  }
}

#[derive(Debug, Clone)]
pub struct ModelIndex {
  entries: Vec<SearchEntry>,
  diagnostics: Vec<Diagnostic>,
}

impl ModelIndex {
  pub fn build(model: &Model) -> Self {
    let mut entries = Vec::new();
    for graph in &model.graphs {
      let graph_index = graph.id.index();
      let graph_name = graph.name.map(|id| model.strings.get(id).to_owned());
      entries.push(SearchEntry::new(
        SearchKind::Graph,
        Some(graph_index),
        graph_index,
        graph_name.clone(),
        None,
        None,
      ));

      for node in &graph.nodes {
        let name = node.name.map(|id| model.strings.get(id).to_owned());
        let operator = Some(model.strings.get(node.operator.name).to_owned());
        entries.push(SearchEntry::new(
          SearchKind::Node,
          Some(graph_index),
          node.id.index(),
          name,
          operator,
          Some(node.operator.origin),
        ));
      }

      for value in &graph.values {
        entries.push(SearchEntry::new(
          SearchKind::Value,
          Some(graph_index),
          value.id.index(),
          Some(model.strings.get(value.name).to_owned()),
          None,
          None,
        ));
      }
    }

    for tensor in &model.tensors {
      entries.push(SearchEntry::new(
        SearchKind::Tensor,
        None,
        tensor.id.index(),
        tensor.name.map(|id| model.strings.get(id).to_owned()),
        None,
        None,
      ));
    }

    for (index, function) in model.functions.iter().enumerate() {
      entries.push(SearchEntry::new(
        SearchKind::Function,
        None,
        index,
        Some(model.strings.get(function.name).to_owned()),
        None,
        None,
      ));
    }

    Self {
      entries,
      diagnostics: Vec::new(),
    }
  }

  pub fn len(&self) -> usize {
    self.entries.len()
  }

  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  pub fn search(&self, query: &str, limit: usize) -> Vec<SearchEntry> {
    search_index_entries(&self.entries, query, limit)
  }
}

impl SearchEntry {
  fn new(
    kind: SearchKind,
    graph: Option<usize>,
    id: usize,
    name: Option<String>,
    operator: Option<String>,
    origin: Option<&'static str>,
  ) -> Self {
    Self::new_with_search_terms(
      kind,
      graph,
      id,
      name,
      operator,
      origin,
      Vec::<String>::new(),
    )
  }

  fn new_with_search_terms<I: IntoIterator<Item = String>>(
    kind: SearchKind,
    graph: Option<usize>,
    id: usize,
    name: Option<String>,
    operator: Option<String>,
    origin: Option<&'static str>,
    searchable_terms: I,
  ) -> Self {
    Self::new_with_handle_and_search_terms(
      kind,
      default_handle(kind, graph, id),
      graph,
      id,
      name,
      operator,
      origin,
      searchable_terms,
    )
  }

  #[allow(clippy::too_many_arguments)]
  fn new_with_handle_and_search_terms<I: IntoIterator<Item = String>>(
    kind: SearchKind,
    handle: EntityHandle,
    graph: Option<usize>,
    id: usize,
    name: Option<String>,
    operator: Option<String>,
    origin: Option<&'static str>,
    searchable_terms: I,
  ) -> Self {
    let mut seen = BTreeSet::new();
    let mut searchable = Vec::new();

    for entry in [name.as_deref(), operator.as_deref(), origin]
      .into_iter()
      .flatten()
    {
      let value = entry.to_lowercase();
      if seen.insert(value.clone()) {
        searchable.push(value);
      }
    }

    for entry in searchable_terms {
      let value = entry.trim().to_lowercase();
      if value.is_empty() {
        continue;
      }
      if seen.insert(value.clone()) {
        searchable.push(value);
      }
    }

    Self {
      kind,
      handle,
      graph,
      id,
      name,
      operator,
      origin,
      searchable: searchable.join(" "),
    }
  }

  pub fn handle(&self) -> EntityHandle {
    self.handle.clone()
  }

  fn score(&self, needle: &str) -> Option<u8> {
    let mut score = best_field_score(self.name.as_deref(), needle, 0);
    score = min_score(score, best_field_score(self.operator.as_deref(), needle, 3));
    score = min_score(score, best_field_score(self.origin, needle, 6));
    if self.searchable.contains(needle) {
      score = min_score(score, Some(9));
    }
    score
  }
}

fn default_handle(kind: SearchKind, graph: Option<usize>, id: usize) -> EntityHandle {
  match kind {
    SearchKind::Graph => EntityHandle::Graph { graph: id },
    SearchKind::Node => EntityHandle::Node {
      graph: graph.unwrap_or(0),
      node: id,
    },
    SearchKind::Value => EntityHandle::Value {
      graph: graph.unwrap_or(0),
      value: id,
    },
    SearchKind::Tensor => EntityHandle::Tensor { tensor: id },
    SearchKind::Function => EntityHandle::Function { function: id },
    SearchKind::Metadata => EntityHandle::Metadata {
      owner: String::new(),
      key: id.to_string(),
    },
    SearchKind::OperatorSet => EntityHandle::OperatorSet {
      domain: None,
      version: id as i64,
    },
    SearchKind::Module => EntityHandle::MlirModule { module: id },
    SearchKind::Operation => EntityHandle::MlirOperation {
      scope: String::new(),
      operation: id,
    },
    SearchKind::Region => EntityHandle::MlirRegion {
      scope: String::new(),
      region: id,
    },
    SearchKind::Block => EntityHandle::MlirBlock {
      scope: String::new(),
      block: id,
    },
    SearchKind::Symbol => EntityHandle::MlirSymbol { symbol: id },
    SearchKind::Dialect => EntityHandle::MlirDialect {
      dialect: String::new(),
    },
    SearchKind::Attribute => EntityHandle::MlirAttribute {
      scope: String::new(),
      attribute: id,
    },
    SearchKind::Resource => EntityHandle::MlirResource { resource: id },
  }
}

fn external_data_count(model: &Model) -> usize {
  model
    .tensors
    .iter()
    .filter(|tensor| matches!(tensor.storage, TensorStorage::External { .. }))
    .count()
}

fn mlir_resource_count(model: &Model) -> usize {
  if model.format.name != "MLIR" {
    return 0;
  }
  let count = model
    .graphs
    .iter()
    .flat_map(|graph| &graph.nodes)
    .flat_map(|node| &node.attributes)
    .chain(
      model
        .functions
        .iter()
        .flat_map(|function| &function.nodes)
        .flat_map(|node| &node.attributes),
    )
    .filter(|attribute| model.strings.get(attribute.name) == "rodata")
    .count();
  if count > 0 {
    return count;
  }
  mlir_bytecode_count(model, "bytecode.resource_count").unwrap_or(0)
}

fn content_identity(data: &[u8]) -> String {
  let hash = data.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
    (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
  });
  format!("fnv1a64:{hash:016x}")
}

fn next_session_id() -> u64 {
  static NEXT: AtomicU64 = AtomicU64::new(1);
  NEXT.fetch_add(1, Ordering::Relaxed)
}

fn clamp_limit(value: usize, default: usize, max: usize) -> usize {
  if value == 0 { default } else { value.min(max) }
}

fn best_field_score(field: Option<&str>, needle: &str, base: u8) -> Option<u8> {
  let field = field?;
  let field = field.to_lowercase();
  if field == needle {
    Some(base)
  } else if field.starts_with(needle) {
    Some(base + 1)
  } else if field.contains(needle) {
    Some(base + 2)
  } else {
    None
  }
}

fn min_score(left: Option<u8>, right: Option<u8>) -> Option<u8> {
  match (left, right) {
    (Some(left), Some(right)) => Some(left.min(right)),
    (Some(value), None) | (None, Some(value)) => Some(value),
    (None, None) => None,
  }
}

#[cfg(test)]
mod tests {
  use std::collections::{BTreeMap, BTreeSet};
  use std::path::{Path, PathBuf};

  use netron_rs_core::{
    Attribute, AttributeValue, FormatInfo, Function, FunctionNode, FunctionValue, Graph, Node,
    Operator, Tensor, TensorElementType, TensorStorage, Value,
  };

  use super::*;

  #[test]
  fn search_ranks_exact_names_before_operator_matches() {
    let model = fixture_model();
    let index = ModelIndex::build(&model);

    let hits = index.search("Add", 4);

    assert_eq!(hits[0].kind, SearchKind::Value);
    assert_eq!(hits[0].name.as_deref(), Some("Add"));
    assert_eq!(hits[1].kind, SearchKind::Node);
    assert_eq!(hits[1].operator.as_deref(), Some("Add"));
  }

  #[test]
  fn search_is_case_insensitive_and_bounded() {
    let model = fixture_model();
    let index = ModelIndex::build(&model);

    let hits = index.search("identity", 1);

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].name.as_deref(), Some("identity_output"));
  }

  #[test]
  fn session_opens_named_memory_mlir() {
    let data = br#"module {
  func.func @main() {
    %0 = arith.constant 0 : i32
    return
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("module.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");

    let summary = session.summary(&SessionLimits::default());

    assert_eq!(summary.format, FormatKind::Mlir);
    assert_eq!(summary.functions, 1);
    assert!(summary.source.content_identity.is_some());
    assert!(!session.diagnostics(&SessionLimits::default()).truncated);
  }

  #[test]
  fn session_opens_mlir_bytecode_summary_with_function_counts() {
    let data = std::fs::read(mlirbc_fixture("model.mlirbc")).expect("fixture exists");
    let session = ModelSession::open(
      &data,
      ModelSource::from_memory(Some("model.mlirbc".to_owned()), data.len()),
    )
    .expect("open MLIR bytecode session");

    let summary = session.summary(&SessionLimits::default());
    let mlir = summary.mlir.expect("mlir summary");

    assert_eq!(summary.format, FormatKind::Mlir);
    assert!(summary.functions > 0);
    assert!(summary.nodes >= summary.functions);
    assert_eq!(summary.functions, mlir.function_count);
    assert_eq!(summary.nodes, mlir.operation_count);
    assert!(
      mlir
        .functions
        .iter()
        .any(|function| function.name.as_deref() == Some("bytecode.func.0"))
    );
    assert!(!mlir.regions.is_empty());
    assert!(!mlir.blocks.is_empty());
    assert!(mlir.dialects.iter().any(|dialect| dialect == "torch"));

    let hits = session.search("func.func", &SessionLimits::default());
    assert!(
      hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirOperation { .. }))
    );
  }

  #[test]
  fn mlir_bytecode_unsupported_sections_become_diagnostics() {
    let mut data = std::fs::read(mlirbc_fixture("model.mlirbc")).expect("fixture exists");
    let insert_at = data[4..]
      .iter()
      .position(|byte| *byte == 0)
      .map(|index| index + 5)
      .expect("producer terminator");
    data.splice(insert_at..insert_at, [9, 1]);

    let session = ModelSession::open(
      &data,
      ModelSource::from_memory(Some("unsupported-section.mlirbc".to_owned()), data.len()),
    )
    .expect("open MLIR bytecode session");
    let summary = session.summary(&SessionLimits::default());
    assert_eq!(summary.format, FormatKind::Mlir);
    assert_eq!(summary.mlir.as_ref().unwrap().diagnostic_count, 1);

    let diagnostics = session.diagnostics(&SessionLimits::default()).diagnostics;
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "mlir.bytecode");
    assert!(
      diagnostics[0]
        .message
        .contains("unsupported bytecode section 9")
    );
  }

  #[test]
  fn mlir_bytecode_resources_surface_in_summary_search_and_detail() {
    let path = mlirbc_fixture("sd-clip-tank.mlirbc");
    let data = std::fs::read(&path).expect("fixture exists");
    let session =
      ModelSession::open(&data, ModelSource::from_file(path, data.len())).expect("open fixture");

    let summary = session.summary(&SessionLimits::default());
    let mlir = summary.mlir.as_ref().expect("mlir summary");
    assert_eq!(summary.mlir_resources, 1);
    assert_eq!(mlir.resource_count, 1);

    let hits = session.search("torch_tensor_1_77_torch.int64", &SessionLimits::default());
    let resource = hits
      .iter()
      .find(|entry| matches!(entry.handle, EntityHandle::MlirResource { resource: 0 }))
      .expect("resource search entry");
    assert_eq!(resource.kind, SearchKind::Resource);

    let detail = session
      .detail(
        &EntityHandle::MlirResource { resource: 0 },
        &SessionLimits::default(),
      )
      .expect("resource detail");
    assert_eq!(
      detail.fields.get("name").map(String::as_str),
      Some("torch_tensor_1_77_torch.int64")
    );
    assert_eq!(
      detail.fields.get("scope").map(String::as_str),
      Some("builtin")
    );
    assert_eq!(detail.fields.get("kind").map(String::as_str), Some("blob"));
  }

  #[test]
  fn mlir_index_summarizes_and_searches_native_terms() {
    let data = br#"module @jit_mlp attributes {mhlo.num_partitions = 1 : i32} {
  func.func @loop() {
  ^bb1(%i: index):
    %c = stablehlo.constant dense<1.0> : tensor<1xf32>
    %0 = arith.index_cast %i : index to i64 loc("kernel.mlir")
    %1 = vm.const.ref.rodata @blob : !vm.buffer
    return
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("model.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");

    let summary = session.summary(&SessionLimits::default());
    let mlir = summary.mlir.expect("mlir summary");
    assert_eq!(mlir.module_count, 1);
    assert_eq!(mlir.function_count, 1);
    assert_eq!(mlir.operation_count, 3);
    assert_eq!(mlir.block_argument_count, 1);
    assert_eq!(mlir.region_count, 2);
    assert_eq!(mlir.block_count, 3);
    assert_eq!(mlir.resource_count, 1);
    assert_eq!(
      mlir.modules[0].handle,
      EntityHandle::MlirModule { module: 0 }
    );
    assert_eq!(mlir.functions[0].scope_id, "function:0");
    assert_eq!(mlir.blocks[1].block_argument_count, 1);
    assert_eq!(
      mlir.blocks[2].handle,
      EntityHandle::MlirBlock {
        scope: "source".to_owned(),
        block: 2,
      }
    );
    assert!(mlir.dialects.contains(&"arith".to_owned()));
    assert!(mlir.dialects.contains(&"stablehlo".to_owned()));
    assert!(mlir.dialects.contains(&"vm".to_owned()));

    let module_hits = session.search("jit_mlp", &SessionLimits::default());
    assert!(
      module_hits
        .iter()
        .any(|hit| hit.handle == EntityHandle::MlirModule { module: 0 })
    );
    assert!(
      module_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirSymbol { .. }))
    );

    let function_hits = session.search("loop", &SessionLimits::default());
    assert!(
      function_hits
        .iter()
        .any(|hit| hit.handle == EntityHandle::MlirFunction { function: 0 })
    );

    let operation_hits = session.search("arith.index_cast", &SessionLimits::default());
    assert!(
      operation_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirOperation { .. }))
    );

    let dialect_hits = session.search("arith", &SessionLimits::default());
    assert!(dialect_hits.iter().any(|hit| {
      hit.handle
        == EntityHandle::MlirDialect {
          dialect: "arith".to_owned(),
        }
    }));

    let attribute_hits = session.search("mhlo.num_partitions", &SessionLimits::default());
    assert!(
      attribute_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirAttribute { .. }))
    );

    let block_arg_hits = session.search("%i", &SessionLimits::default());
    assert!(block_arg_hits.iter().any(|hit| {
      hit.handle
        == EntityHandle::MlirValue {
          scope: "function:0".to_owned(),
          value: 0,
        }
    }));

    let resource_hits = session.search("blob", &SessionLimits::default());
    assert!(
      resource_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirResource { .. }))
    );

    let block_hits = session.search("^bb1", &SessionLimits::default());
    assert!(block_hits.iter().any(|hit| {
      hit.handle
        == EntityHandle::MlirBlock {
          scope: "source".to_owned(),
          block: 2,
        }
    }));

    let type_hits = session.search("tensor<1xf32>", &SessionLimits::default());
    assert!(
      type_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirAttribute { .. }))
    );

    let dense_hits = session.search("dense", &SessionLimits::default());
    assert!(dense_hits.iter().any(|hit| matches!(
      hit.handle,
      EntityHandle::MlirOperation { .. } | EntityHandle::MlirAttribute { .. }
    )));

    let location_hits = session.search("kernel.mlir", &SessionLimits::default());
    assert!(
      location_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirOperation { .. }))
    );
  }

  #[test]
  fn mlir_detail_returns_scoped_entity_metadata_and_relations() {
    let data = br#"module @jit_mlp attributes {mhlo.num_partitions = 1 : i32} {
  func.func @loop(%i: index) {
    %c = stablehlo.constant dense<1.0> : tensor<1xf32>
    %0 = arith.index_cast %i : index to i64 loc("kernel.mlir")
    %1 = vm.const.ref.rodata @blob : !vm.buffer
    return %0 : i64
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("detail.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");
    let summary = session.summary(&SessionLimits::default());
    let mlir = summary.mlir.expect("mlir summary");

    let module_detail = session
      .detail(
        &EntityHandle::MlirModule { module: 0 },
        &SessionLimits::default(),
      )
      .expect("module detail");
    assert_eq!(module_detail.title, "@jit_mlp");
    assert_eq!(
      module_detail.fields.get("scope").map(String::as_str),
      Some("module:0")
    );
    assert!(
      module_detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::MlirRegion { .. }))
    );
    let module_slice = session
      .slice(
        &EntityHandle::MlirModule { module: 0 },
        &SessionLimits::default(),
      )
      .expect("module slice");
    assert!(
      module_slice
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::MlirFunction { .. })))
    );

    let function_handle = session
      .search("loop", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        (matches!(hit.handle, EntityHandle::MlirFunction { .. })).then_some(hit.handle)
      })
      .expect("function hit");
    let function_detail = session
      .detail(&function_handle, &SessionLimits::default())
      .expect("function detail");
    assert_eq!(function_detail.title, "@jit_mlp::@loop");
    assert_eq!(
      function_detail.fields.get("scope").map(String::as_str),
      Some("function:0")
    );
    assert!(
      function_detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::MlirValue { .. }))
    );

    let operation_handle = session
      .search("stablehlo.constant", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirOperation { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("operation hit");
    let operation_detail = session
      .detail(&operation_handle, &SessionLimits::default())
      .expect("operation detail");
    assert_eq!(
      operation_detail.fields.get("operator").map(String::as_str),
      Some("stablehlo.constant")
    );
    assert!(operation_detail.fields.contains_key("origin"));
    assert!(
      operation_detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::MlirValue { .. }))
    );

    let value_handle = session
      .search("%i", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirValue { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("value hit");
    let value_detail = session
      .detail(&value_handle, &SessionLimits::default())
      .expect("value detail");
    assert_eq!(value_detail.title, "%i");
    assert_eq!(
      value_detail.fields.get("name").map(String::as_str),
      Some("%i")
    );
    assert!(value_detail.fields.contains_key("is_block_argument"));

    let block_handle = match mlir.blocks.first() {
      Some(block) => block.handle.clone(),
      None => panic!("module should have a block"),
    };
    let block_detail = session
      .detail(&block_handle, &SessionLimits::default())
      .expect("block detail");
    assert_eq!(
      block_detail.handle, block_handle,
      "block detail should preserve handle"
    );
    assert_eq!(
      block_detail.fields.get("scope").map(String::as_str),
      Some("module:0")
    );

    let region_handle = match mlir.regions.first() {
      Some(region) => region.handle.clone(),
      None => panic!("module should have a region"),
    };
    let region_detail = session
      .detail(&region_handle, &SessionLimits::default())
      .expect("region detail");
    assert_eq!(region_detail.handle, region_handle);
    assert!(
      region_detail
        .fields
        .get("region")
        .is_some_and(|value| value == "0")
    );

    let symbol_handle = session
      .search("jit_mlp", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirSymbol { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("symbol hit");
    let symbol_detail = session
      .detail(&symbol_handle, &SessionLimits::default())
      .expect("symbol detail");
    assert_eq!(
      symbol_detail.fields.get("name").map(String::as_str),
      Some("jit_mlp")
    );
    assert_eq!(
      symbol_detail
        .fields
        .get("symbol_count")
        .and_then(|value| value.parse::<usize>().ok()),
      Some(mlir.symbol_count)
    );

    let dialect_handle = session
      .search("arith", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirDialect { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("dialect hit");
    let dialect_detail = session
      .detail(&dialect_handle, &SessionLimits::default())
      .expect("dialect detail");
    assert_eq!(
      dialect_detail
        .fields
        .get("operation_count")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_default(),
      1
    );
    assert!(
      dialect_detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::MlirOperation { .. }))
    );

    let attribute_handle = session
      .search("mhlo.num_partitions", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirAttribute { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("attribute hit");
    let attribute_detail = session
      .detail(&attribute_handle, &SessionLimits::default())
      .expect("attribute detail");
    assert_eq!(
      attribute_detail.fields.get("name").map(String::as_str),
      Some("mhlo.num_partitions")
    );
    assert_eq!(
      attribute_detail.fields.get("scope").map(String::as_str),
      Some("module:0")
    );
    assert_eq!(
      attribute_detail
        .fields
        .get("value_count")
        .and_then(|value| value.parse::<usize>().ok()),
      Some(1)
    );
    assert!(
      attribute_detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::MlirModule { module: 0 }))
    );

    let resource_handle = session
      .search("blob", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirResource { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("resource hit");
    let resource_detail = session
      .detail(&resource_handle, &SessionLimits::default())
      .expect("resource detail");
    assert_eq!(
      resource_detail.fields.get("kind").map(String::as_str),
      Some("rodata")
    );
    assert_eq!(
      resource_detail.fields.get("name").map(String::as_str),
      Some("blob")
    );
    assert_eq!(
      resource_detail.fields.get("scope").map(String::as_str),
      Some("function:0")
    );
  }

  #[test]
  fn session_slice_and_layout_are_bounded_projection_envelopes() {
    let model = fixture_two_node_onnx_model();
    let session = ModelSession {
      id: 104,
      source: ModelSource::from_memory(Some("bounded.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
    };

    let summary_json = serde_json::to_value(session.summary(&SessionLimits::default())).unwrap();
    assert!(summary_json.get("layout").is_none());
    assert!(summary_json.get("slice").is_none());

    let slice = session
      .slice(
        &EntityHandle::Node { graph: 0, node: 0 },
        &SessionLimits {
          slice: 2,
          ..SessionLimits::default()
        },
      )
      .expect("node slice");
    assert_eq!(slice.limit_used, 2);
    assert!(slice.truncated);
    assert!(slice.omitted_count > 0);
    assert!(!slice.boundaries.is_empty());
    assert!(slice.boundaries.len() <= 2);
    assert!(slice.cache_key.contains("collapse:none"));
    assert!(
      slice
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::Node { node: 0, .. })))
    );
    let entity_ids = slice
      .entities
      .iter()
      .map(|entity| entity.id.as_str())
      .collect::<BTreeSet<_>>();
    assert!(
      slice.edges.iter().all(
        |edge| entity_ids.contains(edge.from.as_str()) && entity_ids.contains(edge.to.as_str())
      )
    );

    let graph_slice = session
      .slice(
        &EntityHandle::Graph { graph: 0 },
        &SessionLimits {
          slice: 4,
          ..SessionLimits::default()
        },
      )
      .expect("graph slice");
    assert!(
      graph_slice
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::Node { node: 1, .. })))
    );

    let layout = session
      .layout(
        &EntityHandle::Graph { graph: 0 },
        &SessionLimits {
          layout: 1,
          ..SessionLimits::default()
        },
      )
      .expect("graph layout");
    assert_eq!(layout.limit_used, 1);
    assert!(layout.truncated);
    assert_eq!(layout.graph.stats.omitted_nodes, 1);
    assert!(layout.cache_key.contains("format:Onnx"));
  }

  #[test]
  #[ignore = "writes Goal 6 evidence artifacts from the local Netron corpus"]
  fn goal6_layout_metadata_artifacts_for_large_fixtures() {
    use std::{fs, path::Path};

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
      .ancestors()
      .nth(3)
      .expect("workspace root");
    let fixture_root = repo_root.join("netron/third_party/test");
    let artifact_root = repo_root.join("artifacts/goal-6/layout-metadata");
    fs::create_dir_all(&artifact_root).expect("artifact directory");

    for (fixture, scope) in [
      ("onnx/gpt-oss-20b.onnx", EntityHandle::Graph { graph: 0 }),
      (
        "onnx/phi3-mini-128k-instruct-cuda-fp16.onnx",
        EntityHandle::Graph { graph: 0 },
      ),
      ("onnx/longformer.onnx.zip", EntityHandle::Graph { graph: 0 }),
      (
        "mlir/examples.mnist_xla.mlir",
        EntityHandle::MlirFunction { function: 0 },
      ),
      (
        "mlir/stablehlo_gpt_125M.mlir",
        EntityHandle::MlirFunction { function: 0 },
      ),
    ] {
      let path = fixture_root.join(fixture);
      let data = fs::read(&path).expect("read Goal 6 fixture");
      let session = ModelSession::open(&data, ModelSource::from_file(path.clone(), data.len()))
        .expect("open Goal 6 fixture");
      let summary = session.summary(&SessionLimits::default());
      let summary_json = serde_json::to_value(&summary).expect("summary json");
      assert!(summary_json.get("layout").is_none());
      assert!(summary_json.get("slice").is_none());

      let slice = session
        .slice(
          &scope,
          &SessionLimits {
            slice: 2,
            ..SessionLimits::default()
          },
        )
        .expect("bounded slice");
      let layout = session
        .layout(
          &scope,
          &SessionLimits {
            layout: 1,
            ..SessionLimits::default()
          },
        )
        .expect("bounded layout");
      let artifact = serde_json::json!({
          "fixture": fixture,
          "format": summary.format,
          "open_has_no_layout_payload": true,
          "open_has_no_slice_payload": true,
          "slice": {
              "limit_used": slice.limit_used,
              "truncated": slice.truncated,
              "omitted_count": slice.omitted_count,
              "warnings": slice.warnings.len(),
              "entities": slice.entities.len(),
              "edges": slice.edges.len(),
              "boundaries": slice.boundaries.len(),
              "cache_key": slice.cache_key,
          },
          "layout": {
              "limit_used": layout.limit_used,
              "truncated": layout.truncated,
              "omitted_count": layout.omitted_count,
              "warnings": layout.warnings.len(),
              "nodes": layout.graph.nodes.len(),
              "edges": layout.graph.edges.len(),
              "cache_key": layout.cache_key,
          },
      });
      let name = Path::new(fixture)
        .file_name()
        .expect("fixture name")
        .to_string_lossy()
        .replace('.', "_");
      fs::write(
        artifact_root.join(format!("{name}.json")),
        serde_json::to_string_pretty(&artifact).expect("artifact json"),
      )
      .expect("write Goal 6 artifact");
    }
  }

  #[test]
  fn mlir_slice_and_layout_cover_function_and_dialect_scopes() {
    let data = br#"module {
  func.func @loop(%i: index) -> i64 {
    %0 = arith.index_cast %i : index to i64
    %1 = arith.addi %0, %0 : i64
    return %1 : i64
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("bounded.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");

    let dialect = session
      .slice(
        &EntityHandle::MlirDialect {
          dialect: "arith".to_owned(),
        },
        &SessionLimits {
          slice: 2,
          ..SessionLimits::default()
        },
      )
      .expect("dialect slice");
    assert_eq!(dialect.limit_used, 2);
    assert!(dialect.truncated);
    assert!(
      dialect
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::MlirDialect { .. })))
    );

    let operation = session
      .slice(
        &EntityHandle::MlirOperation {
          scope: "function:0".to_owned(),
          operation: 1,
        },
        &SessionLimits::default(),
      )
      .expect("operation slice");
    assert!(
      operation
        .edges
        .iter()
        .any(|edge| edge.from.starts_with("mlir_value:function:0:")
          && edge.to == "mlir_operation:function:0:1")
    );

    let region = session
      .slice(
        &EntityHandle::MlirRegion {
          scope: "function:0".to_owned(),
          region: 0,
        },
        &SessionLimits::default(),
      )
      .expect("region slice");
    assert!(
      region
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::MlirRegion { .. })))
    );

    let block = session
      .slice(
        &EntityHandle::MlirBlock {
          scope: "function:0".to_owned(),
          block: 0,
        },
        &SessionLimits::default(),
      )
      .expect("block slice");
    assert!(
      block
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::MlirBlock { .. })))
    );

    let symbol_handle = session
      .search("loop", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| matches!(hit.handle, EntityHandle::MlirSymbol { .. }).then_some(hit.handle))
      .expect("symbol hit");
    let symbol = session
      .slice(&symbol_handle, &SessionLimits::default())
      .expect("symbol slice");
    assert!(
      symbol
        .entities
        .iter()
        .any(|entity| matches!(entity.handle, Some(EntityHandle::MlirFunction { .. })))
    );

    let layout = session
      .layout(
        &EntityHandle::MlirFunction { function: 0 },
        &SessionLimits {
          layout: 1,
          ..SessionLimits::default()
        },
      )
      .expect("MLIR function layout");
    assert_eq!(layout.limit_used, 1);
    assert!(layout.truncated);
    assert_eq!(layout.graph.stats.omitted_nodes, 1);
    assert!(layout.cache_key.contains("mlir_function:0"));
  }

  #[test]
  fn layout_reports_initializer_omissions_as_truncation() {
    let model = fixture_initializer_onnx_model();
    let session = ModelSession {
      id: 105,
      source: ModelSource::from_memory(Some("initializer.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
    };

    let layout = session
      .layout(&EntityHandle::Graph { graph: 0 }, &SessionLimits::default())
      .expect("graph layout");
    assert!(layout.truncated);
    assert!(layout.omitted_count >= 1);
    assert_eq!(layout.graph.stats.omitted_nodes, 0);
    assert_eq!(layout.graph.stats.omitted_initializers, 1);
  }

  #[test]
  fn mlir_layout_preserves_function_graph_id() {
    let data = br#"module {
  func.func @first(%i: index) -> i64 {
    %0 = arith.index_cast %i : index to i64
    return %0 : i64
  }
  func.func @second(%i: index) -> i64 {
    %0 = arith.index_cast %i : index to i64
    return %0 : i64
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("two-functions.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");

    let layout = session
      .layout(
        &EntityHandle::MlirFunction { function: 1 },
        &SessionLimits::default(),
      )
      .expect("MLIR function layout");
    assert_eq!(layout.graph.graph, 1);
    assert!(layout.graph.nodes.iter().all(|node| node.graph == 1));
  }

  #[test]
  fn mlir_index_preserves_generic_and_synthetic_scope_handles() {
    let generic = br#""builtin.module"() ({
  "func.func"() ({
    return
  }) : () -> ()
}) : () -> ()
"#;
    let generic_session = ModelSession::open(
      generic,
      ModelSource::from_memory(Some("generic.mlir".to_owned()), generic.len()),
    )
    .expect("open generic MLIR session");
    let generic_summary = generic_session.summary(&SessionLimits::default());
    let generic_mlir = generic_summary.mlir.expect("generic mlir summary");
    assert_eq!(generic_mlir.function_count, 1);
    assert_eq!(
      generic_mlir.functions[0].handle,
      EntityHandle::MlirFunction { function: 0 }
    );
    assert_eq!(
      generic_mlir.regions[0].handle,
      EntityHandle::MlirRegion {
        scope: "function:0".to_owned(),
        region: 0,
      }
    );

    let repeated = br#"module {
  func.func @main() {
    return
  }
}
module attributes {"triton_gpu.num-warps" = 16 : i32} {
  func.func @main() {
    return
  }
}
"#;
    let repeated_session = ModelSession::open(
      repeated,
      ModelSource::from_memory(Some("repeated.mlir".to_owned()), repeated.len()),
    )
    .expect("open repeated MLIR session");
    let repeated_summary = repeated_session.summary(&SessionLimits::default());
    let repeated_mlir = repeated_summary.mlir.expect("repeated mlir summary");
    assert_eq!(
      repeated_mlir.functions[0].name.as_deref(),
      Some("$0::@main")
    );
    assert_eq!(repeated_mlir.functions[1].scope_id, "function:1");
    assert_eq!(repeated_mlir.modules[0].name.as_deref(), Some("$1"));

    let scope_hits = repeated_session.search("function:1", &SessionLimits::default());
    assert!(
      scope_hits
        .iter()
        .any(|hit| { hit.handle == EntityHandle::MlirFunction { function: 1 } })
    );
  }

  #[test]
  fn session_limits_use_defaults_and_hard_maxima() {
    let limits = SessionLimits {
      search: 0,
      diagnostics: usize::MAX,
      ..SessionLimits::default()
    }
    .clamp();

    assert_eq!(limits.search, SessionLimits::default().search);
    assert_eq!(limits.diagnostics, SessionLimits::HARD_MAX.diagnostics);
  }

  #[test]
  fn onnx_summary_exposes_onnx_specific_counts() {
    let model = fixture_onnx_model();
    let session = ModelSession {
      id: 99,
      source: ModelSource::from_memory(Some("model.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
    };

    let summary = session.summary(&SessionLimits::default());
    let onnx = summary
      .onnx
      .expect("onnx models should include onnx summary section");

    assert_eq!(summary.format, FormatKind::Onnx);
    assert_eq!(summary.initializers, 2);
    assert_eq!(summary.sparse_tensors, 1);
    assert_eq!(summary.opsets, 1);
    assert_eq!(summary.external_data, 1);
    assert_eq!(onnx.producer, Some("query-tests".to_owned()));
    assert_eq!(onnx.graph_count, 1);
    assert_eq!(onnx.function_count, 1);
    assert_eq!(onnx.node_count, 1);
    assert_eq!(onnx.value_count, 3);
    assert_eq!(onnx.tensor_count, 3);
    assert_eq!(onnx.initializer_count, 2);
    assert_eq!(onnx.subgraph_count, 0);
    assert_eq!(onnx.sparse_tensor_count, 1);
    assert_eq!(onnx.opsets.len(), 1);
    assert_eq!(onnx.opsets[0].domain.as_deref(), Some("ai.onnx"));
    assert_eq!(onnx.metadata_keys, vec!["global_meta".to_owned()]);
    assert_eq!(onnx.graph_summaries[0].node_count, 1);
    assert_eq!(onnx.graph_summaries[0].tensor_count, 1);
    assert_eq!(histogram_count(&onnx.histograms.graph_node_counts, "1"), 1);
    assert_eq!(histogram_count(&onnx.histograms.graph_value_counts, "3"), 1);
    assert_eq!(
      histogram_count(&onnx.histograms.graph_tensor_counts, "1"),
      1
    );
    assert_eq!(
      histogram_count(&onnx.histograms.operator_types, "MyNode"),
      1
    );
    assert_eq!(histogram_count(&onnx.histograms.operator_types, "FnOp"), 1);
    assert_eq!(histogram_count(&onnx.histograms.domains, "custom"), 1);
    assert_eq!(histogram_count(&onnx.histograms.dtypes, "float32"), 3);
    assert_eq!(
      histogram_count(&onnx.histograms.storage_kinds, "external"),
      1
    );
    assert_eq!(histogram_count(&onnx.histograms.shape_ranks, "0"), 3);
    assert_eq!(histogram_count(&onnx.histograms.fan_in, "1"), 2);
  }

  #[test]
  fn onnx_search_indexes_metadata_external_data_and_functions() {
    let model = fixture_onnx_model();
    let session = ModelSession {
      id: 100,
      source: ModelSource::from_memory(Some("model.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
    };

    let graph_hits = session.search("global_meta", &SessionLimits::default());
    assert!(graph_hits.iter().any(|hit| hit.kind == SearchKind::Graph));
    assert!(graph_hits.iter().any(|hit| {
      matches!(
          &hit.handle,
          EntityHandle::Metadata { owner, key }
              if owner == "model" && key == "global_meta"
      )
    }));

    let node_hits = session.search("node_meta_key", &SessionLimits::default());
    assert!(node_hits.iter().any(|hit| hit.kind == SearchKind::Node));

    let tensor_hits = session.search("weights.bin", &SessionLimits::default());
    assert_eq!(tensor_hits.len(), 1);
    assert_eq!(tensor_hits[0].kind, SearchKind::Tensor);
    assert_eq!(tensor_hits[0].handle, EntityHandle::Tensor { tensor: 1 });

    let function_hits = session.search("fn_attr", &SessionLimits::default());
    assert!(
      function_hits
        .iter()
        .any(|hit| hit.kind == SearchKind::Function)
    );

    let opset_hits = session.search("ai.onnx", &SessionLimits::default());
    assert!(opset_hits.iter().any(|hit| {
      matches!(
          &hit.handle,
          EntityHandle::OperatorSet {
              domain: Some(domain),
              version: 13
          } if domain == "ai.onnx"
      )
    }));
  }

  #[test]
  fn onnx_tensor_metadata_preserves_external_descriptors() {
    let model = fixture_onnx_model();
    let index = OnnxIndex::build(&model);
    let tensors = index.tensor_metadata(10);
    let external = tensors
      .iter()
      .find(|tensor| tensor.name.as_deref() == Some("external_weight"))
      .expect("external tensor metadata");

    assert_eq!(external.handle, EntityHandle::Tensor { tensor: 1 });
    assert_eq!(external.storage, TensorStorageKind::External);
    assert_eq!(
      external.external_data.get("location").map(String::as_str),
      Some("weights.bin")
    );
    assert_eq!(external.element_type, "float32");
    assert_eq!(external.byte_len, None);

    let session = ModelSession {
      id: 102,
      source: ModelSource::from_memory(Some("model.onnx".to_owned()), 0),
      index: FormatIndex::Onnx(index),
      model,
    };
    let bounded = session.tensor_metadata(&SessionLimits {
      detail: 1,
      ..SessionLimits::default()
    });
    assert_eq!(bounded.len(), 1);
  }

  #[test]
  fn onnx_detail_returns_selected_entity_fields() {
    let model = fixture_onnx_model();
    let session = ModelSession {
      id: 103,
      source: ModelSource::from_memory(Some("model.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
    };

    let node = session
      .detail(
        &EntityHandle::Node { graph: 0, node: 0 },
        &SessionLimits::default(),
      )
      .expect("node detail");
    assert_eq!(node.title, "node_main");
    assert_eq!(
      node.fields.get("domain").map(String::as_str),
      Some("custom")
    );
    assert!(
      node
        .related
        .iter()
        .any(|handle| handle == &EntityHandle::Value { graph: 0, value: 0 })
    );

    let tensor = session
      .detail(
        &EntityHandle::Tensor { tensor: 1 },
        &SessionLimits::default(),
      )
      .expect("tensor detail");
    assert_eq!(
      tensor.fields.get("external.location").map(String::as_str),
      Some("weights.bin")
    );

    let metadata = session
      .detail(
        &EntityHandle::Metadata {
          owner: "model".to_owned(),
          key: "global_meta".to_owned(),
        },
        &SessionLimits::default(),
      )
      .expect("metadata detail");
    assert_eq!(
      metadata.fields.get("value").map(String::as_str),
      Some("present")
    );
  }

  #[test]
  fn onnx_diagnostics_report_unsupported_attributes() {
    let model = fixture_onnx_model();
    let session = ModelSession {
      id: 101,
      source: ModelSource::from_memory(Some("model.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
    };

    let diagnostics = session.diagnostics(&SessionLimits::default()).diagnostics;
    assert_eq!(diagnostics.len(), 2);
    assert!(
      diagnostics
        .iter()
        .all(|item| item.code == "onnx.unsupported_attribute")
    );
    assert!(matches!(
      diagnostics[0].handle.as_ref(),
      Some(EntityHandle::Diagnostic { diagnostic: 0 })
    ));

    let detail = session
      .detail(
        &EntityHandle::Diagnostic { diagnostic: 0 },
        &SessionLimits::default(),
      )
      .expect("diagnostic detail");
    assert_eq!(detail.title, "onnx.unsupported_attribute");
    assert_eq!(
      detail.fields.get("kind").map(String::as_str),
      Some("warning")
    );
  }

  fn fixture_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "test",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let input = graph.add_value(Value::new(model.intern("input")));
    let output = graph.add_value(Value::new(model.intern("identity_output")));
    let exact = graph.add_value(Value::new(model.intern("Add")));
    graph.values[output.index()].producer = Some(netron_rs_core::NodeId::new(0));
    graph.values[exact.index()].producer = Some(netron_rs_core::NodeId::new(0));
    let mut node = Node::new(
      graph_id,
      Operator {
        domain: None,
        name: model.intern("Add"),
        overload: None,
        version: None,
        origin: "test",
      },
    );
    node.inputs.push(Some(input));
    node.outputs.push(Some(output));
    node.outputs.push(Some(exact));
    let node_id = graph.add_node(node);
    graph.values[input.index()].consumers.push(node_id);
    model.replace_graph(graph_id, graph);
    model
  }

  fn histogram_count(entries: &[HistogramEntry], key: &str) -> usize {
    entries
      .iter()
      .find(|entry| entry.key == key)
      .map(|entry| entry.count)
      .unwrap_or(0)
  }

  fn fixture_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });

    model.metadata.producer = Some("query-tests".to_owned());
    model
      .metadata
      .properties
      .insert("global_meta".to_owned(), "present".to_owned());
    model.metadata.opsets.push(netron_rs_core::OperatorSet {
      domain: Some("ai.onnx".to_owned()),
      version: 13,
    });

    let inline_weight_name = model.intern("inline_weight");
    let external_weight_name = model.intern("external_weight");
    let sparse_weight_name = model.intern("sparse_weight");
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));

    let inline_tensor_id = model.add_tensor(Tensor::metadata_only(
      Some(inline_weight_name),
      TensorElementType::Float32,
      Vec::new(),
      TensorStorage::InlineBytes { byte_len: 16 },
    ));

    let mut external_entries = BTreeMap::new();
    external_entries.insert("location".to_owned(), "weights.bin".to_owned());
    let external_tensor_id = model.add_tensor(Tensor::metadata_only(
      Some(external_weight_name),
      TensorElementType::Float32,
      Vec::new(),
      TensorStorage::External {
        entries: external_entries,
      },
    ));

    model.add_tensor(Tensor::metadata_only(
      Some(sparse_weight_name),
      TensorElementType::Float32,
      Vec::new(),
      TensorStorage::Sparse {
        values: inline_tensor_id,
        indices: inline_tensor_id,
      },
    ));

    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    graph
      .metadata
      .insert("graph_meta".to_owned(), "present".to_owned());

    let input = graph.add_value(Value::new(model.intern("input")));
    let weight = graph.add_value(Value::new(model.intern("weight")));
    let output = graph.add_value(Value::new(model.intern("output")));
    graph.values[weight.index()].initializer = Some(external_tensor_id);
    graph.values[weight.index()]
      .metadata
      .insert("value_meta".to_owned(), "set".to_owned());

    let mut node = Node::new(
      graph_id,
      Operator {
        domain: Some(model.intern("custom")),
        name: model.intern("MyNode"),
        overload: None,
        version: Some(1),
        origin: "onnx",
      },
    );
    node.name = Some(model.intern("node_main"));
    node
      .metadata
      .insert("node_meta_key".to_owned(), "present".to_owned());
    node.inputs.push(Some(input));
    node.outputs.push(Some(weight));
    node.outputs.push(Some(output));
    node.attributes.push(Attribute {
      name: model.intern("bad_graph_attr"),
      value: AttributeValue::Unsupported("not supported".to_owned()),
    });
    let node_id = graph.add_node(node);
    graph.values[weight.index()].producer = Some(node_id);
    graph.values[output.index()].producer = Some(node_id);
    graph.values[input.index()].consumers.push(node_id);

    let function_name = model.intern("my_function");
    let mut function_node_metadata = BTreeMap::new();
    function_node_metadata.insert("f_node_attr".to_owned(), "present".to_owned());
    let function = Function {
      name: function_name,
      domain: Some(model.intern("fn_domain")),
      overload: Some(model.intern("v1")),
      description: None,
      metadata: {
        let mut metadata = BTreeMap::new();
        metadata.insert("function_meta".to_owned(), "present".to_owned());
        metadata
      },
      opsets: Vec::new(),
      inputs: vec![model.intern("fn_input")],
      outputs: vec![model.intern("fn_output")],
      attributes: vec![model.intern("fn_attr")],
      values: vec![FunctionValue {
        name: model.intern("fn_weight"),
        type_info: None,
        initializer: Some(external_tensor_id),
      }],
      nodes: vec![FunctionNode {
        name: Some(model.intern("fn_node")),
        description: None,
        metadata: function_node_metadata,
        operator: Operator {
          domain: Some(model.intern("fn_domain")),
          name: model.intern("FnOp"),
          overload: None,
          version: None,
          origin: "onnx-function",
        },
        inputs: vec![Some(model.intern("fn_input"))],
        outputs: vec![Some(model.intern("fn_output"))],
        attributes: vec![Attribute {
          name: model.intern("f_node_attr"),
          value: AttributeValue::Unsupported("unsupported".to_owned()),
        }],
      }],
    };
    model.add_function(function);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_two_node_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let input = graph.add_value(Value::new(model.intern("input")));
    let hidden = graph.add_value(Value::new(model.intern("hidden")));
    let output = graph.add_value(Value::new(model.intern("output")));
    graph.inputs.push(input);
    graph.outputs.push(output);
    graph.values[input.index()].is_graph_input = true;
    graph.values[output.index()].is_graph_output = true;

    let first = add_fixture_node(&mut model, &mut graph, "Relu", &[input], &[hidden]);
    graph.values[input.index()].consumers.push(first);
    graph.values[hidden.index()].producer = Some(first);
    let second = add_fixture_node(&mut model, &mut graph, "Add", &[hidden], &[output]);
    graph.values[hidden.index()].consumers.push(second);
    graph.values[output.index()].producer = Some(second);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_initializer_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let input = graph.add_value(Value::new(model.intern("input")));
    let weight = graph.add_value(Value::new(model.intern("weight")));
    let output = graph.add_value(Value::new(model.intern("output")));
    graph.inputs.push(input);
    graph.outputs.push(output);
    graph.values[input.index()].is_graph_input = true;
    graph.values[weight.index()].initializer = Some(netron_rs_core::TensorId::new(0));
    graph.values[output.index()].is_graph_output = true;
    let weight_name = model.intern("weight");

    let node = add_fixture_node(&mut model, &mut graph, "Add", &[input, weight], &[output]);
    graph.values[input.index()].consumers.push(node);
    graph.values[weight.index()].consumers.push(node);
    graph.values[output.index()].producer = Some(node);
    model.tensors.push(Tensor {
      id: netron_rs_core::TensorId::new(0),
      name: Some(weight_name),
      description: None,
      metadata: BTreeMap::new(),
      element_type: TensorElementType::Int64,
      shape: vec![Dimension {
        value: DimensionValue::Known(1),
        denotation: None,
      }],
      storage: TensorStorage::ElementList { len: 1 },
      quantization: None,
    });

    model.replace_graph(graph_id, graph);
    model
  }

  fn mlirbc_fixture(name: &str) -> PathBuf {
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("../../tests/fixtures/mlir")
      .join(name);
    if local.exists() {
      return local;
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("../../../netron/third_party/test/mlir")
      .join(name)
  }

  fn add_fixture_node(
    model: &mut netron_rs_core::Model,
    graph: &mut Graph,
    operator: &str,
    inputs: &[netron_rs_core::ValueId],
    outputs: &[netron_rs_core::ValueId],
  ) -> netron_rs_core::NodeId {
    let mut node = Node::new(
      graph.id,
      Operator {
        domain: None,
        name: model.intern(operator),
        overload: None,
        version: None,
        origin: "test",
      },
    );
    node.inputs = inputs.iter().copied().map(Some).collect();
    node.outputs = outputs.iter().copied().map(Some).collect();
    graph.add_node(node)
  }
}
