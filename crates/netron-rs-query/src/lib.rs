use std::{
  cell::RefCell,
  collections::{BTreeMap, BTreeSet, VecDeque},
  path::{Path, PathBuf},
  sync::atomic::{AtomicU64, Ordering},
};

use netron_rs_core::{
  Attribute, AttributeValue, Dimension, DimensionValue, Function, FunctionNode, FunctionValue,
  Model, ModelError, ModelInput, Node, NormalizedModel, Operator, Tensor, TensorElementType,
  TensorStorage, ToNormalizedJson, TypeInfo, Value,
};
use netron_rs_layout::{
  CancellationToken, LayoutBounds, LayoutEdge, LayoutError, LayoutGraph, LayoutNode,
  LayoutNodeKind, LayoutOptions, LayoutPoint, LayoutStats,
};
use serde::Serialize;

pub const SESSION_API_VERSION: u32 = 1;
pub const DEFAULT_NODE_SLICE_DEPTH: usize = 1;
pub const MAX_NODE_SLICE_DEPTH: usize = 16;
const ONNX_REPEATED_BLOCK_MAX_WIDTH: usize = 8;
const ONNX_REPEATED_BLOCK_MAX_CONSUMER_SUBSETS: usize = 32;

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CollapseMode {
  #[default]
  None,
  Structural,
}

impl CollapseMode {
  pub fn parse(value: &str) -> Option<Self> {
    match value {
      "none" => Some(Self::None),
      "structural" => Some(Self::Structural),
      _ => None,
    }
  }
}

#[derive(Debug, Clone)]
pub struct ProjectionOptions {
  pub collapse: CollapseMode,
  pub node_depth: usize,
  pub cancel: Option<CancellationToken>,
}

impl Default for ProjectionOptions {
  fn default() -> Self {
    Self {
      collapse: CollapseMode::None,
      node_depth: DEFAULT_NODE_SLICE_DEPTH,
      cancel: None,
    }
  }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProjectionError {
  Canceled,
}

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

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub format: FormatKind,
  pub query: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub cursor: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub next_cursor: Option<usize>,
  pub limit_used: usize,
  pub total_count: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub results: Vec<SearchEntry>,
}

#[derive(Debug, Serialize)]
pub struct SessionExport<'a> {
  pub api_version: u32,
  pub session_id: u64,
  pub format: FormatKind,
  pub limit_used: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub normalized: NormalizedModel<'a>,
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
  OnnxRepeatedBlock {
    graph: usize,
    group: usize,
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

#[allow(clippy::large_enum_variant)]
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
  pub locations: Vec<DetailLocation>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub related: Vec<EntityHandle>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct DetailLocation {
  pub kind: String,
  pub raw: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub file: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub line: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub column: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub end_line: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub end_column: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub index: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub region: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub block: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub format: FormatKind,
  pub scope: EntityHandle,
  pub collapse: CollapseMode,
  pub cache_key: String,
  pub limit_used: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub warnings: Vec<String>,
  pub entities: Vec<SliceEntity>,
  pub edges: Vec<SliceEdge>,
  pub boundaries: Vec<SliceBoundary>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub collapsed_groups: Vec<CollapsedGroup>,
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
pub struct CollapsedGroup {
  pub id: String,
  pub kind: String,
  pub handle: EntityHandle,
  pub label: Option<String>,
  pub item_count: usize,
  pub omitted_count: usize,
  pub expand: EntityHandle,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub format: FormatKind,
  pub scope: EntityHandle,
  pub collapse: CollapseMode,
  pub cache_key: String,
  pub limit_used: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub warnings: Vec<String>,
  pub graph: LayoutGraph,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub collapsed_groups: Vec<CollapsedGroup>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirSymbolEntry {
  pub handle: EntityHandle,
  pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirSymbolTree {
  pub api_version: u32,
  pub limit_used: usize,
  pub truncated: bool,
  pub omitted_count: usize,
  pub roots: Vec<MlirSymbolTreeNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirSymbolTreeNode {
  pub id: String,
  pub kind: String,
  pub label: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub handle: Option<EntityHandle>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub scope_id: Option<String>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub children: Vec<MlirSymbolTreeNode>,
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
  pub resources: Vec<MlirResourceSummary>,
  pub dialects: Vec<String>,
  pub histograms: MlirHistograms,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub bytecode: Option<MlirBytecodeSummary>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MlirHistograms {
  pub operations: Vec<HistogramEntry>,
  pub dialects: Vec<HistogramEntry>,
  pub fan_in: Vec<HistogramEntry>,
  pub fan_out: Vec<HistogramEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirResourceSummary {
  pub handle: EntityHandle,
  pub scope: String,
  pub name: String,
  pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirBytecodeSummary {
  pub version: u64,
  pub producer: String,
  pub string_count: usize,
  pub operation_name_count: usize,
  pub decoded_location_count: usize,
  pub attribute_count: usize,
  pub attributes: Vec<MlirBytecodeAttributeSummary>,
  pub type_count: usize,
  pub types: Vec<MlirBytecodeTypeSummary>,
  pub property_count: usize,
  pub ir_truncated: bool,
  pub sections: Vec<MlirBytecodeSectionSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirBytecodeTypeSummary {
  pub index: usize,
  pub dialect: String,
  pub has_custom_encoding: bool,
  pub len: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub assembly: Option<String>,
  #[serde(skip_serializing_if = "String::is_empty")]
  pub preview_hex: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirBytecodeAttributeSummary {
  pub index: usize,
  pub dialect: String,
  pub has_custom_encoding: bool,
  pub len: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub assembly: Option<String>,
  #[serde(skip_serializing_if = "String::is_empty")]
  pub preview_hex: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirBytecodeSectionSummary {
  pub id: u8,
  pub len: usize,
  pub alignment: Option<usize>,
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
  #[serde(skip_serializing_if = "Option::is_none")]
  pub label: Option<String>,
  pub block_count: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub parent_operation: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub parent_region: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub parent_block: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlirBlockSummary {
  pub handle: EntityHandle,
  pub scope_id: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub label: Option<String>,
  pub region: usize,
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

  pub fn search_page(&self, query: &str, cursor: usize, limit: usize) -> (Vec<SearchEntry>, usize) {
    search_index_page(&self.entries, query, cursor, limit)
  }

  pub fn tensor_metadata(&self, limit: usize) -> Vec<TensorMetadata> {
    self.tensor_metadata.iter().take(limit).cloned().collect()
  }

  pub fn tensor_metadata_by_id(&self, tensor: usize) -> Option<TensorMetadata> {
    self.tensor_metadata.get(tensor).cloned()
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
  bytecode: Option<netron_rs_formats::MlirBytecodeSummary>,
}

impl MlirIndex {
  pub fn build(
    model: &Model,
    source_text: Option<&str>,
    bytecode: Option<netron_rs_formats::MlirBytecodeSummary>,
  ) -> Self {
    let mut entries = Vec::new();
    let mut modules = Vec::new();
    let mut functions = Vec::new();
    let mut regions = Vec::new();
    let mut blocks = Vec::new();
    let mut dialects = BTreeSet::new();
    let mut histograms = MlirHistogramMaps::default();
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
      if bytecode.is_none() && key.starts_with("bytecode.diagnostic.") {
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
          &mut histograms,
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
        terms.extend(value.metadata.keys().cloned());
        terms.extend(value.metadata.values().cloned());
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
          &mut histograms,
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

    if let Some(bytecode) = &bytecode {
      index_bytecode_attributes(
        &mut entries,
        &mut attributes,
        &mut attribute_count,
        bytecode,
      );
      for resource in &bytecode.resources {
        let resource = MlirResourceDetail {
          scope: resource.scope.clone(),
          name: resource.name.clone(),
          kind: resource.kind.clone(),
        };
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
    if let Some(bytecode) = &bytecode {
      for dialect in &bytecode.dialects {
        dialects.insert(dialect.clone());
      }
      for diagnostic in &bytecode.diagnostics {
        diagnostics.push(Diagnostic {
          handle: None,
          kind: DiagnosticKind::Warning,
          code: "mlir.bytecode",
          message: diagnostic.clone(),
          source: Some("bytecode".to_owned()),
        });
      }
      module_count = bytecode.ir.module_count;
      function_count = bytecode.ir.function_count;
      operation_count = bytecode.ir.operation_count;
      value_count = bytecode.ir.value_count;
      block_argument_count = bytecode.ir.block_argument_count;
      region_count = bytecode.ir.region_count;
      block_count = bytecode.ir.block_count;
      summary_attribute_count = bytecode.attribute_count;
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
      let bytecode_scope = if !model.functions.is_empty() {
        mlir_function_scope(0)
      } else {
        mlir_module_scope(0)
      };
      let operation_symbols = bytecode_operation_symbol_map(model);
      replace_bytecode_region_block_summaries(
        &mut entries,
        &mut regions,
        &mut blocks,
        &bytecode_scope,
        bytecode,
        &operation_symbols,
      );
    }
    let bytecode_summary = bytecode.as_ref().map(mlir_bytecode_summary);
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
      resources: mlir_resource_summaries(&resources),
      dialects: dialects.into_iter().collect(),
      histograms: histograms.into_histograms(),
      bytecode: bytecode_summary,
    };
    assign_diagnostic_handles(&mut diagnostics);

    Self {
      entries,
      summary,
      symbols,
      attributes,
      resources,
      diagnostics,
      bytecode,
    }
  }

  pub fn search(&self, query: &str, limit: usize) -> Vec<SearchEntry> {
    search_index_entries(&self.entries, query, limit)
  }

  pub fn search_page(&self, query: &str, cursor: usize, limit: usize) -> (Vec<SearchEntry>, usize) {
    search_index_page(&self.entries, query, cursor, limit)
  }

  pub fn detail(&self, model: &Model, handle: &EntityHandle, limit: usize) -> Option<EntityDetail> {
    mlir_detail(model, self, handle, limit)
  }

  pub fn bytecode(&self) -> Option<&netron_rs_formats::MlirBytecodeSummary> {
    self.bytecode.as_ref()
  }
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
    label: Some(format!("{scope} region 0")),
    block_count: 1,
    parent_operation: None,
    parent_region: None,
    parent_block: None,
  };
  let block = MlirBlockSummary {
    handle: EntityHandle::MlirBlock {
      scope: scope.to_owned(),
      block: 0,
    },
    scope_id: scope.to_owned(),
    label: Some(mlir_block_label(0)),
    region: 0,
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

fn replace_bytecode_region_block_summaries(
  entries: &mut Vec<SearchEntry>,
  regions: &mut Vec<MlirRegionSummary>,
  blocks: &mut Vec<MlirBlockSummary>,
  scope: &str,
  bytecode: &netron_rs_formats::MlirBytecodeSummary,
  operation_symbols: &BTreeMap<usize, String>,
) {
  entries.retain(|entry| !matches!(entry.kind, SearchKind::Region | SearchKind::Block));
  regions.clear();
  blocks.clear();

  let region_count = bytecode.ir.region_count.max(1);
  let block_count = bytecode.ir.block_count.max(1);
  let mut block_regions = vec![0usize; block_count];
  let mut region_blocks = vec![BTreeSet::new(); region_count];
  let mut block_operations = vec![0usize; block_count];
  let mut block_values = vec![BTreeSet::new(); block_count];
  let mut region_parents = vec![None; region_count];
  let mut region_labels = vec![None; region_count];

  for operation in &bytecode.ir.operations {
    if operation.region < region_count && operation.block < block_count {
      block_regions[operation.block] = operation.region;
      region_blocks[operation.region].insert(operation.block);
    }
    if operation.block < block_count {
      block_operations[operation.block] += 1;
      block_values[operation.block].extend(operation.operands.iter().copied());
      block_values[operation.block].extend(operation.results.iter().copied());
    }
    for (index, region) in operation.nested_region_ids.iter().enumerate() {
      if *region < region_count {
        region_parents[*region] = Some((operation.operation, operation.region, operation.block));
        region_labels[*region] = bytecode_operation_region_label(
          scope,
          operation.name.as_str(),
          operation_symbols
            .get(&operation.operation)
            .map(String::as_str),
          index,
        );
      }
    }
  }

  if region_blocks[0].is_empty() {
    region_blocks[0].insert(0);
  }

  for (region, block_set) in region_blocks.iter().enumerate() {
    let handle = EntityHandle::MlirRegion {
      scope: scope.to_owned(),
      region,
    };
    let (parent_operation, parent_region, parent_block) = region_parents[region]
      .map(|(operation, parent_region, parent_block)| {
        (Some(operation), Some(parent_region), Some(parent_block))
      })
      .unwrap_or((None, None, None));
    let summary = MlirRegionSummary {
      handle: handle.clone(),
      scope_id: scope.to_owned(),
      label: region_labels[region]
        .clone()
        .or_else(|| Some(format!("{scope} bytecode region {region}"))),
      block_count: block_set.len(),
      parent_operation,
      parent_region,
      parent_block,
    };
    let mut search_terms = vec![scope.to_owned(), "bytecode".to_owned(), "region".to_owned()];
    if let Some(label) = &summary.label {
      search_terms.push(label.clone());
    }
    if let Some(parent_operation) = parent_operation {
      search_terms.push(format!("parent_operation:{parent_operation}"));
    }
    entries.push(SearchEntry::new_with_handle_and_search_terms(
      SearchKind::Region,
      handle,
      None,
      regions.len(),
      summary.label.clone(),
      None,
      Some("MLIR"),
      search_terms,
    ));
    regions.push(summary);
  }

  for block in 0..block_count {
    let handle = EntityHandle::MlirBlock {
      scope: scope.to_owned(),
      block,
    };
    let label = mlir_block_label(block);
    let summary = MlirBlockSummary {
      handle: handle.clone(),
      scope_id: scope.to_owned(),
      label: Some(label.clone()),
      region: block_regions[block],
      operation_count: block_operations[block],
      value_count: block_values[block].len(),
      block_argument_count: 0,
    };
    let label_term = label.trim_start_matches('^').to_owned();
    entries.push(SearchEntry::new_with_handle_and_search_terms(
      SearchKind::Block,
      handle,
      None,
      blocks.len(),
      Some(format!(
        "{scope} bytecode region {} block {block}",
        block_regions[block]
      )),
      None,
      Some("MLIR"),
      [
        scope.to_owned(),
        "bytecode".to_owned(),
        "block".to_owned(),
        label,
        label_term,
        format!("region:{}", block_regions[block]),
      ],
    ));
    blocks.push(summary);
  }
}

fn mlir_block_label(block: usize) -> String {
  format!("^bb{block}")
}

fn bytecode_operation_region_label(
  scope: &str,
  operation: &str,
  symbol: Option<&str>,
  region_index: usize,
) -> Option<String> {
  let region_name = match (operation, region_index) {
    ("builtin.module" | "func.func", 0) => "body",
    _ => {
      return Some(match symbol {
        Some(symbol) => format!(
          "{scope} {operation} {} region {region_index}",
          bytecode_symbol_label(symbol)
        ),
        None => format!("{scope} {operation} region {region_index}"),
      });
    }
  };
  Some(match symbol {
    Some(symbol) => format!(
      "{scope} {operation} {}.{region_name}",
      bytecode_symbol_label(symbol)
    ),
    None => format!("{scope} {operation}.{region_name}"),
  })
}

fn bytecode_operation_symbol_map(model: &Model) -> BTreeMap<usize, String> {
  let mut symbols = BTreeMap::new();
  if let Some(function) = model.functions.first() {
    for node in &function.nodes {
      push_bytecode_operation_symbol(&mut symbols, &node.metadata);
    }
  } else if let Some(graph) = model.graphs.first() {
    for node in &graph.nodes {
      push_bytecode_operation_symbol(&mut symbols, &node.metadata);
    }
  }
  symbols
}

fn push_bytecode_operation_symbol(
  symbols: &mut BTreeMap<usize, String>,
  metadata: &BTreeMap<String, String>,
) {
  let Some(operation) = metadata
    .get("bytecode.operation")
    .and_then(|value| value.parse::<usize>().ok())
  else {
    return;
  };
  if let Some(symbol) = metadata
    .get("bytecode.symbol")
    .filter(|symbol| !symbol.is_empty())
  {
    symbols.insert(operation, symbol.clone());
  }
}

fn bytecode_symbol_label(symbol: &str) -> String {
  if symbol.starts_with('@') {
    symbol.to_owned()
  } else {
    format!("@{symbol}")
  }
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

const BYTECODE_INDEX_ATTRIBUTE_LIMIT: usize = 1_024;

fn index_bytecode_attributes(
  entries: &mut Vec<SearchEntry>,
  attribute_details: &mut Vec<MlirAttributeDetail>,
  attribute_count: &mut usize,
  bytecode: &netron_rs_formats::MlirBytecodeSummary,
) {
  let mut selected = BTreeSet::new();
  for operation in &bytecode.ir.operations {
    if let Some(attribute) = operation.attributes
      && bytecode
        .attributes
        .get(attribute)
        .and_then(|entry| entry.assembly.as_ref())
        .is_some()
    {
      selected.insert(attribute);
      if selected.len() >= BYTECODE_INDEX_ATTRIBUTE_LIMIT {
        break;
      }
    }
  }
  for attribute in &bytecode.attributes {
    if selected.len() >= BYTECODE_INDEX_ATTRIBUTE_LIMIT {
      break;
    }
    if attribute.assembly.is_some() {
      selected.insert(attribute.index);
    }
  }

  for index in selected {
    let Some(attribute) = bytecode.attributes.get(index) else {
      continue;
    };
    if attribute.index != index {
      continue;
    }
    let Some(assembly) = &attribute.assembly else {
      continue;
    };
    let name = format!("bytecode.attr.{index}");
    let values = vec![assembly.clone(), format!("dialect:{}", attribute.dialect)];
    let mut terms = values.clone();
    terms.extend([
      name.clone(),
      "bytecode".to_owned(),
      "mlirbc".to_owned(),
      format!("len:{}", attribute.len),
    ]);
    if attribute.has_custom_encoding {
      terms.push("custom_encoding:true".to_owned());
    }
    entries.push(mlir_attribute_entry(
      "bytecode",
      *attribute_count,
      &name,
      terms,
    ));
    attribute_details.push(MlirAttributeDetail {
      scope: "bytecode".to_owned(),
      name,
      values,
    });
    *attribute_count += 1;
  }
}

#[allow(clippy::too_many_arguments)]
fn index_mlir_operation(
  model: &Model,
  entries: &mut Vec<SearchEntry>,
  dialects: &mut BTreeSet<String>,
  histograms: &mut MlirHistogramMaps,
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
  histograms.record(&operator_name, &dialect, fan_in, fan_out);
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
    if key == "bytecode.symbol" {
      insert_mlir_symbol(entries, symbols, symbol_names, scope, value);
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
      label: Some(label),
      region: 0,
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

fn mlir_resource_summaries(resources: &[MlirResourceDetail]) -> Vec<MlirResourceSummary> {
  resources
    .iter()
    .enumerate()
    .map(|(resource, detail)| MlirResourceSummary {
      handle: EntityHandle::MlirResource { resource },
      scope: detail.scope.clone(),
      name: detail.name.clone(),
      kind: detail.kind.clone(),
    })
    .collect()
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
    .map_or("builtin", |(dialect, _)| dialect)
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

#[derive(Default)]
struct MlirHistogramMaps {
  operations: BTreeMap<String, usize>,
  dialects: BTreeMap<String, usize>,
  fan_in: BTreeMap<String, usize>,
  fan_out: BTreeMap<String, usize>,
}

impl MlirHistogramMaps {
  fn record(&mut self, operation: &str, dialect: &str, inputs: usize, outputs: usize) {
    increment(&mut self.operations, operation.to_owned());
    increment(&mut self.dialects, dialect.to_owned());
    increment(&mut self.fan_in, inputs.to_string());
    increment(&mut self.fan_out, outputs.to_string());
  }

  fn into_histograms(self) -> MlirHistograms {
    MlirHistograms {
      operations: histogram_entries(self.operations),
      dialects: histogram_entries(self.dialects),
      fan_in: histogram_entries(self.fan_in),
      fan_out: histogram_entries(self.fan_out),
    }
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

fn limit_mlir_histograms(histograms: &MlirHistograms, limit: usize) -> MlirHistograms {
  MlirHistograms {
    operations: limit_histogram_entries(&histograms.operations, limit),
    dialects: limit_histogram_entries(&histograms.dialects, limit),
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
    resources: summary.resources.iter().take(limit).cloned().collect(),
    dialects: summary.dialects.iter().take(limit).cloned().collect(),
    histograms: limit_mlir_histograms(&summary.histograms, limit),
    bytecode: summary
      .bytecode
      .as_ref()
      .map(|bytecode| limit_mlir_bytecode_summary(bytecode, limit)),
  }
}

fn mlir_bytecode_summary(bytecode: &netron_rs_formats::MlirBytecodeSummary) -> MlirBytecodeSummary {
  MlirBytecodeSummary {
    version: bytecode.version,
    producer: bytecode.producer.clone(),
    string_count: bytecode.string_count,
    operation_name_count: bytecode.operation_names.len(),
    decoded_location_count: bytecode.locations.len(),
    attribute_count: bytecode.attribute_count,
    attributes: bytecode
      .attributes
      .iter()
      .map(|entry| MlirBytecodeAttributeSummary {
        index: entry.index,
        dialect: entry.dialect.clone(),
        has_custom_encoding: entry.has_custom_encoding,
        len: entry.len,
        assembly: entry.assembly.clone(),
        preview_hex: entry.preview_hex.clone(),
      })
      .collect(),
    type_count: bytecode.type_count,
    types: bytecode
      .types
      .iter()
      .map(|entry| MlirBytecodeTypeSummary {
        index: entry.index,
        dialect: entry.dialect.clone(),
        has_custom_encoding: entry.has_custom_encoding,
        len: entry.len,
        assembly: entry.assembly.clone(),
        preview_hex: entry.preview_hex.clone(),
      })
      .collect(),
    property_count: bytecode.property_count,
    ir_truncated: bytecode.ir.truncated,
    sections: bytecode
      .sections
      .iter()
      .map(|section| MlirBytecodeSectionSummary {
        id: section.id,
        len: section.len,
        alignment: section.alignment,
      })
      .collect(),
  }
}

fn limit_mlir_bytecode_summary(summary: &MlirBytecodeSummary, limit: usize) -> MlirBytecodeSummary {
  MlirBytecodeSummary {
    version: summary.version,
    producer: summary.producer.clone(),
    string_count: summary.string_count,
    operation_name_count: summary.operation_name_count,
    decoded_location_count: summary.decoded_location_count,
    attribute_count: summary.attribute_count,
    attributes: summary.attributes.iter().take(limit).cloned().collect(),
    type_count: summary.type_count,
    types: summary.types.iter().take(limit).cloned().collect(),
    property_count: summary.property_count,
    ir_truncated: summary.ir_truncated,
    sections: summary.sections.iter().take(limit).cloned().collect(),
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
  operator.domain.map_or_else(
    || "default".to_owned(),
    |domain| model.strings.get(domain).to_owned(),
  )
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
    EntityHandle::OnnxRepeatedBlock { graph, group } => {
      onnx_repeated_block_detail(model, *graph, *group, limit)
    }
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
    | EntityHandle::OnnxRepeatedBlock { .. }
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
    title: graph.name.map_or_else(
      || format!("module {module_index}"),
      |id| model.strings.get(id).to_owned(),
    ),
    fields,
    locations: Vec::new(),
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
    locations: Vec::new(),
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
  let locations = mlir_detail_locations(&fields);
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
  push_mlir_bytecode_operation_related(&mut related, limit, scope, &node.metadata);
  EntityDetail {
    handle: EntityHandle::MlirOperation {
      scope: scope.to_owned(),
      operation: operation_index,
    },
    title: mlir_node_title(model, node.name, &node.operator),
    fields,
    locations,
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
  let locations = mlir_detail_locations(&fields);
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
  push_mlir_bytecode_operation_related(&mut related, limit, scope, &node.metadata);
  EntityDetail {
    handle: EntityHandle::MlirOperation {
      scope: scope.to_owned(),
      operation: operation_index,
    },
    title: mlir_node_title(model, node.name, &node.operator),
    fields,
    locations,
    related,
  }
}

fn push_mlir_bytecode_operation_related(
  related: &mut Vec<EntityHandle>,
  limit: usize,
  scope: &str,
  metadata: &BTreeMap<String, String>,
) {
  if let (Some(region), Some(block)) = (
    metadata_usize(metadata, "bytecode.region"),
    metadata_usize(metadata, "bytecode.block"),
  ) {
    push_related(
      related,
      limit,
      EntityHandle::MlirRegion {
        scope: scope.to_owned(),
        region,
      },
    );
    push_related(
      related,
      limit,
      EntityHandle::MlirBlock {
        scope: scope.to_owned(),
        block,
      },
    );
  }
  for region in metadata_usize_list(metadata, "bytecode.nested_region_ids") {
    push_related(
      related,
      limit,
      EntityHandle::MlirRegion {
        scope: scope.to_owned(),
        region,
      },
    );
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
    locations: Vec::new(),
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
  add_metadata_fields(&mut fields, &value.metadata);
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
    locations: Vec::new(),
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
  if let Some(label) = &summary.label {
    fields.insert("label".to_owned(), label.clone());
  }
  fields.insert("block_count".to_owned(), summary.block_count.to_string());
  if let Some(parent_operation) = summary.parent_operation {
    fields.insert("parent_operation".to_owned(), parent_operation.to_string());
  }
  if let Some(parent_region) = summary.parent_region {
    fields.insert("parent_region".to_owned(), parent_region.to_string());
  }
  if let Some(parent_block) = summary.parent_block {
    fields.insert("parent_block".to_owned(), parent_block.to_string());
  }
  let mut related = Vec::new();
  if let Some(parent_operation) = summary.parent_operation {
    push_related(
      &mut related,
      limit,
      EntityHandle::MlirOperation {
        scope: scope.to_owned(),
        operation: parent_operation,
      },
    );
  }
  for block in index
    .summary
    .blocks
    .iter()
    .filter(|block| block.scope_id == summary.scope_id && block.region == region)
  {
    push_related(&mut related, limit, block.handle.clone());
  }
  Some(EntityDetail {
    handle,
    title: summary
      .label
      .clone()
      .unwrap_or_else(|| format!("{scope} region {region}")),
    fields,
    locations: Vec::new(),
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
  if let Some(label) = &summary.label {
    fields.insert("label".to_owned(), label.clone());
  }
  fields.insert("region".to_owned(), summary.region.to_string());
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
    .filter(|region| {
      region.scope_id == summary.scope_id
        && matches!(&region.handle, EntityHandle::MlirRegion { region, .. } if *region == summary.region)
    })
  {
    push_related(&mut related, limit, region.handle.clone());
  }
  Some(EntityDetail {
    handle,
    title: format!("{scope} block {block}"),
    fields,
    locations: Vec::new(),
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
    locations: Vec::new(),
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
    locations: Vec::new(),
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
  let locations = mlir_attribute_locations(&detail.values);

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
    locations,
    related,
  })
}

fn mlir_attribute_locations(values: &[String]) -> Vec<DetailLocation> {
  let mut locations = Vec::new();
  for value in values {
    if !value.trim().starts_with("loc(") {
      continue;
    }
    if let Some(location) = mlir_source_location(value)
      && !locations.contains(&location)
    {
      locations.push(location);
    }
  }
  locations
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
    locations: Vec::new(),
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

fn mlir_detail_locations(fields: &BTreeMap<String, String>) -> Vec<DetailLocation> {
  let mut locations = Vec::new();
  for key in ["metadata.location", "location"] {
    if let Some(location) = fields.get(key).and_then(|raw| mlir_source_location(raw)) {
      locations.push(location);
      break;
    }
  }
  if let Some(location) = mlir_bytecode_source_location(fields) {
    locations.push(location);
  }
  if let Some(location) = mlir_bytecode_location(fields) {
    locations.push(location);
  }
  locations
}

fn mlir_source_location(raw: &str) -> Option<DetailLocation> {
  let raw = raw.trim();
  if raw.is_empty() {
    return None;
  }
  let target = raw
    .strip_prefix("loc(")
    .and_then(|value| value.strip_suffix(')'))
    .unwrap_or(raw)
    .trim();
  let location = split_source_location_range(target);
  Some(DetailLocation {
    kind: "source".to_owned(),
    raw: raw.to_owned(),
    file: location.file,
    line: location.line,
    column: location.column,
    end_line: location.end_line,
    end_column: location.end_column,
    index: None,
    region: None,
    block: None,
  })
}

fn mlir_bytecode_location(fields: &BTreeMap<String, String>) -> Option<DetailLocation> {
  let raw = fields.get("metadata.bytecode.location")?.trim();
  if raw.is_empty() {
    return None;
  }
  Some(DetailLocation {
    kind: "bytecode".to_owned(),
    raw: raw.to_owned(),
    file: None,
    line: None,
    column: None,
    end_line: None,
    end_column: None,
    index: raw.parse().ok(),
    region: fields
      .get("metadata.bytecode.region")
      .and_then(|value| value.parse().ok()),
    block: fields
      .get("metadata.bytecode.block")
      .and_then(|value| value.parse().ok()),
  })
}

fn mlir_bytecode_source_location(fields: &BTreeMap<String, String>) -> Option<DetailLocation> {
  let file = fields.get("metadata.bytecode.location.file").cloned();
  let line: Option<usize> = fields
    .get("metadata.bytecode.location.line")
    .and_then(|value| value.parse().ok());
  let column: Option<usize> = fields
    .get("metadata.bytecode.location.column")
    .and_then(|value| value.parse().ok());
  let end_line: Option<usize> = fields
    .get("metadata.bytecode.location.end_line")
    .and_then(|value| value.parse().ok());
  let end_column: Option<usize> = fields
    .get("metadata.bytecode.location.end_column")
    .and_then(|value| value.parse().ok());
  if file.is_none() && line.is_none() && column.is_none() {
    return None;
  }
  let mut raw = match (&file, line, column) {
    (Some(file), Some(line), Some(column)) => format!("{file}:{line}:{column}"),
    (Some(file), Some(line), None) => format!("{file}:{line}"),
    (Some(file), None, _) => file.clone(),
    (None, Some(line), Some(column)) => format!("{line}:{column}"),
    (None, Some(line), None) => line.to_string(),
    (None, None, Some(column)) => format!(":{column}"),
    (None, None, None) => return None,
  };
  append_location_range(&mut raw, line, end_line, end_column);
  Some(DetailLocation {
    kind: "source".to_owned(),
    raw,
    file,
    line,
    column,
    end_line,
    end_column,
    index: fields
      .get("metadata.bytecode.location")
      .and_then(|value| value.parse().ok()),
    region: fields
      .get("metadata.bytecode.region")
      .and_then(|value| value.parse().ok()),
    block: fields
      .get("metadata.bytecode.block")
      .and_then(|value| value.parse().ok()),
  })
}

struct SourceLocationParts {
  file: Option<String>,
  line: Option<usize>,
  column: Option<usize>,
  end_line: Option<usize>,
  end_column: Option<usize>,
}

fn split_source_location_range(target: &str) -> SourceLocationParts {
  let Some((start, end)) = target.split_once(" to ") else {
    let (file, line, column) = split_source_location(target);
    return SourceLocationParts {
      file,
      line,
      column,
      end_line: None,
      end_column: None,
    };
  };
  let (file, line, column) = split_source_location(start);
  let end = end.trim();
  if end.chars().all(|value| value.is_ascii_digit()) {
    return SourceLocationParts {
      file,
      line,
      column,
      end_line: line,
      end_column: end.parse().ok(),
    };
  }
  let (_, parsed_line, parsed_column) = split_source_location(end);
  SourceLocationParts {
    file,
    line,
    column,
    end_line: parsed_line,
    end_column: parsed_column,
  }
}

fn append_location_range(
  raw: &mut String,
  line: Option<usize>,
  end_line: Option<usize>,
  end_column: Option<usize>,
) {
  match (end_line, end_column) {
    (Some(end_line), Some(end_column)) => {
      if Some(end_line) == line {
        raw.push_str(&format!(" to {end_column}"));
      } else {
        raw.push_str(&format!(" to {end_line}:{end_column}"));
      }
    }
    (Some(end_line), None) => raw.push_str(&format!(" to {end_line}")),
    (None, Some(end_column)) => raw.push_str(&format!(" to {end_column}")),
    (None, None) => {}
  }
}

fn split_source_location(target: &str) -> (Option<String>, Option<usize>, Option<usize>) {
  let target = target.trim();
  let mut file = target;
  let mut line = None;
  let mut column = None;
  if let Some((prefix, last)) = split_trailing_usize(target) {
    if let Some((file_prefix, middle)) = split_trailing_usize(prefix) {
      file = file_prefix.trim();
      line = Some(middle);
      column = Some(last);
    } else if prefix.trim().chars().all(|value| value.is_ascii_digit()) {
      file = "";
      line = prefix.trim().parse().ok();
      column = Some(last);
    } else {
      file = prefix.trim();
      line = Some(last);
    }
  }
  (unquote_location_file(file), line, column)
}

fn split_trailing_usize(value: &str) -> Option<(&str, usize)> {
  let (prefix, suffix) = value.rsplit_once(':')?;
  let suffix = suffix.trim();
  if suffix.is_empty() || !suffix.chars().all(|value| value.is_ascii_digit()) {
    return None;
  }
  Some((prefix, suffix.parse().ok()?))
}

fn unquote_location_file(value: &str) -> Option<String> {
  let value = value.trim();
  let value = value
    .strip_prefix('"')
    .and_then(|inner| inner.strip_suffix('"'))
    .unwrap_or(value)
    .trim();
  (!value.is_empty()).then(|| value.to_owned())
}

fn mlir_value_fields(
  model: &Model,
  name: &str,
  type_info: &Option<TypeInfo>,
) -> BTreeMap<String, String> {
  let mut fields = BTreeMap::from([("name".to_owned(), name.to_owned())]);
  if let Some(type_info) = type_info {
    if let Some(layout) = type_info.layout {
      fields.insert("layout".to_owned(), model.strings.get(layout).to_owned());
    }
    if let Some(denotation) = type_info.denotation {
      fields.insert(
        "denotation".to_owned(),
        model.strings.get(denotation).to_owned(),
      );
    }
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
  name.map_or_else(
    || model.strings.get(operator.name).to_owned(),
    |id| model.strings.get(id).to_owned(),
  )
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
    title: graph.name.map_or_else(
      || format!("graph {graph_index}"),
      |id| model.strings.get(id).to_owned(),
    ),
    fields,
    locations: Vec::new(),
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
    title: node.name.map_or_else(
      || model.strings.get(node.operator.name).to_owned(),
      |id| model.strings.get(id).to_owned(),
    ),
    fields,
    locations: Vec::new(),
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
    locations: Vec::new(),
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
    locations: Vec::new(),
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
    locations: Vec::new(),
    related: Vec::new(),
  })
}

fn onnx_repeated_block_detail(
  model: &Model,
  graph_index: usize,
  group_index: usize,
  limit: usize,
) -> Option<EntityDetail> {
  let groups = onnx_repeated_blocks(model, graph_index)?;
  let group = groups.get(group_index)?;
  let mut fields = BTreeMap::new();
  fields.insert("graph".to_owned(), graph_index.to_string());
  fields.insert("group".to_owned(), group_index.to_string());
  fields.insert("operator_path".to_owned(), group.operator_path.clone());
  fields.insert(
    "instance_count".to_owned(),
    group.instances.len().to_string(),
  );
  fields.insert("nodes_per_instance".to_owned(), group.width().to_string());
  fields.insert("node_count".to_owned(), group.item_count().to_string());

  let mut related = Vec::new();
  for instance in &group.instances {
    for node in &instance.nodes {
      push_related(
        &mut related,
        limit,
        EntityHandle::Node {
          graph: graph_index,
          node: *node,
        },
      );
    }
  }

  Some(EntityDetail {
    handle: EntityHandle::OnnxRepeatedBlock {
      graph: graph_index,
      group: group_index,
    },
    title: group.label(),
    fields,
    locations: Vec::new(),
    related,
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
    locations: Vec::new(),
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
    locations: Vec::new(),
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
    locations: Vec::new(),
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
  options: ProjectionOptions,
) -> Result<Option<SliceResponse>, ProjectionError> {
  check_projection_canceled(&options)?;
  let mut builder = SliceBuilder::new(limits.slice, options.cancel.clone());
  let clamped_node_depth = options.node_depth.min(MAX_NODE_SLICE_DEPTH);
  let built = match scope {
    EntityHandle::Graph { graph } if options.collapse == CollapseMode::Structural => {
      onnx_graph_structural_slice(model, *graph, &mut builder)
    }
    EntityHandle::Graph { graph } => onnx_graph_slice(model, *graph, &mut builder),
    EntityHandle::Node { graph, node } => {
      onnx_node_slice(model, *graph, *node, clamped_node_depth, &mut builder)
    }
    EntityHandle::Value { graph, value } => onnx_value_slice(model, *graph, *value, &mut builder),
    EntityHandle::Function { function } => onnx_function_slice(model, *function, &mut builder),
    EntityHandle::OnnxRepeatedBlock { graph, group } => {
      onnx_repeated_block_slice(model, *graph, *group, &mut builder)
    }
    EntityHandle::MlirFunction { function } if options.collapse == CollapseMode::Structural => {
      mlir_function_structural_slice(model, *function, &mut builder)
    }
    EntityHandle::MlirFunction { function } => mlir_function_slice(model, *function, &mut builder),
    EntityHandle::MlirOperation { scope, operation } => {
      mlir_operation_slice(model, scope, *operation, &mut builder)
    }
    EntityHandle::MlirRegion { scope, region } if options.collapse == CollapseMode::Structural => {
      let Some(index) = mlir_index(index) else {
        return Ok(None);
      };
      let built = mlir_structural_scope_slice(model, index, scope, &mut builder);
      if built.is_some() {
        builder.handle(
          EntityHandle::MlirRegion {
            scope: scope.clone(),
            region: *region,
          },
          "mlir_region",
          Some(format!("{scope} region {region}")),
          None,
          true,
        );
      }
      built
    }
    EntityHandle::MlirRegion { scope, region } => {
      let Some(index) = mlir_index(index) else {
        return Ok(None);
      };
      mlir_region_slice(model, index, scope, *region, &mut builder)
    }
    EntityHandle::MlirBlock { scope, block } if options.collapse == CollapseMode::Structural => {
      let Some(index) = mlir_index(index) else {
        return Ok(None);
      };
      let built = mlir_structural_scope_slice(model, index, scope, &mut builder);
      if built.is_some() {
        builder.handle(
          EntityHandle::MlirBlock {
            scope: scope.clone(),
            block: *block,
          },
          "mlir_block",
          Some(format!("{scope} block {block}")),
          None,
          true,
        );
      }
      built
    }
    EntityHandle::MlirBlock { scope, block } => {
      let Some(index) = mlir_index(index) else {
        return Ok(None);
      };
      mlir_block_slice(model, index, scope, *block, &mut builder)
    }
    EntityHandle::MlirSymbol { symbol } => {
      let Some(index) = mlir_index(index) else {
        return Ok(None);
      };
      mlir_symbol_slice(model, index, *symbol, &mut builder)
    }
    EntityHandle::MlirDialect { dialect } => mlir_dialect_slice(model, dialect, &mut builder),
    EntityHandle::MlirModule { module } if options.collapse == CollapseMode::Structural => {
      mlir_module_structural_slice(model, *module, &mut builder)
    }
    EntityHandle::MlirModule { module } => mlir_module_slice(model, *module, &mut builder),
    EntityHandle::Tensor { .. }
    | EntityHandle::Metadata { .. }
    | EntityHandle::OperatorSet { .. }
    | EntityHandle::Diagnostic { .. }
    | EntityHandle::MlirValue { .. }
    | EntityHandle::MlirAttribute { .. }
    | EntityHandle::MlirResource { .. } => return Ok(None),
  };
  if built.is_none() {
    return Ok(None);
  }
  builder.check_canceled()?;
  let truncated = builder.omitted_count > 0;
  let mut warnings = Vec::new();
  if truncated {
    warnings.push(format!(
      "slice truncated to {} visible entities",
      limits.slice
    ));
  }
  let node_depth_key = if matches!(scope, EntityHandle::Node { .. }) {
    if clamped_node_depth < options.node_depth {
      warnings.push(format!(
        "node slice depth clamped to {MAX_NODE_SLICE_DEPTH}"
      ));
    }
    Some(clamped_node_depth)
  } else {
    None
  };
  Ok(Some(SliceResponse {
    api_version: SESSION_API_VERSION,
    session_id,
    format: index.kind(),
    scope: scope.clone(),
    collapse: options.collapse,
    cache_key: slice_cache_key(
      session_id,
      index.kind(),
      scope,
      limits.slice,
      options.collapse,
      node_depth_key,
    ),
    limit_used: limits.slice,
    truncated,
    omitted_count: builder.omitted_count,
    warnings,
    entities: builder.entities,
    edges: builder.edges,
    boundaries: builder.boundaries,
    collapsed_groups: builder.collapsed_groups,
  }))
}

fn session_layout(
  model: &Model,
  index: &FormatIndex,
  session_id: u64,
  scope: &EntityHandle,
  limits: SessionLimits,
  options: ProjectionOptions,
) -> Result<Option<LayoutResponse>, ProjectionError> {
  check_projection_canceled(&options)?;
  let mut collapsed_groups = Vec::new();
  let graph = match scope {
    EntityHandle::Graph { graph } if options.collapse == CollapseMode::Structural => {
      let Some(groups) = onnx_repeated_blocks(model, *graph) else {
        return Ok(None);
      };
      collapsed_groups = onnx_repeated_block_collapsed_groups(*graph, &groups);
      onnx_graph_structural_layout(
        model,
        *graph,
        &groups,
        limits.layout,
        options.cancel.as_ref(),
      )?
    }
    EntityHandle::Graph { graph } => layout_graph_or_none(netron_rs_layout::layout_graph(
      model,
      &LayoutOptions {
        graph: *graph,
        max_nodes: Some(limits.layout),
        cancel: options.cancel.clone(),
        ..LayoutOptions::default()
      },
    ))?,
    EntityHandle::MlirFunction { function } if options.collapse == CollapseMode::Structural => {
      let Some(groups) = mlir_function_collapsed_groups(model, *function) else {
        return Ok(None);
      };
      collapsed_groups = groups;
      mlir_function_structural_layout(model, *function, limits.layout, options.cancel.as_ref())?
    }
    EntityHandle::MlirFunction { function } => {
      mlir_function_layout(model, *function, limits.layout, options.cancel.as_ref())?
    }
    EntityHandle::MlirModule { module } if options.collapse == CollapseMode::Structural => {
      let Some(groups) = mlir_module_collapsed_groups(model, *module) else {
        return Ok(None);
      };
      collapsed_groups = groups;
      mlir_module_structural_layout(model, *module, limits.layout, options.cancel.as_ref())?
    }
    EntityHandle::MlirModule { module } => {
      mlir_module_layout(model, *module, limits.layout, options.cancel.as_ref())?
    }
    EntityHandle::MlirRegion { scope, .. } | EntityHandle::MlirBlock { scope, .. } => {
      if let Some(function) = mlir_scope_index(scope, "function:") {
        if options.collapse == CollapseMode::Structural {
          let Some(groups) = mlir_function_collapsed_groups(model, function) else {
            return Ok(None);
          };
          collapsed_groups = groups;
          mlir_function_structural_layout(model, function, limits.layout, options.cancel.as_ref())?
        } else {
          mlir_function_layout(model, function, limits.layout, options.cancel.as_ref())?
        }
      } else {
        let Some(module) = mlir_scope_index(scope, "module:") else {
          return Ok(None);
        };
        if options.collapse == CollapseMode::Structural {
          let Some(groups) = mlir_module_collapsed_groups(model, module) else {
            return Ok(None);
          };
          collapsed_groups = groups;
          mlir_module_structural_layout(model, module, limits.layout, options.cancel.as_ref())?
        } else {
          mlir_module_layout(model, module, limits.layout, options.cancel.as_ref())?
        }
      }
    }
    _ => return Ok(None),
  };
  let Some(graph) = graph else {
    return Ok(None);
  };
  check_projection_canceled(&options)?;
  let collapsed_groups = visible_collapsed_groups(collapsed_groups, &graph);
  let omitted_count =
    graph.stats.omitted_nodes + graph.stats.omitted_edges + graph.stats.omitted_initializers;
  let truncated = omitted_count > 0;
  let mut warnings = Vec::new();
  if truncated {
    warnings.push(format!("layout truncated under limit {}", limits.layout));
  }
  Ok(Some(LayoutResponse {
    api_version: SESSION_API_VERSION,
    session_id,
    format: index.kind(),
    scope: scope.clone(),
    collapse: options.collapse,
    cache_key: projection_cache_key(
      session_id,
      index.kind(),
      "layout",
      scope,
      limits.layout,
      options.collapse,
    ),
    limit_used: limits.layout,
    truncated,
    omitted_count,
    warnings,
    graph,
    collapsed_groups,
  }))
}

fn check_projection_canceled(options: &ProjectionOptions) -> Result<(), ProjectionError> {
  check_canceled(options.cancel.as_ref())
}

fn check_canceled(cancel: Option<&CancellationToken>) -> Result<(), ProjectionError> {
  if cancel.is_some_and(CancellationToken::is_canceled) {
    return Err(ProjectionError::Canceled);
  }
  Ok(())
}

fn layout_graph_or_none(
  result: Result<LayoutGraph, LayoutError>,
) -> Result<Option<LayoutGraph>, ProjectionError> {
  match result {
    Ok(graph) => Ok(Some(graph)),
    Err(LayoutError::Canceled) => Err(ProjectionError::Canceled),
    Err(LayoutError::UnknownGraph { .. }) => Ok(None),
  }
}

fn mlir_index(index: &FormatIndex) -> Option<&MlirIndex> {
  match index {
    FormatIndex::Mlir(index) => Some(index),
    _ => None,
  }
}

struct SliceBuilder {
  limit: usize,
  cancel: Option<CancellationToken>,
  entities: Vec<SliceEntity>,
  edges: Vec<SliceEdge>,
  boundaries: Vec<SliceBoundary>,
  collapsed_groups: Vec<CollapsedGroup>,
  seen_entities: BTreeSet<String>,
  omitted_entities: BTreeSet<String>,
  seen_edges: BTreeSet<String>,
  omitted_edges: BTreeSet<String>,
  omitted_count: usize,
}

impl SliceBuilder {
  fn new(limit: usize, cancel: Option<CancellationToken>) -> Self {
    Self {
      limit,
      cancel,
      entities: Vec::new(),
      edges: Vec::new(),
      boundaries: Vec::new(),
      collapsed_groups: Vec::new(),
      seen_entities: BTreeSet::new(),
      omitted_entities: BTreeSet::new(),
      seen_edges: BTreeSet::new(),
      omitted_edges: BTreeSet::new(),
      omitted_count: 0,
    }
  }

  fn is_canceled(&self) -> bool {
    self
      .cancel
      .as_ref()
      .is_some_and(CancellationToken::is_canceled)
  }

  fn check_canceled(&self) -> Result<(), ProjectionError> {
    check_canceled(self.cancel.as_ref())
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
    if self.is_canceled() {
      return;
    }
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
    if self.is_canceled() {
      return;
    }
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
    if self.is_canceled() {
      return;
    }
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

  fn collapsed_group(
    &mut self,
    kind: &str,
    handle: EntityHandle,
    label: Option<String>,
    item_count: usize,
    expand: EntityHandle,
  ) {
    if self.is_canceled() {
      return;
    }
    if let Some(group) = collapsed_group_metadata(kind, handle, label, item_count, expand) {
      self.collapsed_groups.push(group);
    }
  }
}

fn collapsed_group_metadata(
  kind: &str,
  handle: EntityHandle,
  label: Option<String>,
  item_count: usize,
  expand: EntityHandle,
) -> Option<CollapsedGroup> {
  (item_count > 0).then(|| CollapsedGroup {
    id: handle_key(&handle),
    kind: kind.to_owned(),
    handle,
    label,
    item_count,
    omitted_count: item_count,
    expand,
  })
}

#[derive(Debug, Clone)]
struct OnnxRepeatedBlockGroup {
  group: usize,
  operator_path: String,
  instances: Vec<OnnxRepeatedBlockInstance>,
}

impl OnnxRepeatedBlockGroup {
  fn handle(&self, graph: usize) -> EntityHandle {
    EntityHandle::OnnxRepeatedBlock {
      graph,
      group: self.group,
    }
  }

  fn label(&self) -> String {
    format!("{} x{}", self.operator_path, self.instances.len())
  }

  fn item_count(&self) -> usize {
    self
      .instances
      .iter()
      .map(|instance| instance.nodes.len())
      .sum()
  }

  fn width(&self) -> usize {
    self
      .instances
      .first()
      .map_or(0, |instance| instance.nodes.len())
  }
}

#[derive(Debug, Clone)]
struct OnnxRepeatedBlockInstance {
  nodes: Vec<usize>,
  internal_values: Vec<usize>,
}

impl OnnxRepeatedBlockInstance {
  fn first(&self) -> usize {
    self.nodes[0]
  }

  fn uses_any(&self, used_nodes: &BTreeSet<usize>) -> bool {
    self.nodes.iter().any(|node| used_nodes.contains(node))
  }
}

fn onnx_repeated_blocks(model: &Model, graph_index: usize) -> Option<Vec<OnnxRepeatedBlockGroup>> {
  let graph = model.graphs.get(graph_index)?;
  let mut by_signature = BTreeMap::<String, (String, Vec<OnnxRepeatedBlockInstance>)>::new();
  for first in 0..graph.nodes.len() {
    for candidate in onnx_repeated_candidate_node_sets(graph, first) {
      let Some(instance) = onnx_repeated_private_instance(graph, candidate) else {
        continue;
      };
      let nodes = instance
        .nodes
        .iter()
        .filter_map(|node| graph.nodes.get(*node))
        .collect::<Vec<_>>();
      let signature = onnx_repeated_instance_signature(model, graph, &instance);
      let operator_path = nodes
        .iter()
        .map(|node| model.strings.get(node.operator.name).to_owned())
        .collect::<Vec<_>>()
        .join(" -> ");
      by_signature
        .entry(signature)
        .or_insert_with(|| (operator_path, Vec::new()))
        .1
        .push(instance);
    }
  }

  let mut candidates = by_signature
    .into_values()
    .filter_map(|(operator_path, instances)| {
      let instances = non_overlapping_instances(instances, &BTreeSet::new());
      (instances.len() >= 2).then(|| {
        (
          instances[0].first(),
          instances[0].nodes.len(),
          operator_path,
          instances,
        )
      })
    })
    .collect::<Vec<_>>();
  candidates.sort_by(|left, right| {
    left
      .0
      .cmp(&right.0)
      .then_with(|| right.1.cmp(&left.1))
      .then_with(|| left.2.cmp(&right.2))
  });

  let mut groups = Vec::new();
  let mut used_nodes = BTreeSet::new();
  for (_, _, operator_path, instances) in candidates {
    let instances = non_overlapping_instances(instances, &used_nodes);
    if instances.len() < 2 {
      continue;
    }
    for instance in &instances {
      used_nodes.extend(&instance.nodes);
    }
    groups.push(OnnxRepeatedBlockGroup {
      group: groups.len(),
      operator_path,
      instances,
    });
  }
  Some(groups)
}

fn onnx_repeated_candidate_node_sets(
  graph: &netron_rs_core::Graph,
  first: usize,
) -> Vec<Vec<usize>> {
  if graph
    .nodes
    .get(first)
    .is_none_or(onnx_node_has_subgraph_attribute)
  {
    return Vec::new();
  }

  let mut candidates = BTreeSet::new();
  let root = BTreeSet::from([first]);
  let mut visited = BTreeSet::from([root.clone()]);
  let mut stack = vec![root];
  while let Some(nodes) = stack.pop() {
    if nodes.len() >= 2 {
      candidates.insert(nodes.iter().copied().collect::<Vec<_>>());
    }
    if nodes.len() == ONNX_REPEATED_BLOCK_MAX_WIDTH {
      continue;
    }

    for next in onnx_repeated_candidate_expansions(graph, first, &nodes) {
      if visited.insert(next.clone()) {
        stack.push(next);
      }
    }
  }

  candidates.into_iter().collect()
}

fn onnx_repeated_candidate_expansions(
  graph: &netron_rs_core::Graph,
  first: usize,
  nodes: &BTreeSet<usize>,
) -> Vec<BTreeSet<usize>> {
  let mut expansions = Vec::new();
  for source in nodes {
    let Some(source_node) = graph.nodes.get(*source) else {
      continue;
    };
    for output in source_node.outputs.iter().flatten() {
      let Some(value) = graph.values.get(output.index()) else {
        continue;
      };
      if value
        .producer
        .is_none_or(|producer| producer.index() != *source)
      {
        continue;
      }
      let consumers = value
        .consumers
        .iter()
        .map(|consumer| consumer.index())
        .filter(|consumer| !nodes.contains(consumer))
        .collect::<BTreeSet<_>>();
      if consumers.is_empty() {
        continue;
      }
      if consumers.iter().any(|consumer| {
        *consumer <= first
          || graph
            .nodes
            .get(*consumer)
            .is_none_or(onnx_node_has_subgraph_attribute)
      }) {
        continue;
      }

      let remaining = ONNX_REPEATED_BLOCK_MAX_WIDTH.saturating_sub(nodes.len());
      for subset in onnx_repeated_consumer_subsets(&consumers, remaining) {
        let mut next = nodes.clone();
        next.extend(subset);
        expansions.push(next);
      }
    }
  }
  expansions
}

fn onnx_repeated_consumer_subsets(consumers: &BTreeSet<usize>, limit: usize) -> Vec<Vec<usize>> {
  if limit == 0 {
    return Vec::new();
  }
  let consumers = consumers.iter().copied().collect::<Vec<_>>();
  let mut subsets = Vec::new();
  let mut current = Vec::new();
  onnx_repeated_consumer_subsets_from(&consumers, limit, 0, &mut current, &mut subsets);
  subsets
}

fn onnx_repeated_consumer_subsets_from(
  consumers: &[usize],
  limit: usize,
  start: usize,
  current: &mut Vec<usize>,
  subsets: &mut Vec<Vec<usize>>,
) {
  if subsets.len() >= ONNX_REPEATED_BLOCK_MAX_CONSUMER_SUBSETS {
    return;
  }
  for index in start..consumers.len() {
    current.push(consumers[index]);
    subsets.push(current.clone());
    if current.len() < limit {
      onnx_repeated_consumer_subsets_from(consumers, limit, index + 1, current, subsets);
    }
    current.pop();
    if subsets.len() >= ONNX_REPEATED_BLOCK_MAX_CONSUMER_SUBSETS {
      return;
    }
  }
}

fn non_overlapping_instances(
  instances: Vec<OnnxRepeatedBlockInstance>,
  used_nodes: &BTreeSet<usize>,
) -> Vec<OnnxRepeatedBlockInstance> {
  let mut kept = Vec::new();
  let mut kept_nodes = BTreeSet::new();
  for instance in instances {
    if instance.uses_any(used_nodes) || instance.uses_any(&kept_nodes) {
      continue;
    }
    kept_nodes.extend(&instance.nodes);
    kept.push(instance);
  }
  kept
}

fn onnx_repeated_private_instance(
  graph: &netron_rs_core::Graph,
  nodes: Vec<usize>,
) -> Option<OnnxRepeatedBlockInstance> {
  if nodes.len() < 2 || nodes.len() > ONNX_REPEATED_BLOCK_MAX_WIDTH {
    return None;
  }
  if nodes.iter().any(|node| {
    graph
      .nodes
      .get(*node)
      .is_none_or(onnx_node_has_subgraph_attribute)
  }) {
    return None;
  }
  let node_set = nodes.iter().copied().collect::<BTreeSet<_>>();
  let mut internal_values = Vec::new();
  for source in &nodes {
    let source_node = graph.nodes.get(*source)?;
    for output in source_node.outputs.iter().flatten() {
      let value = graph.values.get(output.index())?;
      if value.producer?.index() != *source {
        return None;
      }
      let internal_consumers = value
        .consumers
        .iter()
        .map(|consumer| consumer.index())
        .filter(|consumer| node_set.contains(consumer))
        .collect::<Vec<_>>();
      if internal_consumers.is_empty() {
        continue;
      }
      internal_values.push(output.index());
    }
  }
  if internal_values.is_empty() {
    return None;
  }
  if !onnx_repeated_instance_is_connected(graph, &nodes, &node_set, &internal_values) {
    return None;
  }
  Some(OnnxRepeatedBlockInstance {
    nodes,
    internal_values,
  })
}

fn onnx_repeated_instance_is_connected(
  graph: &netron_rs_core::Graph,
  nodes: &[usize],
  node_set: &BTreeSet<usize>,
  internal_values: &[usize],
) -> bool {
  let Some(first) = nodes.first().copied() else {
    return false;
  };
  let mut adjacent = BTreeMap::<usize, Vec<usize>>::new();
  for value in internal_values {
    let Some(value) = graph.values.get(*value) else {
      return false;
    };
    let Some(source) = value.producer.map(|producer| producer.index()) else {
      return false;
    };
    for target in value
      .consumers
      .iter()
      .map(|consumer| consumer.index())
      .filter(|target| node_set.contains(target))
    {
      adjacent.entry(source).or_default().push(target);
      adjacent.entry(target).or_default().push(source);
    }
  }

  let mut stack = vec![first];
  let mut seen = BTreeSet::new();
  while let Some(node) = stack.pop() {
    if !seen.insert(node) {
      continue;
    }
    if let Some(next) = adjacent.get(&node) {
      stack.extend(next.iter().copied());
    }
  }
  nodes.iter().all(|node| seen.contains(node))
}

fn onnx_repeated_instance_signature(
  model: &Model,
  graph: &netron_rs_core::Graph,
  instance: &OnnxRepeatedBlockInstance,
) -> String {
  let positions = onnx_repeated_canonical_positions(model, graph, instance);
  let mut ordered_nodes = instance.nodes.clone();
  ordered_nodes.sort_by_key(|node| positions.get(node).copied().unwrap_or(usize::MAX));
  let node_path = ordered_nodes
    .iter()
    .filter_map(|node| graph.nodes.get(*node))
    .map(|node| onnx_node_collapse_signature(model, node))
    .collect::<Vec<_>>()
    .join("->");
  let mut internal_edges = Vec::new();
  for value in &instance.internal_values {
    let Some(value) = graph.values.get(*value) else {
      continue;
    };
    let Some(source) = value
      .producer
      .and_then(|producer| positions.get(&producer.index()).copied())
    else {
      continue;
    };
    let mut targets = value
      .consumers
      .iter()
      .filter_map(|consumer| positions.get(&consumer.index()).copied())
      .collect::<Vec<_>>();
    targets.sort_unstable();
    let external_consumers = value
      .consumers
      .iter()
      .filter(|consumer| !positions.contains_key(&consumer.index()))
      .count();
    internal_edges.push(format!(
      "{source}>{}:external{external_consumers}:graph_output{}",
      targets
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(","),
      usize::from(value.is_graph_output)
    ));
  }
  internal_edges.sort();
  format!("{node_path}|edges:{}", internal_edges.join(";"))
}

fn onnx_repeated_canonical_positions(
  model: &Model,
  graph: &netron_rs_core::Graph,
  instance: &OnnxRepeatedBlockInstance,
) -> BTreeMap<usize, usize> {
  let node_set = instance.nodes.iter().copied().collect::<BTreeSet<_>>();
  let depths = onnx_repeated_node_depths(graph, instance, &node_set);
  let mut labels = instance
    .nodes
    .iter()
    .filter_map(|node| {
      graph.nodes.get(*node).map(|detail| {
        (
          *node,
          format!(
            "depth{}:{}",
            depths.get(node).copied().unwrap_or_default(),
            onnx_node_collapse_signature(model, detail)
          ),
        )
      })
    })
    .collect::<BTreeMap<_, _>>();

  for _ in 0..ONNX_REPEATED_BLOCK_MAX_WIDTH {
    let mut next = BTreeMap::new();
    for node in &instance.nodes {
      let mut incoming = Vec::new();
      let mut outgoing = Vec::new();
      for value in &instance.internal_values {
        let Some(value) = graph.values.get(*value) else {
          continue;
        };
        let Some(source) = value.producer.map(|producer| producer.index()) else {
          continue;
        };
        let targets = value
          .consumers
          .iter()
          .map(|consumer| consumer.index())
          .filter(|target| node_set.contains(target))
          .collect::<Vec<_>>();
        if targets.contains(node) {
          incoming.push(labels.get(&source).cloned().unwrap_or_default());
        }
        if source == *node {
          let mut target_labels = targets
            .iter()
            .filter_map(|target| labels.get(target).cloned())
            .collect::<Vec<_>>();
          target_labels.sort();
          let external_consumers = value
            .consumers
            .iter()
            .filter(|consumer| !node_set.contains(&consumer.index()))
            .count();
          outgoing.push(format!(
            "{}:external{external_consumers}:graph_output{}",
            target_labels.join(","),
            usize::from(value.is_graph_output)
          ));
        }
      }
      incoming.sort();
      outgoing.sort();
      next.insert(
        *node,
        format!(
          "{}|in:{}|out:{}",
          labels.get(node).cloned().unwrap_or_default(),
          incoming.join(","),
          outgoing.join(";")
        ),
      );
    }
    if next == labels {
      break;
    }
    labels = next;
  }

  let mut ordered = instance.nodes.clone();
  ordered.sort_by(|left, right| {
    depths
      .get(left)
      .copied()
      .unwrap_or_default()
      .cmp(&depths.get(right).copied().unwrap_or_default())
      .then_with(|| labels.get(left).cmp(&labels.get(right)))
      .then_with(|| left.cmp(right))
  });
  ordered
    .into_iter()
    .enumerate()
    .map(|(position, node)| (node, position))
    .collect()
}

fn onnx_repeated_node_depths(
  graph: &netron_rs_core::Graph,
  instance: &OnnxRepeatedBlockInstance,
  node_set: &BTreeSet<usize>,
) -> BTreeMap<usize, usize> {
  let mut depths = instance
    .nodes
    .iter()
    .map(|node| (*node, 0usize))
    .collect::<BTreeMap<_, _>>();
  for _ in 0..instance.nodes.len() {
    let mut changed = false;
    for value in &instance.internal_values {
      let Some(value) = graph.values.get(*value) else {
        continue;
      };
      let Some(source) = value.producer.map(|producer| producer.index()) else {
        continue;
      };
      let Some(source_depth) = depths.get(&source).copied() else {
        continue;
      };
      for target in value
        .consumers
        .iter()
        .map(|consumer| consumer.index())
        .filter(|target| node_set.contains(target))
      {
        let depth = source_depth.saturating_add(1);
        let entry = depths.entry(target).or_default();
        if *entry < depth {
          *entry = depth;
          changed = true;
        }
      }
    }
    if !changed {
      break;
    }
  }
  depths
}

fn onnx_node_has_subgraph_attribute(node: &Node) -> bool {
  node.attributes.iter().any(|attribute| {
    matches!(
      attribute.value,
      AttributeValue::Graph(_) | AttributeValue::Graphs(_)
    )
  })
}

fn onnx_node_collapse_signature(model: &Model, node: &Node) -> String {
  let mut attributes = node
    .attributes
    .iter()
    .map(|attribute| {
      let values = attribute_value_terms(model, &attribute.value).join(",");
      format!("{}={values}", model.strings.get(attribute.name))
    })
    .collect::<Vec<_>>();
  attributes.sort();
  format!(
    "{}:{}:in{}:out{}:attrs[{}]",
    operator_domain(model, &node.operator),
    model.strings.get(node.operator.name),
    node.inputs.iter().flatten().count(),
    node.outputs.iter().flatten().count(),
    attributes.join(";")
  )
}

fn onnx_repeated_block_collapsed_groups(
  graph: usize,
  groups: &[OnnxRepeatedBlockGroup],
) -> Vec<CollapsedGroup> {
  groups
    .iter()
    .filter_map(|group| {
      let handle = group.handle(graph);
      collapsed_group_metadata(
        "onnx_repeated_block",
        handle.clone(),
        Some(group.label()),
        group.item_count(),
        handle,
      )
    })
    .collect()
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
    if builder.is_canceled() {
      return Some(());
    }
    let value = &graph.values[value.index()];
    push_onnx_value_entity(model, graph_index, value, builder);
  }
  push_onnx_boundary_path(model, graph_index, builder);
  for node in &graph.nodes {
    if builder.is_canceled() {
      return Some(());
    }
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

fn onnx_graph_structural_slice(
  model: &Model,
  graph_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let graph = model.graphs.get(graph_index)?;
  let groups = onnx_repeated_blocks(model, graph_index)?;
  if groups.is_empty() {
    return onnx_graph_slice(model, graph_index, builder);
  }
  let node_groups = onnx_repeated_block_node_groups(&groups);
  builder.handle(
    EntityHandle::Graph { graph: graph_index },
    "graph",
    graph.name.map(|name| model.strings.get(name).to_owned()),
    None,
    true,
  );
  for value in graph.inputs.iter().chain(&graph.outputs) {
    if builder.is_canceled() {
      return Some(());
    }
    let value = &graph.values[value.index()];
    push_onnx_value_entity(model, graph_index, value, builder);
  }
  for group in &groups {
    if builder.is_canceled() {
      return Some(());
    }
    let handle = group.handle(graph_index);
    builder.handle(
      handle.clone(),
      "onnx_repeated_block",
      Some(group.label()),
      Some("repeated_block".to_owned()),
      true,
    );
    builder.collapsed_group(
      "onnx_repeated_block",
      handle.clone(),
      Some(group.label()),
      group.item_count(),
      handle,
    );
  }
  for node in &graph.nodes {
    if builder.is_canceled() {
      return Some(());
    }
    if node_groups.contains_key(&node.id.index()) {
      continue;
    }
    push_onnx_node_entity(model, graph_index, node, builder);
  }
  push_onnx_structural_edges(model, graph_index, &node_groups, builder);
  Some(())
}

fn onnx_repeated_block_slice(
  model: &Model,
  graph_index: usize,
  group_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let graph = model.graphs.get(graph_index)?;
  let groups = onnx_repeated_blocks(model, graph_index)?;
  let group = groups.get(group_index)?;
  builder.handle(
    group.handle(graph_index),
    "onnx_repeated_block",
    Some(group.label()),
    Some("repeated_block".to_owned()),
    true,
  );
  for instance in &group.instances {
    if builder.is_canceled() {
      return Some(());
    }
    for node in &instance.nodes {
      let node = graph.nodes.get(*node)?;
      push_onnx_node_entity(model, graph_index, node, builder);
      for value in node.inputs.iter().chain(&node.outputs).flatten() {
        push_onnx_value_entity(model, graph_index, &graph.values[value.index()], builder);
      }
    }
    for value in &instance.internal_values {
      push_onnx_value_entity(model, graph_index, &graph.values[*value], builder);
    }
  }
  push_onnx_edges(model, graph_index, builder);
  Some(())
}

fn onnx_repeated_block_node_groups(groups: &[OnnxRepeatedBlockGroup]) -> BTreeMap<usize, usize> {
  let mut node_groups = BTreeMap::new();
  for group in groups {
    for instance in &group.instances {
      for node in &instance.nodes {
        node_groups.insert(*node, group.group);
      }
    }
  }
  node_groups
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
    if builder.is_canceled() {
      return;
    }
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
  depth: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let graph = model.graphs.get(graph_index)?;
  graph.nodes.get(node_index)?;
  let mut queue = VecDeque::from([(node_index, 0usize)]);
  let mut seen_nodes = BTreeSet::new();
  let mut seen_values = BTreeSet::new();

  while let Some((current_index, current_depth)) = queue.pop_front() {
    if builder.is_canceled() {
      return Some(());
    }
    if !seen_nodes.insert(current_index) {
      continue;
    }
    let node = graph.nodes.get(current_index)?;
    push_onnx_node_entity(model, graph_index, node, builder);
    for value_id in node.inputs.iter().chain(&node.outputs).flatten() {
      let value = &graph.values[value_id.index()];
      if seen_values.insert(value.id.index()) {
        push_onnx_value_entity(model, graph_index, value, builder);
      }
      if current_depth >= depth {
        continue;
      }
      if let Some(producer) = value.producer
        && !seen_nodes.contains(&producer.index())
      {
        queue.push_back((producer.index(), current_depth + 1));
      }
      for consumer in &value.consumers {
        if !seen_nodes.contains(&consumer.index()) {
          queue.push_back((consumer.index(), current_depth + 1));
        }
      }
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
    if builder.is_canceled() {
      return Some(());
    }
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
    if builder.is_canceled() {
      return Some(());
    }
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
    if builder.is_canceled() {
      return;
    }
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

fn push_onnx_structural_edges(
  model: &Model,
  graph_index: usize,
  node_groups: &BTreeMap<usize, usize>,
  builder: &mut SliceBuilder,
) {
  let Some(graph) = model.graphs.get(graph_index) else {
    return;
  };
  for value in &graph.values {
    if builder.is_canceled() {
      return;
    }
    let source = value.producer.map_or_else(
      || {
        let value_id = handle_key(&EntityHandle::Value {
          graph: graph_index,
          value: value.id.index(),
        });
        builder
          .seen_entities
          .contains(&value_id)
          .then_some(value_id)
      },
      |producer| {
        Some(onnx_structural_node_id(
          graph_index,
          producer.index(),
          node_groups,
        ))
      },
    );
    let Some(source) = source else {
      continue;
    };
    for consumer in &value.consumers {
      let target = onnx_structural_node_id(graph_index, consumer.index(), node_groups);
      if source == target {
        continue;
      }
      builder.edge(
        source.clone(),
        target,
        None,
        Some(model.strings.get(value.name).to_owned()),
      );
    }
    if value.is_graph_output {
      let output = handle_key(&EntityHandle::Value {
        graph: graph_index,
        value: value.id.index(),
      });
      if source != output && builder.seen_entities.contains(&output) {
        builder.edge(
          source.clone(),
          output,
          None,
          Some(model.strings.get(value.name).to_owned()),
        );
      }
    }
  }
}

fn onnx_structural_node_id(
  graph: usize,
  node: usize,
  node_groups: &BTreeMap<usize, usize>,
) -> String {
  node_groups.get(&node).map_or_else(
    || handle_key(&EntityHandle::Node { graph, node }),
    |group| {
      handle_key(&EntityHandle::OnnxRepeatedBlock {
        graph,
        group: *group,
      })
    },
  )
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
    if builder.is_canceled() {
      return Some(());
    }
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
    if builder.is_canceled() {
      return Some(());
    }
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

fn mlir_structural_scope_slice(
  model: &Model,
  _index: &MlirIndex,
  scope: &str,
  builder: &mut SliceBuilder,
) -> Option<()> {
  if let Some(function) = mlir_scope_index(scope, "function:") {
    return mlir_function_structural_slice(model, function, builder);
  }
  mlir_module_structural_slice(model, mlir_scope_index(scope, "module:")?, builder)
}

fn mlir_module_structural_slice(
  model: &Model,
  module: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let graph = model.graphs.get(module)?;
  let scope = mlir_module_scope(module);
  let module_handle = EntityHandle::MlirModule { module };
  let region_handle = EntityHandle::MlirRegion {
    scope: scope.clone(),
    region: 0,
  };
  let block_handle = EntityHandle::MlirBlock {
    scope: scope.clone(),
    block: 0,
  };
  let module_id = handle_key(&module_handle);
  let region_id = handle_key(&region_handle);
  let block_id = handle_key(&block_handle);
  builder.handle(
    module_handle,
    "mlir_module",
    graph.name.map(|name| model.strings.get(name).to_owned()),
    None,
    true,
  );
  builder.handle(
    region_handle,
    "mlir_region",
    Some(format!("{scope} region 0")),
    None,
    true,
  );
  builder.handle(
    block_handle.clone(),
    "mlir_block",
    Some(format!("{scope} block 0")),
    None,
    true,
  );
  builder.edge(
    module_id.clone(),
    region_id.clone(),
    None,
    Some("contains".to_owned()),
  );
  builder.edge(region_id, block_id, None, Some("contains".to_owned()));
  builder.collapsed_group(
    "mlir_block",
    block_handle.clone(),
    Some(format!("{scope} block 0")),
    graph.nodes.len() + graph.values.len(),
    block_handle,
  );
  for (function, item) in model.functions.iter().enumerate() {
    if builder.is_canceled() {
      return Some(());
    }
    if !mlir_function_belongs_to_module(model, item, graph) {
      continue;
    }
    let function_handle = EntityHandle::MlirFunction { function };
    let function_id = handle_key(&function_handle);
    let label = Some(model.strings.get(item.name).to_owned());
    builder.handle(
      function_handle.clone(),
      "mlir_function",
      label.clone(),
      None,
      true,
    );
    builder.edge(
      module_id.clone(),
      function_id,
      None,
      Some("contains".to_owned()),
    );
    builder.collapsed_group(
      "mlir_function",
      function_handle.clone(),
      label,
      item.nodes.len() + item.values.len(),
      function_handle,
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
  let function_handle = EntityHandle::MlirFunction {
    function: function_index,
  };
  let region_handle = EntityHandle::MlirRegion {
    scope: scope.clone(),
    region: 0,
  };
  let block_handle = EntityHandle::MlirBlock {
    scope: scope.clone(),
    block: 0,
  };
  let function_id = handle_key(&function_handle);
  let region_id = handle_key(&region_handle);
  let block_id = handle_key(&block_handle);
  builder.handle(
    function_handle,
    "mlir_function",
    Some(model.strings.get(function.name).to_owned()),
    None,
    true,
  );
  builder.handle(
    region_handle,
    "mlir_region",
    Some(format!("{scope} region 0")),
    None,
    true,
  );
  builder.handle(
    block_handle,
    "mlir_block",
    Some(format!("{scope} block 0")),
    None,
    true,
  );
  builder.edge(
    function_id,
    region_id.clone(),
    None,
    Some("contains".to_owned()),
  );
  builder.edge(region_id, block_id, None, Some("contains".to_owned()));
  if mlir_function_has_bytecode_topology(function) {
    return mlir_bytecode_function_slice(model, function, &scope, None, None, builder);
  }
  let block_args = mlir_function_block_argument_names(model, function);
  for (value, item) in function.values.iter().enumerate() {
    if builder.is_canceled() {
      return Some(());
    }
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
    if builder.is_canceled() {
      return Some(());
    }
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

fn mlir_function_has_bytecode_topology(function: &Function) -> bool {
  function
    .nodes
    .iter()
    .any(|node| node.metadata.contains_key("bytecode.operation"))
}

fn mlir_bytecode_function_slice(
  model: &Model,
  function: &Function,
  scope: &str,
  region: Option<usize>,
  block: Option<usize>,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let block_args = mlir_function_block_argument_names(model, function);
  let context = MlirBytecodeSliceContext {
    model,
    function,
    scope,
    block_args: &block_args,
  };
  for require_values in [true, false] {
    for (operation, node) in function.nodes.iter().enumerate() {
      if builder.is_canceled() {
        return Some(());
      }
      let has_values = node.inputs.iter().chain(&node.outputs).any(Option::is_some);
      if has_values != require_values || !mlir_bytecode_operation_matches(node, region, block) {
        continue;
      }
      if !push_mlir_bytecode_operation_slice(&context, operation, node, builder) {
        return Some(());
      }
    }
  }
  Some(())
}

fn mlir_bytecode_operation_matches(
  node: &netron_rs_core::FunctionNode,
  region: Option<usize>,
  block: Option<usize>,
) -> bool {
  if let Some(region) = region
    && metadata_usize(&node.metadata, "bytecode.region") != Some(region)
  {
    return false;
  }
  if let Some(block) = block
    && metadata_usize(&node.metadata, "bytecode.block") != Some(block)
  {
    return false;
  }
  true
}

fn metadata_usize(metadata: &BTreeMap<String, String>, key: &str) -> Option<usize> {
  metadata.get(key)?.parse().ok()
}

fn metadata_usize_list(metadata: &BTreeMap<String, String>, key: &str) -> Vec<usize> {
  metadata
    .get(key)
    .map(|value| {
      value
        .split(',')
        .filter_map(|item| item.trim().parse().ok())
        .collect()
    })
    .unwrap_or_default()
}

struct MlirBytecodeSliceContext<'a> {
  model: &'a Model,
  function: &'a Function,
  scope: &'a str,
  block_args: &'a BTreeSet<String>,
}

fn push_mlir_bytecode_operation_slice(
  context: &MlirBytecodeSliceContext<'_>,
  operation: usize,
  node: &netron_rs_core::FunctionNode,
  builder: &mut SliceBuilder,
) -> bool {
  if builder.is_canceled() {
    return false;
  }
  let operation_handle = EntityHandle::MlirOperation {
    scope: context.scope.to_owned(),
    operation,
  };
  let operation_id = handle_key(&operation_handle);
  builder.handle(
    operation_handle.clone(),
    "mlir_operation",
    node
      .name
      .map(|name| context.model.strings.get(name).to_owned()),
    Some(context.model.strings.get(node.operator.name).to_owned()),
    false,
  );
  if !builder.seen_entities.contains(&operation_id) {
    return false;
  }
  push_mlir_bytecode_operation_topology(context.scope, &operation_handle, &node.metadata, builder);
  for input in node.inputs.iter().flatten() {
    if builder.is_canceled() {
      return false;
    }
    push_mlir_function_value_and_edge(
      context,
      *input,
      None,
      Some(operation_handle.clone()),
      builder,
    );
  }
  for output in node.outputs.iter().flatten() {
    if builder.is_canceled() {
      return false;
    }
    push_mlir_function_value_and_edge(
      context,
      *output,
      Some(operation_handle.clone()),
      None,
      builder,
    );
  }
  true
}

fn push_mlir_bytecode_operation_topology(
  scope: &str,
  operation_handle: &EntityHandle,
  metadata: &BTreeMap<String, String>,
  builder: &mut SliceBuilder,
) {
  let operation_id = handle_key(operation_handle);
  if let (Some(region), Some(block)) = (
    metadata_usize(metadata, "bytecode.region"),
    metadata_usize(metadata, "bytecode.block"),
  ) {
    let region_handle = EntityHandle::MlirRegion {
      scope: scope.to_owned(),
      region,
    };
    let block_handle = EntityHandle::MlirBlock {
      scope: scope.to_owned(),
      block,
    };
    let region_id = handle_key(&region_handle);
    let block_id = handle_key(&block_handle);
    builder.handle(
      region_handle,
      "mlir_region",
      Some(format!("{scope} region {region}")),
      None,
      true,
    );
    builder.handle(
      block_handle,
      "mlir_block",
      Some(format!("{scope} block {block}")),
      None,
      true,
    );
    builder.edge(
      region_id.clone(),
      block_id.clone(),
      None,
      Some("contains".to_owned()),
    );
    builder.edge(
      block_id,
      operation_id.clone(),
      None,
      Some("contains".to_owned()),
    );
  }
  for region in metadata_usize_list(metadata, "bytecode.nested_region_ids") {
    let region_handle = EntityHandle::MlirRegion {
      scope: scope.to_owned(),
      region,
    };
    let region_id = handle_key(&region_handle);
    builder.handle(
      region_handle,
      "mlir_region",
      Some(format!("{scope} region {region}")),
      None,
      true,
    );
    builder.edge(
      operation_id.clone(),
      region_id,
      None,
      Some("contains".to_owned()),
    );
  }
}

fn push_mlir_function_value_and_edge(
  context: &MlirBytecodeSliceContext<'_>,
  value_name: netron_rs_core::StringId,
  source: Option<EntityHandle>,
  target: Option<EntityHandle>,
  builder: &mut SliceBuilder,
) {
  if builder.is_canceled() {
    return;
  }
  let Some(value) = mlir_function_value_index(context.function, value_name) else {
    return;
  };
  let value_handle = EntityHandle::MlirValue {
    scope: context.scope.to_owned(),
    value,
  };
  let value_id = handle_key(&value_handle);
  let name = context.model.strings.get(value_name);
  builder.handle(
    value_handle.clone(),
    "mlir_value",
    Some(name.to_owned()),
    None,
    context.block_args.contains(name)
      || context
        .function
        .outputs
        .iter()
        .any(|id| context.model.strings.get(*id) == name),
  );
  if let Some(source) = source {
    builder.edge(
      handle_key(&source),
      value_id.clone(),
      None,
      Some(name.to_owned()),
    );
  }
  if let Some(target) = target {
    builder.edge(
      handle_key(&value_handle),
      handle_key(&target),
      None,
      Some(name.to_owned()),
    );
  }
}

fn mlir_function_structural_slice(
  model: &Model,
  function_index: usize,
  builder: &mut SliceBuilder,
) -> Option<()> {
  let function = model.functions.get(function_index)?;
  let scope = mlir_function_scope(function_index);
  let function_handle = EntityHandle::MlirFunction {
    function: function_index,
  };
  let region_handle = EntityHandle::MlirRegion {
    scope: scope.clone(),
    region: 0,
  };
  let block_handle = EntityHandle::MlirBlock {
    scope: scope.clone(),
    block: 0,
  };
  let function_id = handle_key(&function_handle);
  let region_id = handle_key(&region_handle);
  let block_id = handle_key(&block_handle);
  builder.handle(
    function_handle,
    "mlir_function",
    Some(model.strings.get(function.name).to_owned()),
    None,
    true,
  );
  builder.handle(
    region_handle,
    "mlir_region",
    Some(format!("{scope} region 0")),
    None,
    true,
  );
  builder.handle(
    block_handle.clone(),
    "mlir_block",
    Some(format!("{scope} block 0")),
    None,
    true,
  );
  builder.edge(
    function_id,
    region_id.clone(),
    None,
    Some("contains".to_owned()),
  );
  builder.edge(region_id, block_id, None, Some("contains".to_owned()));
  builder.collapsed_group(
    "mlir_block",
    block_handle.clone(),
    Some(format!("{scope} block 0")),
    function.nodes.len() + function.values.len(),
    block_handle,
  );
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
    let operation_handle = EntityHandle::MlirOperation {
      scope: scope.to_owned(),
      operation,
    };
    builder.handle(
      operation_handle.clone(),
      "mlir_operation",
      node.name.map(|name| model.strings.get(name).to_owned()),
      Some(model.strings.get(node.operator.name).to_owned()),
      false,
    );
    push_mlir_bytecode_operation_topology(scope, &operation_handle, &node.metadata, builder);
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
  let operation_handle = EntityHandle::MlirOperation {
    scope: scope.to_owned(),
    operation,
  };
  builder.handle(
    operation_handle.clone(),
    "mlir_operation",
    node.name.map(|name| model.strings.get(name).to_owned()),
    Some(model.strings.get(node.operator.name).to_owned()),
    false,
  );
  push_mlir_bytecode_operation_topology(scope, &operation_handle, &node.metadata, builder);
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
  if let Some(function_index) = mlir_scope_index(scope, "function:")
    && let Some(function) = model.functions.get(function_index)
    && mlir_function_has_bytecode_topology(function)
  {
    return mlir_bytecode_function_slice(model, function, scope, Some(region), None, builder);
  }
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
  if let Some(function_index) = mlir_scope_index(scope, "function:")
    && let Some(function) = model.functions.get(function_index)
    && mlir_function_has_bytecode_topology(function)
  {
    return mlir_bytecode_function_slice(model, function, scope, None, Some(block), builder);
  }
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
    if builder.is_canceled() {
      return Some(());
    }
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
      if builder.is_canceled() {
        return Some(());
      }
      if mlir_node_mentions_symbol(model, node.name, &node.attributes, symbol_name) {
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
    if builder.is_canceled() {
      return Some(());
    }
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
      if builder.is_canceled() {
        return Some(());
      }
      if mlir_node_mentions_symbol(model, node.name, &node.attributes, symbol_name) {
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

fn mlir_node_mentions_symbol(
  model: &Model,
  name: Option<netron_rs_core::StringId>,
  attributes: &[Attribute],
  symbol: &str,
) -> bool {
  name.is_some_and(|name| mlir_symbol_matches(model.strings.get(name), symbol))
    || attributes
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
    if builder.is_canceled() {
      return Some(());
    }
    let scope = mlir_module_scope(module);
    for (operation, node) in graph.nodes.iter().enumerate() {
      if builder.is_canceled() {
        return Some(());
      }
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
    if builder.is_canceled() {
      return Some(());
    }
    let scope = mlir_function_scope(function);
    for (operation, node) in item.nodes.iter().enumerate() {
      if builder.is_canceled() {
        return Some(());
      }
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

fn mlir_function_layout(
  model: &Model,
  function_index: usize,
  limit: usize,
  cancel: Option<&CancellationToken>,
) -> Result<Option<LayoutGraph>, ProjectionError> {
  check_canceled(cancel)?;
  let Some(function) = model.functions.get(function_index) else {
    return Ok(None);
  };
  let scope = mlir_function_scope(function_index);
  let visible_ops = function.nodes.len().min(limit);
  let mut nodes = Vec::new();
  let mut value_nodes = BTreeMap::new();
  let mut op_nodes = Vec::new();
  let mut output_nodes = Vec::new();
  let block_args = mlir_function_block_argument_names(model, function);
  for (value, item) in function.values.iter().enumerate() {
    check_canceled(cancel)?;
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
    check_canceled(cancel)?;
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
    check_canceled(cancel)?;
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
  check_canceled(cancel)?;
  position_layout_nodes(&mut nodes);

  let mut edges = Vec::new();
  for (operation, node) in function.nodes.iter().take(visible_ops).enumerate() {
    check_canceled(cancel)?;
    for input in node.inputs.iter().flatten() {
      check_canceled(cancel)?;
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
    check_canceled(cancel)?;
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
  check_canceled(cancel)?;
  let stats = LayoutStats {
    nodes_used: nodes.len(),
    edges_used: edges.len(),
    omitted_nodes: function.nodes.len().saturating_sub(visible_ops),
    ..LayoutStats::default()
  };
  Ok(Some(LayoutGraph {
    graph: function_index,
    parent: None,
    name: Some(model.strings.get(function.name).to_owned()),
    subgraphs: Vec::new(),
    bounds: layout_bounds(&nodes),
    nodes,
    edges,
    stats,
  }))
}

fn mlir_module_layout(
  model: &Model,
  module_index: usize,
  limit: usize,
  cancel: Option<&CancellationToken>,
) -> Result<Option<LayoutGraph>, ProjectionError> {
  layout_graph_or_none(netron_rs_layout::layout_graph(
    model,
    &LayoutOptions {
      graph: module_index,
      max_nodes: Some(limit),
      cancel: cancel.cloned(),
      ..LayoutOptions::default()
    },
  ))
}

fn onnx_graph_structural_layout(
  model: &Model,
  graph_index: usize,
  groups: &[OnnxRepeatedBlockGroup],
  limit: usize,
  cancel: Option<&CancellationToken>,
) -> Result<Option<LayoutGraph>, ProjectionError> {
  check_canceled(cancel)?;
  if groups.is_empty() {
    return layout_graph_or_none(netron_rs_layout::layout_graph(
      model,
      &LayoutOptions {
        graph: graph_index,
        max_nodes: Some(limit),
        cancel: cancel.cloned(),
        ..LayoutOptions::default()
      },
    ));
  }
  let Some(graph) = model.graphs.get(graph_index) else {
    return Ok(None);
  };
  let capacity = limit.max(1);
  let node_groups = onnx_repeated_block_node_groups(groups);
  let mut nodes = Vec::new();
  let mut value_nodes = BTreeMap::new();
  let mut operator_nodes = BTreeMap::new();
  let mut group_nodes = BTreeMap::new();
  let mut capacity_omitted = 0usize;

  for value in &graph.inputs {
    check_canceled(cancel)?;
    let value = &graph.values[value.index()];
    if let Some(index) = push_onnx_structural_layout_node(
      &mut nodes,
      capacity,
      onnx_boundary_layout_node(
        graph_index,
        value.id.index(),
        model.strings.get(value.name).to_owned(),
        LayoutNodeKind::GraphInput,
        0,
      ),
    ) {
      value_nodes.insert(value.id.index(), index);
    } else {
      capacity_omitted += 1;
    }
  }

  for group in groups {
    check_canceled(cancel)?;
    let rank = group.instances[0].first() + 1;
    if let Some(index) = push_onnx_structural_layout_node(
      &mut nodes,
      capacity,
      onnx_group_layout_node(graph_index, group, rank),
    ) {
      group_nodes.insert(group.group, index);
    } else {
      capacity_omitted += 1;
    }
  }

  for node in &graph.nodes {
    check_canceled(cancel)?;
    if node_groups.contains_key(&node.id.index()) {
      continue;
    }
    if let Some(index) = push_onnx_structural_layout_node(
      &mut nodes,
      capacity,
      onnx_operator_layout_node(model, graph_index, node),
    ) {
      operator_nodes.insert(node.id.index(), index);
    } else {
      capacity_omitted += 1;
    }
  }

  for value in &graph.outputs {
    check_canceled(cancel)?;
    let value = &graph.values[value.index()];
    let rank = graph.nodes.len() + 2;
    if let Some(index) = push_onnx_structural_layout_node(
      &mut nodes,
      capacity,
      onnx_boundary_layout_node(
        graph_index,
        value.id.index(),
        model.strings.get(value.name).to_owned(),
        LayoutNodeKind::GraphOutput,
        rank,
      ),
    ) {
      value_nodes.insert(value.id.index(), index);
    } else {
      capacity_omitted += 1;
    }
  }

  check_canceled(cancel)?;
  position_layout_nodes(&mut nodes);
  let mut stats = LayoutStats {
    omitted_nodes: groups
      .iter()
      .map(|group| group.item_count().saturating_sub(1))
      .sum::<usize>()
      + capacity_omitted,
    ..LayoutStats::default()
  };
  let edges = onnx_structural_layout_edges(
    OnnxStructuralLayoutEdgeContext {
      model,
      graph,
      node_groups: &node_groups,
      group_nodes: &group_nodes,
      operator_nodes: &operator_nodes,
      value_nodes: &value_nodes,
      nodes: &nodes,
    },
    cancel,
    &mut stats,
  )?;
  stats.nodes_used = nodes.len();
  stats.edges_used = edges.len();
  Ok(Some(LayoutGraph {
    graph: graph_index,
    parent: graph.parent.map(|parent| parent.index()),
    name: graph.name.map(|name| model.strings.get(name).to_owned()),
    subgraphs: graph.subgraphs.iter().map(|graph| graph.index()).collect(),
    bounds: layout_bounds(&nodes),
    nodes,
    edges,
    stats,
  }))
}

fn push_onnx_structural_layout_node(
  nodes: &mut Vec<LayoutNode>,
  capacity: usize,
  node: LayoutNode,
) -> Option<usize> {
  if nodes.len() >= capacity {
    return None;
  }
  let index = nodes.len();
  nodes.push(node);
  Some(index)
}

fn onnx_boundary_layout_node(
  graph: usize,
  value: usize,
  name: String,
  kind: LayoutNodeKind,
  rank: usize,
) -> LayoutNode {
  let id_prefix = match kind {
    LayoutNodeKind::GraphInput => "input",
    LayoutNodeKind::GraphOutput => "output",
    _ => "value",
  };
  LayoutNode {
    id: format!("{id_prefix}:{value}"),
    kind,
    graph,
    node: None,
    value: Some(value),
    name: Some(name),
    operator: None,
    origin: None,
    rank,
    order: 0,
    x: 0.0,
    y: 0.0,
    width: 160.0,
    height: 48.0,
    hidden_initializers: 0,
  }
}

fn onnx_operator_layout_node(model: &Model, graph: usize, node: &Node) -> LayoutNode {
  LayoutNode {
    id: format!("node:{}", node.id.index()),
    kind: LayoutNodeKind::Operator,
    graph,
    node: Some(node.id.index()),
    value: None,
    name: node.name.map(|name| model.strings.get(name).to_owned()),
    operator: Some(model.strings.get(node.operator.name).to_owned()),
    origin: Some(node.operator.origin),
    rank: node.id.index() + 1,
    order: 0,
    x: 0.0,
    y: 0.0,
    width: 190.0,
    height: 64.0,
    hidden_initializers: hidden_initializers_for_node(model, graph, node),
  }
}

fn onnx_group_layout_node(graph: usize, group: &OnnxRepeatedBlockGroup, rank: usize) -> LayoutNode {
  LayoutNode {
    id: handle_key(&group.handle(graph)),
    kind: LayoutNodeKind::Group,
    graph,
    node: None,
    value: None,
    name: Some(group.label()),
    operator: Some("onnx_repeated_block".to_owned()),
    origin: None,
    rank,
    order: 0,
    x: 0.0,
    y: 0.0,
    width: 210.0,
    height: 72.0,
    hidden_initializers: 0,
  }
}

fn hidden_initializers_for_node(model: &Model, graph: usize, node: &Node) -> usize {
  model.graphs.get(graph).map_or(0, |graph| {
    node
      .inputs
      .iter()
      .flatten()
      .filter(|value| graph.values[value.index()].initializer.is_some())
      .count()
  })
}

struct OnnxStructuralLayoutEdgeContext<'a> {
  model: &'a Model,
  graph: &'a netron_rs_core::Graph,
  node_groups: &'a BTreeMap<usize, usize>,
  group_nodes: &'a BTreeMap<usize, usize>,
  operator_nodes: &'a BTreeMap<usize, usize>,
  value_nodes: &'a BTreeMap<usize, usize>,
  nodes: &'a [LayoutNode],
}

fn onnx_structural_layout_edges(
  context: OnnxStructuralLayoutEdgeContext<'_>,
  cancel: Option<&CancellationToken>,
  stats: &mut LayoutStats,
) -> Result<Vec<LayoutEdge>, ProjectionError> {
  let mut edges = Vec::new();
  for value in &context.graph.values {
    check_canceled(cancel)?;
    let source = value.producer.map_or_else(
      || context.value_nodes.get(&value.id.index()).copied(),
      |producer| {
        onnx_structural_layout_node_index(
          producer.index(),
          context.node_groups,
          context.group_nodes,
          context.operator_nodes,
        )
      },
    );
    for consumer in &value.consumers {
      check_canceled(cancel)?;
      let target = onnx_structural_layout_node_index(
        consumer.index(),
        context.node_groups,
        context.group_nodes,
        context.operator_nodes,
      );
      push_onnx_structural_layout_edge(
        context.model,
        value,
        source,
        target,
        context.nodes,
        &mut edges,
        stats,
      );
    }
    if value.is_graph_output {
      push_onnx_structural_layout_edge(
        context.model,
        value,
        source,
        context.value_nodes.get(&value.id.index()).copied(),
        context.nodes,
        &mut edges,
        stats,
      );
    }
  }
  Ok(edges)
}

fn onnx_structural_layout_node_index(
  node: usize,
  node_groups: &BTreeMap<usize, usize>,
  group_nodes: &BTreeMap<usize, usize>,
  operator_nodes: &BTreeMap<usize, usize>,
) -> Option<usize> {
  node_groups
    .get(&node)
    .and_then(|group| group_nodes.get(group))
    .copied()
    .or_else(|| operator_nodes.get(&node).copied())
}

fn push_onnx_structural_layout_edge(
  model: &Model,
  value: &Value,
  source: Option<usize>,
  target: Option<usize>,
  nodes: &[LayoutNode],
  edges: &mut Vec<LayoutEdge>,
  stats: &mut LayoutStats,
) {
  let Some(source) = source else {
    stats.omitted_edges += 1;
    return;
  };
  let Some(target) = target else {
    stats.omitted_edges += 1;
    return;
  };
  if source == target {
    return;
  }
  edges.push(layout_edge(
    nodes,
    source,
    target,
    edges.len(),
    model.strings.get(value.name),
  ));
}

fn mlir_function_collapsed_groups(
  model: &Model,
  function_index: usize,
) -> Option<Vec<CollapsedGroup>> {
  let function = model.functions.get(function_index)?;
  let scope = mlir_function_scope(function_index);
  let block_handle = EntityHandle::MlirBlock { scope, block: 0 };
  Some(
    collapsed_group_metadata(
      "mlir_block",
      block_handle.clone(),
      Some(format!("function:{function_index} block 0")),
      function.nodes.len() + function.values.len(),
      block_handle,
    )
    .into_iter()
    .collect(),
  )
}

fn mlir_module_collapsed_groups(model: &Model, module_index: usize) -> Option<Vec<CollapsedGroup>> {
  let graph = model.graphs.get(module_index)?;
  let mut groups = Vec::new();
  let scope = mlir_module_scope(module_index);
  let block_handle = EntityHandle::MlirBlock {
    scope: scope.clone(),
    block: 0,
  };
  if let Some(group) = collapsed_group_metadata(
    "mlir_block",
    block_handle.clone(),
    Some(format!("{scope} block 0")),
    graph.nodes.len() + graph.values.len(),
    block_handle,
  ) {
    groups.push(group);
  }
  for (function, item) in model.functions.iter().enumerate() {
    if !mlir_function_belongs_to_module(model, item, graph) {
      continue;
    }
    let function_handle = EntityHandle::MlirFunction { function };
    if let Some(group) = collapsed_group_metadata(
      "mlir_function",
      function_handle.clone(),
      Some(model.strings.get(item.name).to_owned()),
      item.nodes.len() + item.values.len(),
      function_handle,
    ) {
      groups.push(group);
    }
  }
  Some(groups)
}

fn visible_collapsed_groups(
  mut groups: Vec<CollapsedGroup>,
  graph: &LayoutGraph,
) -> Vec<CollapsedGroup> {
  let visible = graph
    .nodes
    .iter()
    .map(|node| node.id.as_str())
    .collect::<BTreeSet<_>>();
  groups.retain(|group| visible.contains(group.id.as_str()));
  groups
}

fn mlir_function_structural_layout(
  model: &Model,
  function_index: usize,
  limit: usize,
  cancel: Option<&CancellationToken>,
) -> Result<Option<LayoutGraph>, ProjectionError> {
  check_canceled(cancel)?;
  let Some(function) = model.functions.get(function_index) else {
    return Ok(None);
  };
  let scope = mlir_function_scope(function_index);
  let capacity = limit.max(1);
  let mut nodes = vec![layout_node(MlirLayoutNodeSpec {
    id: handle_key(&EntityHandle::MlirFunction {
      function: function_index,
    }),
    graph: function_index,
    kind: LayoutNodeKind::Group,
    rank: 0,
    value: None,
    name: Some(model.strings.get(function.name).to_owned()),
    operator: Some("mlir_function".to_owned()),
    order: 0,
    node: None,
  })];
  let mut omitted_nodes = 0usize;
  let mut child_indices = Vec::new();
  check_canceled(cancel)?;
  if nodes.len() < capacity {
    child_indices.push(nodes.len());
    nodes.push(layout_node(MlirLayoutNodeSpec {
      id: handle_key(&EntityHandle::MlirBlock {
        scope: scope.clone(),
        block: 0,
      }),
      graph: function_index,
      kind: LayoutNodeKind::Group,
      rank: 1,
      value: None,
      name: Some(format!("{scope} block 0")),
      operator: Some("mlir_block".to_owned()),
      order: 1,
      node: None,
    }));
  } else {
    omitted_nodes = 1;
  }
  check_canceled(cancel)?;
  position_layout_nodes(&mut nodes);
  let edges = child_indices
    .into_iter()
    .enumerate()
    .map(|(index, child)| layout_edge(&nodes, 0, child, index, "contains"))
    .collect::<Vec<_>>();
  let stats = LayoutStats {
    nodes_used: nodes.len(),
    edges_used: edges.len(),
    omitted_nodes,
    omitted_edges: omitted_nodes,
    ..LayoutStats::default()
  };
  Ok(Some(LayoutGraph {
    graph: function_index,
    parent: None,
    name: Some(model.strings.get(function.name).to_owned()),
    subgraphs: Vec::new(),
    bounds: layout_bounds(&nodes),
    nodes,
    edges,
    stats,
  }))
}

fn mlir_module_structural_layout(
  model: &Model,
  module_index: usize,
  limit: usize,
  cancel: Option<&CancellationToken>,
) -> Result<Option<LayoutGraph>, ProjectionError> {
  check_canceled(cancel)?;
  let Some(graph) = model.graphs.get(module_index) else {
    return Ok(None);
  };
  let scope = mlir_module_scope(module_index);
  let capacity = limit.max(1);
  let mut nodes = vec![layout_node(MlirLayoutNodeSpec {
    id: handle_key(&EntityHandle::MlirModule {
      module: module_index,
    }),
    graph: module_index,
    kind: LayoutNodeKind::Group,
    rank: 0,
    value: None,
    name: graph.name.map(|name| model.strings.get(name).to_owned()),
    operator: Some("mlir_module".to_owned()),
    order: 0,
    node: None,
  })];
  let mut omitted_nodes = 0usize;
  let mut child_indices = Vec::new();
  let module_item_count = graph.nodes.len() + graph.values.len();
  check_canceled(cancel)?;
  if module_item_count > 0 {
    if nodes.len() < capacity {
      child_indices.push(nodes.len());
      nodes.push(layout_node(MlirLayoutNodeSpec {
        id: handle_key(&EntityHandle::MlirBlock {
          scope: scope.clone(),
          block: 0,
        }),
        graph: module_index,
        kind: LayoutNodeKind::Group,
        rank: 1,
        value: None,
        name: Some(format!("{scope} block 0")),
        operator: Some("mlir_block".to_owned()),
        order: nodes.len(),
        node: None,
      }));
    } else {
      omitted_nodes += 1;
    }
  }
  for (function, item) in model.functions.iter().enumerate() {
    check_canceled(cancel)?;
    if !mlir_function_belongs_to_module(model, item, graph) {
      continue;
    }
    if nodes.len() >= capacity {
      omitted_nodes += 1;
      continue;
    }
    child_indices.push(nodes.len());
    nodes.push(layout_node(MlirLayoutNodeSpec {
      id: handle_key(&EntityHandle::MlirFunction { function }),
      graph: module_index,
      kind: LayoutNodeKind::Group,
      rank: 1,
      value: None,
      name: Some(model.strings.get(item.name).to_owned()),
      operator: Some("mlir_function".to_owned()),
      order: nodes.len(),
      node: None,
    }));
  }
  check_canceled(cancel)?;
  position_layout_nodes(&mut nodes);
  let mut edges = Vec::new();
  for child in child_indices {
    check_canceled(cancel)?;
    edges.push(layout_edge(&nodes, 0, child, edges.len(), "contains"));
  }
  let stats = LayoutStats {
    nodes_used: nodes.len(),
    edges_used: edges.len(),
    omitted_nodes,
    omitted_edges: omitted_nodes,
    ..LayoutStats::default()
  };
  Ok(Some(LayoutGraph {
    graph: module_index,
    parent: graph.parent.map(|parent| parent.index()),
    name: graph.name.map(|name| model.strings.get(name).to_owned()),
    subgraphs: graph.subgraphs.iter().map(|graph| graph.index()).collect(),
    bounds: layout_bounds(&nodes),
    nodes,
    edges,
    stats,
  }))
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
  collapse: CollapseMode,
) -> String {
  format!(
    "session:{session_id}:format:{format:?}:scope:{}:collapse:{}:{kind}:{limit}",
    handle_key(scope),
    collapse_cache_key(collapse)
  )
}

fn collapse_cache_key(collapse: CollapseMode) -> &'static str {
  match collapse {
    CollapseMode::None => "none",
    CollapseMode::Structural => "structural",
  }
}

fn slice_cache_key(
  session_id: u64,
  format: FormatKind,
  scope: &EntityHandle,
  limit: usize,
  collapse: CollapseMode,
  node_depth: Option<usize>,
) -> String {
  let mut key = projection_cache_key(session_id, format, "slice", scope, limit, collapse);
  if let Some(depth) = node_depth {
    key.push_str(&format!(":depth:{depth}"));
  }
  key
}

fn handle_key(handle: &EntityHandle) -> String {
  match handle {
    EntityHandle::Graph { graph } => format!("graph:{graph}"),
    EntityHandle::Node { graph, node } => format!("node:{graph}:{node}"),
    EntityHandle::Value { graph, value } => format!("value:{graph}:{value}"),
    EntityHandle::Tensor { tensor } => format!("tensor:{tensor}"),
    EntityHandle::Function { function } => format!("function:{function}"),
    EntityHandle::OnnxRepeatedBlock { graph, group } => {
      format!("onnx_repeated_block:{graph}:{group}")
    }
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
  search_index_page(entries, query, 0, limit).0
}

fn search_index_page(
  entries: &[SearchEntry],
  query: &str,
  cursor: usize,
  limit: usize,
) -> (Vec<SearchEntry>, usize) {
  let needle = query.trim().to_lowercase();
  if needle.is_empty() || limit == 0 {
    return (Vec::new(), 0);
  }

  let mut matches = entries
    .iter()
    .enumerate()
    .filter_map(|(index, entry)| entry.score(&needle).map(|score| (score, index)))
    .collect::<Vec<_>>();
  matches.sort_unstable();
  let total_count = matches.len();
  let results = matches
    .into_iter()
    .skip(cursor.min(total_count))
    .take(limit)
    .map(|(_, index)| entries[index].clone())
    .collect();
  (results, total_count)
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticsResponse {
  pub api_version: u32,
  pub session_id: u64,
  pub diagnostics: Vec<Diagnostic>,
  pub truncated: bool,
}

#[derive(Debug)]
pub struct ModelSession {
  id: u64,
  source: ModelSource,
  index: FormatIndex,
  model: Model,
  projection_cache: ProjectionCache,
}

#[derive(Debug, Default)]
struct ProjectionCache {
  slices: RefCell<BTreeMap<String, SliceResponse>>,
  layouts: RefCell<BTreeMap<String, LayoutResponse>>,
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
      projection_cache: ProjectionCache::default(),
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

  pub fn export(&self, limits: &SessionLimits) -> SessionExport<'_> {
    let limit = limits.clamp().export;
    let bounded = self.model.to_bounded_normalized(limit);
    SessionExport {
      api_version: SESSION_API_VERSION,
      session_id: self.id,
      format: self.index.kind(),
      limit_used: limit,
      truncated: bounded.omitted_count > 0,
      omitted_count: bounded.omitted_count,
      normalized: bounded.normalized,
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

  pub fn search_page(
    &self,
    query: &str,
    cursor: Option<usize>,
    limits: &SessionLimits,
  ) -> SearchResponse {
    let limit = limits.clamp().search;
    let cursor = cursor.unwrap_or(0);
    let (results, total_count) = self.index.search_page(query, cursor, limit);
    let next_offset = cursor.saturating_add(results.len());
    let omitted_count = total_count.saturating_sub(next_offset);
    SearchResponse {
      api_version: SESSION_API_VERSION,
      session_id: self.id,
      format: self.index.kind(),
      query: query.to_owned(),
      cursor: (cursor > 0).then_some(cursor),
      next_cursor: (next_offset < total_count).then_some(next_offset),
      limit_used: limit,
      total_count,
      truncated: omitted_count > 0,
      omitted_count,
      results,
    }
  }

  pub fn tensor_metadata(&self, limits: &SessionLimits) -> Vec<TensorMetadata> {
    self.index.tensor_metadata(limits.clamp().detail)
  }

  pub fn tensor_metadata_by_id(&self, tensor: usize) -> Option<TensorMetadata> {
    self.index.tensor_metadata_by_id(tensor)
  }

  pub fn detail(&self, handle: &EntityHandle, limits: &SessionLimits) -> Option<EntityDetail> {
    self
      .index
      .detail(&self.model, handle, limits.clamp().detail)
  }

  pub fn slice(&self, scope: &EntityHandle, limits: &SessionLimits) -> Option<SliceResponse> {
    self.slice_with_depth(scope, limits, DEFAULT_NODE_SLICE_DEPTH)
  }

  pub fn slice_with_depth(
    &self,
    scope: &EntityHandle,
    limits: &SessionLimits,
    node_depth: usize,
  ) -> Option<SliceResponse> {
    self.slice_with_options(
      scope,
      limits,
      ProjectionOptions {
        collapse: CollapseMode::None,
        node_depth,
        cancel: None,
      },
    )
  }

  pub fn slice_with_options(
    &self,
    scope: &EntityHandle,
    limits: &SessionLimits,
    options: ProjectionOptions,
  ) -> Option<SliceResponse> {
    self.try_slice_with_options(scope, limits, options).ok()?
  }

  pub fn try_slice_with_options(
    &self,
    scope: &EntityHandle,
    limits: &SessionLimits,
    options: ProjectionOptions,
  ) -> Result<Option<SliceResponse>, ProjectionError> {
    let limits = limits.clamp();
    check_projection_canceled(&options)?;
    let cacheable = options.node_depth <= MAX_NODE_SLICE_DEPTH;
    let node_depth_key = matches!(scope, EntityHandle::Node { .. })
      .then_some(options.node_depth.min(MAX_NODE_SLICE_DEPTH));
    let cache_key = slice_cache_key(
      self.id,
      self.index.kind(),
      scope,
      limits.slice,
      options.collapse,
      node_depth_key,
    );
    if cacheable && let Some(response) = self.projection_cache.slices.borrow().get(&cache_key) {
      return Ok(Some(response.clone()));
    }
    let Some(response) = session_slice(&self.model, &self.index, self.id, scope, limits, options)?
    else {
      return Ok(None);
    };
    if cacheable {
      self
        .projection_cache
        .slices
        .borrow_mut()
        .insert(cache_key, response.clone());
    }
    Ok(Some(response))
  }

  pub fn layout(&self, scope: &EntityHandle, limits: &SessionLimits) -> Option<LayoutResponse> {
    self.layout_with_options(scope, limits, ProjectionOptions::default())
  }

  pub fn layout_with_options(
    &self,
    scope: &EntityHandle,
    limits: &SessionLimits,
    options: ProjectionOptions,
  ) -> Option<LayoutResponse> {
    self.try_layout_with_options(scope, limits, options).ok()?
  }

  pub fn try_layout_with_options(
    &self,
    scope: &EntityHandle,
    limits: &SessionLimits,
    options: ProjectionOptions,
  ) -> Result<Option<LayoutResponse>, ProjectionError> {
    let limits = limits.clamp();
    check_projection_canceled(&options)?;
    let cache_key = projection_cache_key(
      self.id,
      self.index.kind(),
      "layout",
      scope,
      limits.layout,
      options.collapse,
    );
    if let Some(response) = self.projection_cache.layouts.borrow().get(&cache_key) {
      return Ok(Some(response.clone()));
    }
    let Some(response) = session_layout(&self.model, &self.index, self.id, scope, limits, options)?
    else {
      return Ok(None);
    };
    self
      .projection_cache
      .layouts
      .borrow_mut()
      .insert(cache_key, response.clone());
    Ok(Some(response))
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

  pub fn mlir_symbol_tree(&self, limits: &SessionLimits) -> Option<MlirSymbolTree> {
    let FormatIndex::Mlir(index) = &self.index else {
      return None;
    };
    Some(build_mlir_symbol_tree(
      &index.summary,
      &index.symbols,
      limits.clamp().detail,
    ))
  }
}

fn build_mlir_symbol_tree(
  summary: &MlirSummary,
  symbols: &[String],
  limit: usize,
) -> MlirSymbolTree {
  let full_roots = full_mlir_symbol_tree(summary, symbols);
  let limit = limit.max(1);
  let mut remaining = limit;
  let mut omitted_count = 0;
  let roots = truncate_mlir_symbol_tree_nodes(&full_roots, &mut remaining, &mut omitted_count);
  MlirSymbolTree {
    api_version: SESSION_API_VERSION,
    limit_used: limit,
    truncated: omitted_count > 0,
    omitted_count,
    roots,
  }
}

fn full_mlir_symbol_tree(summary: &MlirSummary, symbols: &[String]) -> Vec<MlirSymbolTreeNode> {
  let mut roots = Vec::new();
  if let Some(modules) = mlir_symbol_tree_scope_group("modules", "Modules", "module", summary) {
    roots.push(modules);
  }
  if let Some(functions) =
    mlir_symbol_tree_scope_group("functions", "Functions", "function", summary)
  {
    roots.push(functions);
  }
  if !symbols.is_empty() {
    roots.push(MlirSymbolTreeNode {
      id: "mlir_tree:symbols".to_owned(),
      kind: "group".to_owned(),
      label: "Symbols".to_owned(),
      handle: None,
      scope_id: None,
      children: symbols
        .iter()
        .enumerate()
        .map(|(symbol, name)| {
          let handle = EntityHandle::MlirSymbol { symbol };
          MlirSymbolTreeNode {
            id: handle_key(&handle),
            kind: "symbol".to_owned(),
            label: mlir_symbol_label(name),
            handle: Some(handle),
            scope_id: None,
            children: Vec::new(),
          }
        })
        .collect(),
    });
  }
  roots
}

fn mlir_symbol_tree_scope_group(
  id: &str,
  label: &str,
  scope_kind: &str,
  summary: &MlirSummary,
) -> Option<MlirSymbolTreeNode> {
  let scopes = match scope_kind {
    "module" => summary.modules.as_slice(),
    "function" => summary.functions.as_slice(),
    _ => &[],
  };
  if scopes.is_empty() {
    return None;
  }
  Some(MlirSymbolTreeNode {
    id: format!("mlir_tree:{id}"),
    kind: "group".to_owned(),
    label: label.to_owned(),
    handle: None,
    scope_id: None,
    children: scopes
      .iter()
      .map(|scope| mlir_symbol_tree_scope_node(scope, scope_kind, summary))
      .collect(),
  })
}

fn mlir_symbol_tree_scope_node(
  scope: &MlirScopeSummary,
  scope_kind: &str,
  summary: &MlirSummary,
) -> MlirSymbolTreeNode {
  MlirSymbolTreeNode {
    id: handle_key(&scope.handle),
    kind: scope_kind.to_owned(),
    label: scope.name.clone().unwrap_or_else(|| scope.scope_id.clone()),
    handle: Some(scope.handle.clone()),
    scope_id: Some(scope.scope_id.clone()),
    children: summary
      .regions
      .iter()
      .filter(|region| region.scope_id == scope.scope_id)
      .map(|region| mlir_symbol_tree_region_node(region, summary))
      .collect(),
  }
}

fn mlir_symbol_tree_region_node(
  region: &MlirRegionSummary,
  summary: &MlirSummary,
) -> MlirSymbolTreeNode {
  let region_index = mlir_region_index(region);
  MlirSymbolTreeNode {
    id: handle_key(&region.handle),
    kind: "region".to_owned(),
    label: region
      .label
      .clone()
      .unwrap_or_else(|| format!("{} region {region_index}", region.scope_id)),
    handle: Some(region.handle.clone()),
    scope_id: Some(region.scope_id.clone()),
    children: summary
      .blocks
      .iter()
      .filter(|block| block.scope_id == region.scope_id && block.region == region_index)
      .map(mlir_symbol_tree_block_node)
      .collect(),
  }
}

fn mlir_symbol_tree_block_node(block: &MlirBlockSummary) -> MlirSymbolTreeNode {
  MlirSymbolTreeNode {
    id: handle_key(&block.handle),
    kind: "block".to_owned(),
    label: block
      .label
      .clone()
      .unwrap_or_else(|| format!("{} block {}", block.scope_id, mlir_block_index(block))),
    handle: Some(block.handle.clone()),
    scope_id: Some(block.scope_id.clone()),
    children: Vec::new(),
  }
}

fn truncate_mlir_symbol_tree_nodes(
  nodes: &[MlirSymbolTreeNode],
  remaining: &mut usize,
  omitted_count: &mut usize,
) -> Vec<MlirSymbolTreeNode> {
  let mut truncated = Vec::new();
  for node in nodes {
    if *remaining == 0 {
      *omitted_count += mlir_symbol_tree_node_count(node);
      continue;
    }
    *remaining -= 1;
    let mut node = node.clone();
    node.children = truncate_mlir_symbol_tree_nodes(&node.children, remaining, omitted_count);
    truncated.push(node);
  }
  truncated
}

fn mlir_symbol_tree_node_count(node: &MlirSymbolTreeNode) -> usize {
  1 + node
    .children
    .iter()
    .map(mlir_symbol_tree_node_count)
    .sum::<usize>()
}

fn mlir_region_index(region: &MlirRegionSummary) -> usize {
  match &region.handle {
    EntityHandle::MlirRegion { region, .. } => *region,
    _ => 0,
  }
}

fn mlir_block_index(block: &MlirBlockSummary) -> usize {
  match &block.handle {
    EntityHandle::MlirBlock { block, .. } => *block,
    _ => 0,
  }
}

fn mlir_symbol_label(name: &str) -> String {
  if name.starts_with('@') {
    name.to_owned()
  } else {
    format!("@{name}")
  }
}

impl FormatIndex {
  fn build(model: &Model, data: Option<&[u8]>) -> Self {
    match model.format.name {
      "ONNX" | "ONNX Tensor" => Self::Onnx(OnnxIndex::build(model)),
      "MLIR" => {
        let source_text = data.and_then(|data| std::str::from_utf8(data).ok());
        let bytecode = data.and_then(|data| netron_rs_formats::inspect_mlir_bytecode(data).ok()?);
        Self::Mlir(MlirIndex::build(model, source_text, bytecode))
      }
      _ => Self::Unknown(ModelIndex::build(model)),
    }
  }

  fn diagnostics(&self) -> &[Diagnostic] {
    match self {
      Self::Onnx(index) => &index.diagnostics,
      Self::Mlir(index) => &index.diagnostics,
      Self::Unknown(_) => &[],
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

  pub fn search_page(&self, query: &str, cursor: usize, limit: usize) -> (Vec<SearchEntry>, usize) {
    match self {
      Self::Onnx(index) => index.search_page(query, cursor, limit),
      Self::Mlir(index) => index.search_page(query, cursor, limit),
      Self::Unknown(index) => index.search_page(query, cursor, limit),
    }
  }

  pub fn tensor_metadata(&self, limit: usize) -> Vec<TensorMetadata> {
    match self {
      Self::Onnx(index) => index.tensor_metadata(limit),
      Self::Mlir(_) | Self::Unknown(_) => Vec::new(),
    }
  }

  pub fn tensor_metadata_by_id(&self, tensor: usize) -> Option<TensorMetadata> {
    match self {
      Self::Onnx(index) => index.tensor_metadata_by_id(tensor),
      Self::Mlir(_) | Self::Unknown(_) => None,
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

    Self { entries }
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

  pub fn search_page(&self, query: &str, cursor: usize, limit: usize) -> (Vec<SearchEntry>, usize) {
    search_index_page(&self.entries, query, cursor, limit)
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
  model
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
    .count()
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
  fn mlir_source_location_parser_extracts_common_targets() {
    assert_eq!(
      mlir_source_location("12:34"),
      Some(DetailLocation {
        kind: "source".to_owned(),
        raw: "12:34".to_owned(),
        file: None,
        line: Some(12),
        column: Some(34),
        end_line: None,
        end_column: None,
        index: None,
        region: None,
        block: None,
      })
    );
    assert_eq!(
      mlir_source_location(r#""kernel.mlir":12:34"#),
      Some(DetailLocation {
        kind: "source".to_owned(),
        raw: r#""kernel.mlir":12:34"#.to_owned(),
        file: Some("kernel.mlir".to_owned()),
        line: Some(12),
        column: Some(34),
        end_line: None,
        end_column: None,
        index: None,
        region: None,
        block: None,
      })
    );
    assert_eq!(
      mlir_source_location(r#"loc("kernel.mlir")"#).and_then(|location| location.file),
      Some("kernel.mlir".to_owned())
    );
    assert_eq!(
      mlir_source_location(r#"loc("kernel.mlir":12:34 to 13:5)"#),
      Some(DetailLocation {
        kind: "source".to_owned(),
        raw: r#"loc("kernel.mlir":12:34 to 13:5)"#.to_owned(),
        file: Some("kernel.mlir".to_owned()),
        line: Some(12),
        column: Some(34),
        end_line: Some(13),
        end_column: Some(5),
        index: None,
        region: None,
        block: None,
      })
    );
    assert_eq!(
      mlir_source_location(r#"loc("kernel.mlir":12:34 to 40)"#),
      Some(DetailLocation {
        kind: "source".to_owned(),
        raw: r#"loc("kernel.mlir":12:34 to 40)"#.to_owned(),
        file: Some("kernel.mlir".to_owned()),
        line: Some(12),
        column: Some(34),
        end_line: Some(12),
        end_column: Some(40),
        index: None,
        region: None,
        block: None,
      })
    );
    assert_eq!(
      mlir_attribute_locations(&[
        r#"loc("kernel.mlir":12:34)"#.to_owned(),
        "dialect:builtin".to_owned()
      ]),
      vec![DetailLocation {
        kind: "source".to_owned(),
        raw: r#"loc("kernel.mlir":12:34)"#.to_owned(),
        file: Some("kernel.mlir".to_owned()),
        line: Some(12),
        column: Some(34),
        end_line: None,
        end_column: None,
        index: None,
        region: None,
        block: None,
      }]
    );

    let fields = BTreeMap::from([
      ("metadata.bytecode.location".to_owned(), "42".to_owned()),
      ("metadata.bytecode.region".to_owned(), "3".to_owned()),
      ("metadata.bytecode.block".to_owned(), "7".to_owned()),
    ]);
    assert_eq!(
      mlir_detail_locations(&fields),
      vec![DetailLocation {
        kind: "bytecode".to_owned(),
        raw: "42".to_owned(),
        file: None,
        line: None,
        column: None,
        end_line: None,
        end_column: None,
        index: Some(42),
        region: Some(3),
        block: Some(7),
      }]
    );
    let fields = BTreeMap::from([
      ("metadata.bytecode.location".to_owned(), "42".to_owned()),
      (
        "metadata.bytecode.location.file".to_owned(),
        "kernel.mlir".to_owned(),
      ),
      (
        "metadata.bytecode.location.line".to_owned(),
        "12".to_owned(),
      ),
      (
        "metadata.bytecode.location.column".to_owned(),
        "34".to_owned(),
      ),
      (
        "metadata.bytecode.location.end_line".to_owned(),
        "13".to_owned(),
      ),
      (
        "metadata.bytecode.location.end_column".to_owned(),
        "5".to_owned(),
      ),
      ("metadata.bytecode.region".to_owned(), "3".to_owned()),
      ("metadata.bytecode.block".to_owned(), "7".to_owned()),
    ]);
    assert_eq!(
      mlir_detail_locations(&fields)
        .into_iter()
        .find(|location| location.kind == "source"),
      Some(DetailLocation {
        kind: "source".to_owned(),
        raw: "kernel.mlir:12:34 to 13:5".to_owned(),
        file: Some("kernel.mlir".to_owned()),
        line: Some(12),
        column: Some(34),
        end_line: Some(13),
        end_column: Some(5),
        index: Some(42),
        region: Some(3),
        block: Some(7),
      })
    );
  }

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
  fn search_page_returns_next_cursor_and_stable_second_page() {
    let model = fixture_model();
    let session = ModelSession {
      id: 77,
      source: ModelSource::from_memory(Some("search.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let limits = SessionLimits {
      search: 1,
      ..SessionLimits::default()
    };

    let first = session.search_page("Add", None, &limits);
    let second = session.search_page("Add", first.next_cursor, &limits);

    assert_eq!(first.api_version, SESSION_API_VERSION);
    assert_eq!(first.session_id, 77);
    assert_eq!(first.format, FormatKind::Unknown);
    assert_eq!(first.query, "Add");
    assert_eq!(first.limit_used, 1);
    assert_eq!(first.total_count, 2);
    assert_eq!(first.cursor, None);
    assert_eq!(first.next_cursor, Some(1));
    assert!(first.truncated);
    assert_eq!(first.omitted_count, 1);
    assert_eq!(first.results[0].kind, SearchKind::Value);
    assert_eq!(first.results[0].name.as_deref(), Some("Add"));

    assert_eq!(second.cursor, Some(1));
    assert_eq!(second.next_cursor, None);
    assert!(!second.truncated);
    assert_eq!(second.omitted_count, 0);
    assert_eq!(second.results[0].kind, SearchKind::Node);
    assert_eq!(second.results[0].operator.as_deref(), Some("Add"));
  }

  #[test]
  fn session_export_builds_bounded_normalized_model() {
    let model = fixture_model();
    let session = ModelSession {
      id: 78,
      source: ModelSource::from_memory(Some("export.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let limits = SessionLimits {
      export: 1,
      ..SessionLimits::default()
    };

    let export = session.export(&limits);
    assert_eq!(export.api_version, SESSION_API_VERSION);
    assert_eq!(export.session_id, 78);
    assert_eq!(export.limit_used, 1);
    assert!(export.truncated);
    assert_eq!(export.omitted_count, 3);

    let json = serde_json::to_value(&export).unwrap();
    let graph = &json["normalized"]["graphs"][0];
    assert_eq!(json["normalized"]["graphs"].as_array().unwrap().len(), 1);
    assert_eq!(graph["values"].as_array().unwrap().len(), 1);
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 1);
    assert_eq!(graph["nodes"][0]["outputs"].as_array().unwrap().len(), 1);
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
    let bytecode = mlir.bytecode.as_ref().expect("bytecode summary");
    assert_eq!(bytecode.version, 6);
    assert_eq!(bytecode.producer, "MLIR19.0.0git");
    assert!(bytecode.string_count > 0);
    assert!(bytecode.operation_name_count > 0);
    assert!(bytecode.decoded_location_count > 0);
    assert!(bytecode.attributes.len() <= bytecode.attribute_count);
    assert!(bytecode.attributes.iter().any(|entry| {
      entry.index < bytecode.attribute_count
        && !entry.dialect.is_empty()
        && entry.len > 0
        && !entry.preview_hex.is_empty()
    }));
    assert!(bytecode.attributes.iter().any(|entry| {
      entry
        .assembly
        .as_deref()
        .is_some_and(|assembly| assembly.starts_with("\"./stable_diffusion_3_medium_diffusers"))
    }));
    assert!(
      bytecode
        .attributes
        .iter()
        .any(|entry| entry.assembly.as_deref() == Some("0.000001 : f64"))
    );
    assert!(bytecode.types.len() <= bytecode.type_count);
    assert!(bytecode.types.iter().any(|entry| {
      entry.index < bytecode.type_count && !entry.dialect.is_empty() && entry.len > 0
    }));
    assert!(bytecode.types.iter().any(|entry| {
      entry
        .assembly
        .as_deref()
        .is_some_and(|assembly| assembly.starts_with('!'))
    }));
    assert!(
      bytecode
        .types
        .iter()
        .any(|entry| entry.assembly.as_deref() == Some("tensor<1536xf16>"))
    );
    assert!(!bytecode.sections.is_empty());
    let FormatIndex::Mlir(index) = session.format() else {
      panic!("expected MLIR index");
    };
    let format_bytecode = index.bytecode().expect("format bytecode summary");
    assert_eq!(
      format_bytecode
        .attributes
        .iter()
        .find(
          |entry| entry.assembly.as_deref() == Some("{torch.assume_strict_symbolic_shapes = unit}")
        )
        .map(|entry| entry.dialect.as_str()),
      Some("builtin")
    );
    let nested_operation = format_bytecode
      .ir
      .operations
      .iter()
      .find(|operation| !operation.nested_region_ids.is_empty())
      .expect("bytecode operation with nested region")
      .clone();
    let nested_region = nested_operation.nested_region_ids[0];
    assert_eq!(nested_operation.name, "builtin.module");
    let nested_region_label = "function:0 builtin.module @compiled_mmdit.body".to_owned();
    assert!(nested_region > 0);
    assert!(nested_region < format_bytecode.ir.region_count);
    assert!(
      mlir
        .functions
        .iter()
        .any(|function| function.name.as_deref() == Some("run_forward"))
    );
    assert_eq!(mlir.modules[0].name.as_deref(), Some("compiled_mmdit"));
    let module_hits = session.search("builtin.module", &SessionLimits::default());
    let module_handle = module_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("builtin.module") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded builtin.module property search hit");
    let module = session
      .detail(&module_handle, &SessionLimits::default())
      .expect("decoded builtin.module detail");
    assert_eq!(
      module
        .fields
        .get("metadata.bytecode.property.sym_name")
        .map(String::as_str),
      Some("compiled_mmdit")
    );
    assert_eq!(
      module
        .fields
        .get("metadata.bytecode.symbol")
        .map(String::as_str),
      Some("compiled_mmdit")
    );
    assert!(mlir.functions[0].operation_count > 0);
    assert_eq!(mlir.regions.len(), mlir.region_count);
    assert_eq!(mlir.blocks.len(), mlir.block_count);
    assert!(mlir.regions.iter().any(|region| matches!(
      region.handle,
      EntityHandle::MlirRegion { ref scope, region: 1 } if scope == "function:0"
    )));
    assert!(
      mlir
        .regions
        .iter()
        .any(|region| region.label.as_deref() == Some(nested_region_label.as_str()))
    );
    assert!(
      mlir.regions.iter().any(|region| {
        region.label.as_deref() == Some("function:0 func.func @run_forward.body")
      })
    );
    let bytecode_block = mlir
      .blocks
      .iter()
      .find(|block| block.region > 0 && block.operation_count > 0)
      .expect("non-entry bytecode block summary");
    let bytecode_block_handle = bytecode_block.handle.clone();
    let bytecode_block_region = bytecode_block.region;
    let bytecode_block_index = match &bytecode_block_handle {
      EntityHandle::MlirBlock { block, .. } => *block,
      _ => panic!("expected bytecode block handle"),
    };
    let bytecode_block_label = format!("^bb{bytecode_block_index}");
    assert_eq!(
      bytecode_block.label.as_deref(),
      Some(bytecode_block_label.as_str())
    );
    assert!(mlir.dialects.iter().any(|dialect| dialect == "torch"));
    assert!(histogram_count(&mlir.histograms.operations, "func.func") > 0);
    assert!(histogram_count(&mlir.histograms.dialects, "torch") > 0);
    let function = &session.model.functions[0];
    assert!(!function.values.is_empty());
    let operation_with_values = function
      .nodes
      .iter()
      .enumerate()
      .find(|(_, node)| !node.inputs.is_empty() || !node.outputs.is_empty())
      .map(|(operation, _)| operation)
      .expect("bytecode operation with operands or results");

    let hits = session.search("func.func", &SessionLimits::default());
    let operation_handle = hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. } if scope == "function:0" => {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("function-scoped bytecode operation search hit");
    let operation = session
      .detail(&operation_handle, &SessionLimits::default())
      .expect("bytecode function operation detail");
    assert_eq!(
      operation.fields.get("operator").map(String::as_str),
      Some("func.func")
    );
    assert!(
      operation
        .fields
        .contains_key("metadata.bytecode.attributes")
    );
    assert_eq!(
      operation
        .fields
        .get("metadata.bytecode.attributes.assembly")
        .map(String::as_str),
      Some("{torch.assume_strict_symbolic_shapes = unit}")
    );
    assert!(
      operation
        .fields
        .contains_key("metadata.bytecode.properties")
    );
    assert!(
      operation
        .fields
        .get("metadata.bytecode.properties.size")
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|size| size > 0)
    );
    assert!(
      operation
        .fields
        .get("metadata.bytecode.properties.preview_hex")
        .is_some_and(|preview| !preview.is_empty())
    );
    assert_eq!(
      operation
        .fields
        .get("metadata.bytecode.symbol")
        .map(String::as_str),
      Some("run_forward")
    );
    assert_eq!(
      operation
        .fields
        .get("metadata.bytecode.property.sym_name")
        .map(String::as_str),
      Some("run_forward")
    );
    assert!(
      operation
        .fields
        .get("metadata.bytecode.property.function_type")
        .is_some_and(|value| value.contains("->"))
    );
    let symbols = session
      .mlir_symbols(&SessionLimits {
        detail: 1000,
        ..SessionLimits::default()
      })
      .expect("MLIR symbols");
    assert!(symbols.iter().any(|symbol| symbol.name == "compiled_mmdit"));
    assert!(symbols.iter().any(|symbol| symbol.name == "run_forward"));
    assert!(
      symbols
        .iter()
        .any(|symbol| symbol.name == "torch.aten._scaled_dot_product_flash_attention_for_cpu")
    );
    assert!(
      symbols
        .iter()
        .any(|symbol| symbol.name == "__auto.mmdit.pos_embed.proj.weight")
    );
    assert!(!symbols.iter().any(|symbol| symbol.name == "bytecode"));
    let symbol_hits = session.search("run_forward", &SessionLimits::default());
    assert!(symbol_hits.iter().any(
      |hit| matches!(hit.handle, EntityHandle::MlirOperation { ref scope, .. } if scope == "function:0")
    ));
    assert!(
      symbol_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirSymbol { .. }))
    );
    let torch_operator_hits = session.search(
      "torch.aten._scaled_dot_product_flash_attention_for_cpu",
      &SessionLimits::default(),
    );
    assert!(
      torch_operator_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirSymbol { .. }))
    );
    let torch_operator_handle = torch_operator_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("torch.operator") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded torch.operator property search hit");
    let torch_operator = session
      .detail(&torch_operator_handle, &SessionLimits::default())
      .expect("decoded torch.operator detail");
    assert_eq!(
      torch_operator
        .fields
        .get("metadata.bytecode.property.name")
        .map(String::as_str),
      Some("torch.aten._scaled_dot_product_flash_attention_for_cpu")
    );
    assert_eq!(
      torch_operator
        .fields
        .get("metadata.bytecode.symbol")
        .map(String::as_str),
      Some("torch.aten._scaled_dot_product_flash_attention_for_cpu")
    );
    let constant_hits = session.search("2 : i64", &SessionLimits::default());
    let constant_handle = constant_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("torch.constant.int") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded torch.constant.int property search hit");
    let constant = session
      .detail(&constant_handle, &SessionLimits::default())
      .expect("decoded torch.constant.int detail");
    assert_eq!(
      constant
        .fields
        .get("metadata.bytecode.property.value")
        .map(String::as_str),
      Some("2 : i64")
    );
    let global_hits = session.search(
      "__auto.mmdit.pos_embed.proj.weight",
      &SessionLimits::default(),
    );
    assert!(
      global_hits
        .iter()
        .any(|hit| matches!(hit.handle, EntityHandle::MlirSymbol { .. }))
    );
    let global_handle = global_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("util.global") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded util.global symbol search hit");
    let global = session
      .detail(&global_handle, &SessionLimits::default())
      .expect("decoded util.global detail");
    assert_eq!(
      global
        .fields
        .get("metadata.bytecode.property.sym_name")
        .map(String::as_str),
      Some("__auto.mmdit.pos_embed.proj.weight")
    );
    assert_eq!(
      global
        .fields
        .get("metadata.bytecode.symbol")
        .map(String::as_str),
      Some("__auto.mmdit.pos_embed.proj.weight")
    );
    let global_load = global_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("util.global.load") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .and_then(|handle| session.detail(&handle, &SessionLimits::default()))
      .expect("decoded util.global.load detail");
    assert_eq!(
      global_load
        .fields
        .get("metadata.bytecode.property.global")
        .map(String::as_str),
      Some("@__auto.mmdit.pos_embed.proj.weight")
    );
    let bytecode_attribute_hits = session.search(
      "torch.assume_strict_symbolic_shapes",
      &SessionLimits::default(),
    );
    let bytecode_attribute_handle = bytecode_attribute_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirAttribute { scope, .. } if scope == "bytecode" => {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded bytecode attribute search hit");
    let bytecode_attribute = session
      .detail(&bytecode_attribute_handle, &SessionLimits::default())
      .expect("decoded bytecode attribute detail");
    assert_eq!(
      bytecode_attribute.fields.get("scope").map(String::as_str),
      Some("bytecode")
    );
    assert_eq!(
      bytecode_attribute.fields.get("name").map(String::as_str),
      Some("bytecode.attr.1415")
    );
    assert_eq!(
      bytecode_attribute.fields.get("value_0").map(String::as_str),
      Some("{torch.assume_strict_symbolic_shapes = unit}")
    );
    let float_attribute_hits = session.search(
      "0.000001",
      &SessionLimits {
        search: 200,
        ..SessionLimits::default()
      },
    );
    assert!(float_attribute_hits.iter().any(
      |hit| matches!(hit.handle, EntityHandle::MlirAttribute { ref scope, .. } if scope == "bytecode")
    ));
    let location_attribute_hits = session.search(
      "loc(\"./stable_diffusion",
      &SessionLimits {
        search: 500,
        ..SessionLimits::default()
      },
    );
    let location_attribute_handle = location_attribute_hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirAttribute { scope, .. } if scope == "bytecode" => {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded bytecode location attribute search hit");
    let location_attribute = session
      .detail(&location_attribute_handle, &SessionLimits::default())
      .expect("decoded bytecode location attribute detail");
    assert!(
      location_attribute
        .fields
        .get("value_0")
        .is_some_and(
          |value| value.starts_with("loc(\"./stable_diffusion") && value.ends_with(":1:1)")
        )
    );
    assert!(location_attribute.locations.iter().any(|location| {
      location.kind == "source"
        && location
          .file
          .as_deref()
          .is_some_and(|file| file.ends_with(".mlir"))
        && location.line == Some(1)
        && location.column == Some(1)
    }));
    assert!(operation.locations.iter().any(|location| {
      location.kind == "source"
        && location
          .file
          .as_deref()
          .is_some_and(|file| file.ends_with(".mlir"))
        && location.line.is_some()
    }));
    assert!(
      operation
        .locations
        .iter()
        .any(|location| location.kind == "bytecode" && location.index.is_some())
    );
    let region_detail = session
      .detail(
        &EntityHandle::MlirRegion {
          scope: "function:0".to_owned(),
          region: 1,
        },
        &SessionLimits::default(),
      )
      .expect("bytecode region detail");
    assert_eq!(
      region_detail.fields.get("region").map(String::as_str),
      Some("1")
    );
    assert!(region_detail.related.iter().any(|handle| matches!(
      handle,
      EntityHandle::MlirBlock { scope, .. } if scope == "function:0"
    )));
    let nested_region_detail = session
      .detail(
        &EntityHandle::MlirRegion {
          scope: "function:0".to_owned(),
          region: nested_region,
        },
        &SessionLimits::default(),
      )
      .expect("nested bytecode region detail");
    assert_eq!(
      nested_region_detail
        .fields
        .get("parent_operation")
        .and_then(|value| value.parse::<usize>().ok()),
      Some(nested_operation.operation)
    );
    assert_eq!(
      nested_region_detail
        .fields
        .get("parent_region")
        .and_then(|value| value.parse::<usize>().ok()),
      Some(nested_operation.region)
    );
    assert_eq!(
      nested_region_detail.fields.get("label").map(String::as_str),
      Some(nested_region_label.as_str())
    );
    let region_label_hits = session.search(&nested_region_label, &SessionLimits::default());
    assert!(region_label_hits.iter().any(|hit| matches!(
      hit.handle,
      EntityHandle::MlirRegion { ref scope, region }
        if scope == "function:0" && region == nested_region
    )));
    assert!(nested_region_detail.related.iter().any(|handle| matches!(
      handle,
      EntityHandle::MlirOperation { scope, operation }
        if scope == "function:0" && *operation == nested_operation.operation
    )));
    let block_detail = session
      .detail(&bytecode_block_handle, &SessionLimits::default())
      .expect("bytecode block detail");
    assert_eq!(
      block_detail
        .fields
        .get("region")
        .and_then(|value| value.parse::<usize>().ok()),
      Some(bytecode_block_region)
    );
    assert_eq!(
      block_detail.fields.get("label").map(String::as_str),
      Some(bytecode_block_label.as_str())
    );
    let block_label_hits = session.search(&bytecode_block_label, &SessionLimits::default());
    assert!(block_label_hits.iter().any(|hit| matches!(
      hit.handle,
      EntityHandle::MlirBlock { ref scope, block }
        if scope == "function:0" && block == bytecode_block_index
    )));
    let block_slice = session
      .slice(&bytecode_block_handle, &SessionLimits::default())
      .expect("bytecode block slice");
    assert!(block_slice.entities.iter().any(|entity| matches!(
      &entity.handle,
      Some(EntityHandle::MlirBlock { scope, .. }) if scope == "function:0"
    )));
    let block_operations = block_slice
      .entities
      .iter()
      .filter_map(|entity| match &entity.handle {
        Some(EntityHandle::MlirOperation { scope, operation }) if scope == "function:0" => {
          Some(*operation)
        }
        _ => None,
      })
      .collect::<Vec<_>>();
    assert!(!block_operations.is_empty());
    for operation in block_operations {
      let detail = session
        .detail(
          &EntityHandle::MlirOperation {
            scope: "function:0".to_owned(),
            operation,
          },
          &SessionLimits::default(),
        )
        .expect("bytecode block operation detail");
      assert_eq!(
        detail
          .fields
          .get("metadata.bytecode.block")
          .and_then(|value| value.parse::<usize>().ok()),
        Some(bytecode_block_index)
      );
    }
    let region_slice = session
      .slice(
        &EntityHandle::MlirRegion {
          scope: "function:0".to_owned(),
          region: bytecode_block_region,
        },
        &SessionLimits::default(),
      )
      .expect("bytecode region slice");
    let region_operations = region_slice
      .entities
      .iter()
      .filter_map(|entity| match &entity.handle {
        Some(EntityHandle::MlirOperation { scope, operation }) if scope == "function:0" => {
          Some(*operation)
        }
        _ => None,
      })
      .collect::<Vec<_>>();
    assert!(!region_operations.is_empty());
    for operation in region_operations {
      let detail = session
        .detail(
          &EntityHandle::MlirOperation {
            scope: "function:0".to_owned(),
            operation,
          },
          &SessionLimits::default(),
        )
        .expect("bytecode region operation detail");
      assert_eq!(
        detail
          .fields
          .get("metadata.bytecode.region")
          .and_then(|value| value.parse::<usize>().ok()),
        Some(bytecode_block_region)
      );
    }
    let operation_with_values = session
      .detail(
        &EntityHandle::MlirOperation {
          scope: "function:0".to_owned(),
          operation: operation_with_values,
        },
        &SessionLimits::default(),
      )
      .expect("bytecode operation with values detail");
    let input_count = operation_with_values
      .fields
      .get("input_count")
      .and_then(|value| value.parse::<usize>().ok())
      .unwrap_or_default();
    let output_count = operation_with_values
      .fields
      .get("output_count")
      .and_then(|value| value.parse::<usize>().ok())
      .unwrap_or_default();
    assert!(input_count + output_count > 0);
    assert!(
      operation_with_values
        .fields
        .contains_key("metadata.bytecode.region")
    );
    assert!(
      operation_with_values
        .fields
        .contains_key("metadata.bytecode.block")
    );
    assert!(operation_with_values.related.iter().any(
      |handle| matches!(handle, EntityHandle::MlirValue { scope, .. } if scope == "function:0")
    ));
    let value_handle = operation_with_values
      .related
      .iter()
      .find_map(|handle| match handle {
        EntityHandle::MlirValue { scope, value } if scope == "function:0" => Some(*value),
        _ => None,
      })
      .expect("bytecode value handle");
    let value_detail = session
      .detail(
        &EntityHandle::MlirValue {
          scope: "function:0".to_owned(),
          value: value_handle,
        },
        &SessionLimits::default(),
      )
      .expect("bytecode value detail");
    assert_eq!(
      value_detail.fields.get("name").map(String::as_str),
      Some(value_detail.title.as_str())
    );
    assert!(
      value_detail
        .fields
        .get("name")
        .is_some_and(|name| name.starts_with('%'))
    );
    assert!(
      value_detail
        .fields
        .get("metadata.bytecode.value")
        .and_then(|value| value.parse::<usize>().ok())
        .is_some()
    );
    assert!(
      value_detail
        .fields
        .get("metadata.bytecode.type")
        .and_then(|value| value.parse::<usize>().ok())
        .is_some()
    );
    let assembly_value = function
      .values
      .iter()
      .enumerate()
      .find_map(|(value, detail)| {
        detail
          .metadata
          .get("bytecode.type.assembly")
          .is_some_and(|assembly| assembly.starts_with('!'))
          .then_some(value)
      })
      .expect("bytecode value with assembly fallback type");
    let assembly_value_detail = session
      .detail(
        &EntityHandle::MlirValue {
          scope: "function:0".to_owned(),
          value: assembly_value,
        },
        &SessionLimits::default(),
      )
      .expect("assembly fallback bytecode value detail");
    assert!(
      assembly_value_detail
        .fields
        .get("metadata.bytecode.type.assembly")
        .is_some_and(|assembly| assembly.starts_with('!'))
    );
    let builtin_value = function
      .values
      .iter()
      .enumerate()
      .find_map(|(value, detail)| {
        detail
          .metadata
          .get("bytecode.type.assembly")
          .is_some_and(|assembly| assembly.starts_with("tensor<"))
          .then_some(value)
      })
      .expect("bytecode value with decoded builtin type");
    let builtin_value_detail = session
      .detail(
        &EntityHandle::MlirValue {
          scope: "function:0".to_owned(),
          value: builtin_value,
        },
        &SessionLimits::default(),
      )
      .expect("decoded builtin bytecode value detail");
    assert!(
      builtin_value_detail
        .fields
        .get("metadata.bytecode.type.assembly")
        .is_some_and(|assembly| assembly.starts_with("tensor<"))
    );
    assert_eq!(
      builtin_value_detail
        .fields
        .get("element_type")
        .map(String::as_str),
      Some("float16")
    );
    assert!(
      builtin_value_detail
        .fields
        .get("shape")
        .is_some_and(|shape| !shape.is_empty())
    );
    let assembly_hits = session.search("!torch", &SessionLimits::default());
    assert!(assembly_hits.iter().any(
      |hit| matches!(hit.handle, EntityHandle::MlirValue { ref scope, .. } if scope == "function:0")
    ));
    let value_hits = session.search("bytecode.value", &SessionLimits::default());
    assert!(value_hits.iter().any(
      |hit| matches!(hit.handle, EntityHandle::MlirValue { ref scope, .. } if scope == "function:0")
    ));
    let nested_operation_detail = session
      .detail(
        &EntityHandle::MlirOperation {
          scope: "function:0".to_owned(),
          operation: nested_operation.operation,
        },
        &SessionLimits::default(),
      )
      .expect("nested bytecode operation detail");
    assert!(
      nested_operation_detail
        .fields
        .get("metadata.bytecode.nested_region_ids")
        .is_some_and(|regions| regions
          .split(',')
          .any(|region| region.parse::<usize>().ok() == Some(nested_region)))
    );
    assert!(nested_operation_detail.related.iter().any(|handle| matches!(
      handle,
      EntityHandle::MlirRegion { scope, region } if scope == "function:0" && *region == nested_region
    )));
    let nested_operation_slice = session
      .slice(
        &EntityHandle::MlirOperation {
          scope: "function:0".to_owned(),
          operation: nested_operation.operation,
        },
        &SessionLimits::default(),
      )
      .expect("nested bytecode operation slice");
    let nested_operation_id = handle_key(&EntityHandle::MlirOperation {
      scope: "function:0".to_owned(),
      operation: nested_operation.operation,
    });
    let nested_region_id = handle_key(&EntityHandle::MlirRegion {
      scope: "function:0".to_owned(),
      region: nested_region,
    });
    assert!(nested_operation_slice.entities.iter().any(|entity| {
      matches!(
        &entity.handle,
        Some(EntityHandle::MlirRegion { scope, region })
          if scope == "function:0" && *region == nested_region
      )
    }));
    assert!(
      nested_operation_slice
        .edges
        .iter()
        .any(|edge| { edge.from == nested_operation_id && edge.to == nested_region_id })
    );
    let function_slice = session
      .slice(
        &EntityHandle::MlirFunction { function: 0 },
        &SessionLimits::default(),
      )
      .expect("bytecode function slice");
    assert!(function_slice.entities.iter().any(|entity| matches!(
      &entity.handle,
      Some(EntityHandle::MlirOperation { scope, .. }) if scope == "function:0"
    )));
    assert!(
      function_slice
        .entities
        .iter()
        .any(|entity| entity.kind == "mlir_value")
    );
    assert!(!function_slice.edges.is_empty());
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
    let function_names = mlir
      .functions
      .iter()
      .filter_map(|function| function.name.as_deref())
      .collect::<Vec<_>>();
    assert!(function_names.contains(&"main"));
    assert!(function_names.contains(&"forward"));
    assert_eq!(mlir.resources[0].name, "torch_tensor_1_77_torch.int64");
    assert_eq!(
      mlir.resources[0].handle,
      EntityHandle::MlirResource { resource: 0 }
    );

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
  fn mlir_bytecode_vtensor_literal_property_is_searchable_and_detailable() {
    let path = mlirbc_fixture("sd-clip-tank.mlirbc");
    let data = std::fs::read(&path).expect("fixture exists");
    let session =
      ModelSession::open(&data, ModelSource::from_file(path, data.len())).expect("open fixture");

    let hits = session.search(
      "torch_tensor_1_77_torch.int64",
      &SessionLimits {
        search: 100,
        ..SessionLimits::default()
      },
    );
    let literal_handle = hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("torch.vtensor.literal") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded torch.vtensor.literal value search hit");
    let literal = session
      .detail(&literal_handle, &SessionLimits::default())
      .expect("decoded torch.vtensor.literal detail");

    assert_eq!(
      literal
        .fields
        .get("metadata.bytecode.property.value")
        .map(String::as_str),
      Some("dense_resource<torch_tensor_1_77_torch.int64> : tensor<1x77xsi64>")
    );
  }

  #[test]
  fn mlir_bytecode_constant_none_is_searchable_and_detailable() {
    let path = mlirbc_fixture("model.mlirbc");
    let data = std::fs::read(&path).expect("fixture exists");
    let session =
      ModelSession::open(&data, ModelSource::from_file(path, data.len())).expect("open fixture");

    let hits = session.search(
      "bytecode.constant.value",
      &SessionLimits {
        search: 100,
        ..SessionLimits::default()
      },
    );
    let none_handle = hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("torch.constant.none") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded torch.constant.none metadata search hit");
    let none = session
      .detail(&none_handle, &SessionLimits::default())
      .expect("decoded torch.constant.none detail");

    assert_eq!(
      none
        .fields
        .get("metadata.bytecode.constant.value")
        .map(String::as_str),
      Some("none")
    );
    assert_eq!(
      none
        .fields
        .get("metadata.bytecode.constant.type")
        .map(String::as_str),
      Some("none")
    );
    assert!(!none.fields.contains_key("metadata.bytecode.properties"));
  }

  #[test]
  fn mlir_bytecode_func_call_callee_property_is_searchable_and_detailable() {
    let path = mlirbc_fixture("sd-clip-tank.mlirbc");
    let data = std::fs::read(&path).expect("fixture exists");
    let session =
      ModelSession::open(&data, ModelSource::from_file(path, data.len())).expect("open fixture");

    let hits = session.search(
      "@forward",
      &SessionLimits {
        search: 100,
        ..SessionLimits::default()
      },
    );
    let call_handle = hits
      .iter()
      .find_map(|hit| match &hit.handle {
        EntityHandle::MlirOperation { scope, .. }
          if scope == "function:0" && hit.operator.as_deref() == Some("func.call") =>
        {
          Some(hit.handle.clone())
        }
        _ => None,
      })
      .expect("decoded func.call callee search hit");
    let call = session
      .detail(&call_handle, &SessionLimits::default())
      .expect("decoded func.call detail");

    assert_eq!(
      call
        .fields
        .get("metadata.bytecode.property.callee")
        .map(String::as_str),
      Some("@forward")
    );
    assert!(!call.fields.contains_key("metadata.bytecode.symbol"));
  }

  #[test]
  fn mlir_bytecode_index_does_not_read_model_metadata_properties() {
    let path = mlirbc_fixture("sd-clip-tank.mlirbc");
    let data = std::fs::read(&path).expect("fixture exists");
    let mut model = netron_rs_formats::parse(netron_rs_core::ModelInput {
      data: &data,
      path: Some(path.as_path()),
      allow_unsafe_paths: false,
    })
    .expect("parse fixture");
    model.metadata.properties.clear();

    let index = MlirIndex::build(
      &model,
      None,
      Some(
        netron_rs_formats::inspect_mlir_bytecode(&data)
          .expect("inspect bytecode")
          .expect("bytecode summary"),
      ),
    );
    assert!(index.bytecode().is_some());
    assert!(model.metadata.properties.is_empty());
    assert_eq!(index.summary.resource_count, 1);
    assert_eq!(
      index.summary.resources[0].name,
      "torch_tensor_1_77_torch.int64"
    );
    assert_eq!(index.summary.bytecode.as_ref().unwrap().version, 6);
    assert!(
      index
        .search("torch_tensor_1_77_torch.int64", 10)
        .iter()
        .any(|entry| matches!(entry.handle, EntityHandle::MlirResource { resource: 0 }))
    );
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
    assert_eq!(mlir.resources[0].name, "blob");
    assert_eq!(mlir.resources[0].kind, "rodata");
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
    assert_eq!(
      histogram_count(&mlir.histograms.operations, "stablehlo.constant"),
      1
    );
    assert_eq!(histogram_count(&mlir.histograms.dialects, "arith"), 1);
    assert_eq!(histogram_count(&mlir.histograms.dialects, "stablehlo"), 1);
    assert_eq!(histogram_count(&mlir.histograms.dialects, "vm"), 1);

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
  fn mlir_symbol_tree_reports_backend_navigation_and_limits() {
    let data = br#"module @jit_mlp {
  func.func @loop(%i: index) {
    return
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("tree.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");

    let tree = session
      .mlir_symbol_tree(&SessionLimits::default())
      .expect("MLIR symbol tree");
    assert_eq!(tree.api_version, SESSION_API_VERSION);
    assert!(!tree.truncated);
    assert_eq!(tree.omitted_count, 0);
    assert_eq!(
      tree
        .roots
        .iter()
        .map(|node| node.label.as_str())
        .collect::<Vec<_>>(),
      vec!["Modules", "Functions", "Symbols"]
    );

    let modules = tree
      .roots
      .iter()
      .find(|node| node.label == "Modules")
      .expect("module group");
    let module = modules.children.first().expect("module node");
    assert_eq!(module.kind, "module");
    assert_eq!(module.label, "@jit_mlp");
    assert!(matches!(
      &module.handle,
      Some(EntityHandle::MlirModule { module: 0 })
    ));
    assert!(
      module
        .children
        .iter()
        .any(|node| node.kind == "region" && !node.children.is_empty())
    );

    let functions = tree
      .roots
      .iter()
      .find(|node| node.label == "Functions")
      .expect("function group");
    let function = functions.children.first().expect("function node");
    assert_eq!(function.kind, "function");
    assert!(function.label.ends_with("@loop"));
    assert!(matches!(
      &function.handle,
      Some(EntityHandle::MlirFunction { function: 0 })
    ));
    assert!(
      function
        .children
        .iter()
        .any(|node| node.kind == "region" && !node.children.is_empty())
    );

    let symbols = tree
      .roots
      .iter()
      .find(|node| node.label == "Symbols")
      .expect("symbol group");
    assert!(symbols.children.iter().any(|node| {
      node.kind == "symbol"
        && node.label == "@jit_mlp"
        && matches!(&node.handle, Some(EntityHandle::MlirSymbol { .. }))
    }));

    let limited = session
      .mlir_symbol_tree(&SessionLimits {
        detail: 2,
        ..SessionLimits::default()
      })
      .expect("limited MLIR symbol tree");
    assert_eq!(limited.limit_used, 2);
    assert!(limited.truncated);
    assert!(limited.omitted_count > 0);
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
    assert!(operation_detail.locations.iter().any(|location| {
      location.kind == "source" && location.line == Some(3) && location.column.is_some()
    }));
    assert!(
      operation_detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::MlirValue { .. }))
    );

    let located_operation_handle = session
      .search("arith.index_cast", &SessionLimits::default())
      .into_iter()
      .find_map(|hit| {
        if let EntityHandle::MlirOperation { .. } = hit.handle {
          Some(hit.handle)
        } else {
          None
        }
      })
      .expect("located operation hit");
    let located_operation_detail = session
      .detail(&located_operation_handle, &SessionLimits::default())
      .expect("located operation detail");
    assert_eq!(
      located_operation_detail
        .locations
        .first()
        .and_then(|location| location.file.as_deref()),
      Some("kernel.mlir")
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
      projection_cache: ProjectionCache::default(),
    };

    let summary_json = serde_json::to_value(session.summary(&SessionLimits::default())).unwrap();
    assert!(summary_json.get("layout").is_none());
    assert!(summary_json.get("slice").is_none());
    assert_eq!(session.projection_cache.slices.borrow().len(), 0);
    assert_eq!(session.projection_cache.layouts.borrow().len(), 0);

    let slice = session
      .slice(
        &EntityHandle::Node { graph: 0, node: 0 },
        &SessionLimits {
          slice: 2,
          ..SessionLimits::default()
        },
      )
      .expect("node slice");
    assert_eq!(session.projection_cache.slices.borrow().len(), 1);
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
    let repeated_slice = session
      .slice(
        &EntityHandle::Node { graph: 0, node: 0 },
        &SessionLimits {
          slice: 2,
          ..SessionLimits::default()
        },
      )
      .expect("cached node slice");
    assert_eq!(repeated_slice.cache_key, slice.cache_key);
    assert_eq!(session.projection_cache.slices.borrow().len(), 1);

    let graph_slice = session
      .slice(
        &EntityHandle::Graph { graph: 0 },
        &SessionLimits {
          slice: 4,
          ..SessionLimits::default()
        },
      )
      .expect("graph slice");
    assert_eq!(session.projection_cache.slices.borrow().len(), 2);
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
    assert_eq!(session.projection_cache.layouts.borrow().len(), 1);
    assert_eq!(layout.limit_used, 1);
    assert!(layout.truncated);
    assert_eq!(layout.graph.stats.omitted_nodes, 1);
    assert!(layout.cache_key.contains("format:Onnx"));
    let repeated_layout = session
      .layout(
        &EntityHandle::Graph { graph: 0 },
        &SessionLimits {
          layout: 1,
          ..SessionLimits::default()
        },
      )
      .expect("cached graph layout");
    assert_eq!(repeated_layout.cache_key, layout.cache_key);
    assert_eq!(session.projection_cache.layouts.borrow().len(), 1);
  }

  #[test]
  fn try_layout_with_options_reports_canceled_without_caching() {
    let model = fixture_two_node_onnx_model();
    let session = ModelSession {
      id: 107,
      source: ModelSource::from_memory(Some("canceled-layout.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let cancel = CancellationToken::default();
    cancel.cancel();
    let result = session.try_layout_with_options(
      &EntityHandle::Graph { graph: 0 },
      &SessionLimits::default(),
      ProjectionOptions {
        cancel: Some(cancel),
        ..ProjectionOptions::default()
      },
    );

    assert!(matches!(result, Err(ProjectionError::Canceled)));
    assert_eq!(session.projection_cache.layouts.borrow().len(), 0);
  }

  #[test]
  fn try_slice_with_options_reports_canceled_without_caching() {
    let model = fixture_two_node_onnx_model();
    let session = ModelSession {
      id: 108,
      source: ModelSource::from_memory(Some("canceled-slice.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let cancel = CancellationToken::default();
    cancel.cancel();
    let result = session.try_slice_with_options(
      &EntityHandle::Graph { graph: 0 },
      &SessionLimits::default(),
      ProjectionOptions {
        cancel: Some(cancel),
        ..ProjectionOptions::default()
      },
    );

    assert!(matches!(result, Err(ProjectionError::Canceled)));
    assert_eq!(session.projection_cache.slices.borrow().len(), 0);
  }

  #[test]
  fn onnx_node_slice_depth_controls_neighborhood() {
    let model = fixture_three_node_onnx_model();
    let session = ModelSession {
      id: 106,
      source: ModelSource::from_memory(Some("chain.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let limits = SessionLimits {
      slice: 32,
      ..SessionLimits::default()
    };

    let depth_zero = session
      .slice_with_depth(&EntityHandle::Node { graph: 0, node: 0 }, &limits, 0)
      .expect("depth zero slice");
    let depth_zero_nodes = slice_node_ids(&depth_zero);
    assert_eq!(depth_zero_nodes, BTreeSet::from([0]));
    assert!(depth_zero.cache_key.contains("depth:0"));

    let depth_one = session
      .slice_with_depth(&EntityHandle::Node { graph: 0, node: 0 }, &limits, 1)
      .expect("depth one slice");
    let depth_one_nodes = slice_node_ids(&depth_one);
    assert!(depth_one_nodes.contains(&0));
    assert!(depth_one_nodes.contains(&1));
    assert!(!depth_one_nodes.contains(&2));
    assert!(depth_one.cache_key.contains("depth:1"));

    let depth_two = session
      .slice_with_depth(&EntityHandle::Node { graph: 0, node: 0 }, &limits, 2)
      .expect("depth two slice");
    let depth_two_nodes = slice_node_ids(&depth_two);
    assert!(depth_two_nodes.contains(&0));
    assert!(depth_two_nodes.contains(&1));
    assert!(depth_two_nodes.contains(&2));
    assert!(depth_two.cache_key.contains("depth:2"));
  }

  #[test]
  fn onnx_structural_collapse_slice_reports_repeated_blocks() {
    let model = fixture_repeated_pair_onnx_model();
    let session = ModelSession {
      id: 109,
      source: ModelSource::from_memory(Some("repeated-pair.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let scope = EntityHandle::Graph { graph: 0 };
    let flat = session
      .slice(&scope, &SessionLimits::default())
      .expect("flat slice");
    assert_eq!(flat.collapse, CollapseMode::None);
    assert!(flat.collapsed_groups.is_empty());
    assert_eq!(slice_node_ids(&flat), BTreeSet::from([0, 1, 2, 3]));

    let structural = session
      .slice_with_options(
        &scope,
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");

    assert_eq!(structural.collapse, CollapseMode::Structural);
    assert!(structural.cache_key.contains("collapse:structural"));
    assert_eq!(structural.collapsed_groups.len(), 1);
    let group = &structural.collapsed_groups[0];
    assert_eq!(group.kind, "onnx_repeated_block");
    assert_eq!(group.item_count, 4);
    assert!(matches!(
      group.handle,
      EntityHandle::OnnxRepeatedBlock { graph: 0, group: 0 }
    ));
    assert_eq!(group.handle, group.expand);
    assert!(
      structural
        .entities
        .iter()
        .any(|entity| entity.kind == "onnx_repeated_block")
    );
    assert!(slice_node_ids(&structural).is_empty());

    let detail = session
      .detail(&group.expand, &SessionLimits::default())
      .expect("repeated block detail");
    assert_eq!(
      detail.fields.get("operator_path").map(String::as_str),
      Some("Relu -> Add")
    );
    assert_eq!(
      detail.fields.get("instance_count").map(String::as_str),
      Some("2")
    );
    assert_eq!(
      detail.fields.get("nodes_per_instance").map(String::as_str),
      Some("2")
    );
    assert!(
      detail
        .related
        .iter()
        .any(|handle| matches!(handle, EntityHandle::Node { node: 0, .. }))
    );

    let expanded = session
      .slice(&group.expand, &SessionLimits::default())
      .expect("repeated block expansion slice");
    assert_eq!(expanded.collapse, CollapseMode::None);
    assert_eq!(slice_node_ids(&expanded), BTreeSet::from([0, 1, 2, 3]));
  }

  #[test]
  fn onnx_structural_collapse_prefers_repeated_linear_motifs() {
    let model = fixture_repeated_triple_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].operator_path, "Relu -> Add -> Sigmoid");
    assert_eq!(groups[0].width(), 3);
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 1, 2], vec![3, 4, 5]]
    );

    let session = ModelSession {
      id: 111,
      source: ModelSource::from_memory(Some("repeated-triple.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let scope = EntityHandle::Graph { graph: 0 };
    let structural = session
      .slice_with_options(
        &scope,
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].item_count, 6);
    assert!(slice_node_ids(&structural).is_empty());

    let detail = session
      .detail(
        &structural.collapsed_groups[0].expand,
        &SessionLimits::default(),
      )
      .expect("repeated triple detail");
    assert_eq!(
      detail.fields.get("operator_path").map(String::as_str),
      Some("Relu -> Add -> Sigmoid")
    );
    assert_eq!(
      detail.fields.get("nodes_per_instance").map(String::as_str),
      Some("3")
    );

    let expanded = session
      .slice(
        &structural.collapsed_groups[0].expand,
        &SessionLimits::default(),
      )
      .expect("repeated triple expansion slice");
    assert_eq!(
      slice_node_ids(&expanded),
      BTreeSet::from([0, 1, 2, 3, 4, 5])
    );
  }

  #[test]
  fn onnx_structural_collapse_detects_larger_linear_motifs() {
    let model = fixture_repeated_six_node_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");
    assert_eq!(groups.len(), 1);
    assert_eq!(
      groups[0].operator_path,
      "Relu -> Add -> Sigmoid -> Tanh -> Mul -> Identity"
    );
    assert_eq!(groups[0].width(), 6);
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]]
    );

    let session = ModelSession {
      id: 116,
      source: ModelSource::from_memory(Some("repeated-six.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let structural = session
      .slice_with_options(
        &EntityHandle::Graph { graph: 0 },
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].item_count, 12);
    assert!(slice_node_ids(&structural).is_empty());

    let detail = session
      .detail(
        &structural.collapsed_groups[0].expand,
        &SessionLimits::default(),
      )
      .expect("larger repeated motif detail");
    assert_eq!(
      detail.fields.get("nodes_per_instance").map(String::as_str),
      Some("6")
    );
  }

  #[test]
  fn onnx_structural_collapse_detects_repeated_branch_motifs() {
    let model = fixture_repeated_branch_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].operator_path, "Relu -> Tanh -> Sigmoid -> Add");
    assert_eq!(groups[0].width(), 4);
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]
    );
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.internal_values.len())
        .collect::<Vec<_>>(),
      vec![3, 3]
    );

    let session = ModelSession {
      id: 112,
      source: ModelSource::from_memory(Some("repeated-branch.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let scope = EntityHandle::Graph { graph: 0 };
    let structural = session
      .slice_with_options(
        &scope,
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].item_count, 8);
    assert!(slice_node_ids(&structural).is_empty());

    let expanded = session
      .slice(
        &structural.collapsed_groups[0].expand,
        &SessionLimits::default(),
      )
      .expect("repeated branch expansion slice");
    assert_eq!(
      slice_node_ids(&expanded),
      BTreeSet::from([0, 1, 2, 3, 4, 5, 6, 7])
    );
  }

  #[test]
  fn onnx_structural_collapse_detects_branch_motifs_with_swapped_node_order() {
    let model = fixture_swapped_order_repeated_branch_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].operator_path, "Relu -> Tanh -> Sigmoid -> Add");
    assert_eq!(groups[0].width(), 4);
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]
    );

    let session = ModelSession {
      id: 117,
      source: ModelSource::from_memory(Some("swapped-order-repeated-branch.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let structural = session
      .slice_with_options(
        &EntityHandle::Graph { graph: 0 },
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].item_count, 8);
    assert!(slice_node_ids(&structural).is_empty());
  }

  #[test]
  fn onnx_structural_collapse_detects_interleaved_private_motifs() {
    let model = fixture_interleaved_repeated_pair_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].operator_path, "Relu -> Tanh");
    assert_eq!(groups[0].width(), 2);
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 2], vec![1, 3]]
    );

    let session = ModelSession {
      id: 113,
      source: ModelSource::from_memory(Some("interleaved-repeated-pair.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let scope = EntityHandle::Graph { graph: 0 };
    let structural = session
      .slice_with_options(
        &scope,
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].item_count, 4);
    assert!(slice_node_ids(&structural).is_empty());

    let expanded = session
      .slice(
        &structural.collapsed_groups[0].expand,
        &SessionLimits::default(),
      )
      .expect("interleaved repeated pair expansion slice");
    assert_eq!(slice_node_ids(&expanded), BTreeSet::from([0, 1, 2, 3]));
  }

  #[test]
  fn onnx_structural_collapse_detects_shared_boundary_motifs() {
    let model = fixture_shared_boundary_repeated_pair_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].operator_path, "Relu -> Add");
    assert_eq!(groups[0].width(), 2);
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 1], vec![3, 4]]
    );

    let session = ModelSession {
      id: 114,
      source: ModelSource::from_memory(Some("shared-boundary-repeated.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let scope = EntityHandle::Graph { graph: 0 };
    let structural = session
      .slice_with_options(
        &scope,
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX slice");
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].item_count, 4);
    assert_eq!(slice_node_ids(&structural), BTreeSet::from([2, 5]));

    let group_id = handle_key(&EntityHandle::OnnxRepeatedBlock { graph: 0, group: 0 });
    assert!(structural.edges.iter().any(|edge| {
      edge.from == group_id && edge.to == handle_key(&EntityHandle::Node { graph: 0, node: 2 })
    }));
    assert!(structural.edges.iter().any(|edge| {
      edge.from == group_id && edge.to == handle_key(&EntityHandle::Node { graph: 0, node: 5 })
    }));

    let expanded = session
      .slice(
        &structural.collapsed_groups[0].expand,
        &SessionLimits::default(),
      )
      .expect("shared-boundary repeated pair expansion slice");
    assert_eq!(slice_node_ids(&expanded), BTreeSet::from([0, 1, 3, 4]));
  }

  #[test]
  fn onnx_structural_collapse_layout_uses_group_nodes_and_cache_key() {
    let model = fixture_repeated_pair_onnx_model();
    let session = ModelSession {
      id: 110,
      source: ModelSource::from_memory(Some("repeated-layout.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
    };
    let scope = EntityHandle::Graph { graph: 0 };
    let flat = session
      .layout(&scope, &SessionLimits::default())
      .expect("flat layout");
    let structural = session
      .layout_with_options(
        &scope,
        &SessionLimits::default(),
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural ONNX layout");

    assert_ne!(flat.cache_key, structural.cache_key);
    assert!(flat.cache_key.contains("collapse:none"));
    assert!(structural.cache_key.contains("collapse:structural"));
    assert_eq!(structural.collapse, CollapseMode::Structural);
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert!(
      structural
        .graph
        .nodes
        .iter()
        .any(|node| node.kind == LayoutNodeKind::Group && node.id == "onnx_repeated_block:0:0")
    );
    assert!(structural.graph.nodes.len() < flat.graph.nodes.len());
    assert!(structural.graph.stats.omitted_nodes >= 3);
  }

  #[test]
  fn onnx_repeated_block_detector_keeps_non_overlapping_instances() {
    let model = fixture_overlapping_repeated_pair_onnx_model();
    let groups = onnx_repeated_blocks(&model, 0).expect("graph exists");

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].group, 0);
    assert_eq!(groups[0].operator_path, "Relu -> Relu");
    assert_eq!(
      groups[0]
        .instances
        .iter()
        .map(|instance| instance.nodes.clone())
        .collect::<Vec<_>>(),
      vec![vec![0, 1], vec![2, 3]]
    );
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
  fn mlir_structural_collapse_slice_reports_expandable_group() {
    let data = br#"module {
  func.func @main(%i: index) -> i64 {
    %0 = arith.index_cast %i : index to i64
    %1 = arith.addi %0, %0 : i64
    return %1 : i64
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("structural-slice.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");
    let limits = SessionLimits::default();

    let flat = session
      .slice(&EntityHandle::MlirFunction { function: 0 }, &limits)
      .expect("flat function slice");
    assert_eq!(flat.collapse, CollapseMode::None);
    assert!(flat.collapsed_groups.is_empty());
    assert!(
      flat
        .entities
        .iter()
        .any(|entity| entity.kind == "mlir_operation")
    );

    let structural = session
      .slice_with_options(
        &EntityHandle::MlirFunction { function: 0 },
        &limits,
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural function slice");
    assert_eq!(structural.collapse, CollapseMode::Structural);
    assert!(structural.cache_key.contains("collapse:structural"));
    assert!(!structural.truncated);
    assert!(
      structural
        .entities
        .iter()
        .all(|entity| entity.kind != "mlir_operation" && entity.kind != "mlir_value")
    );
    assert!(
      structural
        .entities
        .iter()
        .any(|entity| entity.kind == "mlir_block")
    );
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert_eq!(structural.collapsed_groups[0].kind, "mlir_block");
    assert!(matches!(
      structural.collapsed_groups[0].expand,
      EntityHandle::MlirBlock { ref scope, block: 0 } if scope == "function:0"
    ));
    assert!(structural.collapsed_groups[0].item_count > 0);
  }

  #[test]
  fn mlir_structural_collapse_layout_uses_group_nodes_and_cache_key() {
    let data = br#"module {
  func.func @main(%i: index) -> i64 {
    %0 = arith.index_cast %i : index to i64
    return %0 : i64
  }
}
"#;
    let session = ModelSession::open(
      data,
      ModelSource::from_memory(Some("structural-layout.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");
    let scope = EntityHandle::MlirFunction { function: 0 };
    let limits = SessionLimits::default();

    let flat = session.layout(&scope, &limits).expect("flat MLIR layout");
    let structural = session
      .layout_with_options(
        &scope,
        &limits,
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("structural MLIR layout");

    assert_ne!(flat.cache_key, structural.cache_key);
    assert!(flat.cache_key.contains("collapse:none"));
    assert!(structural.cache_key.contains("collapse:structural"));
    assert_eq!(structural.collapse, CollapseMode::Structural);
    assert_eq!(session.projection_cache.layouts.borrow().len(), 2);
    assert!(
      structural
        .graph
        .nodes
        .iter()
        .all(|node| node.kind == LayoutNodeKind::Group)
    );
    assert_eq!(structural.graph.edges.len(), 1);
    assert_eq!(structural.collapsed_groups.len(), 1);
    assert!(!structural.truncated);

    let limited = session
      .layout_with_options(
        &scope,
        &SessionLimits {
          layout: 1,
          ..SessionLimits::default()
        },
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("limited structural MLIR layout");
    assert_eq!(limited.limit_used, 1);
    assert_eq!(limited.graph.nodes.len(), 1);
    assert!(limited.truncated);
    assert_eq!(limited.graph.stats.omitted_nodes, 1);
    assert!(limited.collapsed_groups.is_empty());
  }

  #[test]
  fn mlir_module_structural_layout_filters_groups_to_visible_nodes() {
    let data = br#"module @m {
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
      ModelSource::from_memory(Some("structural-module-layout.mlir".to_owned()), data.len()),
    )
    .expect("open MLIR session");
    let scope = EntityHandle::MlirModule { module: 0 };
    let limited = session
      .layout_with_options(
        &scope,
        &SessionLimits {
          layout: 1,
          ..SessionLimits::default()
        },
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("limited structural module layout");

    assert_eq!(limited.graph.nodes.len(), 1);
    assert!(
      limited
        .graph
        .nodes
        .iter()
        .all(|node| node.kind == LayoutNodeKind::Group)
    );
    assert!(limited.truncated);
    assert_eq!(limited.graph.stats.omitted_nodes, 2);
    assert!(limited.collapsed_groups.is_empty());

    let partial = session
      .layout_with_options(
        &scope,
        &SessionLimits {
          layout: 2,
          ..SessionLimits::default()
        },
        ProjectionOptions {
          collapse: CollapseMode::Structural,
          ..ProjectionOptions::default()
        },
      )
      .expect("partial structural module layout");
    let visible = partial
      .graph
      .nodes
      .iter()
      .map(|node| node.id.as_str())
      .collect::<BTreeSet<_>>();
    assert!(
      partial
        .collapsed_groups
        .iter()
        .all(|group| visible.contains(group.id.as_str()))
    );
  }

  #[test]
  fn layout_reports_initializer_omissions_as_truncation() {
    let model = fixture_initializer_onnx_model();
    let session = ModelSession {
      id: 105,
      source: ModelSource::from_memory(Some("initializer.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
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
      projection_cache: ProjectionCache::default(),
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
      projection_cache: ProjectionCache::default(),
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
      projection_cache: ProjectionCache::default(),
    };
    let bounded = session.tensor_metadata(&SessionLimits {
      detail: 1,
      ..SessionLimits::default()
    });
    assert_eq!(bounded.len(), 1);

    let direct = session
      .tensor_metadata_by_id(1)
      .expect("direct tensor lookup should ignore overview bounds");
    assert_eq!(direct.name.as_deref(), Some("external_weight"));
    assert_eq!(direct.handle, EntityHandle::Tensor { tensor: 1 });
  }

  #[test]
  fn onnx_detail_returns_selected_entity_fields() {
    let model = fixture_onnx_model();
    let session = ModelSession {
      id: 103,
      source: ModelSource::from_memory(Some("model.onnx".to_owned()), 0),
      index: FormatIndex::build(&model, None),
      model,
      projection_cache: ProjectionCache::default(),
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
      projection_cache: ProjectionCache::default(),
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
      .map_or(0, |entry| entry.count)
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
        metadata: BTreeMap::new(),
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

  fn fixture_three_node_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let input = graph.add_value(Value::new(model.intern("input")));
    let first_hidden = graph.add_value(Value::new(model.intern("first_hidden")));
    let second_hidden = graph.add_value(Value::new(model.intern("second_hidden")));
    let output = graph.add_value(Value::new(model.intern("output")));
    graph.inputs.push(input);
    graph.outputs.push(output);
    graph.values[input.index()].is_graph_input = true;
    graph.values[output.index()].is_graph_output = true;

    let first = add_fixture_node(&mut model, &mut graph, "Relu", &[input], &[first_hidden]);
    graph.values[input.index()].consumers.push(first);
    graph.values[first_hidden.index()].producer = Some(first);
    let second = add_fixture_node(
      &mut model,
      &mut graph,
      "Add",
      &[first_hidden],
      &[second_hidden],
    );
    graph.values[first_hidden.index()].consumers.push(second);
    graph.values[second_hidden.index()].producer = Some(second);
    let third = add_fixture_node(
      &mut model,
      &mut graph,
      "Sigmoid",
      &[second_hidden],
      &[output],
    );
    graph.values[second_hidden.index()].consumers.push(third);
    graph.values[output.index()].producer = Some(third);

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

  fn fixture_repeated_pair_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let x0 = graph.add_value(Value::new(model.intern("x0")));
    let h0 = graph.add_value(Value::new(model.intern("h0")));
    let y0 = graph.add_value(Value::new(model.intern("y0")));
    let x1 = graph.add_value(Value::new(model.intern("x1")));
    let h1 = graph.add_value(Value::new(model.intern("h1")));
    let y1 = graph.add_value(Value::new(model.intern("y1")));
    graph.inputs.extend([x0, x1]);
    graph.outputs.extend([y0, y1]);
    graph.values[x0.index()].is_graph_input = true;
    graph.values[x1.index()].is_graph_input = true;
    graph.values[y0.index()].is_graph_output = true;
    graph.values[y1.index()].is_graph_output = true;

    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x0], &[h0]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[h0], &[y0]);
    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x1], &[h1]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[h1], &[y1]);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_repeated_triple_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let x0 = graph.add_value(Value::new(model.intern("x0")));
    let h0 = graph.add_value(Value::new(model.intern("h0")));
    let m0 = graph.add_value(Value::new(model.intern("m0")));
    let y0 = graph.add_value(Value::new(model.intern("y0")));
    let x1 = graph.add_value(Value::new(model.intern("x1")));
    let h1 = graph.add_value(Value::new(model.intern("h1")));
    let m1 = graph.add_value(Value::new(model.intern("m1")));
    let y1 = graph.add_value(Value::new(model.intern("y1")));
    graph.inputs.extend([x0, x1]);
    graph.outputs.extend([y0, y1]);
    graph.values[x0.index()].is_graph_input = true;
    graph.values[x1.index()].is_graph_input = true;
    graph.values[y0.index()].is_graph_output = true;
    graph.values[y1.index()].is_graph_output = true;

    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x0], &[h0]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[h0], &[m0]);
    add_connected_fixture_node(&mut model, &mut graph, "Sigmoid", &[m0], &[y0]);
    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x1], &[h1]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[h1], &[m1]);
    add_connected_fixture_node(&mut model, &mut graph, "Sigmoid", &[m1], &[y1]);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_repeated_six_node_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let operators = ["Relu", "Add", "Sigmoid", "Tanh", "Mul", "Identity"];
    let mut graph_outputs = Vec::new();

    for instance in 0..2 {
      let input = graph.add_value(Value::new(model.intern(format!("x{instance}"))));
      graph.inputs.push(input);
      graph.values[input.index()].is_graph_input = true;
      let mut current = input;
      for (offset, operator) in operators.iter().enumerate() {
        let output = graph.add_value(Value::new(model.intern(format!("v{instance}_{offset}"))));
        add_connected_fixture_node(&mut model, &mut graph, operator, &[current], &[output]);
        current = output;
      }
      graph_outputs.push(current);
    }
    for output in graph_outputs {
      graph.outputs.push(output);
      graph.values[output.index()].is_graph_output = true;
    }

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_repeated_branch_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let x0 = graph.add_value(Value::new(model.intern("x0")));
    let h0 = graph.add_value(Value::new(model.intern("h0")));
    let a0 = graph.add_value(Value::new(model.intern("a0")));
    let b0 = graph.add_value(Value::new(model.intern("b0")));
    let y0 = graph.add_value(Value::new(model.intern("y0")));
    let x1 = graph.add_value(Value::new(model.intern("x1")));
    let h1 = graph.add_value(Value::new(model.intern("h1")));
    let a1 = graph.add_value(Value::new(model.intern("a1")));
    let b1 = graph.add_value(Value::new(model.intern("b1")));
    let y1 = graph.add_value(Value::new(model.intern("y1")));
    graph.inputs.extend([x0, x1]);
    graph.outputs.extend([y0, y1]);
    graph.values[x0.index()].is_graph_input = true;
    graph.values[x1.index()].is_graph_input = true;
    graph.values[y0.index()].is_graph_output = true;
    graph.values[y1.index()].is_graph_output = true;

    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x0], &[h0]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h0], &[a0]);
    add_connected_fixture_node(&mut model, &mut graph, "Sigmoid", &[h0], &[b0]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[a0, b0], &[y0]);
    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x1], &[h1]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h1], &[a1]);
    add_connected_fixture_node(&mut model, &mut graph, "Sigmoid", &[h1], &[b1]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[a1, b1], &[y1]);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_swapped_order_repeated_branch_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let x0 = graph.add_value(Value::new(model.intern("x0")));
    let h0 = graph.add_value(Value::new(model.intern("h0")));
    let a0 = graph.add_value(Value::new(model.intern("a0")));
    let b0 = graph.add_value(Value::new(model.intern("b0")));
    let y0 = graph.add_value(Value::new(model.intern("y0")));
    let x1 = graph.add_value(Value::new(model.intern("x1")));
    let h1 = graph.add_value(Value::new(model.intern("h1")));
    let a1 = graph.add_value(Value::new(model.intern("a1")));
    let b1 = graph.add_value(Value::new(model.intern("b1")));
    let y1 = graph.add_value(Value::new(model.intern("y1")));
    graph.inputs.extend([x0, x1]);
    graph.outputs.extend([y0, y1]);
    graph.values[x0.index()].is_graph_input = true;
    graph.values[x1.index()].is_graph_input = true;
    graph.values[y0.index()].is_graph_output = true;
    graph.values[y1.index()].is_graph_output = true;

    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x0], &[h0]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h0], &[a0]);
    add_connected_fixture_node(&mut model, &mut graph, "Sigmoid", &[h0], &[b0]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[a0, b0], &[y0]);
    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x1], &[h1]);
    add_connected_fixture_node(&mut model, &mut graph, "Sigmoid", &[h1], &[b1]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h1], &[a1]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[a1, b1], &[y1]);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_interleaved_repeated_pair_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let x0 = graph.add_value(Value::new(model.intern("x0")));
    let h0 = graph.add_value(Value::new(model.intern("h0")));
    let y0 = graph.add_value(Value::new(model.intern("y0")));
    let x1 = graph.add_value(Value::new(model.intern("x1")));
    let h1 = graph.add_value(Value::new(model.intern("h1")));
    let y1 = graph.add_value(Value::new(model.intern("y1")));
    graph.inputs.extend([x0, x1]);
    graph.outputs.extend([y0, y1]);
    graph.values[x0.index()].is_graph_input = true;
    graph.values[x1.index()].is_graph_input = true;
    graph.values[y0.index()].is_graph_output = true;
    graph.values[y1.index()].is_graph_output = true;

    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x0], &[h0]);
    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x1], &[h1]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h0], &[y0]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h1], &[y1]);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_shared_boundary_repeated_pair_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let x0 = graph.add_value(Value::new(model.intern("x0")));
    let h0 = graph.add_value(Value::new(model.intern("h0")));
    let y0 = graph.add_value(Value::new(model.intern("y0")));
    let z0 = graph.add_value(Value::new(model.intern("z0")));
    let x1 = graph.add_value(Value::new(model.intern("x1")));
    let h1 = graph.add_value(Value::new(model.intern("h1")));
    let y1 = graph.add_value(Value::new(model.intern("y1")));
    let z1 = graph.add_value(Value::new(model.intern("z1")));
    graph.inputs.extend([x0, x1]);
    graph.outputs.extend([y0, z0, y1, z1]);
    graph.values[x0.index()].is_graph_input = true;
    graph.values[x1.index()].is_graph_input = true;
    graph.values[y0.index()].is_graph_output = true;
    graph.values[z0.index()].is_graph_output = true;
    graph.values[y1.index()].is_graph_output = true;
    graph.values[z1.index()].is_graph_output = true;

    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x0], &[h0]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[h0], &[y0]);
    add_connected_fixture_node(&mut model, &mut graph, "Identity", &[h0], &[z0]);
    add_connected_fixture_node(&mut model, &mut graph, "Relu", &[x1], &[h1]);
    add_connected_fixture_node(&mut model, &mut graph, "Add", &[h1], &[y1]);
    add_connected_fixture_node(&mut model, &mut graph, "Tanh", &[h1], &[z1]);

    model.replace_graph(graph_id, graph);
    model
  }

  fn fixture_overlapping_repeated_pair_onnx_model() -> netron_rs_core::Model {
    let mut model = netron_rs_core::Model::new(FormatInfo {
      name: "ONNX",
      version: None,
    });
    let graph_name = model.intern("main");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let values = (0..5)
      .map(|index| graph.add_value(Value::new(model.intern(format!("v{index}")))))
      .collect::<Vec<_>>();
    graph.inputs.push(values[0]);
    graph.outputs.push(values[4]);
    graph.values[values[0].index()].is_graph_input = true;
    graph.values[values[4].index()].is_graph_output = true;

    for index in 0..4 {
      add_connected_fixture_node(
        &mut model,
        &mut graph,
        "Relu",
        &[values[index]],
        &[values[index + 1]],
      );
    }

    model.replace_graph(graph_id, graph);
    model
  }

  fn slice_node_ids(slice: &SliceResponse) -> BTreeSet<usize> {
    slice
      .entities
      .iter()
      .filter_map(|entity| match &entity.handle {
        Some(EntityHandle::Node { node, .. }) => Some(node),
        _ => None,
      })
      .copied()
      .collect()
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

  fn add_connected_fixture_node(
    model: &mut netron_rs_core::Model,
    graph: &mut Graph,
    operator: &str,
    inputs: &[netron_rs_core::ValueId],
    outputs: &[netron_rs_core::ValueId],
  ) -> netron_rs_core::NodeId {
    let node = add_fixture_node(model, graph, operator, inputs, outputs);
    for input in inputs {
      graph.values[input.index()].consumers.push(node);
    }
    for output in outputs {
      graph.values[output.index()].producer = Some(node);
    }
    node
  }
}
