use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use netron_rs_core::{
    AttributeValue, Dimension, DimensionValue, Model, ModelError, ModelInput, Operator, Tensor,
    TensorElementType, TensorStorage,
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
    Mlir(ModelIndex),
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

    pub fn detail(
        &self,
        model: &Model,
        handle: &EntityHandle,
        limit: usize,
    ) -> Option<EntityDetail> {
        onnx_detail(model, self, handle, limit)
    }
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
    let mut operator_types = BTreeMap::new();
    let mut domains = BTreeMap::new();
    let mut dtypes = BTreeMap::new();
    let mut storage_kinds = BTreeMap::new();
    let mut shape_ranks = BTreeMap::new();
    let mut fan_in = BTreeMap::new();
    let mut fan_out = BTreeMap::new();

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
            record_operator_histograms(
                model,
                &node.operator,
                node.inputs.iter().flatten().count(),
                node.outputs.iter().flatten().count(),
                &mut operator_types,
                &mut domains,
                &mut fan_in,
                &mut fan_out,
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
            record_operator_histograms(
                model,
                &node.operator,
                node.inputs.iter().flatten().count(),
                node.outputs.iter().flatten().count(),
                &mut operator_types,
                &mut domains,
                &mut fan_in,
                &mut fan_out,
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
        operator_types: histogram_entries(operator_types),
        domains: histogram_entries(domains),
        dtypes: histogram_entries(dtypes),
        storage_kinds: histogram_entries(storage_kinds),
        shape_ranks: histogram_entries(shape_ranks),
        fan_in: histogram_entries(fan_in),
        fan_out: histogram_entries(fan_out),
    }
}

fn record_operator_histograms(
    model: &Model,
    operator: &Operator,
    inputs: usize,
    outputs: usize,
    operator_types: &mut BTreeMap<String, usize>,
    domains: &mut BTreeMap<String, usize>,
    fan_in: &mut BTreeMap<String, usize>,
    fan_out: &mut BTreeMap<String, usize>,
) {
    increment(operator_types, model.strings.get(operator.name).to_owned());
    increment(domains, operator_domain(model, operator));
    increment(fan_in, inputs.to_string());
    increment(fan_out, outputs.to_string());
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
        EntityHandle::Diagnostic { diagnostic } => diagnostic_detail(index, *diagnostic),
    }
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

fn diagnostic_detail(index: &OnnxIndex, diagnostic_index: usize) -> Option<EntityDetail> {
    let diagnostic = index.diagnostics.get(diagnostic_index)?;
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
        let index = FormatIndex::build(&model);
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

        SessionSummary {
            api_version: SESSION_API_VERSION,
            session_id: self.id,
            source: self.source.clone(),
            format: self.index.kind(),
            source_format_name: self.model.format.name.to_owned(),
            byte_len: self.source.byte_len,
            graphs: self.model.graphs.len(),
            functions: self.model.functions.len(),
            nodes: self
                .model
                .graphs
                .iter()
                .map(|graph| graph.nodes.len())
                .sum(),
            values: self
                .model
                .graphs
                .iter()
                .map(|graph| graph.values.len())
                .sum(),
            tensors: self.model.tensors.len(),
            initializers,
            subgraphs,
            sparse_tensors,
            metadata,
            opsets,
            external_data: external_data_count(&self.model),
            mlir_resources: mlir_resource_count(&self.model),
            onnx,
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
        self.index
            .detail(&self.model, handle, limits.clamp().detail)
    }
}

impl FormatIndex {
    fn build(model: &Model) -> Self {
        match model.format.name {
            "ONNX" | "ONNX Tensor" => Self::Onnx(OnnxIndex::build(model)),
            "MLIR" => Self::Mlir(ModelIndex::build(model)),
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

    pub fn detail(
        &self,
        model: &Model,
        handle: &EntityHandle,
        limit: usize,
    ) -> Option<EntityDetail> {
        match self {
            Self::Onnx(index) => index.detail(model, handle, limit),
            Self::Mlir(_) | Self::Unknown(_) => None,
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
    use std::collections::BTreeMap;

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
            index: FormatIndex::build(&model),
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
            index: FormatIndex::build(&model),
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
            index: FormatIndex::build(&model),
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
            node.related
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
            index: FormatIndex::build(&model),
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
        node.metadata
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
}
