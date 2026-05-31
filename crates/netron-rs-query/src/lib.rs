use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use netron_rs_core::{Model, ModelError, ModelInput, TensorStorage};
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
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchEntry {
    pub kind: SearchKind,
    pub graph: Option<usize>,
    pub id: usize,
    pub name: Option<String>,
    pub operator: Option<String>,
    pub origin: Option<&'static str>,
    #[serde(skip)]
    searchable: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EntityHandle {
    Graph { graph: usize },
    Node { graph: usize, node: usize },
    Value { graph: usize, value: usize },
    Tensor { tensor: usize },
    Function { function: usize },
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
    Onnx(ModelIndex),
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
    pub external_data: usize,
    pub mlir_resources: usize,
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
    pub kind: DiagnosticKind,
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
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
    diagnostics: Vec<Diagnostic>,
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
            diagnostics: Vec::new(),
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

    pub fn summary(&self, _limits: &SessionLimits) -> SessionSummary {
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
            external_data: external_data_count(&self.model),
            mlir_resources: mlir_resource_count(&self.model),
        }
    }

    pub fn diagnostics(&self, limits: &SessionLimits) -> DiagnosticsResponse {
        let limit = limits.clamp().diagnostics;
        let mut diagnostics = self.diagnostics.clone();
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
}

impl FormatIndex {
    fn build(model: &Model) -> Self {
        let index = ModelIndex::build(model);
        match model.format.name {
            "ONNX" | "ONNX Tensor" => Self::Onnx(index),
            "MLIR" => Self::Mlir(index),
            _ => Self::Unknown(index),
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
            Self::Onnx(index) | Self::Mlir(index) | Self::Unknown(index) => {
                index.search(query, limit)
            }
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
        let needle = query.trim().to_lowercase();
        if needle.is_empty() || limit == 0 {
            return Vec::new();
        }

        let mut matches = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.score(&needle).map(|score| (score, index)))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        matches.truncate(limit);
        matches
            .into_iter()
            .map(|(_, index)| self.entries[index].clone())
            .collect()
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
        let searchable = [name.as_deref(), operator.as_deref(), origin]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        Self {
            kind,
            graph,
            id,
            name,
            operator,
            origin,
            searchable,
        }
    }

    pub fn handle(&self) -> EntityHandle {
        match self.kind {
            SearchKind::Graph => EntityHandle::Graph { graph: self.id },
            SearchKind::Node => EntityHandle::Node {
                graph: self.graph.unwrap_or(0),
                node: self.id,
            },
            SearchKind::Value => EntityHandle::Value {
                graph: self.graph.unwrap_or(0),
                value: self.id,
            },
            SearchKind::Tensor => EntityHandle::Tensor { tensor: self.id },
            SearchKind::Function => EntityHandle::Function { function: self.id },
        }
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
    use netron_rs_core::{FormatInfo, Graph, Node, Operator, Value};

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
}
