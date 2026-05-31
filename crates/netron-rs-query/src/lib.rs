use netron_rs_core::Model;
use serde::Serialize;

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
