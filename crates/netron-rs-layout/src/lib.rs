use std::collections::{BTreeMap, BTreeSet, VecDeque};

use netron_rs_core::{Graph, Model, Node, Value};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
  #[error("graph {graph} does not resolve; model has {graphs} graph(s)")]
  UnknownGraph { graph: usize, graphs: usize },
}

#[derive(Debug, Clone)]
pub struct LayoutOptions {
  pub graph: usize,
  pub max_nodes: Option<usize>,
  pub include_initializers: bool,
  pub direction: LayoutDirection,
  pub rank_spacing: f32,
  pub node_spacing: f32,
}

impl Default for LayoutOptions {
  fn default() -> Self {
    Self {
      graph: 0,
      max_nodes: None,
      include_initializers: false,
      direction: LayoutDirection::LeftToRight,
      rank_spacing: 260.0,
      node_spacing: 96.0,
    }
  }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayoutDirection {
  LeftToRight,
  TopToBottom,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutModel {
  pub graphs: Vec<LayoutGraph>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutGraph {
  pub graph: usize,
  pub parent: Option<usize>,
  pub name: Option<String>,
  pub subgraphs: Vec<usize>,
  pub nodes: Vec<LayoutNode>,
  pub edges: Vec<LayoutEdge>,
  pub stats: LayoutStats,
  pub bounds: LayoutBounds,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutNode {
  pub id: String,
  pub kind: LayoutNodeKind,
  pub graph: usize,
  pub node: Option<usize>,
  pub value: Option<usize>,
  pub name: Option<String>,
  pub operator: Option<String>,
  pub origin: Option<&'static str>,
  pub rank: usize,
  pub order: usize,
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub hidden_initializers: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutNodeKind {
  GraphInput,
  Initializer,
  Operator,
  GraphOutput,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayoutEdge {
  pub id: String,
  pub value: usize,
  pub name: String,
  pub from: String,
  pub to: String,
  pub points: Vec<LayoutPoint>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct LayoutPoint {
  pub x: f32,
  pub y: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LayoutStats {
  pub nodes_used: usize,
  pub edges_used: usize,
  pub omitted_nodes: usize,
  pub omitted_edges: usize,
  pub omitted_initializers: usize,
  pub cycles_detected: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LayoutBounds {
  pub width: f32,
  pub height: f32,
}

pub fn layout_model(model: &Model, options: &LayoutOptions) -> Result<LayoutModel, LayoutError> {
  Ok(LayoutModel {
    graphs: vec![layout_graph(model, options)?],
  })
}

pub fn layout_graph(model: &Model, options: &LayoutOptions) -> Result<LayoutGraph, LayoutError> {
  let graph = model
    .graphs
    .get(options.graph)
    .ok_or(LayoutError::UnknownGraph {
      graph: options.graph,
      graphs: model.graphs.len(),
    })?;

  let visible = visible_operators(graph, options.max_nodes);
  let rank_plan = rank_operators(graph, &visible);
  let mut nodes = Vec::new();
  let mut input_nodes = BTreeMap::new();
  let mut initializer_nodes = BTreeMap::new();
  let mut operator_nodes = vec![None; graph.nodes.len()];
  let mut output_nodes = BTreeMap::new();

  for value in &graph.inputs {
    let value = &graph.values[value.index()];
    let node_index = push_node(
      &mut nodes,
      LayoutNode::boundary(
        format!("input:{}", value.id.index()),
        LayoutNodeKind::GraphInput,
        graph.id.index(),
        value.id.index(),
        model.strings.get(value.name).to_owned(),
        0,
      ),
    );
    input_nodes.insert(value.id.index(), node_index);
  }

  if options.include_initializers {
    for value in graph
      .values
      .iter()
      .filter(|value| value.initializer.is_some() && has_visible_consumer(value, &visible))
    {
      let node_index = push_node(
        &mut nodes,
        LayoutNode::boundary(
          format!("initializer:{}", value.id.index()),
          LayoutNodeKind::Initializer,
          graph.id.index(),
          value.id.index(),
          model.strings.get(value.name).to_owned(),
          0,
        ),
      );
      initializer_nodes.insert(value.id.index(), node_index);
    }
  }

  for node in graph.nodes.iter().filter(|node| visible[node.id.index()]) {
    let rank = rank_plan.ranks[node.id.index()];
    let node_index = push_node(
      &mut nodes,
      LayoutNode::operator(
        format!("node:{}", node.id.index()),
        graph.id.index(),
        node,
        model,
        rank,
        hidden_initializers(graph, node, options.include_initializers),
      ),
    );
    operator_nodes[node.id.index()] = Some(node_index);
  }

  for value_id in &graph.outputs {
    let value = &graph.values[value_id.index()];
    let rank = source_rank(value, &rank_plan.ranks, &visible).unwrap_or(rank_plan.max_rank) + 1;
    let node_index = push_node(
      &mut nodes,
      LayoutNode::boundary(
        format!("output:{}", value.id.index()),
        LayoutNodeKind::GraphOutput,
        graph.id.index(),
        value.id.index(),
        model.strings.get(value.name).to_owned(),
        rank,
      ),
    );
    output_nodes.insert(value.id.index(), node_index);
  }

  position_nodes(&mut nodes, options);

  let mut stats = LayoutStats {
    omitted_nodes: graph.nodes.len() - visible.iter().filter(|visible| **visible).count(),
    omitted_initializers: omitted_initializers(graph, &visible, options.include_initializers),
    cycles_detected: rank_plan.cycles_detected,
    ..LayoutStats::default()
  };
  let edges = build_edges(
    EdgeBuildContext {
      model,
      graph,
      visible: &visible,
      input_nodes: &input_nodes,
      initializer_nodes: &initializer_nodes,
      operator_nodes: &operator_nodes,
      output_nodes: &output_nodes,
      nodes: &nodes,
    },
    &mut stats,
  );
  stats.nodes_used = nodes.len();
  stats.edges_used = edges.len();

  Ok(LayoutGraph {
    graph: graph.id.index(),
    parent: graph.parent.map(|parent| parent.index()),
    name: graph.name.map(|name| model.strings.get(name).to_owned()),
    subgraphs: graph.subgraphs.iter().map(|graph| graph.index()).collect(),
    bounds: bounds(&nodes),
    nodes,
    edges,
    stats,
  })
}

struct EdgeBuildContext<'a> {
  model: &'a Model,
  graph: &'a Graph,
  visible: &'a [bool],
  input_nodes: &'a BTreeMap<usize, usize>,
  initializer_nodes: &'a BTreeMap<usize, usize>,
  operator_nodes: &'a [Option<usize>],
  output_nodes: &'a BTreeMap<usize, usize>,
  nodes: &'a [LayoutNode],
}

impl LayoutNode {
  fn boundary(
    id: String,
    kind: LayoutNodeKind,
    graph: usize,
    value: usize,
    name: String,
    rank: usize,
  ) -> Self {
    Self {
      id,
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

  fn operator(
    id: String,
    graph: usize,
    node: &Node,
    model: &Model,
    rank: usize,
    hidden_initializers: usize,
  ) -> Self {
    Self {
      id,
      kind: LayoutNodeKind::Operator,
      graph,
      node: Some(node.id.index()),
      value: None,
      name: node.name.map(|name| model.strings.get(name).to_owned()),
      operator: Some(model.strings.get(node.operator.name).to_owned()),
      origin: Some(node.operator.origin),
      rank,
      order: 0,
      x: 0.0,
      y: 0.0,
      width: 190.0,
      height: 64.0,
      hidden_initializers,
    }
  }

  fn center_left(&self) -> LayoutPoint {
    LayoutPoint {
      x: self.x,
      y: self.y + self.height / 2.0,
    }
  }

  fn center_right(&self) -> LayoutPoint {
    LayoutPoint {
      x: self.x + self.width,
      y: self.y + self.height / 2.0,
    }
  }
}

fn push_node(nodes: &mut Vec<LayoutNode>, node: LayoutNode) -> usize {
  let index = nodes.len();
  nodes.push(node);
  index
}

fn visible_operators(graph: &Graph, max_nodes: Option<usize>) -> Vec<bool> {
  let limit = max_nodes.unwrap_or(usize::MAX);
  graph
    .nodes
    .iter()
    .enumerate()
    .map(|(index, _)| index < limit)
    .collect()
}

struct RankPlan {
  ranks: Vec<usize>,
  max_rank: usize,
  cycles_detected: usize,
}

fn rank_operators(graph: &Graph, visible: &[bool]) -> RankPlan {
  let mut outgoing = vec![BTreeSet::new(); graph.nodes.len()];
  let mut indegree = vec![0usize; graph.nodes.len()];
  for value in &graph.values {
    let Some(producer) = value.producer else {
      continue;
    };
    if !visible[producer.index()] {
      continue;
    }
    for consumer in &value.consumers {
      if !visible[consumer.index()] || producer == *consumer {
        continue;
      }
      if outgoing[producer.index()].insert(consumer.index()) {
        indegree[consumer.index()] += 1;
      }
    }
  }

  let mut ranks = vec![1usize; graph.nodes.len()];
  let mut queue = VecDeque::new();
  for node in &graph.nodes {
    if visible[node.id.index()] && indegree[node.id.index()] == 0 {
      queue.push_back(node.id.index());
    }
  }

  let mut visited = 0usize;
  while let Some(node) = queue.pop_front() {
    visited += 1;
    for next in &outgoing[node] {
      ranks[*next] = ranks[*next].max(ranks[node] + 1);
      indegree[*next] -= 1;
      if indegree[*next] == 0 {
        queue.push_back(*next);
      }
    }
  }

  let visible_count = visible.iter().filter(|visible| **visible).count();
  let cycles_detected = visible_count.saturating_sub(visited);
  let max_rank = graph
    .nodes
    .iter()
    .filter(|node| visible[node.id.index()])
    .map(|node| ranks[node.id.index()])
    .max()
    .unwrap_or(0);
  RankPlan {
    ranks,
    max_rank,
    cycles_detected,
  }
}

fn source_rank(value: &Value, ranks: &[usize], visible: &[bool]) -> Option<usize> {
  if let Some(producer) = value.producer
    && visible[producer.index()]
  {
    return Some(ranks[producer.index()]);
  }
  if value.is_graph_input || value.initializer.is_some() {
    return Some(0);
  }
  None
}

fn has_visible_consumer(value: &Value, visible: &[bool]) -> bool {
  value
    .consumers
    .iter()
    .any(|consumer| visible[consumer.index()])
}

fn hidden_initializers(graph: &Graph, node: &Node, include_initializers: bool) -> usize {
  if include_initializers {
    return 0;
  }
  node
    .inputs
    .iter()
    .flatten()
    .filter(|value| graph.values[value.index()].initializer.is_some())
    .count()
}

fn omitted_initializers(graph: &Graph, visible: &[bool], include_initializers: bool) -> usize {
  if include_initializers {
    return 0;
  }
  graph
    .values
    .iter()
    .filter(|value| value.initializer.is_some() && has_visible_consumer(value, visible))
    .count()
}

fn position_nodes(nodes: &mut [LayoutNode], options: &LayoutOptions) {
  let mut by_rank: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
  for (index, node) in nodes.iter().enumerate() {
    by_rank.entry(node.rank).or_default().push(index);
  }

  for (rank, indices) in by_rank {
    for (order, index) in indices.into_iter().enumerate() {
      let node = &mut nodes[index];
      node.order = order;
      match options.direction {
        LayoutDirection::LeftToRight => {
          node.x = rank as f32 * options.rank_spacing;
          node.y = order as f32 * options.node_spacing;
        }
        LayoutDirection::TopToBottom => {
          node.x = order as f32 * options.node_spacing * 2.0;
          node.y = rank as f32 * options.rank_spacing;
        }
      }
    }
  }
}

fn build_edges(context: EdgeBuildContext<'_>, stats: &mut LayoutStats) -> Vec<LayoutEdge> {
  let mut edges = Vec::new();
  for value in &context.graph.values {
    let source = source_node(
      value,
      context.visible,
      context.input_nodes,
      context.initializer_nodes,
      context.operator_nodes,
    );
    for consumer in &value.consumers {
      if let Some(target) = context.operator_nodes[consumer.index()] {
        if let Some(source) = source {
          edges.push(make_edge(
            context.model,
            context.graph,
            value,
            source,
            target,
            context.nodes,
            edges.len(),
          ));
        } else {
          stats.omitted_edges += 1;
        }
      } else if source.is_some() {
        stats.omitted_edges += 1;
      }
    }
    if let Some(target) = context.output_nodes.get(&value.id.index()).copied() {
      if let Some(source) = source {
        edges.push(make_edge(
          context.model,
          context.graph,
          value,
          source,
          target,
          context.nodes,
          edges.len(),
        ));
      } else {
        stats.omitted_edges += 1;
      }
    }
  }
  edges
}

fn source_node(
  value: &Value,
  visible: &[bool],
  input_nodes: &BTreeMap<usize, usize>,
  initializer_nodes: &BTreeMap<usize, usize>,
  operator_nodes: &[Option<usize>],
) -> Option<usize> {
  if let Some(producer) = value.producer
    && visible[producer.index()]
  {
    return operator_nodes[producer.index()];
  }
  input_nodes
    .get(&value.id.index())
    .copied()
    .or_else(|| initializer_nodes.get(&value.id.index()).copied())
}

fn make_edge(
  model: &Model,
  graph: &Graph,
  value: &Value,
  source: usize,
  target: usize,
  nodes: &[LayoutNode],
  edge_index: usize,
) -> LayoutEdge {
  let from = nodes[source].id.clone();
  let to = nodes[target].id.clone();
  LayoutEdge {
    id: format!("edge:{}:{edge_index}", value.id.index()),
    value: value.id.index(),
    name: model
      .strings
      .get(graph.values[value.id.index()].name)
      .to_owned(),
    from,
    to,
    points: route(&nodes[source], &nodes[target]),
  }
}

fn route(source: &LayoutNode, target: &LayoutNode) -> Vec<LayoutPoint> {
  let start = source.center_right();
  let end = target.center_left();
  let middle = (start.x + end.x) / 2.0;
  vec![
    start,
    LayoutPoint {
      x: middle,
      y: start.y,
    },
    LayoutPoint {
      x: middle,
      y: end.y,
    },
    end,
  ]
}

fn bounds(nodes: &[LayoutNode]) -> LayoutBounds {
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

#[cfg(test)]
mod tests {
  use netron_rs_core::{
    FormatInfo, Graph, Model, Node, NodeId, Operator, Tensor, TensorElementType, TensorStorage,
    Value,
  };

  use super::*;

  #[test]
  fn layouts_graph_inputs_operators_and_outputs_in_rank_order() {
    let model = fixture_model(false);
    let layout = layout_graph(&model, &LayoutOptions::default()).unwrap();

    assert_eq!(layout.nodes.len(), 4);
    assert_eq!(layout.edges.len(), 3);
    assert_eq!(layout.nodes[0].kind, LayoutNodeKind::GraphInput);
    assert_eq!(layout.nodes[1].operator.as_deref(), Some("Relu"));
    assert_eq!(layout.nodes[2].operator.as_deref(), Some("Add"));
    assert_eq!(layout.nodes[3].kind, LayoutNodeKind::GraphOutput);
    assert!(layout.nodes[2].rank > layout.nodes[1].rank);
    assert!(layout.nodes[3].rank > layout.nodes[2].rank);
  }

  #[test]
  fn hides_initializers_by_default_but_keeps_node_summary() {
    let model = fixture_model(true);
    let layout = layout_graph(&model, &LayoutOptions::default()).unwrap();

    assert_eq!(layout.stats.omitted_initializers, 1);
    assert!(
      layout
        .nodes
        .iter()
        .any(|node| node.operator.as_deref() == Some("Add") && node.hidden_initializers == 1)
    );
    assert!(
      !layout
        .nodes
        .iter()
        .any(|node| node.kind == LayoutNodeKind::Initializer)
    );
  }

  #[test]
  fn can_include_initializer_nodes_when_requested() {
    let model = fixture_model(true);
    let options = LayoutOptions {
      include_initializers: true,
      ..LayoutOptions::default()
    };
    let layout = layout_graph(&model, &options).unwrap();

    assert_eq!(layout.stats.omitted_initializers, 0);
    assert!(
      layout
        .nodes
        .iter()
        .any(|node| node.kind == LayoutNodeKind::Initializer)
    );
    assert!(
      layout
        .edges
        .iter()
        .any(|edge| edge.from.starts_with("initializer:"))
    );
  }

  #[test]
  fn max_nodes_returns_bounded_partial_layout() {
    let model = fixture_model(false);
    let options = LayoutOptions {
      max_nodes: Some(1),
      ..LayoutOptions::default()
    };
    let layout = layout_graph(&model, &options).unwrap();

    assert_eq!(layout.stats.omitted_nodes, 1);
    assert_eq!(
      layout
        .nodes
        .iter()
        .filter(|node| node.kind == LayoutNodeKind::Operator)
        .count(),
      1
    );
  }

  #[test]
  fn cycles_do_not_prevent_layout() {
    let model = cycle_model();
    let layout = layout_graph(&model, &LayoutOptions::default()).unwrap();

    assert_eq!(layout.stats.cycles_detected, 2);
    assert_eq!(
      layout
        .nodes
        .iter()
        .filter(|node| node.kind == LayoutNodeKind::Operator)
        .count(),
      2
    );
  }

  fn fixture_model(with_initializer: bool) -> Model {
    let mut model = Model::new(FormatInfo {
      name: "test",
      version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let input = graph.add_value(Value::new(model.intern("input")));
    let hidden = graph.add_value(Value::new(model.intern("hidden")));
    let output = graph.add_value(Value::new(model.intern("output")));
    let weight = graph.add_value(Value::new(model.intern("weight")));
    graph.inputs.push(input);
    graph.outputs.push(output);
    graph.values[input.index()].is_graph_input = true;
    graph.values[output.index()].is_graph_output = true;
    if with_initializer {
      let weight_name = model.intern("weight");
      let tensor = Tensor::metadata_only(
        Some(weight_name),
        TensorElementType::Float32,
        Vec::new(),
        TensorStorage::InlineBytes { byte_len: 4 },
      );
      graph.values[weight.index()].initializer = Some(model.add_tensor(tensor));
    }

    let first = add_node(&mut model, &mut graph, "Relu", &[input], &[hidden]);
    graph.values[input.index()].consumers.push(first);
    graph.values[hidden.index()].producer = Some(first);
    let add_inputs = if with_initializer {
      vec![hidden, weight]
    } else {
      vec![hidden]
    };
    let second = add_node(&mut model, &mut graph, "Add", &add_inputs, &[output]);
    graph.values[hidden.index()].consumers.push(second);
    graph.values[output.index()].producer = Some(second);
    if with_initializer {
      graph.values[weight.index()].consumers.push(second);
    }

    model.replace_graph(graph_id, graph);
    model
  }

  fn cycle_model() -> Model {
    let mut model = Model::new(FormatInfo {
      name: "test",
      version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let left = graph.add_value(Value::new(model.intern("left")));
    let right = graph.add_value(Value::new(model.intern("right")));
    let left_node = add_node(&mut model, &mut graph, "Left", &[right], &[left]);
    let right_node = add_node(&mut model, &mut graph, "Right", &[left], &[right]);
    graph.values[left.index()].producer = Some(left_node);
    graph.values[left.index()].consumers.push(right_node);
    graph.values[right.index()].producer = Some(right_node);
    graph.values[right.index()].consumers.push(left_node);
    model.replace_graph(graph_id, graph);
    model
  }

  fn add_node(
    model: &mut Model,
    graph: &mut Graph,
    op: &str,
    inputs: &[netron_rs_core::ValueId],
    outputs: &[netron_rs_core::ValueId],
  ) -> NodeId {
    let mut node = Node::new(
      graph.id,
      Operator {
        domain: None,
        name: model.intern(op),
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
