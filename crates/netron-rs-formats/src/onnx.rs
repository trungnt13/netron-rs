use std::{
  collections::{BTreeMap, HashMap, HashSet},
  path::Path,
};

use netron_rs_core::{
  Attribute, AttributeValue, Confidence, Dimension, DimensionValue, FormatInfo, FormatMetadata,
  Function, FunctionNode, FunctionValue, Graph, GraphId, Model, ModelError, ModelFormat,
  ModelInput, Node, Operator, OperatorSet, QuantizationAnnotation, Tensor, TensorElementType,
  TensorStorage, TypeInfo, Value, ValueId,
};
use serde_json::Value as JsonValue;

use crate::validate_external_path;

const FORMAT: &str = "ONNX";

pub struct OnnxFormat;

impl ModelFormat for OnnxFormat {
  fn metadata(&self) -> FormatMetadata {
    FormatMetadata {
      name: FORMAT,
      extensions: &["onnx", "pb"],
    }
  }

  fn detect(&self, input: ModelInput<'_>) -> Confidence {
    let extension = input
      .path
      .and_then(|path| path.extension())
      .and_then(|extension| extension.to_str())
      .map(|extension| extension.to_ascii_lowercase());

    if extension.as_deref() == Some("onnx") {
      if looks_like_json_object(input.data)
        && parse_json_model_proto(input.data).is_ok_and(|model| model.is_onnx_like())
      {
        return Confidence::Medium;
      }
      if ModelProto::decode(input.data).is_ok_and(|model| model.is_onnx_like()) {
        return Confidence::Medium;
      }
      if GraphProto::decode(input.data).is_ok_and(|graph| graph.is_graph_like()) {
        return Confidence::Medium;
      }
      if TensorProto::decode(input.data).is_ok_and(|tensor| tensor.is_tensor_like()) {
        return Confidence::Medium;
      }
      return Confidence::None;
    }

    if looks_like_json_object(input.data)
      && parse_json_model_proto(input.data).is_ok_and(|model| model.is_onnx_like())
    {
      return Confidence::Medium;
    }

    if ModelProto::decode(input.data).is_ok_and(|model| model.is_onnx_like()) {
      return Confidence::Medium;
    }

    if GraphProto::decode(input.data).is_ok_and(|graph| graph.is_graph_like()) {
      return Confidence::Medium;
    }

    if extension.as_deref() == Some("pb")
      && TensorProto::decode(input.data).is_ok_and(|tensor| tensor.is_tensor_like())
    {
      Confidence::Medium
    } else {
      Confidence::None
    }
  }

  fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
    if looks_like_json_object(input.data)
      && let Ok(proto) = parse_json_model_proto(input.data)
      && proto.is_onnx_like()
    {
      validate_model_external_paths(&proto, input.path, input.allow_unsafe_paths)?;
      return lower_model(proto);
    }

    let model_error = match ModelProto::decode(input.data) {
      Ok(proto) if proto.is_onnx_like() => {
        validate_model_external_paths(&proto, input.path, input.allow_unsafe_paths)?;
        return lower_model(proto);
      }
      Ok(_) => None,
      Err(error) => Some(error),
    };

    if let Ok(graph) = GraphProto::decode(input.data)
      && graph.is_graph_like()
    {
      validate_graph_external_paths(&graph, input.path, input.allow_unsafe_paths)?;
      return lower_graph_model(graph);
    }

    if let Ok(tensor) = TensorProto::decode(input.data)
      && tensor.is_tensor_like()
    {
      validate_tensor_external_paths(&tensor, input.path, input.allow_unsafe_paths)?;
      return Ok(lower_tensor_model(tensor));
    }

    if let Some(error) = model_error {
      Err(error)
    } else {
      Err(invalid("ONNX data is not ModelProto or TensorProto"))
    }
  }
}

fn lower_model(mut proto: ModelProto) -> Result<Model, ModelError> {
  let mut model = Model::new(FormatInfo {
    name: FORMAT,
    version: proto.ir_version.map(|version| version.to_string()),
  });

  model.metadata.producer = proto.producer_name;
  model.metadata.producer_version = proto.producer_version;
  model.metadata.domain = proto.domain;
  model.metadata.model_version = proto.model_version;
  model.metadata.description = proto.doc_string;
  model.metadata.opsets = proto
    .opsets
    .into_iter()
    .filter_map(|opset| {
      opset.version.map(|version| OperatorSet {
        domain: opset.domain,
        version,
      })
    })
    .collect();
  model.metadata.properties = proto.metadata_props;

  let mut functions = proto.functions;
  let last_function_indices = last_function_indices(&functions);
  let function_input_types = proto
    .graph
    .as_ref()
    .map(|graph| function_input_type_specializations(&mut model, graph, &functions))
    .unwrap_or_default();
  let mut initializer_moves = proto
    .graph
    .as_mut()
    .map(|graph| move_graph_initializers_to_functions(graph, &functions))
    .unwrap_or_default();
  propagate_function_initializer_moves(&mut functions, &mut initializer_moves);

  if let Some(graph) = proto.graph {
    lower_graph(&mut model, graph, None)?;
  }
  for (index, function) in functions.into_iter().enumerate() {
    let key = function_key_for_function(&function);
    let moved_initializers = if last_function_indices.get(&key) == Some(&index) {
      initializer_moves
        .by_function
        .remove(&key)
        .unwrap_or_default()
    } else {
      Vec::new()
    };
    let hidden_inputs = initializer_moves
      .hidden_inputs
      .remove(&key)
      .unwrap_or_default();
    let input_types = function_input_types.get(&key).cloned().unwrap_or_default();
    let function = lower_function(
      &mut model,
      function,
      moved_initializers,
      hidden_inputs,
      input_types,
    )?;
    model.add_function(function);
  }
  Ok(model)
}

fn lower_graph_model(graph: GraphProto) -> Result<Model, ModelError> {
  let mut model = Model::new(FormatInfo {
    name: FORMAT,
    version: None,
  });
  lower_graph(&mut model, graph, None)?;
  Ok(model)
}

fn lower_tensor_model(proto: TensorProto) -> Model {
  let mut model = Model::new(FormatInfo {
    name: "ONNX Tensor",
    version: None,
  });

  let graph_id = model.add_graph_placeholder(None, None);
  let mut graph = Graph::new(graph_id, None, None);
  let tensor = lower_tensor(&mut model, proto);
  let tensor_id = model.add_tensor(tensor);

  let domain = model.intern("ai.onnx");
  let name = model.intern("Constant");
  let mut node = Node::new(
    graph_id,
    Operator {
      domain: Some(domain),
      name,
      overload: None,
      version: Some(1),
      origin: FORMAT,
    },
  );
  let value = model.intern("value");
  node.attributes.push(Attribute {
    name: value,
    value: AttributeValue::Tensor(tensor_id),
  });
  graph.add_node(node);
  model.replace_graph(graph_id, graph);
  model
}

fn validate_model_external_paths(
  proto: &ModelProto,
  source: Option<&Path>,
  allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
  if let Some(graph) = &proto.graph {
    validate_graph_external_paths(graph, source, allow_unsafe_paths)?;
  }
  for function in &proto.functions {
    for node in &function.nodes {
      validate_node_external_paths(node, source, allow_unsafe_paths)?;
    }
  }
  Ok(())
}

fn validate_graph_external_paths(
  graph: &GraphProto,
  source: Option<&Path>,
  allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
  for tensor in &graph.initializers {
    validate_tensor_external_paths(tensor, source, allow_unsafe_paths)?;
  }
  for tensor in &graph.sparse_initializers {
    validate_sparse_tensor_external_paths(tensor, source, allow_unsafe_paths)?;
  }
  for node in &graph.nodes {
    validate_node_external_paths(node, source, allow_unsafe_paths)?;
  }
  Ok(())
}

fn validate_node_external_paths(
  node: &NodeProto,
  source: Option<&Path>,
  allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
  for attribute in &node.attributes {
    validate_attribute_external_paths(attribute, source, allow_unsafe_paths)?;
  }
  Ok(())
}

fn validate_attribute_external_paths(
  attribute: &AttributeProto,
  source: Option<&Path>,
  allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
  match &attribute.kind {
    Some(AttributeKind::Tensor(tensor)) => {
      validate_tensor_external_paths(tensor, source, allow_unsafe_paths)?
    }
    Some(AttributeKind::SparseTensor(tensor)) => {
      validate_sparse_tensor_external_paths(tensor, source, allow_unsafe_paths)?
    }
    Some(AttributeKind::Graph(graph)) => {
      validate_graph_external_paths(graph, source, allow_unsafe_paths)?
    }
    Some(AttributeKind::Tensors(tensors)) => {
      for tensor in tensors {
        validate_tensor_external_paths(tensor, source, allow_unsafe_paths)?;
      }
    }
    Some(AttributeKind::SparseTensors(tensors)) => {
      for tensor in tensors {
        validate_sparse_tensor_external_paths(tensor, source, allow_unsafe_paths)?;
      }
    }
    Some(AttributeKind::Graphs(graphs)) => {
      for graph in graphs {
        validate_graph_external_paths(graph, source, allow_unsafe_paths)?;
      }
    }
    _ => {}
  }
  Ok(())
}

fn validate_sparse_tensor_external_paths(
  tensor: &SparseTensorProto,
  source: Option<&Path>,
  allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
  if let Some(values) = &tensor.values {
    validate_tensor_external_paths(values, source, allow_unsafe_paths)?;
  }
  if let Some(indices) = &tensor.indices {
    validate_tensor_external_paths(indices, source, allow_unsafe_paths)?;
  }
  Ok(())
}

fn validate_tensor_external_paths(
  tensor: &TensorProto,
  source: Option<&Path>,
  allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
  if let Some(location) = tensor.external_data.get("location") {
    validate_external_path(source, location, allow_unsafe_paths)?;
  }
  Ok(())
}

fn lower_graph(
  model: &mut Model,
  proto: GraphProto,
  parent: Option<GraphId>,
) -> Result<GraphId, ModelError> {
  let name = proto.name.as_ref().map(|name| model.intern(name));
  let graph_id = model.add_graph_placeholder(parent, name);
  let mut graph = Graph::new(graph_id, parent, name);
  graph.description = proto.doc_string;
  graph.metadata = proto.metadata_props;
  let mut values = ValueTable::default();

  for input in proto.inputs {
    let Some(value_id) = ensure_value_info(model, &mut graph, &mut values, input, true) else {
      continue;
    };
    graph.values[value_id.index()].is_graph_input = true;
    graph.inputs.push(value_id);
  }

  for output in proto.outputs {
    let Some(value_id) = ensure_value_info(model, &mut graph, &mut values, output, true) else {
      continue;
    };
    graph.values[value_id.index()].is_graph_output = true;
    normalize_graph_output_type(&mut graph.values[value_id.index()]);
    graph.outputs.push(value_id);
  }

  for value_info in proto.value_info {
    ensure_value_info(model, &mut graph, &mut values, value_info, false);
  }

  for initializer in proto.initializers {
    let Some(name) = initializer.name.clone() else {
      continue;
    };
    if name.is_empty() {
      continue;
    }

    let value_id = ensure_value(model, &mut graph, &mut values, &name);
    attach_initializer(
      model,
      &mut graph,
      value_id,
      ConstantInitializer::Tensor(initializer),
      None,
    );
  }
  graph
    .inputs
    .retain(|value| graph.values[value.index()].initializer.is_none());
  graph
    .outputs
    .retain(|value| graph.values[value.index()].initializer.is_none());

  for initializer in proto.sparse_initializers {
    let Some(name) = initializer
      .values
      .as_ref()
      .and_then(|values| values.name.as_ref())
      .filter(|name| !name.is_empty())
      .cloned()
    else {
      continue;
    };

    let value_id = ensure_value(model, &mut graph, &mut values, &name);
    attach_initializer(
      model,
      &mut graph,
      value_id,
      ConstantInitializer::SparseTensor(initializer),
      None,
    );
  }
  graph
    .inputs
    .retain(|value| graph.values[value.index()].initializer.is_none());
  graph
    .outputs
    .retain(|value| graph.values[value.index()].initializer.is_none());

  for annotation in proto.quantization_annotations {
    if annotation.tensor_name.is_empty() {
      continue;
    }
    let value_id = ensure_value(model, &mut graph, &mut values, &annotation.tensor_name);
    let entries = annotation
      .parameters
      .into_iter()
      .map(|entry| QuantizationAnnotation {
        key: model.intern(entry.0),
        value: model.intern(entry.1),
      })
      .collect();
    graph.values[value_id.index()].quantization = entries;
  }

  let (input_counts, output_counts) = node_value_counts(&proto.nodes, &graph, model);
  for mut proto_node in proto.nodes {
    if let Some((output, initializer)) =
      take_constant_initializer(&mut proto_node, &input_counts, &output_counts)
    {
      let value_id = ensure_value(model, &mut graph, &mut values, &output);
      attach_initializer(model, &mut graph, value_id, initializer, None);
      continue;
    }

    let op_domain = proto_node
      .domain
      .as_ref()
      .filter(|domain| !domain.is_empty());
    let (op_type, overload) = node_operator_parts(
      op_domain.map(String::as_str),
      &proto_node.op_type,
      proto_node.overload.as_deref(),
    );
    let op_name = model.intern(op_type);
    let op_domain = op_domain.map(|domain| model.intern(domain));
    let mut node = Node::new(
      graph_id,
      Operator {
        domain: op_domain,
        name: op_name,
        overload: overload.map(|overload| model.intern(overload)),
        version: opset_version(model, proto_node.domain.as_deref()),
        origin: FORMAT,
      },
    );
    node.name = proto_node.name.as_ref().map(|name| model.intern(name));
    node.description = proto_node.doc_string;
    node.metadata = node_metadata_props(proto_node.metadata_props);
    node.inputs = proto_node
      .inputs
      .iter()
      .filter_map(|name| ensure_optional_value(model, &mut graph, &mut values, name).map(Some))
      .collect();
    node.outputs = proto_node
      .outputs
      .iter()
      .filter_map(|name| ensure_optional_value(model, &mut graph, &mut values, name).map(Some))
      .collect();

    for attribute in proto_node.attributes {
      node
        .attributes
        .push(lower_attribute(model, &mut graph, attribute)?);
    }

    let input_ids = node.inputs.clone();
    let output_ids = node.outputs.clone();
    let node_id = graph.add_node(node);

    for value_id in input_ids.into_iter().flatten() {
      let consumers = &mut graph.values[value_id.index()].consumers;
      if !consumers.contains(&node_id) {
        consumers.push(node_id);
      }
    }
    for value_id in output_ids.into_iter().flatten() {
      graph.values[value_id.index()].producer = Some(node_id);
    }
  }

  apply_simple_node_inference(model, &mut graph);
  model.replace_graph(graph_id, graph);
  Ok(graph_id)
}

fn apply_simple_node_inference(model: &Model, graph: &mut Graph) {
  let mut element_type_updates = Vec::new();
  let mut type_updates = Vec::new();
  for node in &graph.nodes {
    let Some(Some(output)) = node.outputs.first().copied() else {
      continue;
    };
    if !graph.values[output.index()].is_graph_output {
      continue;
    }
    match model.strings.get(node.operator.name) {
      "Cast" => {
        let Some(element_type) = node.attributes.iter().find_map(|attribute| {
          if model.strings.get(attribute.name) != "to" {
            return None;
          }
          match &attribute.value {
            AttributeValue::Type(value) => element_type_from_attribute_name(value),
            _ => None,
          }
        }) else {
          continue;
        };
        element_type_updates.push((output, element_type));
      }
      "Concat" => {
        if let Some(mut type_info) = infer_concat_output_type(model, graph, node) {
          preserve_unknown_graph_output_dimensions(
            model,
            &graph.values[output.index()],
            &mut type_info,
          );
          type_updates.push((output, type_info));
        }
      }
      _ => {}
    }
  }

  for (value_id, type_info) in type_updates {
    graph.values[value_id.index()].type_info = Some(type_info);
  }
  for (value_id, element_type) in element_type_updates {
    let value = &mut graph.values[value_id.index()];
    match &mut value.type_info {
      Some(type_info) => type_info.element_type = Some(element_type),
      None => {
        value.type_info = Some(TypeInfo {
          element_type: Some(element_type),
          layout: None,
          denotation: None,
          shape: Vec::new(),
        });
      }
    }
  }
}

fn infer_concat_output_type(model: &Model, graph: &Graph, node: &Node) -> Option<TypeInfo> {
  let axis = node.attributes.iter().find_map(|attribute| {
    if model.strings.get(attribute.name) != "axis" {
      return None;
    }
    match attribute.value {
      AttributeValue::Int(value) => Some(value),
      _ => None,
    }
  })?;
  let input_types = node
    .inputs
    .iter()
    .map(|value| {
      value
        .and_then(|value| graph.values[value.index()].type_info.as_ref())
        .filter(|type_info| !type_info.shape.is_empty())
    })
    .collect::<Option<Vec<_>>>()?;
  let first = (*input_types.first()?).clone();
  let rank = i64::try_from(first.shape.len()).ok()?;
  let axis = if axis < 0 { axis + rank } else { axis };
  if !(0..rank).contains(&axis) {
    return None;
  }
  let axis = usize::try_from(axis).ok()?;
  if input_types
    .iter()
    .any(|type_info| type_info.shape.len() != first.shape.len())
  {
    return None;
  }

  let mut axis_size = 0_i64;
  let mut shape = first.shape.clone();
  for type_info in &input_types {
    for (index, dimension) in type_info.shape.iter().enumerate() {
      if index == axis {
        let DimensionValue::Known(value) = dimension.value else {
          return None;
        };
        axis_size = axis_size.checked_add(value)?;
      } else if !same_dimension(model, &shape[index], dimension) {
        return None;
      }
    }
  }
  shape[axis] = Dimension::known(axis_size);
  Some(TypeInfo {
    element_type: first.element_type.clone(),
    layout: first.layout,
    denotation: first.denotation,
    shape,
  })
}

fn preserve_unknown_graph_output_dimensions(model: &Model, value: &Value, inferred: &mut TypeInfo) {
  let Some(existing) = &value.type_info else {
    return;
  };
  if existing.shape.len() != inferred.shape.len() {
    return;
  }
  for (existing, inferred) in existing.shape.iter().zip(inferred.shape.iter_mut()) {
    let existing_is_unknown_marker = match existing.value {
      DimensionValue::Unknown => true,
      DimensionValue::Symbolic(id) => model.strings.get(id) == "None",
      _ => false,
    };
    let inferred_is_unknown_marker = match inferred.value {
      DimensionValue::Known(0) => true,
      DimensionValue::Symbolic(id) => model.strings.get(id) == "None",
      _ => false,
    };
    if existing_is_unknown_marker && inferred_is_unknown_marker {
      inferred.value = DimensionValue::Unknown;
    }
  }
}

fn same_dimension(model: &Model, left: &Dimension, right: &Dimension) -> bool {
  if left.denotation != right.denotation {
    return false;
  }
  match (&left.value, &right.value) {
    (DimensionValue::Known(left), DimensionValue::Known(right)) => left == right,
    (DimensionValue::Symbolic(left), DimensionValue::Symbolic(right)) => {
      model.strings.get(*left) == model.strings.get(*right)
    }
    (DimensionValue::Unknown, DimensionValue::Unknown) => true,
    _ => false,
  }
}

fn node_value_counts(
  nodes: &[NodeProto],
  graph: &Graph,
  model: &Model,
) -> (HashMap<String, usize>, HashMap<String, usize>) {
  let mut input_counts = HashMap::new();
  let mut output_counts = HashMap::new();
  for node in nodes {
    for input in node.inputs.iter().filter(|name| !name.is_empty()) {
      *input_counts.entry(input.clone()).or_insert(0) += 1;
    }
    for output in node.outputs.iter().filter(|name| !name.is_empty()) {
      *output_counts.entry(output.clone()).or_insert(0) += 1;
    }
  }
  for input in &graph.inputs {
    input_counts.remove(model.strings.get(graph.values[input.index()].name));
  }
  for output in &graph.outputs {
    output_counts.remove(model.strings.get(graph.values[output.index()].name));
  }
  (input_counts, output_counts)
}

fn take_constant_initializer(
  proto: &mut NodeProto,
  input_counts: &HashMap<String, usize>,
  output_counts: &HashMap<String, usize>,
) -> Option<(String, ConstantInitializer)> {
  if proto.op_type != "Constant"
    || !proto.inputs.is_empty()
    || proto.outputs.len() != 1
    || proto.attributes.len() != 1
  {
    return None;
  }

  let output = proto.outputs[0].clone();
  if output.is_empty()
    || input_counts.get(&output).copied() != Some(1)
    || output_counts.get(&output).copied() != Some(1)
  {
    return None;
  }

  let attribute = proto.attributes.first()?;
  match (&*attribute.name, attribute.kind.as_ref()?) {
    ("value", AttributeKind::Tensor(_)) | ("sparse_value", AttributeKind::SparseTensor(_)) => {}
    _ => return None,
  }

  let attribute = proto.attributes.pop()?;
  let initializer = match attribute.kind? {
    AttributeKind::Tensor(tensor) => ConstantInitializer::Tensor(tensor),
    AttributeKind::SparseTensor(tensor) => ConstantInitializer::SparseTensor(tensor),
    _ => return None,
  };
  Some((output, initializer))
}

enum ConstantInitializer {
  Tensor(TensorProto),
  SparseTensor(SparseTensorProto),
}

#[derive(Debug, Default)]
struct FunctionInitializerMoves {
  by_function: HashMap<FunctionKey, Vec<TensorProto>>,
  hidden_inputs: HashMap<FunctionKey, HashSet<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FunctionKey {
  domain: String,
  name: String,
  overload: Option<String>,
}

fn move_graph_initializers_to_functions(
  graph: &mut GraphProto,
  functions: &[FunctionProto],
) -> FunctionInitializerMoves {
  let function_indices = last_function_indices(functions);
  if function_indices.is_empty() || graph.initializers.is_empty() {
    return FunctionInitializerMoves::default();
  }

  let mut function_uses = HashMap::new();
  let mut initializer_uses: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
  let initializer_names = graph
    .initializers
    .iter()
    .filter_map(|initializer| initializer.name.as_deref())
    .filter(|name| !name.is_empty())
    .collect::<HashSet<_>>();

  for (node_index, node) in graph.nodes.iter().enumerate() {
    let key = function_key_for_node(node);
    if function_indices.contains_key(&key) {
      *function_uses.entry(key).or_insert(0) += 1;
    }
    for (input_index, input) in node.inputs.iter().enumerate() {
      if initializer_names.contains(input.as_str()) {
        initializer_uses
          .entry(input.clone())
          .or_default()
          .push((node_index, input_index));
      }
    }
  }

  let mut moved_initializer_names = HashMap::new();
  let mut node_input_removals: HashMap<usize, Vec<usize>> = HashMap::new();
  for initializer in &graph.initializers {
    let Some(name) = initializer.name.as_deref().filter(|name| !name.is_empty()) else {
      continue;
    };
    let Some(uses) = initializer_uses.get(name) else {
      continue;
    };
    let [(node_index, input_index)] = uses.as_slice() else {
      continue;
    };
    let node = &graph.nodes[*node_index];
    let key = function_key_for_node(node);
    let Some(function_index) = function_indices.get(&key) else {
      continue;
    };
    if function_uses.get(&key).copied() != Some(1) {
      continue;
    }
    let function = &functions[*function_index];
    if function.inputs.get(*input_index).map(String::as_str) != Some(name) {
      continue;
    }
    moved_initializer_names.insert(name.to_owned(), key);
    node_input_removals
      .entry(*node_index)
      .or_default()
      .push(*input_index);
  }

  if moved_initializer_names.is_empty() {
    return FunctionInitializerMoves::default();
  }

  for (node_index, mut input_indices) in node_input_removals {
    input_indices.sort_unstable_by(|left, right| right.cmp(left));
    let node = &mut graph.nodes[node_index];
    for input_index in input_indices {
      if input_index < node.inputs.len() {
        node.inputs.remove(input_index);
      }
    }
  }

  let mut moves = FunctionInitializerMoves::default();
  let mut kept_initializers = Vec::with_capacity(graph.initializers.len());
  for initializer in graph.initializers.drain(..) {
    if let Some(key) = initializer
      .name
      .as_deref()
      .and_then(|name| moved_initializer_names.remove(name))
    {
      if let Some(name) = initializer.name.as_deref() {
        moves
          .hidden_inputs
          .entry(key.clone())
          .or_default()
          .insert(name.to_owned());
      }
      moves.by_function.entry(key).or_default().push(initializer);
    } else {
      kept_initializers.push(initializer);
    }
  }
  graph.initializers = kept_initializers;
  moves
}

fn propagate_function_initializer_moves(
  functions: &mut [FunctionProto],
  moves: &mut FunctionInitializerMoves,
) {
  let function_indices = last_function_indices(functions);
  if function_indices.is_empty() {
    return;
  }

  loop {
    let mut changed = false;
    for function_index in 0..functions.len() {
      let parent_key = function_key_for_function(&functions[function_index]);
      let Some(parent_initializers) = moves.by_function.get(&parent_key) else {
        continue;
      };
      let initializer_names = parent_initializers
        .iter()
        .filter_map(|initializer| initializer.name.as_deref())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect::<HashSet<_>>();
      if initializer_names.is_empty() {
        continue;
      }

      let mut function_uses = HashMap::new();
      let mut initializer_uses: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
      for (node_index, node) in functions[function_index].nodes.iter().enumerate() {
        let key = function_key_for_node(node);
        if function_indices.contains_key(&key) {
          *function_uses.entry(key).or_insert(0) += 1;
        }
        for (input_index, input) in node.inputs.iter().enumerate() {
          if initializer_names.contains(input) {
            initializer_uses
              .entry(input.clone())
              .or_default()
              .push((node_index, input_index));
          }
        }
      }

      let mut moved_names = HashMap::new();
      let mut node_input_removals: HashMap<usize, Vec<usize>> = HashMap::new();
      for name in &initializer_names {
        let Some(uses) = initializer_uses.get(name) else {
          continue;
        };
        let [(node_index, input_index)] = uses.as_slice() else {
          continue;
        };
        let node = &functions[function_index].nodes[*node_index];
        let child_key = function_key_for_node(node);
        if child_key == parent_key {
          continue;
        }
        let Some(child_index) = function_indices.get(&child_key) else {
          continue;
        };
        if function_uses.get(&child_key).copied() != Some(1) {
          continue;
        }
        let child = &functions[*child_index];
        if child.inputs.get(*input_index).map(String::as_str) != Some(name) {
          continue;
        }
        moved_names.insert(name.clone(), child_key);
        node_input_removals
          .entry(*node_index)
          .or_default()
          .push(*input_index);
      }

      if moved_names.is_empty() {
        continue;
      }

      for (node_index, mut input_indices) in node_input_removals {
        input_indices.sort_unstable_by(|left, right| right.cmp(left));
        let node = &mut functions[function_index].nodes[node_index];
        for input_index in input_indices {
          if input_index < node.inputs.len() {
            node.inputs.remove(input_index);
          }
        }
      }

      let mut kept = Vec::new();
      let mut moved = Vec::new();
      for initializer in moves.by_function.remove(&parent_key).unwrap_or_default() {
        if let Some(child_key) = initializer
          .name
          .as_deref()
          .and_then(|name| moved_names.remove(name))
        {
          if let Some(name) = initializer.name.as_deref() {
            moves
              .hidden_inputs
              .entry(child_key.clone())
              .or_default()
              .insert(name.to_owned());
          }
          moved.push((child_key, initializer));
        } else {
          kept.push(initializer);
        }
      }
      if !kept.is_empty() {
        moves.by_function.insert(parent_key, kept);
      }
      for (child_key, initializer) in moved {
        moves
          .by_function
          .entry(child_key)
          .or_default()
          .push(initializer);
      }
      changed = true;
    }
    if !changed {
      break;
    }
  }
}

fn last_function_indices(functions: &[FunctionProto]) -> HashMap<FunctionKey, usize> {
  let mut indices = HashMap::new();
  for (index, function) in functions.iter().enumerate() {
    indices.insert(function_key_for_function(function), index);
  }
  indices
}

fn function_input_type_specializations(
  model: &mut Model,
  graph: &GraphProto,
  functions: &[FunctionProto],
) -> HashMap<FunctionKey, HashMap<String, TypeInfo>> {
  let function_indices = last_function_indices(functions);
  if function_indices.is_empty() {
    return HashMap::new();
  }

  let mut value_types = HashMap::new();
  for value_info in graph
    .inputs
    .iter()
    .chain(graph.outputs.iter())
    .chain(graph.value_info.iter())
  {
    if value_info.name.is_empty() {
      continue;
    }
    if let Some(type_info) = value_info
      .r#type
      .clone()
      .and_then(|value| lower_type(model, value))
    {
      value_types.insert(value_info.name.clone(), type_info);
    }
  }

  let mut function_uses: HashMap<FunctionKey, Vec<&NodeProto>> = HashMap::new();
  for node in &graph.nodes {
    let key = function_key_for_node(node);
    if function_indices.contains_key(&key) {
      function_uses.entry(key).or_default().push(node);
    }
  }

  let mut specializations = HashMap::new();
  for (key, uses) in function_uses {
    if key.domain != "custom" {
      continue;
    }
    let [node] = uses.as_slice() else {
      continue;
    };
    let Some(function_index) = function_indices.get(&key).copied() else {
      continue;
    };
    let function = &functions[function_index];
    if !function.attributes.is_empty() {
      continue;
    }
    let mut input_types = HashMap::new();
    for (input_index, function_input) in function.inputs.iter().enumerate() {
      if function_input.is_empty() {
        continue;
      }
      let Some(call_input) = node.inputs.get(input_index).filter(|name| !name.is_empty()) else {
        continue;
      };
      if function_input != call_input {
        continue;
      }
      let Some(type_info) = value_types.get(call_input).cloned() else {
        continue;
      };
      input_types.insert(function_input.clone(), type_info);
    }
    if !input_types.is_empty() {
      specializations.insert(key, input_types);
    }
  }
  specializations
}

fn function_key_for_function(function: &FunctionProto) -> FunctionKey {
  function_key(
    function.domain.as_deref(),
    &function.name,
    function.overload.as_deref(),
  )
}

fn function_key_for_node(node: &NodeProto) -> FunctionKey {
  function_key(
    node.domain.as_deref(),
    &node.op_type,
    node.overload.as_deref(),
  )
}

fn function_key(domain: Option<&str>, name: &str, overload: Option<&str>) -> FunctionKey {
  FunctionKey {
    domain: domain.unwrap_or_default().to_owned(),
    name: name.to_owned(),
    overload: overload
      .filter(|overload| !overload.is_empty())
      .map(ToOwned::to_owned),
  }
}

fn attach_initializer(
  model: &mut Model,
  graph: &mut Graph,
  value_id: ValueId,
  initializer: ConstantInitializer,
  layout: Option<netron_rs_core::StringId>,
) {
  let (tensor, layout) = match initializer {
    ConstantInitializer::Tensor(tensor) => (lower_tensor(model, tensor), layout),
    ConstantInitializer::SparseTensor(tensor) => {
      let sparse = model.intern("sparse");
      (lower_sparse_tensor(model, tensor), Some(sparse))
    }
  };
  let element_type = tensor.element_type.clone();
  let shape = tensor.shape.clone();
  let tensor_id = model.add_tensor(tensor);
  let value = &mut graph.values[value_id.index()];
  value.initializer = Some(tensor_id);
  value.is_graph_input = false;
  value.type_info = Some(TypeInfo {
    element_type: Some(element_type),
    layout,
    denotation: None,
    shape,
  });
}

fn lower_function(
  model: &mut Model,
  proto: FunctionProto,
  moved_initializers: Vec<TensorProto>,
  hidden_inputs: HashSet<String>,
  input_types: HashMap<String, TypeInfo>,
) -> Result<Function, ModelError> {
  let mut values = Vec::new();
  let mut value_table = FunctionValueTable::default();
  for name in proto
    .inputs
    .iter()
    .filter(|name| !hidden_inputs.contains(*name))
  {
    let index = ensure_function_value(model, &mut values, &mut value_table, name);
    if let Some(type_info) = input_types.get(name).cloned() {
      values[index].type_info.get_or_insert(type_info);
    }
  }
  for name in &proto.outputs {
    ensure_function_value(model, &mut values, &mut value_table, name);
  }
  let mut moved_initializer_names = hidden_inputs;
  for initializer in moved_initializers {
    let Some(name) = initializer.name.clone().filter(|name| !name.is_empty()) else {
      continue;
    };
    moved_initializer_names.insert(name.clone());
    let value_index = ensure_function_value(model, &mut values, &mut value_table, &name);
    attach_function_initializer(
      model,
      &mut values,
      value_index,
      ConstantInitializer::Tensor(initializer),
    );
  }

  let (input_counts, output_counts) =
    function_value_counts(&proto.nodes, &proto.inputs, &proto.outputs);
  let mut nodes = Vec::with_capacity(proto.nodes.len());
  for mut proto_node in proto.nodes {
    if let Some((output, initializer)) =
      take_constant_initializer(&mut proto_node, &input_counts, &output_counts)
    {
      let value_index = ensure_function_value(model, &mut values, &mut value_table, &output);
      attach_function_initializer(model, &mut values, value_index, initializer);
      continue;
    }

    for name in proto_node
      .inputs
      .iter()
      .chain(proto_node.outputs.iter())
      .filter(|name| !name.is_empty())
    {
      ensure_function_value(model, &mut values, &mut value_table, name);
    }
    nodes.push(lower_function_node(model, proto_node)?);
  }

  Ok(Function {
    name: model.intern(&proto.name),
    domain: proto
      .domain
      .as_ref()
      .filter(|domain| !domain.is_empty())
      .map(|domain| model.intern(domain)),
    overload: proto
      .overload
      .as_ref()
      .filter(|overload| !overload.is_empty())
      .map(|overload| model.intern(overload)),
    description: proto.doc_string,
    metadata: proto.metadata_props,
    opsets: proto
      .opsets
      .into_iter()
      .filter_map(|opset| {
        opset.version.map(|version| OperatorSet {
          domain: opset.domain,
          version,
        })
      })
      .collect(),
    inputs: proto
      .inputs
      .iter()
      .filter(|name| !moved_initializer_names.contains(*name))
      .map(|name| model.intern(name))
      .collect(),
    outputs: proto
      .outputs
      .iter()
      .map(|name| model.intern(name))
      .collect(),
    attributes: proto
      .attributes
      .iter()
      .map(|name| model.intern(name))
      .collect(),
    values,
    nodes,
  })
}

fn function_value_counts(
  nodes: &[NodeProto],
  inputs: &[String],
  outputs: &[String],
) -> (HashMap<String, usize>, HashMap<String, usize>) {
  let mut input_counts = HashMap::new();
  let mut output_counts = HashMap::new();
  for node in nodes {
    for input in node.inputs.iter().filter(|name| !name.is_empty()) {
      *input_counts.entry(input.clone()).or_insert(0) += 1;
    }
    for output in node.outputs.iter().filter(|name| !name.is_empty()) {
      *output_counts.entry(output.clone()).or_insert(0) += 1;
    }
  }
  for input in inputs {
    input_counts.remove(input);
  }
  for output in outputs {
    output_counts.remove(output);
  }
  (input_counts, output_counts)
}

fn ensure_function_value(
  model: &mut Model,
  values: &mut Vec<FunctionValue>,
  table: &mut FunctionValueTable,
  name: &str,
) -> usize {
  if let Some(index) = table.by_name.get(name) {
    return *index;
  }
  let index = values.len();
  values.push(FunctionValue {
    name: model.intern(name),
    type_info: None,
    initializer: None,
  });
  table.by_name.insert(name.to_owned(), index);
  index
}

fn attach_function_initializer(
  model: &mut Model,
  values: &mut [FunctionValue],
  value_index: usize,
  initializer: ConstantInitializer,
) {
  let (tensor, layout) = match initializer {
    ConstantInitializer::Tensor(tensor) => (lower_tensor(model, tensor), None),
    ConstantInitializer::SparseTensor(tensor) => {
      let sparse = model.intern("sparse");
      (lower_sparse_tensor(model, tensor), Some(sparse))
    }
  };
  let element_type = tensor.element_type.clone();
  let shape = tensor.shape.clone();
  let tensor_id = model.add_tensor(tensor);
  let value = &mut values[value_index];
  value.initializer = Some(tensor_id);
  match &mut value.type_info {
    Some(type_info) => {
      if let Some(layout) = layout {
        type_info.layout.get_or_insert(layout);
      }
    }
    None => {
      value.type_info = Some(TypeInfo {
        element_type: Some(element_type),
        layout,
        denotation: None,
        shape,
      });
    }
  }
}

fn lower_function_node(model: &mut Model, proto: NodeProto) -> Result<FunctionNode, ModelError> {
  let op_domain = proto.domain.as_ref().filter(|domain| !domain.is_empty());
  let (op_type, overload) = node_operator_parts(
    op_domain.map(String::as_str),
    &proto.op_type,
    proto.overload.as_deref(),
  );
  let op_name = model.intern(op_type);
  let op_domain = op_domain.map(|domain| model.intern(domain));
  let attributes = proto
    .attributes
    .into_iter()
    .map(|attribute| lower_function_attribute(model, attribute))
    .collect::<Result<Vec<_>, _>>()?;
  Ok(FunctionNode {
    name: proto.name.as_ref().map(|name| model.intern(name)),
    description: proto.doc_string,
    metadata: node_metadata_props(proto.metadata_props),
    operator: Operator {
      domain: op_domain,
      name: op_name,
      overload: overload.map(|overload| model.intern(overload)),
      version: opset_version(model, proto.domain.as_deref()),
      origin: FORMAT,
    },
    inputs: proto
      .inputs
      .iter()
      .filter_map(|name| {
        if name.is_empty() {
          None
        } else {
          Some(Some(model.intern(name)))
        }
      })
      .collect(),
    outputs: proto
      .outputs
      .iter()
      .filter_map(|name| {
        if name.is_empty() {
          None
        } else {
          Some(Some(model.intern(name)))
        }
      })
      .collect(),
    attributes,
  })
}

fn node_operator_parts<'a>(
  domain: Option<&str>,
  op_type: &'a str,
  overload: Option<&'a str>,
) -> (&'a str, Option<&'a str>) {
  if domain == Some("pkg.torch.ops")
    && let Some((name, suffix)) = op_type.rsplit_once('.')
  {
    return (name, Some(suffix));
  }
  (
    op_type,
    overload.and_then(|overload| (!overload.is_empty()).then_some(overload)),
  )
}

fn node_metadata_props(mut metadata_props: BTreeMap<String, String>) -> BTreeMap<String, String> {
  if metadata_props
    .get("input_names")
    .is_some_and(|value| value != "[]")
  {
    metadata_props.remove("input_names");
  }
  metadata_props
}

fn lower_function_attribute(
  model: &mut Model,
  proto: AttributeProto,
) -> Result<Attribute, ModelError> {
  let name = model.intern(&proto.name);
  let value = match proto.kind {
    Some(AttributeKind::Float(value)) => AttributeValue::Float(value),
    Some(AttributeKind::Int(value)) => lower_attribute_int(&proto.name, value),
    Some(AttributeKind::String(value)) => match String::from_utf8(value) {
      Ok(value) => AttributeValue::String(model.intern(value)),
      Err(error) => AttributeValue::Bytes {
        byte_len: error.into_bytes().len(),
      },
    },
    Some(AttributeKind::Reference(value)) => AttributeValue::Reference(model.intern(value)),
    Some(AttributeKind::Tensor(value)) => {
      let tensor = lower_tensor(model, value);
      let tensor_id = model.add_tensor(tensor);
      AttributeValue::Tensor(tensor_id)
    }
    Some(AttributeKind::SparseTensor(value)) => {
      let tensor = lower_sparse_tensor(model, value);
      let tensor_id = model.add_tensor(tensor);
      AttributeValue::Tensor(tensor_id)
    }
    Some(AttributeKind::Floats(value)) => AttributeValue::Floats(value),
    Some(AttributeKind::Ints(value)) => AttributeValue::Ints(value),
    Some(AttributeKind::Strings(values)) => {
      let mut strings = Vec::with_capacity(values.len());
      for value in values {
        match String::from_utf8(value) {
          Ok(value) => strings.push(model.intern(value)),
          Err(error) => {
            return Ok(Attribute {
              name,
              value: AttributeValue::Unsupported(format!(
                "{} byte string is not UTF-8",
                error.into_bytes().len()
              )),
            });
          }
        }
      }
      AttributeValue::Strings(strings)
    }
    Some(AttributeKind::Tensors(values)) => AttributeValue::Tensors(
      values
        .into_iter()
        .map(|value| {
          let tensor = lower_tensor(model, value);
          model.add_tensor(tensor)
        })
        .collect(),
    ),
    Some(AttributeKind::SparseTensors(values)) => AttributeValue::Tensors(
      values
        .into_iter()
        .map(|value| {
          let tensor = lower_sparse_tensor(model, value);
          model.add_tensor(tensor)
        })
        .collect(),
    ),
    Some(AttributeKind::Graph(_)) | Some(AttributeKind::Graphs(_)) => {
      AttributeValue::Unsupported("graph attribute in function node".to_owned())
    }
    Some(AttributeKind::Type(value)) => AttributeValue::Type(value),
    None => AttributeValue::Unsupported("missing attribute value".to_owned()),
  };

  Ok(Attribute { name, value })
}

fn ensure_value_info(
  model: &mut Model,
  graph: &mut Graph,
  values: &mut ValueTable,
  proto: ValueInfoProto,
  preserve_metadata: bool,
) -> Option<ValueId> {
  if proto.name.is_empty() {
    return None;
  }
  let value_id = ensure_value(model, graph, values, &proto.name);
  if let Some(type_info) = proto.r#type.and_then(|value| lower_type(model, value)) {
    let value = &mut graph.values[value_id.index()];
    if preserve_metadata || !(value.is_graph_input || value.is_graph_output) {
      value.type_info = Some(type_info);
    }
  }
  if proto.doc_string.is_some() {
    graph.values[value_id.index()].description = proto.doc_string;
  }
  if preserve_metadata && !proto.metadata_props.is_empty() {
    graph.values[value_id.index()].metadata = proto.metadata_props;
  }
  Some(value_id)
}

fn normalize_graph_output_type(value: &mut Value) {
  let Some(type_info) = &mut value.type_info else {
    return;
  };
  let Some(dimension) = type_info.shape.first_mut() else {
    return;
  };
  let is_unknown_output_dimension = match dimension.value {
    DimensionValue::Known(0) => true,
    DimensionValue::Symbolic(_) => false,
    _ => false,
  };
  if is_unknown_output_dimension {
    dimension.value = DimensionValue::Unknown;
  }
}

fn ensure_optional_value(
  model: &mut Model,
  graph: &mut Graph,
  values: &mut ValueTable,
  name: &str,
) -> Option<ValueId> {
  if name.is_empty() {
    None
  } else {
    Some(ensure_value(model, graph, values, name))
  }
}

fn ensure_value(
  model: &mut Model,
  graph: &mut Graph,
  values: &mut ValueTable,
  name: &str,
) -> ValueId {
  if let Some(value_id) = values.by_name.get(name) {
    return *value_id;
  }

  let name_id = model.intern(name);
  let value_id = graph.add_value(Value::new(name_id));
  values.by_name.insert(name.to_owned(), value_id);
  value_id
}

fn lower_attribute(
  model: &mut Model,
  graph: &mut Graph,
  proto: AttributeProto,
) -> Result<Attribute, ModelError> {
  let name = model.intern(&proto.name);
  let value = match proto.kind {
    Some(AttributeKind::Float(value)) => AttributeValue::Float(value),
    Some(AttributeKind::Int(value)) => lower_attribute_int(&proto.name, value),
    Some(AttributeKind::String(value)) => match String::from_utf8(value) {
      Ok(value) => AttributeValue::String(model.intern(value)),
      Err(error) => AttributeValue::Bytes {
        byte_len: error.into_bytes().len(),
      },
    },
    Some(AttributeKind::Reference(value)) => AttributeValue::Reference(model.intern(value)),
    Some(AttributeKind::Tensor(value)) => {
      let tensor = lower_tensor(model, value);
      let tensor_id = model.add_tensor(tensor);
      AttributeValue::Tensor(tensor_id)
    }
    Some(AttributeKind::SparseTensor(value)) => {
      let tensor = lower_sparse_tensor(model, value);
      let tensor_id = model.add_tensor(tensor);
      AttributeValue::Tensor(tensor_id)
    }
    Some(AttributeKind::Graph(value)) => {
      let graph_id = lower_graph(model, value, Some(graph.id))?;
      graph.subgraphs.push(graph_id);
      AttributeValue::Graph(graph_id)
    }
    Some(AttributeKind::Floats(value)) => AttributeValue::Floats(value),
    Some(AttributeKind::Ints(value)) => AttributeValue::Ints(value),
    Some(AttributeKind::Strings(values)) => {
      let mut strings = Vec::with_capacity(values.len());
      for value in values {
        match String::from_utf8(value) {
          Ok(value) => strings.push(model.intern(value)),
          Err(error) => {
            return Ok(Attribute {
              name,
              value: AttributeValue::Unsupported(format!(
                "{} byte string is not UTF-8",
                error.into_bytes().len()
              )),
            });
          }
        }
      }
      AttributeValue::Strings(strings)
    }
    Some(AttributeKind::Tensors(values)) => AttributeValue::Tensors(
      values
        .into_iter()
        .map(|value| {
          let tensor = lower_tensor(model, value);
          model.add_tensor(tensor)
        })
        .collect(),
    ),
    Some(AttributeKind::SparseTensors(values)) => AttributeValue::Tensors(
      values
        .into_iter()
        .map(|value| {
          let tensor = lower_sparse_tensor(model, value);
          model.add_tensor(tensor)
        })
        .collect(),
    ),
    Some(AttributeKind::Graphs(values)) => {
      let mut graph_ids = Vec::with_capacity(values.len());
      for value in values {
        let graph_id = lower_graph(model, value, Some(graph.id))?;
        graph.subgraphs.push(graph_id);
        graph_ids.push(graph_id);
      }
      AttributeValue::Graphs(graph_ids)
    }
    Some(AttributeKind::Type(value)) => AttributeValue::Type(value),
    None => AttributeValue::Unsupported("missing attribute value".to_owned()),
  };

  Ok(Attribute { name, value })
}

fn lower_tensor(model: &mut Model, proto: TensorProto) -> Tensor {
  let TensorProto {
    dims,
    data_type,
    name,
    doc_string,
    raw_data_len,
    value_count,
    external_data,
    metadata_props,
  } = proto;
  let name = name
    .filter(|name| !name.is_empty())
    .map(|name| model.intern(name));
  let storage = if !external_data.is_empty() {
    TensorStorage::External {
      entries: external_data,
    }
  } else if raw_data_len > 0 {
    TensorStorage::InlineBytes {
      byte_len: raw_data_len,
    }
  } else if value_count > 0 {
    TensorStorage::ElementList { len: value_count }
  } else {
    TensorStorage::Absent
  };

  let mut tensor = Tensor::metadata_only(
    name,
    lower_element_type(data_type.unwrap_or_default()),
    dims.into_iter().map(Dimension::known).collect(),
    storage,
  );
  tensor.description = doc_string;
  tensor.metadata = metadata_props;
  tensor
}

fn lower_attribute_int(name: &str, value: i64) -> AttributeValue {
  if name == "to" {
    AttributeValue::Type(element_type_name_for_attribute(value as i32).to_owned())
  } else {
    AttributeValue::Int(value)
  }
}

fn element_type_name_for_attribute(value: i32) -> &'static str {
  match lower_element_type(value) {
    TensorElementType::Unknown => "unknown",
    TensorElementType::Float16 => "float16",
    TensorElementType::Float32 => "float32",
    TensorElementType::Float64 => "float64",
    TensorElementType::Float8e4m3fn => "float8e4m3fn",
    TensorElementType::Float8e4m3fnuz => "float8e4m3fnuz",
    TensorElementType::Float8e5m2 => "float8e5m2",
    TensorElementType::Float8e5m2fnuz => "float8e5m2fnuz",
    TensorElementType::Float8e8m0 => "float8e8m0",
    TensorElementType::Float4e2m1 => "float4e2m1",
    TensorElementType::BFloat16 => "bfloat16",
    TensorElementType::Int2 => "int2",
    TensorElementType::Int4 => "int4",
    TensorElementType::Int8 => "int8",
    TensorElementType::Int16 => "int16",
    TensorElementType::Int32 => "int32",
    TensorElementType::Int64 => "int64",
    TensorElementType::Uint2 => "uint2",
    TensorElementType::Uint4 => "uint4",
    TensorElementType::Uint8 => "uint8",
    TensorElementType::Uint16 => "uint16",
    TensorElementType::Uint32 => "uint32",
    TensorElementType::Uint64 => "uint64",
    TensorElementType::Bool => "boolean",
    TensorElementType::String => "string",
    TensorElementType::Complex64 => "complex<float32>",
    TensorElementType::Complex128 => "complex<float64>",
    TensorElementType::Other(_) => "unknown",
  }
}

fn element_type_from_attribute_name(value: &str) -> Option<TensorElementType> {
  match value {
    "unknown" => Some(TensorElementType::Unknown),
    "float16" => Some(TensorElementType::Float16),
    "float32" => Some(TensorElementType::Float32),
    "float64" => Some(TensorElementType::Float64),
    "float8e4m3fn" => Some(TensorElementType::Float8e4m3fn),
    "float8e4m3fnuz" => Some(TensorElementType::Float8e4m3fnuz),
    "float8e5m2" => Some(TensorElementType::Float8e5m2),
    "float8e5m2fnuz" => Some(TensorElementType::Float8e5m2fnuz),
    "float8e8m0" => Some(TensorElementType::Float8e8m0),
    "float4e2m1" => Some(TensorElementType::Float4e2m1),
    "bfloat16" => Some(TensorElementType::BFloat16),
    "int2" => Some(TensorElementType::Int2),
    "int4" => Some(TensorElementType::Int4),
    "int8" => Some(TensorElementType::Int8),
    "int16" => Some(TensorElementType::Int16),
    "int32" => Some(TensorElementType::Int32),
    "int64" => Some(TensorElementType::Int64),
    "uint2" => Some(TensorElementType::Uint2),
    "uint4" => Some(TensorElementType::Uint4),
    "uint8" => Some(TensorElementType::Uint8),
    "uint16" => Some(TensorElementType::Uint16),
    "uint32" => Some(TensorElementType::Uint32),
    "uint64" => Some(TensorElementType::Uint64),
    "boolean" => Some(TensorElementType::Bool),
    "string" => Some(TensorElementType::String),
    "complex<float32>" => Some(TensorElementType::Complex64),
    "complex<float64>" => Some(TensorElementType::Complex128),
    _ => None,
  }
}

fn lower_sparse_tensor(model: &mut Model, proto: SparseTensorProto) -> Tensor {
  let values = proto.values.unwrap_or_default();
  let indices = proto.indices.unwrap_or_default();
  let element_type = lower_element_type(values.data_type.unwrap_or_default());
  let name = values
    .name
    .as_ref()
    .filter(|name| !name.is_empty())
    .map(|name| model.intern(name));
  let values_tensor = lower_tensor(model, values);
  let values_id = model.add_tensor(values_tensor);
  let indices_tensor = lower_tensor(model, indices);
  let indices_id = model.add_tensor(indices_tensor);
  Tensor::metadata_only(
    name,
    element_type,
    proto.dims.into_iter().map(Dimension::known).collect(),
    TensorStorage::Sparse {
      values: values_id,
      indices: indices_id,
    },
  )
}

fn lower_type(model: &mut Model, proto: TypeProto) -> Option<TypeInfo> {
  let denotation = proto
    .denotation
    .filter(|value| !value.is_empty())
    .map(|value| model.intern(value));
  let sparse = model.intern("sparse");
  if let Some(tensor) = proto.tensor {
    Some(lower_tensor_type(model, tensor, None, denotation))
  } else if let Some(tensor) = proto.sparse_tensor {
    Some(lower_tensor_type(model, tensor, Some(sparse), denotation))
  } else if proto.has_non_tensor || denotation.is_some() {
    Some(TypeInfo {
      element_type: None,
      layout: None,
      denotation,
      shape: Vec::new(),
    })
  } else {
    None
  }
}

fn lower_tensor_type(
  model: &mut Model,
  tensor: TensorTypeProto,
  layout: Option<netron_rs_core::StringId>,
  denotation: Option<netron_rs_core::StringId>,
) -> TypeInfo {
  TypeInfo {
    element_type: tensor.elem_type.map(lower_element_type),
    layout,
    denotation,
    shape: tensor
      .shape
      .into_iter()
      .map(|dimension| lower_dimension(model, dimension))
      .collect(),
  }
}

fn lower_dimension(model: &mut Model, dimension: ShapeDimension) -> Dimension {
  let denotation = dimension
    .denotation
    .filter(|value| !value.is_empty())
    .map(|value| model.intern(value));
  match dimension.value {
    ShapeDimensionValue::Known(value) => Dimension {
      value: DimensionValue::Known(value),
      denotation,
    },
    ShapeDimensionValue::Symbolic(value) => Dimension {
      value: DimensionValue::Symbolic(model.intern(value)),
      denotation,
    },
    ShapeDimensionValue::Unknown => Dimension {
      value: DimensionValue::Unknown,
      denotation,
    },
  }
}

fn opset_version(model: &Model, domain: Option<&str>) -> Option<i64> {
  let domain = domain.filter(|domain| !domain.is_empty());
  model
    .metadata
    .opsets
    .iter()
    .find(|opset| opset.domain.as_deref().filter(|value| !value.is_empty()) == domain)
    .map(|opset| opset.version)
}

fn lower_element_type(value: i32) -> TensorElementType {
  match value {
    1 => TensorElementType::Float32,
    2 => TensorElementType::Uint8,
    3 => TensorElementType::Int8,
    4 => TensorElementType::Uint16,
    5 => TensorElementType::Int16,
    6 => TensorElementType::Int32,
    7 => TensorElementType::Int64,
    8 => TensorElementType::String,
    9 => TensorElementType::Bool,
    10 => TensorElementType::Float16,
    11 => TensorElementType::Float64,
    12 => TensorElementType::Uint32,
    13 => TensorElementType::Uint64,
    14 => TensorElementType::Complex64,
    15 => TensorElementType::Complex128,
    16 => TensorElementType::BFloat16,
    17 => TensorElementType::Float8e4m3fn,
    18 => TensorElementType::Float8e4m3fnuz,
    19 => TensorElementType::Float8e5m2,
    20 => TensorElementType::Float8e5m2fnuz,
    21 => TensorElementType::Uint4,
    22 => TensorElementType::Int4,
    23 => TensorElementType::Float4e2m1,
    24 => TensorElementType::Float8e8m0,
    25 => TensorElementType::Uint2,
    26 => TensorElementType::Int2,
    0 => TensorElementType::Unknown,
    value => TensorElementType::Other(value.to_string()),
  }
}

fn looks_like_json_object(data: &[u8]) -> bool {
  data
    .iter()
    .copied()
    .find(|byte| !byte.is_ascii_whitespace())
    == Some(b'{')
}

fn parse_json_model_proto(data: &[u8]) -> Result<ModelProto, ModelError> {
  let value = serde_json::from_slice::<JsonValue>(data)
    .map_err(|error| invalid(format!("ONNX JSON parse failed: {error}")))?;
  json_model_proto(&value)
}

fn json_model_proto(value: &JsonValue) -> Result<ModelProto, ModelError> {
  let mut model = ModelProto {
    ir_version: json_i64_field(value, &["irVersion", "ir_version"]),
    producer_name: json_string_field(value, &["producerName", "producer_name"]),
    producer_version: json_string_field(value, &["producerVersion", "producer_version"]),
    domain: json_string_field(value, &["domain"]),
    model_version: json_i64_field(value, &["modelVersion", "model_version"]),
    doc_string: json_string_field(value, &["docString", "doc_string"]),
    metadata_props: json_string_entry_map(value, &["metadataProps", "metadata_props"]),
    ..Default::default()
  };
  if let Some(graph) = json_field(value, &["graph"]) {
    model.graph = Some(json_graph_proto(graph)?);
  }
  if let Some(opsets) = json_array_field(value, &["opsetImport", "opsets", "opset_import"]) {
    model.opsets = opsets.iter().map(json_operator_set_id_proto).collect();
  }
  if let Some(functions) = json_array_field(value, &["functions", "function"]) {
    model.functions = functions
      .iter()
      .map(json_function_proto)
      .collect::<Result<_, _>>()?;
  }
  Ok(model)
}

fn json_operator_set_id_proto(value: &JsonValue) -> OperatorSetIdProto {
  OperatorSetIdProto {
    domain: json_string_field(value, &["domain"]),
    version: json_i64_field(value, &["version"]),
  }
}

fn json_function_proto(value: &JsonValue) -> Result<FunctionProto, ModelError> {
  Ok(FunctionProto {
    name: json_string_field(value, &["name"]).unwrap_or_default(),
    inputs: json_string_list_field(value, &["input", "inputs"]),
    outputs: json_string_list_field(value, &["output", "outputs"]),
    attributes: json_string_list_field(value, &["attribute", "attributes"]),
    nodes: json_array_field(value, &["node", "nodes"])
      .into_iter()
      .flatten()
      .map(json_node_proto)
      .collect::<Result<_, _>>()?,
    doc_string: json_string_field(value, &["docString", "doc_string"]),
    opsets: json_array_field(value, &["opsetImport", "opsets", "opset_import"])
      .into_iter()
      .flatten()
      .map(json_operator_set_id_proto)
      .collect(),
    domain: json_string_field(value, &["domain"]),
    overload: json_string_field(value, &["overload"]),
    metadata_props: json_string_entry_map(value, &["metadataProps", "metadata_props"]),
  })
}

fn json_graph_proto(value: &JsonValue) -> Result<GraphProto, ModelError> {
  Ok(GraphProto {
    nodes: json_array_field(value, &["node", "nodes"])
      .into_iter()
      .flatten()
      .map(json_node_proto)
      .collect::<Result<_, _>>()?,
    name: json_string_field(value, &["name"]),
    doc_string: json_string_field(value, &["docString", "doc_string"]),
    initializers: json_array_field(value, &["initializer", "initializers"])
      .into_iter()
      .flatten()
      .map(json_tensor_proto)
      .collect(),
    sparse_initializers: json_array_field(value, &["sparseInitializer", "sparse_initializers"])
      .into_iter()
      .flatten()
      .map(json_sparse_tensor_proto)
      .collect(),
    inputs: json_array_field(value, &["input", "inputs"])
      .into_iter()
      .flatten()
      .map(json_value_info_proto)
      .collect(),
    outputs: json_array_field(value, &["output", "outputs"])
      .into_iter()
      .flatten()
      .map(json_value_info_proto)
      .collect(),
    value_info: json_array_field(value, &["valueInfo", "value_info"])
      .into_iter()
      .flatten()
      .map(json_value_info_proto)
      .collect(),
    quantization_annotations: json_array_field(
      value,
      &["quantizationAnnotation", "quantization_annotations"],
    )
    .into_iter()
    .flatten()
    .map(json_tensor_annotation_proto)
    .collect(),
    metadata_props: json_string_entry_map(value, &["metadataProps", "metadata_props"]),
  })
}

fn json_node_proto(value: &JsonValue) -> Result<NodeProto, ModelError> {
  Ok(NodeProto {
    inputs: json_string_list_field(value, &["input", "inputs"]),
    outputs: json_string_list_field(value, &["output", "outputs"]),
    name: json_string_field(value, &["name"]),
    op_type: json_string_field(value, &["opType", "op_type"]).unwrap_or_default(),
    attributes: json_array_field(value, &["attribute", "attributes"])
      .into_iter()
      .flatten()
      .map(json_attribute_proto)
      .collect::<Result<_, _>>()?,
    doc_string: json_string_field(value, &["docString", "doc_string"]),
    domain: json_string_field(value, &["domain"]),
    overload: json_string_field(value, &["overload"]),
    metadata_props: json_string_entry_map(value, &["metadataProps", "metadata_props"]),
  })
}

fn json_attribute_proto(value: &JsonValue) -> Result<AttributeProto, ModelError> {
  let name = json_string_field(value, &["name"]).unwrap_or_default();
  let kind = if let Some(values) = json_f32_list_field(value, &["floats"]) {
    Some(AttributeKind::Floats(values))
  } else if let Some(values) = json_i64_list_field(value, &["ints"]) {
    Some(AttributeKind::Ints(values))
  } else if let Some(values) = json_array_field(value, &["strings"]) {
    Some(AttributeKind::Strings(
      values
        .iter()
        .map(json_bytes_value)
        .collect::<Result<_, _>>()?,
    ))
  } else if let Some(value) = json_field(value, &["t"]) {
    Some(AttributeKind::Tensor(json_tensor_proto(value)))
  } else if let Some(value) = json_field(value, &["g"]) {
    Some(AttributeKind::Graph(json_graph_proto(value)?))
  } else if let Some(values) = json_array_field(value, &["tensors"]) {
    Some(AttributeKind::Tensors(
      values.iter().map(json_tensor_proto).collect(),
    ))
  } else if let Some(values) = json_array_field(value, &["graphs"]) {
    Some(AttributeKind::Graphs(
      values
        .iter()
        .map(json_graph_proto)
        .collect::<Result<_, _>>()?,
    ))
  } else if let Some(value) = json_field(value, &["sparseTensor", "sparse_tensor"]) {
    Some(AttributeKind::SparseTensor(json_sparse_tensor_proto(value)))
  } else if let Some(values) = json_array_field(value, &["sparseTensors", "sparse_tensors"]) {
    Some(AttributeKind::SparseTensors(
      values.iter().map(json_sparse_tensor_proto).collect(),
    ))
  } else if json_field(value, &["tp", "typeProto", "type_proto"]).is_some() {
    Some(AttributeKind::Type("type_proto".to_owned()))
  } else if let Some(value) = json_string_field(value, &["refAttrName", "ref_attr_name"]) {
    Some(AttributeKind::Reference(value))
  } else if let Some(value) = json_f32_field(value, &["f"]) {
    Some(AttributeKind::Float(value))
  } else if let Some(value) = json_i64_field(value, &["i"]) {
    Some(AttributeKind::Int(value))
  } else if let Some(value) = json_field(value, &["s"]) {
    Some(AttributeKind::String(json_bytes_value(value)?))
  } else {
    None
  };
  Ok(AttributeProto { name, kind })
}

fn json_tensor_proto(value: &JsonValue) -> TensorProto {
  let raw_data_len = json_field(value, &["rawData", "raw_data"])
    .and_then(|value| json_bytes_value(value).ok())
    .map(|value| value.len())
    .unwrap_or_default();
  let mut value_count = 0;
  for key in [
    "floatData",
    "int32Data",
    "int64Data",
    "doubleData",
    "uint64Data",
    "stringData",
  ] {
    if let Some(values) = json_array_field(value, &[key]) {
      value_count += values.len();
    }
  }
  TensorProto {
    dims: json_i64_list_field(value, &["dims"]).unwrap_or_default(),
    data_type: json_i32_field(value, &["dataType", "data_type"]),
    name: json_string_field(value, &["name"]),
    doc_string: json_string_field(value, &["docString", "doc_string"]),
    raw_data_len,
    value_count,
    external_data: json_string_entry_map(value, &["externalData", "external_data"]),
    metadata_props: json_string_entry_map(value, &["metadataProps", "metadata_props"]),
  }
}

fn json_sparse_tensor_proto(value: &JsonValue) -> SparseTensorProto {
  SparseTensorProto {
    values: json_field(value, &["values"]).map(json_tensor_proto),
    indices: json_field(value, &["indices"]).map(json_tensor_proto),
    dims: json_i64_list_field(value, &["dims"]).unwrap_or_default(),
  }
}

fn json_tensor_annotation_proto(value: &JsonValue) -> TensorAnnotationProto {
  TensorAnnotationProto {
    tensor_name: json_string_field(value, &["tensorName", "tensor_name"]).unwrap_or_default(),
    parameters: json_string_entry_map(
      value,
      &["quantParameterTensorNames", "quant_parameter_tensor_names"],
    )
    .into_iter()
    .collect(),
  }
}

fn json_value_info_proto(value: &JsonValue) -> ValueInfoProto {
  ValueInfoProto {
    name: json_string_field(value, &["name"]).unwrap_or_default(),
    r#type: json_field(value, &["type"]).map(json_type_proto),
    doc_string: json_string_field(value, &["docString", "doc_string"]),
    metadata_props: json_string_entry_map(value, &["metadataProps", "metadata_props"]),
  }
}

fn json_type_proto(value: &JsonValue) -> TypeProto {
  TypeProto {
    tensor: json_field(value, &["tensorType", "tensor_type"]).map(json_tensor_type_proto),
    sparse_tensor: json_field(value, &["sparseTensorType", "sparse_tensor_type"])
      .map(json_tensor_type_proto),
    denotation: json_string_field(value, &["denotation"]),
    has_non_tensor: json_field(
      value,
      &[
        "sequenceType",
        "mapType",
        "optionalType",
        "opaqueType",
        "sequence_type",
        "map_type",
        "optional_type",
        "opaque_type",
      ],
    )
    .is_some(),
  }
}

fn json_tensor_type_proto(value: &JsonValue) -> TensorTypeProto {
  TensorTypeProto {
    elem_type: json_i32_field(value, &["elemType", "elem_type"]),
    shape: json_field(value, &["shape"])
      .and_then(|shape| json_array_field(shape, &["dim"]))
      .into_iter()
      .flatten()
      .map(json_shape_dimension)
      .collect(),
  }
}

fn json_shape_dimension(value: &JsonValue) -> ShapeDimension {
  let denotation = json_string_field(value, &["denotation"]);
  let value = if let Some(value) = json_i64_field(value, &["dimValue", "dim_value"]) {
    ShapeDimensionValue::Known(value)
  } else if let Some(value) = json_string_field(value, &["dimParam", "dim_param"]) {
    ShapeDimensionValue::Symbolic(value)
  } else {
    ShapeDimensionValue::Unknown
  };
  ShapeDimension { value, denotation }
}

fn json_field<'a>(value: &'a JsonValue, names: &[&str]) -> Option<&'a JsonValue> {
  let object = value.as_object()?;
  names.iter().find_map(|name| object.get(*name))
}

fn json_array_field<'a>(value: &'a JsonValue, names: &[&str]) -> Option<&'a Vec<JsonValue>> {
  json_field(value, names).and_then(JsonValue::as_array)
}

fn json_string_field(value: &JsonValue, names: &[&str]) -> Option<String> {
  json_field(value, names).and_then(json_string_value)
}

fn json_string_value(value: &JsonValue) -> Option<String> {
  match value {
    JsonValue::String(value) => Some(value.clone()),
    JsonValue::Number(value) => Some(value.to_string()),
    _ => None,
  }
}

fn json_string_list_field(value: &JsonValue, names: &[&str]) -> Vec<String> {
  json_array_field(value, names)
    .into_iter()
    .flatten()
    .filter_map(json_string_value)
    .collect()
}

fn json_i64_field(value: &JsonValue, names: &[&str]) -> Option<i64> {
  json_field(value, names).and_then(json_i64_value)
}

fn json_i32_field(value: &JsonValue, names: &[&str]) -> Option<i32> {
  json_i64_field(value, names).and_then(|value| i32::try_from(value).ok())
}

fn json_i64_value(value: &JsonValue) -> Option<i64> {
  match value {
    JsonValue::Number(value) => value
      .as_i64()
      .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok())),
    JsonValue::String(value) => value.parse::<i64>().ok(),
    _ => None,
  }
}

fn json_i64_list_field(value: &JsonValue, names: &[&str]) -> Option<Vec<i64>> {
  json_array_field(value, names).map(|values| values.iter().filter_map(json_i64_value).collect())
}

fn json_f32_field(value: &JsonValue, names: &[&str]) -> Option<f32> {
  json_field(value, names).and_then(json_f32_value)
}

fn json_f32_value(value: &JsonValue) -> Option<f32> {
  match value {
    JsonValue::Number(value) => value.as_f64().map(|value| value as f32),
    JsonValue::String(value) => value.parse::<f32>().ok(),
    _ => None,
  }
}

fn json_f32_list_field(value: &JsonValue, names: &[&str]) -> Option<Vec<f32>> {
  json_array_field(value, names).map(|values| values.iter().filter_map(json_f32_value).collect())
}

fn json_bytes_value(value: &JsonValue) -> Result<Vec<u8>, ModelError> {
  let Some(value) = value.as_str() else {
    return Err(invalid("ONNX JSON bytes field is not a string"));
  };
  base64_decode(value)
}

fn json_string_entry_map(value: &JsonValue, names: &[&str]) -> BTreeMap<String, String> {
  let mut entries = BTreeMap::new();
  for entry in json_array_field(value, names).into_iter().flatten() {
    let Some(key) = json_string_field(entry, &["key"]) else {
      continue;
    };
    entries.insert(
      key,
      json_string_field(entry, &["value"]).unwrap_or_default(),
    );
  }
  entries
}

fn base64_decode(value: &str) -> Result<Vec<u8>, ModelError> {
  let mut output = Vec::with_capacity(value.len() * 3 / 4);
  let mut buffer = 0_u32;
  let mut bits = 0_u8;
  let mut padded = false;
  for byte in value.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
    if byte == b'=' {
      padded = true;
      continue;
    }
    if padded {
      return Err(invalid("ONNX JSON base64 has data after padding"));
    }
    let value = match byte {
      b'A'..=b'Z' => u32::from(byte - b'A'),
      b'a'..=b'z' => u32::from(byte - b'a' + 26),
      b'0'..=b'9' => u32::from(byte - b'0' + 52),
      b'+' => 62,
      b'/' => 63,
      _ => return Err(invalid("ONNX JSON base64 contains invalid character")),
    };
    buffer = (buffer << 6) | value;
    bits += 6;
    while bits >= 8 {
      bits -= 8;
      output.push((buffer >> bits) as u8);
      buffer &= (1 << bits) - 1;
    }
  }
  Ok(output)
}

#[derive(Default)]
struct ValueTable {
  by_name: HashMap<String, ValueId>,
}

#[derive(Default)]
struct FunctionValueTable {
  by_name: HashMap<String, usize>,
}

#[derive(Debug, Default)]
struct ModelProto {
  ir_version: Option<i64>,
  producer_name: Option<String>,
  producer_version: Option<String>,
  domain: Option<String>,
  model_version: Option<i64>,
  doc_string: Option<String>,
  graph: Option<GraphProto>,
  opsets: Vec<OperatorSetIdProto>,
  metadata_props: BTreeMap<String, String>,
  functions: Vec<FunctionProto>,
}

impl ModelProto {
  fn is_onnx_like(&self) -> bool {
    self.graph.is_some()
      || self.ir_version.is_some()
      || self.producer_name.is_some()
      || !self.opsets.is_empty()
      || !self.metadata_props.is_empty()
      || !self.functions.is_empty()
  }

  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut model = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => model.ir_version = Some(field.int64()?),
        2 => model.producer_name = Some(field.string()?),
        3 => model.producer_version = Some(field.string()?),
        4 => model.domain = Some(field.string()?),
        5 => model.model_version = Some(field.int64()?),
        6 => model.doc_string = Some(field.string()?),
        7 => model.graph = Some(GraphProto::decode(field.bytes()?)?),
        8 => model
          .opsets
          .push(OperatorSetIdProto::decode(field.bytes()?)?),
        14 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            model
              .metadata_props
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        25 => model.functions.push(FunctionProto::decode(field.bytes()?)?),
        _ => {}
      }
    }
    Ok(model)
  }
}

#[derive(Debug, Default)]
struct OperatorSetIdProto {
  domain: Option<String>,
  version: Option<i64>,
}

#[derive(Debug, Default)]
struct FunctionProto {
  name: String,
  inputs: Vec<String>,
  outputs: Vec<String>,
  attributes: Vec<String>,
  nodes: Vec<NodeProto>,
  doc_string: Option<String>,
  opsets: Vec<OperatorSetIdProto>,
  domain: Option<String>,
  overload: Option<String>,
  metadata_props: BTreeMap<String, String>,
}

impl FunctionProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut function = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => function.name = field.string()?,
        4 => function.inputs.push(field.string()?),
        5 => function.outputs.push(field.string()?),
        6 => function.attributes.push(field.string()?),
        7 => function.nodes.push(NodeProto::decode(field.bytes()?)?),
        8 => function.doc_string = Some(field.string()?),
        9 => function
          .opsets
          .push(OperatorSetIdProto::decode(field.bytes()?)?),
        10 => function.domain = Some(field.string()?),
        13 => function.overload = Some(field.string()?),
        14 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            function
              .metadata_props
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        _ => {}
      }
    }
    Ok(function)
  }
}

impl OperatorSetIdProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut opset = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => opset.domain = Some(field.string()?),
        2 => opset.version = Some(field.int64()?),
        _ => {}
      }
    }
    Ok(opset)
  }
}

#[derive(Debug, Default)]
struct GraphProto {
  nodes: Vec<NodeProto>,
  name: Option<String>,
  doc_string: Option<String>,
  initializers: Vec<TensorProto>,
  sparse_initializers: Vec<SparseTensorProto>,
  inputs: Vec<ValueInfoProto>,
  outputs: Vec<ValueInfoProto>,
  value_info: Vec<ValueInfoProto>,
  quantization_annotations: Vec<TensorAnnotationProto>,
  metadata_props: BTreeMap<String, String>,
}

impl GraphProto {
  fn is_graph_like(&self) -> bool {
    !self.nodes.is_empty()
      || !self.initializers.is_empty()
      || !self.sparse_initializers.is_empty()
      || !self.inputs.is_empty()
      || !self.outputs.is_empty()
      || !self.value_info.is_empty()
      || !self.quantization_annotations.is_empty()
      || !self.metadata_props.is_empty()
  }

  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut graph = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => graph.nodes.push(NodeProto::decode(field.bytes()?)?),
        2 => graph.name = Some(field.string()?),
        5 => graph
          .initializers
          .push(TensorProto::decode(field.bytes()?)?),
        10 => graph.doc_string = Some(field.string()?),
        11 => graph.inputs.push(ValueInfoProto::decode(field.bytes()?)?),
        12 => graph.outputs.push(ValueInfoProto::decode(field.bytes()?)?),
        13 => graph
          .value_info
          .push(ValueInfoProto::decode(field.bytes()?)?),
        14 => graph
          .quantization_annotations
          .push(TensorAnnotationProto::decode(field.bytes()?)?),
        15 => graph
          .sparse_initializers
          .push(SparseTensorProto::decode(field.bytes()?)?),
        16 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            graph
              .metadata_props
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        _ => {}
      }
    }
    Ok(graph)
  }
}

#[derive(Debug, Default)]
struct NodeProto {
  inputs: Vec<String>,
  outputs: Vec<String>,
  name: Option<String>,
  op_type: String,
  attributes: Vec<AttributeProto>,
  doc_string: Option<String>,
  domain: Option<String>,
  overload: Option<String>,
  metadata_props: BTreeMap<String, String>,
}

impl NodeProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut node = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => node.inputs.push(field.string()?),
        2 => node.outputs.push(field.string()?),
        3 => node.name = Some(field.string()?),
        4 => node.op_type = field.string()?,
        5 => node
          .attributes
          .push(AttributeProto::decode(field.bytes()?)?),
        6 => node.doc_string = Some(field.string()?),
        7 => node.domain = Some(field.string()?),
        8 => node.overload = Some(field.string()?),
        9 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            node
              .metadata_props
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        _ => {}
      }
    }
    Ok(node)
  }
}

#[derive(Debug, Default)]
struct AttributeProto {
  name: String,
  kind: Option<AttributeKind>,
}

impl AttributeProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut attribute = Self::default();
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut strings = Vec::new();
    let mut tensors = Vec::new();
    let mut graphs = Vec::new();
    let mut sparse_tensors = Vec::new();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => attribute.name = field.string()?,
        2 => attribute.kind = Some(AttributeKind::Float(field.float32()?)),
        3 => attribute.kind = Some(AttributeKind::Int(field.int64()?)),
        4 => attribute.kind = Some(AttributeKind::String(field.bytes()?.to_vec())),
        5 => attribute.kind = Some(AttributeKind::Tensor(TensorProto::decode(field.bytes()?)?)),
        6 => attribute.kind = Some(AttributeKind::Graph(GraphProto::decode(field.bytes()?)?)),
        7 => floats.extend(field.repeated_f32()?),
        8 => ints.extend(field.repeated_i64()?),
        9 => strings.push(field.bytes()?.to_vec()),
        10 => tensors.push(TensorProto::decode(field.bytes()?)?),
        11 => graphs.push(GraphProto::decode(field.bytes()?)?),
        22 => {
          attribute.kind = Some(AttributeKind::SparseTensor(SparseTensorProto::decode(
            field.bytes()?,
          )?))
        }
        23 => sparse_tensors.push(SparseTensorProto::decode(field.bytes()?)?),
        14 => {
          field.bytes()?;
          attribute.kind = Some(AttributeKind::Type("type_proto".to_owned()));
        }
        21 => attribute.kind = Some(AttributeKind::Reference(field.string()?)),
        _ => {}
      }
    }
    if !strings.is_empty() {
      attribute.kind = Some(AttributeKind::Strings(strings));
    }
    if !floats.is_empty() {
      attribute.kind = Some(AttributeKind::Floats(floats));
    }
    if !ints.is_empty() {
      attribute.kind = Some(AttributeKind::Ints(ints));
    }
    if !tensors.is_empty() {
      attribute.kind = Some(AttributeKind::Tensors(tensors));
    }
    if !graphs.is_empty() {
      attribute.kind = Some(AttributeKind::Graphs(graphs));
    }
    if !sparse_tensors.is_empty() {
      attribute.kind = Some(AttributeKind::SparseTensors(sparse_tensors));
    }
    Ok(attribute)
  }
}

#[derive(Debug)]
enum AttributeKind {
  Float(f32),
  Int(i64),
  String(Vec<u8>),
  Reference(String),
  Tensor(TensorProto),
  Graph(GraphProto),
  SparseTensor(SparseTensorProto),
  Floats(Vec<f32>),
  Ints(Vec<i64>),
  Strings(Vec<Vec<u8>>),
  Tensors(Vec<TensorProto>),
  SparseTensors(Vec<SparseTensorProto>),
  Graphs(Vec<GraphProto>),
  Type(String),
}

#[derive(Debug, Default)]
struct TensorProto {
  dims: Vec<i64>,
  data_type: Option<i32>,
  name: Option<String>,
  doc_string: Option<String>,
  raw_data_len: usize,
  value_count: usize,
  external_data: BTreeMap<String, String>,
  metadata_props: BTreeMap<String, String>,
}

impl TensorProto {
  fn is_tensor_like(&self) -> bool {
    self.data_type.is_some()
      && (!self.dims.is_empty()
        || self.name.as_ref().is_some_and(|name| !name.is_empty())
        || self.raw_data_len > 0
        || self.value_count > 0
        || !self.external_data.is_empty())
  }

  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut tensor = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => tensor.dims.extend(field.repeated_i64()?),
        2 => tensor.data_type = Some(field.int32()?),
        4 => tensor.value_count += field.repeated_f32()?.len(),
        5 | 7 | 11 => tensor.value_count += field.repeated_varint_len()?,
        6 => tensor.value_count += 1,
        8 => tensor.name = Some(field.string()?),
        9 => tensor.raw_data_len = field.bytes()?.len(),
        10 => tensor.value_count += field.repeated_f64_len()?,
        12 => tensor.doc_string = Some(field.string()?),
        13 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            tensor
              .external_data
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        16 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            tensor
              .metadata_props
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        _ => {}
      }
    }
    Ok(tensor)
  }
}

#[derive(Debug, Default)]
struct SparseTensorProto {
  values: Option<TensorProto>,
  indices: Option<TensorProto>,
  dims: Vec<i64>,
}

impl SparseTensorProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut tensor = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => tensor.values = Some(TensorProto::decode(field.bytes()?)?),
        2 => tensor.indices = Some(TensorProto::decode(field.bytes()?)?),
        3 => tensor.dims.extend(field.repeated_i64()?),
        _ => {}
      }
    }
    Ok(tensor)
  }
}

#[derive(Debug, Default)]
struct TensorAnnotationProto {
  tensor_name: String,
  parameters: Vec<(String, String)>,
}

impl TensorAnnotationProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut annotation = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => annotation.tensor_name = field.string()?,
        2 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          annotation.parameters.push((
            entry.key.unwrap_or_default(),
            entry.value.unwrap_or_default(),
          ));
        }
        _ => {}
      }
    }
    Ok(annotation)
  }
}

#[derive(Debug, Default)]
struct ValueInfoProto {
  name: String,
  r#type: Option<TypeProto>,
  doc_string: Option<String>,
  metadata_props: BTreeMap<String, String>,
}

impl ValueInfoProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut value = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => value.name = field.string()?,
        2 => value.r#type = Some(TypeProto::decode(field.bytes()?)?),
        3 => value.doc_string = Some(field.string()?),
        4 => {
          let entry = StringStringEntryProto::decode(field.bytes()?)?;
          if let Some(key) = entry.key {
            value
              .metadata_props
              .insert(key, entry.value.unwrap_or_default());
          }
        }
        _ => {}
      }
    }
    Ok(value)
  }
}

#[derive(Clone, Debug, Default)]
struct TypeProto {
  tensor: Option<TensorTypeProto>,
  sparse_tensor: Option<TensorTypeProto>,
  denotation: Option<String>,
  has_non_tensor: bool,
}

impl TypeProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut value = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => value.tensor = Some(TensorTypeProto::decode(field.bytes()?)?),
        4 | 5 | 7 | 9 => {
          field.bytes()?;
          value.has_non_tensor = true;
        }
        6 => value.denotation = Some(field.string()?),
        8 => value.sparse_tensor = Some(TensorTypeProto::decode(field.bytes()?)?),
        _ => {}
      }
    }
    Ok(value)
  }
}

#[derive(Clone, Debug, Default)]
struct TensorTypeProto {
  elem_type: Option<i32>,
  shape: Vec<ShapeDimension>,
}

impl TensorTypeProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut value = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => value.elem_type = Some(field.int32()?),
        2 => value.shape = TensorShapeProto::decode(field.bytes()?)?.dimensions,
        _ => {}
      }
    }
    Ok(value)
  }
}

#[derive(Debug, Default)]
struct TensorShapeProto {
  dimensions: Vec<ShapeDimension>,
}

impl TensorShapeProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut value = Self::default();
    while let Some(field) = reader.next_field()? {
      if field.number == 1 {
        value
          .dimensions
          .push(ShapeDimension::decode(field.bytes()?)?);
      }
    }
    Ok(value)
  }
}

#[derive(Clone, Debug)]
struct ShapeDimension {
  value: ShapeDimensionValue,
  denotation: Option<String>,
}

impl Default for ShapeDimension {
  fn default() -> Self {
    Self {
      value: ShapeDimensionValue::Unknown,
      denotation: None,
    }
  }
}

#[derive(Clone, Debug)]
enum ShapeDimensionValue {
  Known(i64),
  Symbolic(String),
  Unknown,
}

impl ShapeDimension {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut dimension = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => dimension.value = ShapeDimensionValue::Known(field.int64()?),
        2 => dimension.value = ShapeDimensionValue::Symbolic(field.string()?),
        3 => dimension.denotation = Some(field.string()?),
        _ => {}
      }
    }
    Ok(dimension)
  }
}

#[derive(Debug, Default)]
struct StringStringEntryProto {
  key: Option<String>,
  value: Option<String>,
}

impl StringStringEntryProto {
  fn decode(data: &[u8]) -> Result<Self, ModelError> {
    let mut reader = PbReader::new(data);
    let mut entry = Self::default();
    while let Some(field) = reader.next_field()? {
      match field.number {
        1 => entry.key = Some(field.string()?),
        2 => entry.value = Some(field.string()?),
        _ => {}
      }
    }
    Ok(entry)
  }
}

#[derive(Clone, Copy)]
struct Field<'a> {
  number: u32,
  value: FieldValue<'a>,
}

impl<'a> Field<'a> {
  fn int64(self) -> Result<i64, ModelError> {
    Ok(self.varint()? as i64)
  }

  fn int32(self) -> Result<i32, ModelError> {
    Ok(self.varint()? as i32)
  }

  fn string(self) -> Result<String, ModelError> {
    String::from_utf8(self.bytes()?.to_vec())
      .map_err(|error| invalid(format!("string field is not UTF-8: {error}")))
  }

  fn bytes(self) -> Result<&'a [u8], ModelError> {
    match self.value {
      FieldValue::Bytes(value) => Ok(value),
      _ => Err(invalid("expected length-delimited field")),
    }
  }

  fn float32(self) -> Result<f32, ModelError> {
    match self.value {
      FieldValue::Fixed32(value) => Ok(f32::from_bits(value)),
      FieldValue::Varint(_) | FieldValue::Fixed64(_) | FieldValue::Bytes(_) => {
        Err(invalid("expected fixed32 field"))
      }
    }
  }

  fn repeated_f32(self) -> Result<Vec<f32>, ModelError> {
    match self.value {
      FieldValue::Fixed32(value) => Ok(vec![f32::from_bits(value)]),
      FieldValue::Bytes(bytes) => {
        if bytes.len() % 4 != 0 {
          return Err(invalid("packed float field has non-4 byte length"));
        }
        Ok(
          bytes
            .chunks_exact(4)
            .map(|chunk| {
              f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect(),
        )
      }
      _ => Err(invalid("expected repeated float field")),
    }
  }

  fn repeated_i64(self) -> Result<Vec<i64>, ModelError> {
    match self.value {
      FieldValue::Varint(value) => Ok(vec![value as i64]),
      FieldValue::Bytes(bytes) => {
        let mut reader = ScalarReader::new(bytes);
        let mut values = Vec::new();
        while !reader.is_empty() {
          values.push(reader.varint()? as i64);
        }
        Ok(values)
      }
      _ => Err(invalid("expected repeated int64 field")),
    }
  }

  fn repeated_varint_len(self) -> Result<usize, ModelError> {
    match self.value {
      FieldValue::Bytes(bytes) => {
        let mut reader = ScalarReader::new(bytes);
        let mut count = 0;
        while !reader.is_empty() {
          reader.varint()?;
          count += 1;
        }
        Ok(count)
      }
      FieldValue::Varint(_) | FieldValue::Fixed32(_) | FieldValue::Fixed64(_) => Ok(1),
    }
  }

  fn repeated_f64_len(self) -> Result<usize, ModelError> {
    match self.value {
      FieldValue::Fixed64(_) => Ok(1),
      FieldValue::Bytes(bytes) => {
        if bytes.len() % 8 != 0 {
          return Err(invalid("packed double field has non-8 byte length"));
        }
        Ok(bytes.len() / 8)
      }
      _ => Err(invalid("expected repeated double field")),
    }
  }

  fn varint(self) -> Result<u64, ModelError> {
    match self.value {
      FieldValue::Varint(value) => Ok(value),
      _ => Err(invalid("expected varint field")),
    }
  }
}

#[derive(Clone, Copy)]
enum FieldValue<'a> {
  Varint(u64),
  Fixed32(u32),
  Fixed64(()),
  Bytes(&'a [u8]),
}

struct PbReader<'a> {
  data: &'a [u8],
  position: usize,
}

impl<'a> PbReader<'a> {
  fn new(data: &'a [u8]) -> Self {
    Self { data, position: 0 }
  }

  fn next_field(&mut self) -> Result<Option<Field<'a>>, ModelError> {
    if self.position == self.data.len() {
      return Ok(None);
    }

    let key = self.varint()?;
    let number = (key >> 3) as u32;
    let wire_type = (key & 0x07) as u8;
    if number == 0 {
      return Err(invalid("field number 0 is invalid"));
    }

    let value = match wire_type {
      0 => FieldValue::Varint(self.varint()?),
      1 => {
        self.take(8)?;
        FieldValue::Fixed64(())
      }
      2 => {
        let len = self.varint()? as usize;
        FieldValue::Bytes(self.take(len)?)
      }
      5 => {
        let bytes = self.take(4)?;
        FieldValue::Fixed32(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
      }
      _ => {
        return Err(invalid(format!(
          "unsupported protobuf wire type {wire_type}"
        )));
      }
    };

    Ok(Some(Field { number, value }))
  }

  fn varint(&mut self) -> Result<u64, ModelError> {
    let mut result = 0_u64;
    for shift in (0..64).step_by(7) {
      let Some(byte) = self.data.get(self.position).copied() else {
        return Err(invalid("unexpected end of protobuf varint"));
      };
      self.position += 1;
      result |= u64::from(byte & 0x7f) << shift;
      if byte & 0x80 == 0 {
        return Ok(result);
      }
    }
    Err(invalid("protobuf varint is too long"))
  }

  fn take(&mut self, len: usize) -> Result<&'a [u8], ModelError> {
    let Some(end) = self.position.checked_add(len) else {
      return Err(invalid("protobuf length overflow"));
    };
    if end > self.data.len() {
      return Err(invalid("unexpected end of protobuf field"));
    }
    let value = &self.data[self.position..end];
    self.position = end;
    Ok(value)
  }
}

struct ScalarReader<'a> {
  data: &'a [u8],
  position: usize,
}

impl<'a> ScalarReader<'a> {
  fn new(data: &'a [u8]) -> Self {
    Self { data, position: 0 }
  }

  fn is_empty(&self) -> bool {
    self.position == self.data.len()
  }

  fn varint(&mut self) -> Result<u64, ModelError> {
    let mut result = 0_u64;
    for shift in (0..64).step_by(7) {
      let Some(byte) = self.data.get(self.position).copied() else {
        return Err(invalid("unexpected end of packed varint"));
      };
      self.position += 1;
      result |= u64::from(byte & 0x7f) << shift;
      if byte & 0x80 == 0 {
        return Ok(result);
      }
    }
    Err(invalid("packed varint is too long"))
  }
}

fn invalid(message: impl Into<String>) -> ModelError {
  ModelError::InvalidData {
    format: FORMAT,
    message: message.into(),
  }
}
