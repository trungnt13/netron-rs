use std::collections::HashMap;

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
    TypeInfo, Value, ValueId,
};
use serde::Deserialize;
use serde_json::Value as JsonValue;

const FORMAT: &str = "Message";

pub struct MessageFormat;

impl ModelFormat for MessageFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["message", "maxviz"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        let Ok(text) = std::str::from_utf8(
            input
                .data
                .get(..input.data.len().min(128))
                .unwrap_or(input.data),
        ) else {
            return Confidence::None;
        };
        if !text.trim_start().starts_with('{') || !text.contains("\"signature\"") {
            return Confidence::None;
        }
        let Ok(data) = serde_json::from_slice::<MessageModel>(input.data) else {
            return Confidence::None;
        };
        if data.signature.starts_with("netron:") {
            Confidence::High
        } else {
            Confidence::None
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        let data: MessageModel =
            serde_json::from_slice(input.data).map_err(|error| ModelError::InvalidData {
                format: FORMAT,
                message: format!("invalid Netron message JSON: {error}"),
            })?;
        lower_model(data)
    }
}

fn lower_model(data: MessageModel) -> Result<Model, ModelError> {
    let (format_name, format_version) = parse_format(&data.format);
    let mut model = Model::new(FormatInfo {
        name: format_name,
        version: format_version,
    });
    model.metadata.producer = non_empty(data.producer);
    model.metadata.producer_version = non_empty(data.version);
    model.metadata.description = non_empty(data.description);
    for entry in data.metadata {
        if let Some(name) = entry.name {
            model
                .metadata
                .properties
                .insert(name, json_text(&entry.value));
        }
    }

    let graphs = if data.modules.is_empty() {
        data.graphs
    } else {
        data.modules
    };
    for graph_data in graphs {
        let graph_id = model.add_graph_placeholder(None, None);
        let graph = lower_graph(&mut model, graph_id, graph_data)?;
        model.replace_graph(graph_id, graph);
    }
    Ok(model)
}

fn lower_graph(
    model: &mut Model,
    graph_id: netron_rs_core::GraphId,
    graph_data: MessageGraph,
) -> Result<Graph, ModelError> {
    let mut graph = Graph::new(graph_id, None, None);
    let mut values = MessageValues::new(graph_data.values, graph_data.arguments);

    for argument in graph_data.inputs {
        for value_id in values.argument_values(model, &mut graph, &argument) {
            if graph.values[value_id.index()].initializer.is_none() {
                graph.values[value_id.index()].is_graph_input = true;
                if !graph.inputs.contains(&value_id) {
                    graph.inputs.push(value_id);
                }
            }
        }
    }
    for argument in graph_data.outputs {
        let mut visible = false;
        let mut output_ids = Vec::new();
        for value_id in values.argument_values(model, &mut graph, &argument) {
            if graph.values[value_id.index()].initializer.is_none() {
                visible = true;
            }
            output_ids.push(value_id);
        }
        if visible {
            for value_id in output_ids {
                graph.values[value_id.index()].is_graph_output = true;
                if !graph.outputs.contains(&value_id) {
                    graph.outputs.push(value_id);
                }
            }
        }
    }

    for node_data in graph_data.nodes {
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(
                    node_data
                        .op_type
                        .name
                        .as_deref()
                        .filter(|name| !name.is_empty())
                        .unwrap_or(""),
                ),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        if let Some(name) = node_data.name.filter(|name| !name.is_empty()) {
            node.name = Some(model.intern(name));
        }
        node.attributes = node_data
            .attributes
            .into_iter()
            .map(|attribute| lower_attribute(model, attribute))
            .collect();

        for argument in &node_data.inputs {
            for value_id in values.argument_values(model, &mut graph, argument) {
                node.inputs.push(Some(value_id));
            }
        }
        for argument in &node_data.outputs {
            for value_id in values.argument_values(model, &mut graph, argument) {
                node.outputs.push(Some(value_id));
            }
        }

        let node_id = graph.add_node(node);
        let node = &graph.nodes[node_id.index()];
        for value_id in node.inputs.iter().flatten().copied().collect::<Vec<_>>() {
            if !graph.values[value_id.index()].consumers.contains(&node_id) {
                graph.values[value_id.index()].consumers.push(node_id);
            }
        }
        for value_id in node.outputs.iter().flatten().copied().collect::<Vec<_>>() {
            graph.values[value_id.index()].producer = Some(node_id);
        }
    }

    Ok(graph)
}

struct MessageValues {
    definitions: HashMap<String, MessageValue>,
    ids: HashMap<String, ValueId>,
}

impl MessageValues {
    fn new(values: Vec<MessageValue>, arguments: Vec<MessageValue>) -> Self {
        let mut definitions = HashMap::new();
        for (index, value) in values.into_iter().enumerate() {
            definitions.insert(index.to_string(), value);
        }
        for value in arguments {
            if let Some(key) = value.name.as_ref().map(json_text) {
                definitions.insert(key, value);
            }
        }
        Self {
            definitions,
            ids: HashMap::new(),
        }
    }

    fn argument_values(
        &mut self,
        model: &mut Model,
        graph: &mut Graph,
        argument: &MessageArgument,
    ) -> Vec<ValueId> {
        argument
            .value
            .as_ref()
            .or(argument.arguments.as_ref())
            .map(|values| {
                values
                    .iter()
                    .map(|value| self.value(model, graph, &json_text(value)))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn value(&mut self, model: &mut Model, graph: &mut Graph, key: &str) -> ValueId {
        if let Some(value_id) = self.ids.get(key).copied() {
            return value_id;
        }
        let definition = self
            .definitions
            .get(key)
            .cloned()
            .unwrap_or_else(|| MessageValue {
                name: Some(JsonValue::String(key.to_owned())),
                value_type: None,
                initializer: None,
            });
        let name = definition.name.as_ref().map(json_text).unwrap_or_default();
        let mut value = Value::new(model.intern(name));
        if let Some(type_info) = definition
            .initializer
            .as_ref()
            .and_then(|initializer| initializer.tensor_type.as_ref())
            .or(definition.value_type.as_ref())
            .map(|tensor_type| lower_type(model, tensor_type))
        {
            value.type_info = Some(type_info.clone());
        }
        if let Some(initializer) = definition.initializer {
            let tensor_type = initializer
                .tensor_type
                .as_ref()
                .or(definition.value_type.as_ref())
                .map(|tensor_type| lower_type(model, tensor_type))
                .unwrap_or(TypeInfo {
                    element_type: Some(TensorElementType::Unknown),
                    layout: None,
                    denotation: None,
                    shape: Vec::new(),
                });
            let tensor = Tensor::metadata_only(
                None,
                tensor_type
                    .element_type
                    .clone()
                    .unwrap_or(TensorElementType::Unknown),
                tensor_type.shape.clone(),
                TensorStorage::Absent,
            );
            value.initializer = Some(model.add_tensor(tensor));
            value.type_info = Some(tensor_type);
        }
        let value_id = graph.add_value(value);
        self.ids.insert(key.to_owned(), value_id);
        value_id
    }
}

fn lower_attribute(model: &mut Model, attribute: MessageAttribute) -> Attribute {
    Attribute {
        name: model.intern(attribute.name.unwrap_or_default()),
        value: lower_attribute_value(model, attribute.attribute_type.as_deref(), &attribute.value),
    }
}

fn lower_attribute_value(
    model: &mut Model,
    attribute_type: Option<&str>,
    value: &JsonValue,
) -> AttributeValue {
    if value.is_null() {
        return AttributeValue::Null;
    }
    match attribute_type {
        Some("boolean") => value
            .as_bool()
            .map(|value| AttributeValue::Int(i64::from(value)))
            .or_else(|| json_i64(value).map(AttributeValue::Int))
            .unwrap_or_else(|| AttributeValue::String(model.intern(json_text(value)))),
        Some("int64") => json_i64(value)
            .map(AttributeValue::Int)
            .unwrap_or_else(|| AttributeValue::String(model.intern(json_text(value)))),
        Some("float32") | Some("float64") | Some("float") => value
            .as_f64()
            .map(|value| AttributeValue::Float(value as f32))
            .unwrap_or_else(|| AttributeValue::String(model.intern(json_text(value)))),
        Some("SymInt") | Some("SymInt?") | Some("Scalar") | Some("Tensor?") | Some("int64?") => {
            AttributeValue::String(model.intern(json_text(value)))
        }
        _ => match value {
            JsonValue::Bool(value) => AttributeValue::Bool(*value),
            JsonValue::Number(_) => json_i64(value).map(AttributeValue::Int).unwrap_or_else(|| {
                AttributeValue::Float(value.as_f64().unwrap_or_default() as f32)
            }),
            JsonValue::String(value) => AttributeValue::String(model.intern(value)),
            JsonValue::Array(values) => lower_array_attribute(model, values),
            JsonValue::Object(_) => AttributeValue::String(model.intern("[object Object]")),
            JsonValue::Null => AttributeValue::Null,
        },
    }
}

fn lower_array_attribute(model: &mut Model, values: &[JsonValue]) -> AttributeValue {
    if values.iter().all(|value| json_i64(value).is_some()) {
        return AttributeValue::Ints(values.iter().filter_map(json_i64).collect());
    }
    if values.iter().all(|value| value.as_f64().is_some()) {
        return AttributeValue::Floats(
            values
                .iter()
                .filter_map(|value| value.as_f64().map(|value| value as f32))
                .collect(),
        );
    }
    AttributeValue::Strings(
        values
            .iter()
            .map(|value| model.intern(json_text(value)))
            .collect(),
    )
}

fn lower_type(model: &mut Model, data: &MessageTensorType) -> TypeInfo {
    TypeInfo {
        element_type: data.data_type.as_deref().map(element_type),
        layout: None,
        denotation: None,
        shape: data
            .shape
            .as_ref()
            .map(|shape| {
                shape
                    .dimensions
                    .iter()
                    .map(|dimension| lower_dimension(model, dimension))
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn lower_dimension(model: &mut Model, value: &JsonValue) -> Dimension {
    if let Some(value) = json_i64(value) {
        Dimension::known(value)
    } else if let Some(value) = value.as_str() {
        Dimension::symbolic(model.intern(value))
    } else {
        Dimension::unknown()
    }
}

fn element_type(value: &str) -> TensorElementType {
    match value {
        "float16" => TensorElementType::Float16,
        "float32" => TensorElementType::Float32,
        "float64" => TensorElementType::Float64,
        "int8" => TensorElementType::Int8,
        "int16" => TensorElementType::Int16,
        "int32" => TensorElementType::Int32,
        "int64" => TensorElementType::Int64,
        "uint8" => TensorElementType::Uint8,
        "uint16" => TensorElementType::Uint16,
        "uint32" => TensorElementType::Uint32,
        "uint64" => TensorElementType::Uint64,
        "bool" | "boolean" => TensorElementType::Bool,
        "string" => TensorElementType::String,
        value => TensorElementType::Other(value.to_owned()),
    }
}

fn parse_format(format: &str) -> (&'static str, Option<String>) {
    let mut parts = format.split_whitespace();
    let name = parts.next().unwrap_or(FORMAT);
    let version = format
        .split_whitespace()
        .find_map(|part| {
            let part = part.strip_prefix('v').unwrap_or(part);
            if part.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
                Some(
                    part.chars()
                        .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
                        .collect::<String>()
                        .trim_end_matches('.')
                        .to_owned(),
                )
            } else {
                None
            }
        })
        .filter(|version| !version.is_empty());
    (Box::leak(name.to_owned().into_boxed_str()), version)
}

fn json_i64(value: &JsonValue) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
}

fn json_text(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => String::new(),
        JsonValue::String(value) => value.clone(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => value.to_string(),
        JsonValue::Array(_) | JsonValue::Object(_) => "[object Object]".to_owned(),
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

#[derive(Debug, Deserialize)]
struct MessageModel {
    signature: String,
    #[serde(default)]
    format: String,
    #[serde(default)]
    producer: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    metadata: Vec<MessageMetadata>,
    #[serde(default)]
    modules: Vec<MessageGraph>,
    #[serde(default)]
    graphs: Vec<MessageGraph>,
}

#[derive(Debug, Deserialize)]
struct MessageMetadata {
    name: Option<String>,
    #[serde(default)]
    value: JsonValue,
}

#[derive(Debug, Deserialize)]
struct MessageGraph {
    #[serde(default)]
    values: Vec<MessageValue>,
    #[serde(default)]
    arguments: Vec<MessageValue>,
    #[serde(default)]
    inputs: Vec<MessageArgument>,
    #[serde(default)]
    outputs: Vec<MessageArgument>,
    #[serde(default)]
    nodes: Vec<MessageNode>,
}

#[derive(Debug, Clone, Deserialize)]
struct MessageValue {
    #[serde(default)]
    name: Option<JsonValue>,
    #[serde(default, rename = "type")]
    value_type: Option<MessageTensorType>,
    #[serde(default)]
    initializer: Option<MessageTensor>,
}

#[derive(Debug, Clone, Deserialize)]
struct MessageTensor {
    #[serde(default, rename = "type")]
    tensor_type: Option<MessageTensorType>,
}

#[derive(Debug, Clone, Deserialize)]
struct MessageTensorType {
    #[serde(default, rename = "dataType")]
    data_type: Option<String>,
    #[serde(default)]
    shape: Option<MessageTensorShape>,
}

#[derive(Debug, Clone, Deserialize)]
struct MessageTensorShape {
    #[serde(default)]
    dimensions: Vec<JsonValue>,
}

#[derive(Debug, Deserialize)]
struct MessageNode {
    #[serde(default, rename = "type")]
    op_type: MessageNodeType,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    inputs: Vec<MessageArgument>,
    #[serde(default)]
    outputs: Vec<MessageArgument>,
    #[serde(default)]
    attributes: Vec<MessageAttribute>,
}

#[derive(Debug, Default, Deserialize)]
struct MessageNodeType {
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageArgument {
    #[serde(default)]
    value: Option<Vec<JsonValue>>,
    #[serde(default)]
    arguments: Option<Vec<JsonValue>>,
}

#[derive(Debug, Deserialize)]
struct MessageAttribute {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    value: JsonValue,
    #[serde(default, rename = "type")]
    attribute_type: Option<String>,
}
