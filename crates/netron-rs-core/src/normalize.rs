use std::collections::BTreeMap;

use serde::{Serialize, Serializer, ser::SerializeMap};

use crate::{
  AttributeValue, Dimension, DimensionValue, Function, FunctionNode, FunctionValue, Graph, GraphId,
  Model, Node, Tensor, TensorElementType, TensorStorage, TypeInfo, Value,
};

pub trait ToNormalizedJson {
  fn to_normalized(&self) -> NormalizedModel<'_>;

  fn to_normalized_json(&self) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&self.to_normalized())
  }
}

impl ToNormalizedJson for Model {
  fn to_normalized(&self) -> NormalizedModel<'_> {
    NormalizedModel {
      format: NormalizedFormat {
        name: self.format.name,
        version: self.format.version.as_deref(),
      },
      metadata: NormalizedMetadata {
        producer: self.metadata.producer.as_deref(),
        producer_version: self.metadata.producer_version.as_deref(),
        domain: self.metadata.domain.as_deref(),
        model_version: self.metadata.model_version,
        description: self.metadata.description.as_deref(),
        opsets: self
          .metadata
          .opsets
          .iter()
          .map(|opset| NormalizedOperatorSet {
            domain: opset.domain.as_deref(),
            version: opset.version,
          })
          .collect(),
        properties: &self.metadata.properties,
      },
      graphs: self
        .graphs
        .iter()
        .map(|graph| normalize_graph(self, graph))
        .collect(),
      functions: self
        .functions
        .iter()
        .map(|function| normalize_function(self, function))
        .collect(),
      tensors: self
        .tensors
        .iter()
        .map(|tensor| normalize_tensor(self, tensor))
        .collect(),
    }
  }
}

#[derive(Debug, Serialize)]
pub struct NormalizedModel<'a> {
  pub format: NormalizedFormat<'a>,
  pub metadata: NormalizedMetadata<'a>,
  pub graphs: Vec<NormalizedGraph<'a>>,
  pub functions: Vec<NormalizedFunction<'a>>,
  pub tensors: Vec<NormalizedTensor<'a>>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedFormat<'a> {
  pub name: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub version: Option<&'a str>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedMetadata<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub producer: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub producer_version: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub domain: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub model_version: Option<i64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub opsets: Vec<NormalizedOperatorSet<'a>>,
  pub properties: &'a BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedOperatorSet<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub domain: Option<&'a str>,
  pub version: i64,
}

#[derive(Debug, Serialize)]
pub struct NormalizedGraph<'a> {
  pub id: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub parent: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub name: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub metadata: &'a BTreeMap<String, String>,
  pub inputs: Vec<&'a str>,
  pub outputs: Vec<&'a str>,
  pub values: Vec<NormalizedValue<'a>>,
  pub nodes: Vec<NormalizedNode<'a>>,
  pub subgraphs: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedValue<'a> {
  pub id: usize,
  pub name: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub metadata: &'a BTreeMap<String, String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub r#type: Option<NormalizedType<'a>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub producer: Option<usize>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub consumers: Vec<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub initializer: Option<usize>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub quantization: Vec<NormalizedQuantizationAnnotation<'a>>,
  #[serde(default, skip_serializing_if = "is_false")]
  pub graph_input: bool,
  #[serde(default, skip_serializing_if = "is_false")]
  pub graph_output: bool,
}

#[derive(Debug, Serialize)]
pub struct NormalizedNode<'a> {
  pub id: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub name: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub metadata: &'a BTreeMap<String, String>,
  pub operator: NormalizedOperator<'a>,
  pub inputs: Vec<Option<&'a str>>,
  pub outputs: Vec<Option<&'a str>>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub attributes: Vec<NormalizedAttribute<'a>>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedFunction<'a> {
  pub name: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub domain: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub overload: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub metadata: &'a BTreeMap<String, String>,
  pub opsets: Vec<NormalizedOperatorSet<'a>>,
  pub inputs: Vec<&'a str>,
  pub outputs: Vec<&'a str>,
  pub attributes: Vec<&'a str>,
  pub values: Vec<NormalizedFunctionValue<'a>>,
  pub nodes: Vec<NormalizedFunctionNode<'a>>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedFunctionValue<'a> {
  pub name: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub r#type: Option<NormalizedType<'a>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub initializer: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedFunctionNode<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub name: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub metadata: &'a BTreeMap<String, String>,
  pub operator: NormalizedOperator<'a>,
  pub inputs: Vec<Option<&'a str>>,
  pub outputs: Vec<Option<&'a str>>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub attributes: Vec<NormalizedAttribute<'a>>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedOperator<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub domain: Option<&'a str>,
  pub name: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub overload: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub version: Option<i64>,
  pub origin: &'a str,
}

#[derive(Debug, Serialize)]
pub struct NormalizedAttribute<'a> {
  pub name: &'a str,
  pub value: NormalizedAttributeValue<'a>,
}

#[derive(Debug)]
pub enum NormalizedAttributeValue<'a> {
  Null,
  Bool(bool),
  Float(f32),
  Int(i64),
  String(&'a str),
  Reference(&'a str),
  Bytes { byte_len: usize },
  Tensor(usize),
  Graph(usize),
  Floats(Vec<f32>),
  Ints(Vec<i64>),
  Strings(Vec<&'a str>),
  Tensors(Vec<usize>),
  Graphs(Vec<usize>),
  Type(&'a str),
  TypeList(Vec<&'a str>),
  Unsupported(&'a str),
}

impl Serialize for NormalizedAttributeValue<'_> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut map = serializer.serialize_map(None)?;
    match self {
      Self::Null => {
        map.serialize_entry("kind", "null")?;
      }
      Self::Bool(value) => {
        map.serialize_entry("kind", "bool")?;
        map.serialize_entry("value", value)?;
      }
      Self::Float(value) => {
        map.serialize_entry("kind", "float")?;
        map.serialize_entry("value", value)?;
      }
      Self::Int(value) => {
        map.serialize_entry("kind", "int")?;
        map.serialize_entry("value", &ExactJsonI64(*value))?;
      }
      Self::String(value) => {
        map.serialize_entry("kind", "string")?;
        map.serialize_entry("value", value)?;
      }
      Self::Reference(value) => {
        map.serialize_entry("kind", "reference")?;
        map.serialize_entry("value", value)?;
      }
      Self::Bytes { byte_len } => {
        map.serialize_entry("kind", "bytes")?;
        map.serialize_entry("byte_len", byte_len)?;
      }
      Self::Tensor(value) => {
        map.serialize_entry("kind", "tensor")?;
        map.serialize_entry("value", value)?;
      }
      Self::Graph(value) => {
        map.serialize_entry("kind", "graph")?;
        map.serialize_entry("value", value)?;
      }
      Self::Floats(value) => {
        map.serialize_entry("kind", "floats")?;
        map.serialize_entry("value", value)?;
      }
      Self::Ints(value) => {
        let value = value.iter().copied().map(ExactJsonI64).collect::<Vec<_>>();
        map.serialize_entry("kind", "ints")?;
        map.serialize_entry("value", &value)?;
      }
      Self::Strings(value) => {
        map.serialize_entry("kind", "strings")?;
        map.serialize_entry("value", value)?;
      }
      Self::Tensors(value) => {
        map.serialize_entry("kind", "tensors")?;
        map.serialize_entry("value", value)?;
      }
      Self::Graphs(value) => {
        map.serialize_entry("kind", "graphs")?;
        map.serialize_entry("value", value)?;
      }
      Self::Type(value) => {
        map.serialize_entry("kind", "type")?;
        map.serialize_entry("value", value)?;
      }
      Self::TypeList(value) => {
        map.serialize_entry("kind", "type[]")?;
        map.serialize_entry("value", value)?;
      }
      Self::Unsupported(value) => {
        map.serialize_entry("kind", "unsupported")?;
        map.serialize_entry("value", value)?;
      }
    }
    map.end()
  }
}

struct ExactJsonI64(i64);

impl Serialize for ExactJsonI64 {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    const MAX_SAFE_JSON_INTEGER: i64 = 9_007_199_254_740_991;
    if (-MAX_SAFE_JSON_INTEGER..=MAX_SAFE_JSON_INTEGER).contains(&self.0) {
      serializer.serialize_i64(self.0)
    } else {
      serializer.serialize_str(&self.0.to_string())
    }
  }
}

#[derive(Debug, Serialize)]
pub struct NormalizedQuantizationAnnotation<'a> {
  pub key: &'a str,
  pub value: &'a str,
}

#[derive(Debug, Serialize)]
pub struct NormalizedType<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub element_type: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub layout: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub denotation: Option<&'a str>,
  pub shape: Vec<NormalizedDimension<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalizedDimension<'a> {
  Known {
    value: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    denotation: Option<&'a str>,
  },
  Symbolic {
    value: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    denotation: Option<&'a str>,
  },
  Unknown {
    #[serde(skip_serializing_if = "Option::is_none")]
    denotation: Option<&'a str>,
  },
}

#[derive(Debug, Serialize)]
pub struct NormalizedTensor<'a> {
  pub id: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub name: Option<&'a str>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub description: Option<&'a str>,
  pub metadata: &'a BTreeMap<String, String>,
  pub element_type: &'a str,
  pub shape: Vec<NormalizedDimension<'a>>,
  pub storage: NormalizedTensorStorage<'a>,
  #[serde(default, skip_serializing_if = "is_false")]
  pub quantized: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalizedTensorStorage<'a> {
  Absent,
  InlineBytes {
    byte_len: usize,
  },
  ElementList {
    len: usize,
  },
  External {
    entries: &'a BTreeMap<String, String>,
  },
  Sparse {
    values: usize,
    indices: usize,
  },
}

fn normalize_graph<'a>(model: &'a Model, graph: &'a Graph) -> NormalizedGraph<'a> {
  NormalizedGraph {
    id: graph.id.index(),
    parent: graph.parent.map(GraphId::index),
    name: graph.name.map(|id| model.strings.get(id)),
    description: graph.description.as_deref(),
    metadata: &graph.metadata,
    inputs: graph
      .inputs
      .iter()
      .map(|id| model.strings.get(graph.values[id.index()].name))
      .collect(),
    outputs: graph
      .outputs
      .iter()
      .map(|id| model.strings.get(graph.values[id.index()].name))
      .collect(),
    values: graph
      .values
      .iter()
      .map(|value| normalize_value(model, value))
      .collect(),
    nodes: graph
      .nodes
      .iter()
      .map(|node| normalize_node(model, graph, node))
      .collect(),
    subgraphs: graph.subgraphs.iter().map(|id| id.index()).collect(),
  }
}

fn normalize_value<'a>(model: &'a Model, value: &'a Value) -> NormalizedValue<'a> {
  NormalizedValue {
    id: value.id.index(),
    name: model.strings.get(value.name),
    description: value.description.as_deref(),
    metadata: &value.metadata,
    r#type: value
      .type_info
      .as_ref()
      .map(|type_info| normalize_type(model, type_info)),
    producer: value.producer.map(|id| id.index()),
    consumers: value.consumers.iter().map(|id| id.index()).collect(),
    initializer: value.initializer.map(|id| id.index()),
    quantization: value
      .quantization
      .iter()
      .map(|entry| NormalizedQuantizationAnnotation {
        key: model.strings.get(entry.key),
        value: model.strings.get(entry.value),
      })
      .collect(),
    graph_input: value.is_graph_input,
    graph_output: value.is_graph_output,
  }
}

fn normalize_node<'a>(model: &'a Model, graph: &'a Graph, node: &'a Node) -> NormalizedNode<'a> {
  NormalizedNode {
    id: node.id.index(),
    name: node.name.map(|id| model.strings.get(id)),
    description: node.description.as_deref(),
    metadata: &node.metadata,
    operator: NormalizedOperator {
      domain: node.operator.domain.map(|id| model.strings.get(id)),
      name: model.strings.get(node.operator.name),
      overload: node.operator.overload.map(|id| model.strings.get(id)),
      version: node.operator.version,
      origin: node.operator.origin,
    },
    inputs: node
      .inputs
      .iter()
      .map(|id| id.map(|id| model.strings.get(graph.values[id.index()].name)))
      .collect(),
    outputs: node
      .outputs
      .iter()
      .map(|id| id.map(|id| model.strings.get(graph.values[id.index()].name)))
      .collect(),
    attributes: node
      .attributes
      .iter()
      .map(|attribute| NormalizedAttribute {
        name: model.strings.get(attribute.name),
        value: normalize_attribute_value(model, &attribute.value),
      })
      .collect(),
  }
}

fn normalize_function<'a>(model: &'a Model, function: &'a Function) -> NormalizedFunction<'a> {
  NormalizedFunction {
    name: model.strings.get(function.name),
    domain: function.domain.map(|id| model.strings.get(id)),
    overload: function.overload.map(|id| model.strings.get(id)),
    description: function.description.as_deref(),
    metadata: &function.metadata,
    opsets: function
      .opsets
      .iter()
      .map(|opset| NormalizedOperatorSet {
        domain: opset.domain.as_deref(),
        version: opset.version,
      })
      .collect(),
    inputs: function
      .inputs
      .iter()
      .map(|id| model.strings.get(*id))
      .collect(),
    outputs: function
      .outputs
      .iter()
      .map(|id| model.strings.get(*id))
      .collect(),
    attributes: function
      .attributes
      .iter()
      .map(|id| model.strings.get(*id))
      .collect(),
    values: function
      .values
      .iter()
      .map(|value| normalize_function_value(model, value))
      .collect(),
    nodes: function
      .nodes
      .iter()
      .map(|node| normalize_function_node(model, node))
      .collect(),
  }
}

fn normalize_function_value<'a>(
  model: &'a Model,
  value: &'a FunctionValue,
) -> NormalizedFunctionValue<'a> {
  NormalizedFunctionValue {
    name: model.strings.get(value.name),
    r#type: value
      .type_info
      .as_ref()
      .map(|type_info| normalize_type(model, type_info)),
    initializer: value.initializer.map(|id| id.index()),
  }
}

fn normalize_function_node<'a>(
  model: &'a Model,
  node: &'a FunctionNode,
) -> NormalizedFunctionNode<'a> {
  NormalizedFunctionNode {
    name: node.name.map(|id| model.strings.get(id)),
    description: node.description.as_deref(),
    metadata: &node.metadata,
    operator: NormalizedOperator {
      domain: node.operator.domain.map(|id| model.strings.get(id)),
      name: model.strings.get(node.operator.name),
      overload: node.operator.overload.map(|id| model.strings.get(id)),
      version: node.operator.version,
      origin: node.operator.origin,
    },
    inputs: node
      .inputs
      .iter()
      .map(|id| id.map(|id| model.strings.get(id)))
      .collect(),
    outputs: node
      .outputs
      .iter()
      .map(|id| id.map(|id| model.strings.get(id)))
      .collect(),
    attributes: node
      .attributes
      .iter()
      .map(|attribute| NormalizedAttribute {
        name: model.strings.get(attribute.name),
        value: normalize_attribute_value(model, &attribute.value),
      })
      .collect(),
  }
}

fn normalize_attribute_value<'a>(
  model: &'a Model,
  value: &'a AttributeValue,
) -> NormalizedAttributeValue<'a> {
  match value {
    AttributeValue::Null => NormalizedAttributeValue::Null,
    AttributeValue::Bool(value) => NormalizedAttributeValue::Bool(*value),
    AttributeValue::Float(value) => NormalizedAttributeValue::Float(*value),
    AttributeValue::Int(value) => NormalizedAttributeValue::Int(*value),
    AttributeValue::String(value) => NormalizedAttributeValue::String(model.strings.get(*value)),
    AttributeValue::Reference(value) => {
      NormalizedAttributeValue::Reference(model.strings.get(*value))
    }
    AttributeValue::Bytes { byte_len } => NormalizedAttributeValue::Bytes {
      byte_len: *byte_len,
    },
    AttributeValue::Tensor(value) => NormalizedAttributeValue::Tensor(value.index()),
    AttributeValue::Graph(value) => NormalizedAttributeValue::Graph(value.index()),
    AttributeValue::Floats(value) => NormalizedAttributeValue::Floats(value.clone()),
    AttributeValue::Ints(value) => NormalizedAttributeValue::Ints(value.clone()),
    AttributeValue::Strings(value) => {
      NormalizedAttributeValue::Strings(value.iter().map(|id| model.strings.get(*id)).collect())
    }
    AttributeValue::Tensors(value) => {
      NormalizedAttributeValue::Tensors(value.iter().map(|id| id.index()).collect())
    }
    AttributeValue::Graphs(value) => {
      NormalizedAttributeValue::Graphs(value.iter().map(|id| id.index()).collect())
    }
    AttributeValue::Type(value) => NormalizedAttributeValue::Type(value),
    AttributeValue::TypeList(value) => {
      NormalizedAttributeValue::TypeList(value.iter().map(String::as_str).collect())
    }
    AttributeValue::Unsupported(value) => NormalizedAttributeValue::Unsupported(value),
  }
}

fn normalize_tensor<'a>(model: &'a Model, tensor: &'a Tensor) -> NormalizedTensor<'a> {
  NormalizedTensor {
    id: tensor.id.index(),
    name: tensor.name.map(|id| model.strings.get(id)),
    description: tensor.description.as_deref(),
    metadata: &tensor.metadata,
    element_type: element_type_name(&tensor.element_type),
    shape: tensor
      .shape
      .iter()
      .map(|dimension| normalize_dimension(model, dimension))
      .collect(),
    storage: match &tensor.storage {
      TensorStorage::Absent => NormalizedTensorStorage::Absent,
      TensorStorage::InlineBytes { byte_len } => NormalizedTensorStorage::InlineBytes {
        byte_len: *byte_len,
      },
      TensorStorage::ElementList { len } => NormalizedTensorStorage::ElementList { len: *len },
      TensorStorage::External { entries } => NormalizedTensorStorage::External { entries },
      TensorStorage::Sparse { values, indices } => NormalizedTensorStorage::Sparse {
        values: values.index(),
        indices: indices.index(),
      },
    },
    quantized: tensor.quantization.is_some(),
  }
}

fn normalize_type<'a>(model: &'a Model, value: &'a TypeInfo) -> NormalizedType<'a> {
  NormalizedType {
    element_type: value.element_type.as_ref().map(element_type_name),
    layout: value.layout.map(|id| model.strings.get(id)),
    denotation: value.denotation.map(|id| model.strings.get(id)),
    shape: value
      .shape
      .iter()
      .map(|dimension| normalize_dimension(model, dimension))
      .collect(),
  }
}

fn normalize_dimension<'a>(model: &'a Model, dimension: &'a Dimension) -> NormalizedDimension<'a> {
  let denotation = dimension.denotation.map(|id| model.strings.get(id));
  match dimension.value {
    DimensionValue::Known(value) => NormalizedDimension::Known { value, denotation },
    DimensionValue::Symbolic(value) => NormalizedDimension::Symbolic {
      value: model.strings.get(value),
      denotation,
    },
    DimensionValue::Unknown => NormalizedDimension::Unknown { denotation },
  }
}

fn element_type_name(value: &TensorElementType) -> &str {
  match value {
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
    TensorElementType::Uint4 => "uint4",
    TensorElementType::Uint2 => "uint2",
    TensorElementType::Uint8 => "uint8",
    TensorElementType::Uint16 => "uint16",
    TensorElementType::Uint32 => "uint32",
    TensorElementType::Uint64 => "uint64",
    TensorElementType::Bool => "boolean",
    TensorElementType::String => "string",
    TensorElementType::Complex64 => "complex<float32>",
    TensorElementType::Complex128 => "complex<float64>",
    TensorElementType::Other(value) => value,
  }
}

fn is_false(value: &bool) -> bool {
  !*value
}
