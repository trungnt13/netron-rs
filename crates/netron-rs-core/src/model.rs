use std::collections::BTreeMap;

use crate::{GraphId, ModelError, NodeId, StringId, StringInterner, TensorId, ValueId};

#[derive(Debug, Clone)]
pub struct Model {
    pub format: FormatInfo,
    pub metadata: ModelMetadata,
    pub graphs: Vec<Graph>,
    pub functions: Vec<Function>,
    pub tensors: Vec<Tensor>,
    pub strings: StringInterner,
}

impl Model {
    pub fn new(format: FormatInfo) -> Self {
        Self {
            format,
            metadata: ModelMetadata::default(),
            graphs: Vec::new(),
            functions: Vec::new(),
            tensors: Vec::new(),
            strings: StringInterner::default(),
        }
    }

    pub fn intern(&mut self, value: impl AsRef<str>) -> StringId {
        self.strings.intern(value)
    }

    pub fn add_graph_placeholder(
        &mut self,
        parent: Option<GraphId>,
        name: Option<StringId>,
    ) -> GraphId {
        let id = GraphId::new(self.graphs.len());
        self.graphs.push(Graph::new(id, parent, name));
        id
    }

    pub fn replace_graph(&mut self, id: GraphId, graph: Graph) {
        assert_eq!(id.index(), graph.id.index());
        self.graphs[id.index()] = graph;
    }

    pub fn add_tensor(&mut self, mut tensor: Tensor) -> TensorId {
        let id = TensorId::new(self.tensors.len());
        tensor.id = id;
        self.tensors.push(tensor);
        id
    }

    pub fn add_function(&mut self, function: Function) {
        if let Some(existing) = self.functions.iter_mut().find(|existing| {
            existing.name == function.name
                && existing.domain == function.domain
                && existing.overload == function.overload
        }) {
            *existing = function;
            return;
        }
        self.functions.push(function);
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        for (graph_index, graph) in self.graphs.iter().enumerate() {
            if graph.id.index() != graph_index {
                return Err(ModelError::Invariant(format!(
                    "graph id {} stored at index {}",
                    graph.id.index(),
                    graph_index
                )));
            }

            if let Some(parent) = graph.parent
                && parent.index() >= self.graphs.len()
            {
                return Err(ModelError::Invariant(format!(
                    "graph {} parent {} does not resolve",
                    graph_index,
                    parent.index()
                )));
            }
            if let Some(parent) = graph.parent
                && !self.graphs[parent.index()].subgraphs.contains(&graph.id)
            {
                return Err(ModelError::Invariant(format!(
                    "graph {} parent {} does not list it as a subgraph",
                    graph.id.index(),
                    parent.index()
                )));
            }
            if let Some(name) = graph.name {
                self.validate_string_id(name, "graph name")?;
            }
            for subgraph in &graph.subgraphs {
                self.validate_graph_id(*subgraph, "graph subgraph")?;
                if self.graphs[subgraph.index()].parent != Some(graph.id) {
                    return Err(ModelError::Invariant(format!(
                        "graph {} lists subgraph {} but the subgraph parent is {:?}",
                        graph.id.index(),
                        subgraph.index(),
                        self.graphs[subgraph.index()].parent.map(GraphId::index)
                    )));
                }
            }

            for (node_index, node) in graph.nodes.iter().enumerate() {
                if node.id.index() != node_index {
                    return Err(ModelError::Invariant(format!(
                        "node id {} stored at index {} in graph {}",
                        node.id.index(),
                        node_index,
                        graph.id.index()
                    )));
                }
                if node.graph != graph.id {
                    return Err(ModelError::Invariant(format!(
                        "node {} belongs to graph {} but is stored in graph {}",
                        node.id.index(),
                        node.graph.index(),
                        graph.id.index()
                    )));
                }
                self.validate_node_strings(node)?;
                for value in node.inputs.iter().chain(node.outputs.iter()).flatten() {
                    if value.index() >= graph.values.len() {
                        return Err(ModelError::Invariant(format!(
                            "node {} references missing value {}",
                            node.id.index(),
                            value.index()
                        )));
                    }
                }
                for value in node.inputs.iter().flatten() {
                    if !graph.values[value.index()].consumers.contains(&node.id) {
                        return Err(ModelError::Invariant(format!(
                            "node {} input value {} does not list the node as a consumer",
                            node.id.index(),
                            value.index()
                        )));
                    }
                }
                for value in node.outputs.iter().flatten() {
                    if graph.values[value.index()].producer != Some(node.id) {
                        return Err(ModelError::Invariant(format!(
                            "node {} output value {} does not list the node as producer",
                            node.id.index(),
                            value.index()
                        )));
                    }
                }
            }

            for value in graph.inputs.iter().chain(graph.outputs.iter()) {
                if value.index() >= graph.values.len() {
                    return Err(ModelError::Invariant(format!(
                        "graph {} references missing boundary value {}",
                        graph.id.index(),
                        value.index()
                    )));
                }
            }
            for value in &graph.inputs {
                if !graph.values[value.index()].is_graph_input {
                    return Err(ModelError::Invariant(format!(
                        "graph {} input value {} is not marked as graph input",
                        graph.id.index(),
                        value.index()
                    )));
                }
            }
            for value in &graph.outputs {
                if !graph.values[value.index()].is_graph_output {
                    return Err(ModelError::Invariant(format!(
                        "graph {} output value {} is not marked as graph output",
                        graph.id.index(),
                        value.index()
                    )));
                }
            }

            for (value_index, value) in graph.values.iter().enumerate() {
                if value.id.index() != value_index {
                    return Err(ModelError::Invariant(format!(
                        "value id {} stored at index {} in graph {}",
                        value.id.index(),
                        value_index,
                        graph.id.index()
                    )));
                }
                self.validate_string_id(value.name, "value name")?;
                if let Some(type_info) = &value.type_info {
                    self.validate_type_info(type_info)?;
                }
                if let Some(producer) = value.producer
                    && producer.index() >= graph.nodes.len()
                {
                    return Err(ModelError::Invariant(format!(
                        "value {} producer {} does not resolve",
                        value.id.index(),
                        producer.index()
                    )));
                }
                if let Some(producer) = value.producer
                    && !graph.nodes[producer.index()]
                        .outputs
                        .iter()
                        .flatten()
                        .any(|output| *output == value.id)
                {
                    return Err(ModelError::Invariant(format!(
                        "value {} producer {} does not output it",
                        value.id.index(),
                        producer.index()
                    )));
                }
                for consumer in &value.consumers {
                    if consumer.index() >= graph.nodes.len() {
                        return Err(ModelError::Invariant(format!(
                            "value {} consumer {} does not resolve",
                            value.id.index(),
                            consumer.index()
                        )));
                    }
                    if !graph.nodes[consumer.index()]
                        .inputs
                        .iter()
                        .flatten()
                        .any(|input| *input == value.id)
                    {
                        return Err(ModelError::Invariant(format!(
                            "value {} consumer {} does not input it",
                            value.id.index(),
                            consumer.index()
                        )));
                    }
                }
                if let Some(tensor) = value.initializer
                    && tensor.index() >= self.tensors.len()
                {
                    return Err(ModelError::Invariant(format!(
                        "value {} initializer {} does not resolve",
                        value.id.index(),
                        tensor.index()
                    )));
                }
                for quantization in &value.quantization {
                    self.validate_string_id(quantization.key, "quantization key")?;
                    self.validate_string_id(quantization.value, "quantization value")?;
                }
            }
        }
        for (tensor_index, tensor) in self.tensors.iter().enumerate() {
            if tensor.id.index() != tensor_index {
                return Err(ModelError::Invariant(format!(
                    "tensor id {} stored at index {}",
                    tensor.id.index(),
                    tensor_index
                )));
            }
            if let Some(name) = tensor.name {
                self.validate_string_id(name, "tensor name")?;
            }
            for dimension in &tensor.shape {
                self.validate_dimension(dimension)?;
            }
            if let TensorStorage::Sparse { values, indices } = &tensor.storage {
                self.validate_tensor_id(*values, "sparse values tensor")?;
                self.validate_tensor_id(*indices, "sparse indices tensor")?;
            }
            if let Some(quantization) = &tensor.quantization {
                if let Some(scale) = quantization.scale {
                    self.validate_tensor_id(scale, "quantization scale tensor")?;
                }
                if let Some(zero_point) = quantization.zero_point {
                    self.validate_tensor_id(zero_point, "quantization zero point tensor")?;
                }
            }
        }
        for function in &self.functions {
            self.validate_string_id(function.name, "function name")?;
            if let Some(domain) = function.domain {
                self.validate_string_id(domain, "function domain")?;
            }
            if let Some(overload) = function.overload {
                self.validate_string_id(overload, "function overload")?;
            }
            for id in function
                .inputs
                .iter()
                .chain(&function.outputs)
                .chain(&function.attributes)
            {
                self.validate_string_id(*id, "function signature string")?;
            }
            for value in &function.values {
                self.validate_string_id(value.name, "function value name")?;
                if let Some(type_info) = &value.type_info {
                    self.validate_type_info(type_info)?;
                }
                if let Some(tensor) = value.initializer
                    && tensor.index() >= self.tensors.len()
                {
                    return Err(ModelError::Invariant(format!(
                        "function value initializer {} does not resolve",
                        tensor.index()
                    )));
                }
            }
            for node in &function.nodes {
                if let Some(name) = node.name {
                    self.validate_string_id(name, "function node name")?;
                }
                self.validate_operator(&node.operator)?;
                for id in node.inputs.iter().chain(&node.outputs).flatten() {
                    self.validate_string_id(*id, "function node value")?;
                }
                for attribute in &node.attributes {
                    self.validate_attribute(attribute)?;
                }
            }
        }
        Ok(())
    }

    fn validate_string_id(&self, id: StringId, label: &str) -> Result<(), ModelError> {
        if id.index() >= self.strings.len() {
            return Err(ModelError::Invariant(format!(
                "{label} string id {} does not resolve",
                id.index()
            )));
        }
        Ok(())
    }

    fn validate_tensor_id(&self, id: TensorId, label: &str) -> Result<(), ModelError> {
        if id.index() >= self.tensors.len() {
            return Err(ModelError::Invariant(format!(
                "{label} {} does not resolve",
                id.index()
            )));
        }
        Ok(())
    }

    fn validate_graph_id(&self, id: GraphId, label: &str) -> Result<(), ModelError> {
        if id.index() >= self.graphs.len() {
            return Err(ModelError::Invariant(format!(
                "{label} {} does not resolve",
                id.index()
            )));
        }
        Ok(())
    }

    fn validate_node_strings(&self, node: &Node) -> Result<(), ModelError> {
        if let Some(name) = node.name {
            self.validate_string_id(name, "node name")?;
        }
        self.validate_operator(&node.operator)?;
        for attribute in &node.attributes {
            self.validate_attribute(attribute)?;
        }
        Ok(())
    }

    fn validate_operator(&self, operator: &Operator) -> Result<(), ModelError> {
        if let Some(domain) = operator.domain {
            self.validate_string_id(domain, "operator domain")?;
        }
        self.validate_string_id(operator.name, "operator name")?;
        if let Some(overload) = operator.overload {
            self.validate_string_id(overload, "operator overload")?;
        }
        Ok(())
    }

    fn validate_attribute(&self, attribute: &Attribute) -> Result<(), ModelError> {
        self.validate_string_id(attribute.name, "attribute name")?;
        match &attribute.value {
            AttributeValue::Null => {}
            AttributeValue::String(value) => self.validate_string_id(*value, "attribute string")?,
            AttributeValue::Reference(value) => {
                self.validate_string_id(*value, "attribute reference")?
            }
            AttributeValue::Tensor(value) => self.validate_tensor_id(*value, "attribute tensor")?,
            AttributeValue::Graph(value) => self.validate_graph_id(*value, "attribute graph")?,
            AttributeValue::Strings(values) => {
                for value in values {
                    self.validate_string_id(*value, "attribute string")?;
                }
            }
            AttributeValue::Tensors(values) => {
                for value in values {
                    self.validate_tensor_id(*value, "attribute tensor")?;
                }
            }
            AttributeValue::Graphs(values) => {
                for value in values {
                    self.validate_graph_id(*value, "attribute graph")?;
                }
            }
            AttributeValue::Bool(_)
            | AttributeValue::Float(_)
            | AttributeValue::Int(_)
            | AttributeValue::Bytes { .. }
            | AttributeValue::Floats(_)
            | AttributeValue::Ints(_)
            | AttributeValue::Type(_)
            | AttributeValue::TypeList(_)
            | AttributeValue::Unsupported(_) => {}
        }
        Ok(())
    }

    fn validate_type_info(&self, type_info: &TypeInfo) -> Result<(), ModelError> {
        if let Some(layout) = type_info.layout {
            self.validate_string_id(layout, "type layout")?;
        }
        if let Some(denotation) = type_info.denotation {
            self.validate_string_id(denotation, "type denotation")?;
        }
        for dimension in &type_info.shape {
            self.validate_dimension(dimension)?;
        }
        Ok(())
    }

    fn validate_dimension(&self, dimension: &Dimension) -> Result<(), ModelError> {
        if let DimensionValue::Symbolic(value) = dimension.value {
            self.validate_string_id(value, "symbolic dimension")?;
        }
        if let Some(denotation) = dimension.denotation {
            self.validate_string_id(denotation, "dimension denotation")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FormatInfo {
    pub name: &'static str,
    pub version: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct ModelMetadata {
    pub producer: Option<String>,
    pub producer_version: Option<String>,
    pub domain: Option<String>,
    pub model_version: Option<i64>,
    pub description: Option<String>,
    pub opsets: Vec<OperatorSet>,
    pub properties: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct OperatorSet {
    pub domain: Option<String>,
    pub version: i64,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub id: GraphId,
    pub parent: Option<GraphId>,
    pub name: Option<StringId>,
    pub description: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub nodes: Vec<Node>,
    pub values: Vec<Value>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub subgraphs: Vec<GraphId>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: StringId,
    pub domain: Option<StringId>,
    pub overload: Option<StringId>,
    pub description: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub opsets: Vec<OperatorSet>,
    pub inputs: Vec<StringId>,
    pub outputs: Vec<StringId>,
    pub attributes: Vec<StringId>,
    pub values: Vec<FunctionValue>,
    pub nodes: Vec<FunctionNode>,
}

#[derive(Debug, Clone)]
pub struct FunctionValue {
    pub name: StringId,
    pub type_info: Option<TypeInfo>,
    pub initializer: Option<TensorId>,
}

#[derive(Debug, Clone)]
pub struct FunctionNode {
    pub name: Option<StringId>,
    pub description: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub operator: Operator,
    pub inputs: Vec<Option<StringId>>,
    pub outputs: Vec<Option<StringId>>,
    pub attributes: Vec<Attribute>,
}

impl Graph {
    pub fn new(id: GraphId, parent: Option<GraphId>, name: Option<StringId>) -> Self {
        Self {
            id,
            parent,
            name,
            description: None,
            metadata: BTreeMap::new(),
            nodes: Vec::new(),
            values: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            subgraphs: Vec::new(),
        }
    }

    pub fn add_value(&mut self, mut value: Value) -> ValueId {
        let id = ValueId::new(self.values.len());
        value.id = id;
        self.values.push(value);
        id
    }

    pub fn add_node(&mut self, mut node: Node) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        node.id = id;
        self.nodes.push(node);
        id
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub graph: GraphId,
    pub name: Option<StringId>,
    pub description: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub operator: Operator,
    pub inputs: Vec<Option<ValueId>>,
    pub outputs: Vec<Option<ValueId>>,
    pub attributes: Vec<Attribute>,
}

impl Node {
    pub fn new(graph: GraphId, operator: Operator) -> Self {
        Self {
            id: NodeId::new(0),
            graph,
            name: None,
            description: None,
            metadata: BTreeMap::new(),
            operator,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Operator {
    pub domain: Option<StringId>,
    pub name: StringId,
    pub overload: Option<StringId>,
    pub version: Option<i64>,
    pub origin: &'static str,
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: StringId,
    pub value: AttributeValue,
}

#[derive(Debug, Clone)]
pub enum AttributeValue {
    Null,
    Bool(bool),
    Float(f32),
    Int(i64),
    String(StringId),
    Reference(StringId),
    Bytes { byte_len: usize },
    Tensor(TensorId),
    Graph(GraphId),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<StringId>),
    Tensors(Vec<TensorId>),
    Graphs(Vec<GraphId>),
    Type(String),
    TypeList(Vec<String>),
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct Value {
    pub id: ValueId,
    pub name: StringId,
    pub description: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub type_info: Option<TypeInfo>,
    pub producer: Option<NodeId>,
    pub consumers: Vec<NodeId>,
    pub initializer: Option<TensorId>,
    pub quantization: Vec<QuantizationAnnotation>,
    pub is_graph_input: bool,
    pub is_graph_output: bool,
}

impl Value {
    pub fn new(name: StringId) -> Self {
        Self {
            id: ValueId::new(0),
            name,
            description: None,
            metadata: BTreeMap::new(),
            type_info: None,
            producer: None,
            consumers: Vec::new(),
            initializer: None,
            quantization: Vec::new(),
            is_graph_input: false,
            is_graph_output: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub element_type: Option<TensorElementType>,
    pub layout: Option<StringId>,
    pub denotation: Option<StringId>,
    pub shape: Vec<Dimension>,
}

#[derive(Debug, Clone)]
pub struct Dimension {
    pub value: DimensionValue,
    pub denotation: Option<StringId>,
}

impl Dimension {
    pub fn known(value: i64) -> Self {
        Self {
            value: DimensionValue::Known(value),
            denotation: None,
        }
    }

    pub fn symbolic(value: StringId) -> Self {
        Self {
            value: DimensionValue::Symbolic(value),
            denotation: None,
        }
    }

    pub fn unknown() -> Self {
        Self {
            value: DimensionValue::Unknown,
            denotation: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DimensionValue {
    Known(i64),
    Symbolic(StringId),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub name: Option<StringId>,
    pub description: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub element_type: TensorElementType,
    pub shape: Vec<Dimension>,
    pub storage: TensorStorage,
    pub quantization: Option<Quantization>,
}

impl Tensor {
    pub fn metadata_only(
        name: Option<StringId>,
        element_type: TensorElementType,
        shape: Vec<Dimension>,
        storage: TensorStorage,
    ) -> Self {
        Self {
            id: TensorId::new(0),
            name,
            description: None,
            metadata: BTreeMap::new(),
            element_type,
            shape,
            storage,
            quantization: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorElementType {
    Unknown,
    Float16,
    Float32,
    Float64,
    Float8e4m3fn,
    Float8e4m3fnuz,
    Float8e5m2,
    Float8e5m2fnuz,
    Float8e8m0,
    Float4e2m1,
    BFloat16,
    Int2,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint4,
    Uint2,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    String,
    Complex64,
    Complex128,
    Other(String),
}

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Absent,
    InlineBytes { byte_len: usize },
    ElementList { len: usize },
    External { entries: BTreeMap<String, String> },
    Sparse { values: TensorId, indices: TensorId },
}

#[derive(Debug, Clone)]
pub struct Quantization {
    pub scale: Option<TensorId>,
    pub zero_point: Option<TensorId>,
}

#[derive(Debug, Clone)]
pub struct QuantizationAnnotation {
    pub key: StringId,
    pub value: StringId,
}
