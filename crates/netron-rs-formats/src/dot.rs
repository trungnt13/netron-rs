use std::collections::{BTreeMap, HashMap};

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
    TypeInfo, Value, ValueId,
};

const FORMAT: &str = "DOT";

pub struct DotFormat;

impl ModelFormat for DotFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["dot"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        let Ok(text) = std::str::from_utf8(input.data) else {
            return Confidence::None;
        };
        for line in text.lines().take(64) {
            let line = line.trim_start();
            if line.starts_with("//") || line.starts_with('#') {
                continue;
            }
            if line.starts_with("digraph")
                || line
                    .strip_prefix("strict")
                    .is_some_and(|rest| rest.trim_start().starts_with("digraph"))
            {
                return Confidence::High;
            }
        }
        Confidence::None
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        let text = std::str::from_utf8(input.data)
            .map_err(|error| invalid(format!("DOT text is not UTF-8: {error}")))?;
        let graph = Parser::new(text).parse()?;
        lower_graph(graph)
    }
}

fn lower_graph(parsed: ParsedGraph) -> Result<Model, ModelError> {
    if parsed.kind != "digraph" {
        return Err(invalid(format!(
            "graph type '{}' is not supported",
            parsed.kind
        )));
    }

    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: None,
    });
    let graph_name = parsed
        .name
        .as_deref()
        .filter(|name| !name.is_empty())
        .map(|name| model.intern(name));
    let graph_id = model.add_graph_placeholder(None, graph_name);
    let mut graph = Graph::new(graph_id, None, graph_name);

    let lowered = LowerGraph::from_parsed(parsed);
    let mut values = ValueMap::default();
    for node in lowered.nodes.iter().flatten() {
        let mut lowered_node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(&node.operator),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        lowered_node.name = Some(model.intern(&node.name.key));
        lowered_node.metadata = node.metadata.clone();

        for &edge_id in &node.inputs {
            let edge = &lowered.edges[edge_id];
            let value_id = values.value(
                &mut model,
                &mut graph,
                &edge.name.key,
                edge.initializer.as_ref(),
            );
            lowered_node.inputs.push(Some(value_id));
        }
        for &edge_id in &node.outputs {
            let edge = &lowered.edges[edge_id];
            let value_id = values.value(&mut model, &mut graph, &edge.name.key, None);
            lowered_node.outputs.push(Some(value_id));
        }

        lowered_node.attributes = node
            .attributes
            .iter()
            .map(|(name, value)| Attribute {
                name: model.intern(name),
                value: lower_attribute_value(&mut model, value),
            })
            .collect();

        let node_id = graph.add_node(lowered_node);
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

    for edge in &lowered.edges {
        if let Some(value_id) = values.ids.get(&edge.name.key).copied() {
            graph.values[value_id.index()].metadata = edge.metadata.clone();
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_attribute_value(model: &mut Model, value: &DotValue) -> AttributeValue {
    match value {
        DotValue::String(value) => AttributeValue::String(model.intern(value)),
        DotValue::Ints(values) => AttributeValue::Ints(values.clone()),
        DotValue::Floats(values) => AttributeValue::Floats(values.clone()),
    }
}

#[derive(Default)]
struct ValueMap {
    ids: HashMap<String, ValueId>,
}

impl ValueMap {
    fn value(
        &mut self,
        model: &mut Model,
        graph: &mut Graph,
        name: &str,
        initializer: Option<&Initializer>,
    ) -> ValueId {
        if let Some(initializer) = initializer {
            let (element_type, shape) = lower_tensor_type(&initializer.tensor_type);
            let tensor = Tensor::metadata_only(
                None,
                element_type.clone(),
                shape.clone(),
                TensorStorage::Absent,
            );
            let tensor_id = model.add_tensor(tensor);
            let value_id = *self.ids.entry(name.to_owned()).or_insert_with(|| {
                let string_id = model.intern(name);
                graph.add_value(Value::new(string_id))
            });
            let value = &mut graph.values[value_id.index()];
            value.initializer = Some(tensor_id);
            value.type_info = Some(TypeInfo {
                element_type: Some(element_type),
                layout: None,
                denotation: None,
                shape,
            });
            value_id
        } else if let Some(value_id) = self.ids.get(name).copied() {
            value_id
        } else {
            let string_id = model.intern(name);
            let value_id = graph.add_value(Value::new(string_id));
            self.ids.insert(name.to_owned(), value_id);
            value_id
        }
    }
}

fn lower_tensor_type(value: &str) -> (TensorElementType, Vec<Dimension>) {
    let Some(index) = value.find('[') else {
        return (TensorElementType::Other(String::new()), Vec::new());
    };
    let data_type = value[..index].rsplit('.').next().unwrap_or("").to_owned();
    let element_type = match data_type.as_str() {
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
        value => TensorElementType::Other(value.to_owned()),
    };
    let shape_text = value[index..].trim();
    let shape = parse_int_list(shape_text)
        .unwrap_or_default()
        .into_iter()
        .map(Dimension::known)
        .collect();
    (element_type, shape)
}

#[derive(Debug)]
struct ParsedGraph {
    kind: String,
    name: Option<String>,
    statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
enum Statement {
    Node {
        name: NodeName,
        attributes: BTreeMap<String, DotValue>,
        defaults: BTreeMap<String, DotValue>,
    },
    Edge {
        name: NodeName,
        to: NodeName,
        attributes: BTreeMap<String, DotValue>,
    },
    Subgraph,
}

#[derive(Debug, Clone)]
struct NodeName {
    id: String,
    key: String,
}

#[derive(Debug, Clone, PartialEq)]
enum DotValue {
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

impl DotValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn to_metadata_string(&self) -> String {
        match self {
            Self::String(value) => value.clone(),
            Self::Ints(values) => values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            Self::Floats(values) => values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

struct LowerGraph {
    nodes: Vec<Option<LowerNode>>,
    node_ids: HashMap<String, usize>,
    edges: Vec<LowerEdge>,
}

impl LowerGraph {
    fn from_parsed(parsed: ParsedGraph) -> Self {
        let mut graph = Self {
            nodes: Vec::new(),
            node_ids: HashMap::new(),
            edges: Vec::new(),
        };

        for statement in &parsed.statements {
            if let Statement::Node {
                name,
                attributes,
                defaults,
            } = statement
            {
                let mut metadata = defaults.clone();
                metadata.extend(attributes.clone());
                let mut node = LowerNode {
                    name: name.clone(),
                    operator: String::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    attributes: BTreeMap::new(),
                    metadata: metadata
                        .iter()
                        .map(|(key, value)| (key.clone(), value.to_metadata_string()))
                        .collect(),
                };
                infer_node_type(&mut node, &metadata);
                graph.set_node(name.id.clone(), node);
            }
        }

        for statement in &parsed.statements {
            if let Statement::Edge {
                name,
                to,
                attributes,
            } = statement
            {
                let to_id = graph.ensure_node(to.id.clone());
                let from_id = graph.ensure_node(name.id.clone());
                let edge_id = graph.edges.len();
                let metadata = if name.id == name.key {
                    attributes.clone()
                } else {
                    BTreeMap::new()
                };
                graph.edges.push(LowerEdge {
                    name: name.clone(),
                    to: to_id,
                    from: from_id,
                    metadata: metadata
                        .iter()
                        .map(|(key, value)| (key.clone(), value.to_metadata_string()))
                        .collect(),
                    initializer: None,
                });
                graph.nodes[to_id].as_mut().unwrap().inputs.push(edge_id);
                graph.nodes[from_id].as_mut().unwrap().outputs.push(edge_id);
            }
        }

        graph.remove_octagon_passthroughs();
        graph.fold_initializers();
        graph
    }

    fn set_node(&mut self, key: String, node: LowerNode) -> usize {
        if let Some(index) = self.node_ids.get(&key).copied() {
            self.nodes[index] = Some(node);
            index
        } else {
            let index = self.nodes.len();
            self.node_ids.insert(key, index);
            self.nodes.push(Some(node));
            index
        }
    }

    fn ensure_node(&mut self, key: String) -> usize {
        if let Some(index) = self.node_ids.get(&key).copied() {
            index
        } else {
            let name = NodeName {
                id: key.clone(),
                key: key.clone(),
            };
            let node = LowerNode {
                name: name.clone(),
                operator: key.clone(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                attributes: BTreeMap::new(),
                metadata: BTreeMap::new(),
            };
            self.set_node(key, node)
        }
    }

    fn remove_octagon_passthroughs(&mut self) {
        let allowed = ["pos", "height", "width", "shape", "label"];
        for index in 0..self.nodes.len() {
            let Some(node) = &self.nodes[index] else {
                continue;
            };
            if node.metadata.get("shape").map(String::as_str) != Some("octagon")
                || !node
                    .metadata
                    .keys()
                    .all(|key| allowed.iter().any(|allowed| allowed == key))
                || node.inputs.len() != 1
                || node.outputs.is_empty()
            {
                continue;
            }
            let input_edge = node.inputs[0];
            let from_node = self.edges[input_edge].from;
            let upstream_outputs = self.nodes[from_node]
                .as_ref()
                .map(|node| node.outputs.clone())
                .unwrap_or_default();
            if upstream_outputs.len() != 1 {
                continue;
            }
            let replacement = upstream_outputs[0];
            let output_names = node
                .outputs
                .iter()
                .map(|edge| self.edges[*edge].name.id.as_str())
                .collect::<std::collections::HashSet<_>>();
            if output_names.len() != 1 {
                continue;
            }
            let outputs = node.outputs.clone();
            for edge_id in outputs {
                let target = self.edges[edge_id].to;
                if let Some(target_node) = &mut self.nodes[target] {
                    for input in &mut target_node.inputs {
                        if *input == edge_id {
                            *input = replacement;
                        }
                    }
                }
            }
            self.nodes[index] = None;
        }
    }

    fn fold_initializers(&mut self) {
        for index in 0..self.nodes.len() {
            let Some(node) = &self.nodes[index] else {
                continue;
            };
            if !matches!(
                node.operator.as_str(),
                "get_parameter" | "buffer" | "Constant"
            ) || !node.inputs.is_empty()
                || node.outputs.len() != 1
            {
                continue;
            }
            let edge_id = node.outputs[0];
            let uses = self
                .nodes
                .iter()
                .flatten()
                .filter(|candidate| candidate.inputs.contains(&edge_id))
                .count();
            if uses != 1 {
                continue;
            }
            let tensor_type = node
                .attributes
                .get("type")
                .and_then(DotValue::as_str)
                .unwrap_or("?");
            self.edges[edge_id].initializer = Some(Initializer {
                tensor_type: tensor_type.to_owned(),
            });
            self.nodes[index] = None;
        }
    }
}

#[derive(Clone)]
struct LowerNode {
    name: NodeName,
    operator: String,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    attributes: BTreeMap<String, DotValue>,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone)]
struct LowerEdge {
    name: NodeName,
    to: usize,
    from: usize,
    metadata: BTreeMap<String, String>,
    initializer: Option<Initializer>,
}

#[derive(Clone)]
struct Initializer {
    tensor_type: String,
}

fn infer_node_type(node: &mut LowerNode, metadata: &BTreeMap<String, DotValue>) {
    if let Some(label) = metadata.get("label").and_then(DotValue::as_str) {
        if label.starts_with('{') && label.ends_with('}') {
            let lines = label[1..label.len() - 1].split('|').collect::<Vec<_>>();
            if lines.len() > 1 && node.name.id == lines[0] && lines[1].starts_with("op_code=") {
                let def = lines[1].split("\\l").collect::<Vec<_>>();
                let op_code = def[0].split('=').next_back().unwrap_or("").to_owned();
                node.operator = op_code.clone();
                if op_code == "call_module" {
                    node.operator = def.get(1).copied().unwrap_or("").to_owned();
                } else if op_code == "call_function" {
                    node.operator = lines
                        .get(2)
                        .and_then(|line| line.split("\\l").next())
                        .unwrap_or("")
                        .to_owned();
                } else if op_code.starts_with("get_parameter") {
                    node.attributes.insert(
                        "type".to_owned(),
                        DotValue::String(op_code[13..].trim().to_owned()),
                    );
                    node.operator = "get_parameter".to_owned();
                }
                if let Some(line) = lines.get(2) {
                    for attribute in line.split("\\l") {
                        let parts = attribute.split(':').collect::<Vec<_>>();
                        if parts.len() == 2 {
                            let key = parts[0].trim();
                            let value = parts[1].trim();
                            node.attributes
                                .insert(key.to_owned(), parse_dot_attribute_value(value));
                        }
                    }
                }
                node.metadata.remove("label");
            } else if lines.len() == 1 && lines[0].starts_with("buffer\\l") {
                let def = lines[0].split("\\l").collect::<Vec<_>>();
                node.operator = def[0].to_owned();
                if let Some(tensor_type) = def.get(1) {
                    node.attributes.insert(
                        "type".to_owned(),
                        DotValue::String((*tensor_type).to_owned()),
                    );
                }
                node.metadata.remove("label");
            }
        } else if let Some((name, operator)) = parse_name_type_label(label)
            && node.name.id == name
        {
            node.operator = operator;
            node.metadata.remove("label");
        }
    }

    if node.operator.is_empty() {
        let first = node.name.id.split("\\n").next().unwrap_or(&node.name.id);
        if let Some((_, operator)) = first.split_once('/') {
            if let Some(operator) = operator.split(" (op#").next() {
                node.operator = operator.to_owned();
            }
        } else if let Some(operator) = first.split(" (op#").next()
            && first.contains(" (op#")
        {
            node.operator = operator.to_owned();
        }
    }
    if node.operator.is_empty() {
        node.operator = node.name.id.clone();
    }
}

fn parse_name_type_label(label: &str) -> Option<(String, String)> {
    let rest = label.strip_prefix("name:")?.trim_start();
    let parts = rest.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 3 || parts[1] != "type:" {
        return None;
    }
    let name = parts[0];
    if !is_simple_identifier(name) {
        return None;
    }
    let operator = parts[2];
    if !is_simple_identifier(operator) {
        return None;
    }
    Some((name.to_owned(), operator.to_owned()))
}

fn is_simple_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars.next().is_some_and(|ch| ch.is_ascii_alphabetic())
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn parse_dot_attribute_value(value: &str) -> DotValue {
    if value.starts_with('(')
        && value.ends_with(')')
        && let Some(values) = parse_number_tuple(value)
    {
        return values;
    }
    DotValue::String(value.to_owned())
}

fn parse_number_tuple(value: &str) -> Option<DotValue> {
    let inner = value.strip_prefix('(')?.strip_suffix(')')?;
    let mut ints = Vec::new();
    let mut floats = Vec::new();
    let mut all_int = true;
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Ok(value) = part.parse::<i64>() {
            ints.push(value);
            floats.push(value as f32);
        } else if let Ok(value) = part.parse::<f32>() {
            all_int = false;
            floats.push(value);
        } else {
            return None;
        }
    }
    Some(if all_int {
        DotValue::Ints(ints)
    } else {
        DotValue::Floats(floats)
    })
}

fn parse_int_list(value: &str) -> Option<Vec<i64>> {
    let inner = value.strip_prefix('[')?.strip_suffix(']')?;
    let mut values = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        values.push(part.parse::<i64>().ok()?);
    }
    Some(values)
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Id(String),
    Symbol(&'static str),
    Eof,
}

struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    token: Token,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        let mut tokenizer = Tokenizer::new(text);
        let token = tokenizer.read().unwrap_or(Token::Eof);
        Self { tokenizer, token }
    }

    fn parse(&mut self) -> Result<ParsedGraph, ModelError> {
        if self.eat_id_value("strict") {
            // Graph strictness does not affect Netron's normalized model.
        }
        let kind = if self.match_id_value("graph") {
            return Err(invalid("undirected graph is not supported"));
        } else if self.match_id_value("digraph") {
            self.read_id()?
        } else {
            return Err(invalid("invalid graph type"));
        };
        let name = if matches!(self.token, Token::Id(_)) {
            Some(self.read_id()?)
        } else {
            None
        };
        let defaults = Defaults::default();
        let statements = self.parse_block(defaults, "->")?;
        Ok(ParsedGraph {
            kind,
            name,
            statements,
        })
    }

    fn parse_block(
        &mut self,
        mut defaults: Defaults,
        edgeop: &'static str,
    ) -> Result<Vec<Statement>, ModelError> {
        self.read_symbol("{")?;
        let mut list = Vec::new();
        while !self.match_symbol("}") && !matches!(self.token, Token::Eof) {
            if self.eat_id_value("subgraph") {
                if matches!(self.token, Token::Id(_)) {
                    self.read_id()?;
                }
                let _ = self.parse_block(defaults.clone(), edgeop)?;
                list.push(Statement::Subgraph);
            } else if self.match_symbol("{") {
                let statements = self.parse_block(defaults.clone(), edgeop)?;
                if self.eat_symbol(edgeop) {
                    let sources = statements
                        .into_iter()
                        .filter_map(|statement| match statement {
                            Statement::Node {
                                name, attributes, ..
                            } if attributes.is_empty() => Some(name),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    list.extend(self.parse_edges(sources, edgeop)?);
                } else {
                    list.push(Statement::Subgraph);
                }
            } else if matches!(self.token, Token::Id(_)) {
                let name = self.parse_node_id()?;
                if self.eat_symbol("=") {
                    let value = DotValue::String(self.read_id()?);
                    defaults.graph.insert(name.key, value);
                } else if self.eat_symbol(edgeop) {
                    list.extend(self.parse_edges(vec![name], edgeop)?);
                } else {
                    let attributes = self.parse_attributes()?;
                    if name.key == "node" || name.key == "edge" || name.key == "graph" {
                        let target = match name.key.as_str() {
                            "node" => &mut defaults.node,
                            "edge" => &mut defaults.edge,
                            _ => &mut defaults.graph,
                        };
                        target.extend(attributes);
                    } else {
                        list.push(Statement::Node {
                            name,
                            attributes,
                            defaults: defaults.node.clone(),
                        });
                    }
                }
            } else {
                return Err(invalid(format!("unexpected token {:?}", self.token)));
            }
            if self.match_symbol(";") || self.match_symbol(",") {
                self.advance()?;
            }
        }
        self.read_symbol("}")?;
        Ok(list)
    }

    fn parse_node_ids(&mut self) -> Result<Vec<NodeName>, ModelError> {
        let mut list = Vec::new();
        let open = self.eat_symbol("{");
        while !self.match_symbol("}") && !matches!(self.token, Token::Eof) {
            list.push(self.parse_node_id()?);
            if self.match_symbol(",") {
                self.advance()?;
            } else if self.match_symbol(";") {
                self.advance()?;
                if !open {
                    break;
                }
            } else if !open {
                break;
            }
        }
        if open {
            self.read_symbol("}")?;
        }
        Ok(list)
    }

    fn parse_node_id(&mut self) -> Result<NodeName, ModelError> {
        let id = self.read_id()?;
        let mut parts = vec![id.clone()];
        if self.eat_symbol(":") {
            parts.push(self.read_id()?);
            if self.eat_symbol(":") {
                parts.push(self.read_id()?);
            }
        }
        Ok(NodeName {
            id,
            key: parts.join(":"),
        })
    }

    fn parse_attributes(&mut self) -> Result<BTreeMap<String, DotValue>, ModelError> {
        let mut table = BTreeMap::new();
        if self.eat_symbol("[") {
            while matches!(self.token, Token::Id(_)) {
                let name = self.read_id()?;
                self.read_symbol("=")?;
                let value = DotValue::String(self.read_id()?);
                table.insert(name, value);
                if self.match_symbol(";") || self.match_symbol(",") {
                    self.advance()?;
                }
            }
            self.read_symbol("]")?;
        }
        Ok(table)
    }

    fn parse_edges(
        &mut self,
        mut sources: Vec<NodeName>,
        edgeop: &'static str,
    ) -> Result<Vec<Statement>, ModelError> {
        let mut list = Vec::new();
        loop {
            let targets = self.parse_node_ids()?;
            for name in &sources {
                for to in &targets {
                    list.push(Statement::Edge {
                        name: name.clone(),
                        to: to.clone(),
                        attributes: BTreeMap::new(),
                    });
                }
            }
            sources = targets;
            if !self.eat_symbol(edgeop) {
                break;
            }
        }
        let attributes = self.parse_attributes()?;
        for statement in &mut list {
            if let Statement::Edge {
                attributes: target, ..
            } = statement
            {
                *target = attributes.clone();
            }
        }
        Ok(list)
    }

    fn read_id(&mut self) -> Result<String, ModelError> {
        match std::mem::replace(&mut self.token, Token::Eof) {
            Token::Id(value) => {
                self.token = self.tokenizer.read()?;
                Ok(value)
            }
            token => Err(invalid(format!("expected identifier, got {token:?}"))),
        }
    }

    fn read_symbol(&mut self, value: &'static str) -> Result<(), ModelError> {
        if self.match_symbol(value) {
            self.advance()
        } else {
            Err(invalid(format!("expected '{value}', got {:?}", self.token)))
        }
    }

    fn advance(&mut self) -> Result<(), ModelError> {
        self.token = self.tokenizer.read()?;
        Ok(())
    }

    fn eat_symbol(&mut self, value: &'static str) -> bool {
        if self.match_symbol(value) {
            self.advance().is_ok()
        } else {
            false
        }
    }

    fn eat_id_value(&mut self, value: &str) -> bool {
        if self.match_id_value(value) {
            self.advance().is_ok()
        } else {
            false
        }
    }

    fn match_symbol(&self, value: &str) -> bool {
        matches!(self.token, Token::Symbol(symbol) if symbol == value)
    }

    fn match_id_value(&self, value: &str) -> bool {
        matches!(&self.token, Token::Id(id) if id == value)
    }
}

#[derive(Clone, Default)]
struct Defaults {
    graph: BTreeMap<String, DotValue>,
    node: BTreeMap<String, DotValue>,
    edge: BTreeMap<String, DotValue>,
}

struct Tokenizer<'a> {
    chars: std::str::Chars<'a>,
    peeked: Option<char>,
}

impl<'a> Tokenizer<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            chars: text.chars(),
            peeked: None,
        }
    }

    fn read(&mut self) -> Result<Token, ModelError> {
        loop {
            let Some(ch) = self.peek() else {
                return Ok(Token::Eof);
            };
            if ch.is_whitespace() {
                self.next();
                continue;
            }
            if ch == '#' || ch == '/' {
                self.skip_comment()?;
                continue;
            }
            break;
        }

        let Some(ch) = self.next() else {
            return Ok(Token::Eof);
        };
        Ok(match ch {
            '{' => Token::Symbol("{"),
            '}' => Token::Symbol("}"),
            '[' => Token::Symbol("["),
            ']' => Token::Symbol("]"),
            '=' => Token::Symbol("="),
            ':' => Token::Symbol(":"),
            ';' => Token::Symbol(";"),
            ',' => Token::Symbol(","),
            '-' => match self.next() {
                Some('>') => Token::Symbol("->"),
                Some('-') => Token::Symbol("--"),
                _ => return Err(invalid("unexpected '-'")),
            },
            '"' => Token::Id(self.quoted_string()?),
            '<' => Token::Id(self.html_string()?),
            value if is_identifier_start(value) => {
                let mut text = String::new();
                text.push(value);
                while self.peek().is_some_and(is_identifier_continue) {
                    text.push(self.next().unwrap());
                }
                Token::Id(text)
            }
            value => return Err(invalid(format!("unexpected character '{value}'"))),
        })
    }

    fn quoted_string(&mut self) -> Result<String, ModelError> {
        let mut value = String::new();
        while let Some(ch) = self.next() {
            if ch == '"' {
                return Ok(value);
            }
            value.push(ch);
        }
        Err(invalid("unterminated string"))
    }

    fn html_string(&mut self) -> Result<String, ModelError> {
        let mut value = String::from("<");
        let mut depth = 0_i32;
        loop {
            let Some(ch) = self.next() else {
                return Err(invalid("unterminated HTML string"));
            };
            value.push(ch);
            if ch == '<' {
                depth += 1;
            } else if ch == '>' {
                if depth == 0 {
                    return Ok(value);
                }
                depth -= 1;
            }
        }
    }

    fn skip_comment(&mut self) -> Result<(), ModelError> {
        if self.peek() == Some('#') {
            while self.next().is_some_and(|ch| ch != '\n') {}
            return Ok(());
        }
        if self.peek() == Some('/') {
            self.next();
            match self.peek() {
                Some('/') => {
                    while self.next().is_some_and(|ch| ch != '\n') {}
                    Ok(())
                }
                Some('*') => {
                    self.next();
                    loop {
                        let Some(ch) = self.next() else {
                            return Err(invalid("unterminated block comment"));
                        };
                        if ch == '*' && self.peek() == Some('/') {
                            self.next();
                            return Ok(());
                        }
                    }
                }
                _ => Err(invalid("invalid comment")),
            }
        } else {
            Err(invalid("invalid comment"))
        }
    }

    fn peek(&mut self) -> Option<char> {
        if self.peeked.is_none() {
            self.peeked = self.chars.next();
        }
        self.peeked
    }

    fn next(&mut self) -> Option<char> {
        if let Some(ch) = self.peeked.take() {
            Some(ch)
        } else {
            self.chars.next()
        }
    }
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '$' | '<')
}

fn is_identifier_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '$' | '.' | '*')
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
