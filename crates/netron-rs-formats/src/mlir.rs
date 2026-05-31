use std::collections::{BTreeMap, HashMap, HashSet};

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Function,
    FunctionNode, FunctionValue, Graph, Model, ModelError, ModelFormat, ModelInput, Node, Operator,
    Tensor, TensorElementType, TensorStorage, TypeInfo, Value, ValueId,
};

use crate::validate_external_path;

const FORMAT: &str = "MLIR";
const INTERNAL_LOCATION_REF: &str = "__mlir_location_ref";

pub struct MlirFormat;

impl ModelFormat for MlirFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["mlir"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        let name = input
            .path
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !name.ends_with(".mlir") {
            return Confidence::None;
        }
        let Ok(text) = std::str::from_utf8(
            input
                .data
                .get(..input.data.len().min(65536))
                .unwrap_or(input.data),
        ) else {
            return Confidence::None;
        };
        if text.lines().take(256).any(is_mlir_line) {
            Confidence::High
        } else {
            Confidence::Low
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        let text = std::str::from_utf8(input.data)
            .map_err(|error| invalid(format!("MLIR text is not UTF-8: {error}")))?;
        let parsed = parse_text(text);
        validate_external_resource_paths(input.path, &parsed, input.allow_unsafe_paths)?;
        lower_model(parsed)
    }
}

fn is_mlir_line(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("module ")
        || line.starts_with("\"builtin.module\"")
        || line.starts_with("builtin.module ")
        || line.starts_with("vm.module ")
        || line.starts_with("spv.module ")
        || line.starts_with("spirv.module ")
        || line.starts_with("func ")
        || line.starts_with("func.func ")
        || line.starts_with('#') && line.contains('=')
        || line.split_whitespace().next().is_some_and(|token| {
            normalized_operator_token_from_token(token).is_some_and(is_function_start_token)
        })
        || line.contains("tensor<")
        || line.contains("memref<")
        || line.contains("stablehlo.")
        || line.contains("tosa.")
        || line.contains("torch.")
        || line.contains("onnx.")
        || line.contains("affine.")
        || line.contains("arith.")
}

fn normalized_operator_token_from_token(token: &str) -> Option<&str> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    let token = if let Some(rest) = token.strip_prefix('"') {
        rest.find('"')
            .and_then(|end| token.get(1..1 + end))
            .unwrap_or_default()
    } else {
        token
    };
    token
        .split_once('(')
        .map(|(token, _)| token)
        .or(Some(token))
}

fn is_function_start_token(token: &str) -> bool {
    token == "func"
        || token == "builtin.func"
        || token.ends_with(".func")
        || token
            .rsplit_once(".func_v")
            .is_some_and(|(_, version)| version.chars().all(|ch| ch.is_ascii_digit()))
}

fn lower_model(parsed: ParsedModel) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: None,
    });
    model.metadata.properties = parsed.metadata;

    for module in post_order_modules(parsed.modules) {
        if module.nodes.is_empty()
            && (module.name.is_empty() || module.synthetic)
            && (module.metadata.is_empty() || module.function_count == 0)
        {
            continue;
        }
        let name = (!module.name.is_empty()).then(|| model.intern(&module.name));
        let graph_id = model.add_graph_placeholder(None, name);
        let mut graph = Graph::new(graph_id, None, name);
        graph.metadata = module.metadata;
        let mut values = HashMap::<String, ValueId>::new();
        for node in module.nodes {
            lower_graph_node(&mut model, &mut graph, &mut values, node);
        }
        model.replace_graph(graph_id, graph);
    }

    for function in parsed.functions {
        let mut lowered = Function {
            name: model.intern(&function.name),
            domain: None,
            overload: None,
            description: None,
            metadata: function.metadata,
            opsets: Vec::new(),
            inputs: function
                .inputs
                .iter()
                .map(|value| model.intern(&value.name))
                .collect(),
            outputs: function
                .outputs
                .iter()
                .map(|value| model.intern(&value.name))
                .collect(),
            attributes: Vec::new(),
            values: Vec::new(),
            nodes: Vec::new(),
        };
        let mut values = FunctionValues::default();
        for value in function
            .inputs
            .iter()
            .chain(&function.outputs)
            .chain(&function.values)
        {
            values.ensure(&mut model, &mut lowered, value);
        }
        for node in function.nodes {
            for input in &node.inputs {
                if !input.is_empty() {
                    values.ensure_name(&mut model, &mut lowered, input);
                }
            }
            for output in &node.outputs {
                values.ensure_name(&mut model, &mut lowered, output);
            }
            let lowered_node = FunctionNode {
                name: None,
                description: None,
                metadata: node.metadata,
                operator: Operator {
                    domain: None,
                    name: model.intern(&node.operator),
                    overload: None,
                    version: None,
                    origin: FORMAT,
                },
                inputs: node
                    .inputs
                    .iter()
                    .map(|value| (!value.is_empty()).then(|| model.intern(value)))
                    .collect(),
                outputs: node
                    .outputs
                    .iter()
                    .map(|value| Some(model.intern(value)))
                    .collect(),
                attributes: node
                    .attributes
                    .into_iter()
                    .map(|attribute| lower_attribute(&mut model, attribute))
                    .collect(),
            };
            lowered.nodes.push(lowered_node);
        }
        for initializer in function.initializers {
            let type_info = parse_type_info(&mut model, initializer.type_text.as_deref());
            let element_type = type_info
                .as_ref()
                .and_then(|info| info.element_type.clone())
                .unwrap_or(TensorElementType::Unknown);
            let shape = type_info
                .as_ref()
                .map(|info| info.shape.clone())
                .unwrap_or_default();
            let tensor = Tensor::metadata_only(None, element_type, shape, TensorStorage::Absent);
            let tensor_id = model.add_tensor(tensor);
            let entry = values.ensure_name(&mut model, &mut lowered, &initializer.name);
            lowered.values[entry].type_info = type_info;
            lowered.values[entry].initializer = Some(tensor_id);
        }
        model.add_function(lowered);
    }

    Ok(model)
}

fn post_order_modules(mut modules: Vec<ParsedModule>) -> Vec<ParsedModule> {
    modules.sort_by(|left, right| {
        if !left.name.is_empty() && right.name.starts_with(&format!("{}::", left.name)) {
            std::cmp::Ordering::Greater
        } else if !right.name.is_empty() && left.name.starts_with(&format!("{}::", right.name)) {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    });
    modules
}

fn lower_attribute(model: &mut Model, attribute: ParsedAttribute) -> Attribute {
    let value = match attribute.value {
        ParsedAttributeValue::String(value) => AttributeValue::String(model.intern(value)),
        ParsedAttributeValue::Reference(value) => AttributeValue::Reference(model.intern(value)),
        ParsedAttributeValue::Int(value) => AttributeValue::Int(value),
        ParsedAttributeValue::Float(value) => AttributeValue::Float(value),
        ParsedAttributeValue::Tensor { type_text, len } => {
            let type_info = parse_type_info(model, type_text.as_deref());
            let element_type = type_info
                .as_ref()
                .and_then(|info| info.element_type.clone())
                .unwrap_or(TensorElementType::Unknown);
            let shape = type_info
                .as_ref()
                .map(|info| info.shape.clone())
                .unwrap_or_default();
            let storage = if len > 0 {
                TensorStorage::ElementList { len }
            } else {
                TensorStorage::Absent
            };
            AttributeValue::Tensor(model.add_tensor(Tensor::metadata_only(
                None,
                element_type,
                shape,
                storage,
            )))
        }
        ParsedAttributeValue::Ints(values) => AttributeValue::Ints(values),
        ParsedAttributeValue::Strings(values) => AttributeValue::Strings(
            values
                .into_iter()
                .map(|value| model.intern(value))
                .collect(),
        ),
    };
    Attribute {
        name: model.intern(attribute.name),
        value,
    }
}

#[derive(Default)]
struct FunctionValues {
    indices: HashMap<String, usize>,
}

impl FunctionValues {
    fn ensure(&mut self, model: &mut Model, function: &mut Function, value: &ParsedValue) -> usize {
        let index = self.ensure_name(model, function, &value.name);
        if function.values[index].type_info.is_none() {
            function.values[index].type_info = parse_type_info(model, value.type_text.as_deref());
        }
        index
    }

    fn ensure_name(&mut self, model: &mut Model, function: &mut Function, name: &str) -> usize {
        if let Some(index) = self.indices.get(name).copied() {
            return index;
        }
        let index = function.values.len();
        function.values.push(FunctionValue {
            name: model.intern(name),
            type_info: None,
            initializer: None,
        });
        self.indices.insert(name.to_owned(), index);
        index
    }
}

fn lower_graph_node(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    parsed: ParsedNode,
) {
    let inputs = parsed
        .inputs
        .iter()
        .map(|name| (!name.is_empty()).then(|| ensure_graph_value(model, graph, values, name)))
        .collect::<Vec<_>>();
    let outputs = parsed
        .outputs
        .iter()
        .map(|name| Some(ensure_graph_value(model, graph, values, name)))
        .collect::<Vec<_>>();
    let mut node = Node::new(
        graph.id,
        Operator {
            domain: None,
            name: model.intern(&parsed.operator),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.metadata = parsed.metadata;
    node.inputs = inputs.clone();
    node.outputs = outputs.clone();
    node.attributes = parsed
        .attributes
        .into_iter()
        .map(|attribute| lower_attribute(model, attribute))
        .collect();
    let node_id = graph.add_node(node);
    for value_id in inputs.into_iter().flatten() {
        graph.values[value_id.index()].consumers.push(node_id);
    }
    for value_id in outputs.into_iter().flatten() {
        graph.values[value_id.index()].producer = Some(node_id);
    }
}

fn ensure_graph_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    name: &str,
) -> ValueId {
    if let Some(id) = values.get(name).copied() {
        return id;
    }
    let id = graph.add_value(Value::new(model.intern(name)));
    values.insert(name.to_owned(), id);
    id
}

#[derive(Default)]
struct ParsedModel {
    metadata: BTreeMap<String, String>,
    modules: Vec<ParsedModule>,
    functions: Vec<ParsedFunction>,
}

struct ParsedModule {
    name: String,
    synthetic: bool,
    function_count: usize,
    metadata: BTreeMap<String, String>,
    nodes: Vec<ParsedNode>,
}

#[derive(Default)]
struct ParsedFunction {
    name: String,
    metadata: BTreeMap<String, String>,
    inputs: Vec<ParsedValue>,
    outputs: Vec<ParsedValue>,
    values: Vec<ParsedValue>,
    initializers: Vec<ParsedValue>,
    nodes: Vec<ParsedNode>,
}

#[derive(Clone)]
struct ParsedValue {
    name: String,
    type_text: Option<String>,
}

struct ParsedNode {
    operator: String,
    metadata: BTreeMap<String, String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<ParsedAttribute>,
    initializer: Option<ParsedValue>,
}

struct ParsedAttribute {
    name: String,
    value: ParsedAttributeValue,
}

enum ParsedAttributeValue {
    String(String),
    Reference(String),
    Int(i64),
    Float(f32),
    Tensor {
        type_text: Option<String>,
        len: usize,
    },
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

struct ModuleScope {
    name: String,
    synthetic: bool,
    open_depth: isize,
    parsed_index: Option<usize>,
    anonymous_child_counter: usize,
}

struct FunctionCapture {
    header: String,
    start_line: usize,
    body_depth: isize,
    prefix: Option<String>,
    lines: Vec<CapturedLine>,
}

#[derive(Clone)]
struct CapturedLine {
    number: usize,
    text: String,
    local_depth: isize,
}

fn parse_text(text: &str) -> ParsedModel {
    let mut parsed = ParsedModel::default();
    let mut depth = 0isize;
    let synthesize_anonymous_modules = anonymous_module_count(text) > 1;
    let mut anonymous_module_counter = 0usize;
    let mut modules = Vec::<ModuleScope>::new();
    let mut root_nodes = Vec::<ParsedNode>::new();
    let mut alias: Option<(String, String)> = None;
    let mut header: Option<(usize, String, Option<String>)> = None;
    let mut function: Option<FunctionCapture> = None;

    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        let line = strip_comment(raw);
        let trimmed = line.trim();
        let before = depth;

        if let Some((name, mut value)) = alias.take() {
            if !line.trim().is_empty() {
                value.push('\n');
                value.push_str(line.trim_end());
            }
            if delimiters_balanced(&value) {
                parsed.metadata.insert(name, normalize_alias_value(&value));
            } else {
                alias = Some((name, value));
            }
            depth = before + brace_delta(&line);
            continue;
        }

        if let Some(active) = function.as_mut() {
            let after = before + brace_delta(&line);
            if before >= active.body_depth
                && !(trimmed.starts_with('}') && after < active.body_depth)
            {
                active.lines.push(CapturedLine {
                    number,
                    text: line.clone(),
                    local_depth: before - active.body_depth,
                });
            }
            depth = after;
            if depth < active.body_depth
                && let Some(done) = function.take()
            {
                parsed.functions.push(parse_function(done));
                pop_modules(&mut modules, depth);
            }
            continue;
        }

        if should_finish_pending_header(header.as_ref(), trimmed)
            && let Some((start_line, text, prefix)) = header.take()
        {
            parsed.functions.push(parse_function(FunctionCapture {
                header: text,
                start_line,
                body_depth: before,
                prefix,
                lines: Vec::new(),
            }));
        }

        if let Some((start_line, mut text, prefix)) = header.take() {
            if trimmed.is_empty() {
                header = Some((start_line, text, prefix));
                depth = before + brace_delta(&line);
                continue;
            }
            text.push(' ');
            text.push_str(trimmed);
            let after = before + brace_delta(&line);
            if has_body_open(trimmed) {
                function = Some(FunctionCapture {
                    header: text,
                    start_line,
                    body_depth: after,
                    prefix,
                    lines: Vec::new(),
                });
            } else {
                header = Some((start_line, text, prefix));
            }
            depth = after;
            continue;
        }

        if let Some((name, value)) = parse_alias_definition(trimmed) {
            if delimiters_balanced(&value) {
                parsed.metadata.insert(name, normalize_alias_value(&value));
            } else {
                alias = Some((name, value));
            }
        }

        if should_parse_module(trimmed, before, &modules)
            && let Some(mut module) = parse_module(trimmed, synthesize_anonymous_modules, || {
                next_anonymous_module_name(
                    trimmed,
                    before,
                    &mut modules,
                    &mut anonymous_module_counter,
                )
            })
        {
            let scope_name = module.name.clone();
            let synthetic = module.synthetic;
            if module.synthetic
                && !module.name.is_empty()
                && let Some(prefix) = module_prefix(&modules)
            {
                module.name = format!("{prefix}::{}", module.name);
            }
            parsed.modules.push(module);
            let parsed_index = parsed.modules.len() - 1;
            modules.push(ModuleScope {
                name: scope_name,
                synthetic,
                open_depth: before,
                parsed_index: Some(parsed_index),
                anonymous_child_counter: 0,
            });
        }

        if is_function_start(trimmed) && should_parse_function(before, &modules) {
            if suppress_executable_function(&parsed, &modules, trimmed) {
                depth += brace_delta(&line);
                pop_modules(&mut modules, depth);
                continue;
            }
            mark_module_function(&mut parsed, &modules, before);
            let prefix = module_prefix(&modules);
            if has_body_open(trimmed) {
                let after = before + brace_delta(&line);
                function = Some(FunctionCapture {
                    header: trimmed.to_owned(),
                    start_line: number,
                    body_depth: after,
                    prefix,
                    lines: Vec::new(),
                });
                depth = after;
                continue;
            }
            header = Some((number, trimmed.to_owned(), prefix));
        }

        if before == 0
            && !trimmed.is_empty()
            && !trimmed.starts_with('}')
            && !is_module_start(trimmed)
            && !is_spirv_module_start(trimmed)
            && !is_function_start(trimmed)
            && !is_alias_definition(trimmed)
            && !trimmed.starts_with('#')
            && !trimmed.starts_with('!')
            && let Some(parsed_op) = parse_operation(
                &CapturedLine {
                    number,
                    text: line.clone(),
                    local_depth: 0,
                },
                None,
            )
            && let Some(node) = parsed_op.node
        {
            root_nodes.push(node);
        }

        if before
            == modules
                .last()
                .map(|module| module.open_depth + 1)
                .unwrap_or(-1)
            && !trimmed.is_empty()
            && !trimmed.starts_with('}')
            && !trimmed.starts_with("module ")
            && !is_function_start(trimmed)
        {
            let Some(last_module) = modules.last() else {
                break;
            };
            let prefix = module_prefix(&modules);
            if let Some(index) = last_module.parsed_index
                && let Some(parsed_op) = parse_operation(
                    &CapturedLine {
                        number,
                        text: line.clone(),
                        local_depth: 0,
                    },
                    prefix.as_deref(),
                )
                && let Some(node) = parsed_op.node
            {
                parsed.modules[index].nodes.push(node);
            }
        }

        depth += brace_delta(&line);
        pop_modules(&mut modules, depth);
    }

    if let Some(done) = function {
        parsed.functions.push(parse_function(done));
    }
    if let Some((name, value)) = alias {
        parsed.metadata.insert(name, normalize_alias_value(&value));
    }
    if let Some((start_line, text, prefix)) = header
        && is_complete_function_header(&text)
    {
        parsed.functions.push(parse_function(FunctionCapture {
            header: text,
            start_line,
            body_depth: depth,
            prefix,
            lines: Vec::new(),
        }));
    }
    if !root_nodes.is_empty() {
        parsed.modules.push(ParsedModule {
            name: String::new(),
            synthetic: false,
            function_count: 0,
            metadata: BTreeMap::new(),
            nodes: root_nodes,
        });
    }
    defer_unscoped_functions(&mut parsed.functions);
    finalize_location_metadata(&mut parsed);
    parsed
}

fn validate_external_resource_paths(
    source: Option<&std::path::Path>,
    parsed: &ParsedModel,
    allow_unsafe_paths: bool,
) -> Result<(), ModelError> {
    fn check_location(
        source: Option<&std::path::Path>,
        location: &str,
        allow_unsafe_paths: bool,
    ) -> Result<(), ModelError> {
        let location = location.trim().trim_matches('"');
        if location.starts_with('#') {
            return Ok(());
        }
        validate_external_path(source, location, allow_unsafe_paths)
    }

    fn check_attributes(
        source: Option<&std::path::Path>,
        attributes: &[ParsedAttribute],
        allow_unsafe_paths: bool,
    ) -> Result<(), ModelError> {
        for attribute in attributes {
            if attribute.name != "rodata" {
                continue;
            }
            if let ParsedAttributeValue::String(value) = &attribute.value
                && !value.is_empty()
            {
                validate_external_path(source, value, allow_unsafe_paths)?;
            }
        }
        Ok(())
    }

    for module in &parsed.modules {
        for node in &module.nodes {
            if let Some(location) = node.metadata.get("location") {
                check_location(source, location, allow_unsafe_paths)?;
            }
            check_attributes(source, &node.attributes, allow_unsafe_paths)?;
        }
    }
    for function in &parsed.functions {
        for node in &function.nodes {
            if let Some(location) = node.metadata.get("location") {
                check_location(source, location, allow_unsafe_paths)?;
            }
            check_attributes(source, &node.attributes, allow_unsafe_paths)?;
        }
    }
    Ok(())
}

fn anonymous_module_count(text: &str) -> usize {
    text.lines()
        .filter(|line| {
            let line = line.trim_start();
            (is_module_start(line) || is_spirv_module_start(line))
                && parse_module_name(line).is_none()
        })
        .count()
}

fn should_parse_module(line: &str, depth: isize, modules: &[ModuleScope]) -> bool {
    (is_module_start(line) || is_spirv_module_start(line))
        && (depth == 0
            || modules
                .last()
                .is_some_and(|module| depth == module.open_depth + 1))
}

fn should_parse_function(depth: isize, modules: &[ModuleScope]) -> bool {
    depth == 0
        || modules
            .last()
            .is_some_and(|module| depth == module.open_depth + 1)
}

fn next_anonymous_module_name(
    line: &str,
    depth: isize,
    modules: &mut [ModuleScope],
    anonymous_counter: &mut usize,
) -> String {
    if is_spirv_module_start(line)
        && let Some(parent) = modules
            .last_mut()
            .filter(|module| depth == module.open_depth + 1)
    {
        let name = format!("${}", parent.anonymous_child_counter);
        parent.anonymous_child_counter += 1;
        name
    } else {
        let name = format!("${anonymous_counter}");
        *anonymous_counter += 1;
        name
    }
}

fn is_module_start(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("module ")
        || line.starts_with("\"builtin.module\"")
        || line.starts_with("builtin.module ")
        || line.starts_with("vm.module ")
}

fn is_spirv_module_start(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("spv.module ") || line.starts_with("spirv.module ")
}

fn defer_unscoped_functions(functions: &mut Vec<ParsedFunction>) {
    if !functions
        .iter()
        .any(|function| function.name.contains("::"))
    {
        return;
    }
    let mut scoped = Vec::with_capacity(functions.len());
    let mut unscoped = Vec::new();
    for function in functions.drain(..) {
        if function.name.contains("::") {
            scoped.push(function);
        } else {
            unscoped.push(function);
        }
    }
    scoped.extend(unscoped);
    *functions = scoped;
}

fn mark_module_function(parsed: &mut ParsedModel, modules: &[ModuleScope], depth: isize) {
    let Some(module) = modules
        .last()
        .filter(|module| depth == module.open_depth + 1)
        .and_then(|module| module.parsed_index)
    else {
        return;
    };
    if let Some(module) = parsed.modules.get_mut(module) {
        module.function_count += 1;
    }
}

fn finalize_location_metadata(parsed: &mut ParsedModel) {
    fn finalize_node(node: &mut ParsedNode, aliases: &BTreeMap<String, String>) {
        let Some(location) = node.metadata.remove(INTERNAL_LOCATION_REF) else {
            return;
        };
        match resolve_location_metadata(&location, aliases) {
            Some(Some(value)) => {
                node.metadata.insert("location".to_owned(), value);
            }
            Some(None) => {
                node.metadata.remove("location");
            }
            None => {}
        }
    }

    for module in &mut parsed.modules {
        for node in &mut module.nodes {
            finalize_node(node, &parsed.metadata);
        }
    }
    for function in &mut parsed.functions {
        for node in &mut function.nodes {
            finalize_node(node, &parsed.metadata);
        }
    }
}

fn resolve_location_metadata(
    location: &str,
    aliases: &BTreeMap<String, String>,
) -> Option<Option<String>> {
    let value = aliases
        .get(location)
        .map(String::as_str)
        .unwrap_or(location)
        .trim();
    if value == "unknown" || value == "loc(unknown)" {
        return Some(None);
    }
    if value.starts_with('"') && value.ends_with('"') {
        return Some(Some(value.to_owned()));
    }
    if let Some(inner) = value
        .strip_prefix("loc(")
        .and_then(|value| value.strip_suffix(')'))
    {
        let inner = inner.trim();
        if inner == "unknown" {
            Some(None)
        } else if inner.starts_with('"') && inner.ends_with('"') {
            Some(Some(inner.to_owned()))
        } else {
            None
        }
    } else {
        None
    }
}

fn should_finish_pending_header(
    header: Option<&(usize, String, Option<String>)>,
    line: &str,
) -> bool {
    let Some((_, text, _)) = header else {
        return false;
    };
    is_complete_function_header(text)
        && !line.is_empty()
        && (is_function_start(line) || line.starts_with("module ") || is_alias_definition(line))
}

fn is_complete_function_header(header: &str) -> bool {
    let header = header.trim();
    if header.is_empty() || header.ends_with("->") || header.ends_with(',') || header.ends_with(':')
    {
        return false;
    }
    delimiters_balanced(header)
}

fn delimiters_balanced(text: &str) -> bool {
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    for ch in text.chars() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
        if angle < 0 || paren < 0 || bracket < 0 || brace < 0 {
            return false;
        }
    }
    !quote && angle == 0 && paren == 0 && bracket == 0 && brace == 0
}

fn is_alias_definition(line: &str) -> bool {
    let Some((name, value)) = line.split_once('=') else {
        return false;
    };
    name.trim().starts_with('#') && !value.trim().is_empty()
}

fn parse_alias_definition(line: &str) -> Option<(String, String)> {
    let (name, value) = line.split_once('=')?;
    let name = name.trim();
    let value = value.trim();
    if !name.starts_with('#') || value.is_empty() {
        return None;
    }
    Some((name.to_owned(), normalize_alias_value(value)))
}

fn normalize_alias_value(value: &str) -> String {
    let value = value.trim().trim_end_matches(',');
    if value.starts_with('(') && value.contains("->") {
        format!("affine_map<{value}>")
    } else {
        value.to_owned()
    }
}

fn pop_modules(modules: &mut Vec<ModuleScope>, depth: isize) {
    while modules
        .last()
        .is_some_and(|module| depth <= module.open_depth)
    {
        modules.pop();
    }
}

fn module_prefix(modules: &[ModuleScope]) -> Option<String> {
    let named_start = modules
        .iter()
        .position(|module| !module.synthetic && !module.name.is_empty())
        .unwrap_or(0);
    let parts = modules
        .iter()
        .skip(named_start)
        .map(|module| module.name.as_str())
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("::"))
    }
}

fn suppress_executable_function(
    parsed: &ParsedModel,
    modules: &[ModuleScope],
    header: &str,
) -> bool {
    let Some(function_name) = parse_function_name(header) else {
        return false;
    };
    let Some(module_index) = modules.last().and_then(|module| module.parsed_index) else {
        return false;
    };
    parsed.modules.get(module_index).is_some_and(|module| {
        module.nodes.iter().any(|node| {
            matches!(node.operator.as_str(), "flow.executable" | "hal.executable")
                && node_symbol_name(node).is_some_and(|name| name == function_name)
        })
    })
}

fn node_symbol_name(node: &ParsedNode) -> Option<&str> {
    node.attributes.iter().find_map(|attribute| {
        (attribute.name == "sym_name").then(|| match &attribute.value {
            ParsedAttributeValue::String(value) | ParsedAttributeValue::Reference(value) => {
                Some(value.as_str())
            }
            _ => None,
        })?
    })
}

fn parse_module<F>(
    line: &str,
    synthesize_anonymous: bool,
    next_anonymous_name: F,
) -> Option<ParsedModule>
where
    F: FnOnce() -> String,
{
    let trimmed = line.trim_start();
    let rest = if let Some(rest) = trimmed.strip_prefix("module ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("builtin.module ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("\"builtin.module\"") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("vm.module ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("spv.module ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("spirv.module ") {
        rest
    } else {
        return None;
    };
    let mut metadata = extract_attribute_dict(rest)
        .map(parse_metadata_dict)
        .unwrap_or_default();
    if let Some(visibility) = rest
        .split_whitespace()
        .find(|token| matches!(*token, "public" | "private" | "nested"))
    {
        metadata.insert("sym_visibility".to_owned(), visibility.to_owned());
    }
    if is_spirv_module_start(trimmed) {
        metadata.extend(parse_spirv_module_metadata(rest));
    }
    let explicit_name = parse_module_name(line).or_else(|| metadata.get("sym_name").cloned());
    let synthetic = explicit_name.is_none() && synthesize_anonymous;
    let name = explicit_name.unwrap_or_else(|| {
        if synthesize_anonymous {
            next_anonymous_name()
        } else {
            String::new()
        }
    });
    Some(ParsedModule {
        name,
        synthetic,
        function_count: 0,
        metadata,
        nodes: Vec::new(),
    })
}

fn parse_module_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = trimmed
        .strip_prefix("module ")
        .or_else(|| trimmed.strip_prefix("builtin.module "))
        .or_else(|| trimmed.strip_prefix("\"builtin.module\""))
        .or_else(|| trimmed.strip_prefix("vm.module "))
        .or_else(|| trimmed.strip_prefix("spv.module "))
        .or_else(|| trimmed.strip_prefix("spirv.module "))?;
    rest.split_whitespace()
        .find(|value| value.starts_with('@'))
        .map(str::to_owned)
}

fn parse_spirv_module_metadata(rest: &str) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    let prefix = rest
        .split_once('{')
        .map(|(prefix, _)| prefix)
        .unwrap_or(rest);
    let mut tokens = prefix.split_whitespace().peekable();
    if tokens.peek().is_some_and(|token| token.starts_with('@')) {
        tokens.next();
    }
    if let Some(addressing_model) = tokens
        .next()
        .filter(|token| *token != "attributes" && *token != "requires")
    {
        metadata.insert("addressing_model".to_owned(), addressing_model.to_owned());
    }
    if let Some(memory_model) = tokens.next().filter(|token| *token != "requires") {
        metadata.insert("memory_model".to_owned(), memory_model.to_owned());
    }
    if let Some(index) = rest.find("requires ") {
        let value = rest[index + "requires ".len()..]
            .split_once('{')
            .map(|(value, _)| value)
            .unwrap_or(&rest[index + "requires ".len()..])
            .trim();
        if !value.is_empty() {
            metadata.insert("vce_triple".to_owned(), value.to_owned());
        }
    }
    metadata
}

fn is_function_start(line: &str) -> bool {
    normalized_operator_token(line).is_some_and(is_function_start_token)
}

fn normalized_operator_token(line: &str) -> Option<&str> {
    normalized_operator_token_from_token(line.split_whitespace().next()?)
}

fn parse_function(capture: FunctionCapture) -> ParsedFunction {
    let header = capture.header.replace('\n', " ");
    let base = parse_function_name(&header).unwrap_or_else(|| format!("${}", capture.start_line));
    let name = capture
        .prefix
        .as_ref()
        .map(|prefix| format!("{prefix}::@{base}"))
        .unwrap_or_else(|| format!("@{base}"));
    let (function_inputs, function_outputs) = parse_function_type(&header).unwrap_or_default();
    let mut function = ParsedFunction {
        name,
        ..ParsedFunction::default()
    };
    if let Some(visibility) = function_visibility(&header) {
        function
            .metadata
            .insert("sym_visibility".to_owned(), visibility);
    } else if header.contains(" public @") || header.contains("func.func public ") {
        function
            .metadata
            .insert("sym_visibility".to_owned(), "public".to_owned());
    } else if header.contains(" private @") || header.contains("func.func private ") {
        function
            .metadata
            .insert("sym_visibility".to_owned(), "private".to_owned());
    }
    function.inputs = parse_function_inputs(&header);
    function.outputs = parse_function_outputs(&header);
    if capture.lines.is_empty() && header.trim_start().starts_with("func ") {
        normalize_legacy_declaration_outputs(&mut function.outputs);
    }

    for statement in top_level_statements(&capture.lines) {
        let block_arguments = parse_block_arguments(&statement.text);
        for value in block_arguments {
            upsert_value(&mut function.values, value);
        }
        if statement.text.trim_start().starts_with('^') {
            continue;
        }
        if let Some(return_values) = parse_return(&statement.text) {
            if !return_values.is_empty() {
                function.outputs = return_values;
            }
            continue;
        }
        if let Some(parsed) = parse_operation(&statement, capture.prefix.as_deref()) {
            for value in &parsed.values {
                upsert_value(&mut function.values, value.clone());
            }
            if let Some(node) = parsed.node {
                function.nodes.push(node);
            }
        }
    }
    if function.inputs.is_empty() && !function_inputs.is_empty() {
        function.inputs = function_inputs
            .into_iter()
            .enumerate()
            .map(|(index, type_text)| ParsedValue {
                name: format!("%arg{index}"),
                type_text: Some(type_text),
            })
            .collect();
    }
    if function.outputs.is_empty() && !function_outputs.is_empty() {
        function.outputs = function_outputs
            .into_iter()
            .enumerate()
            .map(|(index, type_text)| ParsedValue {
                name: format!("%result{index}"),
                type_text: Some(type_text),
            })
            .collect();
    }
    fold_torch_constants(&mut function);
    fold_dense_initializers(&mut function);
    normalize_convolution_cast_inputs(&mut function);
    prune_unreferenced_values(&mut function);

    function
}

fn parse_block_arguments(text: &str) -> Vec<ParsedValue> {
    let text = text.trim_start();
    if !text.starts_with('^') {
        return Vec::new();
    }
    let Some(open) = text.find('(') else {
        return Vec::new();
    };
    let Some(close) = matching_delimiter(text, open, '(', ')') else {
        return Vec::new();
    };
    split_top_level(&text[open + 1..close], ',')
        .into_iter()
        .filter_map(|arg| {
            let (name, type_text) = arg.split_once(':')?;
            Some(ParsedValue {
                name: name.trim().to_owned(),
                type_text: Some(clean_type_token(type_text.trim()).to_owned()),
            })
        })
        .collect()
}

fn normalize_legacy_declaration_outputs(outputs: &mut [ParsedValue]) {
    for output in outputs {
        let Some(type_text) = output.type_text.as_deref() else {
            continue;
        };
        if !type_text.starts_with("tensor<")
            && !type_text.starts_with("memref<")
            && !type_text.starts_with("vector<")
            && !type_text.starts_with('!')
        {
            output.name = type_text.to_owned();
            output.type_text = None;
        }
    }
}

fn fold_torch_constants(function: &mut ParsedFunction) {
    let constants = function
        .nodes
        .iter()
        .filter(|node| node.operator.starts_with("torch.constant."))
        .flat_map(|node| node.outputs.iter().cloned())
        .collect::<std::collections::HashSet<_>>();
    if constants.is_empty() {
        return;
    }
    let constant_lists = function
        .nodes
        .iter()
        .filter(|node| {
            node.operator == "prim.ListConstruct"
                && node.outputs.len() == 1
                && !node.inputs.is_empty()
                && node.inputs.iter().all(|input| constants.contains(input))
        })
        .map(|node| (node.outputs[0].clone(), node.inputs.len()))
        .collect::<HashMap<_, _>>();
    let constant_list_outputs = constant_lists
        .keys()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    function.values.retain(|value| {
        !constants.contains(&value.name) && !constant_list_outputs.contains(&value.name)
    });
    function.initializers.retain(|value| {
        !constants.contains(&value.name) && !constant_list_outputs.contains(&value.name)
    });
    for node in &mut function.nodes {
        if node.operator.starts_with("torch.constant.")
            || node
                .outputs
                .iter()
                .any(|output| constant_list_outputs.contains(output))
        {
            continue;
        }
        let mut inputs = Vec::new();
        for input in node.inputs.drain(..) {
            if constants.contains(&input) {
                continue;
            }
            if let Some(count) = constant_lists.get(&input).copied() {
                for _ in 0..count {
                    inputs.push(String::new());
                }
            } else {
                inputs.push(input);
            }
        }
        node.inputs = inputs;
    }
    function.nodes.retain(|node| {
        !node.operator.starts_with("torch.constant.")
            && !node
                .outputs
                .iter()
                .any(|output| constant_list_outputs.contains(output))
    });
}

fn fold_dense_initializers(function: &mut ParsedFunction) {
    let mut input_counts = HashMap::<String, usize>::new();
    for node in &function.nodes {
        for input in &node.inputs {
            *input_counts.entry(input.clone()).or_insert(0) += 1;
        }
    }
    let value_types = function
        .values
        .iter()
        .map(|value| (value.name.clone(), value.type_text.clone()))
        .collect::<HashMap<_, _>>();
    let mut convolution_result_types = HashMap::<String, String>::new();
    for node in &function.nodes {
        if !matches!(
            node.operator.as_str(),
            "stablehlo.convolution" | "mhlo.convolution"
        ) {
            continue;
        }
        let Some(output_type) = node
            .outputs
            .first()
            .and_then(|name| value_types.get(name))
            .cloned()
            .flatten()
        else {
            continue;
        };
        for input in &node.inputs {
            convolution_result_types.insert(input.clone(), output_type.clone());
        }
    }
    let returned = function
        .outputs
        .iter()
        .map(|value| value.name.clone())
        .collect::<std::collections::HashSet<_>>();
    let mut nodes = Vec::with_capacity(function.nodes.len());
    for mut node in function.nodes.drain(..) {
        let fold = node.initializer.as_ref().is_some_and(|initializer| {
            input_counts.get(&initializer.name).copied() == Some(1)
                && !returned.contains(&initializer.name)
        });
        if fold {
            if let Some(mut initializer) = node.initializer.take() {
                if let Some(type_text) = convolution_result_types.get(&initializer.name) {
                    initializer.type_text = Some(type_text.clone());
                }
                upsert_value(&mut function.values, initializer.clone());
                function.initializers.push(initializer);
            }
        } else {
            node.initializer = None;
            nodes.push(node);
        }
    }
    function.nodes = nodes;
}

fn normalize_convolution_cast_inputs(function: &mut ParsedFunction) {
    let mut cast_outputs = HashSet::<String>::new();
    let mut convolution_result_types = HashMap::<String, String>::new();

    for node in &function.nodes {
        if node.operator == "hal.tensor.cast" {
            cast_outputs.extend(node.outputs.iter().cloned());
            continue;
        }
        if !matches!(
            node.operator.as_str(),
            "stablehlo.convolution" | "mhlo.convolution"
        ) {
            continue;
        }
        let Some(output_type) = node
            .outputs
            .first()
            .and_then(|name| {
                function
                    .values
                    .iter()
                    .find(|value| value.name == *name)
                    .and_then(|value| value.type_text.clone())
            })
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        for input in &node.inputs {
            convolution_result_types.insert(input.clone(), output_type.clone());
        }
    }

    for value in &mut function.values {
        if !cast_outputs.contains(&value.name) {
            continue;
        }
        if let Some(output_type) = convolution_result_types.get(&value.name) {
            value.type_text = Some(output_type.clone());
        }
    }
}

fn prune_unreferenced_values(function: &mut ParsedFunction) {
    let mut referenced = std::collections::HashSet::<String>::new();
    referenced.extend(function.inputs.iter().map(|value| value.name.clone()));
    referenced.extend(function.outputs.iter().map(|value| value.name.clone()));
    referenced.extend(function.initializers.iter().map(|value| value.name.clone()));
    for node in &function.nodes {
        referenced.extend(node.inputs.iter().cloned());
        referenced.extend(node.outputs.iter().cloned());
    }
    function
        .values
        .retain(|value| referenced.contains(&value.name));
}

fn upsert_value(values: &mut Vec<ParsedValue>, value: ParsedValue) {
    if let Some(existing) = values
        .iter_mut()
        .find(|existing| existing.name == value.name)
    {
        if existing.type_text.is_none() {
            existing.type_text = value.type_text;
        }
    } else {
        values.push(value);
    }
}

struct ParsedOperation {
    values: Vec<ParsedValue>,
    node: Option<ParsedNode>,
}

fn parse_operation(statement: &CapturedLine, prefix: Option<&str>) -> Option<ParsedOperation> {
    let text = statement.text.trim();
    if text.is_empty() || text.starts_with('}') || text.starts_with('^') || text.starts_with('#') {
        return None;
    }
    let (result_part, rhs) = split_assignment(text);
    let results = result_part
        .map(extract_result_names)
        .unwrap_or_default()
        .into_iter()
        .filter(|name| name.starts_with('%'))
        .collect::<Vec<_>>();
    let (operator, remainder, op_column) = parse_operator(rhs, &statement.text)?;
    let operator = normalize_operator_name(operator);
    if operator.ends_with(".return") || operator == "return" {
        return None;
    }

    let output_types = result_types(&operator, rhs, results.len());
    let mut values = results
        .iter()
        .enumerate()
        .map(|(index, name)| ParsedValue {
            name: name.clone(),
            type_text: output_types.get(index).cloned(),
        })
        .collect::<Vec<_>>();
    if operator == "tt.load" {
        for value in &mut values {
            if let Some(type_text) = value.type_text.as_deref().and_then(triton_load_result_type) {
                value.type_text = Some(type_text);
            }
        }
    }
    if operator == "torch.constant" && output_types.is_empty() {
        for value in &mut values {
            value.type_text = Some("none".to_owned());
        }
    }

    let initializer = if is_dense_constant(&operator, rhs) {
        results.first().map(|name| ParsedValue {
            name: name.clone(),
            type_text: output_types.first().cloned().or_else(|| constant_type(rhs)),
        })
    } else {
        None
    };

    let mut inputs = parse_inputs(&operator, remainder);
    let mut attributes = parse_attributes(&operator, remainder, prefix);
    if operator == "memref.dim" {
        inputs.extend(parse_dimension_indices(remainder));
        for input in inputs.iter().filter(|value| !value.starts_with('%')) {
            values.push(ParsedValue {
                name: input.clone(),
                type_text: Some("index".to_owned()),
            });
        }
    }
    if operator == "hal.interface.binding.subspan" {
        inputs.extend(subspan_dynamic_dim_inputs(remainder));
    }
    if operator == "memref.subview" {
        inputs = parse_memref_subview_inputs(remainder);
    }
    if matches!(operator.as_str(), "memref.alloc" | "alloc") {
        stringify_named_attributes(&mut attributes, &["alignment"]);
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![inputs.len() as i64, 0]),
        });
    }
    if operator == "affine.for" {
        attributes.extend(parse_affine_for_attrs(remainder));
        inputs.clear();
    }
    if operator.starts_with("linalg.")
        && let Some(segments) = parse_linalg_operand_segments(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(segments),
        });
    }
    if matches!(
        operator.as_str(),
        "flow.dispatch" | "flow.dispatch.workgroups"
    ) {
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(parse_flow_dispatch_operand_segments(remainder)),
        });
    }
    if operator == "scf.for" {
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(parse_scf_for_operand_segments(remainder)),
        });
    }
    if operator == "arm_sme.tile_load" {
        if let Some(layout) = angle_argument(remainder, "layout") {
            attributes.push(ParsedAttribute {
                name: "layout".to_owned(),
                value: ParsedAttributeValue::String(layout),
            });
        }
        let index_count =
            bracket_value_count(remainder).unwrap_or_else(|| inputs.len().saturating_sub(1));
        let extra = inputs.len().saturating_sub(1 + index_count);
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![
                1,
                index_count as i64,
                i64::from(extra > 0),
                i64::from(extra > 1),
            ]),
        });
    }
    if operator == "arm_sme.tile_store" {
        if let Some(layout) = angle_argument(remainder, "layout") {
            attributes.push(ParsedAttribute {
                name: "layout".to_owned(),
                value: ParsedAttributeValue::String(layout),
            });
        }
        let index_count =
            bracket_value_count(remainder).unwrap_or_else(|| inputs.len().saturating_sub(2));
        let extra = inputs.len().saturating_sub(2 + index_count);
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![1, 1, index_count as i64, i64::from(extra > 0)]),
        });
    }
    if operator == "arm_sme.load_tile_slice" {
        let index_count = bracket_value_count(remainder).unwrap_or(0);
        reorder_arm_sme_tile_slice_inputs(&mut inputs, index_count);
        if let Some(layout) = angle_argument(remainder, "layout") {
            attributes.push(ParsedAttribute {
                name: "layout".to_owned(),
                value: ParsedAttributeValue::String(layout),
            });
        }
    }
    if matches!(
        operator.as_str(),
        "arm_sme.store_tile_slice" | "arm_sme.insert_tile_slice" | "arm_sme.extract_tile_slice"
    ) && let Some(layout) = angle_argument(remainder, "layout")
    {
        attributes.push(ParsedAttribute {
            name: "layout".to_owned(),
            value: ParsedAttributeValue::String(layout),
        });
    }
    if operator == "tt.load" && inputs.len() > 1 {
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![
                1,
                i64::from(inputs.len() > 1),
                i64::from(inputs.len() > 2),
            ]),
        });
    }
    if is_arm_sme_product(&operator) {
        if let Some(kind) = angle_argument(remainder, "kind") {
            attributes.push(ParsedAttribute {
                name: "kind".to_owned(),
                value: ParsedAttributeValue::String(kind),
            });
        }
        let has_acc = remainder.contains("acc(");
        let mask_count = call_argument_count(remainder, "masks");
        if has_acc && mask_count > 0 && inputs.len() >= 2 + 1 + mask_count {
            let acc = inputs.remove(2);
            inputs.insert(2 + mask_count, acc);
        }
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![
                1,
                1,
                i64::from(mask_count > 0),
                i64::from(mask_count > 1),
                i64::from(has_acc),
            ]),
        });
    }
    if operator == "arm_sme.streaming_vl"
        && let Some(type_size) = first_angle_argument(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "type_size".to_owned(),
            value: ParsedAttributeValue::String(type_size),
        });
    }

    let mut metadata = BTreeMap::from([(
        "location".to_owned(),
        format!("{}:{op_column}", statement.number),
    )]);
    if let Some(location) = parse_location_reference(remainder) {
        metadata.insert(INTERNAL_LOCATION_REF.to_owned(), location);
    }

    Some(ParsedOperation {
        values,
        node: Some(ParsedNode {
            operator,
            metadata,
            inputs,
            outputs: results,
            attributes,
            initializer,
        }),
    })
}

fn split_assignment(text: &str) -> (Option<&str>, &str) {
    if text.trim_start().starts_with('%')
        && let Some(index) = text.find('=')
    {
        return (Some(text[..index].trim()), text[index + 1..].trim());
    }
    (None, text)
}

fn parse_operator<'a>(rhs: &'a str, original_line: &str) -> Option<(String, &'a str, usize)> {
    let trimmed = rhs.trim_start();
    let skipped = rhs.len() - trimmed.len();
    if let Some(rest) = trimmed.strip_prefix('"') {
        let end = rest.find('"')?;
        let operator = rest[..end].to_owned();
        let before = &trimmed[..end + 2];
        let column = original_line
            .find(before)
            .map(|index| index + 1)
            .unwrap_or(1);
        return Some((operator, rest[end + 1..].trim_start(), column));
    }
    let end = trimmed
        .find(|ch: char| ch.is_whitespace() || matches!(ch, '(' | '<' | '[' | '{'))
        .unwrap_or(trimmed.len());
    if end == 0 {
        return None;
    }
    let operator = trimmed[..end].to_owned();
    let token = &trimmed[..end];
    let column = original_line
        .find('=')
        .and_then(|eq| {
            original_line[eq + 1..]
                .find(token)
                .map(|index| eq + index + 2)
        })
        .or_else(|| original_line.find(token).map(|index| index + 1))
        .unwrap_or(skipped + 1);
    Some((operator, trimmed[end..].trim_start(), column))
}

fn parse_inputs(operator: &str, remainder: &str) -> Vec<String> {
    if matches!(operator, "cf.br" | "vm.br") {
        return Vec::new();
    }
    if operator == "cf.cond_br" {
        return extract_value_names(remainder).into_iter().take(1).collect();
    }
    if operator == "scf.for" {
        return parse_scf_for_inputs(remainder);
    }
    if operator.ends_with(".for") {
        return Vec::new();
    }
    if operator == "vm.import" {
        return Vec::new();
    }
    if (operator == "call" || operator.ends_with(".call"))
        && let Some(open) = remainder.find('(')
        && let Some(close) = matching_delimiter(remainder, open, '(', ')')
    {
        return extract_value_names(&remainder[open + 1..close]);
    }
    if remainder.trim_start().starts_with('(')
        && let Some(close) = matching_delimiter(remainder, remainder.find('(').unwrap(), '(', ')')
    {
        return extract_value_names(&remainder[..=close]);
    }
    let end = first_top_level_type_separator(remainder).unwrap_or(remainder.len());
    extract_value_names(&remainder[..end])
}

fn parse_dimension_indices(remainder: &str) -> Vec<String> {
    let end = first_top_level_type_separator(remainder).unwrap_or(remainder.len());
    remainder[..end]
        .split(',')
        .skip(1)
        .filter_map(|part| {
            let value = part.trim();
            if value.chars().all(|ch| ch.is_ascii_digit()) {
                Some(value.to_owned())
            } else {
                None
            }
        })
        .collect()
}

fn parse_scf_for_inputs(remainder: &str) -> Vec<String> {
    let mut inputs = Vec::new();
    let Some(equal) = remainder.find('=') else {
        return inputs;
    };
    let after_equal = remainder[equal + 1..].trim_start();
    let Some(to_index) = after_equal.find(" to ") else {
        return inputs;
    };
    let lower = after_equal[..to_index].trim();
    if lower.starts_with('%') {
        inputs.push(lower.to_owned());
    }
    let after_to = &after_equal[to_index + 4..];
    let Some(step_index) = after_to.find(" step ") else {
        return inputs;
    };
    let upper = after_to[..step_index].trim();
    if upper.starts_with('%') {
        inputs.push(upper.to_owned());
    }
    let after_step = &after_to[step_index + 6..];
    let step_end = after_step
        .find(" iter_args(")
        .or_else(|| after_step.find(" {"))
        .or_else(|| after_step.find(" ->"))
        .unwrap_or(after_step.len());
    let step = after_step[..step_end].trim();
    if step.starts_with('%') {
        inputs.push(step.to_owned());
    }
    if let Some(iter_inputs) = parse_scf_for_iter_arg_inputs(remainder) {
        inputs.extend(iter_inputs);
    }
    inputs
}

fn parse_scf_for_iter_arg_inputs(remainder: &str) -> Option<Vec<String>> {
    let marker = "iter_args(";
    let open = remainder.find(marker)? + marker.len() - 1;
    let close = matching_delimiter(remainder, open, '(', ')')?;
    Some(
        split_top_level(&remainder[open + 1..close], ',')
            .into_iter()
            .filter_map(|part| {
                let (_, value) = part.split_once('=')?;
                let value = value.trim();
                value.starts_with('%').then(|| value.to_owned())
            })
            .collect(),
    )
}

fn parse_scf_for_operand_segments(remainder: &str) -> Vec<i64> {
    let iter_args = parse_scf_for_iter_arg_inputs(remainder)
        .map(|inputs| inputs.len() as i64)
        .unwrap_or(0);
    vec![1, 1, 1, iter_args]
}

fn parse_linalg_operand_segments(remainder: &str) -> Option<Vec<i64>> {
    Some(vec![
        named_call_value_count(remainder, "ins")? as i64,
        named_call_value_count(remainder, "outs")? as i64,
    ])
}

fn parse_attributes(operator: &str, remainder: &str, prefix: Option<&str>) -> Vec<ParsedAttribute> {
    let mut attributes = if operator == "scf.for" {
        Vec::new()
    } else {
        extract_attribute_dict(remainder)
            .map(parse_attribute_dict)
            .unwrap_or_default()
    };
    attributes.extend(parse_symbol_attributes(operator, remainder));
    if operator == "hal.executable.variant" {
        if let Some(target) = hal_executable_variant_target(remainder) {
            attributes.push(ParsedAttribute {
                name: "target".to_owned(),
                value: ParsedAttributeValue::String(target),
            });
        }
    }
    if operator == "util.global" {
        return parse_util_global_attributes(remainder);
    }
    if operator == "util.global.load"
        && let Some(global) = parse_global_symbol(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "global".to_owned(),
            value: ParsedAttributeValue::String(global),
        });
    }
    if operator.starts_with("vm.global.load.")
        && let Some(global) = parse_global_symbol(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "global".to_owned(),
            value: ParsedAttributeValue::String(global),
        });
    }
    if operator.starts_with("vm.global.store.")
        && let Some(global) = parse_global_symbol(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "global".to_owned(),
            value: ParsedAttributeValue::String(global),
        });
    }
    if operator.starts_with("vm.global.")
        && !operator.starts_with("vm.global.load.")
        && !operator.starts_with("vm.global.store.")
        && let Some(value) = vm_global_type_attribute(operator, remainder)
    {
        attributes.push(ParsedAttribute {
            name: "type".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if operator == "vm.import" {
        attributes.extend(parse_vm_import_attributes(remainder));
    }
    if operator == "vm.export" {
        attributes.extend(parse_vm_export_attributes(remainder));
    }
    if operator == "torch.symbolic_int" {
        return parse_torch_symbolic_int_attributes(remainder);
    }
    let message = if operator == "vm.fail" {
        quoted_strings(remainder).into_iter().next()
    } else if operator == "util.unreachable" {
        leading_quoted_string(remainder)
    } else {
        None
    };
    if let Some(message) = message {
        attributes.push(ParsedAttribute {
            name: "message".to_owned(),
            value: ParsedAttributeValue::String(message),
        });
    }
    if has_constant_attribute(operator) {
        stringify_constant_value_attributes(&mut attributes);
    }
    if matches!(operator, "stablehlo.convolution" | "mhlo.convolution") {
        attributes = parse_convolution_attributes(remainder);
    }
    if (operator == "call"
        || operator.ends_with(".call")
        || operator.ends_with(".generic_call")
        || operator.starts_with("vm.call"))
        && let Some(callee) = parse_call_callee(remainder)
    {
        let value = if operator.starts_with("vm.call") {
            attributes.push(ParsedAttribute {
                name: "callee".to_owned(),
                value: ParsedAttributeValue::String(callee.trim_start_matches('@').to_owned()),
            });
            String::new()
        } else if callee.starts_with('@') {
            prefix
                .map(|prefix| format!("{prefix}::{callee}"))
                .unwrap_or(callee)
        } else {
            callee
        };
        if !value.is_empty() {
            attributes.push(ParsedAttribute {
                name: "callee".to_owned(),
                value: ParsedAttributeValue::Reference(value),
            });
        }
    }
    if operator == "bufferization.materialize_in_destination" && remainder.contains(" writable ") {
        attributes.push(ParsedAttribute {
            name: "writable".to_owned(),
            value: ParsedAttributeValue::String(String::new()),
        });
    }
    if matches!(operator, "cond_br" | "cf.cond_br" | "vm.cond_br") {
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![1, 0, 0]),
        });
    }
    if operator.ends_with("broadcast_in_dim")
        && let Some(value) = named_bracket_value(remainder, "dims")
    {
        attributes.push(ParsedAttribute {
            name: "broadcast_dimensions".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if operator == "stablehlo.concatenate"
        && let Some(value) = named_scalar_value(remainder, "dim")
    {
        attributes.push(ParsedAttribute {
            name: "dimension".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if operator == "stablehlo.compare"
        && let Some(direction) = remainder.split(',').next().map(str::trim)
        && !direction.is_empty()
    {
        attributes.push(ParsedAttribute {
            name: "comparison_direction".to_owned(),
            value: ParsedAttributeValue::String(direction.to_owned()),
        });
    }
    if matches!(operator, "arith.cmpi" | "arith.cmpf")
        && let Some(predicate) = remainder.split(',').next().map(str::trim)
        && !predicate.is_empty()
    {
        attributes.push(ParsedAttribute {
            name: "predicate".to_owned(),
            value: ParsedAttributeValue::String(predicate.trim_matches('"').to_owned()),
        });
    }
    if operator == "linalg.init_tensor"
        && let Some(sizes) = first_bracket_items(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "static_sizes".to_owned(),
            value: ParsedAttributeValue::Strings(sizes),
        });
    }
    if operator == "hal.interface.binding.subspan"
        && let Some(layout) = leading_symbol_reference(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "layout".to_owned(),
            value: ParsedAttributeValue::String(layout),
        });
    }
    if operator == "flow.dispatch"
        && let Some(entry_points) = leading_symbol_reference(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "entry_points".to_owned(),
            value: ParsedAttributeValue::String(entry_points),
        });
    }
    if is_workgroup_dimension_op(operator)
        && let Some(dimension) =
            first_bracket_items(remainder).and_then(|items| items.first().cloned())
    {
        attributes.push(ParsedAttribute {
            name: "dimension".to_owned(),
            value: ParsedAttributeValue::String(dimension),
        });
    }
    if operator == "affine.apply"
        && let Some(map) = affine_map_attribute(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "map".to_owned(),
            value: ParsedAttributeValue::String(map),
        });
    }
    if operator == "memref.subview" {
        attributes.extend(parse_memref_subview_attributes(remainder));
    }
    if is_scalar_literal_attribute_op(operator)
        && let Some(value) = leading_scalar_literal(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "value".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if operator == "vm.const.ref.rodata"
        && let Some(rodata) = leading_symbol_reference(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "rodata".to_owned(),
            value: ParsedAttributeValue::String(rodata.trim_start_matches('@').to_owned()),
        });
    }
    if matches!(
        operator,
        "spv.Load" | "spirv.Load" | "spv.Store" | "spirv.Store"
    ) && let Some(storage_class) = leading_quoted_string(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "storage_class".to_owned(),
            value: ParsedAttributeValue::String(storage_class),
        });
    }
    if operator == "spv.mlir.addressof"
        && let Some(variable) = leading_symbol_reference(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "variable".to_owned(),
            value: ParsedAttributeValue::String(variable.trim_start_matches('@').to_owned()),
        });
    }
    if operator == "hal.allocator.allocate" {
        if let Some(value) = named_call_string(remainder, "type") {
            attributes.push(ParsedAttribute {
                name: "type".to_owned(),
                value: ParsedAttributeValue::String(value),
            });
        }
        if let Some(value) = named_call_string(remainder, "usage") {
            attributes.push(ParsedAttribute {
                name: "usage".to_owned(),
                value: ParsedAttributeValue::String(value),
            });
        }
    }
    if operator == "hal.command_buffer.create" {
        for name in ["mode", "categories", "affinity", "bindings"] {
            if let Some(value) = named_call_string(remainder, name) {
                attributes.push(ParsedAttribute {
                    name: name.to_owned(),
                    value: ParsedAttributeValue::String(value),
                });
            }
        }
    }
    if operator == "hal.command_buffer.execution_barrier" {
        for (name, attribute_name) in [
            ("flags", "flags"),
            ("source", "source_stage_mask"),
            ("target", "target_stage_mask"),
        ] {
            if let Some(value) = named_call_string(remainder, name) {
                attributes.push(ParsedAttribute {
                    name: attribute_name.to_owned(),
                    value: ParsedAttributeValue::String(value),
                });
            }
        }
    }
    if operator == "hal.command_buffer.dispatch.symbol" {
        if let Some(target) = named_call_string(remainder, "target") {
            attributes.push(ParsedAttribute {
                name: "target".to_owned(),
                value: if target.starts_with('@') {
                    ParsedAttributeValue::Reference(target)
                } else {
                    ParsedAttributeValue::String(target)
                },
            });
        }
        if let Some(workgroups) = named_bracket_value(remainder, "workgroups") {
            attributes.push(ParsedAttribute {
                name: "workgroups".to_owned(),
                value: ParsedAttributeValue::String(workgroups),
            });
        }
    }
    if operator == "hal.executable_layout.lookup"
        && let Some(layouts) = named_call_bracket_body(remainder, "layouts")
    {
        attributes.push(ParsedAttribute {
            name: "layouts".to_owned(),
            value: ParsedAttributeValue::String(layouts),
        });
    }
    if operator == "hal.device.query" {
        attributes.extend(parse_hal_device_query_attributes(remainder));
    }
    if matches!(operator, "spv.GlobalVariable" | "spirv.GlobalVariable") {
        attributes.extend(parse_spv_global_variable_attributes(remainder));
    }
    if matches!(operator, "spv.CompositeExtract" | "spirv.CompositeExtract")
        && let Some(indices) = first_bracket_items(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "indices".to_owned(),
            value: ParsedAttributeValue::Ints(
                indices
                    .into_iter()
                    .filter_map(|item| item.split_whitespace().next()?.parse::<i64>().ok())
                    .collect(),
            ),
        });
    }
    if operator == "hal.interface.binding" {
        attributes.extend(parse_hal_interface_binding_attributes(remainder));
    }
    if matches!(operator, "spv.EntryPoint" | "spirv.EntryPoint") {
        attributes.extend(parse_spv_entry_point_attributes(remainder));
    }
    if matches!(operator, "spv.ExecutionMode" | "spirv.ExecutionMode") {
        attributes.extend(parse_spv_execution_mode_attributes(remainder));
    }
    if operator == "arith.truncf"
        && let Some(mode) = parse_rounding_mode(remainder)
    {
        attributes.push(ParsedAttribute {
            name: "roundingmode".to_owned(),
            value: ParsedAttributeValue::String(mode),
        });
    }
    if let Some(flags) = angle_argument(remainder, "overflow") {
        attributes.push(ParsedAttribute {
            name: "overflowFlags".to_owned(),
            value: ParsedAttributeValue::String(normalize_comma_list(&flags)),
        });
    }
    if let Some(flags) = angle_argument(remainder, "fastmath") {
        attributes.push(ParsedAttribute {
            name: "fastmath".to_owned(),
            value: ParsedAttributeValue::String(normalize_comma_list(&flags)),
        });
    }
    if has_keyword_before_type(remainder, "exact") {
        attributes.push(ParsedAttribute {
            name: "isExact".to_owned(),
            value: ParsedAttributeValue::String(String::new()),
        });
    }
    if operator == "stablehlo.dot_general" {
        if remainder.contains("contracting_dims") {
            attributes.push(ParsedAttribute {
                name: "dot_dimension_numbers".to_owned(),
                value: ParsedAttributeValue::String("[object Object]".to_owned()),
            });
        }
        if let Some(value) = named_bracket_items(remainder, "precision") {
            attributes.push(ParsedAttribute {
                name: "precision_config".to_owned(),
                value: ParsedAttributeValue::Strings(value),
            });
        }
    }
    if has_constant_attribute(operator) && !has_dense_payload(remainder) {
        if !attributes.iter().any(|attribute| attribute.name == "value")
            && let Some(value) = remainder
                .split(':')
                .next()
                .and_then(|value| value.split_whitespace().next())
        {
            let type_text = constant_type(remainder);
            attributes.push(ParsedAttribute {
                name: "value".to_owned(),
                value: ParsedAttributeValue::String(normalize_scalar_literal(
                    value,
                    type_text.as_deref(),
                )),
            });
        }
    }
    if has_constant_attribute(operator) && has_dense_payload(remainder) {
        let value = if uses_tensor_dense_attribute(&operator) {
            ParsedAttributeValue::Tensor {
                type_text: constant_type(remainder),
                len: dense_literal_count(remainder),
            }
        } else {
            dense_literal(remainder).unwrap_or_else(|| ParsedAttributeValue::String(String::new()))
        };
        attributes.push(ParsedAttribute {
            name: "value".to_owned(),
            value,
        });
    }
    if operator == "torch.vtensor.literal" && has_dense_payload(remainder) {
        attributes.push(ParsedAttribute {
            name: "value".to_owned(),
            value: ParsedAttributeValue::Tensor {
                type_text: dense_payload_type(remainder).or_else(|| type_annotation(remainder)),
                len: dense_literal_count(remainder),
            },
        });
    }
    if (operator == "vm.rodata" || operator.ends_with(".rodata")) && has_dense_payload(remainder) {
        attributes.push(ParsedAttribute {
            name: "value".to_owned(),
            value: ParsedAttributeValue::Tensor {
                type_text: dense_payload_type(remainder).or_else(|| type_annotation(remainder)),
                len: dense_literal_count(remainder),
            },
        });
    }
    if operator == "tt.get_program_id"
        && let Some(axis) = remainder
            .split_once(':')
            .map(|(prefix, _)| prefix)
            .unwrap_or(remainder)
            .split_whitespace()
            .next()
        && matches!(axis, "x" | "y" | "z")
    {
        attributes.push(ParsedAttribute {
            name: "axis".to_owned(),
            value: ParsedAttributeValue::String(axis.to_owned()),
        });
    }
    attributes
}

fn parse_symbol_attributes(operator: &str, remainder: &str) -> Vec<ParsedAttribute> {
    let vm_global_decl = operator.starts_with("vm.global.")
        && !operator.starts_with("vm.global.load.")
        && !operator.starts_with("vm.global.store.");
    if !matches!(
        operator,
        "flow.executable"
            | "hal.executable"
            | "hal.interface"
            | "hal.executable.variant"
            | "hal.executable.entry_point"
            | "vm.rodata"
            | "vm.import"
    ) && !vm_global_decl
    {
        return Vec::new();
    }
    let prefix = remainder
        .split_once('=')
        .map(|(prefix, _)| prefix)
        .unwrap_or(remainder);
    let prefix = prefix
        .split_once('{')
        .map(|(prefix, _)| prefix)
        .unwrap_or(prefix);
    let mut attributes = Vec::new();
    if let Some(sym_visibility) = prefix.split_whitespace().find_map(|token| {
        matches!(token, "public" | "private" | "nested").then(|| token.to_owned())
    }) {
        attributes.push(ParsedAttribute {
            name: "sym_visibility".to_owned(),
            value: ParsedAttributeValue::String(sym_visibility),
        });
    }
    if let Some(sym_name) = prefix
        .split_whitespace()
        .find_map(|token| token.strip_prefix('@'))
        .map(|name| {
            name.split_once('(')
                .map(|(name, _)| name)
                .unwrap_or(name)
                .trim_end_matches(',')
                .to_owned()
        })
    {
        attributes.push(ParsedAttribute {
            name: "sym_name".to_owned(),
            value: ParsedAttributeValue::String(sym_name),
        });
    }
    attributes
}

fn hal_executable_variant_target(remainder: &str) -> Option<String> {
    let value = named_assignment_value(remainder, "target")?;
    if let Some(open) = value.find('<')
        && let Some(close) = matching_angle_delimiter(&value, open)
    {
        return Some(value[..=close].to_owned());
    }
    Some(value.trim_end_matches('{').trim().to_owned())
}

fn parse_vm_import_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let Some(signature) = vm_import_function_type(remainder) else {
        return Vec::new();
    };
    vec![ParsedAttribute {
        name: "function_type".to_owned(),
        value: ParsedAttributeValue::String(signature),
    }]
}

fn vm_import_function_type(remainder: &str) -> Option<String> {
    let at = remainder.find('@')?;
    let open = remainder[at..].find('(').map(|index| at + index)?;
    let close = matching_delimiter(remainder, open, '(', ')')?;
    let inputs = split_top_level(&remainder[open + 1..close], ',')
        .into_iter()
        .filter_map(vm_import_argument_type)
        .collect::<Vec<_>>();
    let after = remainder[close + 1..].trim_start();
    let outputs = if let Some(rest) = after.strip_prefix("->") {
        let end = rest
            .find(" attributes ")
            .or_else(|| rest.find('{'))
            .unwrap_or(rest.len());
        let output = rest[..end].trim();
        if output.is_empty() {
            "()".to_owned()
        } else {
            clean_type_list(output).to_owned()
        }
    } else {
        "()".to_owned()
    };
    Some(format!("({}) -> {outputs}", inputs.join(", ")))
}

fn vm_import_argument_type(argument: &str) -> Option<String> {
    let argument = argument.trim();
    if argument.is_empty() {
        return None;
    }
    let type_text = argument
        .split_once(':')
        .map(|(_, type_text)| type_text)
        .unwrap_or(argument);
    Some(clean_type_token(type_text.trim()).to_owned())
}

fn parse_vm_export_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let Some(function_ref) = leading_symbol_reference(remainder) else {
        return Vec::new();
    };
    let function_ref = function_ref.trim_start_matches('@').to_owned();
    let export_name = named_call_string(remainder, "as").unwrap_or_else(|| function_ref.clone());
    vec![
        ParsedAttribute {
            name: "export_name".to_owned(),
            value: ParsedAttributeValue::String(export_name),
        },
        ParsedAttribute {
            name: "function_ref".to_owned(),
            value: ParsedAttributeValue::String(function_ref),
        },
    ]
}

fn vm_global_type_attribute(operator: &str, remainder: &str) -> Option<String> {
    if operator == "vm.global.ref" {
        return Some("[object Object]".to_owned());
    }
    if let Some(type_text) = type_annotation(remainder)
        && !type_text.is_empty()
    {
        return Some(type_text);
    }
    operator
        .strip_prefix("vm.global.")
        .and_then(|suffix| suffix.split('.').next())
        .filter(|suffix| !suffix.is_empty())
        .map(str::to_owned)
}

fn parse_global_symbol(remainder: &str) -> Option<String> {
    let token = remainder
        .split_whitespace()
        .find(|token| token.starts_with('@'))?;
    Some(
        token
            .trim_start_matches('@')
            .trim_end_matches(',')
            .to_owned(),
    )
}

fn parse_torch_symbolic_int_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut min_val = None;
    let mut max_val = None;
    if let Some(dict) = extract_attribute_dict(remainder) {
        for attribute in parse_attribute_dict(dict) {
            let value = symbolic_int_attribute_text(attribute.value);
            match attribute.name.as_str() {
                "min_val" => min_val = Some(value),
                "max_val" => max_val = Some(value),
                _ => {}
            }
        }
    }
    let symbol_name = first_quoted_string(remainder);

    let mut attributes = Vec::new();
    if let Some(value) = max_val {
        attributes.push(ParsedAttribute {
            name: "max_val".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if let Some(value) = min_val {
        attributes.push(ParsedAttribute {
            name: "min_val".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if let Some(value) = symbol_name {
        attributes.push(ParsedAttribute {
            name: "symbol_name".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    attributes
}

fn symbolic_int_attribute_text(value: ParsedAttributeValue) -> String {
    match value {
        ParsedAttributeValue::Int(value) => js_integer_string(value),
        ParsedAttributeValue::Float(value) => value.to_string(),
        ParsedAttributeValue::String(value) | ParsedAttributeValue::Reference(value) => value,
        ParsedAttributeValue::Ints(values) => values
            .into_iter()
            .map(js_integer_string)
            .collect::<Vec<_>>()
            .join(", "),
        ParsedAttributeValue::Strings(values) => values.join(", "),
        ParsedAttributeValue::Tensor { .. } => "[object Object]".to_owned(),
    }
}

fn js_integer_string(value: i64) -> String {
    if (value as f64).abs() > 9_007_199_254_740_991.0 {
        (value as f64).to_string()
    } else {
        value.to_string()
    }
}

fn first_quoted_string(text: &str) -> Option<String> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

fn leading_quoted_string(text: &str) -> Option<String> {
    let text = text.trim_start();
    if !text.starts_with('"') {
        return None;
    }
    let end = text[1..].find('"')?;
    Some(text[1..1 + end].to_owned())
}

fn parse_util_global_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut initial_value = None;
    let mut type_text = None;
    if let Some((_, value)) = remainder.split_once('=') {
        let type_index = first_top_level_type_separator(value);
        let initial = type_index
            .map(|index| &value[..index])
            .unwrap_or(value)
            .trim();
        if !initial.is_empty() {
            initial_value = Some(strip_type_annotation(initial).to_owned());
        }
        if let Some(index) = type_index {
            let text = clean_type_token(&value[index + 1..]);
            if !text.is_empty() {
                type_text = Some(text.to_owned());
            }
        }
    } else if let Some(index) = first_top_level_type_separator(remainder) {
        let text = clean_type_token(&remainder[index + 1..]);
        if !text.is_empty() {
            type_text = Some(text.to_owned());
        }
    }

    let prefix = remainder
        .split_once('=')
        .map(|(prefix, _)| prefix)
        .unwrap_or(remainder);
    let sym_visibility = prefix.split_whitespace().find_map(|token| {
        matches!(token, "public" | "private" | "nested").then(|| token.to_owned())
    });
    let sym_name = prefix
        .split_whitespace()
        .find_map(|token| token.strip_prefix('@'))
        .map(|name| name.trim_end_matches(',').to_owned());

    let mut attributes = Vec::new();
    if let Some(value) = initial_value {
        attributes.push(ParsedAttribute {
            name: "initial_value".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if let Some(value) = sym_name {
        attributes.push(ParsedAttribute {
            name: "sym_name".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if let Some(value) = sym_visibility {
        attributes.push(ParsedAttribute {
            name: "sym_visibility".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if let Some(value) = type_text {
        attributes.push(ParsedAttribute {
            name: "type".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    attributes
}

fn parse_convolution_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    if let Some(value) = named_assignment_value(remainder, "dim_numbers") {
        attributes.push(ParsedAttribute {
            name: "dimension_numbers".to_owned(),
            value: ParsedAttributeValue::String(value),
        });
    }
    if let Some(window) = named_brace_value(remainder, "window") {
        for attribute in parse_attribute_dict(&window) {
            match attribute.name.as_str() {
                "stride" => attributes.push(ParsedAttribute {
                    name: "window_strides".to_owned(),
                    value: int_list_attribute(attribute.value),
                }),
                "rhs_dilate" => attributes.push(ParsedAttribute {
                    name: "rhs_dilation".to_owned(),
                    value: int_list_attribute(attribute.value),
                }),
                "pad" => attributes.push(ParsedAttribute {
                    name: "padding".to_owned(),
                    value: padding_attribute(attribute.value),
                }),
                _ => {}
            }
        }
    }
    for dict in brace_dicts_before_type(remainder) {
        for attribute in parse_attribute_dict(dict) {
            if matches!(
                attribute.name.as_str(),
                "batch_group_count" | "feature_group_count"
            ) {
                attributes.push(attribute);
            }
        }
    }
    attributes
}

fn int_list_attribute(value: ParsedAttributeValue) -> ParsedAttributeValue {
    match value {
        ParsedAttributeValue::String(value) => ParsedAttributeValue::Ints(
            split_top_level(value.trim().trim_matches(['[', ']']), ',')
                .into_iter()
                .filter_map(|item| item.trim().parse::<i64>().ok())
                .collect(),
        ),
        value => value,
    }
}

fn padding_attribute(value: ParsedAttributeValue) -> ParsedAttributeValue {
    let ParsedAttributeValue::String(value) = value else {
        return value;
    };
    let mut body = value.trim();
    if body.starts_with("[[") && body.ends_with("]]") {
        body = &body[1..body.len() - 1];
    }
    ParsedAttributeValue::Strings(
        split_top_level(body, ',')
            .into_iter()
            .map(|item| {
                item.trim()
                    .trim_matches(['[', ']'])
                    .split_whitespace()
                    .collect::<String>()
            })
            .filter(|item| !item.is_empty())
            .collect(),
    )
}

fn stringify_constant_value_attributes(attributes: &mut [ParsedAttribute]) {
    stringify_named_attributes(attributes, &["value"]);
}

fn stringify_named_attributes(attributes: &mut [ParsedAttribute], names: &[&str]) {
    for attribute in attributes {
        if !names.contains(&attribute.name.as_str()) {
            continue;
        }
        let value = match &attribute.value {
            ParsedAttributeValue::Int(value) => Some(value.to_string()),
            ParsedAttributeValue::Float(value) => Some(value.to_string()),
            _ => None,
        };
        if let Some(value) = value {
            attribute.value = ParsedAttributeValue::String(value);
        }
    }
}

fn normalize_operator_name(operator: String) -> String {
    if operator == "torch.constant.device" {
        return "torch.constant".to_owned();
    }
    let Some(rest) = operator.strip_prefix("torch.") else {
        return operator;
    };
    if let Some(after_aten) = rest.strip_prefix("aten.") {
        let mut parts = after_aten.split('.');
        let Some(name) = parts.next() else {
            return rest.to_owned();
        };
        return format!("aten.{name}");
    }
    if rest.starts_with("prim.") || rest.starts_with("prims.") {
        return rest.to_owned();
    }
    operator
}

fn uses_tensor_dense_attribute(operator: &str) -> bool {
    matches!(
        operator,
        "tosa.const"
            | "stablehlo.constant"
            | "arith.constant"
            | "mhlo.constant"
            | "util.unfoldable_constant"
            | "torch.constant.tensor"
            | "vhlo.constant_v1"
    )
}

fn has_constant_attribute(operator: &str) -> bool {
    operator == "constant"
        || operator.ends_with(".constant")
        || operator == "util.unfoldable_constant"
}

fn angle_argument(text: &str, name: &str) -> Option<String> {
    let start = text.find(&format!("{name}<"))? + name.len();
    let end = matching_delimiter(text, start, '<', '>')?;
    Some(text[start + 1..end].trim().to_owned())
}

fn first_angle_argument(text: &str) -> Option<String> {
    let open = text.find('<')?;
    let end = matching_delimiter(text, open, '<', '>')?;
    Some(text[open + 1..end].trim().to_owned())
}

fn normalize_comma_list(value: &str) -> String {
    split_top_level(value, ',')
        .into_iter()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

fn reorder_arm_sme_tile_slice_inputs(inputs: &mut Vec<String>, index_count: usize) {
    if inputs.len() < 1 + index_count + 2 {
        return;
    }
    let base = inputs[0].clone();
    let indices = inputs[1..1 + index_count].to_vec();
    let rest = inputs[1 + index_count..].to_vec();
    let mut reordered = Vec::with_capacity(inputs.len());
    reordered.push(base);
    reordered.extend(rest.iter().take(2).cloned());
    reordered.extend(indices);
    reordered.extend(rest.into_iter().skip(2));
    *inputs = reordered;
}

fn call_argument_count(text: &str, name: &str) -> usize {
    let Some(index) = text.find(&format!("{name}(")) else {
        return 0;
    };
    let open = index + name.len();
    let Some(close) = matching_delimiter(text, open, '(', ')') else {
        return 0;
    };
    extract_value_names(&text[open + 1..close]).len()
}

fn is_arm_sme_product(operator: &str) -> bool {
    operator == "arm_sme.outerproduct"
        || (operator.starts_with("arm_sme.")
            && (operator.contains("mopa_") || operator.contains("mops_")))
}

fn bracket_value_count(text: &str) -> Option<usize> {
    let open = text.find('[')?;
    let close = matching_delimiter(text, open, '[', ']')?;
    Some(extract_value_names(&text[open + 1..close]).len())
}

fn first_bracket_items(text: &str) -> Option<Vec<String>> {
    let open = text.find('[')?;
    let close = matching_delimiter(text, open, '[', ']')?;
    Some(
        split_top_level(&text[open + 1..close], ',')
            .into_iter()
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_owned)
            .collect(),
    )
}

fn leading_symbol_reference(text: &str) -> Option<String> {
    let text = text.trim_start();
    if !text.starts_with('@') {
        return None;
    }
    let end = text
        .find(|ch: char| ch.is_whitespace() || matches!(ch, '[' | '(' | '{'))
        .unwrap_or(text.len());
    Some(text[..end].trim_end_matches(',').to_owned())
}

fn parse_flow_dispatch_operand_segments(remainder: &str) -> Vec<i64> {
    vec![
        bracket_value_count(remainder).unwrap_or(0) as i64,
        first_paren_value_count_after_bracket(remainder) as i64,
        0,
        0,
    ]
}

fn first_paren_value_count_after_bracket(text: &str) -> usize {
    let offset = text
        .find('[')
        .and_then(|open| matching_delimiter(text, open, '[', ']').map(|close| close + 1))
        .unwrap_or(0);
    let Some(open) = text[offset..].find('(').map(|index| offset + index) else {
        return 0;
    };
    let Some(close) = matching_delimiter(text, open, '(', ')') else {
        return 0;
    };
    extract_value_names(&text[open + 1..close]).len()
}

fn named_call_value_count(text: &str, name: &str) -> Option<usize> {
    let marker = format!("{name}(");
    let open = text.find(&marker)? + name.len();
    let close = matching_delimiter(text, open, '(', ')')?;
    Some(extract_value_names(&text[open + 1..close]).len())
}

fn named_call_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("{name}(");
    let open = text.find(&marker)? + name.len();
    let close = matching_delimiter(text, open, '(', ')')?;
    Some(text[open + 1..close].trim())
}

fn named_call_string(text: &str, name: &str) -> Option<String> {
    Some(named_call_body(text, name)?.trim_matches('"').to_owned())
}

fn named_call_bracket_body(text: &str, name: &str) -> Option<String> {
    let mut value = named_call_string(text, name)?;
    loop {
        let trimmed = value.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            value = trimmed[1..trimmed.len() - 1].trim().to_owned();
        } else {
            return Some(value);
        }
    }
}

fn parse_hal_device_query_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    let mut key_value = None;
    if let Some(key) = named_call_body(remainder, "key") {
        let parts = quoted_strings(&key);
        if let Some(category) = parts.first() {
            attributes.push(ParsedAttribute {
                name: "category".to_owned(),
                value: ParsedAttributeValue::String(category.clone()),
            });
        }
        key_value = parts.last().cloned();
    }
    if let Some((_, value)) = remainder.rsplit_once('=')
        && let Some(default) = value.split_whitespace().next()
        && matches!(default, "true" | "false")
    {
        attributes.push(ParsedAttribute {
            name: "default".to_owned(),
            value: ParsedAttributeValue::String(default.to_owned()),
        });
    }
    if let Some(key) = key_value {
        attributes.push(ParsedAttribute {
            name: "key".to_owned(),
            value: ParsedAttributeValue::String(key),
        });
    }
    attributes
}

fn quoted_strings(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('"') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('"') else {
            break;
        };
        values.push(rest[..end].to_owned());
        rest = &rest[end + 1..];
    }
    values
}

fn parse_spv_global_variable_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    if let Some(name) = leading_symbol_reference(remainder) {
        attributes.push(ParsedAttribute {
            name: "sym_name".to_owned(),
            value: ParsedAttributeValue::String(name.trim_start_matches('@').to_owned()),
        });
    }
    if let Some(built_in) = named_call_string(remainder, "built_in") {
        attributes.push(ParsedAttribute {
            name: "built_in".to_owned(),
            value: ParsedAttributeValue::String(built_in),
        });
    }
    if let Some(initializer) = named_call_symbol(remainder, "initializer") {
        attributes.push(ParsedAttribute {
            name: "initializer".to_owned(),
            value: ParsedAttributeValue::String(initializer),
        });
    }
    if let Some((binding, descriptor_set)) = spv_bind_attribute(remainder) {
        attributes.push(ParsedAttribute {
            name: "binding".to_owned(),
            value: ParsedAttributeValue::Int(binding),
        });
        attributes.push(ParsedAttribute {
            name: "descriptor_set".to_owned(),
            value: ParsedAttributeValue::Int(descriptor_set),
        });
    }
    attributes
}

fn parse_hal_interface_binding_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    let prefix = remainder
        .split_once(',')
        .map(|(prefix, _)| prefix)
        .unwrap_or(remainder);
    if let Some(visibility) = prefix
        .split_whitespace()
        .find(|token| matches!(token, &"public" | &"private" | &"nested"))
    {
        attributes.push(ParsedAttribute {
            name: "sym_visibility".to_owned(),
            value: ParsedAttributeValue::String((*visibility).to_owned()),
        });
    }
    if let Some(symbol) = leading_symbol_reference(
        prefix
            .split_whitespace()
            .find(|token| token.starts_with('@'))
            .unwrap_or_default(),
    ) {
        attributes.push(ParsedAttribute {
            name: "sym_name".to_owned(),
            value: ParsedAttributeValue::String(symbol.trim_start_matches('@').to_owned()),
        });
    }
    for item in split_top_level(
        remainder
            .split_once(',')
            .map(|(_, suffix)| suffix)
            .unwrap_or_default(),
        ',',
    ) {
        let Some((name, value)) = item.split_once('=') else {
            continue;
        };
        attributes.push(ParsedAttribute {
            name: name.trim().to_owned(),
            value: parse_attribute_value(value.trim()),
        });
    }
    attributes
}

fn parse_spv_entry_point_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    if let Some(model) = leading_quoted_string(remainder) {
        attributes.push(ParsedAttribute {
            name: "execution_model".to_owned(),
            value: ParsedAttributeValue::String(model),
        });
    }
    if let Some(function) = symbol_references_after_prefix(remainder).last().cloned() {
        attributes.push(ParsedAttribute {
            name: "fn".to_owned(),
            value: ParsedAttributeValue::Reference(function),
        });
    }
    attributes
}

fn parse_spv_execution_mode_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    if let Some(function) = leading_symbol_reference(remainder) {
        attributes.push(ParsedAttribute {
            name: "fn".to_owned(),
            value: ParsedAttributeValue::Reference(function),
        });
    }
    if let Some(mode) = quoted_string_after_symbol(remainder) {
        attributes.push(ParsedAttribute {
            name: "execution_mode".to_owned(),
            value: ParsedAttributeValue::String(mode),
        });
    }
    let values = spv_execution_mode_values(remainder);
    if !values.is_empty() {
        attributes.push(ParsedAttribute {
            name: "values".to_owned(),
            value: ParsedAttributeValue::Strings(values),
        });
    }
    attributes
}

fn named_call_symbol(text: &str, name: &str) -> Option<String> {
    let value = named_call_string(text, name)?;
    Some(value.trim_start_matches('@').to_owned())
}

fn spv_bind_attribute(text: &str) -> Option<(i64, i64)> {
    let marker = "bind(";
    let open = text.find(marker)? + marker.len() - 1;
    let close = matching_delimiter(text, open, '(', ')')?;
    let items = split_top_level(&text[open + 1..close], ',')
        .into_iter()
        .filter_map(|item| item.trim().parse::<i64>().ok())
        .collect::<Vec<_>>();
    (items.len() >= 2).then(|| (items[0], items[1]))
}

fn symbol_references_after_prefix(text: &str) -> Vec<String> {
    let mut after = text.trim_start();
    if after.starts_with('"')
        && let Some(end) = after[1..].find('"')
    {
        after = after[1 + end + 1..].trim_start();
    }
    after
        .split(',')
        .filter_map(|part| leading_symbol_reference(part.trim()))
        .collect()
}

fn quoted_string_after_symbol(text: &str) -> Option<String> {
    let symbol = leading_symbol_reference(text)?;
    let start = text.find(&symbol)? + symbol.len();
    leading_quoted_string(&text[start..])
}

fn spv_execution_mode_values(text: &str) -> Vec<String> {
    let Some(mode) = quoted_string_after_symbol(text) else {
        return Vec::new();
    };
    let Some(start) = text
        .find(&format!("\"{mode}\""))
        .map(|index| index + mode.len() + 2)
    else {
        return Vec::new();
    };
    text[start..]
        .split(',')
        .skip(1)
        .filter_map(|item| {
            let token = item.split_whitespace().next()?.trim();
            (!token.is_empty()).then(|| token.trim_matches('"').to_owned())
        })
        .collect()
}

fn is_workgroup_dimension_op(operator: &str) -> bool {
    matches!(
        operator,
        "flow.dispatch.workgroup.size"
            | "flow.dispatch.workgroup.id"
            | "flow.dispatch.workgroup.count"
            | "hal.interface.workgroup.size"
            | "hal.interface.workgroup.id"
            | "hal.interface.workgroup.count"
    )
}

fn subspan_dynamic_dim_inputs(remainder: &str) -> Vec<String> {
    let Some(type_index) = first_top_level_type_separator(remainder) else {
        return Vec::new();
    };
    let suffix = &remainder[type_index + 1..];
    let Some(open) = suffix.find('{') else {
        return Vec::new();
    };
    let Some(close) = matching_delimiter(suffix, open, '{', '}') else {
        return Vec::new();
    };
    extract_value_names(&suffix[open + 1..close])
}

fn parse_memref_subview_inputs(remainder: &str) -> Vec<String> {
    let source = remainder
        .find('[')
        .map(|index| extract_value_names(&remainder[..index]))
        .unwrap_or_default();
    let mut inputs = memref_subview_bracket_groups(remainder)
        .into_iter()
        .flat_map(|group| extract_value_names(group))
        .collect::<Vec<_>>();
    inputs.extend(source);
    inputs
}

fn parse_memref_subview_attributes(remainder: &str) -> Vec<ParsedAttribute> {
    let groups = memref_subview_bracket_groups(remainder);
    if groups.len() < 3 {
        return Vec::new();
    }
    let dynamic_counts = groups
        .iter()
        .map(|group| extract_value_names(group).len() as i64)
        .collect::<Vec<_>>();
    vec![
        ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![
                dynamic_counts[0],
                dynamic_counts[1],
                dynamic_counts[2],
                0,
            ]),
        },
        ParsedAttribute {
            name: "static_offsets".to_owned(),
            value: ParsedAttributeValue::Strings(static_subview_items(groups[0])),
        },
        ParsedAttribute {
            name: "static_sizes".to_owned(),
            value: subview_static_ints_or_strings(groups[1]),
        },
        ParsedAttribute {
            name: "static_strides".to_owned(),
            value: subview_static_ints_or_strings(groups[2]),
        },
    ]
}

fn memref_subview_bracket_groups(remainder: &str) -> Vec<&str> {
    let end = first_top_level_type_separator(remainder).unwrap_or(remainder.len());
    let mut groups = Vec::new();
    let mut index = 0usize;
    while let Some(relative) = remainder[index..end].find('[') {
        let open = index + relative;
        let Some(close) = matching_delimiter(remainder, open, '[', ']') else {
            break;
        };
        groups.push(remainder[open + 1..close].trim());
        index = close + 1;
    }
    groups
}

fn static_subview_items(group: &str) -> Vec<String> {
    split_top_level(group, ',')
        .into_iter()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            if item.starts_with('%') {
                i64::MIN.to_string()
            } else {
                item.to_owned()
            }
        })
        .collect()
}

fn subview_static_ints_or_strings(group: &str) -> ParsedAttributeValue {
    let items = static_subview_items(group);
    let ints = items
        .iter()
        .map(|item| item.parse::<i64>().ok())
        .collect::<Vec<_>>();
    if ints.iter().all(Option::is_some) {
        ParsedAttributeValue::Ints(ints.into_iter().map(|item| item.unwrap_or(0)).collect())
    } else {
        ParsedAttributeValue::Strings(items)
    }
}

fn affine_map_attribute(remainder: &str) -> Option<String> {
    let index = remainder.find("affine_map<")?;
    let open = index + "affine_map".len();
    let close = matching_angle_delimiter(remainder, open)?;
    Some(normalize_affine_map_attribute(&format!(
        "affine_map<{}>",
        &remainder[open + 1..close]
    )))
}

fn matching_angle_delimiter(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0isize;
    let mut quote = false;
    let mut previous = None;
    for (index, ch) in text.char_indices().filter(|(index, _)| *index >= open) {
        if quote {
            if ch == '"' {
                quote = false;
            }
            previous = Some(ch);
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => depth += 1,
            '>' if previous != Some('-') => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        previous = Some(ch);
    }
    None
}

fn normalize_affine_map_attribute(value: &str) -> String {
    let Some((prefix, suffix)) = value.split_once("->") else {
        return value.to_owned();
    };
    format!("{prefix}->{}", strip_affine_expr_ids(suffix))
}

fn strip_affine_expr_ids(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut output = String::with_capacity(value.len());
    while index < bytes.len() {
        let ch = bytes[index] as char;
        if matches!(ch, 's' | 'd')
            && bytes
                .get(index + 1)
                .is_some_and(|value| (*value as char).is_ascii_digit())
        {
            index += 2;
            while index < bytes.len() && (bytes[index] as char).is_ascii_digit() {
                index += 1;
            }
            continue;
        }
        output.push(ch);
        index += 1;
    }
    output
}

fn is_scalar_literal_attribute_op(operator: &str) -> bool {
    operator == "spv.Constant" || operator.starts_with("vm.const.")
}

fn leading_scalar_literal(remainder: &str) -> Option<String> {
    let end = first_top_level_type_separator(remainder).unwrap_or(remainder.len());
    let token = remainder[..end]
        .split_whitespace()
        .next()?
        .trim_end_matches(',');
    if token.is_empty() || token.starts_with(['%', '@', '[', '(', '{', '<']) || token.contains("::")
    {
        return None;
    }
    Some(normalize_scalar_literal(
        token,
        type_annotation(remainder).as_deref(),
    ))
}

fn dense_literal_count(text: &str) -> usize {
    let Some(dense) = text.find("dense<") else {
        return 0;
    };
    let open = dense + "dense".len();
    let Some(end) = matching_delimiter(text, open, '<', '>') else {
        return 0;
    };
    let body = &text[open + 1..end];
    if body.trim().is_empty() {
        return 0;
    }
    if !body.contains('[') {
        return 1;
    }
    let mut values = Vec::new();
    let mut token = String::new();
    for ch in body.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '+' | '.' | 'e' | 'E') {
            token.push(ch);
        } else if !token.is_empty() {
            values.push(token.clone());
            token.clear();
        }
    }
    values.len()
}

fn dense_literal(text: &str) -> Option<ParsedAttributeValue> {
    let dense = text.find("dense<")?;
    let open = dense + "dense".len();
    let end = matching_delimiter(text, open, '<', '>')?;
    let body = &text[open + 1..end];
    let mut values = Vec::new();
    let mut token = String::new();
    for ch in body.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E') {
            token.push(ch);
        } else if !token.is_empty() {
            if let Ok(value) = token.parse::<f64>() {
                values.push(value);
            }
            token.clear();
        }
    }
    if values.is_empty() {
        return Some(ParsedAttributeValue::String(body.to_owned()));
    }
    if values.iter().all(|value| value.fract() == 0.0) {
        return Some(ParsedAttributeValue::Ints(
            values.into_iter().map(|value| value as i64).collect(),
        ));
    }
    Some(ParsedAttributeValue::String(
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    ))
}

fn parse_call_callee(remainder: &str) -> Option<String> {
    remainder
        .split_whitespace()
        .find(|part| part.starts_with('@'))
        .map(|part| {
            part.split_once('(')
                .map(|(callee, _)| callee)
                .unwrap_or(part)
                .trim_end_matches(',')
                .to_owned()
        })
}

fn parse_location_reference(text: &str) -> Option<String> {
    let index = text.rfind(" loc(")? + 1;
    let open = index + "loc".len();
    let close = matching_delimiter(text, open, '(', ')')?;
    Some(text[open + 1..close].trim().to_owned())
}

fn parse_affine_for_attrs(remainder: &str) -> Vec<ParsedAttribute> {
    let mut attributes = Vec::new();
    if let Some(after_eq) = remainder.split_once('=').map(|(_, value)| value.trim()) {
        let mut parts = after_eq.split_whitespace();
        let lower = parts.next().unwrap_or("0");
        let _to = parts.next();
        let upper = parts.next().unwrap_or("0");
        let step = after_eq
            .split(" step ")
            .nth(1)
            .and_then(|value| value.split_whitespace().next())
            .unwrap_or("1");
        attributes.push(ParsedAttribute {
            name: "lowerBoundMap".to_owned(),
            value: ParsedAttributeValue::String(format!("affine_map<() -> ({lower})>")),
        });
        attributes.push(ParsedAttribute {
            name: "upperBoundMap".to_owned(),
            value: ParsedAttributeValue::String(format!("affine_map<() -> ({upper})>")),
        });
        attributes.push(ParsedAttribute {
            name: "step".to_owned(),
            value: ParsedAttributeValue::String(step.to_owned()),
        });
        attributes.push(ParsedAttribute {
            name: "operandSegmentSizes".to_owned(),
            value: ParsedAttributeValue::Ints(vec![0, 0, 0]),
        });
    }
    attributes
}

fn parse_attribute_dict(text: &str) -> Vec<ParsedAttribute> {
    split_top_level(text, ',')
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.trim();
            if entry.is_empty() {
                return None;
            }
            let (name, value) = entry
                .split_once('=')
                .map(|(name, value)| (name.trim(), value.trim()))
                .unwrap_or((entry, ""));
            let name = normalize_attr_name(name);
            let value = if name == "qtype" && value.trim_start().starts_with("tensor<") {
                ParsedAttributeValue::String("[object Object]".to_owned())
            } else if has_dense_payload(value) {
                ParsedAttributeValue::Tensor {
                    type_text: type_annotation(value),
                    len: dense_literal_count(value),
                }
            } else {
                parse_attribute_value(value)
            };
            Some(ParsedAttribute { name, value })
        })
        .collect()
}

fn parse_metadata_dict(text: &str) -> BTreeMap<String, String> {
    split_top_level(text, ',')
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.trim();
            if entry.is_empty() {
                return None;
            }
            let (name, value) = entry
                .split_once('=')
                .map(|(name, value)| (name.trim(), value.trim()))
                .unwrap_or((entry, ""));
            let name = normalize_attr_name(name);
            let value = if value.trim_start().starts_with('[') && value.trim_end().ends_with(']') {
                metadata_list_text(value)
            } else {
                parse_attribute_value(value).metadata_text()
            };
            Some((name, value))
        })
        .collect()
}

impl ParsedAttributeValue {
    fn metadata_text(&self) -> String {
        match self {
            Self::String(value) | Self::Reference(value) => value.clone(),
            Self::Int(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
            Self::Tensor { .. } => "[object Object]".to_owned(),
            Self::Ints(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Strings(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(|value| json_string_literal(value))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

fn metadata_list_text(value: &str) -> String {
    let value = value.trim();
    let inner = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(value);
    let items = split_top_level(inner, ',')
        .into_iter()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| json_string_literal(&parse_attribute_value(item).metadata_text()))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}

fn json_string_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn parse_attribute_value(value: &str) -> ParsedAttributeValue {
    if first_top_level_type_separator(value).is_some() {
        return ParsedAttributeValue::String(
            strip_type_annotation(value.trim())
                .trim_matches('"')
                .to_owned(),
        );
    }
    let value = strip_type_annotation(value.trim());
    if value.is_empty() {
        return ParsedAttributeValue::String(String::new());
    }
    if value.starts_with('@') {
        return ParsedAttributeValue::Reference(value.to_owned());
    }
    if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
        return ParsedAttributeValue::String(unescape_mlir_string(&value[1..value.len() - 1]));
    }
    if value.starts_with('[') && value.ends_with(']') {
        let items = split_top_level(&value[1..value.len() - 1], ',')
            .into_iter()
            .map(|item| item.trim().trim_matches('"').to_owned())
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>();
        return ParsedAttributeValue::String(items.join(", "));
    }
    if let Some(value) = normalize_hex_integer(value) {
        return ParsedAttributeValue::String(value);
    }
    if let Ok(value) = value.parse::<i64>() {
        return ParsedAttributeValue::Int(value);
    }
    if let Ok(value) = value.parse::<f32>() {
        return ParsedAttributeValue::Float(value);
    }
    ParsedAttributeValue::String(value.to_owned())
}

fn unescape_mlir_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\\'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) = (
                hex_value(bytes[index + 1] as char),
                hex_value(bytes[index + 2] as char),
            )
        {
            output.push(char::from((high << 4) | low));
            index += 3;
            continue;
        }
        output.push(bytes[index] as char);
        index += 1;
    }
    output
}

fn hex_value(ch: char) -> Option<u8> {
    match ch {
        '0'..='9' => Some(ch as u8 - b'0'),
        'a'..='f' => Some(ch as u8 - b'a' + 10),
        'A'..='F' => Some(ch as u8 - b'A' + 10),
        _ => None,
    }
}

fn normalize_attr_name(name: &str) -> String {
    name.trim()
        .trim_matches('"')
        .trim_start_matches('#')
        .to_owned()
}

fn named_assignment_value(text: &str, name: &str) -> Option<String> {
    let index = text.find(name)?;
    let after = &text[index + name.len()..];
    let eq = after.find('=')?;
    let rest = after[eq + 1..].trim_start();
    let end = first_top_level_delimiter(rest, ',').unwrap_or(rest.len());
    Some(rest[..end].trim().to_owned())
}

fn named_brace_value(text: &str, name: &str) -> Option<String> {
    let index = text.find(name)?;
    let after = &text[index + name.len()..];
    let open = after.find('{')?;
    let close = matching_delimiter(after, open, '{', '}')?;
    Some(after[open + 1..close].trim().to_owned())
}

fn brace_dicts_before_type(text: &str) -> Vec<&str> {
    let end = first_top_level_type_separator(text).unwrap_or(text.len());
    let prefix = &text[..end];
    let mut result = Vec::new();
    let mut index = 0usize;
    while let Some(relative) = prefix[index..].find('{') {
        let open = index + relative;
        let Some(close) = matching_delimiter(prefix, open, '{', '}') else {
            break;
        };
        result.push(prefix[open + 1..close].trim());
        index = close + 1;
    }
    result
}

fn first_top_level_delimiter(text: &str, delimiter: char) -> Option<usize> {
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    for (index, ch) in text.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            value
                if value == delimiter && angle == 0 && paren == 0 && bracket == 0 && brace == 0 =>
            {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

fn named_bracket_value(text: &str, name: &str) -> Option<String> {
    let index = text.find(name)?;
    let after = &text[index + name.len()..];
    let eq = after.find('=')?;
    let rest = after[eq + 1..].trim_start();
    let open = rest.find('[')?;
    let close = matching_delimiter(rest, open, '[', ']')?;
    Some(rest[open..=close].to_owned())
}

fn named_bracket_items(text: &str, name: &str) -> Option<Vec<String>> {
    let value = named_bracket_value(text, name)?;
    Some(
        split_top_level(&value[1..value.len() - 1], ',')
            .into_iter()
            .map(|item| item.trim().trim_matches('"').to_owned())
            .filter(|item| !item.is_empty())
            .collect(),
    )
}

fn named_scalar_value(text: &str, name: &str) -> Option<String> {
    let index = text.find(name)?;
    let after = &text[index + name.len()..];
    let eq = after.find('=')?;
    let rest = after[eq + 1..].trim_start();
    let end = rest
        .find(|ch: char| ch == ',' || ch == ':' || ch.is_whitespace())
        .unwrap_or(rest.len());
    Some(rest[..end].trim().to_owned())
}

fn normalize_scalar_literal(value: &str, type_text: Option<&str>) -> String {
    let value = value.trim().trim_matches('"');
    if matches!(value, "true" | "false") {
        return value.to_owned();
    }
    if value.starts_with("0x") && type_text.is_some_and(is_scalar_float_type) {
        return value.to_owned();
    }
    if let Some(hex) = normalize_hex_integer(value) {
        return hex;
    }
    if let Ok(parsed) = value.parse::<f64>() {
        if parsed.is_finite() {
            if parsed == 0.0 {
                return "0".to_owned();
            }
            if let Some(scientific) = normalize_scientific_literal(value, parsed) {
                return scientific;
            }
            let abs = parsed.abs();
            if abs >= 1e21 || (abs > 0.0 && abs < 1e-6) {
                return format_js_scientific(parsed);
            }
            return parsed.to_string();
        }
    }
    value.to_owned()
}

fn is_scalar_float_type(type_text: &str) -> bool {
    matches!(
        type_text.trim(),
        "f16" | "f32" | "f64" | "f80" | "f128" | "bf16"
    )
}

fn normalize_hex_integer(value: &str) -> Option<String> {
    let (sign, digits) = value
        .strip_prefix("-0x")
        .map(|digits| (-1i128, digits))
        .or_else(|| value.strip_prefix("+0x").map(|digits| (1i128, digits)))
        .or_else(|| value.strip_prefix("0x").map(|digits| (1i128, digits)))?;
    let parsed = i128::from_str_radix(digits, 16).ok()?;
    Some((sign * parsed).to_string())
}

fn normalize_scientific_literal(value: &str, parsed: f64) -> Option<String> {
    let exponent_index = value.find(['e', 'E'])?;
    let abs = parsed.abs();
    if parsed == 0.0 || (abs >= 1e-6 && abs < 1e21) {
        return Some(parsed.to_string());
    }
    let (mantissa, exponent) = value.split_at(exponent_index);
    let exponent = exponent[1..].parse::<i32>().ok()?;
    let mantissa = trim_decimal_zeros(mantissa);
    let sign = if exponent >= 0 { "+" } else { "" };
    Some(format!("{mantissa}e{sign}{exponent}"))
}

fn format_js_scientific(value: f64) -> String {
    let formatted = format!("{value:e}");
    let Some((mantissa, exponent)) = formatted.split_once('e') else {
        return formatted;
    };
    let mantissa = trim_decimal_zeros(mantissa);
    let exponent = exponent.parse::<i32>().unwrap_or_default();
    let sign = if exponent >= 0 { "+" } else { "" };
    format!("{mantissa}e{sign}{exponent}")
}

fn trim_decimal_zeros(value: &str) -> String {
    let Some(dot) = value.find('.') else {
        return value.to_owned();
    };
    let mut end = value.len();
    while end > dot + 1 && value.as_bytes()[end - 1] == b'0' {
        end -= 1;
    }
    if end == dot + 1 {
        end -= 1;
    }
    value[..end].to_owned()
}

fn parse_rounding_mode(text: &str) -> Option<String> {
    let prefix = text
        .split_once(':')
        .map(|(prefix, _)| prefix)
        .unwrap_or(text);
    [
        "to_nearest_even",
        "toward_zero",
        "downward",
        "upward",
        "to_nearest_away",
    ]
    .into_iter()
    .find(|mode| prefix.split_whitespace().any(|token| token == *mode))
    .map(str::to_owned)
}

fn has_keyword_before_type(text: &str, keyword: &str) -> bool {
    let prefix = text
        .split_once(':')
        .map(|(prefix, _)| prefix)
        .unwrap_or(text);
    prefix.split_whitespace().any(|token| token == keyword)
}

fn is_dense_constant(operator: &str, rhs: &str) -> bool {
    matches!(
        operator,
        "stablehlo.constant" | "mhlo.constant" | "arith.constant" | "tosa.const"
    ) && has_dense_payload(rhs)
        && constant_type(rhs)
            .as_deref()
            .is_some_and(|type_text| type_text.starts_with("tensor<"))
}

fn has_dense_payload(text: &str) -> bool {
    text.contains("dense<") || text.contains("dense_resource<")
}

fn dense_payload_type(text: &str) -> Option<String> {
    let marker = text
        .find("dense_resource<")
        .map(|index| (index, "dense_resource".len()))
        .or_else(|| text.find("dense<").map(|index| (index, "dense".len())))?;
    let open = marker.0 + marker.1;
    let close = matching_delimiter(text, open, '<', '>')?;
    let rest = &text[close + 1..];
    let index = first_top_level_type_separator(rest)?;
    Some(clean_type_token(&rest[index + 1..]).to_owned())
}

fn constant_type(rhs: &str) -> Option<String> {
    type_annotation(rhs)
}

fn type_annotation(text: &str) -> Option<String> {
    let index = first_top_level_type_separator(text)?;
    Some(clean_type_token(text[index + 1..].trim()).to_owned())
}

fn result_types(operator: &str, rhs: &str, count: usize) -> Vec<String> {
    if count == 0 {
        return Vec::new();
    }
    if operator == "memref.dim" {
        return vec!["index".to_owned()];
    }
    if matches!(operator, "affine.apply" | "affine.min" | "affine.max") {
        return vec!["index".to_owned(); count];
    }
    if operator == "torch.constant" && rhs.contains("torch.constant.device") {
        return vec!["!torch.Device".to_owned()];
    }
    if matches!(operator, "arm_sme.tile_load" | "arm_sme.load_tile_slice")
        && let Some(index) = rhs.rfind(':')
    {
        let suffix = rhs[index + 1..].trim();
        let types = split_top_level(suffix, ',');
        if let Some(last) = types.last() {
            return vec![clean_type_token(last.trim()).to_owned()];
        }
    }
    if let Some(index) = rhs.rfind("->") {
        let mut types = split_type_list(&rhs[index + 2..], count);
        if operator == "stablehlo.convert" {
            for value in &mut types {
                if let Some(converted) = scalar_tensor_as_vector(value) {
                    *value = converted;
                }
            }
        }
        return types;
    }
    if let Some(index) = rhs.rfind(" to ") {
        return split_type_list(&rhs[index + 4..], count);
    }
    if let Some(index) = first_top_level_type_separator(rhs) {
        let suffix = rhs[index + 1..].trim();
        if suffix.starts_with('(') && suffix.contains("->") {
            if let Some(arrow) = suffix.rfind("->") {
                return split_type_list(&suffix[arrow + 2..], count);
            }
        }
        if operator == "tensor.extract"
            && let Some(element) = scalar_result_from_shaped_type(suffix)
        {
            return vec![element];
        }
        if operator.ends_with(".select") {
            let types = split_type_list(suffix, usize::MAX);
            if let Some(type_text) = types.last() {
                return vec![type_text.clone()];
            }
        }
        if operator == "tt.load"
            && let Some(type_text) = triton_load_result_type(clean_type_token(suffix))
        {
            return vec![type_text];
        }
        if count > 1 {
            return split_type_list(suffix, count);
        }
        return vec![clean_type_token(suffix).to_owned()];
    }
    if operator == "arith.constant"
        && matches!(rhs.split_whitespace().nth(1), Some("true" | "false"))
    {
        return vec!["i1".to_owned()];
    }
    Vec::new()
}

fn scalar_tensor_as_vector(type_text: &str) -> Option<String> {
    let inner = bracket_inner(type_text.trim(), "tensor<")?;
    (!inner.contains('x')).then(|| format!("tensor<1x{inner}>"))
}

fn triton_load_result_type(type_text: &str) -> Option<String> {
    let inner = bracket_inner(type_text.trim(), "tensor<")?;
    let parts = split_top_level(inner, ',');
    let shaped = parts.first()?.trim();
    let (shape, pointer) = shaped.rsplit_once('x')?;
    let element = bracket_inner(pointer.trim(), "!tt.ptr<")?;
    let layout = parts
        .get(1..)
        .filter(|parts| !parts.is_empty())
        .map(|parts| format!(", {}", parts.join(", ")))
        .unwrap_or_default();
    Some(format!("tensor<{shape}x{element}{layout}>"))
}

fn split_type_list(text: &str, count: usize) -> Vec<String> {
    let cleaned = clean_type_list(text.trim());
    let inner = if cleaned.starts_with('(') {
        matching_delimiter(cleaned, 0, '(', ')')
            .map(|close| &cleaned[1..close])
            .unwrap_or(cleaned)
    } else {
        cleaned
    };
    let mut values = split_top_level(inner, ',')
        .into_iter()
        .map(|item| clean_type_token(item.trim()).to_owned())
        .filter(|item| !item.is_empty() && item != "()")
        .collect::<Vec<_>>();
    if values.len() == 1 && count > 1 && count != usize::MAX {
        values.resize(count, values[0].clone());
    }
    values
}

fn clean_type_list(text: &str) -> &str {
    let text = text.trim();
    let mut end = text.len();
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    for (index, ch) in text.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => {
                if angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
                    end = index;
                    break;
                }
                brace += 1;
            }
            '}' => brace -= 1,
            _ => {}
        }
    }
    text[..end].trim().trim_end_matches(',')
}

fn clean_type_token(text: &str) -> &str {
    let text = text.trim();
    let mut end = text.len();
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    for (index, ch) in text.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => {
                if angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
                    end = index;
                    break;
                }
                brace += 1;
            }
            '}' => brace -= 1,
            ',' => {
                if angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
                    end = index;
                    break;
                }
            }
            value if value.is_whitespace() => {
                if angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
                    end = index;
                    break;
                }
            }
            _ => {}
        }
    }
    text[..end].trim().trim_end_matches(',')
}

fn parse_return(text: &str) -> Option<Vec<ParsedValue>> {
    let trimmed = text.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let token = parts.next().unwrap_or_default();
    if token != "return" && !token.ends_with(".return") {
        return None;
    }
    let rest = parts.next().unwrap_or_default();
    let operands = rest
        .split_once(':')
        .map(|(operands, _)| operands)
        .unwrap_or(rest);
    let names = extract_value_names(operands);
    let types = rest
        .split_once(':')
        .map(|(_, suffix)| split_type_list(suffix, names.len()))
        .unwrap_or_default();
    Some(
        names
            .into_iter()
            .enumerate()
            .map(|(index, name)| ParsedValue {
                name,
                type_text: types.get(index).cloned(),
            })
            .collect(),
    )
}

fn top_level_statements(lines: &[CapturedLine]) -> Vec<CapturedLine> {
    let mut statements = Vec::new();
    let mut current: Option<CapturedLine> = None;
    for line in lines {
        let text = line.text.trim();
        if let Some(active) = current.as_mut()
            && is_bindings_continuation(&active.text)
        {
            active.text.push(' ');
            active.text.push_str(text);
            continue;
        }
        if let Some(active) = current.as_mut()
            && is_attribute_continuation(&active.text, text)
        {
            active.text.push(' ');
            active.text.push_str(text);
            continue;
        }
        if line.local_depth > 0 {
            continue;
        }
        if text.is_empty() || text.starts_with('}') {
            continue;
        }
        if is_continuation(text) {
            if let Some(active) = current.as_mut() {
                active.text.push(' ');
                active.text.push_str(text);
            }
            continue;
        }
        if let Some(active) = current.take() {
            statements.push(active);
        }
        current = Some(line.clone());
    }
    if let Some(active) = current {
        statements.push(active);
    }
    statements
}

fn is_bindings_continuation(active: &str) -> bool {
    active.contains("bindings([") && !active.contains("])")
}

fn is_attribute_continuation(active: &str, text: &str) -> bool {
    let attribute_line = text.starts_with('}')
        || text.starts_with('(')
        || (text.contains('=') && !text.starts_with('%') && !text.starts_with('^'));
    attribute_line && (active.contains("linalg.") || active.contains("convolution"))
}

fn is_continuation(text: &str) -> bool {
    text.starts_with(':')
        || text.starts_with("->")
        || text.starts_with(',')
        || text.starts_with("tensor<")
        || text.starts_with("vector<")
        || text.starts_with("memref<")
}

fn parse_function_name(header: &str) -> Option<String> {
    if let Some(name) = function_sym_name(header) {
        return Some(name);
    }
    let at = header.find('@')?;
    let rest = &header[at + 1..];
    let end = rest
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '$' | '.')))
        .unwrap_or(rest.len());
    Some(rest[..end].to_owned())
}

fn function_sym_name(header: &str) -> Option<String> {
    extract_attribute_dict(header)
        .map(parse_attribute_dict)
        .into_iter()
        .flatten()
        .find_map(|attribute| (attribute.name == "sym_name").then_some(attribute))
        .map(|attribute| match attribute.value {
            ParsedAttributeValue::String(value) | ParsedAttributeValue::Reference(value) => value,
            _ => String::new(),
        })
        .filter(|value| !value.is_empty())
}

fn function_visibility(header: &str) -> Option<String> {
    let mut visibility = extract_attribute_dict(header)
        .and_then(|dict| {
            parse_attribute_dict(dict)
                .into_iter()
                .find_map(|attribute| (attribute.name == "sym_visibility").then_some(attribute))
        })
        .and_then(|attribute| match attribute.value {
            ParsedAttributeValue::String(value) => Some(value),
            _ => None,
        });
    if visibility.is_none() {
        visibility = header.split_whitespace().find_map(|token| {
            matches!(token, "public" | "private" | "nested").then(|| token.to_owned())
        });
    }
    visibility
}

fn parse_function_type(header: &str) -> Option<(Vec<String>, Vec<String>)> {
    let function_type = extract_attribute_dict(header)
        .map(|dict| {
            parse_attribute_dict(dict)
                .into_iter()
                .find_map(|attribute| (attribute.name == "function_type").then_some(attribute))
        })?
        .and_then(|attribute| match attribute.value {
            ParsedAttributeValue::String(value) => Some(value),
            ParsedAttributeValue::Reference(value) => Some(value),
            _ => None,
        })?;

    parse_function_signature_types(&function_type)
}

fn parse_function_signature_types(function_type: &str) -> Option<(Vec<String>, Vec<String>)> {
    let function_type = function_type.trim();
    let input_open = function_type.find('(')?;
    let input_close = matching_delimiter(function_type, input_open, '(', ')')?;
    let input_types = split_type_list(&function_type[input_open + 1..input_close], usize::MAX);

    let output_suffix = function_type
        .find("->")
        .map(|index| function_type[index + 2..].trim())?;
    let output_open = output_suffix.find('(')?;
    let output_close = matching_delimiter(output_suffix, output_open, '(', ')')?;
    let output_types = split_type_list(&output_suffix[output_open + 1..output_close], usize::MAX);
    Some((input_types, output_types))
}

fn parse_function_inputs(header: &str) -> Vec<ParsedValue> {
    let Some(name_at) = header.find('@') else {
        return Vec::new();
    };
    let Some(open) = header[name_at..].find('(').map(|index| index + name_at) else {
        return Vec::new();
    };
    let Some(close) = matching_delimiter(header, open, '(', ')') else {
        return Vec::new();
    };
    split_top_level(&header[open + 1..close], ',')
        .into_iter()
        .enumerate()
        .filter_map(|(index, arg)| {
            let arg = arg.trim();
            if arg.is_empty() {
                return None;
            }
            let (name, type_text) = arg
                .split_once(':')
                .map(|(name, type_text)| {
                    let name = name
                        .split_whitespace()
                        .last()
                        .unwrap_or_default()
                        .trim()
                        .to_owned();
                    (name, type_text)
                })
                .unwrap_or_else(|| (format!("%arg{index}"), arg));
            Some(ParsedValue {
                name,
                type_text: Some(clean_type_token(type_text.trim()).to_owned()),
            })
        })
        .collect()
}

fn parse_function_outputs(header: &str) -> Vec<ParsedValue> {
    let Some(arrow) = header.rfind("->") else {
        return Vec::new();
    };
    let suffix = &header[arrow + 2..];
    let body = suffix
        .rfind('{')
        .map(|index| &suffix[..index])
        .unwrap_or(suffix)
        .trim();
    split_type_list(body, usize::MAX)
        .into_iter()
        .enumerate()
        .map(|(index, type_text)| ParsedValue {
            name: format!("%result{index}"),
            type_text: Some(type_text),
        })
        .collect()
}

fn parse_type_info(model: &mut Model, type_text: Option<&str>) -> Option<TypeInfo> {
    let text = type_text?.trim();
    if text.is_empty() || text == "()" {
        return None;
    }
    let text = clean_type_token(text);
    if let Some(inner) = bracket_inner(text, "!torch.vtensor<") {
        return Some(parse_torch_vtensor_type(model, inner));
    }
    if let Some(inner) = bracket_inner(text, "tensor<")
        .or_else(|| bracket_inner(text, "memref<"))
        .or_else(|| bracket_inner(text, "vector<"))
    {
        return Some(parse_shaped_type(model, inner));
    }
    Some(TypeInfo {
        element_type: scalar_element_type(text),
        layout: None,
        denotation: None,
        shape: Vec::new(),
    })
}

fn parse_shaped_type(model: &mut Model, inner: &str) -> TypeInfo {
    let (dims, element) = split_shape_and_element(inner);
    TypeInfo {
        element_type: shaped_element_type(element),
        layout: None,
        denotation: None,
        shape: dims
            .iter()
            .map(|dim| {
                let dim = dim.trim();
                if dim == "?" || dim == "*" {
                    Dimension::symbolic(model.intern("?"))
                } else if dim.starts_with('[')
                    && dim.ends_with(']')
                    && dim[1..dim.len() - 1].parse::<i64>().is_ok()
                {
                    Dimension::known(dim[1..dim.len() - 1].parse::<i64>().unwrap())
                } else if let Ok(value) = dim.parse::<i64>() {
                    Dimension::known(value)
                } else {
                    Dimension::symbolic(model.intern(dim))
                }
            })
            .collect(),
    }
}

fn parse_torch_vtensor_type(model: &mut Model, inner: &str) -> TypeInfo {
    let parts = split_top_level(inner, ',');
    let dims_text = parts
        .first()
        .and_then(|value| value.trim().strip_prefix('['))
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or_default();
    let element = parts.get(1).copied().unwrap_or_default().trim();
    TypeInfo {
        element_type: shaped_element_type(element),
        layout: None,
        denotation: None,
        shape: if dims_text.trim().is_empty() {
            Vec::new()
        } else {
            split_top_level(dims_text, ',')
                .into_iter()
                .map(|dim| parse_dimension(model, dim.trim()))
                .collect()
        },
    }
}

fn parse_dimension(model: &mut Model, dim: &str) -> Dimension {
    if dim == "?" || dim == "*" || dim == "-1" {
        Dimension::symbolic(model.intern("?"))
    } else if let Ok(value) = dim.parse::<i64>() {
        Dimension::known(value)
    } else {
        Dimension::symbolic(model.intern(dim))
    }
}

fn split_shape_and_element(inner: &str) -> (Vec<&str>, &str) {
    let separators = shape_separators(inner);
    let Some(last) = separators.last().copied() else {
        return (Vec::new(), inner.trim());
    };
    let element_text = inner[last + 1..].trim();
    let element = split_top_level(element_text, ',')
        .into_iter()
        .next()
        .unwrap_or(element_text)
        .trim();
    let dims_text = inner[..last].trim();
    if dims_text == "*" {
        return (Vec::new(), element);
    }
    let mut dims = Vec::new();
    let mut start = 0usize;
    for index in separators.into_iter().filter(|index| *index < last) {
        dims.push(inner[start..index].trim());
        start = index + 1;
    }
    dims.push(inner[start..last].trim());
    (dims, element)
}

fn scalar_result_from_shaped_type(type_text: &str) -> Option<String> {
    let type_text = clean_type_token(type_text);
    let inner = bracket_inner(type_text, "tensor<")
        .or_else(|| bracket_inner(type_text, "memref<"))
        .or_else(|| bracket_inner(type_text, "vector<"))?;
    let (_, element) = split_shape_and_element(inner);
    Some(clean_type_token(element).to_owned())
}

fn shape_separators(text: &str) -> Vec<usize> {
    let mut result = Vec::new();
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    let mut previous = None;
    for (index, ch) in text.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            previous = Some(ch);
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            'x' if angle == 0
                && paren == 0
                && bracket == 0
                && brace == 0
                && previous.is_some_and(|value: char| !value.is_ascii_alphabetic()) =>
            {
                result.push(index);
            }
            _ => {}
        }
        previous = Some(ch);
    }
    result
}

fn scalar_element_type(text: &str) -> Option<TensorElementType> {
    let text = text.trim();
    match text {
        "f16" | "f32" | "f64" | "f80" | "f128" | "bf16" => None,
        "i1" | "i8" | "si8" | "i16" | "si16" | "i32" | "si32" | "i64" | "si64" | "i128"
        | "si128" | "ui1" | "ui8" | "ui16" | "ui32" | "ui64" => None,
        "index" => None,
        value if integer_type_name(value).is_some() => None,
        value if value.starts_with('!') => None,
        value if !value.is_empty() && value != "none" => {
            Some(TensorElementType::Other(value.to_owned()))
        }
        _ => None,
    }
}

fn shaped_element_type(text: &str) -> Option<TensorElementType> {
    match text.trim() {
        "f16" => Some(TensorElementType::Float16),
        "f32" => Some(TensorElementType::Float32),
        "f64" => Some(TensorElementType::Float64),
        "f80" => Some(TensorElementType::Other("float80".to_owned())),
        "f128" => Some(TensorElementType::Other("float128".to_owned())),
        "bf16" => Some(TensorElementType::BFloat16),
        "i1" => Some(TensorElementType::Other("int1".to_owned())),
        "i8" | "si8" => Some(TensorElementType::Int8),
        "i16" | "si16" => Some(TensorElementType::Int16),
        "i32" | "si32" => Some(TensorElementType::Int32),
        "i64" | "si64" => Some(TensorElementType::Int64),
        "i128" | "si128" => Some(TensorElementType::Other("int128".to_owned())),
        "index" => Some(TensorElementType::Int64),
        "ui8" => Some(TensorElementType::Uint8),
        "ui16" => Some(TensorElementType::Uint16),
        "ui32" => Some(TensorElementType::Uint32),
        "ui64" => Some(TensorElementType::Uint64),
        value if value.starts_with('!') => {
            Some(TensorElementType::Other(normalize_type_text(value)))
        }
        value => integer_type_name(value)
            .map(TensorElementType::Other)
            .or_else(|| scalar_element_type(value)),
    }
}

fn normalize_type_text(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    let mut index = 0usize;
    while index < bytes.len() {
        let ch = bytes[index] as char;
        if ch == ',' {
            output.push_str(", ");
            index += 1;
            while index < bytes.len() && (bytes[index] as char).is_whitespace() {
                index += 1;
            }
            continue;
        }
        if is_number_start(bytes, index) {
            let start = index;
            index = number_token_end(bytes, index);
            output.push_str(&normalize_number_token(&text[start..index]));
            continue;
        }
        output.push(ch);
        index += ch.len_utf8();
    }
    output
}

fn is_number_start(bytes: &[u8], index: usize) -> bool {
    let ch = bytes[index] as char;
    let next = bytes.get(index + 1).copied().map(char::from);
    let numeric = ch.is_ascii_digit()
        || (matches!(ch, '+' | '-' | '.') && next.is_some_and(|value| value.is_ascii_digit()));
    if !numeric {
        return false;
    }
    if index == 0 {
        return true;
    }
    let previous = bytes[index - 1] as char;
    !(previous.is_ascii_alphanumeric() || previous == '_' || previous == '.')
}

fn number_token_end(bytes: &[u8], mut index: usize) -> usize {
    if matches!(bytes[index] as char, '+' | '-') {
        index += 1;
    }
    while index < bytes.len() && (bytes[index] as char).is_ascii_digit() {
        index += 1;
    }
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        while index < bytes.len() && (bytes[index] as char).is_ascii_digit() {
            index += 1;
        }
    }
    if index < bytes.len() && matches!(bytes[index] as char, 'e' | 'E') {
        let exponent = index;
        index += 1;
        if index < bytes.len() && matches!(bytes[index] as char, '+' | '-') {
            index += 1;
        }
        let digits = index;
        while index < bytes.len() && (bytes[index] as char).is_ascii_digit() {
            index += 1;
        }
        if digits == index {
            return exponent;
        }
    }
    index
}

fn normalize_number_token(token: &str) -> String {
    if token.contains(['.', 'e', 'E']) {
        normalize_scalar_literal(token, None)
    } else {
        token.to_owned()
    }
}

fn integer_type_name(text: &str) -> Option<String> {
    let text = text.trim();
    let (prefix, digits) = text
        .strip_prefix("si")
        .map(|digits| ("int", digits))
        .or_else(|| text.strip_prefix("ui").map(|digits| ("uint", digits)))
        .or_else(|| text.strip_prefix('i').map(|digits| ("int", digits)))?;
    (!digits.is_empty() && digits.chars().all(|ch| ch.is_ascii_digit()))
        .then(|| format!("{prefix}{digits}"))
}

fn bracket_inner<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let inner = text.strip_prefix(prefix)?;
    inner.strip_suffix('>')
}

fn extract_value_names(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let start = index;
            index += 1;
            while index < bytes.len() {
                let ch = bytes[index] as char;
                if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-') {
                    index += 1;
                } else {
                    break;
                }
            }
            let mut name = text[start..index].to_owned();
            if index < bytes.len() && bytes[index] == b'#' {
                index += 1;
                let suffix_start = index;
                while index < bytes.len() && (bytes[index] as char).is_ascii_digit() {
                    index += 1;
                }
                if suffix_start < index {
                    let suffix = &text[suffix_start..index];
                    if suffix != "0" {
                        name.push('.');
                        name.push_str(suffix);
                    }
                }
            }
            values.push(name);
        } else {
            index += 1;
        }
    }
    values
}

fn extract_result_names(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    for part in split_top_level(text, ',') {
        let names = extract_value_names(part);
        let Some(name) = names.first() else {
            continue;
        };
        let count = result_count(part, name).unwrap_or(1);
        results.push(name.clone());
        for index in 1..count {
            results.push(format!("{name}.{index}"));
        }
    }
    results
}

fn result_count(part: &str, name: &str) -> Option<usize> {
    let start = part.find(name)? + name.len();
    let rest = part[start..].trim_start();
    let digits = rest.strip_prefix(':')?;
    let end = digits
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(digits.len());
    (end > 0)
        .then(|| digits[..end].parse::<usize>().ok())
        .flatten()
}

fn extract_attribute_dict(text: &str) -> Option<&str> {
    let colon = first_top_level_type_separator(text).unwrap_or(text.len());
    let prefix = &text[..colon];
    let open = prefix.find('{')?;
    let close = matching_delimiter(prefix, open, '{', '}')?;
    Some(&prefix[open + 1..close])
}

fn first_top_level_type_separator(text: &str) -> Option<usize> {
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    for (index, ch) in text.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            ':' if angle == 0
                && paren == 0
                && bracket == 0
                && brace == 0
                && !text[..index].ends_with(':')
                && !text[index + 1..].starts_with(':') =>
            {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

fn split_top_level(text: &str, delimiter: char) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    let mut quote = false;
    for (index, ch) in text.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            value
                if value == delimiter && angle == 0 && paren == 0 && bracket == 0 && brace == 0 =>
            {
                result.push(text[start..index].trim());
                start = index + value.len_utf8();
            }
            _ => {}
        }
    }
    result.push(text[start..].trim());
    result
}

fn matching_delimiter(text: &str, open: usize, left: char, right: char) -> Option<usize> {
    let mut depth = 0isize;
    let mut quote = false;
    for (index, ch) in text.char_indices().filter(|(index, _)| *index >= open) {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        if ch == '"' {
            quote = true;
        } else if ch == left {
            depth += 1;
        } else if ch == right {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn strip_type_annotation(value: &str) -> &str {
    let mut angle = 0isize;
    let mut bracket = 0isize;
    let mut quote = false;
    for (index, ch) in value.char_indices() {
        if quote {
            if ch == '"' {
                quote = false;
            }
            continue;
        }
        match ch {
            '"' => quote = true,
            '<' => angle += 1,
            '>' if angle > 0 => angle -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            ':' if angle == 0 && bracket == 0 => return value[..index].trim(),
            _ => {}
        }
    }
    value.trim()
}

fn strip_comment(line: &str) -> String {
    let mut quote = false;
    let chars = line.char_indices().peekable();
    for (index, ch) in chars {
        if ch == '"' {
            quote = !quote;
        }
        if !quote && ch == '/' && line[index..].starts_with("//") {
            return line[..index].trim_end().to_owned();
        }
    }
    line.trim_end().to_owned()
}

fn brace_delta(line: &str) -> isize {
    let mut delta = 0isize;
    let mut quote = false;
    for ch in line.chars() {
        if ch == '"' {
            quote = !quote;
        }
        if quote {
            continue;
        }
        match ch {
            '{' => delta += 1,
            '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

fn has_body_open(line: &str) -> bool {
    line.contains('{')
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
