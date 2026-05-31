use netron_rs_core::{
    Attribute, AttributeValue, Confidence, FormatInfo, FormatMetadata, Graph, Model, ModelError,
    ModelFormat, ModelInput, Node, Operator, Value,
};

const FORMAT: &str = "LightGBM";
const PICKLE_FORMAT: &str = "LightGBM Pickle";

pub struct LightGbmFormat;

impl ModelFormat for LightGbmFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["txt", "model", "pkl"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if input.data.starts_with(b"tree\n")
            || input.data.starts_with(b"tree\r\n")
            || lightgbm_pickle_text(input.data).is_some()
        {
            Confidence::High
        } else {
            Confidence::None
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        if let Some(text) = lightgbm_pickle_text(input.data) {
            return lower_model(PICKLE_FORMAT, parse_text_model(text)?);
        }
        let text = std::str::from_utf8(input.data)
            .map_err(|error| invalid(format!("LightGBM text is not UTF-8: {error}")))?;
        lower_model(FORMAT, parse_text_model(text)?)
    }
}

fn lightgbm_pickle_text(data: &[u8]) -> Option<&str> {
    if !data
        .windows(b"clightgbm.basic\nBooster\n".len())
        .any(|window| window == b"clightgbm.basic\nBooster\n")
    {
        return None;
    }
    for offset in 0..data.len().saturating_sub(5) {
        if data[offset] != b'X' {
            continue;
        }
        let len = u32::from_le_bytes(data.get(offset + 1..offset + 5)?.try_into().ok()?) as usize;
        let start = offset + 5;
        let end = start.checked_add(len)?;
        if data.get(start..start + 5) == Some(b"tree\n") {
            return std::str::from_utf8(data.get(start..end)?).ok();
        }
    }
    None
}

fn lower_model(format: &'static str, parsed: LightGbmText) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: format,
        version: parsed
            .version
            .strip_prefix('v')
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);

    let feature_value_ids = parsed
        .feature_names
        .iter()
        .map(|name| {
            let mut value = Value::new(model.intern(name));
            value.is_graph_input = parsed.feature_names.len() < 1000;
            let value_id = graph.add_value(value);
            if parsed.feature_names.len() < 1000 {
                graph.inputs.push(value_id);
            }
            value_id
        })
        .collect::<Vec<_>>();

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("lightgbm.basic.Booster"),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs = feature_value_ids.iter().copied().map(Some).collect();
    node.attributes = parsed.attributes(&mut model);
    let node_id = graph.add_node(node);
    for value_id in feature_value_ids {
        graph.values[value_id.index()].consumers.push(node_id);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

#[derive(Debug)]
struct LightGbmText {
    version: String,
    num_class: Option<i64>,
    num_tree_per_iteration: Option<i64>,
    label_index: Option<i64>,
    max_feature_idx: Option<i64>,
    objective: Option<String>,
    average_output: bool,
    feature_names: Vec<String>,
    tree_count: usize,
    loaded_parameter: String,
}

impl LightGbmText {
    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        push_attr(
            model,
            &mut attributes,
            "average_output",
            AttributeValue::Bool(self.average_output),
        );
        let model_placeholders = std::iter::repeat_n("[object Object]", self.tree_count.min(16))
            .map(|value| model.intern(value))
            .collect();
        push_attr(
            model,
            &mut attributes,
            "models",
            AttributeValue::Strings(model_placeholders),
        );
        let loaded_parameter = model.intern(&self.loaded_parameter);
        push_attr(
            model,
            &mut attributes,
            "loaded_parameter",
            AttributeValue::String(loaded_parameter),
        );
        let version = model.intern(&self.version);
        push_attr(
            model,
            &mut attributes,
            "version",
            AttributeValue::String(version),
        );
        if let Some(value) = self.num_class {
            push_attr(
                model,
                &mut attributes,
                "num_class",
                AttributeValue::Int(value),
            );
        }
        if let Some(value) = self.num_tree_per_iteration {
            push_attr(
                model,
                &mut attributes,
                "num_tree_per_iteration",
                AttributeValue::Int(value),
            );
        }
        if let Some(value) = self.label_index {
            push_attr(
                model,
                &mut attributes,
                "label_index",
                AttributeValue::Int(value),
            );
        }
        if let Some(value) = self.max_feature_idx {
            push_attr(
                model,
                &mut attributes,
                "max_feature_idx",
                AttributeValue::Int(value),
            );
        }
        if let Some(value) = &self.objective {
            let value = model.intern(value);
            push_attr(
                model,
                &mut attributes,
                "objective",
                AttributeValue::String(value),
            );
        }
        attributes
    }
}

fn push_attr(
    model: &mut Model,
    attributes: &mut Vec<Attribute>,
    name: &str,
    value: AttributeValue,
) {
    attributes.push(Attribute {
        name: model.intern(name),
        value,
    });
}

fn parse_text_model(text: &str) -> Result<LightGbmText, ModelError> {
    let mut lines = text.lines();
    if lines.next().map(str::trim_end) != Some("tree") {
        return Err(invalid("missing LightGBM tree header"));
    }

    let mut parsed = LightGbmText {
        version: String::new(),
        num_class: None,
        num_tree_per_iteration: None,
        label_index: None,
        max_feature_idx: None,
        objective: None,
        average_output: false,
        feature_names: Vec::new(),
        tree_count: 0,
        loaded_parameter: String::new(),
    };
    let mut in_parameters = false;

    for line in lines {
        if in_parameters {
            if line == "end of parameters" {
                in_parameters = false;
            } else if line.is_empty() {
                continue;
            } else {
                parsed.loaded_parameter.push_str(line);
                parsed.loaded_parameter.push('\n');
            }
            continue;
        }

        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if line == "parameters:" {
            in_parameters = true;
            continue;
        }
        if line == "average_output" {
            parsed.average_output = true;
            continue;
        }
        if let Some(index) = line.find('=') {
            let key = &line[..index];
            let value = &line[index + 1..];
            match key {
                "version" => parsed.version = value.to_owned(),
                "num_class" => parsed.num_class = parse_i64(value),
                "num_tree_per_iteration" => parsed.num_tree_per_iteration = parse_i64(value),
                "label_index" => parsed.label_index = parse_i64(value),
                "max_feature_idx" => parsed.max_feature_idx = parse_i64(value),
                "objective" => parsed.objective = Some(value.to_owned()),
                "feature_names" => {
                    parsed.feature_names = value
                        .split_whitespace()
                        .filter(|value| !value.is_empty())
                        .map(ToOwned::to_owned)
                        .collect();
                }
                "Tree" => parsed.tree_count += 1,
                _ => {}
            }
        }
    }

    if parsed.version.is_empty() {
        return Err(invalid("missing LightGBM version"));
    }
    Ok(parsed)
}

fn parse_i64(value: &str) -> Option<i64> {
    value.parse::<i64>().ok()
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
