use netron_rs_core::{
    Attribute, AttributeValue, Confidence, FormatInfo, FormatMetadata, Graph, Model, ModelError,
    ModelFormat, ModelInput, Node, Operator, Value,
};
use serde::Deserialize;
use serde_json::{Number as JsonNumber, Value as JsonValue};

const FORMAT: &str = "XGBoost JSON";
const UBJSON_FORMAT: &str = "XGBoost UBJSON";
const SCIKIT_FORMAT: &str = "scikit-learn";

pub struct XGBoostFormat;

impl ModelFormat for XGBoostFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["json", "ubj", "model", "pkl"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if xgboost_json_root(input.data)
            .as_ref()
            .is_some_and(is_xgboost_root)
            || parse_ubjson(input.data)
                .ok()
                .as_ref()
                .is_some_and(is_xgboost_root)
            || is_xgboost_sklearn_pickle(input.data)
        {
            Confidence::High
        } else {
            Confidence::None
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        if is_xgboost_sklearn_pickle(input.data) {
            return lower_sklearn_classifier();
        }
        if let Some(json) = xgboost_json_root(input.data) {
            let parsed: XGBoostJson =
                serde_json::from_value(json).map_err(|error| ModelError::InvalidData {
                    format: FORMAT,
                    message: format!("invalid XGBoost JSON: {error}"),
                })?;
            return lower_model(FORMAT, parsed);
        }
        let json = parse_ubjson(input.data)?;
        let parsed: XGBoostJson =
            serde_json::from_value(json).map_err(|error| ModelError::InvalidData {
                format: UBJSON_FORMAT,
                message: format!("invalid XGBoost UBJSON: {error}"),
            })?;
        lower_model(UBJSON_FORMAT, parsed)
    }
}

fn is_xgboost_sklearn_pickle(data: &[u8]) -> bool {
    data.starts_with(b"\x80")
        && data
            .windows(b"xgboost.sklearn".len())
            .any(|window| window == b"xgboost.sklearn")
        && data
            .windows(b"XGBClassifier".len())
            .any(|window| window == b"XGBClassifier")
}

fn xgboost_json_root(data: &[u8]) -> Option<JsonValue> {
    serde_json::from_slice::<JsonValue>(data).ok()
}

fn is_xgboost_root(json: &JsonValue) -> bool {
    let Some(object) = json.as_object() else {
        return false;
    };
    object.len() < 256 && object.contains_key("learner") && object.contains_key("version")
}

fn parse_ubjson(data: &[u8]) -> Result<JsonValue, ModelError> {
    UbjsonReader { data, offset: 0 }.value()
}

struct UbjsonReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl UbjsonReader<'_> {
    fn value(&mut self) -> Result<JsonValue, ModelError> {
        let marker = self.byte()?;
        self.value_with_marker(marker)
    }

    fn value_with_marker(&mut self, marker: u8) -> Result<JsonValue, ModelError> {
        match marker {
            b'{' => self.object(),
            b'[' => self.array(),
            b'S' => Ok(JsonValue::String(self.string()?)),
            b'Z' => Ok(JsonValue::Null),
            b'T' => Ok(JsonValue::Bool(true)),
            b'F' => Ok(JsonValue::Bool(false)),
            b'U' | b'i' | b'I' | b'l' | b'L' => Ok(JsonValue::Number(JsonNumber::from(
                self.integer_with_marker(marker)?,
            ))),
            b'd' | b'D' => {
                let value = self.float_with_marker(marker)?;
                Ok(JsonNumber::from_f64(value)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null))
            }
            _ => Err(invalid(format!(
                "unsupported XGBoost UBJSON marker '{}'",
                marker as char
            ))),
        }
    }

    fn object(&mut self) -> Result<JsonValue, ModelError> {
        let mut object = serde_json::Map::new();
        while self.peek()? != b'}' {
            let key = self.key()?;
            let value = self.value()?;
            object.insert(key, value);
        }
        self.byte()?;
        Ok(JsonValue::Object(object))
    }

    fn array(&mut self) -> Result<JsonValue, ModelError> {
        let value_marker = if self.peek()? == b'$' {
            self.byte()?;
            Some(self.byte()?)
        } else {
            None
        };
        let count = if self.peek()? == b'#' {
            self.byte()?;
            Some(self.length()?)
        } else {
            None
        };
        let mut values = Vec::new();
        if let Some(count) = count {
            for _ in 0..count {
                values.push(if let Some(marker) = value_marker {
                    self.value_with_marker(marker)?
                } else {
                    self.value()?
                });
            }
        } else {
            while self.peek()? != b']' {
                values.push(if let Some(marker) = value_marker {
                    self.value_with_marker(marker)?
                } else {
                    self.value()?
                });
            }
            self.byte()?;
        }
        Ok(JsonValue::Array(values))
    }

    fn key(&mut self) -> Result<String, ModelError> {
        let len = self.length()?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| invalid(format!("invalid XGBoost UBJSON key: {error}")))
    }

    fn string(&mut self) -> Result<String, ModelError> {
        let len = self.length()?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| invalid(format!("invalid XGBoost UBJSON string: {error}")))
    }

    fn length(&mut self) -> Result<usize, ModelError> {
        let marker = self.byte()?;
        let value = self.integer_with_marker(marker)?;
        usize::try_from(value).map_err(|_| invalid("XGBoost UBJSON length is negative"))
    }

    fn integer_with_marker(&mut self, marker: u8) -> Result<i64, ModelError> {
        Ok(match marker {
            b'U' => self.byte()? as i64,
            b'i' => i8::from_be_bytes([self.byte()?]) as i64,
            b'I' => i16::from_be_bytes(self.take_array()?) as i64,
            b'l' => i32::from_be_bytes(self.take_array()?) as i64,
            b'L' => i64::from_be_bytes(self.take_array()?),
            _ => return Err(invalid("XGBoost UBJSON integer marker is invalid")),
        })
    }

    fn float_with_marker(&mut self, marker: u8) -> Result<f64, ModelError> {
        Ok(match marker {
            b'd' => f32::from_be_bytes(self.take_array()?) as f64,
            b'D' => f64::from_be_bytes(self.take_array()?),
            _ => return Err(invalid("XGBoost UBJSON float marker is invalid")),
        })
    }

    fn peek(&self) -> Result<u8, ModelError> {
        self.data
            .get(self.offset)
            .copied()
            .ok_or_else(|| invalid("unexpected end of XGBoost UBJSON"))
    }

    fn byte(&mut self) -> Result<u8, ModelError> {
        let value = self.peek()?;
        self.offset += 1;
        Ok(value)
    }

    fn take(&mut self, len: usize) -> Result<&[u8], ModelError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid("XGBoost UBJSON offset overflows usize"))?;
        let bytes = self
            .data
            .get(self.offset..end)
            .ok_or_else(|| invalid("unexpected end of XGBoost UBJSON"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        self.take(N)?
            .try_into()
            .map_err(|_| invalid("unexpected end of XGBoost UBJSON"))
    }
}

fn lower_model(format: &'static str, parsed: XGBoostJson) -> Result<Model, ModelError> {
    let version = parsed.version.as_ref().map(|version| {
        version
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(".")
    });
    let mut model = Model::new(FormatInfo {
        name: format,
        version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);

    let feature_value_ids = parsed
        .learner
        .feature_names
        .iter()
        .map(|name| {
            let mut value = Value::new(model.intern(name));
            value.is_graph_input = parsed.learner.feature_names.len() < 1000;
            let value_id = graph.add_value(value);
            if parsed.learner.feature_names.len() < 1000 {
                graph.inputs.push(value_id);
            }
            value_id
        })
        .collect::<Vec<_>>();

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("xgboost.core.Booster"),
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

fn lower_sklearn_classifier() -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: SCIKIT_FORMAT,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    graph.add_node(Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("xgboost.sklearn.XGBClassifier"),
            overload: None,
            version: None,
            origin: SCIKIT_FORMAT,
        },
    ));
    model.replace_graph(graph_id, graph);
    Ok(model)
}

#[derive(Debug, Deserialize)]
struct XGBoostJson {
    learner: Learner,
    version: Option<Vec<u64>>,
}

impl XGBoostJson {
    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if let Some(version) = &self.version {
            push_attr(
                model,
                &mut attributes,
                "version",
                AttributeValue::Ints(version.iter().map(|value| *value as i64).collect()),
            );
        }
        let params = &self.learner.learner_model_param;
        push_param_string(
            model,
            &mut attributes,
            "base_score",
            params.base_score.as_deref(),
        );
        push_param_i64(
            model,
            &mut attributes,
            "num_feature",
            params.num_feature.as_deref(),
        );
        push_param_i64(
            model,
            &mut attributes,
            "num_class",
            params.num_class.as_deref(),
        );
        push_param_i64(
            model,
            &mut attributes,
            "num_target",
            params.num_target.as_deref(),
        );
        if let Some(name) = &self.learner.objective.name {
            let value = model.intern(name);
            push_attr(
                model,
                &mut attributes,
                "objective",
                AttributeValue::String(value),
            );
        }
        if self.learner.attributes.is_some() {
            let value = model.intern("[object Object]");
            push_attr(
                model,
                &mut attributes,
                "attributes",
                AttributeValue::String(value),
            );
        }
        if let Some(name) = &self.learner.gradient_booster.name {
            let value = model.intern(name);
            push_attr(
                model,
                &mut attributes,
                "booster_type",
                AttributeValue::String(value),
            );
        }
        let booster_model = &self.learner.gradient_booster.model;
        push_param_i64(
            model,
            &mut attributes,
            "num_trees",
            booster_model
                .gbtree_model_param
                .as_ref()
                .and_then(|param| param.num_trees.as_deref()),
        );
        push_param_i64(
            model,
            &mut attributes,
            "num_parallel_tree",
            booster_model
                .gbtree_model_param
                .as_ref()
                .and_then(|param| param.num_parallel_tree.as_deref()),
        );
        let tree_placeholders =
            std::iter::repeat_n("[object Object]", booster_model.trees.len().min(16))
                .map(|value| model.intern(value))
                .collect();
        push_attr(
            model,
            &mut attributes,
            "trees",
            AttributeValue::Strings(tree_placeholders),
        );
        push_attr(
            model,
            &mut attributes,
            "tree_info",
            AttributeValue::Ints(booster_model.tree_info.clone()),
        );
        attributes
    }
}

#[derive(Debug, Deserialize)]
struct Learner {
    #[serde(default)]
    attributes: Option<JsonValue>,
    #[serde(default)]
    feature_names: Vec<String>,
    gradient_booster: GradientBooster,
    learner_model_param: LearnerModelParam,
    objective: Objective,
}

#[derive(Debug, Deserialize)]
struct LearnerModelParam {
    base_score: Option<String>,
    num_class: Option<String>,
    num_feature: Option<String>,
    num_target: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GradientBooster {
    model: BoosterModel,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BoosterModel {
    gbtree_model_param: Option<GbTreeModelParam>,
    #[serde(default)]
    trees: Vec<JsonValue>,
    #[serde(default)]
    tree_info: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct GbTreeModelParam {
    num_parallel_tree: Option<String>,
    num_trees: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Objective {
    name: Option<String>,
}

fn push_param_string(
    model: &mut Model,
    attributes: &mut Vec<Attribute>,
    name: &str,
    value: Option<&str>,
) {
    if let Some(value) = value {
        let value = model.intern(value);
        push_attr(model, attributes, name, AttributeValue::String(value));
    }
}

fn push_param_i64(
    model: &mut Model,
    attributes: &mut Vec<Attribute>,
    name: &str,
    value: Option<&str>,
) {
    if let Some(value) = value.and_then(|value| value.parse::<i64>().ok()) {
        push_attr(model, attributes, name, AttributeValue::Int(value));
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

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: UBJSON_FORMAT,
        message: message.into(),
    }
}
