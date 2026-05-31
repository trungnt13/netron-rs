use std::path::Path;

use crate::archive::ZipArchive;
use netron_rs_core::{
    Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model, ModelError, ModelFormat,
    ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage, TypeInfo, Value, ValueId,
};

const ARRAY_FORMAT: &str = "NumPy Array";
const ARCHIVE_FORMAT: &str = "NumPy Archive";

pub struct NumpyFormat;

impl ModelFormat for NumpyFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: "NumPy",
            extensions: &["npy", "npz"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if NpyArray::read(input.data, None).is_ok() {
            return Confidence::High;
        }
        if is_numpy_path(input.path) {
            return Confidence::High;
        }
        Confidence::None
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        match NpyArray::read(input.data, None) {
            Ok(array) => return lower_array(array),
            Err(error) if is_npy_path(input.path) => return Err(error),
            Err(_) => {}
        }
        if is_npz_path(input.path) {
            return lower_archive(ZipArchive::open(input.data)?);
        }
        Err(invalid("missing NumPy array header"))
    }
}

fn is_numpy_path(path: Option<&Path>) -> bool {
    is_npy_path(path) || is_npz_path(path)
}

fn is_npy_path(path: Option<&Path>) -> bool {
    has_extension(path, "npy")
}

fn is_npz_path(path: Option<&Path>) -> bool {
    has_extension(path, "npz")
}

fn has_extension(path: Option<&Path>, extension: &str) -> bool {
    path.and_then(Path::extension)
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
}

fn lower_array(array: NpyArray) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: ARRAY_FORMAT,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let value_id = lower_value(&mut model, &mut graph, None, array);

    let mut node = Node::new(graph_id, operator(&mut model, "numpy.ndarray"));
    node.inputs.push(Some(value_id));
    let node_id = graph.add_node(node);
    graph.values[value_id.index()].consumers.push(node_id);

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_archive(archive: ZipArchive<'_>) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: ARCHIVE_FORMAT,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut inputs = Vec::new();

    for entry in archive.entries {
        if !entry.name.ends_with(".npy") {
            continue;
        }
        let data = entry.bytes()?;
        let name = entry.name.strip_suffix(".npy").unwrap_or(entry.name);
        let array = NpyArray::read(&data, Some(name))?;
        inputs.push(lower_value(&mut model, &mut graph, Some(name), array));
    }
    if inputs.is_empty() {
        return Err(invalid("NumPy archive has no .npy entries"));
    }

    let mut node = Node::new(graph_id, operator(&mut model, "Object"));
    node.inputs = inputs.iter().copied().map(Some).collect();
    let node_id = graph.add_node(node);
    for value_id in inputs {
        graph.values[value_id.index()].consumers.push(node_id);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_value(
    model: &mut Model,
    graph: &mut Graph,
    name: Option<&str>,
    array: NpyArray,
) -> ValueId {
    let shape = array
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let tensor = Tensor::metadata_only(
        None,
        array.element_type.clone(),
        shape.clone(),
        TensorStorage::InlineBytes {
            byte_len: array.data_len,
        },
    );
    let tensor_id = model.add_tensor(tensor);
    let value_name = model.intern(name.unwrap_or(""));
    let value_id = graph.add_value(Value::new(value_name));
    let value = &mut graph.values[value_id.index()];
    value.initializer = Some(tensor_id);
    value.type_info = Some(TypeInfo {
        element_type: Some(array.element_type),
        layout: None,
        denotation: None,
        shape,
    });
    value_id
}

fn operator(model: &mut Model, name: &str) -> Operator {
    Operator {
        domain: None,
        name: model.intern(name),
        overload: None,
        version: None,
        origin: "NumPy",
    }
}

struct NpyArray {
    element_type: TensorElementType,
    shape: Vec<i64>,
    data_len: usize,
}

impl NpyArray {
    fn read(data: &[u8], name: Option<&str>) -> Result<Self, ModelError> {
        if data.get(0..6) != Some(b"\x93NUMPY") {
            return Err(invalid("NumPy magic is missing"));
        }
        let major = *data
            .get(6)
            .ok_or_else(|| invalid("NumPy version is truncated"))?;
        let (header_offset, header_len): (usize, usize) = match major {
            1 => {
                let header = data
                    .get(8..10)
                    .ok_or_else(|| invalid("NumPy v1 header length is truncated"))?;
                (10, u16::from_le_bytes(header.try_into().unwrap()) as usize)
            }
            2 | 3 => {
                let header = data
                    .get(8..12)
                    .ok_or_else(|| invalid("NumPy v2/v3 header length is truncated"))?;
                (12, u32::from_le_bytes(header.try_into().unwrap()) as usize)
            }
            version => return Err(invalid(format!("unsupported NumPy version '{version}'"))),
        };
        let data_offset = header_offset
            .checked_add(header_len)
            .ok_or_else(|| invalid("NumPy header offset overflows usize"))?;
        let header = data
            .get(header_offset..data_offset)
            .ok_or_else(|| invalid("NumPy header is truncated"))?;
        let header = std::str::from_utf8(header)
            .map_err(|error| invalid(format!("NumPy header is not UTF-8: {error}")))?;
        let descriptor = parse_descriptor(header)?;
        let element_type =
            lower_element_type(&descriptor, data.get(data_offset..).unwrap_or(&[]), name);
        let shape = parse_shape(header)?;
        let data_len = data
            .len()
            .checked_sub(data_offset)
            .ok_or_else(|| invalid("NumPy data offset overflows usize"))?;
        Ok(Self {
            element_type,
            shape,
            data_len,
        })
    }
}

fn parse_descriptor(header: &str) -> Result<String, ModelError> {
    let mut parser = HeaderParser::after_key(header, "descr")?;
    parser.skip_ws();
    match parser.peek() {
        Some('\'') | Some('"') => parser.string(),
        Some('[') => parser.bracketed('[', ']'),
        _ => Err(invalid("NumPy descriptor is missing")),
    }
}

fn parse_shape(header: &str) -> Result<Vec<i64>, ModelError> {
    let mut parser = HeaderParser::after_key(header, "shape")?;
    parser.skip_ws();
    let tuple = parser.bracketed('(', ')')?;
    let inner = tuple
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .ok_or_else(|| invalid("NumPy shape is not a tuple"))?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.trim_end_matches(['L', 'l'])
                .parse::<i64>()
                .map_err(|error| invalid(format!("NumPy shape dimension '{part}': {error}")))
        })
        .collect()
}

fn lower_element_type(descriptor: &str, payload: &[u8], name: Option<&str>) -> TensorElementType {
    let descriptor = descriptor.trim();
    if descriptor.starts_with('[') {
        return TensorElementType::Other("void".to_owned());
    }

    let body = descriptor
        .strip_prefix(['<', '>', '|', '='])
        .unwrap_or(descriptor);
    let Some(kind) = body.chars().next() else {
        return TensorElementType::Unknown;
    };
    let size = body.get(kind.len_utf8()..).unwrap_or("");
    match (kind, size) {
        ('?', _) => TensorElementType::Bool,
        ('b', "1") | ('i', "1") => TensorElementType::Int8,
        ('B', "1") | ('u', "1") => TensorElementType::Uint8,
        ('i', "2") => TensorElementType::Int16,
        ('u', "2") => TensorElementType::Uint16,
        ('i', "4") => TensorElementType::Int32,
        ('u', "4") => TensorElementType::Uint32,
        ('i', "8") => TensorElementType::Int64,
        ('u', "8") => TensorElementType::Uint64,
        ('f', "1") => TensorElementType::Float8e5m2,
        ('f', "2") => TensorElementType::Float16,
        ('f', "4") => TensorElementType::Float32,
        ('f', "8") => TensorElementType::Float64,
        ('c', "8") => TensorElementType::Complex64,
        ('c', "16") => TensorElementType::Complex128,
        ('S', _) | ('U', _) => TensorElementType::String,
        ('O', _) if is_string_dtype_payload(payload, name) => TensorElementType::String,
        ('O', _) => TensorElementType::Other("object".to_owned()),
        ('V', _) => TensorElementType::Other("void".to_owned()),
        _ => TensorElementType::Other(descriptor.to_owned()),
    }
}

fn is_string_dtype_payload(payload: &[u8], name: Option<&str>) -> bool {
    name.is_some_and(|name| name.eq_ignore_ascii_case("StringDType"))
        || payload
            .windows(b"_convert_to_stringdtype_kwargs".len())
            .any(|window| window == b"_convert_to_stringdtype_kwargs")
}

struct HeaderParser<'a> {
    text: &'a str,
    offset: usize,
}

impl<'a> HeaderParser<'a> {
    fn after_key(text: &'a str, key: &str) -> Result<Self, ModelError> {
        let single = format!("'{key}'");
        let double = format!("\"{key}\"");
        let start = text
            .find(&single)
            .or_else(|| text.find(&double))
            .ok_or_else(|| invalid(format!("NumPy header field '{key}' is missing")))?;
        let colon = text[start..]
            .find(':')
            .ok_or_else(|| invalid(format!("NumPy header field '{key}' has no value")))?;
        Ok(Self {
            text,
            offset: start + colon + 1,
        })
    }

    fn peek(&self) -> Option<char> {
        self.text.get(self.offset..)?.chars().next()
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.offset += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    fn string(&mut self) -> Result<String, ModelError> {
        let quote = self
            .peek()
            .ok_or_else(|| invalid("NumPy string value is truncated"))?;
        if quote != '\'' && quote != '"' {
            return Err(invalid("NumPy string value is not quoted"));
        }
        self.offset += quote.len_utf8();
        let start = self.offset;
        while let Some(ch) = self.peek() {
            if ch == quote {
                let value = self
                    .text
                    .get(start..self.offset)
                    .ok_or_else(|| invalid("NumPy string slice is invalid"))?
                    .to_owned();
                self.offset += quote.len_utf8();
                return Ok(value);
            }
            if ch == '\\' {
                self.offset += ch.len_utf8();
                if let Some(escaped) = self.peek() {
                    self.offset += escaped.len_utf8();
                }
            } else {
                self.offset += ch.len_utf8();
            }
        }
        Err(invalid("NumPy string value is unterminated"))
    }

    fn bracketed(&mut self, open: char, close: char) -> Result<String, ModelError> {
        if self.peek() != Some(open) {
            return Err(invalid("NumPy bracketed value has unexpected start"));
        }
        let start = self.offset;
        let mut depth = 0_i32;
        while let Some(ch) = self.peek() {
            if ch == '\'' || ch == '"' {
                self.string()?;
                continue;
            }
            if ch == open {
                depth += 1;
            } else if ch == close {
                depth -= 1;
                if depth == 0 {
                    self.offset += ch.len_utf8();
                    return self
                        .text
                        .get(start..self.offset)
                        .map(ToOwned::to_owned)
                        .ok_or_else(|| invalid("NumPy bracketed slice is invalid"));
                }
            }
            self.offset += ch.len_utf8();
        }
        Err(invalid("NumPy bracketed value is unterminated"))
    }
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: "NumPy",
        message: message.into(),
    }
}
