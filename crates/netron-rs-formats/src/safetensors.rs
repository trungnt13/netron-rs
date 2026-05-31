use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path};

use netron_rs_core::{
    Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model, ModelError, ModelFormat,
    ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage, TypeInfo, Value, ValueId,
};
use serde::Deserialize;

const FORMAT: &str = "Safetensors";
const MAX_HEADER_LEN: usize = 100 * 1024 * 1024;

pub struct SafetensorsFormat;

impl ModelFormat for SafetensorsFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["safetensors", "safetensors.index.json"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        let is_safetensors_path = input
            .path
            .and_then(|path| path.extension())
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"));
        if is_safetensors_path || is_index_path(input.path) {
            return Confidence::High;
        }

        if Reader::open(input.data).is_some() {
            Confidence::Medium
        } else {
            Confidence::None
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        if let Some(reader) = Reader::open(input.data) {
            return lower_entries(reader.read()?);
        }
        if let Some(index) = read_index(input.data)? {
            return lower_index(input.path, index, |base, file| {
                fs::read(base.join(file))
                    .map_err(|error| invalid(format!("failed to read shard '{file}': {error}")))
            });
        }
        Err(invalid("missing safetensors header"))
    }
}

pub(crate) fn parse_index_with_shards(
    path: Option<&Path>,
    data: &[u8],
    read_shard: impl FnMut(&Path, &str) -> Result<Vec<u8>, ModelError>,
) -> Result<Option<Model>, ModelError> {
    let Some(index) = read_index(data)? else {
        return Ok(None);
    };
    lower_index(path, index, read_shard).map(Some)
}

fn is_index_path(path: Option<&Path>) -> bool {
    path.and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
}

fn read_index(data: &[u8]) -> Result<Option<IndexFile>, ModelError> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return Ok(None);
    };
    let Some(weight_map) = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(None);
    };
    let mut entries = Vec::new();
    for (key, value) in weight_map {
        let Some(value) = value.as_str() else {
            return Ok(None);
        };
        if !value.ends_with(".safetensors") {
            return Ok(None);
        }
        entries.push((key.clone(), value.to_owned()));
    }
    if entries.is_empty() {
        return Ok(None);
    }
    Ok(Some(IndexFile {
        weight_map: entries,
    }))
}

fn lower_index(
    path: Option<&Path>,
    index: IndexFile,
    mut read_shard: impl FnMut(&Path, &str) -> Result<Vec<u8>, ModelError>,
) -> Result<Model, ModelError> {
    let path = path.ok_or_else(|| invalid("safetensors index requires a file path"))?;
    let base = path
        .parent()
        .ok_or_else(|| invalid("safetensors index path has no parent directory"))?;
    let mut files = Vec::<String>::new();
    for (_, file) in &index.weight_map {
        validate_index_file(file)?;
        if !files.contains(file) {
            files.push(file.clone());
        }
    }

    let mut entries_by_file = HashMap::<String, HashMap<String, Entry>>::new();
    for file in files {
        let data = read_shard(base, &file)?;
        let reader = Reader::open(&data)
            .ok_or_else(|| invalid(format!("shard '{file}' is not a safetensors file")))?;
        entries_by_file.insert(
            file,
            reader
                .read()?
                .into_iter()
                .map(|entry| (entry.name.clone(), entry))
                .collect(),
        );
    }

    let mut entries = Vec::new();
    for (name, file) in index.weight_map {
        let shard = entries_by_file
            .get_mut(&file)
            .ok_or_else(|| invalid(format!("index references missing shard '{file}'")))?;
        let entry = shard.remove(&name).ok_or_else(|| {
            invalid(format!(
                "index maps tensor '{name}' to shard '{file}', but the shard does not contain it"
            ))
        })?;
        entries.push(entry);
    }
    lower_entries(entries)
}

fn validate_index_file(file: &str) -> Result<(), ModelError> {
    let path = Path::new(file);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid(format!(
            "safetensors shard path '{file}' is unsafe"
        )));
    }
    Ok(())
}

fn lower_entries(entries: Vec<Entry>) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut groups = Vec::<Group>::new();

    for entry in entries {
        let element_type = lower_element_type(&entry.dtype)?;
        let shape = entry
            .shape
            .iter()
            .copied()
            .map(Dimension::known)
            .collect::<Vec<_>>();
        let tensor_name = model.intern(&entry.name);
        let tensor = Tensor::metadata_only(
            Some(tensor_name),
            element_type.clone(),
            shape.clone(),
            TensorStorage::InlineBytes {
                byte_len: entry.byte_len,
            },
        );
        let tensor_id = model.add_tensor(tensor);
        let value_id = graph.add_value(Value::new(tensor_name));
        let value = &mut graph.values[value_id.index()];
        value.initializer = Some(tensor_id);
        value.type_info = Some(TypeInfo {
            element_type: Some(element_type),
            layout: None,
            denotation: None,
            shape,
        });

        let (group_name, _) = split_group_name(&entry.name);
        if let Some(group) = groups.iter_mut().find(|group| group.name == group_name) {
            group.inputs.push(value_id);
        } else {
            groups.push(Group {
                name: group_name.to_owned(),
                inputs: vec![value_id],
            });
        }
    }

    let op_name = model.intern("Module");
    for group in groups {
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: op_name,
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        if !group.name.is_empty() {
            node.name = Some(model.intern(&group.name));
        }
        node.inputs = group.inputs.iter().copied().map(Some).collect();
        let node_id = graph.add_node(node);
        for value_id in group.inputs {
            let consumers = &mut graph.values[value_id.index()].consumers;
            if !consumers.contains(&node_id) {
                consumers.push(node_id);
            }
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn split_group_name(name: &str) -> (&str, &str) {
    name.rsplit_once('.').unwrap_or(("", name))
}

fn lower_element_type(dtype: &str) -> Result<TensorElementType, ModelError> {
    let element_type = match dtype {
        "I8" => TensorElementType::Int8,
        "I16" => TensorElementType::Int16,
        "I32" => TensorElementType::Int32,
        "I64" => TensorElementType::Int64,
        "U8" => TensorElementType::Uint8,
        "U16" => TensorElementType::Uint16,
        "U32" => TensorElementType::Uint32,
        "U64" => TensorElementType::Uint64,
        "BF16" => TensorElementType::BFloat16,
        "F16" => TensorElementType::Float16,
        "F32" => TensorElementType::Float32,
        "F64" => TensorElementType::Float64,
        "BOOL" => TensorElementType::Bool,
        "F8_E4M3" => TensorElementType::Float8e4m3fn,
        "F8_E5M2" => TensorElementType::Float8e5m2,
        "F8_E8M0" => TensorElementType::Float8e8m0,
        _ => return Err(invalid(format!("unsupported dtype '{dtype}'"))),
    };
    Ok(element_type)
}

struct Group {
    name: String,
    inputs: Vec<ValueId>,
}

struct Entry {
    name: String,
    dtype: String,
    shape: Vec<i64>,
    byte_len: usize,
}

struct Reader<'a> {
    data: &'a [u8],
    header_len: usize,
}

impl<'a> Reader<'a> {
    fn open(data: &'a [u8]) -> Option<Self> {
        if data.len() <= 9 {
            return None;
        }
        let header_len = u64::from_le_bytes(data.get(0..8)?.try_into().ok()?) as usize;
        if header_len == 0
            || header_len > MAX_HEADER_LEN
            || 8_usize.checked_add(header_len)? >= data.len()
            || data.get(8) != Some(&b'{')
        {
            return None;
        }
        Some(Self { data, header_len })
    }

    fn read(&self) -> Result<Vec<Entry>, ModelError> {
        let header_end = 8 + self.header_len;
        let header = self
            .data
            .get(8..header_end)
            .ok_or_else(|| invalid("header extends past file"))?;
        let object = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(header)
            .map_err(|error| invalid(format!("header is not valid JSON: {error}")))?;
        let data_len = self.data.len() - header_end;
        let mut entries = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let entry = serde_json::from_value::<HeaderEntry>(value).map_err(|error| {
                invalid(format!("tensor '{name}' metadata is invalid: {error}"))
            })?;
            if entry.shape.iter().any(|dimension| *dimension < 0) {
                return Err(invalid(format!("tensor '{name}' has negative dimension")));
            }
            let [start, end] = entry.data_offsets;
            if start > end || end > data_len {
                return Err(invalid(format!(
                    "tensor '{name}' data offsets are outside file"
                )));
            }
            entries.push(Entry {
                name,
                dtype: entry.dtype,
                shape: entry.shape,
                byte_len: end - start,
            });
        }
        Ok(entries)
    }
}

#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<i64>,
    data_offsets: [usize; 2],
}

struct IndexFile {
    weight_map: Vec<(String, String)>,
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
