mod archive;
mod coreml;
mod darknet;
mod dot;
mod gguf;
mod lightgbm;
mod message;
mod mlir;
mod ncnn;
mod numpy;
mod onnx;
mod pytorch;
mod safetensors;
mod sentencepiece;
mod tflite;
mod xgboost;

use std::path::Path;

use archive::ZipArchive;
use netron_rs_core::{Confidence, Model, ModelError, ModelFormat};

static COREML: coreml::CoreMlFormat = coreml::CoreMlFormat;
static DARKNET: darknet::DarknetFormat = darknet::DarknetFormat;
static DOT: dot::DotFormat = dot::DotFormat;
static GGUF: gguf::GgufFormat = gguf::GgufFormat;
static LIGHTGBM: lightgbm::LightGbmFormat = lightgbm::LightGbmFormat;
static MESSAGE: message::MessageFormat = message::MessageFormat;
static MLIR: mlir::MlirFormat = mlir::MlirFormat;
static NCNN: ncnn::NcnnFormat = ncnn::NcnnFormat;
static NUMPY: numpy::NumpyFormat = numpy::NumpyFormat;
static ONNX: onnx::OnnxFormat = onnx::OnnxFormat;
static PYTORCH: pytorch::PyTorchFormat = pytorch::PyTorchFormat;
static SAFETENSORS: safetensors::SafetensorsFormat = safetensors::SafetensorsFormat;
static SENTENCEPIECE: sentencepiece::SentencePieceFormat = sentencepiece::SentencePieceFormat;
static TFLITE: tflite::TfliteFormat = tflite::TfliteFormat;
static XGBOOST: xgboost::XGBoostFormat = xgboost::XGBoostFormat;

pub fn parse(input: ModelInput<'_>) -> Result<Model, ModelError> {
    let model = parse_with_containers(input, 0)?;
    model.validate()?;
    Ok(model)
}

fn parse_with_containers(input: ModelInput<'_>, depth: usize) -> Result<Model, ModelError> {
    let formats: [&dyn ModelFormat; 15] = [
        &ONNX,
        &SAFETENSORS,
        &GGUF,
        &TFLITE,
        &COREML,
        &NUMPY,
        &LIGHTGBM,
        &XGBOOST,
        &MESSAGE,
        &DARKNET,
        &MLIR,
        &NCNN,
        &SENTENCEPIECE,
        &DOT,
        &PYTORCH,
    ];
    if let Some(format) = formats
        .into_iter()
        .map(|format| (format.detect(input), format))
        .filter(|(confidence, _)| *confidence != Confidence::None)
        .max_by_key(|(confidence, _)| *confidence)
        .map(|(_, format)| format)
    {
        return format.parse(input);
    }

    parse_zip_container(input, depth)?.ok_or(ModelError::UnsupportedFormat)
}

fn parse_zip_container(input: ModelInput<'_>, depth: usize) -> Result<Option<Model>, ModelError> {
    if depth >= 4 {
        return Ok(None);
    }
    let Ok(archive) = ZipArchive::open(input.data) else {
        return Ok(None);
    };
    if let Some(model) = parse_ncnn_zip_container(&archive)? {
        return Ok(Some(model));
    }
    if let Some(model) = parse_safetensors_zip_container(&archive)? {
        return Ok(Some(model));
    }
    let mut model = None;
    let mut first_error = None;
    let entries = archive
        .entries
        .iter()
        .filter(|entry| is_supported_archive_entry(entry.name))
        .collect::<Vec<_>>();
    let has_primary_entry = entries
        .iter()
        .any(|entry| !is_auxiliary_archive_entry(entry.name));
    for entry in entries
        .into_iter()
        .filter(|entry| !has_primary_entry || !is_auxiliary_archive_entry(entry.name))
    {
        let data = entry.bytes()?;
        let input = ModelInput {
            data: &data,
            path: Some(Path::new(entry.name)),
        };
        match parse_with_containers(input, depth + 1) {
            Ok(candidate) => {
                if model.is_some() {
                    return Err(ModelError::InvalidData {
                        format: "ZIP",
                        message: "archive contains multiple model files".to_owned(),
                    });
                }
                model = Some(candidate);
            }
            Err(ModelError::UnsupportedFormat) => {}
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    if model.is_none()
        && let Some(error) = first_error
    {
        return Err(error);
    }
    Ok(model)
}

fn parse_ncnn_zip_container(archive: &ZipArchive<'_>) -> Result<Option<Model>, ModelError> {
    let mut model = None;
    for entry in &archive.entries {
        let Some(sidecar_name) = ncnn_zip_sidecar_name(entry.name) else {
            continue;
        };
        let Some(sidecar) = archive
            .entries
            .iter()
            .find(|candidate| candidate.name == sidecar_name.as_str())
        else {
            continue;
        };
        let data = entry.bytes()?;
        let blob_data = sidecar.bytes()?;
        let candidate =
            ncnn::parse_with_blob_data(&data, Some(Path::new(entry.name)), Some(blob_data))?;
        if model.is_some() {
            return Err(ModelError::InvalidData {
                format: "ZIP",
                message: "archive contains multiple ncnn model files".to_owned(),
            });
        }
        model = Some(candidate);
    }
    Ok(model)
}

fn parse_safetensors_zip_container(archive: &ZipArchive<'_>) -> Result<Option<Model>, ModelError> {
    let mut model = None;
    for entry in &archive.entries {
        if !is_safetensors_index_archive_entry(entry.name) {
            continue;
        }
        let data = entry.bytes()?;
        let base = archive_entry_parent(entry.name);
        let candidate =
            safetensors::parse_index_with_shards(Some(Path::new(entry.name)), &data, |_, file| {
                let shard_name = archive_sibling_name(base, file)?;
                let shard = archive
                    .entries
                    .iter()
                    .find(|candidate| candidate.name == shard_name.as_str())
                    .ok_or_else(|| ModelError::InvalidData {
                        format: "ZIP",
                        message: format!("archive is missing safetensors shard '{shard_name}'"),
                    })?;
                shard.bytes()
            })?;
        let Some(candidate) = candidate else {
            continue;
        };
        if model.is_some() {
            return Err(ModelError::InvalidData {
                format: "ZIP",
                message: "archive contains multiple safetensors index files".to_owned(),
            });
        }
        model = Some(candidate);
    }
    Ok(model)
}

fn ncnn_zip_sidecar_name(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".cfg.ncnn") {
        return Some(format!(
            "{}{}",
            &name[..name.len() - ".cfg.ncnn".len()],
            ".weights.ncnn"
        ));
    }
    if lower.ends_with(".param.bin") {
        return Some(format!(
            "{}{}",
            &name[..name.len() - ".param.bin".len()],
            ".bin"
        ));
    }
    if lower.ends_with(".param") {
        return Some(format!(
            "{}{}",
            &name[..name.len() - ".param".len()],
            ".bin"
        ));
    }
    None
}

fn is_auxiliary_archive_entry(name: &str) -> bool {
    let file_name = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let name = file_name.to_ascii_lowercase();
    name.ends_with(".dot")
        || name.ends_with(".svg")
        || (name.ends_with(".json") && !name.ends_with(".safetensors.index.json"))
}

fn is_safetensors_index_archive_entry(name: &str) -> bool {
    let file_name = name.rsplit(['/', '\\']).next().unwrap_or(name);
    file_name
        .to_ascii_lowercase()
        .ends_with(".safetensors.index.json")
}

fn archive_entry_parent(name: &str) -> &str {
    name.rsplit_once('/')
        .map(|(parent, _)| parent)
        .unwrap_or("")
}

fn archive_sibling_name(base: &str, file: &str) -> Result<String, ModelError> {
    if file.starts_with('/') || file.contains('\\') {
        return Err(ModelError::InvalidData {
            format: "ZIP",
            message: format!("archive safetensors shard path '{file}' is unsafe"),
        });
    }
    let mut parts = base
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    for part in file.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                return Err(ModelError::InvalidData {
                    format: "ZIP",
                    message: format!("archive safetensors shard path '{file}' is unsafe"),
                });
            }
            part => parts.push(part),
        }
    }
    Ok(parts.join("/"))
}

fn is_supported_archive_entry(name: &str) -> bool {
    let file_name = name.rsplit(['/', '\\']).next().unwrap_or(name);
    if file_name.is_empty() || file_name.starts_with('.') {
        return false;
    }
    let name = file_name.to_ascii_lowercase();
    [
        ".gguf",
        ".lite",
        ".mlmodel",
        ".model",
        ".message",
        ".maxviz",
        ".cfg",
        ".mlir",
        ".param.bin",
        ".param",
        ".ncnn",
        ".dot",
        ".npy",
        ".npz",
        ".onnx",
        ".pb",
        ".json",
        ".pkl",
        ".pickle",
        ".pt",
        ".pt2",
        ".pth",
        ".safetensors",
        ".tflite",
    ]
    .iter()
    .any(|extension| name.ends_with(extension))
}

pub use netron_rs_core::{ModelInput, ToNormalizedJson};
