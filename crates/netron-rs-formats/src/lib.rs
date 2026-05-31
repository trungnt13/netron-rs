mod archive;
mod mlir;
mod onnx;

use std::path::Path;

use archive::ZipArchive;
use netron_rs_core::{Confidence, Model, ModelError, ModelFormat};

static MLIR: mlir::MlirFormat = mlir::MlirFormat;
static ONNX: onnx::OnnxFormat = onnx::OnnxFormat;

pub fn parse(input: ModelInput<'_>) -> Result<Model, ModelError> {
    let model = parse_with_containers(input, 0)?;
    model.validate()?;
    Ok(model)
}

fn parse_with_containers(input: ModelInput<'_>, depth: usize) -> Result<Model, ModelError> {
    let formats: [&dyn ModelFormat; 2] = [&ONNX, &MLIR];
    if let Some(format) = formats
        .into_iter()
        .map(|format| (format.detect(input), format))
        .filter(|(confidence, _)| *confidence != Confidence::None)
        .max_by_key(|(confidence, _)| *confidence)
        .map(|(_, format)| format)
    {
        return format.parse(input);
    }

    if !is_supported_zip_wrapper(input.path) {
        return Err(ModelError::UnsupportedFormat);
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
    let mut model = None;
    let mut first_error = None;
    let entries = archive
        .entries
        .iter()
        .filter(|entry| is_supported_archive_entry(entry.name));
    for entry in entries {
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

fn is_supported_archive_entry(name: &str) -> bool {
    let file_name = name.rsplit(['/', '\\']).next().unwrap_or(name);
    if file_name.is_empty() || file_name.starts_with('.') {
        return false;
    }
    let name = file_name.to_ascii_lowercase();
    [".onnx", ".pb", ".json"]
        .iter()
        .any(|extension| name.ends_with(extension))
}

fn is_supported_zip_wrapper(path: Option<&Path>) -> bool {
    let Some(file_name) = path
        .and_then(|path| path.file_name())
        .and_then(|file_name| file_name.to_str())
    else {
        return false;
    };
    file_name.to_ascii_lowercase().ends_with(".onnx.zip")
}

pub use netron_rs_core::{ModelInput, ToNormalizedJson};
