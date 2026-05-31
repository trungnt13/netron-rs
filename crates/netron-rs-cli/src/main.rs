use std::env;
use std::fs::{self, File};
use std::io::{Error as IoError, ErrorKind};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use memmap2::Mmap;
use netron_rs_core::{Model, ModelInput, TensorStorage};
use netron_rs_formats::ToNormalizedJson;
use serde::Serialize;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let command = Command::parse(env::args().skip(1).collect())?;
    match command {
        Command::Parse { path } => {
            let mapped = MappedModel::open(&path)?;
            let model = netron_rs_formats::parse(mapped.input())?;
            println!("{}", model.to_normalized_json()?);
        }
        Command::Stats { path } => {
            let mapped = MappedModel::open(&path)?;
            let model = netron_rs_formats::parse(mapped.input())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&ModelStats::from_model(&model, mapped.len()))?
            );
        }
        Command::Bench { path, iterations } => {
            let mapped = MappedModel::open(&path)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&Benchmark::run(&mapped, iterations)?)?
            );
        }
        Command::Layout {
            path,
            graph,
            max_nodes,
        } => {
            let mapped = MappedModel::open(&path)?;
            let model = netron_rs_formats::parse(mapped.input())?;
            let options = netron_rs_layout::LayoutOptions {
                graph,
                max_nodes,
                ..netron_rs_layout::LayoutOptions::default()
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&netron_rs_layout::layout_model(&model, &options)?)?
            );
        }
        Command::Search { path, query, limit } => {
            let mapped = MappedModel::open(&path)?;
            let model = netron_rs_formats::parse(mapped.input())?;
            let index = netron_rs_query::ModelIndex::build(&model);
            println!(
                "{}",
                serde_json::to_string_pretty(&index.search(&query, limit))?
            );
        }
    }
    Ok(())
}

enum Command {
    Parse {
        path: PathBuf,
    },
    Stats {
        path: PathBuf,
    },
    Bench {
        path: PathBuf,
        iterations: usize,
    },
    Layout {
        path: PathBuf,
        graph: usize,
        max_nodes: Option<usize>,
    },
    Search {
        path: PathBuf,
        query: String,
        limit: usize,
    },
}

impl Command {
    fn parse(args: Vec<String>) -> Result<Self, Box<dyn std::error::Error>> {
        match args.as_slice() {
            [path] => Ok(Self::Parse {
                path: PathBuf::from(path),
            }),
            [command, path] if command == "parse" => Ok(Self::Parse {
                path: PathBuf::from(path),
            }),
            [command, path] if command == "stats" => Ok(Self::Stats {
                path: PathBuf::from(path),
            }),
            [command, path] if command == "bench" => Ok(Self::Bench {
                path: PathBuf::from(path),
                iterations: 10,
            }),
            [command, path, iterations] if command == "bench" => Ok(Self::Bench {
                path: PathBuf::from(path),
                iterations: iterations.parse::<usize>()?.max(1),
            }),
            [command, path] if command == "layout" => Ok(Self::Layout {
                path: PathBuf::from(path),
                graph: 0,
                max_nodes: None,
            }),
            [command, path, graph] if command == "layout" => Ok(Self::Layout {
                path: PathBuf::from(path),
                graph: graph.parse::<usize>()?,
                max_nodes: None,
            }),
            [command, path, graph, max_nodes] if command == "layout" => Ok(Self::Layout {
                path: PathBuf::from(path),
                graph: graph.parse::<usize>()?,
                max_nodes: Some(max_nodes.parse::<usize>()?),
            }),
            [command, path, query] if command == "search" => Ok(Self::Search {
                path: PathBuf::from(path),
                query: query.clone(),
                limit: 20,
            }),
            [command, path, query, limit] if command == "search" => Ok(Self::Search {
                path: PathBuf::from(path),
                query: query.clone(),
                limit: limit.parse::<usize>()?.max(1),
            }),
            _ => Err(
                "usage: netron-rs [parse|stats|bench|layout|search] <model> [query|iterations|graph] [limit|max_nodes]"
                    .into(),
            ),
        }
    }
}

struct MappedModel {
    path: PathBuf,
    data: ModelData,
}

enum ModelData {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl AsRef<[u8]> for ModelData {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Mapped(data) => data,
            Self::Owned(data) => data,
        }
    }
}

impl MappedModel {
    fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata = fs::metadata(path)?;
        if metadata.is_dir() {
            return Self::open_directory(path);
        }
        let file = File::open(path)?;
        // SAFETY: the mmap is read-only and lives as long as every borrowed parser input.
        let data = ModelData::Mapped(unsafe { Mmap::map(&file)? });
        Ok(Self {
            path: path.to_owned(),
            data,
        })
    }

    fn open_directory(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !has_extension(path, "mlpackage") {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                format!("unsupported model directory '{}'", path.display()),
            )
            .into());
        }
        let model_path = coreml_package_model_path(path)?;
        let data = fs::read(&model_path)?;
        Ok(Self {
            path: model_path,
            data: ModelData::Owned(data),
        })
    }

    fn input(&self) -> ModelInput<'_> {
        ModelInput {
            data: self.data.as_ref(),
            path: Some(self.path.as_path()),
        }
    }

    fn len(&self) -> usize {
        self.data.as_ref().len()
    }
}

fn coreml_package_model_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest_path = path.join("Manifest.json");
    if manifest_path.is_file() {
        let manifest = fs::read(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest)?;
        let entries = manifest
            .get("itemInfoEntries")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                IoError::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Core ML package '{}' has no itemInfoEntries",
                        path.display()
                    ),
                )
            })?;
        if let Some(root) = manifest
            .get("rootModelIdentifier")
            .and_then(serde_json::Value::as_str)
        {
            if let Some(entry_path) = entries
                .get(root)
                .and_then(|entry| entry.get("path"))
                .and_then(serde_json::Value::as_str)
                .filter(|entry_path| entry_path.to_ascii_lowercase().ends_with(".mlmodel"))
            {
                if let Some(model_path) = coreml_manifest_model_path(path, entry_path)? {
                    return Ok(model_path);
                }
            }
        }
        let mut candidates = entries
            .values()
            .filter_map(|entry| entry.get("path").and_then(serde_json::Value::as_str))
            .filter(|entry_path| entry_path.to_ascii_lowercase().ends_with(".mlmodel"));
        if let Some(entry_path) = candidates.next() {
            if let Some(model_path) = coreml_manifest_model_path(path, entry_path)? {
                return Ok(model_path);
            }
        }
    }

    find_first_file(path, |candidate| has_extension(candidate, "mlmodel")).ok_or_else(|| {
        IoError::new(
            ErrorKind::NotFound,
            format!(
                "Core ML package '{}' contains no .mlmodel file",
                path.display()
            ),
        )
        .into()
    })
}

fn coreml_manifest_model_path(
    package_path: &Path,
    entry_path: &str,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    if entry_path.contains('\\')
        || Path::new(entry_path).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("Core ML package entry path '{entry_path}' escapes package Data directory"),
        )
        .into());
    }

    let data_root = package_path.join("Data");
    let model_path = data_root.join(entry_path);
    if !model_path.is_file() {
        return Ok(None);
    }

    let data_root = data_root.canonicalize()?;
    let model_path = model_path.canonicalize()?;
    if !model_path.starts_with(&data_root) {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("Core ML package entry path '{entry_path}' escapes package Data directory"),
        )
        .into());
    }
    Ok(Some(model_path))
}

fn find_first_file(path: &Path, predicate: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut stack = vec![path.to_owned()];
    while let Some(directory) = stack.pop() {
        let entries = fs::read_dir(directory).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = entry.file_type().ok()?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() && predicate(&path) {
                return Some(path);
            }
        }
    }
    None
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

#[derive(Serialize)]
struct ModelStats {
    bytes: usize,
    format: &'static str,
    graphs: usize,
    functions: usize,
    nodes: usize,
    values: usize,
    tensors: usize,
    tensor_inline_bytes: usize,
    tensor_external: usize,
    tensor_sparse: usize,
    strings: usize,
}

impl ModelStats {
    fn from_model(model: &Model, bytes: usize) -> Self {
        let nodes = model.graphs.iter().map(|graph| graph.nodes.len()).sum();
        let values = model.graphs.iter().map(|graph| graph.values.len()).sum();
        let mut tensor_inline_bytes = 0;
        let mut tensor_external = 0;
        let mut tensor_sparse = 0;
        for tensor in &model.tensors {
            match &tensor.storage {
                TensorStorage::InlineBytes { byte_len } => tensor_inline_bytes += byte_len,
                TensorStorage::External { .. } => tensor_external += 1,
                TensorStorage::Sparse { .. } => tensor_sparse += 1,
                TensorStorage::Absent | TensorStorage::ElementList { .. } => {}
            }
        }
        Self {
            bytes,
            format: model.format.name,
            graphs: model.graphs.len(),
            functions: model.functions.len(),
            nodes,
            values,
            tensors: model.tensors.len(),
            tensor_inline_bytes,
            tensor_external,
            tensor_sparse,
            strings: model.strings.len(),
        }
    }
}

#[derive(Serialize)]
struct Benchmark {
    iterations: usize,
    bytes: usize,
    parse_bytes_per_second_mean: f64,
    parse_ms_min: f64,
    parse_ms_mean: f64,
    parse_ms_max: f64,
    parse_ms_last: f64,
    json_ms_last: f64,
    layout_ms_last: f64,
    search_index_ms_last: f64,
    parse_and_json_ms_last: f64,
    parse_json_layout_ms_last: f64,
    time_to_first_graph_ms_last: f64,
    stats: ModelStats,
}

impl Benchmark {
    fn run(mapped: &MappedModel, iterations: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut parse_times = Vec::with_capacity(iterations);
        let mut last_model = None;
        for _ in 0..iterations {
            let start = Instant::now();
            let model = netron_rs_formats::parse(mapped.input())?;
            parse_times.push(start.elapsed().as_secs_f64() * 1000.0);
            last_model = Some(model);
        }
        let model = last_model.expect("iterations is at least one");
        let json_start = Instant::now();
        let _json = model.to_normalized_json()?;
        let json_ms_last = json_start.elapsed().as_secs_f64() * 1000.0;
        let layout_start = Instant::now();
        if !model.graphs.is_empty() {
            let _layout = netron_rs_layout::layout_model(
                &model,
                &netron_rs_layout::LayoutOptions::default(),
            )?;
        }
        let layout_ms_last = layout_start.elapsed().as_secs_f64() * 1000.0;
        let search_index_start = Instant::now();
        let _index = netron_rs_query::ModelIndex::build(&model);
        let search_index_ms_last = search_index_start.elapsed().as_secs_f64() * 1000.0;
        let stats = ModelStats::from_model(&model, mapped.len());
        let parse_ms_min = parse_times.iter().copied().fold(f64::INFINITY, f64::min);
        let parse_ms_max = parse_times
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let parse_ms_mean = parse_times.iter().sum::<f64>() / parse_times.len() as f64;
        let parse_ms_last = parse_times.last().copied().unwrap_or(0.0);
        let parse_bytes_per_second_mean = if parse_ms_mean > 0.0 {
            mapped.len() as f64 / (parse_ms_mean / 1000.0)
        } else {
            f64::INFINITY
        };
        Ok(Self {
            iterations,
            bytes: mapped.len(),
            parse_bytes_per_second_mean,
            parse_ms_min,
            parse_ms_mean,
            parse_ms_max,
            parse_ms_last,
            json_ms_last,
            layout_ms_last,
            search_index_ms_last,
            parse_and_json_ms_last: parse_ms_last + json_ms_last,
            parse_json_layout_ms_last: parse_ms_last + json_ms_last + layout_ms_last,
            time_to_first_graph_ms_last: parse_ms_last + layout_ms_last,
            stats,
        })
    }
}
