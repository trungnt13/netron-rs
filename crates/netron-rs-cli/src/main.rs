use std::env;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::Mmap;
use netron_rs_core::{Model, ModelError, ModelInput, TensorStorage};
use netron_rs_formats::ToNormalizedJson;
use netron_rs_query::{EntityHandle, FormatKind, ModelSession, ModelSource, SessionLimits};
use serde::Serialize;

const CLI_SCHEMA_VERSION: u32 = 1;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let json = args.iter().any(|arg| arg == "--json");
    if let Err(mut error) = run(args) {
        error.json |= json;
        error.print();
        std::process::exit(error.code);
    }
}

fn run(args: Vec<String>) -> Result<(), CliFailure> {
    let command = Command::parse(args)?;
    let command_name = command.name();
    run_command(command).map_err(|error| error.or_command(command_name))
}

fn run_command(command: Command) -> Result<(), CliFailure> {
    match command {
        Command::Parse { path } => {
            let mapped = MappedModel::open(&path)?;
            let model = netron_rs_formats::parse(mapped.input()).map_err(CliFailure::from)?;
            println!("{}", model.to_normalized_json().map_err(CliFailure::from)?);
        }
        Command::Summary { path, json } => {
            let mapped = MappedModel::open(&path)?;
            let session = ModelSession::open(mapped.bytes(), mapped.source())?;
            let summary = session.summary(&SessionLimits::default());
            print_output("summary", json, &summary)?;
        }
        Command::Stats { path } => {
            let mapped = MappedModel::open(&path)?;
            let model = netron_rs_formats::parse(mapped.input()).map_err(CliFailure::from)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&ModelStats::from_model(&model, mapped.len()))
                    .map_err(CliFailure::from)?
            );
        }
        Command::Bench { path, iterations } => {
            let mapped = MappedModel::open(&path)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&Benchmark::run(&mapped, iterations)?)
                    .map_err(CliFailure::from)?
            );
        }
        Command::Layout {
            path,
            selector,
            max_nodes,
            json,
        } => {
            let mapped = MappedModel::open(&path)?;
            if json {
                let session = ModelSession::open(mapped.bytes(), mapped.source())?;
                let limit = max_nodes.unwrap_or(SessionLimits::default().layout);
                let limits = SessionLimits {
                    layout: limit,
                    slice: limit,
                    ..SessionLimits::default()
                };
                match layout_scope(&session, selector)? {
                    LayoutRequest::Layout(handle) => {
                        let layout = session.layout(&handle, &limits).ok_or_else(|| {
                            CliFailure::invalid("layout scope is not available for this model")
                        })?;
                        print_json("layout", &layout)?;
                    }
                    LayoutRequest::Slice(handle) => {
                        let slice = session.slice(&handle, &limits).ok_or_else(|| {
                            CliFailure::invalid("slice scope is not available for this model")
                        })?;
                        print_json("layout", &slice)?;
                    }
                }
            } else {
                let graph = match selector {
                    LayoutSelector::Graph(graph) => graph,
                    _ => 0,
                };
                let model = netron_rs_formats::parse(mapped.input()).map_err(CliFailure::from)?;
                let options = netron_rs_layout::LayoutOptions {
                    graph,
                    max_nodes,
                    ..netron_rs_layout::LayoutOptions::default()
                };
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &netron_rs_layout::layout_model(&model, &options)
                            .map_err(|error| CliFailure::internal(error.to_string()))?
                    )
                    .map_err(CliFailure::from)?
                );
            }
        }
        Command::Search {
            path,
            query,
            limit,
            json,
        } => {
            let mapped = MappedModel::open(&path)?;
            let session = ModelSession::open(mapped.bytes(), mapped.source())?;
            let limits = SessionLimits {
                search: limit,
                ..SessionLimits::default()
            };
            let hits = session.search(&query, &limits);
            print_output("search", json, &hits)?;
        }
        Command::Detail { path, handle, json } => {
            let mapped = MappedModel::open(&path)?;
            let session = ModelSession::open(mapped.bytes(), mapped.source())?;
            let detail = session
                .detail(&handle, &SessionLimits::default())
                .ok_or_else(|| CliFailure::invalid("detail handle is not available"))?;
            print_output("detail", json, &detail)?;
        }
        Command::MlirSymbols { path, json } => {
            let mapped = MappedModel::open(&path)?;
            let session = ModelSession::open(mapped.bytes(), mapped.source())?;
            let symbols = session
                .mlir_symbols(&SessionLimits::default())
                .ok_or_else(|| CliFailure::invalid("MLIR symbols require an MLIR model"))?;
            print_output("mlir.symbols", json, &symbols)?;
        }
        Command::OnnxTensor { path, tensor, json } => {
            let mapped = MappedModel::open(&path)?;
            let session = ModelSession::open(mapped.bytes(), mapped.source())?;
            if session.summary(&SessionLimits::default()).format != FormatKind::Onnx {
                return Err(CliFailure::invalid(
                    "ONNX tensor metadata requires an ONNX model",
                ));
            }
            let metadata = session
                .tensor_metadata(&SessionLimits::default())
                .into_iter()
                .find(|entry| matches!(entry.handle, EntityHandle::Tensor { tensor: id } if id == tensor))
                .ok_or_else(|| CliFailure::invalid("tensor id is not available"))?;
            print_output("onnx.tensor", json, &metadata)?;
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct CliSuccess<'a, T: Serialize + ?Sized> {
    schema_version: u32,
    status: &'static str,
    command: &'a str,
    data: &'a T,
}

fn print_output<T: Serialize>(
    command: &'static str,
    json: bool,
    data: &T,
) -> Result<(), CliFailure> {
    if json {
        print_json(command, data)
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(data).map_err(CliFailure::from)?
        );
        Ok(())
    }
}

fn print_json<T: Serialize>(command: &'static str, data: &T) -> Result<(), CliFailure> {
    let envelope = CliSuccess {
        schema_version: CLI_SCHEMA_VERSION,
        status: "ok",
        command,
        data,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&envelope).map_err(CliFailure::from)?
    );
    Ok(())
}

#[derive(Serialize)]
struct CliErrorEnvelope<'a> {
    schema_version: u32,
    status: &'static str,
    command: Option<&'a str>,
    error: CliErrorBody<'a>,
}

#[derive(Serialize)]
struct CliErrorBody<'a> {
    code: &'static str,
    message: &'a str,
}

#[derive(Debug)]
struct CliFailure {
    code: i32,
    kind: &'static str,
    message: String,
    command: Option<&'static str>,
    json: bool,
}

impl CliFailure {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: 4,
            kind: "invalid_request",
            message: message.into(),
            command: None,
            json: false,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            code: 1,
            kind: "internal_error",
            message: message.into(),
            command: None,
            json: false,
        }
    }

    fn access_denied(message: impl Into<String>) -> Self {
        Self {
            code: 3,
            kind: "access_denied",
            message: message.into(),
            command: None,
            json: false,
        }
    }

    fn with_command(mut self, command: &'static str) -> Self {
        self.command = Some(command);
        self
    }

    fn or_command(mut self, command: &'static str) -> Self {
        if self.command.is_none() {
            self.command = Some(command);
        }
        self
    }

    fn print(&self) {
        if self.json {
            let envelope = CliErrorEnvelope {
                schema_version: CLI_SCHEMA_VERSION,
                status: "error",
                command: self.command,
                error: CliErrorBody {
                    code: self.kind,
                    message: &self.message,
                },
            };
            match serde_json::to_string_pretty(&envelope) {
                Ok(json) => eprintln!("{json}"),
                Err(_) => eprintln!("{}: {}", self.kind, self.message),
            }
        } else {
            eprintln!("{}", self.message);
        }
    }
}

impl From<ModelError> for CliFailure {
    fn from(error: ModelError) -> Self {
        match error {
            ModelError::UnsupportedFormat => Self {
                code: 2,
                kind: "unsupported_format",
                message: error.to_string(),
                command: None,
                json: false,
            },
            ModelError::AccessDenied { .. } => Self {
                code: 3,
                kind: "access_denied",
                message: error.to_string(),
                command: None,
                json: false,
            },
            ModelError::InvalidData { .. } => Self {
                code: 5,
                kind: "parse_error",
                message: error.to_string(),
                command: None,
                json: false,
            },
            ModelError::Invariant(_) => Self::internal(error.to_string()),
        }
    }
}

impl From<serde_json::Error> for CliFailure {
    fn from(error: serde_json::Error) -> Self {
        Self::internal(error.to_string())
    }
}

impl From<std::io::Error> for CliFailure {
    fn from(error: std::io::Error) -> Self {
        match error.kind() {
            ErrorKind::PermissionDenied => Self::access_denied(error.to_string()),
            ErrorKind::NotFound | ErrorKind::InvalidInput => Self::invalid(error.to_string()),
            _ => Self::internal(error.to_string()),
        }
    }
}

enum Command {
    Parse {
        path: PathBuf,
    },
    Summary {
        path: PathBuf,
        json: bool,
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
        selector: LayoutSelector,
        max_nodes: Option<usize>,
        json: bool,
    },
    Search {
        path: PathBuf,
        query: String,
        limit: usize,
        json: bool,
    },
    Detail {
        path: PathBuf,
        handle: EntityHandle,
        json: bool,
    },
    MlirSymbols {
        path: PathBuf,
        json: bool,
    },
    OnnxTensor {
        path: PathBuf,
        tensor: usize,
        json: bool,
    },
}

enum LayoutSelector {
    Graph(usize),
    Node {
        graph: usize,
        node: usize,
    },
    MlirFunction {
        function: FunctionSelector,
        region: Option<usize>,
    },
}

enum FunctionSelector {
    Index(usize),
    Name(String),
}

enum LayoutRequest {
    Layout(EntityHandle),
    Slice(EntityHandle),
}

impl Command {
    fn parse(args: Vec<String>) -> Result<Self, CliFailure> {
        if args.len() == 1 {
            return Ok(Self::Parse {
                path: PathBuf::from(&args[0]),
            });
        }
        let Some(command) = args.first().map(String::as_str) else {
            return Err(Self::usage());
        };
        match command {
            "parse" => single_path(args, "parse").map(|path| Self::Parse { path }),
            "summary" => parse_summary(args).map_err(|error| error.or_command("summary")),
            "stats" => single_path(args, "stats").map(|path| Self::Stats { path }),
            "bench" => parse_bench(args).map_err(|error| error.or_command("bench")),
            "layout" => parse_layout(args).map_err(|error| error.or_command("layout")),
            "search" => parse_search(args).map_err(|error| error.or_command("search")),
            "detail" => parse_detail(args).map_err(|error| error.or_command("detail")),
            "mlir" => parse_mlir(args).map_err(|error| error.or_command("mlir")),
            "onnx" => parse_onnx(args).map_err(|error| error.or_command("onnx")),
            _ => Err(Self::usage()),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Parse { .. } => "parse",
            Self::Summary { .. } => "summary",
            Self::Stats { .. } => "stats",
            Self::Bench { .. } => "bench",
            Self::Layout { .. } => "layout",
            Self::Search { .. } => "search",
            Self::Detail { .. } => "detail",
            Self::MlirSymbols { .. } => "mlir.symbols",
            Self::OnnxTensor { .. } => "onnx.tensor",
        }
    }

    fn usage() -> CliFailure {
        CliFailure::invalid(
            "usage: netron-rs [parse|summary|stats|bench|layout|search|detail|mlir|onnx] ...",
        )
    }
}

fn single_path(mut args: Vec<String>, command: &'static str) -> Result<PathBuf, CliFailure> {
    args.remove(0);
    if args.len() == 1 {
        Ok(PathBuf::from(args.remove(0)))
    } else {
        Err(CliFailure::invalid(format!("{command} requires <model>")).with_command(command))
    }
}

fn parse_summary(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    let json = take_flag(&mut args, "--json");
    if args.len() != 1 {
        return Err(CliFailure::invalid("summary requires <model>").with_command("summary"));
    }
    Ok(Command::Summary {
        path: PathBuf::from(args.remove(0)),
        json,
    })
}

fn parse_bench(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    let iterations = match args.len() {
        1 => 10,
        2 => parse_usize_arg(&args[1], "iterations")?.max(1),
        _ => {
            return Err(
                CliFailure::invalid("bench requires <model> [iterations]").with_command("bench")
            );
        }
    };
    Ok(Command::Bench {
        path: PathBuf::from(args.remove(0)),
        iterations,
    })
}

fn parse_layout(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    let json = take_flag(&mut args, "--json");
    let mut max_nodes = take_value(&mut args, "--max-nodes")?
        .map(|value| parse_usize_arg(&value, "--max-nodes"))
        .transpose()?;
    if max_nodes.is_none() {
        max_nodes = take_value(&mut args, "--max-ops")?
            .map(|value| parse_usize_arg(&value, "--max-ops"))
            .transpose()?;
    }
    let graph = take_value(&mut args, "--graph")?
        .map(|value| parse_usize_arg(&value, "--graph"))
        .transpose()?;
    let node = take_value(&mut args, "--node")?
        .map(|value| parse_usize_arg(&value, "--node"))
        .transpose()?;
    let function = take_value(&mut args, "--function")?;
    if let Some(depth) = take_value(&mut args, "--depth")? {
        parse_usize_arg(&depth, "--depth")?;
    }
    let region = take_value(&mut args, "--region")?
        .map(|value| parse_usize_arg(&value, "--region"))
        .transpose()?;

    let selector =
        if let Some(node) = node {
            if args.len() != 1 {
                return Err(
                    CliFailure::invalid("layout --node requires <model>").with_command("layout")
                );
            }
            LayoutSelector::Node {
                graph: graph.unwrap_or(0),
                node,
            }
        } else if let Some(function) = function {
            if args.len() != 1 {
                return Err(CliFailure::invalid("layout --function requires <model>")
                    .with_command("layout"));
            }
            LayoutSelector::MlirFunction {
                function: parse_function_selector(function),
                region,
            }
        } else {
            if region.is_some() {
                return Err(CliFailure::invalid("layout --region requires --function")
                    .with_command("layout"));
            }
            let (path, selected_graph, positional_max) = match args.len() {
                1 => (PathBuf::from(args.remove(0)), graph.unwrap_or(0), None),
                2 if graph.is_none() => (
                    PathBuf::from(args.remove(0)),
                    parse_usize_arg(&args.remove(0), "graph")?,
                    None,
                ),
                3 if graph.is_none() && max_nodes.is_none() => (
                    PathBuf::from(args.remove(0)),
                    parse_usize_arg(&args.remove(0), "graph")?,
                    Some(parse_usize_arg(&args.remove(0), "max_nodes")?),
                ),
                _ => {
                    return Err(
                        CliFailure::invalid("layout requires <model> [graph] [max_nodes]")
                            .with_command("layout"),
                    );
                }
            };
            return Ok(Command::Layout {
                path,
                selector: LayoutSelector::Graph(selected_graph),
                max_nodes: max_nodes.or(positional_max),
                json,
            });
        };

    Ok(Command::Layout {
        path: PathBuf::from(args.remove(0)),
        selector,
        max_nodes,
        json,
    })
}

fn parse_search(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    let json = take_flag(&mut args, "--json");
    let flagged_limit = take_value(&mut args, "--limit")?
        .map(|value| parse_usize_arg(&value, "--limit"))
        .transpose()?;
    let limit = match (flagged_limit, args.len()) {
        (Some(limit), 2) => limit.max(1),
        (None, 2) => 20,
        (None, 3) => parse_usize_arg(&args[2], "limit")?.max(1),
        _ => {
            return Err(
                CliFailure::invalid("search requires <model> <query> [limit]")
                    .with_command("search"),
            );
        }
    };
    Ok(Command::Search {
        path: PathBuf::from(args.remove(0)),
        query: args.remove(0),
        limit,
        json,
    })
}

fn parse_detail(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    let json = take_flag(&mut args, "--json");
    let graph = take_value(&mut args, "--graph")?
        .map(|value| parse_usize_arg(&value, "--graph"))
        .transpose()?
        .unwrap_or(0);
    let scope = take_value(&mut args, "--scope")?;
    let handle = if let Some(node) = take_value(&mut args, "--node")? {
        EntityHandle::Node {
            graph,
            node: parse_usize_arg(&node, "--node")?,
        }
    } else if let Some(value) = take_value(&mut args, "--value")? {
        EntityHandle::Value {
            graph,
            value: parse_usize_arg(&value, "--value")?,
        }
    } else if let Some(tensor) = take_value(&mut args, "--tensor")? {
        EntityHandle::Tensor {
            tensor: parse_usize_arg(&tensor, "--tensor")?,
        }
    } else if let Some(function) = take_value(&mut args, "--function")? {
        EntityHandle::Function {
            function: parse_usize_arg(&function, "--function")?,
        }
    } else if let Some(function) = take_value(&mut args, "--mlir-function")? {
        EntityHandle::MlirFunction {
            function: parse_usize_arg(&function, "--mlir-function")?,
        }
    } else if let Some(operation) = take_value(&mut args, "--mlir-operation")? {
        EntityHandle::MlirOperation {
            scope: required_scope(scope.as_deref(), "--mlir-operation")?,
            operation: parse_usize_arg(&operation, "--mlir-operation")?,
        }
    } else if let Some(region) = take_value(&mut args, "--mlir-region")? {
        EntityHandle::MlirRegion {
            scope: required_scope(scope.as_deref(), "--mlir-region")?,
            region: parse_usize_arg(&region, "--mlir-region")?,
        }
    } else if let Some(block) = take_value(&mut args, "--mlir-block")? {
        EntityHandle::MlirBlock {
            scope: required_scope(scope.as_deref(), "--mlir-block")?,
            block: parse_usize_arg(&block, "--mlir-block")?,
        }
    } else if let Some(symbol) = take_value(&mut args, "--mlir-symbol")? {
        EntityHandle::MlirSymbol {
            symbol: parse_usize_arg(&symbol, "--mlir-symbol")?,
        }
    } else {
        return Err(CliFailure::invalid("detail requires a handle selector").with_command("detail"));
    };
    if args.len() != 1 {
        return Err(CliFailure::invalid("detail requires <model>").with_command("detail"));
    }
    Ok(Command::Detail {
        path: PathBuf::from(args.remove(0)),
        handle,
        json,
    })
}

fn parse_mlir(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    if args.first().map(String::as_str) != Some("symbols") {
        return Err(CliFailure::invalid("mlir requires subcommand: symbols").with_command("mlir"));
    }
    args.remove(0);
    let json = take_flag(&mut args, "--json");
    if args.len() != 1 {
        return Err(
            CliFailure::invalid("mlir symbols requires <model>").with_command("mlir.symbols")
        );
    }
    Ok(Command::MlirSymbols {
        path: PathBuf::from(args.remove(0)),
        json,
    })
}

fn parse_onnx(mut args: Vec<String>) -> Result<Command, CliFailure> {
    args.remove(0);
    if args.first().map(String::as_str) != Some("tensor") {
        return Err(CliFailure::invalid("onnx requires subcommand: tensor").with_command("onnx"));
    }
    args.remove(0);
    let json = take_flag(&mut args, "--json");
    let flagged_tensor = take_value(&mut args, "--tensor")?
        .map(|value| parse_usize_arg(&value, "--tensor"))
        .transpose()?;
    let tensor = match (flagged_tensor, args.len()) {
        (Some(tensor), 1) => tensor,
        (None, 2) => parse_usize_arg(&args.remove(1), "tensor")?,
        _ => {
            return Err(
                CliFailure::invalid("onnx tensor requires <model> --tensor <id>")
                    .with_command("onnx.tensor"),
            );
        }
    };
    Ok(Command::OnnxTensor {
        path: PathBuf::from(args.remove(0)),
        tensor,
        json,
    })
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}

fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, CliFailure> {
    let Some(index) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    args.remove(index);
    if index >= args.len() {
        return Err(CliFailure::invalid(format!("{flag} requires a value")));
    }
    Ok(Some(args.remove(index)))
}

fn parse_usize_arg(value: &str, name: &str) -> Result<usize, CliFailure> {
    value
        .parse::<usize>()
        .map_err(|_| CliFailure::invalid(format!("{name} must be a non-negative integer")))
}

fn parse_function_selector(value: String) -> FunctionSelector {
    value
        .parse::<usize>()
        .map(FunctionSelector::Index)
        .unwrap_or(FunctionSelector::Name(value))
}

fn required_scope(scope: Option<&str>, selector: &'static str) -> Result<String, CliFailure> {
    scope
        .map(str::to_owned)
        .ok_or_else(|| CliFailure::invalid(format!("{selector} requires --scope")))
}

fn layout_scope(
    session: &ModelSession,
    selector: LayoutSelector,
) -> Result<LayoutRequest, CliFailure> {
    match selector {
        LayoutSelector::Graph(graph) => Ok(LayoutRequest::Layout(EntityHandle::Graph { graph })),
        LayoutSelector::Node { graph, node } => {
            Ok(LayoutRequest::Slice(EntityHandle::Node { graph, node }))
        }
        LayoutSelector::MlirFunction {
            function: FunctionSelector::Index(function),
            region,
        } => Ok(LayoutRequest::Layout(mlir_function_or_region(
            function, region,
        ))),
        LayoutSelector::MlirFunction {
            function: FunctionSelector::Name(name),
            region,
        } => {
            let handle = find_mlir_function(session, &name)?;
            let EntityHandle::MlirFunction { function } = handle else {
                return Err(CliFailure::invalid(format!(
                    "MLIR function '{name}' is not available"
                )));
            };
            Ok(LayoutRequest::Layout(mlir_function_or_region(
                function, region,
            )))
        }
    }
}

fn mlir_function_or_region(function: usize, region: Option<usize>) -> EntityHandle {
    region.map_or(EntityHandle::MlirFunction { function }, |region| {
        EntityHandle::MlirRegion {
            scope: format!("function:{function}"),
            region,
        }
    })
}

fn find_mlir_function(session: &ModelSession, name: &str) -> Result<EntityHandle, CliFailure> {
    let limits = SessionLimits {
        search: SessionLimits::HARD_MAX.search,
        ..SessionLimits::default()
    };
    let hits = session.search(name, &limits);
    let mut fallback = None;
    for hit in hits {
        if let EntityHandle::MlirFunction { .. } = hit.handle {
            if hit.name.as_deref() == Some(name) {
                return Ok(hit.handle);
            }
            fallback.get_or_insert(hit.handle);
        }
    }
    fallback.ok_or_else(|| CliFailure::invalid(format!("MLIR function '{name}' is not available")))
}

struct MappedModel {
    path: PathBuf,
    data: ModelData,
}

enum ModelData {
    Mapped(Mmap),
}

impl AsRef<[u8]> for ModelData {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Mapped(data) => data,
        }
    }
}

impl MappedModel {
    fn open(path: &Path) -> Result<Self, CliFailure> {
        let metadata = fs::metadata(path)?;
        if metadata.is_dir() {
            return Err(CliFailure::invalid(format!(
                "unsupported model directory '{}'",
                path.display()
            )));
        }
        let file = File::open(path)?;
        // SAFETY: the mmap is read-only and lives as long as every borrowed parser input.
        let data = ModelData::Mapped(unsafe { Mmap::map(&file)? });
        Ok(Self {
            path: path.to_owned(),
            data,
        })
    }

    fn input(&self) -> ModelInput<'_> {
        ModelInput {
            data: self.data.as_ref(),
            path: Some(self.path.as_path()),
            allow_unsafe_paths: false,
        }
    }

    fn source(&self) -> ModelSource {
        ModelSource::from_file(self.path.clone(), self.len()).with_allow_unsafe_paths(false)
    }

    fn bytes(&self) -> &[u8] {
        self.data.as_ref()
    }

    fn len(&self) -> usize {
        self.data.as_ref().len()
    }
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
    fn run(mapped: &MappedModel, iterations: usize) -> Result<Self, CliFailure> {
        let mut parse_times = Vec::with_capacity(iterations);
        let mut last_model = None;
        for _ in 0..iterations {
            let start = Instant::now();
            let model = netron_rs_formats::parse(mapped.input()).map_err(CliFailure::from)?;
            parse_times.push(start.elapsed().as_secs_f64() * 1000.0);
            last_model = Some(model);
        }
        let model = last_model.expect("iterations is at least one");
        let json_start = Instant::now();
        let _json = model.to_normalized_json().map_err(CliFailure::from)?;
        let json_ms_last = json_start.elapsed().as_secs_f64() * 1000.0;
        let layout_start = Instant::now();
        if !model.graphs.is_empty() {
            let _layout =
                netron_rs_layout::layout_model(&model, &netron_rs_layout::LayoutOptions::default())
                    .map_err(|error| CliFailure::internal(error.to_string()))?;
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
