use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Output, Stdio};

#[test]
fn stats_reports_model_size_without_tensor_materialization() {
    let model = write_fixture("stats.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("stats")
        .arg(&model)
        .output()
        .expect("run stats");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stats: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(stats["format"], "ONNX");
    assert_eq!(stats["graphs"], 1);
    assert_eq!(stats["nodes"], 1);
    assert_eq!(stats["tensors"], 1);
    assert_eq!(stats["tensor_inline_bytes"], 12);
}

#[test]
fn bench_reports_repeatable_parse_timing() {
    let model = write_fixture("bench.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("bench")
        .arg(&model)
        .arg("2")
        .output()
        .expect("run bench");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bench: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(bench["iterations"], 2);
    assert_eq!(bench["stats"]["format"], "ONNX");
    assert!(bench["parse_bytes_per_second_mean"].as_f64().unwrap() > 0.0);
    assert!(bench["parse_ms_mean"].as_f64().unwrap() >= 0.0);
    assert!(bench["parse_ms_last"].as_f64().unwrap() >= 0.0);
    assert!(bench["json_ms_last"].as_f64().unwrap() >= 0.0);
    assert!(bench["layout_ms_last"].as_f64().unwrap() >= 0.0);
    assert!(bench["search_index_ms_last"].as_f64().unwrap() >= 0.0);
    assert!(bench["parse_and_json_ms_last"].as_f64().unwrap() >= 0.0);
    assert!(bench["parse_json_layout_ms_last"].as_f64().unwrap() >= 0.0);
    assert!(bench["time_to_first_graph_ms_last"].as_f64().unwrap() >= 0.0);
}

#[test]
fn search_reports_bounded_query_hits() {
    let model = write_fixture("search.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("search")
        .arg(&model)
        .arg("add")
        .arg("1")
        .output()
        .expect("run search");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hits: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(hits.as_array().unwrap().len(), 1);
    assert_eq!(hits[0]["kind"], "node");
    assert_eq!(hits[0]["handle"]["kind"], "node");
    assert_eq!(hits[0]["handle"]["graph"], 0);
    assert_eq!(hits[0]["operator"], "Add");
    assert_eq!(hits[0]["graph"], 0);
}

#[test]
fn search_json_wraps_bounded_query_hits() {
    let model = write_fixture("search-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("search")
        .arg(&model)
        .arg("add")
        .arg("--limit")
        .arg("1")
        .arg("--json")
        .output()
        .expect("run search");

    let envelope = success_json(output);
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["status"], "ok");
    assert_eq!(envelope["command"], "search");
    assert_eq!(envelope["data"].as_array().unwrap().len(), 1);
    assert_eq!(envelope["data"][0]["handle"]["kind"], "node");
}

#[test]
fn summary_reports_indexed_session_info() {
    let model = write_fixture("summary.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("summary")
        .arg(&model)
        .output()
        .expect("run summary");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["format"], "onnx");
    assert_eq!(summary["graphs"], 1);
    assert_eq!(summary["nodes"], 1);
    assert_eq!(summary["tensors"], 1);
    assert_eq!(summary["opsets"], 1);
    assert_eq!(summary["onnx"]["opsets"][0]["version"], 18);
    assert_eq!(summary["source"]["kind"]["kind"], "file");
    assert!(
        summary["source"]["content_identity"]
            .as_str()
            .unwrap()
            .starts_with("fnv1a64:")
    );
}

#[test]
fn summary_json_wraps_session_info() {
    let model = write_fixture("summary-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("summary")
        .arg(&model)
        .arg("--json")
        .output()
        .expect("run summary");

    let envelope = success_json(output);
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["status"], "ok");
    assert_eq!(envelope["command"], "summary");
    assert_eq!(envelope["data"]["format"], "onnx");
    assert_eq!(envelope["data"]["graphs"], 1);
}

#[test]
fn summary_supports_mlir_input() {
    let path = std::env::temp_dir()
        .join(format!(
            "netron-rs-summary-mlir-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
        .with_extension("mlir");
    fs::write(
        &path,
        "module {\n  func.func @main() {\n    %c0 = arith.constant 0 : i32\n    return\n  }\n}\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("summary")
        .arg(&path)
        .output()
        .expect("run summary");
    fs::remove_file(&path).unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["format"], "mlir");
    assert_eq!(summary["functions"], 1);
}

#[test]
fn mlir_symbols_json_reports_symbol_handles() {
    let path = write_mlir_fixture(
        "mlir-symbols",
        "module @m {\n  func.func @main() {\n    return\n  }\n}\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("mlir")
        .arg("symbols")
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run mlir symbols");
    fs::remove_file(&path).unwrap();

    let envelope = success_json(output);
    assert_eq!(envelope["command"], "mlir.symbols");
    let symbols = envelope["data"].as_array().unwrap();
    assert!(symbols.iter().any(|symbol| {
        symbol["handle"]["kind"] == "mlir_symbol"
            && symbol["name"].as_str().unwrap().contains("main")
    }));
}

#[test]
fn detail_json_reports_entity_fields() {
    let model = write_fixture("detail-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("detail")
        .arg(&model)
        .arg("--node")
        .arg("0")
        .arg("--json")
        .output()
        .expect("run detail");

    let envelope = success_json(output);
    assert_eq!(envelope["command"], "detail");
    assert_eq!(envelope["data"]["handle"]["kind"], "node");
    assert_eq!(envelope["data"]["handle"]["node"], 0);
    assert_eq!(envelope["data"]["fields"]["operator"], "Add");
}

#[test]
fn layout_reports_format_independent_view_graph() {
    let model = write_fixture("layout.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("layout")
        .arg(&model)
        .arg("0")
        .arg("10")
        .output()
        .expect("run layout");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let layout: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let graph = &layout["graphs"][0];
    assert_eq!(graph["graph"], 0);
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 3);
    assert_eq!(graph["edges"].as_array().unwrap().len(), 2);
    assert!(
        graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["kind"] == "operator" && node["operator"] == "Add")
    );
    assert_eq!(graph["stats"]["omitted_initializers"], 1);
}

#[test]
fn layout_mlir_function_json_reports_projection() {
    let path = write_mlir_fixture(
        "layout-mlir-function",
        "module {\n  func.func @main() {\n    return\n  }\n}\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("layout")
        .arg(&path)
        .arg("--function")
        .arg("@main")
        .arg("--region")
        .arg("0")
        .arg("--max-ops")
        .arg("10")
        .arg("--json")
        .output()
        .expect("run mlir function layout");
    fs::remove_file(&path).unwrap();

    let envelope = success_json(output);
    assert_eq!(envelope["command"], "layout");
    assert_eq!(envelope["data"]["scope"]["kind"], "mlir_region");
    assert_eq!(envelope["data"]["scope"]["region"], 0);
    assert_eq!(envelope["data"]["limit_used"], 10);
    assert!(
        envelope["data"]["cache_key"]
            .as_str()
            .unwrap()
            .contains("layout")
    );
}

#[test]
fn layout_json_reports_session_projection() {
    let model = write_fixture("layout-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("layout")
        .arg(&model)
        .arg("--graph")
        .arg("0")
        .arg("--max-nodes")
        .arg("1")
        .arg("--json")
        .output()
        .expect("run layout");

    let envelope = success_json(output);
    assert_eq!(envelope["command"], "layout");
    assert_eq!(envelope["data"]["scope"]["kind"], "graph");
    assert_eq!(envelope["data"]["limit_used"], 1);
    assert!(
        envelope["data"]["cache_key"]
            .as_str()
            .unwrap()
            .contains("layout")
    );
    assert!(envelope["data"]["warnings"].is_array());
}

#[test]
fn layout_node_json_returns_slice_projection() {
    let model = write_fixture("layout-node-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("layout")
        .arg(&model)
        .arg("--node")
        .arg("0")
        .arg("--depth")
        .arg("2")
        .arg("--max-nodes")
        .arg("2")
        .arg("--json")
        .output()
        .expect("run layout node");

    let envelope = success_json(output);
    assert_eq!(envelope["command"], "layout");
    assert_eq!(envelope["data"]["scope"]["kind"], "node");
    assert_eq!(envelope["data"]["scope"]["node"], 0);
    assert_eq!(envelope["data"]["limit_used"], 2);
    assert!(!envelope["data"]["entities"].as_array().unwrap().is_empty());
    assert!(
        envelope["data"]["cache_key"]
            .as_str()
            .unwrap()
            .contains("slice")
    );
}

#[test]
fn onnx_tensor_json_reports_tensor_metadata() {
    let model = write_fixture("onnx-tensor-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("onnx")
        .arg("tensor")
        .arg(&model)
        .arg("--tensor")
        .arg("0")
        .arg("--json")
        .output()
        .expect("run onnx tensor");

    let envelope = success_json(output);
    assert_eq!(envelope["command"], "onnx.tensor");
    assert_eq!(envelope["data"]["handle"]["kind"], "tensor");
    assert_eq!(envelope["data"]["handle"]["tensor"], 0);
    assert_eq!(envelope["data"]["storage"], "inline_bytes");
    assert_eq!(envelope["data"]["byte_len"], 12);
}

#[test]
fn json_errors_use_stable_exit_codes() {
    let bad = std::env::temp_dir().join(format!(
        "netron-rs-unsupported-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&bad, b"not a model").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("summary")
        .arg(&bad)
        .arg("--json")
        .output()
        .expect("run summary");
    fs::remove_file(&bad).unwrap();
    let envelope = error_json(output, 2);
    assert_eq!(envelope["command"], "summary");
    assert_eq!(envelope["error"]["code"], "unsupported_format");

    let model = write_fixture("invalid-layout-json.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("layout")
        .arg(&model)
        .arg("--node")
        .arg("missing")
        .arg("--json")
        .output()
        .expect("run layout");
    let envelope = error_json(output, 4);
    assert_eq!(envelope["command"], "layout");
    assert_eq!(envelope["error"]["code"], "invalid_request");

    let missing = std::env::temp_dir().join("netron-rs-missing-json-error.onnx");
    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("summary")
        .arg(&missing)
        .arg("--json")
        .output()
        .expect("run missing summary");
    let envelope = error_json(output, 4);
    assert_eq!(envelope["command"], "summary");
    assert_eq!(envelope["error"]["code"], "invalid_request");
}

#[test]
fn service_stdio_matches_cli_for_shared_operations() {
    let onnx = write_fixture("service-summary.onnx");
    let cli_summary = success_json(
        Command::new(env!("CARGO_BIN_EXE_netron-rs"))
            .arg("summary")
            .arg(&onnx)
            .arg("--json")
            .output()
            .expect("run cli summary"),
    );

    let mlir = write_mlir_fixture(
        "service-search",
        "module {\n  func.func @main() {\n    return\n  }\n}\n",
    );
    let cli_search = success_json(
        Command::new(env!("CARGO_BIN_EXE_netron-rs"))
            .arg("search")
            .arg(&mlir)
            .arg("main")
            .arg("--limit")
            .arg("1")
            .arg("--json")
            .output()
            .expect("run cli search"),
    );

    let mut service = ServiceProcess::spawn();
    let open = service.request(serde_json::json!({
        "id": 1,
        "method": "open",
        "params": { "path": onnx }
    }));
    assert_eq!(open["status"], "ok");
    let onnx_session = open["data"]["session"].as_u64().unwrap();
    let summary = service.request(serde_json::json!({
        "id": 2,
        "method": "summary",
        "params": { "session": onnx_session }
    }));
    assert_eq!(summary["command"], "summary");
    assert_eq!(summary["data"]["format"], cli_summary["data"]["format"]);
    assert_eq!(summary["data"]["graphs"], cli_summary["data"]["graphs"]);
    assert_eq!(summary["data"]["nodes"], cli_summary["data"]["nodes"]);
    let diagnostics = service.request(serde_json::json!({
        "id": 21,
        "method": "diagnostics",
        "params": { "session": onnx_session, "limit": 4 }
    }));
    assert_eq!(diagnostics["command"], "diagnostics");
    assert!(diagnostics["data"]["diagnostics"].is_array());
    let detail = service.request(serde_json::json!({
        "id": 22,
        "method": "detail",
        "params": {
            "session": onnx_session,
            "handle": { "kind": "node", "graph": 0, "node": 0 }
        }
    }));
    assert_eq!(detail["data"]["fields"]["operator"], "Add");
    let slice = service.request(serde_json::json!({
        "id": 23,
        "method": "slice",
        "params": {
            "session": onnx_session,
            "max_nodes": 2,
            "handle": { "kind": "node", "graph": 0, "node": 0 }
        }
    }));
    assert_eq!(slice["data"]["scope"]["kind"], "node");
    assert_eq!(slice["data"]["limit_used"], 2);
    let layout = service.request(serde_json::json!({
        "id": 24,
        "method": "layout",
        "params": {
            "session": onnx_session,
            "max_nodes": 1,
            "handle": { "kind": "graph", "graph": 0 }
        }
    }));
    assert_eq!(layout["data"]["scope"]["kind"], "graph");
    assert_eq!(layout["data"]["limit_used"], 1);
    let export = service.request(serde_json::json!({
        "id": 25,
        "method": "export",
        "params": { "session": onnx_session, "limit": 5 }
    }));
    assert_eq!(export["command"], "export");
    assert_eq!(export["data"]["limit_used"], 5);
    assert!(export["data"]["omitted_count"].is_number());
    assert!(export["data"]["normalized"].is_object());

    let open = service.request(serde_json::json!({
        "id": 3,
        "method": "open",
        "params": { "path": mlir }
    }));
    let mlir_session = open["data"]["session"].as_u64().unwrap();
    let search = service.request(serde_json::json!({
        "id": 4,
        "method": "search",
        "params": { "session": mlir_session, "query": "main", "limit": 1 }
    }));
    assert_eq!(search["command"], "search");
    assert_eq!(search["data"], cli_search["data"]);
    let close = service.request(serde_json::json!({
        "id": 5,
        "method": "close",
        "params": { "session": mlir_session }
    }));
    assert_eq!(close["data"]["closed"], true);
}

#[test]
fn parse_rejects_directory_input() {
    let root = temp_root("netron-rs-directory");
    fs::create_dir_all(&root).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
        .arg("parse")
        .arg(&root)
        .output()
        .expect("run parse");
    fs::remove_dir_all(&root).unwrap();

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unsupported model directory"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

struct ServiceProcess {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl ServiceProcess {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
            .arg("serve")
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn service");
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            input,
            output,
        }
    }

    fn request(&mut self, request: serde_json::Value) -> serde_json::Value {
        writeln!(self.input, "{request}").unwrap();
        self.input.flush().unwrap();
        let mut line = String::new();
        self.output.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }
}

impl Drop for ServiceProcess {
    fn drop(&mut self) {
        let _ = self.input.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn success_json(output: Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn error_json(output: Output, code: i32) -> serde_json::Value {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stderr).unwrap()
}

fn temp_root(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn write_mlir_fixture(name: &str, text: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir()
        .join(format!(
            "netron-rs-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
        .with_extension("mlir");
    fs::write(&path, text).unwrap();
    path
}

fn write_fixture(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("netron-rs-{name}"));
    fs::write(&path, fixture_model()).unwrap();
    path
}

fn fixture_model() -> Vec<u8> {
    let mut model = Vec::new();
    varint(&mut model, 1, 9);
    string(&mut model, 2, "netron-rs-cli-test");
    message(&mut model, 7, fixture_graph());
    message(&mut model, 8, opset("", 18));
    model
}

fn fixture_graph() -> Vec<u8> {
    let mut graph = Vec::new();
    string(&mut graph, 2, "main");
    message(&mut graph, 5, tensor("w", &[3], 1, &[0; 12]));
    message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
    message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));
    message(&mut graph, 1, node());
    graph
}

fn node() -> Vec<u8> {
    let mut node = Vec::new();
    string(&mut node, 1, "x");
    string(&mut node, 1, "w");
    string(&mut node, 2, "y");
    string(&mut node, 4, "Add");
    node
}

fn value_info(name: &str, elem_type: u64, dims: &[Vec<u8>]) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in dims {
        message(&mut shape, 1, dim.clone());
    }

    let mut tensor_type = Vec::new();
    varint(&mut tensor_type, 1, elem_type);
    message(&mut tensor_type, 2, shape);

    let mut type_proto = Vec::new();
    message(&mut type_proto, 1, tensor_type);

    let mut value = Vec::new();
    string(&mut value, 1, name);
    message(&mut value, 2, type_proto);
    value
}

fn tensor(name: &str, dims: &[u64], data_type: u64, raw_data: &[u8]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for dim in dims {
        varint(&mut tensor, 1, *dim);
    }
    varint(&mut tensor, 2, data_type);
    string(&mut tensor, 8, name);
    bytes(&mut tensor, 9, raw_data);
    tensor
}

fn opset(domain: &str, version: u64) -> Vec<u8> {
    let mut opset = Vec::new();
    string(&mut opset, 1, domain);
    varint(&mut opset, 2, version);
    opset
}

fn dim_value(value: u64) -> Vec<u8> {
    let mut dim = Vec::new();
    varint(&mut dim, 1, value);
    dim
}

fn varint(output: &mut Vec<u8>, field: u64, value: u64) {
    encode_varint(output, field << 3);
    encode_varint(output, value);
}

fn string(output: &mut Vec<u8>, field: u64, value: &str) {
    bytes(output, field, value.as_bytes());
}

fn message(output: &mut Vec<u8>, field: u64, value: Vec<u8>) {
    bytes(output, field, &value);
}

fn bytes(output: &mut Vec<u8>, field: u64, value: &[u8]) {
    encode_varint(output, (field << 3) | 2);
    encode_varint(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn encode_varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}
