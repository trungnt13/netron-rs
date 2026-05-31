use std::fs;
use std::process::Command;

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
    assert_eq!(hits[0]["operator"], "Add");
    assert_eq!(hits[0]["graph"], 0);
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
    assert_eq!(summary["source"]["kind"]["kind"], "file");
    assert!(
        summary["source"]["content_identity"]
            .as_str()
            .unwrap()
            .starts_with("fnv1a64:")
    );
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

fn temp_root(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
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
