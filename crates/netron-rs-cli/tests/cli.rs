use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, ChildStdin, ChildStdout, Command, Output, Stdio};
use std::thread;
use std::time::Duration;

#[test]
fn stats_reports_model_size_without_tensor_materialization() {
  let model = write_fixture("stats.onnx");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("stats")
    .arg(&model)
    .output()
    .expect("run stats");

  let stats = success_json(output);
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

  let bench = success_json(output);
  assert_eq!(bench["iterations"], 2);
  assert_eq!(bench["stats"]["format"], "ONNX");
  assert!(bench["parse_bytes_per_second_mean"].as_f64().unwrap() > 0.0);
  assert!(bench["parse_ms_mean"].as_f64().unwrap() >= 0.0);
  assert!(bench["parse_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["json_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["layout_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["search_index_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["session_open_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["summary_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["summary_json_bytes_last"].as_u64().unwrap() > 0);
  assert!(bench["search_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["tensor_metadata_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["session_layout_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["peak_rss_kb_last"].as_u64().unwrap() > 0);
  assert!(bench["parse_and_json_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["parse_json_layout_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["time_to_first_graph_ms_last"].as_f64().unwrap() >= 0.0);
}

#[test]
fn bench_covers_repo_local_external_data_fixture() {
  let model = onnx_fixture("external-chain.onnx");
  let payload = onnx_fixture("external-chain.bin");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("bench")
    .arg(&model)
    .arg("1")
    .output()
    .expect("run external-data bench");

  let bench = success_json(output);
  assert_eq!(bench["stats"]["format"], "ONNX");
  assert!(bench["stats"]["nodes"].as_u64().unwrap() >= 1000);
  assert_eq!(bench["stats"]["tensor_external"], 1);
  assert!(bench["session_open_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["summary_json_bytes_last"].as_u64().unwrap() > 0);
  assert!(
    fs::metadata(payload).unwrap().len() > fs::metadata(model).unwrap().len(),
    "external payload should dominate fixture size"
  );
}

#[test]
fn bench_covers_repo_local_mlirbc_fixture() {
  let model = mlirbc_fixture("sd-clip-tank.mlirbc");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("bench")
    .arg(&model)
    .arg("1")
    .output()
    .expect("run mlirbc bench");

  let bench = success_json(output);
  assert_eq!(bench["stats"]["format"], "MLIR");
  assert!(bench["stats"]["functions"].as_u64().unwrap() >= 2);
  assert!(bench["session_open_ms_last"].as_f64().unwrap() >= 0.0);
  assert!(bench["summary_json_bytes_last"].as_u64().unwrap() > 0);
  assert!(bench["peak_rss_kb_last"].as_u64().unwrap() > 0);
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

  let hits = success_json(output);
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
fn search_cursor_returns_paged_envelope() {
  let model = write_fixture("search-page.onnx");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("search")
    .arg(&model)
    .arg("add")
    .arg("--limit")
    .arg("1")
    .arg("--cursor")
    .arg("0")
    .arg("--json")
    .output()
    .expect("run paged search");

  let envelope = success_json(output);
  let data = &envelope["data"];
  assert_eq!(data["api_version"], 1);
  assert_eq!(data["format"], "onnx");
  assert_eq!(data["query"], "add");
  assert_eq!(data["limit_used"], 1);
  assert_eq!(data["total_count"], 1);
  assert_eq!(data["truncated"], false);
  assert_eq!(data["omitted_count"], 0);
  assert_eq!(data["results"].as_array().unwrap().len(), 1);
  assert_eq!(data["results"][0]["handle"]["kind"], "node");
  assert!(!data.as_object().unwrap().contains_key("next_cursor"));
}

#[test]
fn summary_reports_indexed_session_info() {
  let model = write_fixture("summary.onnx");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("summary")
    .arg(&model)
    .output()
    .expect("run summary");

  let summary = success_json(output);
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
  let path = write_mlir_fixture(
    "summary-mlir",
    "module {\n  func.func @main() {\n    %c0 = arith.constant 0 : i32\n    return\n  }\n}\n",
  );
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("summary")
    .arg(&path)
    .output()
    .expect("run summary");
  fs::remove_file(&path).unwrap();

  let summary = success_json(output);
  assert_eq!(summary["format"], "mlir");
  assert_eq!(summary["functions"], 1);
}

#[test]
fn summary_supports_mlir_bytecode_input() {
  let path = mlirbc_fixture("model.mlirbc");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("summary")
    .arg(&path)
    .arg("--json")
    .output()
    .expect("run mlirbc summary");

  let envelope = success_json(output);
  assert_eq!(envelope["command"], "summary");
  assert_eq!(envelope["data"]["format"], "mlir");
  assert_eq!(envelope["data"]["source_format_name"], "MLIR");
  assert!(envelope["data"]["functions"].as_u64().unwrap() > 0);
  assert_eq!(
    envelope["data"]["functions"],
    envelope["data"]["mlir"]["function_count"]
  );
  assert_eq!(
    envelope["data"]["mlir"]["modules"][0]["name"],
    "compiled_mmdit"
  );
  assert_eq!(
    envelope["data"]["mlir"]["functions"][0]["name"],
    "run_forward"
  );
  assert!(
    !envelope["data"]["mlir"]["regions"]
      .as_array()
      .unwrap()
      .is_empty()
  );
  assert!(
    !envelope["data"]["mlir"]["blocks"]
      .as_array()
      .unwrap()
      .is_empty()
  );
  assert_eq!(envelope["data"]["mlir"]["blocks"][0]["label"], "^bb0");
  assert!(
    envelope["data"]["mlir"]["dialects"]
      .as_array()
      .unwrap()
      .iter()
      .any(|dialect| dialect == "torch")
  );
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
    symbol["handle"]["kind"] == "mlir_symbol" && symbol["name"].as_str().unwrap().contains("main")
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
fn detail_json_reports_mlir_bytecode_attribute_fields() {
  let model = mlirbc_fixture("model.mlirbc");
  let search = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("search")
    .arg("--json")
    .arg(&model)
    .arg("0.000001")
    .arg("--limit")
    .arg("200")
    .output()
    .expect("run bytecode attribute search");
  let hits = success_json(search);
  let handle = hits["data"]
    .as_array()
    .unwrap()
    .iter()
    .find_map(|hit| {
      (hit["handle"]["kind"] == "mlir_attribute" && hit["handle"]["scope"] == "bytecode")
        .then(|| hit["handle"].clone())
    })
    .expect("bytecode attribute search hit");

  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("detail")
    .arg("--json")
    .arg("--scope")
    .arg(handle["scope"].as_str().unwrap())
    .arg("--mlir-attribute")
    .arg(handle["attribute"].as_u64().unwrap().to_string())
    .arg(&model)
    .output()
    .expect("run bytecode attribute detail");

  let envelope = success_json(output);
  assert_eq!(envelope["command"], "detail");
  assert_eq!(envelope["data"]["handle"]["kind"], "mlir_attribute");
  assert_eq!(envelope["data"]["handle"]["scope"], "bytecode");
  assert_eq!(envelope["data"]["fields"]["value_0"], "0.000001 : f64");
}

#[test]
fn detail_json_reports_mlir_bytecode_value_fields() {
  let model = mlirbc_fixture("model.mlirbc");
  let search = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("search")
    .arg("--json")
    .arg(&model)
    .arg("operation_result")
    .arg("--limit")
    .arg("50")
    .output()
    .expect("run bytecode value search");
  let hits = success_json(search);
  let handle = hits["data"]
    .as_array()
    .unwrap()
    .iter()
    .find_map(|hit| {
      (hit["handle"]["kind"] == "mlir_value" && hit["handle"]["scope"] == "function:0")
        .then(|| hit["handle"].clone())
    })
    .expect("bytecode value search hit");

  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("detail")
    .arg("--json")
    .arg("--scope")
    .arg(handle["scope"].as_str().unwrap())
    .arg("--mlir-value")
    .arg(handle["value"].as_u64().unwrap().to_string())
    .arg(&model)
    .output()
    .expect("run bytecode value detail");

  let envelope = success_json(output);
  assert_eq!(envelope["command"], "detail");
  assert_eq!(envelope["data"]["handle"]["kind"], "mlir_value");
  assert_eq!(envelope["data"]["handle"]["scope"], "function:0");
  assert!(
    envelope["data"]["fields"]["name"]
      .as_str()
      .unwrap()
      .starts_with('%')
  );
  assert_eq!(
    envelope["data"]["fields"]["metadata.bytecode.value.kind"],
    "operation_result"
  );
  assert!(
    envelope["data"]["fields"]["metadata.bytecode.value"]
      .as_str()
      .unwrap()
      .parse::<usize>()
      .is_ok()
  );
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

  let layout = success_json(output);
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
fn layout_mlir_function_json_accepts_structural_collapse() {
  let path = write_mlir_fixture(
    "layout-mlir-structural",
    "module {\n  func.func @main(%i: index) -> i64 {\n    %0 = arith.index_cast %i : index to i64\n    return %0 : i64\n  }\n}\n",
  );
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("layout")
    .arg(&path)
    .arg("--function")
    .arg("@main")
    .arg("--collapse")
    .arg("structural")
    .arg("--json")
    .output()
    .expect("run structural mlir layout");
  fs::remove_file(&path).unwrap();

  let envelope = success_json(output);
  assert_eq!(envelope["command"], "layout");
  assert_eq!(envelope["data"]["collapse"], "structural");
  assert!(
    envelope["data"]["cache_key"]
      .as_str()
      .unwrap()
      .contains("collapse:structural")
  );
  assert!(
    envelope["data"]["collapsed_groups"]
      .as_array()
      .unwrap()
      .iter()
      .any(|group| group["kind"] == "mlir_block")
  );
  assert!(
    envelope["data"]["graph"]["nodes"]
      .as_array()
      .unwrap()
      .iter()
      .all(|node| node["kind"] == "group")
  );
}

#[test]
fn layout_onnx_json_accepts_structural_collapse() {
  let path = write_repeated_pair_fixture("layout-onnx-structural.onnx");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("layout")
    .arg(&path)
    .arg("--collapse")
    .arg("structural")
    .arg("--json")
    .output()
    .expect("run structural onnx layout");
  fs::remove_file(&path).unwrap();

  let envelope = success_json(output);
  assert_eq!(envelope["command"], "layout");
  assert_eq!(envelope["data"]["collapse"], "structural");
  assert!(
    envelope["data"]["collapsed_groups"]
      .as_array()
      .unwrap()
      .iter()
      .any(|group| group["kind"] == "onnx_repeated_block")
  );
  assert!(
    envelope["data"]["graph"]["nodes"]
      .as_array()
      .unwrap()
      .iter()
      .any(|node| node["kind"] == "group")
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
      .contains("depth:2")
  );
}

#[test]
fn layout_node_without_json_returns_slice_projection() {
  let model = write_fixture("layout-node.onnx");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("layout")
    .arg(&model)
    .arg("--node")
    .arg("0")
    .arg("--depth")
    .arg("0")
    .arg("--max-nodes")
    .arg("10")
    .output()
    .expect("run layout node");

  let slice = success_json(output);
  assert_eq!(slice["scope"]["kind"], "node");
  assert_eq!(slice["scope"]["node"], 0);
  assert_eq!(slice["limit_used"], 10);
  assert!(slice["cache_key"].as_str().unwrap().contains("depth:0"));
}

#[test]
fn layout_depth_requires_node_selector() {
  let model = write_fixture("layout-depth-requires-node.onnx");
  let output = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
    .arg("layout")
    .arg(&model)
    .arg("--depth")
    .arg("2")
    .arg("--json")
    .output()
    .expect("run layout");

  let envelope = error_json(output, 4);
  assert_eq!(envelope["command"], "layout");
  assert_eq!(envelope["error"]["code"], "invalid_request");
  assert!(
    envelope["error"]["message"]
      .as_str()
      .unwrap()
      .contains("--depth requires --node")
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
  let bad = temp_root("netron-rs-unsupported");
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
  let cli_tensor = success_json(
    Command::new(env!("CARGO_BIN_EXE_netron-rs"))
      .arg("onnx")
      .arg("tensor")
      .arg(&onnx)
      .arg("--tensor")
      .arg("0")
      .arg("--json")
      .output()
      .expect("run cli tensor"),
  );

  let mlir = write_mlir_fixture(
    "service-search",
    "module {\n  func.func @main(%i: index) -> i64 {\n    %0 = arith.index_cast %i : index to i64\n    return %0 : i64\n  }\n}\n",
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
  let cli_symbols = success_json(
    Command::new(env!("CARGO_BIN_EXE_netron-rs"))
      .arg("mlir")
      .arg("symbols")
      .arg(&mlir)
      .arg("--json")
      .output()
      .expect("run cli mlir symbols"),
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
  let tensor = service.request(serde_json::json!({
      "id": 25,
      "method": "onnx.tensor",
      "params": { "session": onnx_session, "tensor": 0 }
  }));
  assert_eq!(tensor["command"], "onnx.tensor");
  assert_eq!(tensor["data"], cli_tensor["data"]);
  let export = service.request(serde_json::json!({
      "id": 26,
      "method": "export",
      "params": { "session": onnx_session, "limit": 5 }
  }));
  assert_eq!(export["command"], "export");
  assert_eq!(export["data"]["session"], onnx_session);
  assert!(export["data"]["session_id"].is_number());
  assert_eq!(export["data"]["limit_used"], 5);
  assert!(export["data"]["omitted_count"].is_number());
  assert!(export["data"]["normalized"].is_object());
  let paged_search = service.request(serde_json::json!({
      "id": 27,
      "method": "search",
      "params": { "session": onnx_session, "query": "add", "limit": 1, "cursor": null }
  }));
  assert_eq!(paged_search["command"], "search");
  assert_eq!(paged_search["data"]["format"], "onnx");
  assert_eq!(paged_search["data"]["limit_used"], 1);
  assert_eq!(paged_search["data"]["results"].as_array().unwrap().len(), 1);
  assert!(
    !paged_search["data"]
      .as_object()
      .unwrap()
      .contains_key("next_cursor")
  );
  let invalid_cursor = service.request(serde_json::json!({
      "id": 28,
      "method": "search",
      "params": { "session": onnx_session, "query": "add", "limit": 1, "cursor": "1" }
  }));
  assert_eq!(invalid_cursor["status"], "error");
  assert_eq!(invalid_cursor["error"]["code"], "invalid_request");

  let repeated = write_repeated_pair_fixture("service-onnx-structural.onnx");
  let open = service.request(serde_json::json!({
      "id": 29,
      "method": "open",
      "params": { "path": repeated }
  }));
  assert_eq!(open["status"], "ok");
  let repeated_session = open["data"]["session"].as_u64().unwrap();
  let onnx_structural_slice = service.request(serde_json::json!({
      "id": 30,
      "method": "slice",
      "params": {
          "session": repeated_session,
          "collapse": "structural",
          "handle": { "kind": "graph", "graph": 0 }
      }
  }));
  assert_eq!(onnx_structural_slice["command"], "slice");
  assert_eq!(onnx_structural_slice["data"]["collapse"], "structural");
  assert!(
    onnx_structural_slice["data"]["collapsed_groups"]
      .as_array()
      .unwrap()
      .iter()
      .any(|group| group["kind"] == "onnx_repeated_block")
  );
  let onnx_structural_layout = service.request(serde_json::json!({
      "id": 31,
      "method": "layout",
      "params": {
          "session": repeated_session,
          "collapse": "structural",
          "handle": { "kind": "graph", "graph": 0 }
      }
  }));
  assert_eq!(onnx_structural_layout["command"], "layout");
  assert_eq!(onnx_structural_layout["data"]["collapse"], "structural");
  assert!(
    onnx_structural_layout["data"]["collapsed_groups"]
      .as_array()
      .unwrap()
      .iter()
      .any(|group| group["kind"] == "onnx_repeated_block")
  );
  let close = service.request(serde_json::json!({
      "id": 32,
      "method": "close",
      "params": { "session": repeated_session }
  }));
  assert_eq!(close["data"]["closed"], true);

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
  let symbols = service.request(serde_json::json!({
      "id": 6,
      "method": "mlir.symbols",
      "params": { "session": mlir_session, "limit": 20 }
  }));
  assert_eq!(symbols["command"], "mlir.symbols");
  assert_eq!(symbols["data"], cli_symbols["data"]);
  let symbol_tree = service.request(serde_json::json!({
      "id": 63,
      "method": "mlir.symbol.tree",
      "params": { "session": mlir_session, "limit": 20 }
  }));
  assert_eq!(symbol_tree["command"], "mlir.symbol.tree");
  assert_eq!(symbol_tree["data"]["api_version"], 1);
  let symbol_tree_roots = symbol_tree["data"]["roots"].as_array().unwrap();
  assert!(
    symbol_tree_roots
      .iter()
      .any(|node| node["label"].as_str() == Some("Functions"))
  );
  assert!(
    symbol_tree_roots
      .iter()
      .any(|node| node["label"].as_str() == Some("Symbols"))
  );
  let structural_slice = service.request(serde_json::json!({
      "id": 61,
      "method": "slice",
      "params": {
          "session": mlir_session,
          "collapse": "structural",
          "handle": { "kind": "mlir_function", "function": 0 }
      }
  }));
  assert_eq!(structural_slice["command"], "slice");
  assert_eq!(structural_slice["data"]["collapse"], "structural");
  assert!(
    structural_slice["data"]["collapsed_groups"]
      .as_array()
      .unwrap()
      .iter()
      .any(|group| group["kind"] == "mlir_block")
  );
  let structural_layout = service.request(serde_json::json!({
      "id": 62,
      "method": "layout",
      "params": {
          "session": mlir_session,
          "collapse": "structural",
          "handle": { "kind": "mlir_function", "function": 0 }
      }
  }));
  assert_eq!(structural_layout["command"], "layout");
  assert_eq!(structural_layout["data"]["collapse"], "structural");
  assert!(
    structural_layout["data"]["graph"]["nodes"]
      .as_array()
      .unwrap()
      .iter()
      .all(|node| node["kind"] == "group")
  );
  let close = service.request(serde_json::json!({
      "id": 7,
      "method": "close",
      "params": { "session": mlir_session }
  }));
  assert_eq!(close["data"]["closed"], true);
}

#[test]
fn service_http_accepts_json_requests_with_token() {
  let onnx = write_fixture("service-http.onnx");
  let service = HttpServiceProcess::spawn();

  let open = service.request(serde_json::json!({
      "id": 1,
      "method": "open",
      "params": { "path": onnx }
  }));
  assert_eq!(open["status"], "ok");
  let session = open["data"]["session"].as_u64().unwrap();

  let search = service.request(serde_json::json!({
      "id": 2,
      "method": "search",
      "params": { "session": session, "query": "add", "limit": 1, "cursor": null }
  }));
  assert_eq!(search["status"], "ok");
  assert_eq!(search["command"], "search");
  assert_eq!(search["data"]["results"].as_array().unwrap().len(), 1);

  let unauthorized = service.request_with_token(
    serde_json::json!({
        "id": 3,
        "method": "summary",
        "params": { "session": session }
    }),
    "bad-token",
  );
  assert_eq!(unauthorized["status"], "error");
  assert_eq!(unauthorized["error"]["code"], "access_denied");

  let cancel = service.request(serde_json::json!({
      "id": 4,
      "method": "cancel",
      "params": { "session": session, "cancel_token": "missing-token" }
  }));
  assert_eq!(cancel["status"], "ok");
  assert_eq!(cancel["command"], "cancel");
  assert_eq!(cancel["data"]["session"], session);
  assert_eq!(cancel["data"]["cancel_token"], "missing-token");
  assert_eq!(cancel["data"]["canceled"], false);
}

#[test]
fn service_http_handles_parallel_session_requests() {
  let first = write_fixture("service-http-parallel-a.onnx");
  let second = write_fixture("service-http-parallel-b.onnx");
  let service = HttpServiceProcess::spawn();

  let first_open = service.request(serde_json::json!({
      "id": 1,
      "method": "open",
      "params": { "path": first }
  }));
  let second_open = service.request(serde_json::json!({
      "id": 2,
      "method": "open",
      "params": { "path": second }
  }));
  let first_session = first_open["data"]["session"].as_u64().unwrap();
  let second_session = second_open["data"]["session"].as_u64().unwrap();
  let address = service.address.clone();
  let token = service.token.clone();

  let workers = (0..8)
    .map(|index| {
      let address = address.clone();
      let token = token.clone();
      let session = if index % 2 == 0 {
        first_session
      } else {
        second_session
      };
      thread::spawn(move || {
        http_request(
          &address,
          &token,
          serde_json::json!({
              "id": index,
              "method": if index % 3 == 0 { "search" } else { "summary" },
              "params": if index % 3 == 0 {
                  serde_json::json!({ "session": session, "query": "add", "limit": 1 })
              } else {
                  serde_json::json!({ "session": session })
              }
          }),
        )
      })
    })
    .collect::<Vec<_>>();

  for worker in workers {
    let response = worker.join().unwrap();
    assert_eq!(response["status"], "ok");
  }
}

#[test]
fn service_http_cancels_active_layout_request() {
  let onnx = write_chain_fixture("service-http-cancel.onnx", 50_000);
  let service = HttpServiceProcess::spawn();

  let open = service.request(serde_json::json!({
      "id": 1,
      "method": "open",
      "params": { "path": onnx }
  }));
  assert_eq!(open["status"], "ok");
  let session = open["data"]["session"].as_u64().unwrap();
  let address = service.address.clone();
  let token = service.token.clone();
  let cancel_token = "layout-to-cancel";

  let worker = thread::spawn(move || {
    http_request(
      &address,
      &token,
      serde_json::json!({
          "id": "layout-to-cancel",
          "method": "layout",
          "params": {
              "session": session,
              "handle": { "kind": "graph", "graph": 0 },
              "max_nodes": 50_000,
              "cancel_token": cancel_token
          }
      }),
    )
  });

  let mut cancel_seen = false;
  for _ in 0..200 {
    let response = service.request(serde_json::json!({
        "id": "cancel-layout",
        "method": "cancel",
        "params": { "session": session, "cancel_token": cancel_token }
    }));
    if response["data"]["canceled"] == true {
      cancel_seen = true;
      break;
    }
    if worker.is_finished() {
      break;
    }
    thread::sleep(Duration::from_millis(5));
  }

  assert!(
    cancel_seen,
    "cancel request should find the active layout token"
  );
  let layout = worker.join().unwrap();
  assert_eq!(layout["status"], "error");
  assert_eq!(layout["command"], "layout");
  assert_eq!(layout["error"]["code"], "canceled");
}

#[test]
fn service_stdio_cancels_active_layout_request() {
  let onnx = write_chain_fixture("service-stdio-cancel.onnx", 50_000);
  let mut service = ServiceProcess::spawn();

  let open = service.request(serde_json::json!({
      "id": 1,
      "method": "open",
      "params": { "path": onnx }
  }));
  assert_eq!(open["status"], "ok");
  let session = open["data"]["session"].as_u64().unwrap();
  let cancel_token = "stdio-layout-to-cancel";

  service.send(serde_json::json!({
      "id": "stdio-layout-to-cancel",
      "method": "layout",
      "params": {
          "session": session,
          "handle": { "kind": "graph", "graph": 0 },
          "max_nodes": 50_000,
          "cancel_token": cancel_token
      }
  }));
  for attempt in 0..200 {
    service.send(serde_json::json!({
        "id": format!("cancel-stdio-layout-{attempt}"),
        "method": "cancel",
        "params": { "session": session, "cancel_token": cancel_token }
    }));
  }

  let mut cancel_seen = false;
  let mut layout = None;
  for _ in 0..201 {
    let response = service.read_response();
    match response["command"].as_str() {
      Some("cancel") => {
        cancel_seen |= response["data"]["canceled"] == true;
      }
      Some("layout") => {
        layout = Some(response);
        if cancel_seen {
          break;
        }
      }
      other => panic!("unexpected response command {other:?}: {response}"),
    }
    if cancel_seen
      && layout
        .as_ref()
        .is_some_and(|response| response["status"] == "error")
    {
      break;
    }
  }

  assert!(
    cancel_seen,
    "cancel request should find the active stdio layout token"
  );
  let layout = layout.expect("layout response");
  assert_eq!(layout["status"], "error");
  assert_eq!(layout["command"], "layout");
  assert_eq!(layout["error"]["code"], "canceled");
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
    self.send(request);
    self.read_response()
  }

  fn send(&mut self, request: serde_json::Value) {
    writeln!(self.input, "{request}").unwrap();
    self.input.flush().unwrap();
  }

  fn read_response(&mut self) -> serde_json::Value {
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

struct HttpServiceProcess {
  child: Child,
  address: String,
  token: String,
}

impl HttpServiceProcess {
  fn spawn() -> Self {
    let mut child = Command::new(env!("CARGO_BIN_EXE_netron-rs"))
      .arg("serve")
      .arg("--http")
      .arg("127.0.0.1:0")
      .stdout(Stdio::piped())
      .spawn()
      .expect("spawn HTTP service");
    let stdout = child.stdout.take().unwrap();
    let mut output = BufReader::new(stdout);
    let mut line = String::new();
    output.read_line(&mut line).unwrap();
    let ready: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(ready["status"], "ok");
    assert_eq!(ready["data"]["transport"], "http");
    Self {
      child,
      address: ready["data"]["address"].as_str().unwrap().to_owned(),
      token: ready["data"]["token"].as_str().unwrap().to_owned(),
    }
  }

  fn request(&self, request: serde_json::Value) -> serde_json::Value {
    self.request_with_token(request, &self.token)
  }

  fn request_with_token(&self, request: serde_json::Value, token: &str) -> serde_json::Value {
    http_request(&self.address, token, request)
  }
}

impl Drop for HttpServiceProcess {
  fn drop(&mut self) {
    let _ = self.child.kill();
    let _ = self.child.wait();
  }
}

fn http_request(address: &str, token: &str, request: serde_json::Value) -> serde_json::Value {
  let body = request.to_string();
  let mut stream = TcpStream::connect(address).unwrap();
  write!(
    stream,
    "POST /rpc HTTP/1.1\r\n\
     Host: {address}\r\n\
     Authorization: Bearer {token}\r\n\
     Content-Type: application/json\r\n\
     Content-Length: {}\r\n\
     Connection: close\r\n\
     \r\n\
     {body}",
    body.len()
  )
  .unwrap();
  stream.flush().unwrap();

  let mut response = String::new();
  stream.read_to_string(&mut response).unwrap();
  let (_, body) = response
    .split_once("\r\n\r\n")
    .expect("HTTP response should contain headers and body");
  serde_json::from_str(body).unwrap()
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
  let path = temp_root(&format!("netron-rs-{name}")).with_extension("mlir");
  fs::write(&path, text).unwrap();
  path
}

fn mlirbc_fixture(name: &str) -> std::path::PathBuf {
  let local = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../../tests/fixtures/mlir")
    .join(name);
  if local.exists() {
    return local;
  }

  std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../../../netron/third_party/test/mlir")
    .join(name)
}

fn onnx_fixture(name: &str) -> std::path::PathBuf {
  std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../../tests/fixtures/onnx")
    .join(name)
}

fn write_fixture(name: &str) -> std::path::PathBuf {
  let path = std::env::temp_dir().join(format!("netron-rs-{name}"));
  fs::write(&path, fixture_model()).unwrap();
  path
}

fn write_chain_fixture(name: &str, nodes: usize) -> std::path::PathBuf {
  let path = std::env::temp_dir().join(format!("netron-rs-{name}"));
  fs::write(&path, fixture_chain_model(nodes)).unwrap();
  path
}

fn write_repeated_pair_fixture(name: &str) -> std::path::PathBuf {
  let path = std::env::temp_dir().join(format!("netron-rs-{name}"));
  fs::write(&path, fixture_repeated_pair_model()).unwrap();
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

fn fixture_chain_model(nodes: usize) -> Vec<u8> {
  let mut model = Vec::new();
  varint(&mut model, 1, 9);
  string(&mut model, 2, "netron-rs-cli-chain-test");
  message(&mut model, 7, fixture_chain_graph(nodes));
  message(&mut model, 8, opset("", 18));
  model
}

fn fixture_repeated_pair_model() -> Vec<u8> {
  let mut model = Vec::new();
  varint(&mut model, 1, 9);
  string(&mut model, 2, "netron-rs-cli-repeated-pair-test");
  message(&mut model, 7, fixture_repeated_pair_graph());
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

fn fixture_repeated_pair_graph() -> Vec<u8> {
  let mut graph = Vec::new();
  string(&mut graph, 2, "main");
  for name in ["x0", "x1"] {
    message(&mut graph, 11, value_info(name, 1, &[dim_value(1)]));
  }
  for name in ["y0", "y1"] {
    message(&mut graph, 12, value_info(name, 1, &[dim_value(1)]));
  }
  message(&mut graph, 1, op_node("Relu", "x0", "h0"));
  message(&mut graph, 1, op_node("Add", "h0", "y0"));
  message(&mut graph, 1, op_node("Relu", "x1", "h1"));
  message(&mut graph, 1, op_node("Add", "h1", "y1"));
  graph
}

fn fixture_chain_graph(nodes: usize) -> Vec<u8> {
  let mut graph = Vec::new();
  string(&mut graph, 2, "main");
  message(&mut graph, 11, value_info("x0", 1, &[dim_value(1)]));
  message(
    &mut graph,
    12,
    value_info(&format!("x{nodes}"), 1, &[dim_value(1)]),
  );
  for index in 0..nodes {
    message(
      &mut graph,
      1,
      chain_node(&format!("x{index}"), &format!("x{}", index + 1)),
    );
  }
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

fn op_node(op: &str, input: &str, output: &str) -> Vec<u8> {
  let mut node = Vec::new();
  string(&mut node, 1, input);
  string(&mut node, 2, output);
  string(&mut node, 4, op);
  node
}

fn chain_node(input: &str, output: &str) -> Vec<u8> {
  let mut node = Vec::new();
  string(&mut node, 1, input);
  string(&mut node, 2, output);
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
