use netron_rs_core::{AttributeValue, ModelError, TensorElementType, TensorStorage};
use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;
#[cfg(unix)]
use std::{fs, os::unix::fs::symlink, time::SystemTime};

#[test]
fn parses_onnx_graph_into_canonical_normalized_json() {
  let data = fixture_model();
  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("fixture parses");

  assert_eq!(model.format.name, "ONNX");
  assert_eq!(model.metadata.producer.as_deref(), Some("netron-rs-test"));
  assert_eq!(model.metadata.opsets[0].version, 18);
  assert_eq!(model.graphs.len(), 1);
  assert_eq!(model.tensors.len(), 1);

  let graph = &model.graphs[0];
  assert_eq!(graph.inputs.len(), 1);
  assert_eq!(graph.outputs.len(), 1);
  assert_eq!(graph.nodes.len(), 1);
  assert_eq!(graph.nodes[0].inputs.len(), 2);
  assert_eq!(
    graph.metadata.get("source").map(String::as_str),
    Some("fixture")
  );

  assert!(matches!(
    model.tensors[0].storage,
    TensorStorage::InlineBytes { byte_len: 12 }
  ));

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let value = |name: &str| {
    values
      .iter()
      .find(|value| value["name"] == name)
      .expect("value exists")
  };
  assert_eq!(normalized["format"]["name"], "ONNX");
  assert_eq!(normalized["metadata"]["producer"], "netron-rs-test");
  assert_eq!(normalized["metadata"]["opsets"][0]["version"], 18);
  assert_eq!(normalized["graphs"][0]["name"], "main");
  assert_eq!(normalized["graphs"][0]["metadata"]["source"], "fixture");
  assert_eq!(normalized["graphs"][0]["inputs"], json!(["x"]));
  assert_eq!(normalized["graphs"][0]["outputs"], json!(["y"]));
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["operator"]["overload"],
    "fast"
  );
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["metadata"],
    json!({ "engine": "fixture" })
  );
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["inputs"],
    json!(["x", "w"])
  );
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["attributes"][0]["name"],
    "alpha"
  );
  assert_eq!(
    value("x")["type"]["shape"][1],
    json!({ "kind": "symbolic", "value": "batch" })
  );
  assert_eq!(value("w")["type"]["element_type"], "float32");
  assert_eq!(value("mid")["metadata"], json!({}));
  assert_eq!(value("y")["type"]["denotation"], "IMAGE");
  assert_eq!(
    value("y")["quantization"],
    json!([
        { "key": "scale", "value": "y_scale" },
        { "key": "zero_point", "value": "y_zero" }
    ])
  );
  assert_eq!(
    normalized["tensors"][0]["storage"],
    json!({ "kind": "inline_bytes", "byte_len": 12 })
  );
}

#[test]
fn parses_single_file_zip_wrapper_around_onnx_model() {
  let archive = zip_store("model.onnx", &fixture_model());
  let model = parse(ModelInput {
    data: &archive,
    path: Some(std::path::Path::new("model.onnx.zip")),
    allow_unsafe_paths: false,
  })
  .expect("zip-wrapped ONNX model parses");

  assert_eq!(model.format.name, "ONNX");
  assert_eq!(model.graphs.len(), 1);
  assert_eq!(model.graphs[0].nodes.len(), 1);
}

#[test]
fn rejects_unknown_extension_as_unsupported() {
  let error = parse(ModelInput {
    data: b"not a model",
    path: Some(std::path::Path::new("model.txt")),
    allow_unsafe_paths: false,
  })
  .expect_err("plain text should be unsupported");

  assert!(matches!(
    error,
    netron_rs_core::ModelError::UnsupportedFormat
  ));
}

#[test]
fn rejects_mislabelled_onnx_file_as_unsupported() {
  let error = parse(ModelInput {
    data: b"this looks like text, not ONNX",
    path: Some(std::path::Path::new("model.onnx")),
    allow_unsafe_paths: false,
  })
  .expect_err("bad .onnx payload should be unsupported");

  assert!(matches!(
    error,
    netron_rs_core::ModelError::UnsupportedFormat
  ));
}

#[test]
fn rejects_random_zip_payload_as_unsupported() {
  let archive = zip_store("readme.txt", b"hello archive");
  let error = parse(ModelInput {
    data: &archive,
    path: Some(std::path::Path::new("model.onnx.zip")),
    allow_unsafe_paths: false,
  })
  .expect_err("unsupported zip payload should be unsupported");
  assert!(matches!(
    error,
    netron_rs_core::ModelError::UnsupportedFormat
  ));
}

#[test]
fn rejects_onnx_absolute_external_data_path() {
  let location = std::env::temp_dir()
    .join("netron-rs-denied.bin")
    .to_string_lossy()
    .into_owned();
  let data = tensor_with_external_data("w", &[1], 1, &location);
  let model_path = std::path::Path::new("model.onnx");
  let error = parse(ModelInput {
    data: &data,
    path: Some(model_path),
    allow_unsafe_paths: false,
  })
  .expect_err("absolute external-data paths must be denied");
  assert!(matches!(error, ModelError::AccessDenied { .. }));
  if let ModelError::AccessDenied { path } = error {
    assert!(path.contains("model.onnx"));
    assert!(path.contains(&location));
  }
}

#[test]
fn rejects_onnx_external_data_path_with_traversal() {
  for location in ["../outside.bin", r"..\outside.bin"] {
    let data = tensor_with_external_data("w", &[1], 1, location);
    let data_path = std::path::Path::new("/tmp/model.onnx");
    let error = parse(ModelInput {
      data: &data,
      path: Some(data_path),
      allow_unsafe_paths: false,
    })
    .expect_err("path traversal in external-data location should be denied");
    assert!(matches!(error, ModelError::AccessDenied { .. }));
  }
}

#[test]
fn rejects_onnx_external_data_uri_locations() {
  for location in [
    "http://example.com/t.bin",
    "ftp://example.com/t.bin",
    "file://host/share/t.bin",
  ] {
    let data = tensor_with_external_data("w", &[1], 1, location);
    let data_path = std::path::Path::new("/tmp/model.onnx");
    let error = parse(ModelInput {
      data: &data,
      path: Some(data_path),
      allow_unsafe_paths: false,
    })
    .expect_err("uri external-data locations should be denied");
    assert!(matches!(error, ModelError::AccessDenied { .. }));
  }
}

#[cfg(unix)]
#[test]
fn rejects_onnx_external_data_symlink_escape_when_canonicalized() {
  let unique = SystemTime::now()
    .duration_since(SystemTime::UNIX_EPOCH)
    .unwrap()
    .as_nanos();
  let temp_root = std::env::temp_dir().join(format!("netron-onnx-{}", unique));
  let model_dir = temp_root.join("model");
  let link_dir = model_dir.join("links");
  let target_dir = temp_root.join("external");
  fs::create_dir_all(&link_dir).unwrap();
  fs::create_dir_all(&target_dir).unwrap();
  let model_path = model_dir.join("model.onnx");
  let target_file = target_dir.join("outside.bin");
  fs::write(&target_file, b"").unwrap();
  let link_file = link_dir.join("outside.bin");
  symlink(&target_file, &link_file).unwrap();
  let data = tensor_with_external_data("w", &[1], 1, "links/outside.bin");

  let error = parse(ModelInput {
    data: &data,
    path: Some(model_path.as_path()),
    allow_unsafe_paths: false,
  })
  .expect_err("symlink escape should be denied");

  assert!(matches!(error, ModelError::AccessDenied { .. }));
  fs::remove_dir_all(&temp_root).unwrap();
}

#[test]
fn parses_onnx_external_data_with_allow_unsafe_flag() {
  let location = std::env::temp_dir()
    .join("netron-rs-trusted.bin")
    .to_string_lossy()
    .into_owned();
  let data = tensor_with_external_data("w", &[1], 1, &location);
  let model_path = std::path::Path::new("/tmp/model.onnx");
  let model = parse(ModelInput {
    data: &data,
    path: Some(model_path),
    allow_unsafe_paths: true,
  })
  .expect("unsafe paths should be allowed when explicitly enabled");
  assert!(matches!(
    model.tensors[0].storage,
    TensorStorage::External { .. }
  ));
}

#[test]
fn rejects_generic_zip_wrapper_even_when_it_contains_onnx() {
  let archive = zip_store("model.onnx", &fixture_model());
  let error = parse(ModelInput {
    data: &archive,
    path: Some(std::path::Path::new("bundle.zip")),
    allow_unsafe_paths: false,
  })
  .expect_err("generic zip wrapper should not probe ONNX contents");
  assert!(matches!(
    error,
    netron_rs_core::ModelError::UnsupportedFormat
  ));
}

#[test]
fn rejects_nested_non_matching_archive_entries_as_unsupported() {
  let archive = zip_store_entries(&[
    ("assets/readme.txt", b"unused payload"),
    ("notes/keep.txt", b"not a model"),
  ]);
  let error = parse(ModelInput {
    data: &archive,
    path: Some(std::path::Path::new("bundle.onnx.zip")),
    allow_unsafe_paths: false,
  })
  .expect_err("only nested non-matching entries should be unsupported");
  assert!(matches!(
    error,
    netron_rs_core::ModelError::UnsupportedFormat
  ));
}

#[test]
fn parses_zip_wrapper_around_onnx_json_model() {
  let data = br#"{
        "irVersion": "9",
        "producerName": "json-test",
        "graph": {
            "name": "main",
            "node": [{
                "input": ["x", "w"],
                "output": ["y"],
                "opType": "Pad",
                "attribute": [
                    { "name": "mode", "s": "cmVmbGVjdA==", "type": "STRING" },
                    { "name": "pads", "ints": ["0", "0"], "type": "INTS" }
                ]
            }],
            "initializer": [{
                "dims": ["2"],
                "dataType": 1,
                "floatData": [1.0, 2.0],
                "name": "w"
            }],
            "input": [{
                "name": "x",
                "type": {
                    "tensorType": {
                        "elemType": 1,
                        "shape": { "dim": [{ "dimParam": "batch" }, { "dimValue": "2" }] }
                    }
                }
            }],
            "output": [{
                "name": "y",
                "type": {
                    "tensorType": {
                        "elemType": 1,
                        "shape": { "dim": [{ "dimParam": "batch" }, { "dimValue": "2" }] }
                    }
                }
            }]
        },
        "opsetImport": [{ "domain": "", "version": "18" }]
    }"#;
  let archive = zip_store("model.json", data);
  let model = parse(ModelInput {
    data: &archive,
    path: Some(std::path::Path::new("model.onnx.zip")),
    allow_unsafe_paths: false,
  })
  .expect("zip-wrapped ONNX JSON model parses");

  assert_eq!(model.format.name, "ONNX");
  assert_eq!(model.metadata.producer.as_deref(), Some("json-test"));
  assert_eq!(model.metadata.opsets[0].version, 18);
  assert!(matches!(
    model.tensors[0].storage,
    TensorStorage::ElementList { len: 2 }
  ));

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"][0]["inputs"], json!(["x"]));
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["inputs"],
    json!(["x", "w"])
  );
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["attributes"][0]["value"],
    json!({ "kind": "string", "value": "reflect" })
  );
  assert_eq!(
    normalized["graphs"][0]["values"][0]["type"]["shape"][0],
    json!({ "kind": "symbolic", "value": "batch" })
  );
}

#[test]
fn zip_wrapper_prefers_onnx_model_over_dot_sidecar() {
  let model_data = fixture_model();
  let archive = zip_store_entries(&[
    ("model.onnx", model_data.as_slice()),
    ("model.onnx.dot", b"digraph G { a -> b }"),
  ]);
  let model = parse(ModelInput {
    data: &archive,
    path: Some(std::path::Path::new("model.onnx.zip")),
    allow_unsafe_paths: false,
  })
  .expect("zip-wrapped ONNX model with DOT sidecar parses");

  assert_eq!(model.format.name, "ONNX");
  assert_eq!(model.graphs[0].nodes.len(), 1);
}

#[test]
fn normalizes_torch_package_operator_overload_like_netron() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "main");
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));
  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 2, "y");
  string(&mut node, 4, "aten.convolution.default");
  string(&mut node, 7, "pkg.torch.ops");
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("torch-op.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("torch package op parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let operator = &normalized["graphs"][0]["nodes"][0]["operator"];
  assert_eq!(operator["domain"], "pkg.torch.ops");
  assert_eq!(operator["name"], "aten.convolution");
  assert_eq!(operator["overload"], "default");
}

#[test]
fn cast_output_type_overrides_stale_value_info() {
  let mut cast_to = Vec::new();
  string(&mut cast_to, 1, "to");
  varint(&mut cast_to, 3, 1);

  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Cast");
  message(&mut node, 5, cast_to);

  let mut graph = Vec::new();
  string(&mut graph, 2, "main");
  message(&mut graph, 11, value_info("x", 10, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 10, &[dim_value(1)]));
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("cast.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("cast fixture parses");
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let y = values.iter().find(|value| value["name"] == "y").unwrap();
  assert_eq!(y["type"]["element_type"], "float32");
}

#[test]
fn concat_output_shape_uses_concrete_input_shapes() {
  let mut axis = Vec::new();
  string(&mut axis, 1, "axis");
  varint(&mut axis, 3, 1);

  let mut node = Vec::new();
  string(&mut node, 1, "a");
  string(&mut node, 1, "b");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Concat");
  message(&mut node, 5, axis);

  let mut graph = Vec::new();
  string(&mut graph, 2, "concat");
  message(
    &mut graph,
    13,
    value_info("a", 1, &[dim_value(1), dim_value(2), dim_value(8)]),
  );
  message(
    &mut graph,
    13,
    value_info("b", 1, &[dim_value(1), dim_value(3), dim_value(8)]),
  );
  message(
    &mut graph,
    12,
    value_info("y", 1, &[dim_param("N"), dim_value(5), dim_param("H")]),
  );
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("concat.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("concat fixture parses");
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let y = values.iter().find(|value| value["name"] == "y").unwrap();
  assert_eq!(
    y["type"]["shape"],
    json!([
        { "kind": "known", "value": 1 },
        { "kind": "known", "value": 5 },
        { "kind": "known", "value": 8 }
    ])
  );
}

#[test]
fn concat_preserves_unknown_graph_output_dimension_markers() {
  let mut axis = Vec::new();
  string(&mut axis, 1, "axis");
  varint(&mut axis, 3, 2);

  let mut node = Vec::new();
  string(&mut node, 1, "a");
  string(&mut node, 1, "b");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Concat");
  message(&mut node, 5, axis);

  let mut graph = Vec::new();
  string(&mut graph, 2, "concat-none");
  message(
    &mut graph,
    13,
    value_info("a", 1, &[dim_param("None"), dim_value(1), dim_value(3)]),
  );
  message(
    &mut graph,
    13,
    value_info("b", 1, &[dim_param("None"), dim_value(1), dim_value(3)]),
  );
  message(
    &mut graph,
    12,
    value_info("y", 1, &[dim_param("None"), dim_value(1), dim_value(6)]),
  );
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("concat-none.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("concat none fixture parses");
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let y = values.iter().find(|value| value["name"] == "y").unwrap();
  assert_eq!(y["type"]["shape"][0], json!({ "kind": "unknown" }));
  assert_eq!(
    y["type"]["shape"][2],
    json!({ "kind": "known", "value": 6 })
  );
}

#[test]
fn preserves_empty_input_names_metadata_and_exact_large_int_attributes() {
  let mut end = Vec::new();
  string(&mut end, 1, "end");
  varint(&mut end, 3, i64::MAX as u64);

  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Slice");
  message(&mut node, 5, end);
  message(&mut node, 9, string_entry("input_names", "[]"));

  let mut graph = Vec::new();
  string(&mut graph, 2, "metadata");
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("metadata.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("metadata fixture parses");
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let node = &normalized["graphs"][0]["nodes"][0];
  assert_eq!(node["metadata"]["input_names"], "[]");
  assert_eq!(
    node["attributes"][0]["value"],
    json!({ "kind": "int", "value": i64::MAX.to_string() })
  );
}

#[test]
fn value_info_does_not_override_graph_output_type() {
  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Identity");

  let mut graph = Vec::new();
  string(&mut graph, 2, "output-type");
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(
    &mut graph,
    12,
    value_info("y", 1, &[dim_unknown(), dim_value(4)]),
  );
  message(
    &mut graph,
    13,
    value_info("y", 1, &[dim_value(0), dim_value(4)]),
  );
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("output-type.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("output type fixture parses");
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let y = values.iter().find(|value| value["name"] == "y").unwrap();
  assert_eq!(
    y["type"]["shape"],
    json!([
        { "kind": "unknown" },
        { "kind": "known", "value": 4 }
    ])
  );
}

#[test]
fn graph_output_leading_zero_dimension_is_unknown_like_netron() {
  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Identity");

  let mut graph = Vec::new();
  string(&mut graph, 2, "zero-output");
  message(
    &mut graph,
    11,
    value_info("x", 1, &[dim_value(0), dim_value(4)]),
  );
  message(
    &mut graph,
    12,
    value_info("y", 1, &[dim_value(0), dim_value(4)]),
  );
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("zero-output.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("zero output fixture parses");
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let x = values.iter().find(|value| value["name"] == "x").unwrap();
  let y = values.iter().find(|value| value["name"] == "y").unwrap();
  assert_eq!(
    x["type"]["shape"][0],
    json!({ "kind": "known", "value": 0 })
  );
  assert_eq!(y["type"]["shape"][0], json!({ "kind": "unknown" }));
}

#[test]
fn parses_null_graph_without_error() {
  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  string(&mut data, 2, "netron-rs-test");
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("null graph model parses");

  assert_eq!(model.graphs.len(), 0);
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"], json!([]));
}

#[test]
fn parses_standalone_graph_proto() {
  let data = fixture_graph();
  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("graph.onnx")),
    allow_unsafe_paths: false,
  })
  .expect("standalone graph parses");

  assert_eq!(model.format.name, "ONNX");
  assert_eq!(model.metadata.opsets.len(), 0);
  assert_eq!(model.graphs.len(), 1);
  assert_eq!(model.graphs[0].nodes.len(), 1);
  assert_eq!(model.graphs[0].inputs.len(), 1);
  assert_eq!(model.graphs[0].outputs.len(), 1);
}

#[test]
fn parses_sparse_initializer_as_sparse_tensor_metadata() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "sparse");
  message(&mut graph, 15, sparse_tensor("sparse_w", &[4, 4]));
  message(
    &mut graph,
    12,
    value_info_sparse("sparse_w", 1, &[dim_value(4), dim_value(4)]),
  );

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("sparse fixture parses");

  assert_eq!(model.tensors.len(), 3);
  assert!(matches!(
    model.tensors[2].storage,
    TensorStorage::Sparse { .. }
  ));

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["graphs"][0]["values"][0]["type"]["layout"],
    "sparse"
  );
  assert_eq!(
    normalized["tensors"][2]["storage"],
    json!({ "kind": "sparse", "values": 0, "indices": 1 })
  );
}

#[test]
fn hides_initializer_outputs_like_netron() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "initializer-output");
  message(&mut graph, 5, tensor("w", &[1], 1, &[0; 4]));
  message(&mut graph, 12, value_info("w", 1, &[dim_value(1)]));

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("initializer output fixture parses");

  assert_eq!(model.graphs[0].outputs.len(), 0);
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"][0]["outputs"], json!([]));
}

#[test]
fn preserves_optional_value_type_shell() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "optional");
  message(&mut graph, 11, value_info_optional("optional_input"));

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("optional type fixture parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"][0]["inputs"], json!(["optional_input"]));
  assert_eq!(
    normalized["graphs"][0]["values"][0]["type"],
    json!({ "shape": [] })
  );
}

#[test]
fn normalizes_complex_type_names_like_netron() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "complex");
  message(
    &mut graph,
    11,
    value_info("complex64_input", 14, &[dim_value(2)]),
  );
  message(&mut graph, 11, value_info("bool_input", 9, &[dim_value(1)]));
  message(
    &mut graph,
    12,
    value_info("complex128_output", 15, &[dim_value(2)]),
  );

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("complex fixture parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["graphs"][0]["values"].as_array().unwrap();
  let value = |name: &str| {
    values
      .iter()
      .find(|value| value["name"] == name)
      .expect("value exists")
  };
  assert_eq!(
    value("complex64_input")["type"]["element_type"],
    "complex<float32>"
  );
  assert_eq!(
    value("complex128_output")["type"]["element_type"],
    "complex<float64>"
  );
  assert_eq!(value("bool_input")["type"]["element_type"], "boolean");
}

#[test]
fn preserves_unpacked_repeated_attribute_scalars() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "attributes");
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));

  let mut kernel = Vec::new();
  string(&mut kernel, 1, "kernel_shape");
  varint(&mut kernel, 8, 3);
  varint(&mut kernel, 8, 3);

  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Conv");
  message(&mut node, 5, kernel);
  message(&mut graph, 1, node);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("attribute fixture parses");

  let AttributeValue::Ints(values) = &model.graphs[0].nodes[0].attributes[0].value else {
    panic!("expected repeated int attribute");
  };
  assert_eq!(values, &[3, 3]);

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["attributes"][0]["value"],
    json!({ "kind": "ints", "value": [3, 3] })
  );
}

#[test]
fn promotes_single_use_constant_tensor_to_initializer() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "constant");
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));
  message(
    &mut graph,
    1,
    constant_node("c", tensor("c", &[1], 7, &[0; 8])),
  );
  message(&mut graph, 1, binary_node("Add", "x", "c", "y"));

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("constant fixture parses");

  let graph = &model.graphs[0];
  assert_eq!(graph.nodes.len(), 1);
  assert_eq!(model.strings.get(graph.nodes[0].operator.name), "Add");
  let constant = graph
    .values
    .iter()
    .find(|value| model.strings.get(value.name) == "c")
    .expect("constant value exists");
  assert!(constant.initializer.is_some());
  assert!(constant.producer.is_none());

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["operator"]["name"],
    "Add"
  );
  let normalized_values = normalized["graphs"][0]["values"].as_array().unwrap();
  let constant = normalized_values
    .iter()
    .find(|value| value["name"] == "c")
    .expect("constant normalized value exists");
  assert_eq!(constant["initializer"], 0);
  assert_eq!(constant["type"]["element_type"], "int64");
}

#[test]
fn parses_local_function_signature_and_overload() {
  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 8, opset("", 18));
  message(&mut data, 25, function_proto());

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("function-only fixture parses");

  assert_eq!(model.graphs.len(), 0);
  assert_eq!(model.functions.len(), 1);

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["functions"][0]["name"], "Scale");
  assert_eq!(normalized["functions"][0]["domain"], "custom");
  assert_eq!(normalized["functions"][0]["overload"], "float");
  assert_eq!(normalized["functions"][0]["inputs"], json!(["x", "s"]));
  assert_eq!(normalized["functions"][0]["outputs"], json!(["y"]));
  assert_eq!(normalized["functions"][0]["metadata"]["kind"], "local");
  assert_eq!(
    normalized["functions"][0]["nodes"][0]["operator"]["name"],
    "Mul"
  );
  assert_eq!(
    normalized["functions"][0]["nodes"][0]["operator"]["overload"],
    "broadcast"
  );
}

#[test]
fn promotes_local_function_constant_tensor_to_initializer() {
  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 8, opset("", 18));
  message(&mut data, 25, function_proto_with_constant());

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("function constant fixture parses");

  let function = &model.functions[0];
  assert_eq!(function.nodes.len(), 1);
  assert_eq!(model.strings.get(function.nodes[0].operator.name), "Add");
  assert_eq!(model.tensors.len(), 1);
  let constant = function
    .values
    .iter()
    .find(|value| model.strings.get(value.name) == "c")
    .expect("function constant value exists");
  assert!(constant.initializer.is_some());

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["functions"][0]["values"].as_array().unwrap();
  let constant = values
    .iter()
    .find(|value| value["name"] == "c")
    .expect("constant normalized function value exists");
  assert_eq!(constant["initializer"], 0);
  assert_eq!(constant["type"]["element_type"], "int64");
  assert_eq!(
    normalized["functions"][0]["nodes"][0]["operator"]["name"],
    "Add"
  );
}

#[test]
fn preserves_local_function_node_tensor_attributes() {
  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 8, opset("", 18));
  message(&mut data, 25, function_proto_with_reused_constant());

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("function reused constant fixture parses");

  let function = &model.functions[0];
  assert_eq!(function.nodes.len(), 3);
  assert_eq!(
    model.strings.get(function.nodes[0].operator.name),
    "Constant"
  );
  assert!(matches!(
    function.nodes[0].attributes[0].value,
    AttributeValue::Tensor(_)
  ));
  assert_eq!(model.tensors.len(), 1);

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["functions"][0]["nodes"][0]["attributes"][0]["value"],
    json!({ "kind": "tensor", "value": 0 })
  );
}

#[test]
fn moves_single_use_graph_initializer_into_local_function() {
  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, function_initializer_graph());
  message(&mut data, 8, opset("", 18));
  message(&mut data, 25, function_proto());

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("function initializer fixture parses");

  let graph = &model.graphs[0];
  assert_eq!(graph.nodes.len(), 1);
  assert_eq!(graph.nodes[0].inputs.len(), 1);
  assert_eq!(
    model
      .strings
      .get(graph.values[graph.nodes[0].inputs[0].unwrap().index()].name),
    "x"
  );

  let function = &model.functions[0];
  assert_eq!(function.inputs.len(), 1);
  assert_eq!(model.strings.get(function.inputs[0]), "x");
  let moved = function
    .values
    .iter()
    .find(|value| model.strings.get(value.name) == "s")
    .expect("moved initializer value exists");
  assert!(moved.initializer.is_some());

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"][0]["nodes"][0]["inputs"], json!(["x"]));
  assert_eq!(normalized["functions"][0]["inputs"], json!(["x"]));
  let values = normalized["functions"][0]["values"].as_array().unwrap();
  let x_value = values
    .iter()
    .find(|value| value["name"] == "x")
    .expect("function input value exists");
  assert_eq!(x_value["type"]["element_type"], "float32");
  assert_eq!(
    x_value["type"]["shape"],
    json!([{ "kind": "known", "value": 1 }])
  );
  let moved = values
    .iter()
    .find(|value| value["name"] == "s")
    .expect("moved normalized value exists");
  assert_eq!(moved["initializer"], 0);
  assert_eq!(moved["type"]["element_type"], "float32");
}

#[test]
fn propagates_single_use_initializer_through_nested_local_functions() {
  let mut graph = Vec::new();
  string(&mut graph, 2, "caller");
  message(&mut graph, 5, tensor("w", &[1], 1, &[0; 4]));
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));
  let mut call = Vec::new();
  string(&mut call, 1, "x");
  string(&mut call, 1, "w");
  string(&mut call, 2, "y");
  string(&mut call, 4, "Outer");
  string(&mut call, 7, "custom");
  message(&mut graph, 1, call);

  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 7, graph);
  message(&mut data, 8, opset("", 18));
  message(&mut data, 25, nested_function_proto("Outer", "Inner"));
  message(&mut data, 25, nested_function_proto("Inner", "Add"));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("nested function initializer fixture parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"][0]["nodes"][0]["inputs"], json!(["x"]));
  let function = |name: &str| {
    normalized["functions"]
      .as_array()
      .unwrap()
      .iter()
      .find(|function| function["name"] == name)
      .unwrap()
  };
  assert_eq!(function("Outer")["inputs"], json!(["x"]));
  assert_eq!(function("Outer")["nodes"][0]["inputs"], json!(["x"]));
  assert_eq!(function("Inner")["inputs"], json!(["x"]));
  assert_eq!(function("Inner")["nodes"][0]["inputs"], json!(["x", "w"]));
  let inner_values = function("Inner")["values"].as_array().unwrap();
  let moved = inner_values
    .iter()
    .find(|value| value["name"] == "w")
    .expect("nested moved initializer exists");
  assert_eq!(moved["initializer"], 0);
}

#[test]
fn duplicate_local_function_signature_keeps_last_definition() {
  let mut data = Vec::new();
  varint(&mut data, 1, 9);
  message(&mut data, 8, opset("", 18));
  message(&mut data, 25, function_proto_with_op("Add"));
  message(&mut data, 25, function_proto_with_op("Sub"));

  let model = parse(ModelInput {
    data: &data,
    path: None,
    allow_unsafe_paths: false,
  })
  .expect("duplicate function fixture parses");

  assert_eq!(model.functions.len(), 1);
  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["functions"][0]["nodes"][0]["operator"]["name"],
    "Sub"
  );
}

#[test]
fn parses_standalone_tensor_proto_as_constant_graph() {
  let mut packed_values = Vec::new();
  encode_varint(&mut packed_values, 0x10);
  encode_varint(&mut packed_values, 0x72);

  let mut data = Vec::new();
  varint(&mut data, 1, 7);
  varint(&mut data, 2, 23);
  bytes(&mut data, 5, &packed_values);
  string(&mut data, 8, "zero_point");

  let model = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("zero_point.pb")),
    allow_unsafe_paths: false,
  })
  .expect("standalone tensor fixture parses");

  assert_eq!(model.format.name, "ONNX Tensor");
  assert_eq!(model.graphs.len(), 1);
  assert_eq!(model.graphs[0].nodes.len(), 1);
  assert_eq!(model.tensors.len(), 1);
  assert!(matches!(
    model.tensors[0].element_type,
    TensorElementType::Float4e2m1
  ));
  assert!(matches!(
    model.graphs[0].nodes[0].attributes[0].value,
    AttributeValue::Tensor(_)
  ));

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["operator"]["name"],
    "Constant"
  );
  assert_eq!(normalized["tensors"][0]["element_type"], "float4e2m1");
}

fn fixture_model() -> Vec<u8> {
  let mut model = Vec::new();
  varint(&mut model, 1, 9);
  string(&mut model, 2, "netron-rs-test");
  message(&mut model, 7, fixture_graph());
  message(&mut model, 8, opset("", 18));
  model
}

fn fixture_graph() -> Vec<u8> {
  let mut graph = Vec::new();
  string(&mut graph, 2, "main");
  string(&mut graph, 10, "primary graph");
  message(&mut graph, 5, tensor("w", &[3], 1, &[0; 12]));
  message(
    &mut graph,
    14,
    annotation("y", &[("scale", "y_scale"), ("zero_point", "y_zero")]),
  );
  message(&mut graph, 16, string_entry("source", "fixture"));
  message(
    &mut graph,
    11,
    value_info("x", 1, &[dim_value(1), dim_param("batch")]),
  );
  message(&mut graph, 11, value_info("w", 1, &[dim_value(3)]));
  let mut internal_info = value_info("mid", 1, &[dim_value(3)]);
  message(
    &mut internal_info,
    4,
    string_entry(
      "pkg.torch.export.graph_signature.OutputSpec.kind",
      "BUFFER_MUTATION",
    ),
  );
  message(&mut graph, 13, internal_info);
  message(
    &mut graph,
    12,
    value_info_denotation("y", 1, &[dim_value(1), dim_param("batch")], "IMAGE"),
  );
  message(&mut graph, 1, node());
  graph
}

fn function_initializer_graph() -> Vec<u8> {
  let mut graph = Vec::new();
  string(&mut graph, 2, "caller");
  message(&mut graph, 5, tensor("s", &[1], 1, &[0; 4]));
  message(&mut graph, 11, value_info("x", 1, &[dim_value(1)]));
  message(&mut graph, 12, value_info("y", 1, &[dim_value(1)]));
  message(&mut graph, 1, function_call_node());
  graph
}

fn node() -> Vec<u8> {
  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 1, "");
  string(&mut node, 1, "w");
  string(&mut node, 2, "y");
  string(&mut node, 3, "relu1");
  string(&mut node, 4, "Relu");
  string(&mut node, 8, "fast");
  message(&mut node, 9, string_entry("engine", "fixture"));
  message(
    &mut node,
    9,
    string_entry("input_names", "['data', 'weight']"),
  );

  let mut alpha = Vec::new();
  string(&mut alpha, 1, "alpha");
  fixed32(&mut alpha, 2, 0.25_f32.to_bits());
  message(&mut node, 5, alpha);

  let mut labels = Vec::new();
  string(&mut labels, 1, "labels");
  bytes(&mut labels, 9, b"hot");
  bytes(&mut labels, 9, b"cold");
  message(&mut node, 5, labels);

  node
}

fn function_call_node() -> Vec<u8> {
  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 1, "s");
  string(&mut node, 2, "y");
  string(&mut node, 4, "Scale");
  string(&mut node, 7, "custom");
  string(&mut node, 8, "float");
  node
}

fn constant_node(output: &str, tensor: Vec<u8>) -> Vec<u8> {
  let mut attribute = Vec::new();
  string(&mut attribute, 1, "value");
  message(&mut attribute, 5, tensor);

  let mut node = Vec::new();
  string(&mut node, 2, output);
  string(&mut node, 4, "Constant");
  message(&mut node, 5, attribute);
  node
}

fn binary_node(op_type: &str, lhs: &str, rhs: &str, output: &str) -> Vec<u8> {
  let mut node = Vec::new();
  string(&mut node, 1, lhs);
  string(&mut node, 1, rhs);
  string(&mut node, 2, output);
  string(&mut node, 4, op_type);
  node
}

fn function_proto() -> Vec<u8> {
  function_proto_with_op("Mul")
}

fn function_proto_with_constant() -> Vec<u8> {
  let mut function = Vec::new();
  string(&mut function, 1, "WithConstant");
  string(&mut function, 4, "x");
  string(&mut function, 5, "y");
  message(&mut function, 9, opset("", 18));
  message(
    &mut function,
    7,
    constant_node("c", tensor("c", &[1], 7, &[0; 8])),
  );
  message(&mut function, 7, binary_node("Add", "x", "c", "y"));
  function
}

fn function_proto_with_reused_constant() -> Vec<u8> {
  let mut function = Vec::new();
  string(&mut function, 1, "WithReusedConstant");
  string(&mut function, 4, "x");
  string(&mut function, 5, "y");
  message(&mut function, 9, opset("", 18));
  message(
    &mut function,
    7,
    constant_node("c", tensor("c", &[1], 7, &[0; 8])),
  );
  message(&mut function, 7, binary_node("Add", "x", "c", "a"));
  message(&mut function, 7, binary_node("Mul", "a", "c", "y"));
  function
}

fn function_proto_with_op(op_type: &str) -> Vec<u8> {
  let mut function = Vec::new();
  string(&mut function, 1, "Scale");
  string(&mut function, 4, "x");
  string(&mut function, 4, "s");
  string(&mut function, 5, "y");
  string(&mut function, 8, "local function");
  message(&mut function, 9, opset("", 18));
  string(&mut function, 10, "custom");
  string(&mut function, 13, "float");
  message(&mut function, 14, string_entry("kind", "local"));

  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 1, "s");
  string(&mut node, 2, "y");
  string(&mut node, 4, op_type);
  string(&mut node, 8, "broadcast");
  message(&mut function, 7, node);

  function
}

fn nested_function_proto(name: &str, callee: &str) -> Vec<u8> {
  let mut function = Vec::new();
  string(&mut function, 1, name);
  string(&mut function, 4, "x");
  string(&mut function, 4, "w");
  string(&mut function, 5, "y");
  message(&mut function, 9, opset("", 18));
  string(&mut function, 10, "custom");

  let mut node = Vec::new();
  string(&mut node, 1, "x");
  string(&mut node, 1, "w");
  string(&mut node, 2, "y");
  string(&mut node, 4, callee);
  if callee != "Add" {
    string(&mut node, 7, "custom");
  }
  message(&mut function, 7, node);

  function
}

fn value_info(name: &str, elem_type: u64, dims: &[Vec<u8>]) -> Vec<u8> {
  value_info_impl(name, elem_type, dims, false, None)
}

fn value_info_denotation(
  name: &str,
  elem_type: u64,
  dims: &[Vec<u8>],
  denotation: &str,
) -> Vec<u8> {
  value_info_impl(name, elem_type, dims, false, Some(denotation))
}

fn value_info_sparse(name: &str, elem_type: u64, dims: &[Vec<u8>]) -> Vec<u8> {
  value_info_impl(name, elem_type, dims, true, None)
}

fn value_info_optional(name: &str) -> Vec<u8> {
  let mut tensor_type = Vec::new();
  varint(&mut tensor_type, 1, 1);

  let mut elem_type = Vec::new();
  message(&mut elem_type, 1, tensor_type);

  let mut optional_type = Vec::new();
  message(&mut optional_type, 1, elem_type);

  let mut type_proto = Vec::new();
  message(&mut type_proto, 9, optional_type);

  let mut value = Vec::new();
  string(&mut value, 1, name);
  message(&mut value, 2, type_proto);
  value
}

fn value_info_impl(
  name: &str,
  elem_type: u64,
  dims: &[Vec<u8>],
  sparse: bool,
  denotation: Option<&str>,
) -> Vec<u8> {
  let mut shape = Vec::new();
  for dim in dims {
    message(&mut shape, 1, dim.clone());
  }

  let mut tensor_type = Vec::new();
  varint(&mut tensor_type, 1, elem_type);
  message(&mut tensor_type, 2, shape);

  let mut type_proto = Vec::new();
  message(&mut type_proto, if sparse { 8 } else { 1 }, tensor_type);
  if let Some(denotation) = denotation {
    string(&mut type_proto, 6, denotation);
  }

  let mut value = Vec::new();
  string(&mut value, 1, name);
  message(&mut value, 2, type_proto);
  value
}

fn sparse_tensor(name: &str, dims: &[u64]) -> Vec<u8> {
  let mut sparse = Vec::new();
  message(&mut sparse, 1, tensor(name, &[2], 1, &[0; 8]));
  message(&mut sparse, 2, tensor("", &[2, 2], 7, &[0; 32]));
  for dim in dims {
    varint(&mut sparse, 3, *dim);
  }
  sparse
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

fn tensor_with_external_data(name: &str, dims: &[u64], data_type: u64, location: &str) -> Vec<u8> {
  let mut tensor = tensor(name, dims, data_type, &[]);
  let mut entry = Vec::new();
  string(&mut entry, 1, "location");
  string(&mut entry, 2, location);
  bytes(&mut tensor, 13, &entry);
  tensor
}

fn opset(domain: &str, version: u64) -> Vec<u8> {
  let mut opset = Vec::new();
  string(&mut opset, 1, domain);
  varint(&mut opset, 2, version);
  opset
}

fn annotation(tensor_name: &str, entries: &[(&str, &str)]) -> Vec<u8> {
  let mut annotation = Vec::new();
  string(&mut annotation, 1, tensor_name);
  for (key, value) in entries {
    message(&mut annotation, 2, string_entry(key, value));
  }
  annotation
}

fn string_entry(key: &str, value: &str) -> Vec<u8> {
  let mut entry = Vec::new();
  string(&mut entry, 1, key);
  string(&mut entry, 2, value);
  entry
}

fn zip_store(name: &str, data: &[u8]) -> Vec<u8> {
  zip_store_entries(&[(name, data)])
}

fn zip_store_entries(entries: &[(&str, &[u8])]) -> Vec<u8> {
  let mut output = Vec::new();
  let mut central_records = Vec::new();
  for (name, data) in entries {
    let name = name.as_bytes();
    let local_offset = output.len() as u32;
    le_u32(&mut output, 0x0403_4b50);
    le_u16(&mut output, 20);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u32(&mut output, 0);
    le_u32(&mut output, data.len() as u32);
    le_u32(&mut output, data.len() as u32);
    le_u16(&mut output, name.len() as u16);
    le_u16(&mut output, 0);
    output.extend_from_slice(name);
    output.extend_from_slice(data);
    central_records.push((name.to_vec(), data.len() as u32, local_offset));
  }

  let central_offset = output.len() as u32;
  for (name, size, local_offset) in &central_records {
    le_u32(&mut output, 0x0201_4b50);
    le_u16(&mut output, 20);
    le_u16(&mut output, 20);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u32(&mut output, 0);
    le_u32(&mut output, *size);
    le_u32(&mut output, *size);
    le_u16(&mut output, name.len() as u16);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u32(&mut output, 0);
    le_u32(&mut output, *local_offset);
    output.extend_from_slice(name);
  }

  let central_size = output.len() as u32 - central_offset;
  le_u32(&mut output, 0x0605_4b50);
  le_u16(&mut output, 0);
  le_u16(&mut output, 0);
  le_u16(&mut output, central_records.len() as u16);
  le_u16(&mut output, central_records.len() as u16);
  le_u32(&mut output, central_size);
  le_u32(&mut output, central_offset);
  le_u16(&mut output, 0);
  output
}

fn dim_value(value: u64) -> Vec<u8> {
  let mut dim = Vec::new();
  varint(&mut dim, 1, value);
  dim
}

fn dim_param(value: &str) -> Vec<u8> {
  let mut dim = Vec::new();
  string(&mut dim, 2, value);
  dim
}

fn dim_unknown() -> Vec<u8> {
  Vec::new()
}

fn varint(output: &mut Vec<u8>, field: u64, value: u64) {
  encode_varint(output, field << 3);
  encode_varint(output, value);
}

fn fixed32(output: &mut Vec<u8>, field: u64, value: u32) {
  encode_varint(output, (field << 3) | 5);
  output.extend_from_slice(&value.to_le_bytes());
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

fn le_u16(output: &mut Vec<u8>, value: u16) {
  output.extend_from_slice(&value.to_le_bytes());
}

fn le_u32(output: &mut Vec<u8>, value: u32) {
  output.extend_from_slice(&value.to_le_bytes());
}

fn encode_varint(output: &mut Vec<u8>, mut value: u64) {
  while value >= 0x80 {
    output.push((value as u8 & 0x7f) | 0x80);
    value >>= 7;
  }
  output.push(value as u8);
}
