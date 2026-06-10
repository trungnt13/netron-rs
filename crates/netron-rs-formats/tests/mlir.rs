use netron_rs_core::ModelError;
use netron_rs_formats::{ModelInput, ToNormalizedJson, inspect_mlir_bytecode, parse};
use serde_json::json;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::{fs, os::unix::fs::symlink, time::SystemTime};

#[test]
fn parses_mlir_bytecode_header_dialects_and_ir_summary() {
  let path = mlirbc_fixture("model.mlirbc");
  let data = std::fs::read(&path).expect("fixture exists");
  let model = parse(ModelInput {
    data: &data,
    path: Some(path.as_path()),
    allow_unsafe_paths: false,
  })
  .expect("MLIR bytecode parses");

  assert_eq!(model.format.name, "MLIR");
  assert_eq!(model.format.version.as_deref(), Some("Bytecode v6"));
  assert_eq!(model.metadata.producer.as_deref(), Some("MLIR19.0.0git"));
  let summary = inspect_mlir_bytecode(&data)
    .expect("bytecode inspection")
    .expect("bytecode summary");
  assert_eq!(summary.version, 6);
  assert_eq!(summary.producer, "MLIR19.0.0git");
  assert!(summary.string_count > 0);
  assert!(
    summary
      .operation_names
      .iter()
      .any(|name| name == "func.func")
  );
  assert!(!summary.ir.operations.is_empty());
  assert!(
    summary
      .ir
      .operations
      .iter()
      .any(|operation| !operation.operands.is_empty() || !operation.results.is_empty())
  );
  assert!(
    summary
      .ir
      .operations
      .iter()
      .any(|operation| operation.attributes.is_some() || operation.properties.is_some())
  );
  assert_eq!(summary.property_count, summary.properties.len());
  let property_index = summary
    .ir
    .operations
    .iter()
    .find_map(|operation| operation.properties)
    .expect("operation should reference bytecode properties");
  let property = &summary.properties[property_index];
  assert_eq!(property.index, property_index);
  assert!(property.len > 0);
  assert!(!property.preview_hex.is_empty());
  assert_eq!(summary.attribute_count, summary.attributes.len());
  assert!(!summary.attributes.is_empty());
  assert!(
    summary
      .attributes
      .iter()
      .enumerate()
      .all(|(index, entry)| { entry.index == index && !entry.dialect.is_empty() && entry.len > 0 })
  );
  assert!(
    summary
      .attributes
      .iter()
      .any(|entry| !entry.preview_hex.is_empty())
  );
  assert!(summary.attributes.iter().any(|entry| {
    entry
      .assembly
      .as_deref()
      .is_some_and(|assembly| assembly.starts_with("\"./stable_diffusion_3_medium_diffusers"))
  }));
  assert!(
    summary
      .attributes
      .iter()
      .any(|entry| entry.assembly.as_deref() == Some("0.000001 : f64"))
  );
  assert!(summary.attributes.iter().any(|entry| {
    entry.assembly.as_deref().is_some_and(|assembly| {
      assembly.starts_with("loc(\"./stable_diffusion_3_medium_diffusers_bs1_77_1024x1024_fp16")
    })
  }));
  assert_eq!(summary.type_count, summary.types.len());
  assert!(!summary.types.is_empty());
  assert!(
    summary
      .types
      .iter()
      .enumerate()
      .all(|(index, entry)| { entry.index == index && !entry.dialect.is_empty() && entry.len > 0 })
  );
  assert!(
    summary
      .types
      .iter()
      .any(|entry| !entry.preview_hex.is_empty())
  );
  let assembly_type = summary
    .types
    .iter()
    .find(|entry| !entry.has_custom_encoding && entry.assembly.is_some())
    .expect("assembly fallback bytecode type");
  assert!(
    assembly_type
      .assembly
      .as_deref()
      .is_some_and(|assembly| assembly.starts_with('!'))
  );
  assert_eq!(
    summary.types[0].assembly.as_deref(),
    Some("tensor<1536xf16>")
  );
  assert!(summary.types.iter().any(|entry| {
    entry.has_custom_encoding
      && entry.dialect == "builtin"
      && entry
        .assembly
        .as_deref()
        .is_some_and(|assembly| assembly.starts_with("tensor<"))
  }));
  assert!(summary.ir.values.iter().any(|value| {
    value
      .type_index
      .and_then(|type_index| summary.types.get(type_index))
      .and_then(|entry| entry.assembly.as_deref())
      .is_some()
  }));
  assert!(!summary.ir.values.is_empty());
  assert!(summary.ir.values.iter().all(|value| {
    value
      .type_index
      .is_none_or(|type_index| type_index < summary.type_count)
  }));
  assert!(
    summary
      .ir
      .values
      .iter()
      .any(|value| value.kind == "operation_result" && value.type_index.is_some())
  );
  assert!(
    summary
      .ir
      .values
      .iter()
      .any(|value| value.kind == "block_argument" && value.type_index.is_some())
  );
  assert!(!summary.locations.is_empty());
  let decoded_location = summary
    .ir
    .operations
    .iter()
    .find_map(|operation| {
      let location = operation.location?;
      summary
        .locations
        .iter()
        .find(|decoded| decoded.attribute == location)
    })
    .expect("operation location should decode");
  assert!(
    decoded_location
      .file
      .as_deref()
      .is_some_and(|file| file.ends_with(".mlir"))
  );
  assert!(decoded_location.line.is_some());
  assert!(!summary.sections.is_empty());
  assert!(!model.metadata.properties.contains_key("bytecode.version"));
  assert_eq!(model.graphs.len(), 1);
  assert!(!model.functions.is_empty());
  assert_eq!(model.strings.get(model.functions[0].name), "run_forward");
  assert_eq!(model.graphs[0].nodes.len(), 1);
  assert!(model.functions[0].nodes.len() > 10);
  assert!(!model.functions[0].values.is_empty());
  let decoded_bytecode_values = model.functions[0]
    .values
    .iter()
    .filter(|value| value.metadata.contains_key("bytecode.value.kind"))
    .collect::<Vec<_>>();
  assert!(!decoded_bytecode_values.is_empty());
  assert!(decoded_bytecode_values.iter().all(|value| {
    model.strings.get(value.name).starts_with('%')
      && value
        .metadata
        .get("bytecode.value")
        .and_then(|raw| raw.parse::<usize>().ok())
        .is_some()
  }));
  assert!(
    decoded_bytecode_values
      .iter()
      .filter(|value| value
        .metadata
        .get("bytecode.value.kind")
        .is_some_and(|kind| kind == "block_argument"))
      .all(|value| {
        let name = model.strings.get(value.name);
        name.starts_with("%arg") || name.starts_with("%bb")
      })
  );
  assert!(decoded_bytecode_values.iter().any(|value| {
    value
      .metadata
      .get("bytecode.value.kind")
      .is_some_and(|kind| kind == "operation_result")
      && model
        .strings
        .get(value.name)
        .strip_prefix('%')
        .is_some_and(|name| name.chars().all(|ch| ch.is_ascii_digit()))
  }));
  assert!(
    model.functions[0]
      .nodes
      .iter()
      .any(|node| !node.inputs.is_empty() || !node.outputs.is_empty())
  );
  assert!(
    model.functions[0]
      .nodes
      .iter()
      .any(|node| model.strings.get(node.operator.name) == "func.func")
  );
  let func_op = model.functions[0]
    .nodes
    .iter()
    .find(|node| model.strings.get(node.operator.name) == "func.func")
    .expect("decoded bytecode func.func operation");
  assert_eq!(
    func_op.metadata.get("bytecode.symbol").map(String::as_str),
    Some("run_forward")
  );
  assert_eq!(
    func_op
      .metadata
      .get("bytecode.attributes.assembly")
      .map(String::as_str),
    Some("{torch.assume_strict_symbolic_shapes = unit}")
  );
  let func_attributes = func_op
    .metadata
    .get("bytecode.attributes")
    .and_then(|index| index.parse::<usize>().ok())
    .and_then(|index| summary.attributes.get(index))
    .expect("func.func bytecode attributes decode");
  assert_eq!(
    func_attributes.assembly.as_deref(),
    Some("{torch.assume_strict_symbolic_shapes = unit}")
  );
}

#[test]
fn rejects_bad_mlir_bytecode_magic() {
  let error = parse(ModelInput {
    data: b"MLIR-not-bytecode",
    path: Some(Path::new("bad.mlirbc")),
    allow_unsafe_paths: false,
  })
  .expect_err("bad bytecode should fail");
  assert!(matches!(
    error,
    ModelError::InvalidData { format: "MLIR", .. }
  ));
}

#[test]
fn parses_mlir_functions_calls_and_dense_constants() {
  let data = br#"module @jit_mlp attributes {mhlo.num_partitions = 1 : i32} {
  func.func private @relu(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x128xf32>
    return %1 : tensor<1x128xf32>
  }
  func.func public @main(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
    %0 = call @relu(%arg0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    return %0 : tensor<1x128xf32>
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("model.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  assert_eq!(model.format.name, "MLIR");
  assert_eq!(model.graphs.len(), 1);
  assert_eq!(model.functions.len(), 2);
  assert_eq!(model.tensors.len(), 1);

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"][0]["name"], "@jit_mlp");
  assert_eq!(
    normalized["graphs"][0]["metadata"]["mhlo.num_partitions"],
    "1"
  );
  assert_eq!(normalized["functions"][0]["name"], "@jit_mlp::@relu");
  assert_eq!(normalized["functions"][0]["inputs"], json!(["%arg0"]));
  assert_eq!(normalized["functions"][0]["outputs"], json!(["%1"]));
  assert_eq!(
    normalized["functions"][0]["nodes"][0]["operator"]["name"],
    "stablehlo.broadcast_in_dim"
  );
  assert_eq!(
    normalized["functions"][1]["nodes"][0]["attributes"][0]["value"],
    json!({ "kind": "reference", "value": "@jit_mlp::@relu" })
  );
}

#[test]
fn rejects_mlir_absolute_node_location() {
  let location = std::env::temp_dir()
    .join("netron-rs-denied.bin")
    .to_string_lossy()
    .into_owned();
  let data = mlir_with_location(&location);
  let error = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("module.mlir")),
    allow_unsafe_paths: false,
  })
  .expect_err("absolute node locations should be denied");
  assert_access_denied(error, "module.mlir", &location);
}

#[test]
fn rejects_mlir_node_location_with_traversal() {
  for location in ["../outside.bin", r"..\outside.bin"] {
    let data = mlir_with_location(location);
    let error = parse(ModelInput {
      data: &data,
      path: Some(std::path::Path::new("module.mlir")),
      allow_unsafe_paths: false,
    })
    .expect_err("traversal node locations should be denied");
    assert_access_denied(error, "module.mlir", location);
  }
}

#[test]
fn rejects_mlir_node_uri_locations() {
  for location in [
    "http://example.com/t.bin",
    "ftp://example.com/t.bin",
    "file://host/share/t.bin",
  ] {
    let data = mlir_with_location(location);
    let error = parse(ModelInput {
      data: &data,
      path: Some(std::path::Path::new("module.mlir")),
      allow_unsafe_paths: false,
    })
    .expect_err("uri node locations should be denied");
    assert_access_denied(error, "module.mlir", location);
  }
}

#[cfg(unix)]
#[test]
fn rejects_mlir_node_location_symlink_escape_when_canonicalized() {
  let unique = SystemTime::now()
    .duration_since(SystemTime::UNIX_EPOCH)
    .unwrap()
    .as_nanos();
  let temp_root = std::env::temp_dir().join(format!("netron-mlir-{}", unique));
  let model_dir = temp_root.join("model");
  let link_dir = model_dir.join("links");
  let target_dir = temp_root.join("external");
  fs::create_dir_all(&link_dir).unwrap();
  fs::create_dir_all(&target_dir).unwrap();
  let model_path = model_dir.join("module.mlir");
  let target_file = target_dir.join("outside.bin");
  fs::write(&target_file, b"").unwrap();
  symlink(&target_file, link_dir.join("outside.bin")).unwrap();
  let data = mlir_with_location("links/outside.bin");

  let error = parse(ModelInput {
    data: &data,
    path: Some(model_path.as_path()),
    allow_unsafe_paths: false,
  })
  .expect_err("symlink escape should be denied");

  assert_access_denied(error, "module.mlir", "links/outside.bin");
  fs::remove_dir_all(&temp_root).unwrap();
}

#[test]
fn parses_mlir_node_location_when_unsafe_allowed() {
  let location = std::env::temp_dir()
    .join("netron-rs-trusted.bin")
    .to_string_lossy()
    .into_owned();
  let data = mlir_with_location(&location);
  let _ = parse(ModelInput {
    data: &data,
    path: Some(std::path::Path::new("module.mlir")),
    allow_unsafe_paths: true,
  })
  .expect("unsafe paths should be allowed for MLIR when enabled");
}

#[test]
fn parses_legacy_mlir_aliases_prototypes_and_encoded_types() {
  let data = br#"#strided1D = (d0) -> (d0)
func @gpu_alloc(memref<?xi8>)
func @main(%arg0: !torch.vtensor<[3,2],f32>, %flag: i1) -> tensor<3x2xf32, #strided1D> {
  %0 = func.call @helper(%arg0) : (!torch.vtensor<[3,2],f32>) -> tensor<3x2xf32, #strided1D> loc(#loc)
  %1 = scf.if %flag -> (tensor<3x2xf32, #strided1D>) {
    scf.yield %0 : tensor<3x2xf32, #strided1D>
  } else {
    scf.yield %0 : tensor<3x2xf32, #strided1D>
  }
  return %1 : tensor<3x2xf32, #strided1D>
}
func @helper(%arg0: !torch.vtensor<[3,2],f32>) -> tensor<3x2xf32, #strided1D>
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("legacy.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["metadata"]["properties"]["#strided1D"],
    "affine_map<(d0) -> (d0)>"
  );
  assert_eq!(normalized["functions"].as_array().unwrap().len(), 3);
  assert_eq!(normalized["functions"][0]["name"], "@gpu_alloc");
  assert_eq!(normalized["functions"][0]["inputs"], json!(["%arg0"]));
  assert_eq!(normalized["functions"][1]["name"], "@main");
  assert_eq!(
    normalized["functions"][1]["values"][0]["type"]["shape"],
    json!([
        { "kind": "known", "value": 3 },
        { "kind": "known", "value": 2 }
    ])
  );
  assert_eq!(
    normalized["functions"][1]["nodes"][1]["inputs"],
    json!(["%flag"])
  );
  assert_eq!(
    normalized["functions"][1]["nodes"][0]["outputs"],
    json!(["%0"])
  );
  assert_eq!(
    normalized["functions"][1]["values"][2]["type"]["shape"],
    json!([
        { "kind": "known", "value": 3 },
        { "kind": "known", "value": 2 }
    ])
  );
}

#[test]
fn parses_module_metadata_list_values() {
  let data = br#"module attributes {hal.device.targets = [#hal.device.target]} {
  func.func @main() {
    return
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("module-metadata.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let parsed_targets = normalized["graphs"][0]["metadata"]["hal.device.targets"]
    .as_str()
    .unwrap();
  assert_eq!(
    serde_json::from_str::<serde_json::Value>(parsed_targets).unwrap(),
    json!(["#hal.device.target"])
  );
}

#[test]
fn propagates_casted_convolution_input_types() {
  let data = br#"module {
  func.func @conv(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<1x225x225x3xf32>
    %1 = hal.tensor.cast %arg1 : !hal.buffer_view -> tensor<3x3x3x32xf32>
    %2 = mhlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} :
      (tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
    %3 = hal.tensor.cast %2 : tensor<1x112x112x32xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("conv-cast.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let values = normalized["functions"][0]["values"]
    .as_array()
    .expect("values array");
  let value_shape = |name: &str| {
    values
      .iter()
      .find(|value| value["name"] == name)
      .and_then(|value| value["type"]["shape"].as_array())
      .expect("value shape")
      .iter()
      .map(|dimension| dimension["value"].as_i64())
      .collect::<Vec<_>>()
  };

  assert_eq!(
    value_shape("%0"),
    vec![Some(1), Some(112), Some(112), Some(32)]
  );
  assert_eq!(
    value_shape("%1"),
    vec![Some(1), Some(112), Some(112), Some(32)]
  );
}

#[test]
fn parses_module_globals_and_torch_constant_folding() {
  let data = br#"module @module {
  util.global private @weights = #stream.parameter.named<"model"::"weights"> : tensor<2xbf16>
  func.func @main(%arg0: !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],f32> {
    %weights = util.global.load @weights : tensor<2xbf16>
    %0 = torch_c.from_builtin_tensor %weights : tensor<2xbf16> -> !torch.vtensor<[2],bf16>
    %sym = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : !torch.int
    %int-1 = torch.constant.int -1
    %int2 = torch.constant.int 2
    %shape = torch.prim.ListConstruct %int-1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %view = torch.aten.view %arg0, %shape : !torch.vtensor<[2],si64>, !torch.list<int> -> !torch.vtensor<[2],si64>
    %lit = torch.vtensor.literal(dense_resource<token_ids> : tensor<1x2xsi64>) : !torch.vtensor<[1,2],si64>
    %cpu = torch.constant.device "cpu"
    %out = torch.prims.convert_element_type %view, %int2 : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[2],f32>
    return %out : !torch.vtensor<[2],f32>
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("globals.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["operator"]["name"],
    "util.global"
  );
  assert_eq!(
    normalized["graphs"][0]["nodes"][0]["attributes"],
    json!([
        { "name": "initial_value", "value": { "kind": "string", "value": "#stream.parameter.named<\"model\"::\"weights\">" } },
        { "name": "sym_name", "value": { "kind": "string", "value": "weights" } },
        { "name": "sym_visibility", "value": { "kind": "string", "value": "private" } },
        { "name": "type", "value": { "kind": "string", "value": "tensor<2xbf16>" } }
    ])
  );

  let nodes = normalized["functions"][0]["nodes"].as_array().unwrap();
  assert_eq!(nodes[0]["attributes"][0]["value"]["value"], "weights");
  assert_eq!(
    nodes
      .iter()
      .find(|node| node["operator"]["name"] == "torch.symbolic_int")
      .unwrap()["attributes"],
    json!([
        { "name": "max_val", "value": { "kind": "string", "value": "9223372036854776000" } },
        { "name": "min_val", "value": { "kind": "string", "value": "2" } },
        { "name": "symbol_name", "value": { "kind": "string", "value": "s0" } }
    ])
  );
  assert_eq!(
    nodes
      .iter()
      .find(|node| node["operator"]["name"] == "aten.view")
      .unwrap()["inputs"],
    json!(["%arg0", null, null])
  );
  assert!(
    nodes
      .iter()
      .all(|node| node["operator"]["name"] != "prim.ListConstruct")
  );
  assert_eq!(
    nodes
      .iter()
      .find(|node| node["operator"]["name"] == "torch.vtensor.literal")
      .unwrap()["attributes"][0]["value"]["kind"],
    "tensor"
  );
  assert_eq!(
    nodes
      .iter()
      .find(|node| node["outputs"] == json!(["%cpu"]))
      .unwrap()["operator"]["name"],
    "torch.constant"
  );
  assert!(
    nodes
      .iter()
      .any(|node| node["operator"]["name"] == "prims.convert_element_type")
  );
}

#[test]
fn keeps_cfg_block_arguments_out_of_function_inputs() {
  let data = br#"func.func @loop() {
  ^bb1(%i: index):
    return
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("loop.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["functions"][0]["name"], "@loop");
  assert_eq!(normalized["functions"][0]["inputs"], json!([]));
}

#[test]
fn parses_scf_for_bounds_and_iter_args_as_inputs() {
  let data = br#"func.func @loop(%arg0: tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant dense<0.0> : tensor<4xf32>
  %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<4xf32>) {
    scf.yield %acc : tensor<4xf32>
  }
  return
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("scf-for.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let nodes = normalized["functions"][0]["nodes"].as_array().unwrap();
  let scf_for = nodes
    .iter()
    .find(|node| node["operator"]["name"] == "scf.for")
    .unwrap();
  assert_eq!(scf_for["inputs"], json!(["%c0", "%c4", "%c1", "%init"]));
  assert_eq!(
    scf_for["attributes"],
    json!([
        { "name": "operandSegmentSizes", "value": { "kind": "ints", "value": [1, 1, 1, 1] } }
    ])
  );
}

#[test]
fn parses_masked_tt_load_operand_segments() {
  let data = br#"tt.func @masked(%arg0: tensor<4x!tt.ptr<f16>>, %arg1: tensor<4xi1>) {
  %0 = tt.load %arg0, %arg1 : tensor<4x!tt.ptr<f16>>
  tt.return
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("tt-load.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let node = &normalized["functions"][0]["nodes"][0];
  let loaded = normalized["functions"][0]["values"]
    .as_array()
    .unwrap()
    .iter()
    .find(|value| value["name"] == "%0")
    .unwrap();
  assert_eq!(node["operator"]["name"], "tt.load");
  assert_eq!(loaded["type"]["element_type"], "float16");
  assert_eq!(
    node["attributes"],
    json!([
        { "name": "operandSegmentSizes", "value": { "kind": "ints", "value": [1, 1, 0] } }
    ])
  );
}

#[test]
fn parses_iree_dispatch_and_subspan_syntax() {
  let data = br#"func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %wg = hal.interface.workgroup.id[0] : index
  %span = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<1x4xf32>
  %init = linalg.init_tensor [1, %c4] : tensor<1x?xf32>
  %0 = flow.dispatch @dispatch::@main[%c4](%arg0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("iree-dispatch.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let nodes = normalized["functions"][0]["nodes"].as_array().unwrap();
  let subspan = nodes
    .iter()
    .find(|node| node["operator"]["name"] == "hal.interface.binding.subspan")
    .unwrap();
  assert_eq!(subspan["inputs"], json!(["%c0"]));
  assert_eq!(
    subspan["attributes"],
    json!([
        { "name": "layout", "value": { "kind": "string", "value": "@io::@s0b0_ro_external" } }
    ])
  );
  let init = nodes
    .iter()
    .find(|node| node["operator"]["name"] == "linalg.init_tensor")
    .unwrap();
  assert_eq!(
    init["attributes"],
    json!([
        { "name": "static_sizes", "value": { "kind": "strings", "value": ["1", "%c4"] } }
    ])
  );
  let workgroup = nodes
    .iter()
    .find(|node| node["operator"]["name"] == "hal.interface.workgroup.id")
    .unwrap();
  assert_eq!(
    workgroup["attributes"],
    json!([
        { "name": "dimension", "value": { "kind": "string", "value": "0" } }
    ])
  );
  let dispatch = nodes
    .iter()
    .find(|node| node["operator"]["name"] == "flow.dispatch")
    .unwrap();
  assert_eq!(dispatch["inputs"], json!(["%c4", "%arg0"]));
  assert_eq!(
    dispatch["attributes"],
    json!([
        { "name": "entry_points", "value": { "kind": "string", "value": "@dispatch::@main" } },
        { "name": "operandSegmentSizes", "value": { "kind": "ints", "value": [1, 1, 0, 0] } }
    ])
  );
}

#[test]
fn anonymous_module_wrappers_do_not_emit_graphs_or_prefix_functions() {
  let data = br#"module {
  func.func @main() {
    return
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("anonymous.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"].as_array().unwrap().len(), 0);
  assert_eq!(normalized["functions"][0]["name"], "@main");
}

#[test]
fn repeated_anonymous_modules_get_synthetic_scope_prefixes() {
  let data = br#"module {
  tt.func public @a() {
    tt.return
  }
}
module attributes {"triton_gpu.num-warps" = 16 : i32} {
  tt.func public @a() {
    tt.return
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("anonymous-repeated.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["functions"]
      .as_array()
      .unwrap()
      .iter()
      .map(|function| function["name"].as_str().unwrap())
      .collect::<Vec<_>>(),
    vec!["$0::@a", "$1::@a"]
  );
  assert_eq!(normalized["graphs"].as_array().unwrap().len(), 1);
  assert_eq!(normalized["graphs"][0]["name"], "$1");
  assert_eq!(
    normalized["graphs"][0]["metadata"]["triton_gpu.num-warps"],
    "16"
  );
}

#[test]
fn repeated_anonymous_modules_keep_synthetic_function_scope() {
  let data = br#"module {
  func.func @main() {
    return
  }
}
module {
  func.func @main() {
    return
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("multi_dump.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(normalized["graphs"].as_array().unwrap().len(), 0);
  assert_eq!(normalized["functions"][0]["name"], "$0::@main");
  assert_eq!(normalized["functions"][1]["name"], "$1::@main");
}

#[test]
fn unquoted_builtin_module_is_recognized_for_scoping() {
  let data = br#"module {
  func.func @parent() {
    return
  }
  builtin.module {
    func.func @inner() {
      return
    }
  }
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("builtin-module.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["functions"]
      .as_array()
      .unwrap()
      .iter()
      .map(|function| function["name"].as_str().unwrap())
      .collect::<Vec<_>>(),
    vec!["$0::@parent", "$0::$1::@inner"]
  );
}

#[test]
fn parses_hal_executable_variant_target_attribute() {
  let data = br#"hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, GPU, {}>}> {
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("hal-executable-variant.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let attributes = normalized["graphs"][0]["nodes"][0]["attributes"]
    .as_array()
    .unwrap();
  assert_eq!(
    attributes
      .iter()
      .find(|attribute| attribute["name"] == "target")
      .unwrap()["value"]["value"],
    "#hal.executable.target<\"vulkan\", \"vulkan-spirv-fb\", {spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, GPU, {}>}>"
  );
}

#[test]
fn parses_hal_device_query_key_pair() {
  let data = br#"func.func @main(%device: !hal.device) {
  %ok, %value = hal.device.query<%device : !hal.device> key("hal.executable.format" :: "vulkan-spirv-fb") : i1, i1 = false
  return
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("hal-device-query.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  let nodes = normalized["functions"][0]["nodes"].as_array().unwrap();
  let query = nodes
    .iter()
    .find(|node| node["operator"]["name"] == "hal.device.query")
    .unwrap();
  assert_eq!(query["inputs"], json!(["%device"]));
  assert_eq!(
    query["attributes"],
    json!([
        { "name": "category", "value": { "kind": "string", "value": "hal.executable.format" } },
        { "name": "default", "value": { "kind": "string", "value": "false" } },
        { "name": "key", "value": { "kind": "string", "value": "vulkan-spirv-fb" } }
    ])
  );
}

#[test]
fn parses_spirv_and_vm_custom_syntax_attributes() {
  let data = br#"spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.GlobalVariable @resource bind(2, 4) : !spv.ptr<i32, StorageBuffer>
  %addr = spv.mlir.addressof @resource : !spv.ptr<i32, StorageBuffer>
  %0 = spv.Load "StorageBuffer" %addr : i32
  spv.EntryPoint "GLCompute" @main, @resource
  spv.ExecutionMode @main "LocalSize", 8, 2, 1
}
vm.func @main() {
  %c1 = constant 1 : i32
  %buffer = vm.const.ref.rodata @blob : !vm.buffer
  vm.cond_br %buffer, ^bb1, ^bb2
^bb1:
  vm.global.store.i32 %c1, @_flag : i32
  vm.br ^bb3(%buffer : !vm.buffer)
^bb2:
  vm.fail %c2, "device not supported in the compiled configuration"
^bb3(%arg: !vm.buffer):
  vm.return
}
"#;

  let model = parse(ModelInput {
    data,
    path: Some(std::path::Path::new("spirv-vm.mlir")),
    allow_unsafe_paths: false,
  })
  .expect("MLIR parses");

  let normalized: serde_json::Value =
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
  assert_eq!(
    normalized["graphs"][0]["metadata"]["addressing_model"],
    "Logical"
  );
  assert_eq!(
    normalized["graphs"][0]["metadata"]["memory_model"],
    "GLSL450"
  );
  assert_eq!(
    normalized["graphs"][0]["metadata"]["vce_triple"],
    "#spv.vce<v1.0, [Shader], []>"
  );

  let graph_nodes = normalized["graphs"][0]["nodes"].as_array().unwrap();
  assert_eq!(
    graph_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "spv.GlobalVariable")
      .unwrap()["attributes"],
    json!([
        { "name": "sym_name", "value": { "kind": "string", "value": "resource" } },
        { "name": "binding", "value": { "kind": "int", "value": 2 } },
        { "name": "descriptor_set", "value": { "kind": "int", "value": 4 } }
    ])
  );
  assert_eq!(
    graph_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "spv.Load")
      .unwrap()["attributes"][0]["value"]["value"],
    "StorageBuffer"
  );
  assert!(
    graph_nodes
      .iter()
      .any(|node| node["operator"]["name"] == "spv.EntryPoint")
  );

  let function_nodes = normalized["functions"][0]["nodes"].as_array().unwrap();
  assert_eq!(
    function_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "vm.const.ref.rodata")
      .unwrap()["attributes"][0]["value"]["value"],
    "blob"
  );
  assert_eq!(
    function_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "vm.cond_br")
      .unwrap()["attributes"][0]["value"]["value"],
    json!([1, 0, 0])
  );
  assert_eq!(
    function_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "vm.global.store.i32")
      .unwrap()["attributes"][0]["value"]["value"],
    "_flag"
  );
  assert_eq!(
    function_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "vm.br")
      .unwrap()["inputs"],
    json!([])
  );
  assert_eq!(
    function_nodes
      .iter()
      .find(|node| node["operator"]["name"] == "vm.fail")
      .unwrap()["attributes"][0]["value"]["value"],
    "device not supported in the compiled configuration"
  );
}

fn mlirbc_fixture(name: &str) -> PathBuf {
  let local = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../../tests/fixtures/mlir")
    .join(name);
  if local.exists() {
    return local;
  }

  Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../../../netron/third_party/test/mlir")
    .join(name)
}

fn mlir_with_location(location: &str) -> Vec<u8> {
  format!(
    r#"module {{
  func.func @main() {{
    %0 = arith.constant 0 : i32 loc("{location}")
    return
  }}
}}
"#
  )
  .into_bytes()
}

fn assert_access_denied(error: ModelError, source: &str, location: &str) {
  let ModelError::AccessDenied { path } = error else {
    panic!("expected access denied");
  };
  assert!(path.contains(source));
  assert!(path.contains(location));
}
