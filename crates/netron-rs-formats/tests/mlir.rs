use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

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
