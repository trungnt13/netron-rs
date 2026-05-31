use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_message_graph_values_and_initializers() {
    let data = br#"{
        "signature": "netron:pytorch",
        "format": "TorchScript v2.6.0.dev20241105",
        "graphs": [{
            "values": [
                { "name": "input", "type": { "dataType": "float32", "shape": { "dimensions": [1, "N"] } } },
                { "name": "weight", "initializer": { "type": { "dataType": "float32", "shape": { "dimensions": [2, 3] } } } },
                { "name": "out", "type": { "dataType": "float32", "shape": { "dimensions": [1] } } }
            ],
            "inputs": [{ "value": [0] }],
            "outputs": [{ "value": [2] }],
            "nodes": [{
                "name": "linear",
                "type": { "name": "aten::linear" },
                "inputs": [{ "value": [0] }, { "value": [1] }],
                "outputs": [{ "value": [2] }],
                "attributes": [
                    { "name": "training", "type": "boolean", "value": 1 },
                    { "name": "sym", "type": "SymInt", "value": 32 },
                    { "name": "maybe", "type": "Tensor?", "value": null },
                    { "name": "pads", "value": [1, 2] }
                ]
            }]
        }]
    }"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("model.message")),
    })
    .expect("message fixture parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("2.6.0"));

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["input"]));
    assert_eq!(graph["outputs"], json!(["out"]));
    assert_eq!(graph["nodes"][0]["operator"]["name"], "aten::linear");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["input", "weight"]));
    assert_eq!(graph["nodes"][0]["outputs"], json!(["out"]));

    let attributes = graph["nodes"][0]["attributes"].as_array().unwrap();
    let attr = |name: &str| {
        attributes
            .iter()
            .find(|attribute| attribute["name"] == name)
            .unwrap_or_else(|| panic!("missing attribute {name}"))
    };
    assert_eq!(
        attr("training")["value"],
        json!({ "kind": "int", "value": 1 })
    );
    assert_eq!(
        attr("sym")["value"],
        json!({ "kind": "string", "value": "32" })
    );
    assert_eq!(attr("maybe")["value"], json!({ "kind": "null" }));
    assert_eq!(
        attr("pads")["value"],
        json!({ "kind": "ints", "value": [1, 2] })
    );

    let values = graph["values"].as_array().unwrap();
    let weight = values
        .iter()
        .find(|value| value["name"] == "weight")
        .expect("initializer value exists");
    assert_eq!(weight["initializer"], 0);
    assert_eq!(weight["type"]["element_type"], "float32");
}

#[test]
fn parses_message_modules_with_named_arguments() {
    let data = br#"{
        "signature": "netron:modular",
        "format": "Modular 0.0",
        "modules": [{
            "arguments": [
                { "name": "input", "type": { "dataType": "float32", "shape": { "dimensions": [1] } } },
                { "name": "output" }
            ],
            "nodes": [{
                "type": { "name": "conv" },
                "inputs": [{ "arguments": ["input"] }],
                "outputs": [{ "arguments": ["output"] }]
            }]
        }]
    }"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("model.maxviz")),
    })
    .expect("message module fixture parses");

    assert_eq!(model.format.name, "Modular");
    assert_eq!(model.format.version.as_deref(), Some("0.0"));

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let node = &normalized["graphs"][0]["nodes"][0];
    assert_eq!(node["operator"]["name"], "conv");
    assert_eq!(node["inputs"], json!(["input"]));
    assert_eq!(node["outputs"], json!(["output"]));
}
