use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_darknet_cfg_shapes_weights_and_routes() {
    let data = br#"
        [net]
        width=32
        height=32
        channels=3

        [convolutional]
        filters=8
        size=3
        stride=1
        pad=1
        activation=relu

        [maxpool]
        size=2
        stride=2

        [route]
        layers=-1

        [connected]
        output=10
        activation=linear
    "#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("model.cfg")),
    })
    .expect("darknet cfg parses");

    assert_eq!(model.format.name, "Darknet");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.tensors.len(), 4);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["input"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 4);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "convolutional");
    assert_eq!(graph["nodes"][1]["operator"]["name"], "maxpool");
    assert_eq!(graph["nodes"][2]["operator"]["name"], "route");
    assert_eq!(graph["nodes"][3]["operator"]["name"], "connected");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["input", "", ""]));
    assert_eq!(graph["nodes"][0]["outputs"], json!(["0"]));
    assert_eq!(graph["nodes"][3]["outputs"], json!(["3"]));

    let values = graph["values"].as_array().unwrap();
    let input = values
        .iter()
        .find(|value| value["name"] == "input")
        .expect("input value exists");
    assert_eq!(input["type"]["shape"][0]["value"], json!(32));
    assert_eq!(input["type"]["shape"][1]["value"], json!(32));
    assert_eq!(input["type"]["shape"][2]["value"], json!(3));

    let output = values
        .iter()
        .find(|value| value["name"] == "3")
        .expect("connected output exists");
    assert_eq!(output["type"]["shape"][0]["value"], json!(10));

    let attrs = graph["nodes"][0]["attributes"].as_array().unwrap();
    let attr = |name: &str| {
        attrs
            .iter()
            .find(|attribute| attribute["name"] == name)
            .unwrap_or_else(|| panic!("missing attribute {name}"))
    };
    assert_eq!(
        attr("filters")["value"],
        json!({ "kind": "int", "value": 8 })
    );
    assert_eq!(
        attr("activation")["value"],
        json!({ "kind": "string", "value": "relu" })
    );
}
