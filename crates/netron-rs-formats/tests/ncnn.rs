use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_ncnn_text_param_weights_and_attributes() {
    let data = br#"7767517
4 4
Input data 0 1 data 0=1 1=3 2=16 3=16
Convolution conv1 1 1 data conv1 0=8 1=3 11=3 5=1 6=216
Pooling pool1 1 1 conv1 pool1 0=0 1=2 2=2
Softmax prob 1 1 pool1 prob
"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("model.param")),
    })
    .expect("ncnn param parses");

    assert_eq!(model.format.name, "ncnn");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.tensors.len(), 2);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["data"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 3);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "Convolution");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["data", "", ""]));
    assert_eq!(graph["nodes"][0]["outputs"], json!(["conv1"]));
    assert_eq!(graph["nodes"][1]["operator"]["name"], "Pooling");

    let attrs = graph["nodes"][0]["attributes"].as_array().unwrap();
    let attr = |name: &str| {
        attrs
            .iter()
            .find(|attribute| attribute["name"] == name)
            .unwrap_or_else(|| panic!("missing attribute {name}"))
    };
    assert_eq!(
        attr("num_output")["value"],
        json!({ "kind": "int", "value": 8 })
    );
    assert_eq!(
        attr("kernel_w")["value"],
        json!({ "kind": "int", "value": 3 })
    );

    let values = graph["values"].as_array().unwrap();
    let input = values
        .iter()
        .find(|value| value["name"] == "data")
        .expect("input value exists");
    assert_eq!(input["type"]["shape"][0]["value"], json!(1));
    let weights = values
        .iter()
        .filter(|value| value["name"] == "" && value["initializer"].is_number())
        .collect::<Vec<_>>();
    assert_eq!(weights.len(), 2);
    assert_eq!(weights[0]["type"]["element_type"], "0");
    assert_eq!(weights[0]["type"]["shape"][0]["value"], json!(8));
}
