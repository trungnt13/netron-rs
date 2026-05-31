use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_dot_edges_into_graph_nodes() {
    let data = br#"digraph G {
        A -> B;
        B [label="name: B
type: Relu"];
    }"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("graph.dot")),
    })
    .expect("dot fixture parses");

    assert_eq!(model.format.name, "DOT");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.graphs[0].nodes.len(), 2);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(normalized["graphs"][0]["name"], "G");
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "Relu"
    );
    assert_eq!(normalized["graphs"][0]["nodes"][1]["operator"]["name"], "A");
    assert_eq!(normalized["graphs"][0]["nodes"][0]["inputs"], json!(["A"]));
    assert_eq!(normalized["graphs"][0]["nodes"][1]["outputs"], json!(["A"]));
}

#[test]
fn folds_dot_parameter_initializer() {
    let data = br#"digraph fx_graph {
        p [label="{p|op_code=get_parameter torch.float32[2,3]|}"];
        x [label="{x|op_code=placeholder|}"];
        p -> conv;
        x -> conv;
        conv [label="{conv|op_code=call_module\ltorch.nn.Conv2d|stride: (2, 2)\l}"];
    }"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("fx.dot")),
    })
    .expect("dot fixture parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][1]["operator"]["name"],
        "torch.nn.Conv2d"
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][1]["attributes"][0]["value"],
        json!({ "kind": "ints", "value": [2, 2] })
    );
    let values = normalized["graphs"][0]["values"].as_array().unwrap();
    let parameter = values
        .iter()
        .find(|value| value["name"] == "p")
        .expect("parameter edge exists");
    assert_eq!(parameter["initializer"], 0);
    assert_eq!(parameter["type"]["element_type"], "float32");
}
