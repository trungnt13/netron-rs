use std::path::{Path, PathBuf};

use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::{Value, json};

#[test]
fn parses_tflite_split_skipgram_lstm_and_unpack_options() {
    let smartreply = parse_fixture("smartreply.tflite");
    let skip_gram = find_node(&smartreply, "SkipGram");
    assert_eq!(
        skip_gram["attributes"],
        json!([
            { "name": "ngram_size", "value": { "kind": "int", "value": 3 } },
            { "name": "max_skip_size", "value": { "kind": "int", "value": 2 } },
            { "name": "include_all_ngrams", "value": { "kind": "bool", "value": true } }
        ])
    );
    let lsh_projection = find_node(&smartreply, "LSHProjection");
    assert_eq!(
        lsh_projection["attributes"],
        json!([{ "name": "type", "value": { "kind": "string", "value": "SPARSE" } }])
    );

    let speakerid = parse_fixture("speech_speakerid_model_2017_11_14.tflite");
    let lstm = find_node(&speakerid, "LSTM");
    assert_eq!(
        lstm["attributes"],
        json!([{ "name": "cell_clip", "value": { "kind": "float", "value": 50.0 } }])
    );

    let subword = parse_fixture("subword-conformer.latest.tflite");
    let split_v = find_node(&subword, "SplitV");
    assert_eq!(
        split_v["attributes"],
        json!([{ "name": "num_splits", "value": { "kind": "int", "value": 3 } }])
    );
    let split = find_node_in_graph(&subword, 4, "Split");
    assert_eq!(
        split["attributes"],
        json!([{ "name": "num_splits", "value": { "kind": "int", "value": 4 } }])
    );
    let unpack = find_node_in_graph(&subword, 6, "Unpack");
    assert_eq!(
        unpack["attributes"],
        json!([{ "name": "num", "value": { "kind": "int", "value": 2 } }])
    );
    let while_node = find_node(&subword, "While");
    assert_eq!(
        while_node["attributes"],
        json!([
            { "name": "cond_subgraph_index", "value": { "kind": "int", "value": 5 } },
            { "name": "body_subgraph_index", "value": { "kind": "int", "value": 6 } }
        ])
    );
    let if_node = find_node_in_graph(&subword, 6, "If");
    assert_eq!(
        if_node["attributes"],
        json!([
            { "name": "then_subgraph_index", "value": { "kind": "int", "value": 2 } },
            { "name": "else_subgraph_index", "value": { "kind": "int", "value": 1 } }
        ])
    );
}

#[test]
fn hides_tflite_custom_fused_activation_attribute_like_netron() {
    let quicknet = parse_fixture("quicknet.tflite");
    let node = find_node(&quicknet, "LceBconv2d");
    let attributes = node["attributes"].as_array().expect("attributes array");

    assert_eq!(attributes.len(), 7);
    assert!(
        attributes
            .iter()
            .all(|attribute| attribute["name"] != "fused_activation_function")
    );
}

fn parse_fixture(name: &str) -> Value {
    let path = fixture_path(name);
    let data = std::fs::read(&path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", path.display());
    });
    let model = parse(ModelInput {
        data: &data,
        path: Some(&path),
    })
    .expect("TFLite fixture parses");
    serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../netron/third_party/test/tflite")
        .join(name)
}

fn find_node<'a>(model: &'a Value, operator: &str) -> &'a Value {
    find_node_in_graph(model, 0, operator)
}

fn find_node_in_graph<'a>(model: &'a Value, graph_index: usize, operator: &str) -> &'a Value {
    model["graphs"][graph_index]["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .find(|node| node["operator"]["name"] == operator)
        .unwrap_or_else(|| panic!("node {operator} exists in graph {graph_index}"))
}
