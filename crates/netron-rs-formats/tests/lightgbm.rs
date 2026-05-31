use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_lightgbm_text_booster_features_and_attributes() {
    let data = br#"tree
version=v3
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=1
objective=regression
feature_names=f0 f1
tree_sizes=10 10

Tree=0
num_leaves=1
leaf_value=0

Tree=1
num_leaves=1
leaf_value=1

parameters:
[objective: regression]
[num_iterations: 2]
end of parameters
"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("model.txt")),
    })
    .expect("lightgbm text fixture parses");

    assert_eq!(model.format.name, "LightGBM");
    assert_eq!(model.format.version.as_deref(), Some("3"));

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(normalized["graphs"][0]["inputs"], json!(["f0", "f1"]));
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "lightgbm.basic.Booster"
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["f0", "f1"])
    );

    let attributes = normalized["graphs"][0]["nodes"][0]["attributes"]
        .as_array()
        .unwrap();
    let attr = |name: &str| {
        attributes
            .iter()
            .find(|attribute| attribute["name"] == name)
            .unwrap_or_else(|| panic!("missing attribute {name}"))
    };
    assert_eq!(
        attr("models")["value"],
        json!({ "kind": "strings", "value": ["[object Object]", "[object Object]"] })
    );
    assert_eq!(
        attr("loaded_parameter")["value"],
        json!({ "kind": "string", "value": "[objective: regression]\n[num_iterations: 2]\n" })
    );
    assert_eq!(
        attr("average_output")["value"],
        json!({ "kind": "bool", "value": false })
    );
}

#[test]
fn hides_large_lightgbm_feature_sets_as_graph_inputs() {
    let mut data = String::from(
        "tree\nversion=v3\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=1000\nobjective=regression\nfeature_names=",
    );
    for index in 0..1000 {
        if index > 0 {
            data.push(' ');
        }
        data.push_str(&format!("feature_{index}"));
    }
    data.push_str("\nTree=0\n");

    let model = parse(ModelInput {
        data: data.as_bytes(),
        path: Some(std::path::Path::new("large.model")),
    })
    .expect("lightgbm large feature fixture parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(normalized["graphs"][0]["inputs"], json!([]));
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"]
            .as_array()
            .unwrap()
            .len(),
        1000
    );
    assert_eq!(
        normalized["graphs"][0]["values"][0]["graph_input"],
        serde_json::Value::Null
    );
}

#[test]
fn parses_lightgbm_pickle_booster_payload() {
    let text = b"tree\nversion=v2\nfeature_names=f0\nTree=0\n";
    let mut data = b"\x80\x03clightgbm.basic\nBooster\nq\x00X".to_vec();
    data.extend_from_slice(&(text.len() as u32).to_le_bytes());
    data.extend_from_slice(text);
    data.extend_from_slice(b"q\x01.");

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("model.pkl")),
    })
    .expect("lightgbm pickle fixture parses");

    assert_eq!(model.format.name, "LightGBM Pickle");
    assert_eq!(model.format.version.as_deref(), Some("2"));
}
