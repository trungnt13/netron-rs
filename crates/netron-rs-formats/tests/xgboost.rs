use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_xgboost_json_booster_attributes() {
    let data = br#"{
        "version": [2, 1, 4],
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "learner_model_param": {
                "base_score": "5E-1",
                "boost_from_average": "1",
                "num_class": "0",
                "num_feature": "20",
                "num_target": "1"
            },
            "objective": {
                "name": "binary:logistic",
                "reg_loss_param": { "scale_pos_weight": "1" }
            },
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": "2"
                    },
                    "tree_info": [0, 0],
                    "trees": [{}, {}]
                }
            }
        }
    }"#;

    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("booster.json")),
    })
    .expect("xgboost json fixture parses");

    assert_eq!(model.format.name, "XGBoost JSON");
    assert_eq!(model.format.version.as_deref(), Some("2.1.4"));

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let node = &normalized["graphs"][0]["nodes"][0];
    assert_eq!(node["operator"]["name"], "xgboost.core.Booster");
    assert_eq!(node["inputs"], json!([]));

    let attributes = node["attributes"].as_array().unwrap();
    let attr = |name: &str| {
        attributes
            .iter()
            .find(|attribute| attribute["name"] == name)
            .unwrap_or_else(|| panic!("missing attribute {name}"))
    };
    assert_eq!(
        attr("version")["value"],
        json!({ "kind": "ints", "value": [2, 1, 4] })
    );
    assert_eq!(
        attr("objective")["value"],
        json!({ "kind": "string", "value": "binary:logistic" })
    );
    assert_eq!(
        attr("trees")["value"],
        json!({ "kind": "strings", "value": ["[object Object]", "[object Object]"] })
    );
    assert_eq!(
        attr("tree_info")["value"],
        json!({ "kind": "ints", "value": [0, 0] })
    );
}

#[test]
fn parses_xgboost_sklearn_classifier_pickle_stub() {
    let data = b"\x80\x04\x8c\x0fxgboost.sklearn\x94\x8c\rXGBClassifier\x94\x93\x94.";
    let model = parse(ModelInput {
        data,
        path: Some(std::path::Path::new("xgb_classifier.pkl")),
    })
    .expect("xgboost sklearn pickle fixture parses");

    assert_eq!(model.format.name, "scikit-learn");
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "xgboost.sklearn.XGBClassifier"
    );
}
