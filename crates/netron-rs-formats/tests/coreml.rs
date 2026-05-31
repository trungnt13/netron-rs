use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_coreml_pipeline_with_duplicate_intermediate_outputs() {
    let data = pipeline_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("fixture.mlmodel")),
    })
    .expect("coreml fixture parses");

    assert_eq!(model.format.name, "Core ML");
    assert_eq!(model.format.version.as_deref(), Some("1"));
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.graphs[0].nodes.len(), 3);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["description"], "Pipeline");
    assert_eq!(graph["inputs"], json!(["x"]));
    assert_eq!(graph["outputs"], json!(["y"]));
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "arrayFeatureExtractor"
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!(["x"]));
    assert_eq!(graph["nodes"][0]["outputs"], json!(["mid"]));
    assert_eq!(
        graph["nodes"][0]["attributes"],
        json!([{ "name": "extractIndex", "value": { "kind": "strings", "value": ["0"] } }])
    );
    assert_eq!(graph["nodes"][1]["operator"]["name"], "oneHotEncoder");
    assert_eq!(graph["nodes"][1]["inputs"], json!(["mid"]));
    assert_eq!(graph["nodes"][1]["outputs"], json!(["mid|1"]));
    assert_eq!(
        graph["nodes"][1]["attributes"],
        json!([
            { "name": "outputSparse", "value": { "kind": "bool", "value": true } },
            { "name": "int64Categories", "value": { "kind": "string", "value": "[object Object]" } }
        ])
    );
    assert_eq!(graph["nodes"][2]["operator"]["name"], "featureVectorizer");
    assert_eq!(graph["nodes"][2]["inputs"], json!(["mid|1"]));
    assert_eq!(graph["nodes"][2]["outputs"], json!(["y"]));

    let values = graph["values"].as_array().unwrap();
    let value = |name: &str| {
        values
            .iter()
            .find(|value| value["name"] == name)
            .expect("value exists")
    };
    assert_eq!(value("x")["type"]["element_type"], "int64");
    assert_eq!(value("y")["type"]["element_type"], "float32");
    assert_eq!(
        value("y")["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }])
    );
    assert_eq!(value("mid")["producer"], 0);
    assert_eq!(value("mid")["consumers"], json!([1]));
    assert_eq!(value("mid|1")["producer"], 1);
    assert_eq!(value("mid|1")["consumers"], json!([2]));
}

#[test]
fn parses_coreml_tree_classifier_label_outputs() {
    let data = model(
        description_with_predictions(
            &[feature("data", Some(multi_array_type(65_600, &[13])))],
            &[
                feature("target", Some(int64_type())),
                feature("classProbability", Some(dictionary_type())),
            ],
            "target",
            "classProbability",
        ),
        402,
        tree_ensemble_classifier(),
    );

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("tree.mlmodel")),
    })
    .expect("coreml tree classifier parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["description"], "Tree Ensemble Classifier");
    assert_eq!(graph["outputs"], json!(["target", "classProbability"]));
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "treeEnsembleClassifier"
    );
    assert_eq!(
        graph["nodes"][0]["outputs"],
        json!(["target:labelProbabilityLayerName"])
    );
    assert_eq!(
        graph["nodes"][0]["attributes"],
        json!([
            { "name": "nodes", "value": { "kind": "strings", "value": ["[object Object]"] } },
            { "name": "basePredictionValue", "value": { "kind": "ints", "value": [0] } },
            { "name": "numPredictionDimensions", "value": { "kind": "string", "value": "1" } }
        ])
    );
    assert_eq!(graph["nodes"][1]["operator"]["name"], "int64ClassLabels");
    assert_eq!(
        graph["nodes"][1]["inputs"],
        json!(["target:labelProbabilityLayerName"])
    );
    assert_eq!(
        graph["nodes"][1]["outputs"],
        json!(["classProbability", "target"])
    );
    assert_eq!(
        graph["nodes"][1]["attributes"],
        json!([{ "name": "vector", "value": { "kind": "strings", "value": ["0", "1"] } }])
    );

    let values = graph["values"].as_array().unwrap();
    let value = |name: &str| {
        values
            .iter()
            .find(|value| value["name"] == name)
            .expect("value exists")
    };
    assert_eq!(value("classProbability")["type"], json!({ "shape": [] }));
    assert_eq!(value("target")["producer"], 1);
}

#[test]
fn parses_coreml_support_vector_regressor_attributes() {
    let data = model(
        description(
            &[feature("data", Some(multi_array_type(65_600, &[13])))],
            &[feature("target", Some(double_type()))],
        ),
        301,
        support_vector_regressor(),
    );

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("svm.mlmodel")),
    })
    .expect("coreml support vector regressor parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let node = &normalized["graphs"][0]["nodes"][0];
    assert_eq!(node["operator"]["name"], "supportVectorRegressor");
    assert_eq!(node["inputs"], json!(["data"]));
    assert_eq!(node["outputs"], json!(["target"]));
    assert_eq!(
        node["attributes"],
        json!([
            { "name": "coefficients", "value": { "kind": "string", "value": "[object Object]" } },
            { "name": "kernel", "value": { "kind": "string", "value": "[object Object]" } },
            { "name": "rho", "value": { "kind": "float", "value": -20.85139 } },
            { "name": "supportVectors", "value": { "kind": "string", "value": "denseSupportVectors" } }
        ])
    );
}

#[test]
fn parses_coreml_neural_network_embedding_with_anonymous_weights() {
    let data = model(
        description(
            &[feature("data", Some(multi_array_type(65_600, &[1])))],
            &[feature("target", Some(multi_array_type(65_600, &[3])))],
        ),
        500,
        neural_network_embedding(),
    );

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("embedding.mlmodel")),
    })
    .expect("coreml neural network parses");

    assert_eq!(model.tensors.len(), 1);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["description"], "Neural Network");
    let node = &graph["nodes"][0];
    assert_eq!(node["name"], "embedding_1");
    assert_eq!(node["operator"]["name"], "embedding");
    assert_eq!(node["inputs"], json!(["data", ""]));
    assert_eq!(node["outputs"], json!(["target"]));
    assert_eq!(
        node["attributes"],
        json!([
            { "name": "inputDim", "value": { "kind": "string", "value": "2" } },
            { "name": "outputChannels", "value": { "kind": "string", "value": "3" } }
        ])
    );

    let values = graph["values"].as_array().unwrap();
    let value = |name: &str| {
        values
            .iter()
            .find(|value| value["name"] == name)
            .expect("value exists")
    };
    assert_eq!(value("")["initializer"], 0);
    assert_eq!(value("")["type"]["element_type"], "float32");
    assert_eq!(
        value("")["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }, { "kind": "known", "value": 3 }])
    );
}

#[test]
fn parses_coreml_neural_network_shape_layers() {
    let layers = vec![
        neural_layer(
            "const",
            &[],
            &["c"],
            290,
            load_constant_layer(&[2], &[1.0, 2.0]),
        ),
        neural_layer("mul", &["x", "c"], &["m"], 231, multiply_layer(0.5)),
        neural_layer("reshape", &["m"], &["r"], 300, reshape_layer(&[1, 2])),
        neural_layer("slice", &["r"], &["y"], 350, slice_layer(0, 1, 1, 1)),
    ];
    let data = model(
        description(
            &[feature("x", Some(multi_array_type(65_600, &[2])))],
            &[feature("y", Some(multi_array_type(65_600, &[1])))],
        ),
        500,
        neural_network(layers),
    );

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("shape.mlmodel")),
    })
    .expect("coreml neural network shape layers parse");

    assert_eq!(model.tensors.len(), 1);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["attributes"],
        json!([{ "name": "shape", "value": { "kind": "strings", "value": ["2"] } }])
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!([""]));
    assert_eq!(
        graph["nodes"][1]["attributes"],
        json!([{ "name": "alpha", "value": { "kind": "float", "value": 0.5 } }])
    );
    assert_eq!(
        graph["nodes"][2]["attributes"],
        json!([{ "name": "targetShape", "value": { "kind": "strings", "value": ["1", "2"] } }])
    );
    assert_eq!(
        graph["nodes"][3]["attributes"],
        json!([
            { "name": "startIndex", "value": { "kind": "string", "value": "0" } },
            { "name": "endIndex", "value": { "kind": "string", "value": "1" } },
            { "name": "stride", "value": { "kind": "string", "value": "1" } },
            { "name": "axis", "value": { "kind": "int", "value": 1 } }
        ])
    );
}

#[test]
fn parses_coreml_gru_weights() {
    let data = model(
        description(
            &[
                feature("x", Some(multi_array_type(65_600, &[1, 3]))),
                feature("h_in", Some(multi_array_type(65_600, &[2]))),
            ],
            &[
                feature("y", Some(multi_array_type(65_600, &[1, 2]))),
                feature("h_out", Some(multi_array_type(65_600, &[2]))),
            ],
        ),
        500,
        neural_network(vec![neural_layer(
            "gru",
            &["x", "h_in"],
            &["y", "h_out"],
            410,
            gru_layer(),
        )]),
    );

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("gru.mlmodel")),
    })
    .expect("coreml gru parses");

    assert_eq!(model.tensors.len(), 9);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let node = &normalized["graphs"][0]["nodes"][0];
    assert_eq!(node["operator"]["name"], "gru");
    assert_eq!(node["inputs"].as_array().unwrap().len(), 11);
    assert_eq!(node["outputs"], json!(["y", "h_out"]));
    assert_eq!(
        node["attributes"],
        json!([
            { "name": "activations", "value": { "kind": "strings", "value": ["[object Object]", "[object Object]"] } },
            { "name": "inputVectorSize", "value": { "kind": "string", "value": "3" } },
            { "name": "outputVectorSize", "value": { "kind": "string", "value": "2" } },
            { "name": "hasBiasVectors", "value": { "kind": "bool", "value": true } }
        ])
    );
}

fn pipeline_fixture() -> Vec<u8> {
    let extractor = model(
        description(&[feature("x", None)], &[feature("mid", None)]),
        609,
        array_feature_extractor(&[0]),
    );
    let encoder = model(
        description(&[feature("mid", None)], &[feature("mid", None)]),
        600,
        one_hot_encoder(),
    );
    let vectorizer = model(
        description(&[feature("mid", None)], &[feature("y", None)]),
        602,
        feature_vectorizer(1),
    );

    let mut pipeline = Vec::new();
    message(&mut pipeline, 1, &extractor);
    message(&mut pipeline, 1, &encoder);
    message(&mut pipeline, 1, &vectorizer);

    model(
        description(
            &[feature("x", Some(int64_type()))],
            &[feature("y", Some(multi_array_type(65_568, &[2])))],
        ),
        202,
        pipeline,
    )
}

fn model(description: Vec<u8>, kind_field: u32, kind: Vec<u8>) -> Vec<u8> {
    let mut data = Vec::new();
    varint_field(&mut data, 1, 1);
    message(&mut data, 2, &description);
    message(&mut data, kind_field, &kind);
    data
}

fn description(inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
    let mut data = Vec::new();
    for input in inputs {
        message(&mut data, 1, input);
    }
    for output in outputs {
        message(&mut data, 10, output);
    }
    data
}

fn description_with_predictions(
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
    predicted_feature_name: &str,
    predicted_probabilities_name: &str,
) -> Vec<u8> {
    let mut data = description(inputs, outputs);
    string_field(&mut data, 11, predicted_feature_name);
    string_field(&mut data, 12, predicted_probabilities_name);
    data
}

fn feature(name: &str, feature_type: Option<Vec<u8>>) -> Vec<u8> {
    let mut data = Vec::new();
    string_field(&mut data, 1, name);
    if let Some(feature_type) = feature_type {
        message(&mut data, 3, &feature_type);
    }
    data
}

fn int64_type() -> Vec<u8> {
    let mut data = Vec::new();
    message(&mut data, 1, &[]);
    data
}

fn double_type() -> Vec<u8> {
    let mut data = Vec::new();
    message(&mut data, 2, &[]);
    data
}

fn dictionary_type() -> Vec<u8> {
    let mut dictionary = Vec::new();
    message(&mut dictionary, 1, &[]);

    let mut data = Vec::new();
    message(&mut data, 6, &dictionary);
    data
}

fn multi_array_type(data_type: u64, shape: &[u64]) -> Vec<u8> {
    let mut array = Vec::new();
    for dimension in shape {
        varint_field(&mut array, 1, *dimension);
    }
    varint_field(&mut array, 2, data_type);

    let mut data = Vec::new();
    message(&mut data, 5, &array);
    data
}

fn array_feature_extractor(indices: &[u64]) -> Vec<u8> {
    let mut packed = Vec::new();
    for index in indices {
        write_varint(&mut packed, *index);
    }

    let mut data = Vec::new();
    message(&mut data, 1, &packed);
    data
}

fn one_hot_encoder() -> Vec<u8> {
    let mut data = Vec::new();
    message(&mut data, 2, &[]);
    varint_field(&mut data, 10, 1);
    data
}

fn feature_vectorizer(input_count: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..input_count {
        message(&mut data, 1, &[]);
    }
    data
}

fn tree_ensemble_classifier() -> Vec<u8> {
    let mut classifier = Vec::new();
    message(&mut classifier, 1, &tree_ensemble_parameters());
    message(&mut classifier, 101, &int64_vector(&[0, 1]));
    classifier
}

fn tree_ensemble_parameters() -> Vec<u8> {
    let mut data = Vec::new();
    message(&mut data, 1, &[]);
    varint_field(&mut data, 2, 1);
    fixed64_field(&mut data, 3, 0.0);
    data
}

fn support_vector_regressor() -> Vec<u8> {
    let mut kernel = Vec::new();
    message(&mut kernel, 1, &[]);

    let mut data = Vec::new();
    message(&mut data, 1, &kernel);
    message(&mut data, 3, &[]);
    message(&mut data, 4, &[]);
    fixed64_field(&mut data, 5, -20.85139);
    data
}

fn neural_network_embedding() -> Vec<u8> {
    let mut layer = Vec::new();
    string_field(&mut layer, 1, "embedding_1");
    string_field(&mut layer, 2, "data");
    string_field(&mut layer, 3, "target");
    message(&mut layer, 150, &embedding_layer());

    let mut data = Vec::new();
    message(&mut data, 1, &layer);
    data
}

fn neural_network(layers: Vec<Vec<u8>>) -> Vec<u8> {
    let mut data = Vec::new();
    for layer in layers {
        message(&mut data, 1, &layer);
    }
    data
}

fn neural_layer(
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    kind_field: u32,
    kind: Vec<u8>,
) -> Vec<u8> {
    let mut data = Vec::new();
    string_field(&mut data, 1, name);
    for input in inputs {
        string_field(&mut data, 2, input);
    }
    for output in outputs {
        string_field(&mut data, 3, output);
    }
    message(&mut data, kind_field, &kind);
    data
}

fn embedding_layer() -> Vec<u8> {
    let mut data = Vec::new();
    varint_field(&mut data, 1, 2);
    varint_field(&mut data, 2, 3);
    message(
        &mut data,
        20,
        &weight_params_float(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    data
}

fn load_constant_layer(shape: &[u64], values: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    for dimension in shape {
        varint_field(&mut data, 1, *dimension);
    }
    message(&mut data, 2, &weight_params_float(values));
    data
}

fn multiply_layer(alpha: f32) -> Vec<u8> {
    let mut data = Vec::new();
    fixed32_field(&mut data, 1, alpha);
    data
}

fn reshape_layer(target_shape: &[u64]) -> Vec<u8> {
    let mut data = Vec::new();
    for dimension in target_shape {
        varint_field(&mut data, 1, *dimension);
    }
    data
}

fn slice_layer(start_index: u64, end_index: u64, stride: u64, axis: u64) -> Vec<u8> {
    let mut data = Vec::new();
    varint_field(&mut data, 1, start_index);
    varint_field(&mut data, 2, end_index);
    varint_field(&mut data, 3, stride);
    varint_field(&mut data, 4, axis);
    data
}

fn gru_layer() -> Vec<u8> {
    let mut data = Vec::new();
    varint_field(&mut data, 1, 3);
    varint_field(&mut data, 2, 2);
    message(&mut data, 10, &[]);
    message(&mut data, 10, &[]);
    varint_field(&mut data, 20, 1);
    for field in [30, 31, 32] {
        message(&mut data, field, &weight_params_float(&[0.0; 6]));
    }
    for field in [50, 51, 52] {
        message(&mut data, field, &weight_params_float(&[0.0; 4]));
    }
    for field in [70, 71, 72] {
        message(&mut data, field, &weight_params_float(&[0.0; 2]));
    }
    data
}

fn weight_params_float(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    for value in values {
        fixed32_field(&mut data, 1, *value);
    }
    data
}

fn int64_vector(values: &[u64]) -> Vec<u8> {
    let mut data = Vec::new();
    for value in values {
        varint_field(&mut data, 1, *value);
    }
    data
}

fn string_field(data: &mut Vec<u8>, number: u32, value: &str) {
    message(data, number, value.as_bytes());
}

fn message(data: &mut Vec<u8>, number: u32, value: &[u8]) {
    write_varint(data, u64::from(number) << 3 | 2);
    write_varint(data, value.len() as u64);
    data.extend_from_slice(value);
}

fn varint_field(data: &mut Vec<u8>, number: u32, value: u64) {
    write_varint(data, u64::from(number) << 3);
    write_varint(data, value);
}

fn fixed64_field(data: &mut Vec<u8>, number: u32, value: f64) {
    write_varint(data, u64::from(number) << 3 | 1);
    data.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn fixed32_field(data: &mut Vec<u8>, number: u32, value: f32) {
    write_varint(data, u64::from(number) << 3 | 5);
    data.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn write_varint(data: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        data.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    data.push(value as u8);
}
