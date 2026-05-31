use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_sentencepiece_model_metadata_graph() {
    let data = sentencepiece_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("tokenizer.model")),
    })
    .expect("sentencepiece fixture parses");

    assert_eq!(model.format.name, "SentencePiece");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.graphs[0].nodes.len(), 4);
    assert!(model.tensors.is_empty());

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["name"], "pieces");
    assert_eq!(graph["nodes"][0]["operator"]["name"], "SentencePiece[]");
    assert_eq!(
        graph["nodes"][0]["attributes"][0]["value"],
        json!({ "kind": "strings", "value": ["[object Object]", "[object Object]"] })
    );
    assert_eq!(graph["nodes"][1]["name"], "trainer_spec");
    assert_eq!(graph["nodes"][1]["operator"]["name"], "TrainerSpec");
    assert_eq!(graph["nodes"][2]["name"], "normalizer_spec");
    assert_eq!(graph["nodes"][3]["name"], "self_test_data");
}

#[test]
fn rejects_non_sentencepiece_protobuf_shape() {
    let error = parse(ModelInput {
        data: b"not a sentencepiece model",
        path: Some(std::path::Path::new("bad.model")),
    })
    .expect_err("non sentencepiece model is unsupported");

    assert!(error.to_string().contains("unsupported model format"));
}

fn sentencepiece_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    message(&mut data, 1, &piece("a", -1.0, 1));
    message(&mut data, 1, &piece("b", -2.0, 1));
    message(&mut data, 2, &trainer_spec());
    message(&mut data, 3, &normalizer_spec());
    message(&mut data, 4, &self_test_data());
    data
}

fn piece(piece: &str, score: f32, value_type: u64) -> Vec<u8> {
    let mut data = Vec::new();
    string(&mut data, 1, piece);
    fixed32(&mut data, 2, score);
    varint_field(&mut data, 3, value_type);
    data
}

fn trainer_spec() -> Vec<u8> {
    let mut data = Vec::new();
    string(&mut data, 1, "input.txt");
    string(&mut data, 2, "tokenizer");
    varint_field(&mut data, 3, 2);
    varint_field(&mut data, 4, 32000);
    fixed32(&mut data, 10, 1.0);
    string(&mut data, 31, "foo");
    data
}

fn normalizer_spec() -> Vec<u8> {
    let mut data = Vec::new();
    string(&mut data, 1, "nmt_nfkc");
    bytes(&mut data, 2, &[0, 180, 2, 0]);
    varint_field(&mut data, 3, 1);
    data
}

fn self_test_data() -> Vec<u8> {
    let mut sample = Vec::new();
    string(&mut sample, 1, "input");
    string(&mut sample, 2, "expected");
    let mut data = Vec::new();
    message(&mut data, 1, &sample);
    data
}

fn message(data: &mut Vec<u8>, field: u64, payload: &[u8]) {
    key(data, field, 2);
    varint(data, payload.len() as u64);
    data.extend_from_slice(payload);
}

fn string(data: &mut Vec<u8>, field: u64, value: &str) {
    bytes(data, field, value.as_bytes());
}

fn bytes(data: &mut Vec<u8>, field: u64, value: &[u8]) {
    key(data, field, 2);
    varint(data, value.len() as u64);
    data.extend_from_slice(value);
}

fn fixed32(data: &mut Vec<u8>, field: u64, value: f32) {
    key(data, field, 5);
    data.extend_from_slice(&value.to_le_bytes());
}

fn varint_field(data: &mut Vec<u8>, field: u64, value: u64) {
    key(data, field, 0);
    varint(data, value);
}

fn key(data: &mut Vec<u8>, field: u64, wire_type: u64) {
    varint(data, (field << 3) | wire_type);
}

fn varint(data: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        data.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    data.push(value as u8);
}
