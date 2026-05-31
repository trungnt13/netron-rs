use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_bert_style_gguf_into_structured_graph() {
    let data = bert_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("bert.gguf")),
    })
    .expect("gguf fixture parses");

    assert_eq!(model.format.name, "GGUF");
    assert_eq!(model.format.version.as_deref(), Some("3"));
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.graphs[0].nodes.len(), 13);
    assert_eq!(model.tensors.len(), 21);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(normalized["metadata"]["properties"]["file_type"], "1");
    assert_eq!(graph["name"], "bert");
    assert_eq!(graph["metadata"]["block_count"], "1");
    assert_eq!(graph["nodes"][0]["operator"]["name"], "tokenizer");
    assert_eq!(graph["nodes"][0]["attributes"][0]["name"], "model");
    assert_eq!(
        graph["nodes"][0]["attributes"][0]["value"]["kind"],
        "string"
    );
    assert_eq!(graph["nodes"][0]["attributes"][0]["value"]["value"], "bert");
    assert_eq!(graph["nodes"][1]["name"], "token_embd");
    assert_eq!(graph["nodes"][1]["operator"]["name"], "EMBEDDING");
    assert_eq!(
        graph["nodes"][5]["inputs"],
        json!([
            "v4",
            "blk.0.attn_q.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_k.weight",
            "blk.0.attn_k.bias",
            "blk.0.attn_v.weight",
            "blk.0.attn_v.bias",
            "blk.0.attn_output.weight",
            "blk.0.attn_output.bias"
        ])
    );
    assert_eq!(graph["nodes"][12]["operator"]["name"], "tokenizer");
    assert_eq!(graph["nodes"][12]["inputs"], json!(["v11"]));
}

fn bert_fixture() -> Vec<u8> {
    let tensors = [
        ("token_embd.weight", 1, &[4, 16][..]),
        ("token_types.weight", 0, &[4, 2][..]),
        ("token_embd_norm.weight", 0, &[4][..]),
        ("token_embd_norm.bias", 0, &[4][..]),
        ("position_embd.weight", 1, &[4, 8][..]),
        ("blk.0.attn_q.weight", 1, &[4, 4][..]),
        ("blk.0.attn_q.bias", 0, &[4][..]),
        ("blk.0.attn_k.weight", 1, &[4, 4][..]),
        ("blk.0.attn_k.bias", 0, &[4][..]),
        ("blk.0.attn_v.weight", 1, &[4, 4][..]),
        ("blk.0.attn_v.bias", 0, &[4][..]),
        ("blk.0.attn_output.weight", 1, &[4, 4][..]),
        ("blk.0.attn_output.bias", 0, &[4][..]),
        ("blk.0.attn_output_norm.weight", 0, &[4][..]),
        ("blk.0.attn_output_norm.bias", 0, &[4][..]),
        ("blk.0.ffn_up.weight", 1, &[4, 8][..]),
        ("blk.0.ffn_up.bias", 0, &[8][..]),
        ("blk.0.ffn_down.weight", 1, &[8, 4][..]),
        ("blk.0.ffn_down.bias", 0, &[4][..]),
        ("blk.0.layer_output_norm.weight", 0, &[4][..]),
        ("blk.0.layer_output_norm.bias", 0, &[4][..]),
    ];

    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    u32(&mut data, 3);
    u64(&mut data, tensors.len() as u64);
    u64(&mut data, 4);
    metadata_string(&mut data, "general.architecture", "bert");
    metadata_u32(&mut data, "general.file_type", 1);
    metadata_u32(&mut data, "bert.block_count", 1);
    metadata_string(&mut data, "tokenizer.ggml.model", "bert");

    let mut offset = 0_u64;
    for (name, tensor_type, shape) in tensors {
        string(&mut data, name);
        u32(&mut data, shape.len() as u32);
        for dim in shape {
            u64(&mut data, *dim);
        }
        u32(&mut data, tensor_type);
        u64(&mut data, offset);
        offset += byte_len(tensor_type, shape);
    }
    while data.len() % 32 != 0 {
        data.push(0);
    }
    data.resize(data.len() + offset as usize, 0);
    data
}

fn byte_len(tensor_type: u32, shape: &[u64]) -> u64 {
    let elements = shape.iter().product::<u64>();
    match tensor_type {
        0 => elements * 4,
        1 => elements * 2,
        _ => unreachable!("fixture only uses f32/f16"),
    }
}

fn metadata_string(data: &mut Vec<u8>, name: &str, value: &str) {
    string(data, name);
    u32(data, 8);
    string(data, value);
}

fn metadata_u32(data: &mut Vec<u8>, name: &str, value: u32) {
    string(data, name);
    u32(data, 4);
    u32(data, value);
}

fn string(data: &mut Vec<u8>, value: &str) {
    u64(data, value.len() as u64);
    data.extend_from_slice(value.as_bytes());
}

fn u32(data: &mut Vec<u8>, value: u32) {
    data.extend_from_slice(&value.to_le_bytes());
}

fn u64(data: &mut Vec<u8>, value: u64) {
    data.extend_from_slice(&value.to_le_bytes());
}
