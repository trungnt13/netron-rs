use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use netron_rs_core::TensorStorage;
use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_safetensors_header_into_module_graph() {
    let data = safetensors_fixture(&[
        ("layer.weight", "F32", &[2, 3], 24),
        ("layer.bias", "F16", &[3], 6),
        ("other.flag", "BOOL", &[1], 1),
    ]);

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("model.safetensors")),
    })
    .expect("safetensors fixture parses");

    assert_eq!(model.format.name, "Safetensors");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.graphs[0].nodes.len(), 2);
    assert_eq!(model.tensors.len(), 3);
    assert!(matches!(
        model.tensors[0].storage,
        TensorStorage::InlineBytes { byte_len: 24 }
    ));

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(normalized["graphs"][0]["nodes"][0]["name"], "layer");
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "Module"
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["layer.weight", "layer.bias"])
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][1]["inputs"],
        json!(["other.flag"])
    );
    let values = normalized["graphs"][0]["values"].as_array().unwrap();
    let value = |name: &str| {
        values
            .iter()
            .find(|value| value["name"] == name)
            .expect("value exists")
    };
    assert_eq!(value("layer.weight")["type"]["element_type"], "float32");
    assert_eq!(
        value("layer.weight")["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }, { "kind": "known", "value": 3 }])
    );
    assert_eq!(value("other.flag")["type"]["element_type"], "boolean");
}

#[test]
fn rejects_safetensors_offsets_outside_file() {
    let mut data = Vec::new();
    let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#;
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header);
    data.extend_from_slice(&[0; 4]);

    let error = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("bad.safetensors")),
    })
    .expect_err("invalid offsets are rejected");

    assert!(error.to_string().contains("outside file"));
}

#[test]
fn parses_safetensors_index_with_local_shard() {
    let root = std::env::temp_dir().join(format!(
        "netron-rs-safetensors-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let shard_path = root.join("model.safetensors");
    let index_path = root.join("model.safetensors.index.json");
    fs::write(
        &shard_path,
        safetensors_fixture(&[
            ("layer.weight", "F32", &[2], 8),
            ("layer.unused", "I32", &[1], 4),
        ]),
    )
    .unwrap();
    fs::write(
        &index_path,
        br#"{"weight_map":{"layer.weight":"model.safetensors"}}"#,
    )
    .unwrap();
    let index = fs::read(&index_path).unwrap();

    let model = parse(ModelInput {
        data: &index,
        path: Some(index_path.as_path()),
    })
    .expect("safetensors index parses");

    fs::remove_dir_all(&root).unwrap();

    assert_eq!(model.format.name, "Safetensors");
    assert_eq!(model.tensors.len(), 1);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["layer.weight"])
    );
}

#[test]
fn parses_safetensors_index_with_per_shard_tensor_mapping() {
    let root = temp_root("netron-rs-safetensors-shards");
    fs::create_dir(&root).unwrap();
    let first_path = root.join("first.safetensors");
    let second_path = root.join("second.safetensors");
    let index_path = root.join("model.safetensors.index.json");
    fs::write(&first_path, safetensors_fixture(&[("x", "F32", &[1], 4)])).unwrap();
    fs::write(
        &second_path,
        safetensors_fixture(&[("x", "F32", &[2], 8), ("y", "I32", &[1], 4)]),
    )
    .unwrap();
    fs::write(
        &index_path,
        br#"{"weight_map":{"x":"first.safetensors","y":"second.safetensors"}}"#,
    )
    .unwrap();
    let index = fs::read(&index_path).unwrap();

    let model = parse(ModelInput {
        data: &index,
        path: Some(index_path.as_path()),
    })
    .expect("safetensors index parses");

    fs::remove_dir_all(&root).unwrap();

    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["x", "y"])
    );
    assert_eq!(
        normalized["graphs"][0]["values"][0]["type"]["shape"],
        json!([{ "kind": "known", "value": 1 }])
    );
}

#[test]
fn rejects_safetensors_index_when_mapped_tensor_is_missing() {
    let root = temp_root("netron-rs-safetensors-missing");
    fs::create_dir(&root).unwrap();
    let shard_path = root.join("model.safetensors");
    let index_path = root.join("model.safetensors.index.json");
    fs::write(
        &shard_path,
        safetensors_fixture(&[("other", "F32", &[1], 4)]),
    )
    .unwrap();
    fs::write(&index_path, br#"{"weight_map":{"x":"model.safetensors"}}"#).unwrap();
    let index = fs::read(&index_path).unwrap();

    let error = parse(ModelInput {
        data: &index,
        path: Some(index_path.as_path()),
    })
    .expect_err("missing mapped tensor is rejected");

    fs::remove_dir_all(&root).unwrap();

    assert!(error.to_string().contains("does not contain it"));
}

#[test]
fn parses_zip_wrapped_safetensors_index_with_shards() {
    let first = safetensors_fixture(&[("x", "F32", &[1], 4)]);
    let second = safetensors_fixture(&[("y", "I32", &[1], 4)]);
    let archive = zip_store_entries(&[
        (
            "checkpoint/model.safetensors.index.json",
            br#"{"weight_map":{"x":"first.safetensors","y":"second.safetensors"}}"#,
        ),
        ("checkpoint/first.safetensors", &first),
        ("checkpoint/second.safetensors", &second),
    ]);

    let model = parse(ModelInput {
        data: &archive,
        path: Some(std::path::Path::new("checkpoint.zip")),
    })
    .expect("zip-wrapped safetensors index parses");

    assert_eq!(model.format.name, "Safetensors");
    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["x", "y"])
    );
}

fn temp_root(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn safetensors_fixture(entries: &[(&str, &str, &[i64], usize)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0;
    for (index, (name, dtype, shape, byte_len)) in entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let end = offset + byte_len;
        header.push_str(&serde_json::to_string(name).unwrap());
        header.push_str(":{\"dtype\":\"");
        header.push_str(dtype);
        header.push_str("\",\"shape\":");
        header.push_str(&serde_json::to_string(shape).unwrap());
        header.push_str(",\"data_offsets\":[");
        header.push_str(&offset.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");
        offset = end;
    }
    header.push('}');

    let mut data = Vec::new();
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.resize(data.len() + offset, 0);
    data
}

fn zip_store_entries(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut central_records = Vec::new();
    for (name, data) in entries {
        let name = name.as_bytes();
        let local_offset = output.len() as u32;
        le_u32(&mut output, 0x0403_4b50);
        le_u16(&mut output, 20);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u32(&mut output, 0);
        le_u32(&mut output, data.len() as u32);
        le_u32(&mut output, data.len() as u32);
        le_u16(&mut output, name.len() as u16);
        le_u16(&mut output, 0);
        output.extend_from_slice(name);
        output.extend_from_slice(data);
        central_records.push((name.to_vec(), data.len() as u32, local_offset));
    }

    let central_offset = output.len() as u32;
    for (name, size, local_offset) in &central_records {
        le_u32(&mut output, 0x0201_4b50);
        le_u16(&mut output, 20);
        le_u16(&mut output, 20);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u32(&mut output, 0);
        le_u32(&mut output, *size);
        le_u32(&mut output, *size);
        le_u16(&mut output, name.len() as u16);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u16(&mut output, 0);
        le_u32(&mut output, 0);
        le_u32(&mut output, *local_offset);
        output.extend_from_slice(name);
    }

    let central_size = output.len() as u32 - central_offset;
    le_u32(&mut output, 0x0605_4b50);
    le_u16(&mut output, 0);
    le_u16(&mut output, 0);
    le_u16(&mut output, central_records.len() as u16);
    le_u16(&mut output, central_records.len() as u16);
    le_u32(&mut output, central_size);
    le_u32(&mut output, central_offset);
    le_u16(&mut output, 0);
    output
}

fn le_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn le_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}
