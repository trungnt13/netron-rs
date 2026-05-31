use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_npy_array_into_ndarray_node() {
    let data = npy_fixture("<f4", &[2, 3], 24);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("tensor.npy")),
    })
    .expect("npy fixture parses");

    assert_eq!(model.format.name, "NumPy Array");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.graphs[0].nodes.len(), 1);
    assert_eq!(model.tensors.len(), 1);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "numpy.ndarray"
    );
    assert_eq!(normalized["graphs"][0]["nodes"][0]["inputs"], json!([""]));
    assert_eq!(normalized["graphs"][0]["values"][0]["name"], "");
    assert_eq!(
        normalized["graphs"][0]["values"][0]["type"]["element_type"],
        "float32"
    );
    assert_eq!(
        normalized["graphs"][0]["values"][0]["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }, { "kind": "known", "value": 3 }])
    );
}

#[test]
fn parses_npz_archive_in_zip_order() {
    let first = npy_fixture("<i4", &[1], 4);
    let second = npy_fixture("<f8", &[2], 16);
    let data = zip_fixture(&[("arr_10.npy", &first), ("arr_2.npy", &second)]);

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("weights.npz")),
    })
    .expect("npz fixture parses");

    assert_eq!(model.format.name, "NumPy Archive");
    assert_eq!(model.tensors.len(), 2);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "Object"
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["arr_10", "arr_2"])
    );
    assert_eq!(
        normalized["graphs"][0]["values"][0]["type"]["element_type"],
        "int32"
    );
    assert_eq!(
        normalized["graphs"][0]["values"][1]["type"]["element_type"],
        "float64"
    );
}

#[test]
fn recognizes_numpy_string_types() {
    let data = npy_fixture("<U6", &[2, 2], 96);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("strings.npy")),
    })
    .expect("string npy fixture parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["values"][0]["type"]["element_type"],
        "string"
    );
}

#[test]
fn rejects_truncated_npy_header() {
    let error = parse(ModelInput {
        data: b"\x93NUMPY\x01",
        path: Some(std::path::Path::new("bad.npy")),
    })
    .expect_err("truncated header is rejected");

    assert!(error.to_string().contains("header length is truncated"));
}

fn npy_fixture(descriptor: &str, shape: &[i64], byte_len: usize) -> Vec<u8> {
    let shape_text = if shape.is_empty() {
        "()".to_owned()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descriptor, shape_text
    );
    while (10 + header.len() + 1) % 16 != 0 {
        header.push(' ');
    }
    header.push('\n');

    let mut data = Vec::new();
    data.extend_from_slice(b"\x93NUMPY");
    data.extend_from_slice(&[1, 0]);
    data.extend_from_slice(&(header.len() as u16).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.resize(data.len() + byte_len, 0);
    data
}

fn zip_fixture(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut central = Vec::new();
    for (name, content) in entries {
        let local_offset = data.len() as u32;
        data.extend_from_slice(b"PK\x03\x04");
        data.extend_from_slice(&20_u16.to_le_bytes());
        data.extend_from_slice(&0_u16.to_le_bytes());
        data.extend_from_slice(&0_u16.to_le_bytes());
        data.extend_from_slice(&0_u16.to_le_bytes());
        data.extend_from_slice(&0_u16.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());
        data.extend_from_slice(&(content.len() as u32).to_le_bytes());
        data.extend_from_slice(&(content.len() as u32).to_le_bytes());
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(&0_u16.to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(content);

        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&20_u16.to_le_bytes());
        central.extend_from_slice(&20_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u32.to_le_bytes());
        central.extend_from_slice(&(content.len() as u32).to_le_bytes());
        central.extend_from_slice(&(content.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u16.to_le_bytes());
        central.extend_from_slice(&0_u32.to_le_bytes());
        central.extend_from_slice(&local_offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }

    let central_offset = data.len() as u32;
    data.extend_from_slice(&central);
    data.extend_from_slice(b"PK\x05\x06");
    data.extend_from_slice(&0_u16.to_le_bytes());
    data.extend_from_slice(&0_u16.to_le_bytes());
    data.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    data.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    data.extend_from_slice(&(central.len() as u32).to_le_bytes());
    data.extend_from_slice(&central_offset.to_le_bytes());
    data.extend_from_slice(&0_u16.to_le_bytes());
    data
}
