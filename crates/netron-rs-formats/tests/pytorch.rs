use netron_rs_formats::{ModelInput, ToNormalizedJson, parse};
use serde_json::json;

#[test]
fn parses_legacy_pytorch_bool_tensor_pickle() {
    let data = legacy_bool_tensor_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("boolean.pkl.pth")),
    })
    .expect("legacy pytorch pickle parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("0.1.10"));
    assert_eq!(model.tensors.len(), 1);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.object");
    assert_eq!(graph["nodes"][0]["inputs"], json!([""]));

    let value = &graph["values"][0];
    assert_eq!(value["name"], "");
    assert_eq!(value["initializer"], 0);
    assert_eq!(value["type"]["element_type"], "boolean");
    assert_eq!(
        value["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }])
    );
    assert_eq!(
        normalized["tensors"][0]["storage"],
        json!({ "kind": "inline_bytes", "byte_len": 2 })
    );
}

#[test]
fn parses_legacy_pytorch_dict_pickle() {
    let data = legacy_dict_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("ENet.pth")),
    })
    .expect("legacy pytorch dict pickle parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("0.1.10"));
    assert_eq!(model.tensors.len(), 0);

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.dict");
    assert_eq!(graph["nodes"][0]["inputs"], json!([]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_legacy_vgg19x_pickle_as_opaque_dict() {
    let data = legacy_vgg19x_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("ckpt.t7")),
    })
    .expect("legacy VGG19X pickle parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.dict");
    assert_eq!(graph["nodes"][0]["inputs"], json!([]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_legacy_pytorch_plain_dict_state_dict_pickle() {
    let data = legacy_plain_dict_state_dict_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mask_r_cnn.pth")),
    })
    .expect("legacy pytorch plain dict state dict parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("0.1.10"));
    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["name"], "conv");
    assert_eq!(graph["nodes"][0]["operator"]["name"], "Weights");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["weight"]));
    assert_eq!(graph["nodes"][1]["name"], "bn");
    assert_eq!(graph["nodes"][1]["inputs"], json!(["running_mean"]));
}

#[test]
fn parses_pytorch_data_pickle_class_node() {
    let data = b"\x80\x02c__torch__.example.___torch_mangle_1\nTinyModule\nq\x00)\x81.".to_vec();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("data.pkl")),
    })
    .expect("pytorch data pickle parses");

    assert_eq!(model.format.name, "PyTorch Pickle");
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "__torch__.example.___torch_mangle_1.TinyModule"
    );
}

#[test]
fn parses_pytorch_zip_archive_as_empty_module_graph() {
    let data = pytorch_zip_archive_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("alexnet.fx.pth")),
    })
    .expect("pytorch zip archive parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(normalized["graphs"][0]["nodes"], json!([]));
    assert_eq!(normalized["graphs"][0]["values"], json!([]));
}

#[test]
fn parses_pytorch_zip_archive_with_ckpt_extension() {
    let data = pytorch_zip_archive_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("model.ckpt")),
    })
    .expect("pytorch ckpt zip archive parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
}

#[test]
fn parses_single_file_zip_wrapper_around_pytorch_archive() {
    let inner = pytorch_zip_archive_fixture();
    let mut data = Vec::new();
    zip_stored(&mut data, "alexnet.fx.pth", &inner);

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("alexnet.fx.pth.zip")),
    })
    .expect("nested pytorch zip wrapper parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
}

#[test]
fn parses_single_file_zip_wrapper_around_legacy_pickle() {
    let inner = legacy_dict_fixture();
    let mut data = Vec::new();
    zip_stored(&mut data, "alexnet.pkl.pth", &inner);

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("alexnet.pkl.pth.zip")),
    })
    .expect("nested legacy pickle zip wrapper parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("0.1.10"));
}

#[test]
fn parses_pytorch_exported_program_archive() {
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "serialized_exported_program.json",
        exported_program_json_fixture().as_bytes(),
    );
    zip_stored(&mut data, "version", b"7.3");

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("constant_input.pt2")),
    })
    .expect("exported program parses");

    assert_eq!(model.format.name, "PyTorch Export");
    assert_eq!(model.format.version.as_deref(), Some("7.3"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["name"], "model");
    assert_eq!(graph["inputs"], json!(["x", "y"]));
    assert_eq!(graph["outputs"], json!(["div_1"]));
    assert_eq!(graph["nodes"][0]["operator"]["name"], "div");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["x", "y"]));
    assert_eq!(graph["nodes"][0]["outputs"], json!(["div_1"]));
    assert_eq!(graph["values"][0]["type"]["element_type"], "float32");
}

#[test]
fn parses_legacy_pytorch_exported_program_json_variants() {
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "serialized_exported_program.json",
        legacy_exported_program_json_fixture().as_bytes(),
    );

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("gpt2_v3.1.pt2")),
    })
    .expect("legacy exported program JSON parses");

    assert_eq!(model.format.name, "PyTorch Export");
    assert_eq!(model.format.version.as_deref(), Some("3.1"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["x"]));
    assert_eq!(graph["outputs"], json!(["add"]));
    assert_eq!(graph["nodes"][0]["name"], "sym_size_int");
    assert_eq!(graph["nodes"][0]["outputs"], json!(["sym_size_int"]));
    assert_eq!(
        graph["nodes"][1]["inputs"],
        json!(["x", "sym_size_int", null])
    );
    assert_eq!(graph["nodes"][2]["inputs"], json!(["view"]));
}

#[test]
fn parses_pytorch_exported_program_archive_initializers() {
    let mut data = Vec::new();
    zip_stored(&mut data, "draft/archive_format", b"pt2");
    zip_stored(
        &mut data,
        "draft/models/model.json",
        exported_program_with_weight_json_fixture().as_bytes(),
    );
    zip_stored(
        &mut data,
        "draft/data/weights/model_weights_config.json",
        br#"{"config":{"linear.weight":{"path_name":"weight_0","is_param":true,"use_pickle":false,"tensor_meta":{"dtype":7,"sizes":[{"as_int":2}],"requires_grad":true}}}}"#,
    );
    zip_stored(&mut data, "draft/data/weights/weight_0", &[0; 8]);

    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("draft_export.pt2")),
    })
    .expect("exported program with initializers parses");

    assert_eq!(model.tensors.len(), 1);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["x", "threshold"]));
    assert_eq!(graph["nodes"][0]["inputs"], json!(["x", "linear.weight"]));
    assert_eq!(graph["nodes"][1]["name"], "item_default");
    assert_eq!(graph["nodes"][1]["outputs"], json!(["item_default"]));

    let weight = graph["values"]
        .as_array()
        .unwrap()
        .iter()
        .find(|value| value["name"] == "linear.weight")
        .unwrap();
    assert_eq!(weight["initializer"], 0);
    assert_eq!(weight["type"]["element_type"], "float32");

    let threshold = graph["values"]
        .as_array()
        .unwrap()
        .iter()
        .find(|value| value["name"] == "threshold")
        .unwrap();
    assert_eq!(threshold["type"]["element_type"], "int64");
}

#[test]
fn parses_pytorch_zip_data_pickle_module_graph() {
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "archive/data.pkl",
        &densenet_data_pickle_fixture(),
    );
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("alexnet.zip.pth")),
    })
    .expect("pytorch zip data pickle parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "torchvision.models.densenet.DenseNet"
    );
    assert_eq!(
        graph["nodes"][0]["inputs"],
        json!(["features", "classifier"])
    );
}

#[test]
fn parses_pytorch_zip_dict_checkpoint_graph() {
    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", b"\x80\x02}q\x00.");
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("best_mask.pth")),
    })
    .expect("pytorch zip dict checkpoint parses");

    assert_eq!(model.format.name, "PyTorch");
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.dict");
    assert_eq!(graph["nodes"][0]["inputs"], json!([]));
}

#[test]
fn parses_pytorch_zip_dict_checkpoint_null_inputs() {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02}q\x00(X\x0f\x00\x00\x00network_weightsq\x01}q\x02u.");
    let mut data = Vec::new();
    zip_stored(&mut data, "checkpoint_best/data.pkl", &pickle);
    zip_stored(&mut data, "checkpoint_best/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("checkpoint_best.pth")),
    })
    .expect("pytorch zip dict checkpoint parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.dict");
    assert_eq!(graph["nodes"][0]["inputs"], json!([null, null]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_pytorch_zip_deepspeed_dict_checkpoint_inputs() {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02}q\x00(");
    binunicode(&mut pickle, "module");
    pickle.extend_from_slice(b"}q\x01");
    binunicode(&mut pickle, "optimizer");
    pickle.extend_from_slice(b"}q\x02");
    binunicode(&mut pickle, "param_shapes");
    pickle.extend_from_slice(b"]q\x03u.");
    let mut data = Vec::new();
    zip_stored(&mut data, "mp_rank_00_model_states/data.pkl", &pickle);
    zip_stored(&mut data, "mp_rank_00_model_states/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("InternVideo2-stage2_1b-224p-f4.pt")),
    })
    .expect("deepspeed checkpoint dict parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.dict");
    assert_eq!(graph["nodes"][0]["inputs"], json!([null, "", ""]));
    assert_eq!(graph["values"].as_array().unwrap().len(), 2);
}

#[test]
fn parses_pytorch_zip_list_checkpoint_graph() {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02]q\x00(");
    binunicode(&mut pickle, "animal_fibre");
    binunicode(&mut pickle, "metal_thread");
    pickle.extend_from_slice(b"e.");
    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", &pickle);
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("labels.pth")),
    })
    .expect("pytorch zip list checkpoint parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.list");
    assert_eq!(graph["nodes"][0]["inputs"], json!([]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_pytorch_zip_tensorrt_ordered_dict_checkpoint_graph() {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02");
    pickle_global(&mut pickle, "collections", "OrderedDict");
    pickle.extend_from_slice(b"q\x00)Rq\x01(");
    binunicode(&mut pickle, "engine");
    pickle_global(&mut pickle, "__builtin__", "bytearray");
    pickle.extend_from_slice(b"q\x02)Rq\x03u.");
    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", &pickle);
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("model_trt.pth")),
    })
    .expect("TensorRT ordered dict checkpoint parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "collections.OrderedDict"
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!([null, null]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_pytorch_zip_torch_compile_optimized_module_without_children() {
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "mlp.torch.compile/data.pkl",
        &torch_compile_optimized_module_pickle_fixture(),
    );
    zip_stored(&mut data, "mlp.torch.compile/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mlp.torch.compile.pt")),
    })
    .expect("torch compile optimized module parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "torch._dynamo.eval_frame.OptimizedModule"
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!([]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_pytorch_zip_mcunet_dict_checkpoint_inputs() {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02}q\x00(");
    binunicode(&mut pickle, "first_conv");
    pickle.extend_from_slice(b"}q\x01");
    binunicode(&mut pickle, "blocks");
    pickle.extend_from_slice(b"]q\x02u.");
    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", &pickle);
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mcunet-5fps.pkl")),
    })
    .expect("MCUNet dict checkpoint parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.dict");
    assert_eq!(graph["nodes"][0]["inputs"].as_array().unwrap().len(), 14);
    assert_eq!(graph["values"].as_array().unwrap().len(), 14);
}

#[test]
fn parses_pytorch_zip_ordered_state_dict_weights() {
    let data = pytorch_zip_state_dict_archive_fixture(false);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("from_numpy.pth")),
    })
    .expect("pytorch zip state dict parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 3);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["name"], "conv");
    assert_eq!(graph["nodes"][0]["operator"]["name"], "Weights");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["conv.weight"]));
    assert_eq!(graph["nodes"][1]["name"], "bn");
    assert_eq!(
        graph["nodes"][1]["inputs"],
        json!(["bn.weight", "bn.running_mean"])
    );
    assert_eq!(graph["values"][0]["name"], "conv.weight");
    assert_eq!(
        graph["values"][0]["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }, { "kind": "known", "value": 3 }])
    );
}

#[test]
fn parses_pytorch_zip_quantized_packed_state_dict_weights() {
    let data = pytorch_zip_state_dict_archive_fixture(true);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("final_89.pt")),
    })
    .expect("pytorch zip quantized state dict parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["name"], "fc");
    assert_eq!(graph["nodes"][0]["operator"]["name"], "Weights");
    assert_eq!(
        graph["nodes"][0]["inputs"],
        json!(["fc.scale", "fc.zero_point", "", null])
    );
    assert_eq!(graph["values"][2]["name"], "");
    assert!(graph["values"][2]["initializer"].is_null());
    assert_eq!(graph["values"][2]["type"]["element_type"], "quint8");
    assert_eq!(
        graph["values"][2]["type"]["shape"],
        json!([{ "kind": "known", "value": 2 }, { "kind": "known", "value": 3 }])
    );
}

#[test]
fn parses_pytorch_zip_cloudpickle_skeleton_class() {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02ccloudpickle.cloudpickle\n_make_skeleton_class\nq\x00(");
    pickle_global(&mut pickle, "__builtin__", "type");
    pickle.extend_from_slice(b"q\x01");
    binunicode(&mut pickle, "BiRNN");
    pickle.extend_from_slice(b"q\x02}");
    binunicode(&mut pickle, "__module__");
    binunicode(&mut pickle, "__main__");
    pickle.push(b'.');
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "tutorial_bidirectional_recurrent_neural_network/data.pkl",
        &pickle,
    );
    zip_stored(
        &mut data,
        "tutorial_bidirectional_recurrent_neural_network/version",
        b"3\n",
    );
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("cloudpickle.pth")),
    })
    .expect("pytorch zip cloudpickle skeleton parses");

    assert_eq!(model.format.name, "PyTorch");
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "__main__.BiRNN");
    assert_eq!(graph["nodes"][0]["inputs"], json!([]));
}

#[test]
fn parses_pytorch_package_archive_graphs() {
    let data = pytorch_package_archive_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("DCGAN2.pt")),
    })
    .expect("pytorch package archive parses");

    assert_eq!(model.format.name, "PyTorch Package");
    assert_eq!(model.format.version.as_deref(), Some("1.9"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(normalized["graphs"][0]["name"], "DCGAN/model.pkl");
    assert_eq!(normalized["graphs"][1]["name"], "DCGAN2/model2.pkl");
    for graph in normalized["graphs"].as_array().unwrap() {
        assert_eq!(graph["nodes"][0]["operator"]["name"], "models.DCGAN.DCGAN");
        assert_eq!(graph["nodes"][0]["inputs"], json!([]));
    }
}

#[test]
fn parses_pytorch_zip_complex_tensor() {
    let data = pytorch_zip_complex_tensor_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("complex_tensor.pt")),
    })
    .expect("pytorch zip complex tensor parses");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 1);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.object");
    assert_eq!(graph["nodes"][0]["inputs"], json!([""]));
    assert_eq!(
        graph["values"][0]["type"]["element_type"],
        "complex<float32>"
    );
    assert_eq!(
        graph["values"][0]["type"]["shape"],
        json!([{ "kind": "known", "value": 3 }])
    );
}

#[test]
fn parses_transducer_torchscript_archive_as_module_graph() {
    let data = transducer_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("cpu_jit.pt")),
    })
    .expect("transducer torchscript archive parses as module graph");

    assert_eq!(model.format.name, "PyTorch");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 0);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!([]));
    assert_eq!(graph["outputs"], json!([]));
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "__torch__.model.Transducer"
    );
    assert_eq!(
        graph["nodes"][0]["inputs"],
        json!([
            "encoder",
            "decoder",
            "joiner",
            "simple_am_proj",
            "simple_lm_proj",
            "ctc_output"
        ])
    );
    assert_eq!(graph["values"].as_array().unwrap().len(), 6);
}

#[test]
fn parses_yolo_mobile_torchscript_archive() {
    let data = yolo_mobile_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new(
            "coco128-yolov8n-seg_output.torchscript.ptl",
        )),
    })
    .expect("YOLO mobile torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 3);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%626"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 218);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "conv2d_clamp_run");
    assert_eq!(
        graph["nodes"][0]["metadata"]["source"],
        "code/__torch__/ultralytics/nn/tasks/___torch_mangle_1806.py:7:22"
    );
    assert_eq!(graph["nodes"][217]["operator"]["name"], "TupleConstruct");
    assert_eq!(graph["nodes"][217]["outputs"], json!(["%626"]));
}

#[test]
fn parses_deeplabv3_mobile_torchscript_archive() {
    let data = deeplabv3_mobile_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("deeplabv3_scripted.ptl")),
    })
    .expect("DeepLabV3 mobile torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 0);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%result.1"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 35);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "conv2d_clamp_run");
    assert_eq!(graph["nodes"][24]["operator"]["name"], "cat");
    assert_eq!(graph["nodes"][24]["inputs"], json!([]));
    assert_eq!(graph["nodes"][34]["operator"]["name"], "If");
}

#[test]
fn parses_deeplabv3_torchscript_archive() {
    let data = deeplabv3_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("deeplabv3_scripted.pt")),
    })
    .expect("DeepLabV3 torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 293);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%result.1"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 578);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "size");
    assert_eq!(graph["nodes"][577]["operator"]["name"], "If");
    assert_eq!(graph["values"][0]["name"], "backbone.conv1.weight");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_d2go_torchscript_archive() {
    let data = d2go_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("d2go.pt")),
    })
    .expect("D2Go torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 5);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%inputs.1"]));
    assert_eq!(graph["outputs"], json!(["%60"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 316);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "__getitem__");
    assert_eq!(graph["nodes"][315]["operator"]["name"], "TupleConstruct");
    assert_eq!(graph["values"][0]["name"], "model.model.pixel_mean");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_bert_torchscript_archive() {
    let data = bert_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("bert-base-uncased.pt")),
    })
    .expect("BERT torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 204);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["inputs"],
        json!(["%input_ids.1", "%attention_mask.1"])
    );
    assert_eq!(graph["outputs"], json!(["%79"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 671);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "size");
    assert_eq!(graph["nodes"][670]["operator"]["name"], "TupleConstruct");
    assert_eq!(graph["values"][0]["name"], "embeddings.token_type_ids");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_densenet161_torchscript_archive() {
    let data = densenet161_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("densenet161_traced.pt")),
    })
    .expect("DenseNet-161 torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 806);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%input.152"]));
    assert_eq!(graph["outputs"], json!(["%2717"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 572);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "_convolution");
    assert_eq!(graph["nodes"][571]["operator"]["name"], "addmm");
    assert_eq!(graph["values"][0]["name"], "features.conv0.weight");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_densenet161_scripted_torchscript_archive() {
    let data = densenet161_scripted_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("densenet161.pt")),
    })
    .expect("scripted DenseNet-161 torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 414);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%ret"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 1076);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "conv2d");
    assert_eq!(graph["nodes"][1075]["operator"]["name"], "If");
    assert_eq!(graph["values"][0]["name"], "features.conv0.weight");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_fairseq_lightweightconv_torchscript_archive() {
    let data = fairseq_lightweightconv_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("fairseq.lightweightconv.pt")),
    })
    .expect("fairseq lightweight convolution torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 4);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%172"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 42);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "size");
    assert_eq!(graph["nodes"][41]["operator"]["name"], "view");
    assert_eq!(graph["values"][0]["name"], "43");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_fasterrcnn_resnet50_fpn_torchscript_archive() {
    let data = fasterrcnn_resnet50_fpn_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("fasterrcnn_resnet50_fpn.pt")),
    })
    .expect("Faster R-CNN torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%images.1", "%targets.1"]));
    assert_eq!(graph["outputs"], json!(["%246"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 227);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "Uninitialized");
    assert_eq!(graph["nodes"][226]["operator"]["name"], "TupleConstruct");
    assert_eq!(
        graph["values"][0]["name"],
        "backbone.fpn.inner_blocks.3.weight"
    );
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_fbdeit_torchscript_archive() {
    let data = fbdeit_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("fbdeit_scripted.pt")),
    })
    .expect("FBDeiT torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.12"));
    assert_eq!(model.tensors.len(), 148);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%988"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 266);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "size");
    assert_eq!(graph["nodes"][265]["operator"]["name"], "If");
    assert_eq!(graph["values"][0]["name"], "patch_embed.proj.weight");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_gpt2_torchscript_archive() {
    let data = gpt2_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("gpt2.pt")),
    })
    .expect("GPT2 torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 161);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["inputs"],
        json!(["%x.1", "%temperature.1", "%top_k.1"])
    );
    assert_eq!(graph["outputs"], json!(["%128"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 490);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "size");
    assert_eq!(graph["nodes"][489]["operator"]["name"], "multinomial");
    assert_eq!(graph["values"][0]["name"], "transformer.wte.weight");
    assert_eq!(graph["values"][0]["initializer"], 0);
}

#[test]
fn parses_inception_v3_traced_torchscript_archive() {
    let data = inception_v3_traced_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("inception_v3_traced.pt")),
    })
    .expect("Inception v3 traced torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 478);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.4"]));
    assert_eq!(graph["outputs"], json!(["%1786"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 331);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "slice");
    assert_eq!(graph["nodes"][330]["operator"]["name"], "addmm");
}

#[test]
fn parses_inception_v3_scripted_torchscript_archive() {
    let data = inception_v3_scripted_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("inception_v3.pt")),
    })
    .expect("Inception v3 scripted torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 470);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%29"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 801);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "If");
    assert_eq!(graph["nodes"][800]["operator"]["name"], "TupleConstruct");
}

#[test]
fn parses_inception_v3_pertensor_torchscript_archive() {
    let data = inception_v3_pertensor_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("iv3_pertensor.pt")),
    })
    .expect("Inception v3 per-tensor quantized torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%20"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 515);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "If");
    assert_eq!(graph["nodes"][514]["operator"]["name"], "TupleConstruct");
}

#[test]
fn parses_mnist_linear_torchscript2_archive() {
    let data = mnist_linear_torchscript2_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mnist_linear_torchscript_2.pt")),
    })
    .expect("MNIST linear TorchScript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 4);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%input.1"]));
    assert_eq!(graph["outputs"], json!(["%30"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 7);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "t");
    assert_eq!(graph["nodes"][6]["operator"]["name"], "add_");
}

#[test]
fn parses_mobilenet_v2_torchscript_archive() {
    let data = mobilenet_v2_torchscript_archive(false);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mobilenet_v2.pt")),
    })
    .expect("MobileNet v2 TorchScript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.tensors.len(), 10);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%ret"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 47);
}

#[test]
fn parses_mobilenet_v2_traced_torchscript_archive() {
    let data = mobilenet_v2_torchscript_archive(true);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mobilenet_v2_traced.pt")),
    })
    .expect("MobileNet v2 traced TorchScript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.tensors.len(), 262);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%input.10"]));
    assert_eq!(graph["outputs"], json!(["%886"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 157);
}

#[test]
fn parses_mask_depthwise_conv_torchscript_archive() {
    let data = mask_depthwise_conv_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mask_depthwise_conv.pt")),
    })
    .expect("mask depthwise conv TorchScript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 0);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%input.1", "%kernel.1"]));
    assert_eq!(graph["outputs"], json!(["%54"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 16);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "shape");
    assert_eq!(graph["nodes"][15]["operator"]["name"], "view");
}

#[test]
fn parses_generated_torchscript_corpus_archives() {
    for case in generated_torchscript_cases() {
        let data = generated_torchscript_archive(case.root, case.source_path, case.source);
        let model = parse(ModelInput {
            data: &data,
            path: Some(std::path::Path::new(case.path)),
        })
        .unwrap_or_else(|error| panic!("{} parses: {error}", case.path));

        assert_eq!(model.format.name, "TorchScript", "{}", case.path);
        assert_eq!(
            model.format.version.as_deref(),
            Some("1.6"),
            "{}",
            case.path
        );
        assert_eq!(model.tensors.len(), case.tensors, "{}", case.path);
        let normalized: serde_json::Value =
            serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
        let graph = &normalized["graphs"][0];
        assert_eq!(graph["inputs"], json!(case.inputs), "{}", case.path);
        assert_eq!(graph["outputs"], json!(case.outputs), "{}", case.path);
        assert_eq!(
            graph["nodes"].as_array().unwrap().len(),
            case.nodes,
            "{}",
            case.path
        );
    }
}

#[test]
fn parses_pytorch_data_pickle_module_inputs() {
    let data = densenet_data_pickle_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("densenet.data.pkl")),
    })
    .expect("pytorch module data pickle parses");

    assert_eq!(model.format.name, "PyTorch Pickle");
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "torchvision.models.densenet.DenseNet"
    );
    assert_eq!(
        graph["nodes"][0]["inputs"],
        json!(["features", "classifier"])
    );
    assert_eq!(graph["values"][0]["type"]["shape"], json!([]));
}

#[test]
fn parses_pytorch_data_pickle_memoized_module_key() {
    let data = memoized_module_key_data_pickle_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("inception_v3.data.pkl")),
    })
    .expect("memoized module key data pickle parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "torchvision.models.inception.Inception3"
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!(["dropout", "fc"]));
}

#[test]
fn parses_fastai_data_pickle_model_module() {
    let data = fastai_data_pickle_fixture();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("fast.ai.data.pkl")),
    })
    .expect("fastai model data pickle parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "torch.nn.modules.container.Sequential"
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!(["0", "1"]));
}

#[test]
fn parses_fastai_zip_pickle_as_learner() {
    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", &fastai_data_pickle_fixture());
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("fruit_veg_model.pkl")),
    })
    .expect("fastai zip pickle parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "fastai.learner.Learner"
    );
    assert_eq!(graph["nodes"][0]["inputs"], json!([null, null, null]));
    assert_eq!(graph["values"], json!([]));
}

#[test]
fn parses_pose_hrnet_zip_pickle_module_inputs() {
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "archive/data.pkl",
        &pose_hrnet_data_pickle_fixture(),
    );
    zip_stored(&mut data, "archive/version", b"3\n");
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("hrnet_posenet_FP32.pth")),
    })
    .expect("pose hrnet zip pickle parses");

    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(
        graph["nodes"][0]["operator"]["name"],
        "models.pose_hrnet.PoseHighResolutionNet"
    );
    assert_eq!(
        graph["nodes"][0]["inputs"],
        json!([
            "conv1",
            "bn1",
            "conv2",
            "bn2",
            "relu",
            "layer1",
            "transition1",
            "stage2",
            "transition2",
            "stage3",
            "transition3",
            "stage4",
            "final_layer",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
        ])
    );
}

#[test]
fn parses_protocol4_pytorch_tensor_pickle() {
    let data = tensor_pickle_fixture(1, false);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("tensor.pkl")),
    })
    .expect("protocol4 tensor pickle parses");

    assert_eq!(model.format.name, "PyTorch Pickle");
    assert_eq!(model.tensors.len(), 1);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["nodes"][0]["operator"]["name"], "builtins.object");
    assert_eq!(graph["nodes"][0]["inputs"], json!([""]));
    assert_eq!(graph["values"][0]["type"]["element_type"], "int64");
    assert_eq!(
        graph["values"][0]["type"]["shape"],
        json!([{ "kind": "known", "value": 5 }])
    );
    assert_eq!(
        normalized["tensors"][0]["storage"],
        json!({ "kind": "inline_bytes", "byte_len": 40 })
    );
}

#[test]
fn parses_protocol4_pytorch_tensor_list_pickle() {
    let data = tensor_pickle_fixture(2, true);
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("tensors.pkl")),
    })
    .expect("protocol4 tensor list pickle parses");

    assert_eq!(model.tensors.len(), 2);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["operator"]["name"],
        "builtins.list"
    );
    assert_eq!(
        normalized["graphs"][0]["nodes"][0]["inputs"],
        json!(["", ""])
    );
}

#[test]
fn parses_torchscript_source_archive_binop_graph() {
    let data = torchscript_binop_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("binop.pt")),
    })
    .expect("torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%11"]));
    assert_eq!(graph["nodes"][0]["operator"]["name"], "DictConstruct");
    assert_eq!(graph["nodes"][0]["outputs"], json!(["%config.1"]));
    assert_eq!(graph["nodes"][1]["operator"]["name"], "__getitem__");
    assert_eq!(graph["nodes"][1]["inputs"], json!(["%config.1"]));
    assert_eq!(
        graph["nodes"][1]["metadata"]["source"],
        "code/__torch__.py:9:21"
    );
    assert_eq!(graph["nodes"][2]["operator"]["name"], "mul");
    assert_eq!(graph["nodes"][2]["inputs"], json!(["%x.1", "%10"]));
    assert_eq!(graph["nodes"][2]["outputs"], json!(["%11"]));
}

#[test]
fn parses_torchscript_source_archive_direct_le_graph() {
    let data = torchscript_direct_le_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("issue1167.pt")),
    })
    .expect("torchscript direct le archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%a.1", "%b.1"]));
    assert_eq!(graph["outputs"], json!(["%5"]));
    assert_eq!(graph["nodes"][0]["operator"]["name"], "le");
    assert_eq!(graph["nodes"][0]["inputs"], json!(["%a.1", "%b.1"]));
    assert_eq!(graph["nodes"][0]["outputs"], json!(["%5"]));
    assert_eq!(
        graph["nodes"][0]["metadata"]["source"],
        "code/__torch__.py:8:16"
    );
}

#[test]
fn parses_prefixed_torchscript_enum_control_flow_graph() {
    let data = torchscript_enum_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("enum_int_test.pt")),
    })
    .expect("prefixed torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%41"]));
    assert_eq!(graph["nodes"][0]["operator"]["name"], "eq");
    assert_eq!(graph["nodes"][0]["outputs"], json!(["%6"]));
    assert_eq!(
        graph["nodes"][0]["metadata"]["source"],
        "code/__torch__.py:10:14"
    );
    assert_eq!(
        graph["nodes"][0]["metadata"]["generated"],
        "/workspace/test_enum_error.py:15:11"
    );
    assert_eq!(graph["nodes"][1]["operator"]["name"], "If");
    assert_eq!(graph["nodes"][1]["inputs"], json!(["%6"]));
    assert_eq!(graph["nodes"][1]["outputs"], json!(["%41"]));
}

#[test]
fn parses_legacy_torchscript_json_linear_graph() {
    let data = legacy_torchscript_json_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("mnist_linear_torchscript_1.pt")),
    })
    .expect("legacy torchscript json archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.0"));
    assert_eq!(model.metadata.producer.as_deref(), Some("pytorch"));
    assert_eq!(model.metadata.producer_version.as_deref(), Some("1.0"));
    assert_eq!(model.tensors.len(), 4);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%input.1"]));
    assert_eq!(graph["outputs"], json!(["%30"]));
    assert_eq!(graph["nodes"][0]["operator"]["name"], "t");
    assert_eq!(graph["nodes"][1]["operator"]["name"], "matmul");
    assert_eq!(graph["nodes"][6]["operator"]["name"], "add_");
    assert_eq!(graph["nodes"][6]["outputs"], json!(["%30"]));
    assert_eq!(graph["values"][0]["name"], "fc1.weight");
    assert_eq!(graph["values"][0]["initializer"], 0);
    assert_eq!(
        graph["values"][0]["type"]["shape"],
        json!([{ "kind": "known", "value": 500 }, { "kind": "known", "value": 784 }])
    );
}

#[test]
fn parses_legacy_cruise_torchscript_graph() {
    let data = legacy_cruise_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("cruise_cutin_vehicle_model.pt")),
    })
    .expect("legacy cruise torchscript json archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.metadata.producer.as_deref(), Some("pytorch"));
    assert_eq!(model.metadata.producer_version.as_deref(), Some("1.0"));
    assert_eq!(model.tensors.len(), 24);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%281"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 63);
    assert_eq!(graph["values"][0]["name"], "lane_feature_conv.0.weight");
    assert_eq!(
        graph["nodes"][8]["inputs"],
        json!([
            "%input_1.1",
            "lane_feature_conv.0.weight",
            "lane_feature_conv.0.bias",
            null,
            null,
            null,
            null
        ])
    );
    assert_eq!(graph["nodes"][62]["operator"]["name"], "TupleConstruct");
}

#[test]
fn parses_traced_torchscript_blitz_graph() {
    let data = traced_blitz_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new(
            "blitz_neural_networks_tutorial_traced.pt",
        )),
    })
    .expect("traced blitz torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.3"));
    assert_eq!(model.tensors.len(), 11);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%input.1"]));
    assert_eq!(graph["outputs"], json!(["%121"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 25);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "_convolution");
    assert_eq!(
        graph["nodes"][0]["metadata"]["source"],
        "code/__torch__/___torch_mangle_15.py:32:18"
    );
    assert_eq!(graph["nodes"][18]["operator"]["name"], "addmm");
    assert_eq!(
        graph["nodes"][18]["inputs"],
        json!(["fc1.bias", "%input5.1", "%106"])
    );
    assert_eq!(graph["nodes"][24]["outputs"], json!(["%121"]));
}

#[test]
fn parses_scripted_torchscript_blitz_graph() {
    let data = scripted_blitz_torchscript_archive();
    let model = parse(ModelInput {
        data: &data,
        path: Some(std::path::Path::new("blitz_neural_networks_tutorial.pt")),
    })
    .expect("scripted blitz torchscript archive parses");

    assert_eq!(model.format.name, "TorchScript");
    assert_eq!(model.format.version.as_deref(), Some("1.6"));
    assert_eq!(model.tensors.len(), 4);
    let normalized: serde_json::Value =
        serde_json::from_str(&model.to_normalized_json().unwrap()).unwrap();
    let graph = &normalized["graphs"][0];
    assert_eq!(graph["inputs"], json!(["%x.1"]));
    assert_eq!(graph["outputs"], json!(["%x5"]));
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 25);
    assert_eq!(graph["nodes"][0]["operator"]["name"], "conv2d");
    assert_eq!(
        graph["nodes"][0]["metadata"]["source"],
        "code/__torch__.py:250:18"
    );
    assert_eq!(
        graph["nodes"][0]["inputs"],
        json!([
            "%x.1",
            "conv1.weight",
            "conv1.bias",
            null,
            null,
            null,
            null,
            null,
            null
        ])
    );
    assert_eq!(
        graph["nodes"][10]["inputs"],
        json!(["%x1.1", null, "%num_features"])
    );
    assert_eq!(graph["nodes"][24]["outputs"], json!(["%x5"]));
    assert_eq!(
        normalized["tensors"][0]["shape"],
        json!([
            { "kind": "known", "value": 6 },
            { "kind": "known", "value": 1 },
            { "kind": "known", "value": 3 },
            { "kind": "known", "value": 3 }
        ])
    );
}

fn legacy_bool_tensor_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
    ]);
    pickle_int2(&mut data, 1001);
    data.extend_from_slice(&[0x80, 0x02, b'}', b'.']);

    data.extend_from_slice(b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(");
    data.push(b'(');
    binunicode(&mut data, "storage");
    data.extend_from_slice(b"ctorch\nBoolStorage\n");
    binunicode(&mut data, "0");
    binunicode(&mut data, "cpu");
    data.extend_from_slice(&[b'K', 2, b'N', b't', b'Q', b'K', 0, b'K', 2, 0x85]);
    data.extend_from_slice(&[b'K', 1, 0x85, 0x89]);
    data.extend_from_slice(b"ccollections\nOrderedDict\n)RtR.");

    data.extend_from_slice(&[0x80, 0x02, b']', b'q', 0]);
    binunicode(&mut data, "0");
    data.extend_from_slice(&[b'q', 1, b'a', b'.']);

    data.extend_from_slice(&2_u64.to_le_bytes());
    data.extend_from_slice(&[1, 0]);
    data
}

fn legacy_dict_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
    ]);
    pickle_int2(&mut data, 1001);
    data.extend_from_slice(&[0x80, 0x02, b'}', b'.']);
    data.extend_from_slice(b"\x80\x02}q\x00.");
    data.extend_from_slice(&[0x80, 0x02, b']', b'q', 0, b'.']);
    data
}

fn legacy_vgg19x_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
    ]);
    pickle_int2(&mut data, 1001);
    data.extend_from_slice(&[0x80, 0x02, b'}', b'.']);
    data.extend_from_slice(b"\x80\x02c__main__\nVGG19X\nq\x00)\x81q\x01.");
    data.extend_from_slice(&[0x80, 0x02, b']', b'q', 0, b'.']);
    data
}

fn exported_program_json_fixture() -> String {
    json!({
        "graph_module": {
            "graph": {
                "inputs": [
                    { "as_tensor": { "name": "x" } },
                    { "as_tensor": { "name": "y" } },
                    { "as_string": "trunc" }
                ],
                "outputs": [
                    { "as_tensor": { "name": "div_1" } }
                ],
                "nodes": [
                    {
                        "target": "torch.ops.aten.div.Tensor_mode",
                        "inputs": [
                            { "name": "self", "arg": { "as_tensor": { "name": "x" } } },
                            { "name": "other", "arg": { "as_tensor": { "name": "y" } } },
                            { "name": "rounding_mode", "arg": { "as_string": "trunc" } }
                        ],
                        "outputs": [
                            { "as_tensor": { "name": "div_1" } }
                        ],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": { "dtype": 7, "sizes": [{ "as_int": 4 }] },
                    "y": { "dtype": 7, "sizes": [{ "as_int": 4 }] },
                    "div_1": { "dtype": 7, "sizes": [{ "as_int": 4 }] }
                }
            },
            "signature": {
                "input_specs": [
                    { "user_input": { "arg": { "as_tensor": { "name": "x" } } } },
                    { "user_input": { "arg": { "as_tensor": { "name": "y" } } } },
                    { "constant_input": { "name": "div", "value": { "as_string": "trunc" } } }
                ],
                "output_specs": [
                    { "user_output": { "arg": { "as_tensor": { "name": "div_1" } } } }
                ]
            }
        },
        "schema_version": { "major": 7, "minor": 3 }
    })
    .to_string()
}

fn legacy_exported_program_json_fixture() -> String {
    r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    { "$type": "as_tensor", "$value": { "name": "x" } }
                ],
                "outputs": [
                    { "$type": "as_tensor", "$value": { "name": "add" } }
                ],
                "nodes": [
                    {
                        "target": "torch.ops.aten.sym_size.int",
                        "inputs": [
                            { "name": "self", "arg": { "$type": "as_tensor", "$value": { "name": "x" } } },
                            { "name": "dim", "arg": { "$type": "as_int", "$value": 1 } }
                        ],
                        "outputs": [
                            { "$type": "as_sym_int", "$value": { "$type": "as_name", "$value": "sym_size_int_2" } }
                        ],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.view.default",
                        "inputs": [
                            { "name": "self", "arg": { "$type": "as_tensor", "$value": { "name": "x" } } },
                            { "name": "size", "arg": { "$type": "as_sym_ints", "$value": [
                                { "$type": "as_name", "$value": "sym_size_int_2" },
                                { "$type": "as_int", "$value": 768 }
                            ] } }
                        ],
                        "outputs": [
                            { "$type": "as_tensor", "$value": { "name": "view" } }
                        ],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.add.Scalar",
                        "inputs": [
                            { "name": "self", "arg": { "$type": "as_tensor", "$value": { "name": "view" } } },
                            { "name": "other", "arg": { "$type": "as_float", "$value": -Infinity } }
                        ],
                        "outputs": [
                            { "$type": "as_tensor", "$value": { "name": "add" } }
                        ],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "x": { "dtype": 7, "sizes": [
                        { "$type": "as_int", "$value": 1 },
                        { "$type": "as_expr", "$value": { "hint": "s0" } }
                    ] },
                    "view": { "dtype": 7, "sizes": [
                        { "$type": "as_int", "$value": 1 },
                        { "$type": "as_int", "$value": 768 }
                    ] },
                    "add": { "dtype": 7, "sizes": [
                        { "$type": "as_int", "$value": 1 },
                        { "$type": "as_int", "$value": 768 }
                    ] }
                }
            },
            "signature": {
                "input_specs": [
                    { "$type": "user_input", "$value": { "arg": { "$type": "as_tensor", "$value": { "name": "x" } } } }
                ],
                "output_specs": [
                    { "$type": "user_output", "$value": { "arg": { "$type": "as_tensor", "$value": { "name": "add" } } } }
                ]
            }
        },
        "schema_version": { "major": 3, "minor": 1 }
    }"#
    .to_owned()
}

fn exported_program_with_weight_json_fixture() -> String {
    json!({
        "graph_module": {
            "graph": {
                "inputs": [
                    { "as_tensor": { "name": "p_weight" } },
                    { "as_tensor": { "name": "x" } },
                    { "as_tensor": { "name": "threshold" } }
                ],
                "outputs": [
                    { "as_tensor": { "name": "add" } }
                ],
                "nodes": [
                    {
                        "target": "torch.ops.aten.add.Tensor",
                        "inputs": [
                            { "name": "self", "arg": { "as_tensor": { "name": "x" } } },
                            { "name": "other", "arg": { "as_tensor": { "name": "p_weight" } } }
                        ],
                        "outputs": [
                            { "as_tensor": { "name": "add" } }
                        ],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.item.default",
                        "inputs": [
                            { "name": "self", "arg": { "as_tensor": { "name": "threshold" } } }
                        ],
                        "outputs": [
                            { "as_sym_int": { "as_name": "item" } }
                        ],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "p_weight": { "dtype": 7, "sizes": [{ "as_int": 2 }] },
                    "x": { "dtype": 7, "sizes": [{ "as_int": 2 }] },
                    "threshold": { "dtype": 5, "sizes": [] },
                    "add": { "dtype": 7, "sizes": [{ "as_int": 2 }] }
                }
            },
            "signature": {
                "input_specs": [
                    {
                        "parameter": {
                            "arg": { "name": "p_weight" },
                            "parameter_name": "linear.weight"
                        }
                    },
                    { "user_input": { "arg": { "as_tensor": { "name": "x" } } } },
                    { "user_input": { "arg": { "as_tensor": { "name": "threshold" } } } }
                ],
                "output_specs": [
                    { "user_output": { "arg": { "as_tensor": { "name": "add" } } } }
                ]
            }
        },
        "schema_version": { "major": 8, "minor": 0 }
    })
    .to_string()
}

fn legacy_plain_dict_state_dict_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
    ]);
    pickle_int2(&mut data, 1001);
    data.extend_from_slice(&[0x80, 0x02, b'}', b'.']);

    data.extend_from_slice(&[0x80, 0x02, b'}', b'q', 0, b'(']);
    binunicode(&mut data, "conv");
    pickle_empty_ordered_dict(&mut data);
    data.push(b'(');
    binunicode(&mut data, "weight");
    pickle_rebuild_tensor(&mut data, "FloatStorage", "0", 6, &[2, 3]);
    data.push(b'u');
    binunicode(&mut data, "bn");
    pickle_empty_ordered_dict(&mut data);
    data.push(b'(');
    binunicode(&mut data, "running_mean");
    pickle_rebuild_tensor(&mut data, "FloatStorage", "1", 2, &[2]);
    data.push(b'u');
    data.extend_from_slice(b"u.");

    data.extend_from_slice(&[0x80, 0x02, b']', b'q', 0, b'(']);
    binunicode(&mut data, "0");
    binunicode(&mut data, "1");
    data.extend_from_slice(b"e.");
    data.extend_from_slice(&6_u64.to_le_bytes());
    data.extend_from_slice(&[0; 24]);
    data.extend_from_slice(&2_u64.to_le_bytes());
    data.extend_from_slice(&[0; 8]);
    data
}

fn torchscript_enum_archive() -> Vec<u8> {
    let source = b"class MyModule(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  _is_full_backward_hook : Optional[bool]\n  value : __torch__.MyEnum\n  def forward(self: __torch__.MyModule,\n    x: Tensor) -> Tensor:\n    value = self.value\n    _0 = torch.eq(value, __torch__.MyEnum.FIRST)\n    if _0:\n      _1 = torch.mul(x, 1.)\n    else:\n      value0 = self.value\n      _2 = torch.eq(value0, __torch__.MyEnum.SECOND)\n      if _2:\n        _3 = torch.mul(x, 2.)\n      else:\n        _3 = torch.mul(x, 3.)\n      _1 = _3\n    return _1\nclass MyEnum(Enum):\n  FIRST = 1\n  SECOND = 2\n  THIRD = 3\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "enum_int_test/code/__torch__.py", source);
    let mut debug = b"\x80\x02".to_vec();
    binunicode(&mut debug, "/workspace/test_enum_error.py");
    debug.push(b'.');
    zip_stored(
        &mut data,
        "enum_int_test/code/__torch__.py.debug_pkl",
        &debug,
    );
    zip_stored(&mut data, "enum_int_test/version", b"3\n");
    data
}

fn torchscript_binop_archive() -> Vec<u8> {
    let source = b"class SimpleModel(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  _is_full_backward_hook : Optional[bool]\n  def forward(self,\n    x: Tensor) -> Tensor:\n    config = {\"width\": 224, \"height\": 224 + 224}\n    return x * config[\"height\"]\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "code/__torch__.py", source);
    zip_stored(&mut data, "version", b"3\n");
    data
}

fn torchscript_direct_le_archive() -> Vec<u8> {
    let source = b"class PlaceholderModule(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  def forward(self: __torch__.PlaceholderModule,\n    a: Tensor,\n    b: Tensor) -> Tensor:\n    return torch.le(a, b)\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "issue1167/code/__torch__.py", source);
    let mut debug = b"\x80\x02".to_vec();
    binunicode(&mut debug, "/workspace/issue1167.py");
    debug.push(b'.');
    zip_stored(&mut data, "issue1167/code/__torch__.py.debug_pkl", &debug);
    zip_stored(&mut data, "issue1167/version", b"3\n");
    data
}

fn pytorch_zip_archive_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "alexnet.fx/data.pkl",
        b"\x80\x02ctorch.fx.graph_module\nreduce_graph_module\nq\x00.",
    );
    zip_stored(&mut data, "alexnet.fx/version", b"3\n");
    data
}

fn pytorch_zip_complex_tensor_archive() -> Vec<u8> {
    let mut pickle = Vec::new();
    pickle.extend_from_slice(b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((");
    binunicode(&mut pickle, "storage");
    pickle_global(&mut pickle, "torch", "ComplexFloatStorage");
    pickle.extend_from_slice(b"q\x02");
    binunicode(&mut pickle, "0");
    binunicode(&mut pickle, "cpu");
    pickle.extend_from_slice(&[b'K', 3, b't', b'q', 5, b'Q', b'K', 0, b'K', 3, 0x85]);
    pickle.extend_from_slice(&[b'q', 6, b'K', 1, 0x85, b'q', 7, 0x89]);
    pickle_global(&mut pickle, "collections", "OrderedDict");
    pickle.extend_from_slice(b"q\x08)Rq\ttq\nRq\x0b.");

    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", &pickle);
    zip_stored(&mut data, "archive/version", b"4\n");
    data
}

fn pytorch_zip_state_dict_archive_fixture(quantized: bool) -> Vec<u8> {
    let pickle = if quantized {
        pytorch_quantized_state_dict_fixture()
    } else {
        pytorch_ordered_state_dict_fixture()
    };
    let mut data = Vec::new();
    zip_stored(&mut data, "archive/data.pkl", &pickle);
    zip_stored(&mut data, "archive/version", b"3\n");
    data
}

fn pytorch_ordered_state_dict_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_empty_ordered_dict(&mut data);
    data.push(b'(');
    binunicode(&mut data, "conv.weight");
    pickle_rebuild_tensor(&mut data, "FloatStorage", "0", 6, &[2, 3]);
    binunicode(&mut data, "conv.bias");
    data.push(b'N');
    binunicode(&mut data, "bn.weight");
    pickle_rebuild_tensor(&mut data, "FloatStorage", "1", 2, &[2]);
    binunicode(&mut data, "bn.running_mean");
    pickle_rebuild_tensor(&mut data, "FloatStorage", "2", 2, &[2]);
    data.extend_from_slice(b"u.");
    data
}

fn pytorch_quantized_state_dict_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_empty_ordered_dict(&mut data);
    data.push(b'(');
    binunicode(&mut data, "fc.scale");
    pickle_rebuild_tensor(&mut data, "FloatStorage", "0", 1, &[]);
    binunicode(&mut data, "fc.zero_point");
    pickle_rebuild_tensor(&mut data, "LongStorage", "1", 1, &[]);
    binunicode(&mut data, "fc._packed_params.dtype");
    pickle_global(&mut data, "torch", "qint8");
    binunicode(&mut data, "fc._packed_params._packed_params");
    pickle_rebuild_qtensor(&mut data, "QInt8Storage", "2", 6, &[2, 3]);
    data.push(b'N');
    data.push(0x86);
    data.extend_from_slice(b"u.");
    data
}

fn pickle_rebuild_tensor(
    data: &mut Vec<u8>,
    storage: &str,
    key: &str,
    storage_size: u16,
    shape: &[u16],
) {
    pickle_global(data, "torch._utils", "_rebuild_tensor_v2");
    data.push(b'(');
    pickle_storage_persistent_id(data, storage, key, storage_size);
    data.push(b'Q');
    data.extend_from_slice(&[b'K', 0]);
    pickle_int_tuple(data, shape);
    data.push(b')');
    data.push(0x89);
    pickle_empty_ordered_dict(data);
    data.push(b't');
    data.push(b'R');
}

fn pickle_rebuild_qtensor(
    data: &mut Vec<u8>,
    storage: &str,
    key: &str,
    storage_size: u16,
    shape: &[u16],
) {
    pickle_global(data, "torch._utils", "_rebuild_qtensor");
    data.push(b'(');
    pickle_storage_persistent_id(data, storage, key, storage_size);
    data.push(b'Q');
    data.extend_from_slice(&[b'K', 0]);
    pickle_int_tuple(data, shape);
    data.push(b')');
    data.push(b'(');
    pickle_global(data, "torch", "per_tensor_affine");
    data.extend_from_slice(&[b'K', 1, b'K', 0, b't']);
    data.push(0x89);
    pickle_empty_ordered_dict(data);
    data.push(b't');
    data.push(b'R');
}

fn pickle_storage_persistent_id(data: &mut Vec<u8>, storage: &str, key: &str, size: u16) {
    data.push(b'(');
    binunicode(data, "storage");
    pickle_global(data, "torch", storage);
    binunicode(data, key);
    binunicode(data, "cpu");
    pickle_small_int(data, size);
    data.push(b't');
}

fn pickle_int_tuple(data: &mut Vec<u8>, values: &[u16]) {
    match values {
        [] => data.push(b')'),
        [value] => {
            pickle_small_int(data, *value);
            data.push(0x85);
        }
        [left, right] => {
            pickle_small_int(data, *left);
            pickle_small_int(data, *right);
            data.push(0x86);
        }
        values => {
            data.push(b'(');
            for value in values {
                pickle_small_int(data, *value);
            }
            data.push(b't');
        }
    }
}

fn pickle_empty_ordered_dict(data: &mut Vec<u8>) {
    pickle_global(data, "collections", "OrderedDict");
    data.extend_from_slice(b")R");
}

fn transducer_torchscript_archive() -> Vec<u8> {
    let source = b"class Decoder(Module):\n  __parameters__ = []\n  def forward(self: __torch__.decoder.Decoder,\n    y: Tensor) -> Tensor:\n    return y\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "cpu_jit/code/__torch__/decoder.py", source);
    zip_stored(
        &mut data,
        "cpu_jit/data.pkl",
        &transducer_data_pickle_fixture(),
    );
    zip_stored(&mut data, "cpu_jit/version", b"4\n");
    data
}

fn yolo_mobile_torchscript_archive() -> Vec<u8> {
    let source = b"class SegmentationModel(Module):\n  __parameters__ = []\n  __buffers__ = []\n  mobile_optimized : bool\n  def forward(self: __torch__.ultralytics.nn.tasks.___torch_mangle_1806.SegmentationModel,\n    x: Tensor) -> Tuple[Tensor, Tensor]:\n    _0 = ops.prepacked.conv2d_clamp_run(x, CONSTANTS.c0)\n    _1 = torch.split_with_sizes(torch.silu_(_0), [16, 16], 1)\n    return (_0, _1)\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "yolov8n-seg/code/__torch__/ultralytics/nn/tasks/___torch_mangle_1806.py",
        source,
    );
    zip_stored(&mut data, "yolov8n-seg/bytecode.pkl", b"\x80\x02).");
    zip_stored(&mut data, "yolov8n-seg/version", b"4\n");
    data
}

fn deeplabv3_mobile_torchscript_archive() -> Vec<u8> {
    let source = b"class DeepLabV3(Module):\n  __parameters__ = []\n  __buffers__ = []\n  mobile_optimized : bool\n  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.___torch_mangle_269.DeepLabV3,\n    x: Tensor) -> Dict[str, Tensor]:\n    _0 = ops.prepacked.conv2d_clamp_run(x, CONSTANTS.c0)\n    result = torch.dict()\n    torch._set_item(result, \"out\", _0)\n    return result\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "deeplabv3_scripted/code/__torch__/torchvision/models/segmentation/deeplabv3/___torch_mangle_269.py",
        source,
    );
    zip_stored(&mut data, "deeplabv3_scripted/bytecode.pkl", b"\x80\x02).");
    zip_stored(&mut data, "deeplabv3_scripted/version", b"4\n");
    data
}

fn deeplabv3_torchscript_archive() -> Vec<u8> {
    let source = b"class DeepLabV3(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  backbone : __torch__.torchvision.models._utils.IntermediateLayerGetter\n  classifier : __torch__.torchvision.models.segmentation.deeplabv3.DeepLabHead\n  aux_classifier : __torch__.torchvision.models.segmentation.fcn.FCNHead\n  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.DeepLabV3,\n    x: Tensor) -> Dict[str, Tensor]:\n    result = torch.dict()\n    torch._set_item(result, \"out\", x)\n    return result\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "deeplabv3_scripted/code/__torch__/torchvision/models/segmentation/deeplabv3.py",
        source,
    );
    zip_stored(&mut data, "deeplabv3_scripted/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "deeplabv3_scripted/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "deeplabv3_scripted/version", b"3\n");
    data
}

fn d2go_torchscript_archive() -> Vec<u8> {
    let source = b"class Wrapper(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  _is_full_backward_hook : Optional[bool]\n  coco_idx : Tensor\n  model : __torch__.detectron2.export.flatten.___torch_mangle_953.TracingAdapter\n  def forward(self: __torch__.Wrapper,\n    inputs: List[Tensor]) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:\n    res = annotate(Dict[str, Tensor], {})\n    torch._set_item(res, \"boxes\", inputs[0])\n    return (inputs, [res])\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "d2go/code/__torch__.py", source);
    zip_stored(&mut data, "d2go/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "d2go/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "d2go/version", b"4\n");
    data
}

fn bert_torchscript_archive() -> Vec<u8> {
    let source = b"class BertModel(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  _is_full_backward_hook : Optional[bool]\n  embeddings : __torch__.transformers.models.bert.modeling_bert.BertEmbeddings\n  encoder : __torch__.transformers.models.bert.modeling_bert.BertEncoder\n  pooler : __torch__.transformers.models.bert.modeling_bert.BertPooler\n  def forward(self: __torch__.transformers.models.bert.modeling_bert.BertModel,\n    input_ids: Tensor,\n    attention_mask: Tensor) -> Tuple[Tensor, Tensor]:\n    attention_mask0 = torch.mul(attention_mask, CONSTANTS.c0)\n    return (input_ids, attention_mask0)\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "traced_bert/code/__torch__/transformers/models/bert/modeling_bert.py",
        source,
    );
    zip_stored(&mut data, "traced_bert/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "traced_bert/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "traced_bert/version", b"4\n");
    data
}

fn densenet161_torchscript_archive() -> Vec<u8> {
    let source = b"class DenseNet(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  features : __torch__.torch.nn.modules.container.___torch_mangle_768.Sequential\n  classifier : __torch__.torch.nn.modules.linear.___torch_mangle_769.Linear\n  def forward(self: __torch__.torchvision.models.densenet.___torch_mangle_770.DenseNet,\n    input: Tensor) -> Tensor:\n    out = torch.adaptive_avg_pool2d(input, [1, 1])\n    input1 = torch.flatten(out, 1, -1)\n    return input1\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "densenet161_traced/code/__torch__/torchvision/models/densenet/___torch_mangle_770.py",
        source,
    );
    zip_stored(&mut data, "densenet161_traced/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "densenet161_traced/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "densenet161_traced/version", b"3\n");
    data
}

fn densenet161_scripted_torchscript_archive() -> Vec<u8> {
    let source = b"class DenseNet(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  features : __torch__.torch.nn.modules.container.___torch_mangle_196.Sequential\n  classifier : __torch__.torch.nn.modules.linear.___torch_mangle_197.Linear\n  def forward(self: __torch__.torchvision.models.densenet.DenseNet,\n    x: Tensor) -> Tensor:\n    _0 = __torch__.torch.nn.functional.adaptive_avg_pool2d\n    features = (self.features).forward(x, )\n    out = __torch__.torch.nn.functional.relu(features, True, )\n    out0 = _0(out, [1, 1], )\n    out1 = torch.flatten(out0, 1, -1)\n    return (self.classifier).forward(out1, )\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "densenet161/code/__torch__/torchvision/models/densenet.py",
        source,
    );
    zip_stored(&mut data, "densenet161/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "densenet161/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "densenet161/version", b"3\n");
    data
}

fn fairseq_lightweightconv_torchscript_archive() -> Vec<u8> {
    let source = b"class LightweightConv1dTBC(Module):\n  __parameters__ = [\"weight\", ]\n  __buffers__ = []\n  weight : Tensor\n  training : bool\n  _is_full_backward_hook : Optional[bool]\n  weight_dropout_module : __torch__.fairseq.modules.fairseq_dropout.___torch_mangle_8.FairseqDropout\n  def forward(self: __torch__.fairseq.modules.lightweight_convolution.___torch_mangle_9.LightweightConv1dTBC,\n    x: Tensor) -> Tensor:\n    T = ops.prim.NumToTensor(torch.size(x, 0))\n    input = torch.view(self.weight, [2, 9])\n    return torch.view(input, [1, 2, 9])\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "fairseq.lightweightconv/code/__torch__/fairseq/modules/lightweight_convolution/___torch_mangle_9.py",
        source,
    );
    zip_stored(
        &mut data,
        "fairseq.lightweightconv/code/__torch__/fairseq/modules/fairseq_dropout/___torch_mangle_8.py",
        b"class FairseqDropout(Module):\n  def forward(self: __torch__.fairseq.modules.fairseq_dropout.___torch_mangle_8.FairseqDropout) -> None:\n    return None\n",
    );
    zip_stored(&mut data, "fairseq.lightweightconv/data.pkl", b"\x80\x02).");
    zip_stored(
        &mut data,
        "fairseq.lightweightconv/constants.pkl",
        b"\x80\x02).",
    );
    zip_stored(&mut data, "fairseq.lightweightconv/version", b"3\n");
    data
}

fn fasterrcnn_resnet50_fpn_torchscript_archive() -> Vec<u8> {
    let source = b"class FasterRCNN(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  _has_warned : bool\n  transform : __torch__.torchvision.models.detection.transform.GeneralizedRCNNTransform\n  backbone : __torch__.torchvision.models.detection.backbone_utils.BackboneWithFPN\n  rpn : __torch__.torchvision.models.detection.rpn.RegionProposalNetwork\n  roi_heads : __torch__.torchvision.models.detection.roi_heads.RoIHeads\n  def forward(self: __torch__.torchvision.models.detection.faster_rcnn.FasterRCNN,\n    images: List[Tensor],\n    targets: Optional[List[Dict[str, Tensor]]]=None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:\n    _4 = uninitialized(List[Dict[str, Tensor]])\n    losses = annotate(Dict[str, Tensor], {})\n    return (losses, images)\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "fasterrcnn_resnet50_fpn/code/__torch__/torchvision.py",
        b"def _is_tracing() -> bool:\n  return False\n",
    );
    zip_stored(
        &mut data,
        "fasterrcnn_resnet50_fpn/code/__torch__/torchvision/models/detection/faster_rcnn.py",
        source,
    );
    zip_stored(&mut data, "fasterrcnn_resnet50_fpn/data.pkl", b"\x80\x02).");
    zip_stored(
        &mut data,
        "fasterrcnn_resnet50_fpn/constants.pkl",
        b"\x80\x02).",
    );
    zip_stored(&mut data, "fasterrcnn_resnet50_fpn/version", b"3\n");
    data
}

fn fbdeit_torchscript_archive() -> Vec<u8> {
    let source = b"class VisionTransformer(Module):\n  __parameters__ = [\"cls_token\", \"pos_embed\", ]\n  __buffers__ = []\n  cls_token : Tensor\n  pos_embed : Tensor\n  training : bool\n  patch_embed : __torch__.timm.layers.patch_embed.PatchEmbed\n  pos_drop : __torch__.torch.nn.modules.dropout.Dropout\n  blocks : __torch__.torch.nn.modules.container.Sequential\n  norm : __torch__.torch.nn.modules.normalization.LayerNorm\n  head : __torch__.torch.nn.modules.linear.___torch_mangle_3.Linear\n  def forward(self: __torch__.timm.models.vision_transformer.VisionTransformer,\n    x: Tensor) -> Tensor:\n    x0 = (self).forward_features(x, )\n    return (self).forward_head(x0, False, )\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "fbdeit_scripted/code/__torch__/timm/models/vision_transformer.py",
        source,
    );
    zip_stored(&mut data, "fbdeit_scripted/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "fbdeit_scripted/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "fbdeit_scripted/.data/version", b"10\n");
    data
}

fn gpt2_torchscript_archive() -> Vec<u8> {
    let source = b"class GPT2(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  transformer : __torch__.torch.nn.modules.container.ModuleDict\n  lm_head : __torch__.torch.nn.modules.linear.___torch_mangle_5.Linear\n  def forward(self: __torch__.GPT2,\n    x: Tensor,\n    temperature: float=0.69999999999999996,\n    top_k: int=40) -> Tensor:\n    logits = torch.div(x, temperature)\n    return torch.multinomial(logits, 1)\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "gpt2/code/__torch__.py", source);
    zip_stored(&mut data, "gpt2/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "gpt2/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "gpt2/.data/version", b"3\n");
    data
}

fn memoized_module_key_data_pickle_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_global(&mut data, "torchvision.models.inception", "Inception3");
    data.extend_from_slice(&[b'q', 0, b')', 0x81, b'q', 1, b'}', b'q', 2, b'(']);
    binunicode(&mut data, "fc");
    data.extend_from_slice(&[b'q', 10, b'N']);
    binunicode(&mut data, "_modules");
    data.extend_from_slice(&[b'q', 3]);
    pickle_global(&mut data, "collections", "OrderedDict");
    data.extend_from_slice(&[b'q', 4, b')', b'R', b'q', 5, b'(']);
    binunicode(&mut data, "dropout");
    data.extend_from_slice(&[b'q', 6]);
    pickle_global(&mut data, "torch.nn.modules.dropout", "Dropout");
    data.extend_from_slice(&[b'q', 7, b')', 0x81, b'q', 8, b'h', 10]);
    pickle_global(&mut data, "torch.nn.modules.linear", "Linear");
    data.extend_from_slice(&[b'q', 9, b')', 0x81, b'q', 11, b'u', b'u', b'b', b'.']);
    data
}

fn torch_compile_optimized_module_pickle_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_global(&mut data, "torch._dynamo.eval_frame", "OptimizedModule");
    data.extend_from_slice(&[b'q', 0, b')', 0x81, b'q', 1, b'}', b'q', 2, b'(']);
    binunicode(&mut data, "_modules");
    data.extend_from_slice(b"q\x03}q\x04(");
    binunicode(&mut data, "fc1");
    pickle_global(&mut data, "torch.nn.modules.linear", "Linear");
    data.extend_from_slice(&[b'q', 5, b')', 0x81, b'q', 6]);
    binunicode(&mut data, "relu");
    pickle_global(&mut data, "torch.nn.modules.activation", "ReLU");
    data.extend_from_slice(&[b'q', 7, b')', 0x81, b'q', 8, b'u', b'u', b'b', b'.']);
    data
}

fn inception_v3_traced_torchscript_archive() -> Vec<u8> {
    let source = b"class Inception3(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  Conv2d_1a_3x3 : __torch__.torchvision.models.inception.___torch_mangle_1452.BasicConv2d\n  fc : __torch__.torch.nn.modules.linear.___torch_mangle_1755.Linear\n  def forward(self: __torch__.torchvision.models.inception.___torch_mangle_1756.Inception3,\n    x: Tensor) -> Tensor:\n    x0 = torch.flatten(x, 1, -1)\n    return (self.fc).forward(x0, )\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "inception_v3_traced/code/__torch__/torchvision/models/inception/___torch_mangle_1756.py",
        source,
    );
    zip_stored(&mut data, "inception_v3_traced/data.pkl", b"\x80\x02).");
    zip_stored(
        &mut data,
        "inception_v3_traced/constants.pkl",
        b"\x80\x02).",
    );
    zip_stored(&mut data, "inception_v3_traced/version", b"3\n");
    data
}

fn inception_v3_scripted_torchscript_archive() -> Vec<u8> {
    let source = b"class Inception3(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  aux_logits : bool\n  transform_input : bool\n  Conv2d_1a_3x3 : __torch__.torchvision.models.inception.BasicConv2d\n  fc : __torch__.torch.nn.modules.linear.___torch_mangle_1449.Linear\n  def forward(self: __torch__.torchvision.models.inception.Inception3,\n    x: Tensor) -> __torch__.torchvision.models.inception.InceptionOutputs:\n    _0 = torch.flatten(x, 1, -1)\n    return __torch__.torchvision.models.inception.InceptionOutputs(_0, None, )\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "inception_v3/code/__torch__/torchvision/models/inception.py",
        source,
    );
    zip_stored(&mut data, "inception_v3/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "inception_v3/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "inception_v3/version", b"3\n");
    data
}

fn inception_v3_pertensor_torchscript_archive() -> Vec<u8> {
    let named_tuple_source =
        b"class InceptionOutputs(NamedTuple):\n  logits : Tensor\n  aux_logits : Optional[Tensor]\n";
    let source = b"class QuantizableInception3(Module):\n  __parameters__ = []\n  AuxLogits : None\n  training : bool\n  aux_logits : bool\n  transform_input : bool\n  fc : __torch__.torch.nn.quantized.modules.linear.Linear\n  quant : __torch__.torch.nn.quantized.modules.Quantize\n  dequant : __torch__.torch.nn.quantized.modules.DeQuantize\n  def forward(self: __torch__.torchvision.models.quantization.inception.QuantizableInception3,\n    x: Tensor) -> __torch__.torchvision.models.inception.InceptionOutputs:\n    x0 = (self.quant).forward(x, )\n    x1 = (self.dequant).forward(x0, )\n    return __torch__.torchvision.models.inception.InceptionOutputs(x1, None)\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "iv3_pertensor/code/__torch__/torchvision/models/inception.py",
        named_tuple_source,
    );
    zip_stored(
        &mut data,
        "iv3_pertensor/code/__torch__/torchvision/models/quantization/inception.py",
        source,
    );
    zip_stored(&mut data, "iv3_pertensor/data.pkl", b"\x80\x02).");
    zip_stored(&mut data, "iv3_pertensor/constants.pkl", b"\x80\x02).");
    zip_stored(&mut data, "iv3_pertensor/version", b"3\n");
    data
}

fn mnist_linear_torchscript2_archive() -> Vec<u8> {
    let source = b"class NeuralNet(Module):\n  __parameters__ = []\n  fc1 : __torch__.torch.nn.modules.linear.Linear\n  relu : __torch__.torch.nn.modules.activation.ReLU\n  fc2 : __torch__.torch.nn.modules.linear.___torch_mangle_0.Linear\n  def forward(self: __torch__.NeuralNet,\n    input: Tensor) -> Tensor:\n    _0 = self.fc1\n    weight = _0.weight\n    bias = _0.bias\n    _1 = self.fc2\n    weight0 = _1.weight\n    bias0 = _1.bias\n    output = torch.matmul(input, torch.t(weight))\n    input0 = torch.add_(output, bias, alpha=1)\n    input1 = torch.relu(input0)\n    output0 = torch.matmul(input1, torch.t(weight0))\n    return torch.add_(output0, bias0, alpha=1)\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "mnist_linear_torchscript/code/__torch__.py",
        source,
    );
    zip_stored(
        &mut data,
        "mnist_linear_torchscript/data.pkl",
        b"\x80\x02).",
    );
    zip_stored(
        &mut data,
        "mnist_linear_torchscript/constants.pkl",
        b"\x80\x02).",
    );
    zip_stored(&mut data, "mnist_linear_torchscript/version", b"3\n");
    data
}

fn mobilenet_v2_torchscript_archive(traced: bool) -> Vec<u8> {
    let mut data = Vec::new();
    if traced {
        let source = b"class MobileNetV2(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  features : __torch__.torch.nn.modules.container.___torch_mangle_2351.Sequential\n  classifier : __torch__.torch.nn.modules.container.___torch_mangle_2354.Sequential\n  def forward(self: __torch__.torchvision.models.mobilenet.___torch_mangle_2355.MobileNetV2,\n    input: Tensor) -> Tensor:\n    _0 = self.classifier\n    _1 = (self.features).forward(input, )\n    _2 = torch.adaptive_avg_pool2d(_1, [1, 1])\n    _3 = ops.prim.NumToTensor(torch.size(_1, 0))\n    input0 = torch.reshape(_2, [int(_3), -1])\n    return (_0).forward(input0, )\n";
        zip_stored(
            &mut data,
            "mobilenet_v2_traced/code/__torch__/torchvision/models/mobilenet/___torch_mangle_2355.py",
            source,
        );
        zip_stored(&mut data, "mobilenet_v2_traced/data.pkl", b"\x80\x02).");
        zip_stored(
            &mut data,
            "mobilenet_v2_traced/constants.pkl",
            b"\x80\x02).",
        );
        zip_stored(&mut data, "mobilenet_v2_traced/version", b"3\n");
    } else {
        let source = b"class MobileNetV2(Module):\n  __parameters__ = []\n  __buffers__ = []\n  training : bool\n  last_channel : int\n  features : __torch__.torch.nn.modules.container.___torch_mangle_2139.Sequential\n  classifier : __torch__.torch.nn.modules.container.___torch_mangle_2142.Sequential\n  def forward(self: __torch__.torchvision.models.mobilenet.MobileNetV2,\n    x: Tensor) -> Tensor:\n    return (self)._forward_impl(x, )\n  def _forward_impl(self: __torch__.torchvision.models.mobilenet.MobileNetV2,\n    x: Tensor) -> Tensor:\n    x0 = (self.features).forward(x, )\n    return (self.classifier).forward(x0, )\n";
        zip_stored(
            &mut data,
            "mobilenet_v2/code/__torch__/torchvision/models/mobilenet.py",
            source,
        );
        zip_stored(&mut data, "mobilenet_v2/data.pkl", b"\x80\x02).");
        zip_stored(&mut data, "mobilenet_v2/constants.pkl", b"\x80\x02).");
        zip_stored(&mut data, "mobilenet_v2/version", b"3\n");
    }
    data
}

fn mask_depthwise_conv_torchscript_archive() -> Vec<u8> {
    let source = b"op_version_set = 1\nclass DepthwiseConv2Group(Module):\n  __parameters__ = []\n  def forward(self,\n    input: Tensor,\n    kernel: Tensor) -> Tensor:\n    _0 = torch.slice(ops.prim.shape(kernel), 0, 2, 1)\n    batch, channel, = _0\n    _1 = [1, torch.mul(batch, channel), torch.size(input, 2), torch.size(input, 3)]\n    input0 = torch.view(input, _1)\n    _2 = [torch.mul(batch, channel), 1, torch.size(kernel, 2), torch.size(kernel, 3)]\n    kernel0 = torch.view(kernel, _2)\n    feature = torch.conv2d(input0, kernel0, None, [1, 1], [0, 0], [1, 1], torch.mul(batch, channel))\n    _3 = [batch, channel, torch.size(feature, 2), torch.size(feature, 3)]\n    return torch.view(feature, _3)\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "mask_depthwise_conv/code/__torch__/models/rpn/___torch_mangle_438.py",
        source,
    );
    zip_stored(&mut data, "mask_depthwise_conv/data.pkl", b"\x80\x02).");
    zip_stored(
        &mut data,
        "mask_depthwise_conv/constants.pkl",
        b"\x80\x02).",
    );
    zip_stored(&mut data, "mask_depthwise_conv/version", b"3\n");
    data
}

struct GeneratedTorchScriptCase {
    path: &'static str,
    root: &'static str,
    source_path: &'static str,
    source: &'static [u8],
    tensors: usize,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
    nodes: usize,
}

fn generated_torchscript_cases() -> Vec<GeneratedTorchScriptCase> {
    vec![
        GeneratedTorchScriptCase {
            path: "LMModel1.pt",
            root: "LMModel1",
            source_path: "code/__torch__.py",
            source: b"class LMModel1(Module):\n",
            tensors: 5,
            inputs: &["%x.1"],
            outputs: &["%120"],
            nodes: 18,
        },
        GeneratedTorchScriptCase {
            path: "lnf_latest.pt",
            root: "silero_denoiser_8x64x128x256_pt20",
            source_path: "code/__torch__.py",
            source: b"class JITSileroDenoiser(Module):\nclass JitGenerator(Module):\n",
            tensors: 12,
            inputs: &["%x.2"],
            outputs: &["%5549"],
            nodes: 225,
        },
        GeneratedTorchScriptCase {
            path: "m4-sWE-0.1B.script.pt",
            root: "m4-sWE-0.1B.script",
            source_path: "code/__torch__/___torch_mangle_7.py",
            source: b"class WordEmbeddingWrapper(Module):\n",
            tensors: 0,
            inputs: &["%input_ids.1"],
            outputs: &["%inputs_embeds"],
            nodes: 10,
        },
        GeneratedTorchScriptCase {
            path: "mask_model.pt",
            root: "mask_model",
            source_path: "code/__torch__/___torch_mangle_631.py",
            source: b"class WrappedDETR(Module):\n",
            tensors: 22,
            inputs: &["%inputs.1"],
            outputs: &["%1814"],
            nodes: 216,
        },
        GeneratedTorchScriptCase {
            path: "mask_rcnn.pt",
            root: "mask_rcnn",
            source_path: "code/__torch__/torchvision/models/detection/mask_rcnn.py",
            source: b"class MaskRCNN(Module):\n  def forward(self,\n    images: List[Tensor],\n    targets: Optional[List[Dict[str, Tensor]]]=None):\n",
            tensors: 22,
            inputs: &["%images.1", "%targets.1"],
            outputs: &["%273"],
            nodes: 236,
        },
        GeneratedTorchScriptCase {
            path: "mobilefacenet_scripted.pt",
            root: "mobilefacenet_scripted",
            source_path: "code/__torch__/mobilefacenet.py",
            source: b"class MobileFaceNet(Module):\n",
            tensors: 31,
            inputs: &["%x.1"],
            outputs: &["%x13.1"],
            nodes: 69,
        },
        GeneratedTorchScriptCase {
            path: "mobilenet_quantization_scripted_quantized_1.pth",
            root: "mobilenet_quantization_scripted_quantized",
            source_path: "code/__torch__/___torch_mangle_208.py",
            source: b"class InvertedResidual(Module):\n",
            tensors: 2,
            inputs: &["%x.1"],
            outputs: &["%2586"],
            nodes: 34,
        },
        GeneratedTorchScriptCase {
            path: "mobilenet_quantization_scripted_quantized_2.pth",
            root: "mobilenet_quantization_scripted_quantized",
            source_path: "code/__torch__/___torch_mangle_89.py",
            source: b"class InvertedResidual(Module):\n",
            tensors: 2,
            inputs: &["%x.1"],
            outputs: &["%1277"],
            nodes: 34,
        },
        GeneratedTorchScriptCase {
            path: "model_0_epochs.pt",
            root: "model_0_epochs",
            source_path: "code/__torch__/models.py",
            source: b"class MixModel(Module):\n",
            tensors: 4,
            inputs: &["%input_mask.1", "%input_board.1"],
            outputs: &["%111"],
            nodes: 21,
        },
        GeneratedTorchScriptCase {
            path: "model_fnet.pt",
            root: "model",
            source_path: "code/__torch__/torch/fx/graph_module/___torch_mangle_606.py",
            source: b"class GraphModule(Module):\n",
            tensors: 36,
            inputs: &["%argument_1.1"],
            outputs: &["%528"],
            nodes: 118,
        },
        GeneratedTorchScriptCase {
            path: "mobilenetv2-quant_full-nnapi.pt",
            root: "mobilenetv2-quant_full-nnapi",
            source_path: "code/__torch__/___torch_mangle_7876.py",
            source: b"class BundleWrapper(Module):\n",
            tensors: 0,
            inputs: &["%arg.1"],
            outputs: &["%62"],
            nodes: 21,
        },
        GeneratedTorchScriptCase {
            path: "mobilenetv2-quant_full-cpu.pt",
            root: "mobilenetv2-quant_full-cpu",
            source_path: "code/__torch__/torchvision/models/quantization/mobilenetv2/___torch_mangle_7875.py",
            source: b"class QuantizableMobileNetV2(Module):\n",
            tensors: 0,
            inputs: &["%x.1"],
            outputs: &["%x3.1"],
            nodes: 277,
        },
        GeneratedTorchScriptCase {
            path: "resnet18_quantized_cifar10.pt.zip",
            root: "resnet18_quantized_cifar10",
            source_path: "code/__torch__.py",
            source: b"class QuantizedResNet18(Module):\n  model_fp32 : __torch__.resnet.ResNet\n",
            tensors: 2,
            inputs: &["%x.1"],
            outputs: &["%288"],
            nodes: 153,
        },
        GeneratedTorchScriptCase {
            path: "model_static_cpu.pt",
            root: "model_dynamic_cpu",
            source_path: "code/__torch__/___torch_mangle_5.py",
            source: b"class SpeechRecognizer(Module):\n__annotations__[\"quantized_forward._jit_pass_packed_weight_0\"] = __torch__.torch.classes.quantized.Conv2dPackedParamsBase\n",
            tensors: 233,
            inputs: &["%input_signal.1", "%input_signal_length.1"],
            outputs: &["%6409"],
            nodes: 1887,
        },
        GeneratedTorchScriptCase {
            path: "model_dynamic_cpu.pt",
            root: "model_dynamic_cpu",
            source_path: "code/__torch__/___torch_mangle_5.py",
            source: b"class SpeechRecognizer(Module):\n__annotations__[\"quantized_forward._jit_pass_packed_weight_0\"] = __torch__.torch.classes.quantized.LinearPackedParamsBase\n",
            tensors: 347,
            inputs: &["%input_signal.1", "%input_signal_length.1"],
            outputs: &["%5498"],
            nodes: 1478,
        },
        GeneratedTorchScriptCase {
            path: "module_000007.pt",
            root: "module_000007",
            source_path: "code/__torch__/torch/nn/modules/conv.py",
            source: b"class Conv2d(Module):\n  out_channels : Final[int] = 8\n  in_channels : Final[int] = 3\n",
            tensors: 26,
            inputs: &[
                "%images.1",
                "%intrinsics.1",
                "%extrinsics.1",
                "%depth_min.1",
                "%depth_max.1",
            ],
            outputs: &["%564"],
            nodes: 100,
        },
    ]
}

fn generated_torchscript_archive(root: &str, source_path: &str, source: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    zip_stored(&mut data, &format!("{root}/{source_path}"), source);
    zip_stored(&mut data, &format!("{root}/data.pkl"), b"\x80\x02).");
    zip_stored(&mut data, &format!("{root}/constants.pkl"), b"\x80\x02).");
    zip_stored(&mut data, &format!("{root}/version"), b"3\n");
    data
}

fn pytorch_package_archive_fixture() -> Vec<u8> {
    let pickle = b"\x80\x03cmodels.DCGAN\nDCGAN\nq\x00)\x81q\x01.";
    let mut data = Vec::new();
    zip_stored(&mut data, "package/DCGAN/model.pkl", pickle);
    zip_stored(&mut data, "package/DCGAN2/model2.pkl", pickle);
    zip_stored(&mut data, "package/.data/version", b"6\n");
    data
}

fn transducer_data_pickle_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_global(&mut data, "__torch__.model", "Transducer");
    data.extend_from_slice(&[b'q', 0, b')', 0x81, b'q', 1, b'}', b'q', 2, b'(']);
    binunicode(&mut data, "_modules");
    data.extend_from_slice(&[b'q', 3]);
    pickle_global(&mut data, "collections", "OrderedDict");
    data.extend_from_slice(&[b')', b'R', b'q', 4, b'(']);
    for (index, name) in [
        "encoder",
        "decoder",
        "joiner",
        "simple_am_proj",
        "simple_lm_proj",
        "ctc_output",
    ]
    .iter()
    .enumerate()
    {
        binunicode(&mut data, name);
        data.extend_from_slice(&[b'q', 5 + (index as u8 * 2)]);
        pickle_global(&mut data, "__torch__.module", "Placeholder");
        data.extend_from_slice(&[b')', 0x81, b'q', 6 + (index as u8 * 2)]);
    }
    data.extend_from_slice(b"uub.");
    data
}

fn legacy_torchscript_json_archive() -> Vec<u8> {
    let model_json = br#"{"protoVersion":"3","mainModule":{"submodules":[{"parameters":[{"isBuffer":false,"tensorId":"0","name":"weight"},{"isBuffer":false,"tensorId":"1","name":"bias"}],"name":"fc1","optimize":true},{"name":"relu","optimize":true},{"parameters":[{"isBuffer":false,"tensorId":"2","name":"weight"},{"isBuffer":false,"tensorId":"3","name":"bias"}],"name":"fc2","optimize":true}],"torchscriptArena":{"key":"code/mnist_linear_torchscript.py"},"name":"mnist_linear_torchscript","optimize":true},"producerName":"pytorch","producerVersion":"1.0","tensors":[{"dims":["500","784"],"offset":"0","strides":["784","1"],"requiresGrad":true,"dataType":"FLOAT","data":{"key":"tensors/0"},"device":"cpu"},{"dims":["500"],"offset":"0","strides":["1"],"requiresGrad":true,"dataType":"FLOAT","data":{"key":"tensors/1"},"device":"cpu"},{"dims":["10","500"],"offset":"0","strides":["500","1"],"requiresGrad":true,"dataType":"FLOAT","data":{"key":"tensors/2"},"device":"cpu"},{"dims":["10"],"offset":"0","strides":["1"],"requiresGrad":true,"dataType":"FLOAT","data":{"key":"tensors/3"},"device":"cpu"}]}"#;
    let source = b"op_version_set = 0\ndef forward(self,\n    input: Tensor) -> Tensor:\n  _0 = self.fc1\n  weight = _0.weight\n  bias = _0.bias\n  _1 = self.fc2\n  weight0 = _1.weight\n  bias0 = _1.bias\n  output = torch.matmul(input, torch.t(weight))\n  input0 = torch.add_(output, bias, alpha=1)\n  input1 = torch.relu(input0)\n  output0 = torch.matmul(input1, torch.t(weight0))\n  return torch.add_(output0, bias0, alpha=1)\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "mnist_linear_torchscript/model.json", model_json);
    zip_stored(
        &mut data,
        "mnist_linear_torchscript/code/mnist_linear_torchscript.py",
        source,
    );
    zip_stored(&mut data, "mnist_linear_torchscript/version", b"1\n");
    data
}

fn legacy_cruise_torchscript_archive() -> Vec<u8> {
    let shapes = [
        vec!["10", "4", "3"],
        vec!["10"],
        vec!["25", "10", "3"],
        vec!["25"],
        vec!["40", "68"],
        vec!["40"],
        vec!["24", "40"],
        vec!["24"],
        vec!["66", "124"],
        vec!["66"],
        vec!["48", "66"],
        vec!["48"],
        vec!["11", "48"],
        vec!["11"],
        vec!["1", "11"],
        vec!["1"],
        vec!["77", "125"],
        vec!["77"],
        vec!["46", "77"],
        vec!["46"],
        vec!["12", "46"],
        vec!["12"],
        vec!["1", "12"],
        vec!["1"],
    ];
    let tensors = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            json!({
                "dims": shape,
                "offset": "0",
                "strides": ["1"],
                "requiresGrad": true,
                "dataType": "FLOAT",
                "data": { "key": format!("tensors/{index}") },
                "device": "cpu"
            })
        })
        .collect::<Vec<_>>();
    let model_json = json!({
        "protoVersion": "1",
        "mainModule": {
            "submodules": [
                legacy_container_module(
                    "lane_feature_conv",
                    vec![legacy_parameter_module("0", 0, 1), legacy_parameter_module("2", 2, 3)]
                ),
                legacy_container_module(
                    "obs_feature_fc",
                    vec![legacy_parameter_module("0", 4, 5), legacy_parameter_module("3", 6, 7)]
                ),
                legacy_container_module(
                    "classify",
                    vec![
                        legacy_parameter_module("0", 8, 9),
                        legacy_parameter_module("3", 10, 11),
                        legacy_parameter_module("6", 12, 13),
                        legacy_parameter_module("9", 14, 15),
                    ]
                ),
                legacy_container_module(
                    "regress",
                    vec![
                        legacy_parameter_module("0", 16, 17),
                        legacy_parameter_module("3", 18, 19),
                        legacy_parameter_module("6", 20, 21),
                        legacy_parameter_module("9", 22, 23),
                    ]
                )
            ],
            "torchscriptArena": { "key": "code/cruise.py" },
            "name": "cruise",
            "optimize": true
        },
        "producerName": "pytorch",
        "producerVersion": "1.0",
        "tensors": tensors
    })
    .to_string();
    let source = b"op_version_set = 0\nself.lane_feature_conv\ntorch.max_pool1d_with_indices(x)\ngetattr(self.regress, \"9\").bias\n";
    let mut data = Vec::new();
    zip_stored(&mut data, "cruise/model.json", model_json.as_bytes());
    zip_stored(&mut data, "cruise/code/cruise.py", source);
    zip_stored(&mut data, "cruise/version", b"1\n");
    data
}

fn legacy_container_module(name: &str, submodules: Vec<serde_json::Value>) -> serde_json::Value {
    legacy_module(name, Vec::new(), submodules)
}

fn legacy_parameter_module(name: &str, weight: usize, bias: usize) -> serde_json::Value {
    legacy_module(
        name,
        vec![
            legacy_parameter(weight, "weight"),
            legacy_parameter(bias, "bias"),
        ],
        Vec::new(),
    )
}

fn legacy_module(
    name: &str,
    parameters: Vec<serde_json::Value>,
    submodules: Vec<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "name": name,
        "optimize": true,
        "parameters": parameters,
        "submodules": submodules
    })
}

fn legacy_parameter(tensor_id: usize, name: &str) -> serde_json::Value {
    json!({
        "isBuffer": false,
        "tensorId": tensor_id.to_string(),
        "name": name
    })
}

fn traced_blitz_torchscript_archive() -> Vec<u8> {
    let source = b"op_version_set = 1\nimport __torch__.torch.nn.modules.conv.___torch_mangle_16\nimport __torch__.torch.nn.modules.conv.___torch_mangle_17\nimport __torch__.torch.nn.modules.linear.___torch_mangle_18\nimport __torch__.torch.nn.modules.linear.___torch_mangle_19\nimport __torch__.torch.nn.modules.linear.___torch_mangle_20\nclass Net(Module):\n  __parameters__ = []\n  training : bool\n  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_16.Conv2d\n  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_17.Conv2d\n  fc1 : __torch__.torch.nn.modules.linear.___torch_mangle_18.Linear\n  fc2 : __torch__.torch.nn.modules.linear.___torch_mangle_19.Linear\n  fc3 : __torch__.torch.nn.modules.linear.___torch_mangle_20.Linear\n  def forward(self: __torch__.___torch_mangle_15.Net,\n    input: Tensor) -> Tensor:\n    _0 = self.conv1\n    weight = _0.weight\n    _1 = _0.bias\n    _2 = self.conv2\n    weight0 = _2.weight\n    _3 = _2.bias\n    _4 = self.fc1\n    weight1 = _4.weight\n    bias = _4.bias\n    _5 = self.fc2\n    weight2 = _5.weight\n    bias0 = _5.bias\n    _6 = self.fc3\n    weight3 = _6.weight\n    bias1 = _6.bias\n    input0 = torch._convolution(input, weight, _1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True)\n    input1 = torch.relu(input0)\n    input2 = torch.max_pool2d(input1, [2, 2], annotate(List[int], []), [0, 0], [1, 1], False)\n    input3 = torch._convolution(input2, weight0, _3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True)\n    input4 = torch.relu(input3)\n    x = torch.max_pool2d(input4, [2, 2], annotate(List[int], []), [0, 0], [1, 1], False)\n    s = ops.prim.NumToTensor(torch.size(x, 1))\n    s0 = ops.prim.NumToTensor(torch.size(x, 2))\n    s1 = ops.prim.NumToTensor(torch.size(x, 3))\n    num_features = torch.mul(s, CONSTANTS.c0)\n    num_features0 = torch.mul_(num_features, s0)\n    _7 = [-1, int(torch.mul_(num_features0, s1))]\n    input5 = torch.view(x, _7)\n    input6 = torch.addmm(bias, input5, torch.t(weight1), beta=1, alpha=1)\n    input7 = torch.relu(input6)\n    input8 = torch.addmm(bias0, input7, torch.t(weight2), beta=1, alpha=1)\n    input9 = torch.relu(input8)\n    _8 = torch.addmm(bias1, input9, torch.t(weight3), beta=1, alpha=1)\n    return _8\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "blitz_neural_networks_tutorial_traced/code/__torch__/___torch_mangle_15.py",
        source,
    );
    zip_stored(
        &mut data,
        "blitz_neural_networks_tutorial_traced/data.pkl",
        &traced_blitz_data_pickle(),
    );
    zip_stored(
        &mut data,
        "blitz_neural_networks_tutorial_traced/constants.pkl",
        &traced_blitz_constants_pickle(),
    );
    zip_stored(
        &mut data,
        "blitz_neural_networks_tutorial_traced/version",
        b"1\n",
    );
    data
}

fn scripted_blitz_torchscript_archive() -> Vec<u8> {
    let source = b"op_version_set = 1\nimport __torch__.torch.nn.modules.conv\nimport __torch__.torch.nn.modules.conv.___torch_mangle_3\nimport __torch__.torch.nn.modules.linear\nimport __torch__.torch.nn.modules.linear.___torch_mangle_11\nimport __torch__.torch.nn.modules.linear.___torch_mangle_13\nclass Net(Module):\n  __parameters__ = []\n  training : bool\n  conv1 : __torch__.torch.nn.modules.conv.Conv2d\n  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_3.Conv2d\n  fc1 : __torch__.torch.nn.modules.linear.Linear\n  fc2 : __torch__.torch.nn.modules.linear.___torch_mangle_11.Linear\n  fc3 : __torch__.torch.nn.modules.linear.___torch_mangle_13.Linear\n  def forward(self: __torch__.Net,\n    x: Tensor) -> Tensor:\n    x0 = torch.conv2d(x, CONSTANTS.c0, CONSTANTS.c1)\n    x1 = torch.max_pool2d(x0, [2, 2], [2, 2])\n    x2 = self.num_flat_features(x1)\n    x3 = torch.addmm(CONSTANTS.c2, x1, CONSTANTS.c3)\n    return x3\n  def num_flat_features(self: __torch__.Net,\n    x: Tensor) -> int:\n    size = torch.slice(torch.size(x), 1, 9223372036854775807, 1)\n    num_features = 1\n    for _274 in range(torch.len(size)):\n      s = size[_274]\n      num_features = torch.mul(num_features, s)\n    return num_features\n";
    let mut data = Vec::new();
    zip_stored(
        &mut data,
        "blitz_neural_networks_tutorial/code/__torch__.py",
        source,
    );
    zip_stored(
        &mut data,
        "blitz_neural_networks_tutorial/data.pkl",
        &scripted_blitz_data_pickle(),
    );
    zip_stored(&mut data, "blitz_neural_networks_tutorial/version", b"3\n");
    data
}

fn scripted_blitz_data_pickle() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    for shape in [&[6][..], &[6, 1, 3, 3], &[16], &[16, 6, 3, 3]] {
        pickle_tensor_shape(&mut data, "FloatStorage", shape);
    }
    data.push(b'.');
    data
}

fn traced_blitz_data_pickle() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    for shape in [
        &[6, 1, 3, 3][..],
        &[6],
        &[16, 6, 3, 3],
        &[16],
        &[120, 576],
        &[120],
        &[84, 120],
        &[84],
        &[10, 84],
        &[10],
    ] {
        pickle_tensor_shape(&mut data, "FloatStorage", shape);
    }
    data.push(b'.');
    data
}

fn traced_blitz_constants_pickle() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_tensor_shape(&mut data, "LongStorage", &[]);
    data.push(b'.');
    data
}

fn pickle_tensor_shape(data: &mut Vec<u8>, storage: &str, shape: &[u16]) {
    pickle_global(data, "torch", storage);
    data.extend_from_slice(&[b'Q', b'K', 0, b'(']);
    for dimension in shape {
        pickle_small_int(data, *dimension);
    }
    data.push(b't');
}

fn densenet_data_pickle_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_global(&mut data, "torchvision.models.densenet", "DenseNet");
    data.extend_from_slice(&[b'q', 0, b')', 0x81, b'q', 1, b'}', b'q', 2, b'(']);
    binunicode(&mut data, "_modules");
    data.extend_from_slice(&[b'q', 3]);
    pickle_global(&mut data, "collections", "OrderedDict");
    data.extend_from_slice(&[b')', b'R', b'q', 4, b'(']);
    binunicode(&mut data, "features");
    data.extend_from_slice(&[b'q', 5]);
    pickle_global(&mut data, "torch.nn.modules.container", "Sequential");
    data.extend_from_slice(&[b')', 0x81, b'q', 6]);
    binunicode(&mut data, "classifier");
    data.extend_from_slice(&[b'q', 7]);
    pickle_global(&mut data, "torch.nn.modules.linear", "Linear");
    data.extend_from_slice(&[b')', 0x81, b'q', 8, b'u', b'u', b'b', b'.']);
    data
}

fn fastai_data_pickle_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_global(&mut data, "fastai.learner", "Learner");
    data.extend_from_slice(&[b'q', 0, b')', 0x81, b'q', 1, b'}', b'q', 2, b'(']);
    binunicode(&mut data, "model");
    data.extend_from_slice(&[b'q', 3]);
    pickle_global(&mut data, "torch.nn.modules.container", "Sequential");
    data.extend_from_slice(&[b'q', 4, b')', 0x81, b'q', 5, b'}', b'q', 6, b'(']);
    binunicode(&mut data, "_modules");
    data.extend_from_slice(&[b'q', 7]);
    pickle_global(&mut data, "collections", "OrderedDict");
    data.extend_from_slice(&[b')', b'R', b'q', 8, b'(']);
    binunicode(&mut data, "0");
    pickle_global(&mut data, "torch.nn.modules.conv", "Conv2d");
    data.extend_from_slice(&[b')', 0x81]);
    binunicode(&mut data, "1");
    pickle_global(&mut data, "fastai.layers", "AdaptiveConcatPool2d");
    data.extend_from_slice(&[b')', 0x81, b'u', b'u', b'b', b'u', b'b', b'.']);
    data
}

fn pose_hrnet_data_pickle_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x02]);
    pickle_global(&mut data, "models.pose_hrnet", "PoseHighResolutionNet");
    data.extend_from_slice(&[b'q', 0, b')', 0x81, b'q', 1, b'}', b'q', 2, b'(']);
    binunicode(&mut data, "_modules");
    data.extend_from_slice(&[b'q', 3]);
    pickle_global(&mut data, "collections", "OrderedDict");
    data.extend_from_slice(b")Rq\x04(");
    for (index, name) in [
        "conv1",
        "bn1",
        "conv2",
        "bn2",
        "relu",
        "layer1",
        "transition1",
        "stage2",
        "transition2",
        "stage3",
        "transition3",
        "stage4",
        "final_layer",
    ]
    .iter()
    .enumerate()
    {
        binunicode(&mut data, name);
        data.extend_from_slice(&[b'q', 5 + (index as u8 * 2)]);
        pickle_global(&mut data, "torch.nn.modules.module", "Module");
        data.extend_from_slice(&[b')', 0x81, b'q', 6 + (index as u8 * 2)]);
    }
    data.extend_from_slice(b"uub.");
    data
}

fn zip_stored(data: &mut Vec<u8>, name: &str, content: &[u8]) {
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
}

fn tensor_pickle_fixture(count: usize, list: bool) -> Vec<u8> {
    let storage = legacy_long_storage_fixture();
    let mut data = Vec::new();
    data.extend_from_slice(&[0x80, 0x04]);
    if list {
        data.push(b']');
    }
    for _ in 0..count {
        data.push(b'B');
        data.extend_from_slice(&(storage.len() as u32).to_le_bytes());
        data.extend_from_slice(&storage);
        data.extend_from_slice(&[0x85, b'R', b'K', 0, b'K', 5, 0x85]);
    }
    data.push(b'.');
    data
}

fn legacy_long_storage_fixture() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, b'.',
    ]);
    pickle_int2(&mut data, 1001);
    data.extend_from_slice(&[0x80, 0x02, b'}', b'.']);
    data.extend_from_slice(b"\x80\x02(");
    binunicode(&mut data, "storage");
    data.extend_from_slice(b"ctorch\nLongStorage\n");
    binunicode(&mut data, "0");
    binunicode(&mut data, "cpu");
    data.extend_from_slice(&[b'K', 5, b'N', b't', b'Q', b'.']);
    data
}

fn pickle_int2(data: &mut Vec<u8>, value: u16) {
    data.extend_from_slice(&[0x80, 0x02, b'M']);
    data.extend_from_slice(&value.to_le_bytes());
    data.push(b'.');
}

fn pickle_small_int(data: &mut Vec<u8>, value: u16) {
    if let Ok(value) = u8::try_from(value) {
        data.extend_from_slice(&[b'K', value]);
    } else {
        data.push(b'M');
        data.extend_from_slice(&value.to_le_bytes());
    }
}

fn pickle_global(data: &mut Vec<u8>, module: &str, name: &str) {
    data.push(b'c');
    data.extend_from_slice(module.as_bytes());
    data.push(b'\n');
    data.extend_from_slice(name.as_bytes());
    data.push(b'\n');
}

fn binunicode(data: &mut Vec<u8>, value: &str) {
    data.push(b'X');
    data.extend_from_slice(&(value.len() as u32).to_le_bytes());
    data.extend_from_slice(value.as_bytes());
}
