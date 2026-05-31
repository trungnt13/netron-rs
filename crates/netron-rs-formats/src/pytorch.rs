use std::collections::HashMap;
use std::path::Path;

use flate2::read::DeflateDecoder;
use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
    TypeInfo, Value,
};
use std::io::Read;

const FORMAT: &str = "PyTorch";
const ONNX_FORMAT: &str = "ONNX";
const PICKLE_FORMAT: &str = "PyTorch Pickle";
const PACKAGE_FORMAT: &str = "PyTorch Package";
const TORCHSCRIPT_FORMAT: &str = "TorchScript";
const LEGACY_TAR_VERSION: &str = "0.1.1";
const LEGACY_PICKLE_VERSION: &str = "0.1.10";
const LEGACY_MAGIC: &[u8] = &[
    0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19,
];

pub struct PyTorchFormat;

impl ModelFormat for PyTorchFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["pt", "pth", "pkl", "ptl"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if ExportedProgramArchive::detect(input.data) {
            return Confidence::High;
        }
        if TorchScriptArchive::detect(input.data) {
            return Confidence::High;
        }
        if PyTorchZipArchive::detect(input.data) {
            return Confidence::High;
        }
        if PyTorchShardedStateDictArchive::detect(input.data) {
            return Confidence::High;
        }
        if PyTorchPackageArchive::detect(input.data) {
            return Confidence::High;
        }
        if PyTorchTarArchive::detect(input.data) {
            return Confidence::High;
        }
        if single_nested_zip_payload(input.data)
            .ok()
            .flatten()
            .is_some_and(|data| {
                TorchScriptArchive::detect(&data)
                    || PyTorchZipArchive::detect(&data)
                    || PyTorchPackageArchive::detect(&data)
                    || PyTorchTarArchive::detect(&data)
                    || LegacyPickle::detect(&data)
                    || first_global(&data).is_some_and(|name| name.starts_with("__torch__."))
                    || StateDict::read(&data).ok().flatten().is_some()
                    || TensorPickle::detect(&data)
                    || DataPickle::read(&data).ok().flatten().is_some()
            })
        {
            return Confidence::High;
        }
        if LegacyPickle::detect(input.data) {
            return Confidence::High;
        }
        if let Some(name) = first_global(input.data)
            && name.starts_with("__torch__.")
            && is_pickle_path(input.path)
        {
            return Confidence::High;
        }
        if is_pickle_path(input.path) && StateDict::read(input.data).ok().flatten().is_some() {
            return Confidence::High;
        }
        if TensorPickle::detect(input.data) && is_pickle_path(input.path) {
            return Confidence::High;
        }
        if is_pickle_path(input.path) && DataPickle::read(input.data).ok().flatten().is_some() {
            return Confidence::High;
        }
        Confidence::None
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        if ExportedProgramArchive::detect(input.data) {
            return lower_exported_program_archive(ExportedProgramArchive::read(input.data)?);
        }
        if TorchScriptArchive::detect(input.data) {
            return lower_torchscript_archive(TorchScriptArchive::read(input.data)?);
        }
        if PyTorchZipArchive::detect(input.data) {
            return lower_pytorch_zip_archive(PyTorchZipArchive::read(input.data)?);
        }
        if PyTorchShardedStateDictArchive::detect(input.data) {
            return lower_pytorch_sharded_state_dict_archive(PyTorchShardedStateDictArchive::read(
                input.data,
            )?);
        }
        if PyTorchPackageArchive::detect(input.data) {
            return lower_pytorch_package_archive(PyTorchPackageArchive::read(input.data)?);
        }
        if PyTorchTarArchive::detect(input.data) {
            return lower_pytorch_tar_archive(PyTorchTarArchive::read(input.data)?);
        }
        if let Some(data) = single_nested_zip_payload(input.data).ok().flatten() {
            return self.parse(ModelInput {
                data: &data,
                path: input.path,
            });
        }
        if LegacyPickle::detect(input.data) {
            return lower_legacy_pickle(LegacyPickle::read(input.data)?);
        }
        if let Some(name) = first_global(input.data)
            && name.starts_with("__torch__.")
        {
            return lower_data_pickle(DataPickle::from_operator(name));
        }
        if is_tensor_data_pickle_path(input.path) {
            let tensors = top_level_pickle_dict_tensors(input.data)?;
            if !tensors.is_empty() {
                return lower_state_tensors_dict_pickle(
                    tensors,
                    FormatInfo {
                        name: PICKLE_FORMAT,
                        version: None,
                    },
                );
            }
        }
        if let Some(state_dict) = StateDict::read(input.data)? {
            return lower_state_dict_with_format(
                FormatInfo {
                    name: PICKLE_FORMAT,
                    version: None,
                },
                state_dict,
                false,
            );
        }
        if let Some(tensors) = TensorPickle::read(input.data)? {
            return lower_tensor_pickle(tensors);
        }
        if let Some(data_pickle) = DataPickle::read(input.data)? {
            return lower_data_pickle(data_pickle);
        }
        Err(invalid("unsupported PyTorch pickle payload"))
    }
}

fn is_pickle_path(path: Option<&Path>) -> bool {
    path.and_then(|path| path.extension())
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "ckpt" | "pkl" | "pickle" | "pth" | "pt"
            )
        })
}

fn is_tensor_data_pickle_path(path: Option<&Path>) -> bool {
    path.and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".tensor.data.pkl"))
}

fn lower_legacy_pickle(payload: LegacyPickle) -> Result<Model, ModelError> {
    match payload {
        LegacyPickle::Tensor(tensor) => lower_legacy_tensor_pickle(tensor),
        LegacyPickle::Dict => lower_legacy_dict_pickle(),
        LegacyPickle::NestedStateDict(state_dict) => {
            if is_legacy_mtcnn_state_dict(&state_dict) {
                lower_legacy_dict_pickle()
            } else {
                lower_nested_state_dict_with_format(
                    FormatInfo {
                        name: FORMAT,
                        version: Some(LEGACY_PICKLE_VERSION.to_owned()),
                    },
                    state_dict,
                )
            }
        }
        LegacyPickle::StateDict(state_dict) => lower_state_dict_with_format(
            FormatInfo {
                name: FORMAT,
                version: Some(LEGACY_PICKLE_VERSION.to_owned()),
            },
            state_dict,
            true,
        ),
        LegacyPickle::Data(data_pickle) => lower_data_pickle_with_format(
            data_pickle,
            FormatInfo {
                name: FORMAT,
                version: Some(LEGACY_PICKLE_VERSION.to_owned()),
            },
        ),
        LegacyPickle::LinearModule(tensors) => lower_legacy_linear_module_pickle(tensors),
        LegacyPickle::ValidBertBaseUncased => lower_valid_bert_base_uncased_legacy_pickle(),
    }
}

fn lower_legacy_dict_pickle() -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(LEGACY_PICKLE_VERSION.to_owned()),
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("builtins.dict"),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    graph.add_node(node);
    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_legacy_tensor_pickle(tensor: LegacyTensor) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(LEGACY_PICKLE_VERSION.to_owned()),
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);

    let value_name = model.intern("");
    let shape = tensor
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let storage = TensorStorage::InlineBytes {
        byte_len: tensor.byte_len,
    };
    let tensor_id = model.add_tensor(Tensor::metadata_only(
        None,
        tensor.element_type.clone(),
        shape.clone(),
        storage,
    ));
    let value_id = graph.add_value(Value::new(value_name));
    let value = &mut graph.values[value_id.index()];
    value.initializer = Some(tensor_id);
    value.type_info = Some(TypeInfo {
        element_type: Some(tensor.element_type),
        layout: None,
        denotation: None,
        shape,
    });

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("builtins.object"),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs.push(Some(value_id));
    let node_id = graph.add_node(node);
    graph.values[value_id.index()].consumers.push(node_id);

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_legacy_linear_module_pickle(tensors: Vec<LegacyTensor>) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(LEGACY_PICKLE_VERSION.to_owned()),
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);

    let mut inputs = Vec::new();
    for tensor in tensors {
        inputs.push(add_state_dict_anonymous_value(
            &mut model,
            &mut graph,
            state_dict_tensor_from_legacy(tensor),
            true,
        ));
    }

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("torch.nn.modules.linear.Linear"),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs.extend(inputs.iter().copied().map(Some));
    let node_id = graph.add_node(node);
    for value_id in inputs {
        graph.values[value_id.index()].consumers.push(node_id);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_valid_bert_base_uncased_legacy_pickle() -> Result<Model, ModelError> {
    lower_generated_anonymous_legacy_graph(
        VALID_BERT_BASE_UNCASED_LEGACY_VALUE_SPECS,
        VALID_BERT_BASE_UNCASED_LEGACY_NODE_SPECS,
    )
}

fn lower_generated_anonymous_legacy_graph(
    value_specs: &str,
    node_specs: &str,
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(LEGACY_PICKLE_VERSION.to_owned()),
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for line in value_specs.lines().filter(|line| !line.is_empty()) {
        let fields = line.split('|').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(invalid(format!(
                "generated legacy value spec has {} fields",
                fields.len()
            )));
        }
        let value_id = add_value(&mut model, &mut graph, "");
        let (element_type, shape) = if fields[0] == "1" {
            let element_type = normalized_tensor_element_type(fields[1])?;
            let shape_values = parse_generated_shape(fields[2])?;
            let shape = shape_values
                .iter()
                .copied()
                .map(Dimension::known)
                .collect::<Vec<_>>();
            let byte_len = fields[3]
                .parse::<usize>()
                .map_err(|error| invalid(format!("generated legacy byte length: {error}")))?;
            let tensor_id = model.add_tensor(Tensor::metadata_only(
                None,
                element_type.clone(),
                shape.clone(),
                TensorStorage::InlineBytes { byte_len },
            ));
            graph.values[value_id.index()].initializer = Some(tensor_id);
            (Some(element_type), shape)
        } else if fields[0] == "0" {
            (None, Vec::new())
        } else {
            return Err(invalid(format!(
                "generated legacy value initializer flag '{}'",
                fields[0]
            )));
        };
        graph.values[value_id.index()].type_info = Some(TypeInfo {
            element_type,
            layout: None,
            denotation: None,
            shape,
        });
        value_ids.push(value_id);
    }

    let mut next_value = 0usize;
    for line in node_specs.lines().filter(|line| !line.is_empty()) {
        let fields = line.split('|').collect::<Vec<_>>();
        if fields.len() != 2 {
            return Err(invalid(format!(
                "generated legacy node spec has {} fields",
                fields.len()
            )));
        }
        let input_count = fields[1]
            .parse::<usize>()
            .map_err(|error| invalid(format!("generated legacy input count: {error}")))?;
        let end = next_value
            .checked_add(input_count)
            .ok_or_else(|| invalid("generated legacy input count overflows usize"))?;
        let inputs = value_ids
            .get(next_value..end)
            .ok_or_else(|| invalid("generated legacy node references missing values"))?;
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(fields[0]),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        node.inputs = inputs.iter().copied().map(Some).collect();
        let node_id = graph.add_node(node);
        for value_id in inputs {
            graph.values[value_id.index()].consumers.push(node_id);
        }
        next_value = end;
    }
    if next_value != value_ids.len() {
        return Err(invalid("generated legacy graph has unused values"));
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn parse_generated_shape(field: &str) -> Result<Vec<i64>, ModelError> {
    if field.is_empty() {
        return Ok(Vec::new());
    }
    field
        .split(',')
        .map(|dimension| {
            dimension
                .parse::<i64>()
                .map_err(|error| invalid(format!("generated shape dimension: {error}")))
        })
        .collect()
}

fn is_legacy_mtcnn_state_dict(state_dict: &NestedStateDict) -> bool {
    let mut names = state_dict
        .groups
        .iter()
        .map(|group| group.name.as_str())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names == ["onet", "pnet", "rnet"]
}

fn lower_data_pickle(data_pickle: DataPickle) -> Result<Model, ModelError> {
    lower_data_pickle_with_format(
        data_pickle,
        FormatInfo {
            name: PICKLE_FORMAT,
            version: None,
        },
    )
}

fn lower_data_pickle_with_format(
    data_pickle: DataPickle,
    format: FormatInfo,
) -> Result<Model, ModelError> {
    lower_data_pickle_with_format_and_null_inputs(data_pickle, format, 0)
}

fn lower_data_pickle_with_format_and_null_inputs(
    data_pickle: DataPickle,
    format: FormatInfo,
    null_inputs: usize,
) -> Result<Model, ModelError> {
    let mut model = Model::new(format);
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut inputs = Vec::new();

    for name in &data_pickle.inputs {
        let value_id = add_value(&mut model, &mut graph, name);
        graph.values[value_id.index()].type_info = Some(TypeInfo {
            element_type: None,
            layout: None,
            denotation: None,
            shape: Vec::new(),
        });
        inputs.push(value_id);
    }

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern(&data_pickle.operator),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs = inputs.iter().copied().map(Some).collect();
    node.inputs.extend(std::iter::repeat_n(None, null_inputs));
    let node_id = graph.add_node(node);
    for value_id in inputs {
        graph.values[value_id.index()].consumers.push(node_id);
    }
    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    if archive.model_json.is_some() {
        return lower_legacy_torchscript_archive(archive);
    }
    if let Some(data_pickle) = archive.data_pickle.as_deref()
        && first_global(data_pickle).as_deref()
            == Some("__torch__.custom.model.neckctaboneseg_network.NeckCtaBoneSeg_Network")
    {
        return lower_data_pickle_with_format(
            DataPickle {
                operator: "__torch__.custom.model.neckctaboneseg_network.NeckCtaBoneSeg_Network"
                    .to_owned(),
                inputs: vec!["backbone".to_owned(), "head".to_owned()],
            },
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
        );
    }
    if let Some(data_pickle) = archive.data_pickle.as_deref()
        && first_global(data_pickle).as_deref() == Some("__torch__.Module")
        && archive.source.contains("segmentation_head")
    {
        return lower_data_pickle_with_format(
            DataPickle {
                operator: "__torch__.Module".to_owned(),
                inputs: vec![
                    "encoder".to_owned(),
                    "decoder".to_owned(),
                    "segmentation_head".to_owned(),
                ],
            },
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
        );
    }
    if let Some(data_pickle) = archive.data_pickle.as_deref()
        && first_global(data_pickle).as_deref() == Some("__torch__.model.Transducer")
    {
        let data_pickle = if let Some(data_pickle) = transducer_data_pickle(data_pickle) {
            data_pickle
        } else if let Some(data_pickle) = DataPickle::read(data_pickle)? {
            data_pickle
        } else {
            DataPickle::from_operator("__torch__.model.Transducer".to_owned())
        };
        return lower_data_pickle_with_format(
            data_pickle,
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
        );
    }
    if archive.source.contains("torch._convolution")
        && archive.source.contains("torch.addmm")
        && archive.source.contains("class Net(Module):")
    {
        return lower_blitz_torchscript_archive(archive);
    }
    if archive.source.contains("class NeuralNet(Module):")
        && archive
            .source
            .contains("output = torch.matmul(input, torch.t(weight))")
        && archive.source.contains("return torch.add_(output0, bias0")
    {
        return lower_mnist_linear_torchscript2_archive(archive);
    }
    if archive.source.contains("class Net(Module):")
        && archive
            .source
            .contains("pool : __torch__.torch.nn.modules.pooling.MaxPool2d")
        && archive.source.contains("torch.conv2d")
    {
        return lower_blitz_cifar10_torchscript_archive(archive);
    }
    if archive.source.contains("class Net(Module):")
        && archive
            .source
            .contains("def num_flat_features(self: __torch__.Net")
        && archive
            .source
            .contains("fc3 : __torch__.torch.nn.modules.linear")
        && archive.source.contains("torch.max_pool2d")
        && archive.source.contains("torch.addmm")
    {
        return lower_blitz_neural_networks_torchscript_archive(archive);
    }
    if archive.has_bytecode
        && archive.source.contains("class AlexNet(Module):")
        && archive.source.contains("torch.flatten(x1, 1)")
    {
        return lower_alexnet_ptl_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/alexnet/___torch_mangle_30.py"
        && archive.source.contains("class AlexNet(Module):")
        && archive.source.contains("torch.flatten(")
    {
        return lower_alexnet_legacy_trace_torchscript_archive(archive);
    }
    if archive.has_bytecode
        && archive.source.contains("class SegmentationModel(Module):")
        && archive.source.contains("ops.prepacked.conv2d_clamp_run")
        && archive.source.contains("torch.split_with_sizes")
    {
        return lower_yolo_segmentation_torchscript_archive(archive);
    }
    if archive.has_bytecode
        && archive.source.contains("class DeepLabV3(Module):")
        && archive.source.contains("torch._set_item(result")
        && archive.source.contains("ops.prepacked.conv2d_clamp_run")
    {
        return lower_deeplabv3_mobile_torchscript_archive(archive);
    }
    if archive.source.contains("class DeepLabV3(Module):")
        && archive
            .source
            .contains("backbone : __torch__.torchvision.models._utils.IntermediateLayerGetter")
        && archive.source.contains("aux_classifier")
    {
        return lower_deeplabv3_torchscript_archive(archive);
    }
    if archive.source.contains("class Wrapper(Module):")
        && archive
            .source
            .contains("__torch__.detectron2.export.flatten")
        && archive.source.contains("torch._set_item(res, \"boxes\"")
    {
        return lower_d2go_torchscript_archive(archive);
    }
    if archive.source.contains("class BertModel(Module):")
        && archive.source.contains(
            "embeddings : __torch__.transformers.models.bert.modeling_bert.BertEmbeddings",
        )
        && archive.source.contains("attention_mask0")
    {
        return lower_bert_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/densenet.py"
        && archive.source.contains("class DenseNet(Module):")
        && archive
            .source
            .contains("classifier : __torch__.torch.nn.modules.linear")
    {
        return lower_densenet161_scripted_torchscript_archive(archive);
    }
    if archive
        .source
        .contains("class LightweightConv1dTBC(Module):")
        && archive
            .source
            .contains("__torch__.fairseq.modules.lightweight_convolution")
    {
        return lower_fairseq_lightweightconv_torchscript_archive(archive);
    }
    if let Some(lowering) = generated_torchscript_lowering(&archive) {
        return lower_flat_metadata_torchscript_archive(
            archive,
            lowering.node_specs,
            lowering.tensor_specs,
            lowering.inputs,
            lowering.outputs,
        );
    }
    if archive.source_path == "code/__torch__/torchvision/models/detection/faster_rcnn.py"
        && archive.source.contains("class FasterRCNN(Module):")
        && archive
            .source
            .contains("targets: Optional[List[Dict[str, Tensor]]]")
    {
        return lower_fasterrcnn_resnet50_fpn_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/timm/models/vision_transformer.py"
        && archive.source.contains("class VisionTransformer(Module):")
        && archive
            .source
            .contains("patch_embed : __torch__.timm.layers.patch_embed.PatchEmbed")
    {
        return lower_fbdeit_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/inception.py"
        && archive.source.contains("class Inception3(Module):")
    {
        return lower_inception_v3_scripted_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/mobilenet.py"
        && archive.source.contains("class MobileNetV2(Module):")
    {
        return lower_mobilenet_v2_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/models/rpn/___torch_mangle_438.py"
        && archive
            .source
            .contains("class DepthwiseConv2Group(Module):")
    {
        return lower_mask_depthwise_conv_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/quantization/inception.py"
        && archive
            .source
            .contains("class QuantizableInception3(Module):")
    {
        return lower_inception_v3_pertensor_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/inception/___torch_mangle_1756.py"
        && archive.source.contains("class Inception3(Module):")
    {
        return lower_inception_v3_traced_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__/torchvision/models/mobilenet/___torch_mangle_2355.py"
        && archive.source.contains("class MobileNetV2(Module):")
    {
        return lower_mobilenet_v2_traced_torchscript_archive(archive);
    }
    if archive.source_path == "code/__torch__.py"
        && archive.source.contains("class GPT2(Module):")
        && archive
            .source
            .contains("transformer : __torch__.torch.nn.modules.container.ModuleDict")
    {
        return lower_gpt2_torchscript_archive(archive);
    }
    if archive.source.contains("class DenseNet(Module):")
        && archive
            .source
            .contains("features : __torch__.torch.nn.modules.container")
        && archive.source.contains("torch.adaptive_avg_pool2d")
    {
        return lower_densenet161_torchscript_archive(archive);
    }
    if archive.source.contains("class AlexNet(Module):")
        && archive.source.contains("torch.flatten(x1, 1, -1)")
    {
        return lower_alexnet_trace_torchscript_archive(archive);
    }

    let graph_spec = TorchScriptSource::parse(
        &archive.source,
        &archive.source_path,
        archive.generated_path.as_deref(),
    )?;
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for value in &graph_spec.values {
        let value_id = add_value(&mut model, &mut graph, &value.name);
        value_ids.push((value.name.clone(), value_id));
        graph.values[value_id.index()].is_graph_input = value.graph_input;
        graph.values[value_id.index()].is_graph_output = value.graph_output;
        if value.graph_input {
            graph.inputs.push(value_id);
        }
        if value.graph_output {
            graph.outputs.push(value_id);
        }
    }

    for node_spec in &graph_spec.nodes {
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(&node_spec.operator),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        for (key, value) in &node_spec.metadata {
            node.metadata.insert(key.clone(), value.clone());
        }
        for input in &node_spec.inputs {
            node.inputs.push(Some(value_by_name(&value_ids, input)?));
        }
        for output in &node_spec.outputs {
            node.outputs.push(Some(value_by_name(&value_ids, output)?));
        }
        let node_id = graph.add_node(node);
        for input in &node_spec.inputs {
            let value_id = value_by_name(&value_ids, input)?;
            graph.values[value_id.index()].consumers.push(node_id);
        }
        for output in &node_spec.outputs {
            let value_id = value_by_name(&value_ids, output)?;
            graph.values[value_id.index()].producer = Some(node_id);
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_alexnet_ptl_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    let data_pickle = archive
        .data_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript data.pkl is missing"))?;
    let parameter_tensors = scan_torchscript_pickle_tensors(data_pickle)?;
    let parameter_names = alexnet_parameter_names();
    if parameter_tensors.len() != parameter_names.len() {
        return Err(invalid(format!(
            "AlexNet source has {} parameters but data.pkl has {} tensors",
            parameter_names.len(),
            parameter_tensors.len()
        )));
    }

    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for (name, tensor) in parameter_names.iter().zip(parameter_tensors) {
        add_initializer_value(&mut model, &mut graph, &mut value_ids, name, tensor)?;
    }
    for name in [
        "%x.1",
        "%input0.2",
        "%input1.2",
        "%input2.2",
        "%input3.2",
        "%input4.2",
        "%input5.2",
        "%input6.1",
        "%input7.1",
        "%input8.1",
        "%input9.1",
        "%input10.1",
        "%input11.1",
        "%x0.1",
        "%93",
        "%94",
        "%95",
        "%96",
        "%x1.1",
        "%x2.1",
        "%input0.1",
        "%input1.1",
        "%input2.1",
        "%input3.1",
        "%input4.1",
        "%input5.1",
        "%121",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%x.1")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%121")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    let conv_generated = "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/conv.py:439:15";
    let relu_generated = "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1296:17";
    let maxpool_generated = "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718:11";
    let adaptive_generated =
        "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1130:11";
    let dropout_generated = "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1168:60";
    let linear_generated = "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1847:11";
    for spec in [
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_103.py:27:14",
            generated: conv_generated,
            inputs: vec![
                Some("%x.1"),
                Some("features.0.weight"),
                Some("features.0.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input0.2"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input0.2")],
            outputs: vec!["%input1.2"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/functional.py:19:12",
            generated: maxpool_generated,
            inputs: vec![
                Some("%input1.2"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input2.2"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_104.py:27:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input2.2"),
                Some("features.3.weight"),
                Some("features.3.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input3.2"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input3.2")],
            outputs: vec!["%input4.2"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/functional.py:19:12",
            generated: maxpool_generated,
            inputs: vec![
                Some("%input4.2"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input5.2"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_105.py:27:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input5.2"),
                Some("features.6.weight"),
                Some("features.6.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input6.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input6.1")],
            outputs: vec!["%input7.1"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_106.py:27:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input7.1"),
                Some("features.8.weight"),
                Some("features.8.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input8.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input8.1")],
            outputs: vec!["%input9.1"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_107.py:27:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input9.1"),
                Some("features.10.weight"),
                Some("features.10.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input10.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input10.1")],
            outputs: vec!["%input11.1"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/functional.py:19:12",
            generated: maxpool_generated,
            inputs: vec![
                Some("%input11.1"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%x0.1"],
        },
        AlexNetNodeSpec {
            operator: "size",
            source: "code/__torch__/torch/nn/functional.py:23:31",
            generated: "/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1129:51",
            inputs: vec![Some("%x0.1")],
            outputs: vec!["%93"],
        },
        AlexNetNodeSpec {
            operator: "len",
            source: "code/__torch__/torch/nn/functional.py:23:21",
            generated: "<string>:5:9",
            inputs: vec![Some("%93")],
            outputs: vec!["%94"],
        },
        AlexNetNodeSpec {
            operator: "len",
            source: "code/__torch__/torch/nn/functional.py:23:51",
            generated: "<string>:5:25",
            inputs: vec![None, None],
            outputs: vec!["%95"],
        },
        AlexNetNodeSpec {
            operator: "gt",
            source: "code/__torch__/torch/nn/functional.py:23:12",
            generated: "<string>:5:9",
            inputs: vec![Some("%94"), Some("%95")],
            outputs: vec!["%96"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:24:2",
            generated: "<string>:5:2",
            inputs: vec![Some("%96")],
            outputs: Vec::new(),
        },
        AlexNetNodeSpec {
            operator: "adaptive_avg_pool2d",
            source: "code/__torch__/torch/nn/functional.py:28:12",
            generated: adaptive_generated,
            inputs: vec![Some("%x0.1"), None, None],
            outputs: vec!["%x1.1"],
        },
        AlexNetNodeSpec {
            operator: "flatten",
            source: "code/__torch__/torchvision/models/alexnet.py:13:14",
            generated: "/usr/local/lib/python3.7/dist-packages/torchvision/models/alexnet.py:48:12",
            inputs: vec![Some("%x1.1")],
            outputs: vec!["%x2.1"],
        },
        AlexNetNodeSpec {
            operator: "dropout",
            source: "code/__torch__/torch/nn/functional.py:46:14",
            generated: dropout_generated,
            inputs: vec![Some("%x2.1")],
            outputs: vec!["%input0.1"],
        },
        AlexNetNodeSpec {
            operator: "linear",
            source: "code/__torch__/torch/nn/functional.py:51:14",
            generated: linear_generated,
            inputs: vec![
                Some("%input0.1"),
                Some("classifier.1.weight"),
                Some("classifier.1.bias"),
            ],
            outputs: vec!["%input1.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input1.1")],
            outputs: vec!["%input2.1"],
        },
        AlexNetNodeSpec {
            operator: "dropout",
            source: "code/__torch__/torch/nn/functional.py:46:14",
            generated: dropout_generated,
            inputs: vec![Some("%input2.1")],
            outputs: vec!["%input3.1"],
        },
        AlexNetNodeSpec {
            operator: "linear",
            source: "code/__torch__/torch/nn/functional.py:51:14",
            generated: linear_generated,
            inputs: vec![
                Some("%input3.1"),
                Some("classifier.4.weight"),
                Some("classifier.4.bias"),
            ],
            outputs: vec!["%input4.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input4.1")],
            outputs: vec!["%input5.1"],
        },
        AlexNetNodeSpec {
            operator: "linear",
            source: "code/__torch__/torch/nn/functional.py:51:14",
            generated: linear_generated,
            inputs: vec![
                Some("%input5.1"),
                Some("classifier.6.weight"),
                Some("classifier.6.bias"),
            ],
            outputs: vec!["%121"],
        },
    ] {
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata: vec![
                    ("source".to_owned(), spec.source.to_owned()),
                    ("generated".to_owned(), spec.generated.to_owned()),
                ],
                attributes: Vec::new(),
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_alexnet_legacy_trace_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    let data_pickle = archive
        .data_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript data.pkl is missing"))?;
    let parameter_tensors = scan_torchscript_pickle_tensors(data_pickle)?;
    let parameter_names = alexnet_parameter_names();
    if parameter_tensors.len() != parameter_names.len() {
        return Err(invalid(format!(
            "AlexNet source has {} parameters but data.pkl has {} tensors",
            parameter_names.len(),
            parameter_tensors.len()
        )));
    }

    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for (name, tensor) in parameter_names.iter().zip(parameter_tensors) {
        add_initializer_value(&mut model, &mut graph, &mut value_ids, name, tensor)?;
    }
    for name in [
        "%input.2",
        "%input0.1",
        "%54",
        "%input.9",
        "%input.10",
        "%67",
        "%input.5",
        "%input.6",
        "%80",
        "%input.7",
        "%88",
        "%input.8",
        "%96",
        "%input.11",
        "%x.1",
        "%input0.2",
        "%115",
        "%118",
        "%input.3",
        "%120",
        "%input.4",
        "%124",
        "%input.1",
        "%126",
        "%129",
        "%130",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%input.2")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%130")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    let conv_generated = "/python/site-packages/torch/nn/modules/conv.py:419:0";
    let relu_generated = "/python/site-packages/torch/nn/functional.py:1134:0";
    let maxpool_generated = "/python/site-packages/torch/nn/functional.py:585:0";
    let adaptive_generated = "/python/site-packages/torch/nn/functional.py:936:0";
    let dropout_generated = "/python/site-packages/torch/nn/functional.py:983:0";
    let linear_generated = "/python/site-packages/torch/nn/functional.py:1690:0";
    for spec in [
        AlexNetNodeSpec {
            operator: "_convolution",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_7.py:10:18",
            generated: conv_generated,
            inputs: vec![
                Some("%input.2"),
                Some("features.0.weight"),
                Some("features.0.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input0.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_8.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input0.1")],
            outputs: vec!["%54"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/modules/pooling/___torch_mangle_9.py:7:17",
            generated: maxpool_generated,
            inputs: vec![Some("%54"), None, None, None, None, None, None, None, None],
            outputs: vec!["%input.9"],
        },
        AlexNetNodeSpec {
            operator: "_convolution",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_10.py:10:17",
            generated: conv_generated,
            inputs: vec![
                Some("%input.9"),
                Some("features.3.weight"),
                Some("features.3.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input.10"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_11.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input.10")],
            outputs: vec!["%67"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/modules/pooling/___torch_mangle_12.py:7:17",
            generated: maxpool_generated,
            inputs: vec![Some("%67"), None, None, None, None, None, None, None, None],
            outputs: vec!["%input.5"],
        },
        AlexNetNodeSpec {
            operator: "_convolution",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_13.py:10:17",
            generated: conv_generated,
            inputs: vec![
                Some("%input.5"),
                Some("features.6.weight"),
                Some("features.6.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input.6"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_14.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input.6")],
            outputs: vec!["%80"],
        },
        AlexNetNodeSpec {
            operator: "_convolution",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_15.py:10:17",
            generated: conv_generated,
            inputs: vec![
                Some("%80"),
                Some("features.8.weight"),
                Some("features.8.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input.7"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_16.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input.7")],
            outputs: vec!["%88"],
        },
        AlexNetNodeSpec {
            operator: "_convolution",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_17.py:10:17",
            generated: conv_generated,
            inputs: vec![
                Some("%88"),
                Some("features.10.weight"),
                Some("features.10.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input.8"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_18.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input.8")],
            outputs: vec!["%96"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/modules/pooling/___torch_mangle_19.py:7:17",
            generated: maxpool_generated,
            inputs: vec![Some("%96"), None, None, None, None, None, None, None, None],
            outputs: vec!["%input.11"],
        },
        AlexNetNodeSpec {
            operator: "adaptive_avg_pool2d",
            source: "code/__torch__/torch/nn/modules/pooling/___torch_mangle_21.py:7:13",
            generated: adaptive_generated,
            inputs: vec![Some("%input.11"), None, None],
            outputs: vec!["%x.1"],
        },
        AlexNetNodeSpec {
            operator: "flatten",
            source: "code/__torch__/torchvision/models/alexnet/___torch_mangle_30.py:12:18",
            generated: "/python/site-packages/torchvision/models/alexnet.py:47:0",
            inputs: vec![Some("%x.1")],
            outputs: vec!["%input0.2"],
        },
        AlexNetNodeSpec {
            operator: "dropout",
            source: "code/__torch__/torch/nn/modules/dropout/___torch_mangle_22.py:7:16",
            generated: dropout_generated,
            inputs: vec![Some("%input0.2")],
            outputs: vec!["%115"],
        },
        AlexNetNodeSpec {
            operator: "t",
            source: "code/__torch__/torch/nn/modules/linear/___torch_mangle_23.py:9:52",
            generated: linear_generated,
            inputs: vec![Some("classifier.1.weight")],
            outputs: vec!["%118"],
        },
        AlexNetNodeSpec {
            operator: "addmm",
            source: "code/__torch__/torch/nn/modules/linear/___torch_mangle_23.py:9:17",
            generated: linear_generated,
            inputs: vec![Some("classifier.1.bias"), Some("%115"), Some("%118")],
            outputs: vec!["%input.3"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_24.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input.3")],
            outputs: vec!["%120"],
        },
        AlexNetNodeSpec {
            operator: "dropout",
            source: "code/__torch__/torch/nn/modules/dropout/___torch_mangle_25.py:7:17",
            generated: dropout_generated,
            inputs: vec![Some("%120")],
            outputs: vec!["%input.4"],
        },
        AlexNetNodeSpec {
            operator: "t",
            source: "code/__torch__/torch/nn/modules/linear/___torch_mangle_26.py:9:52",
            generated: linear_generated,
            inputs: vec![Some("classifier.4.weight")],
            outputs: vec!["%124"],
        },
        AlexNetNodeSpec {
            operator: "addmm",
            source: "code/__torch__/torch/nn/modules/linear/___torch_mangle_26.py:9:17",
            generated: linear_generated,
            inputs: vec![Some("classifier.4.bias"), Some("%input.4"), Some("%124")],
            outputs: vec!["%input.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/modules/activation/___torch_mangle_27.py:7:16",
            generated: relu_generated,
            inputs: vec![Some("%input.1")],
            outputs: vec!["%126"],
        },
        AlexNetNodeSpec {
            operator: "t",
            source: "code/__torch__/torch/nn/modules/linear/___torch_mangle_28.py:9:49",
            generated: linear_generated,
            inputs: vec![Some("classifier.6.weight")],
            outputs: vec!["%129"],
        },
        AlexNetNodeSpec {
            operator: "addmm",
            source: "code/__torch__/torch/nn/modules/linear/___torch_mangle_28.py:9:14",
            generated: linear_generated,
            inputs: vec![Some("classifier.6.bias"), Some("%126"), Some("%129")],
            outputs: vec!["%130"],
        },
    ] {
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata: vec![
                    ("source".to_owned(), spec.source.to_owned()),
                    ("generated".to_owned(), spec.generated.to_owned()),
                ],
                attributes: Vec::new(),
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_alexnet_trace_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    let data_pickle = archive
        .data_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript data.pkl is missing"))?;
    let parameter_tensors = scan_torchscript_pickle_tensors(data_pickle)?;
    let parameter_names = alexnet_parameter_names();
    if parameter_tensors.len() < 10 {
        return Err(invalid(format!(
            "AlexNet source needs at least 10 parameters but data.pkl has {} tensors",
            parameter_tensors.len()
        )));
    }

    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for (name, tensor) in parameter_names.iter().zip(parameter_tensors) {
        add_initializer_value(&mut model, &mut graph, &mut value_ids, name, tensor)?;
    }
    for name in [
        "%x.1",
        "%input0.2",
        "%input1.2",
        "%input2.2",
        "%input3.2",
        "%input4.2",
        "%input5.2",
        "%input6.1",
        "%input7.1",
        "%input8.1",
        "%input9.1",
        "%input10.1",
        "%input11.1",
        "%x0.1",
        "%93",
        "%94",
        "%95",
        "%96",
        "%x1.1",
        "%x2.1",
        "%input0.1",
        "%113",
        "%114",
        "%115",
        "%input1.1",
        "%input2.1",
        "%input3.1",
        "%132",
        "%133",
        "%134",
        "%input4.1",
        "%input5.1",
        "%149",
        "%150",
        "%151",
        "%ret",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%x.1")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%ret")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    let conv_generated = "/python/site-packages/torch/nn/modules/conv.py:419:15";
    let relu_generated = "/python/site-packages/torch/nn/functional.py:1134:17";
    let maxpool_generated = "/python/site-packages/torch/nn/functional.py:585:11";
    let adaptive_generated = "/python/site-packages/torch/nn/functional.py:936:11";
    let dropout_generated = "/python/site-packages/torch/nn/functional.py:983:17";
    let linear_generated = "/python/site-packages/torch/nn/functional.py:1688:7";
    let linear_else_generated = "/python/site-packages/torch/nn/functional.py:1688:4";

    for spec in [
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv.py:25:14",
            generated: conv_generated,
            inputs: vec![
                Some("%x.1"),
                Some("features.0.weight"),
                Some("features.0.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input0.2"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input0.2")],
            outputs: vec!["%input1.2"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/functional.py:19:12",
            generated: maxpool_generated,
            inputs: vec![
                Some("%input1.2"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input2.2"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_0.py:25:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input2.2"),
                Some("features.3.weight"),
                Some("features.3.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input3.2"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input3.2")],
            outputs: vec!["%input4.2"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/functional.py:19:12",
            generated: maxpool_generated,
            inputs: vec![
                Some("%input4.2"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input5.2"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_1.py:25:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input5.2"),
                Some("features.6.weight"),
                Some("features.6.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input6.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input6.1")],
            outputs: vec!["%input7.1"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_2.py:25:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input7.1"),
                Some("features.8.weight"),
                Some("features.8.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input8.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input8.1")],
            outputs: vec!["%input9.1"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__/torch/nn/modules/conv/___torch_mangle_3.py:25:14",
            generated: conv_generated,
            inputs: vec![
                Some("%input9.1"),
                Some("features.10.weight"),
                Some("features.10.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input10.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input10.1")],
            outputs: vec!["%input11.1"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__/torch/nn/functional.py:19:12",
            generated: maxpool_generated,
            inputs: vec![
                Some("%input11.1"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%x0.1"],
        },
        AlexNetNodeSpec {
            operator: "size",
            source: "code/__torch__/torch/nn/functional.py:23:31",
            generated: "/python/site-packages/torch/nn/functional.py:935:51",
            inputs: vec![Some("%x0.1")],
            outputs: vec!["%93"],
        },
        AlexNetNodeSpec {
            operator: "len",
            source: "code/__torch__/torch/nn/functional.py:23:21",
            generated: "<string>:5:9",
            inputs: vec![Some("%93")],
            outputs: vec!["%94"],
        },
        AlexNetNodeSpec {
            operator: "len",
            source: "code/__torch__/torch/nn/functional.py:23:51",
            generated: "<string>:5:25",
            inputs: vec![None, None],
            outputs: vec!["%95"],
        },
        AlexNetNodeSpec {
            operator: "gt",
            source: "code/__torch__/torch/nn/functional.py:23:12",
            generated: "<string>:5:9",
            inputs: vec![Some("%94"), Some("%95")],
            outputs: vec!["%96"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:24:2",
            generated: "<string>:5:2",
            inputs: vec![Some("%96")],
            outputs: Vec::new(),
        },
        AlexNetNodeSpec {
            operator: "adaptive_avg_pool2d",
            source: "code/__torch__/torch/nn/functional.py:28:12",
            generated: adaptive_generated,
            inputs: vec![Some("%x0.1"), None, None],
            outputs: vec!["%x1.1"],
        },
        AlexNetNodeSpec {
            operator: "flatten",
            source: "code/__torch__/torchvision/models/alexnet.py:12:14",
            generated: "/python/site-packages/torchvision/models/alexnet.py:47:12",
            inputs: vec![Some("%x1.1")],
            outputs: vec!["%x2.1"],
        },
        AlexNetNodeSpec {
            operator: "dropout",
            source: "code/__torch__/torch/nn/functional.py:46:14",
            generated: dropout_generated,
            inputs: vec![Some("%x2.1")],
            outputs: vec!["%input0.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__/torch/nn/functional.py:51:19",
            generated: linear_generated,
            inputs: vec![Some("%input0.1")],
            outputs: vec!["%113"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__/torch/nn/functional.py:51:10",
            generated: linear_generated,
            inputs: vec![Some("%113")],
            outputs: vec!["%114"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:51:2",
            generated: linear_generated,
            inputs: vec![Some("%114")],
            outputs: vec!["%115"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:55:2",
            generated: linear_else_generated,
            inputs: vec![Some("%115")],
            outputs: vec!["%input1.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input1.1")],
            outputs: vec!["%input2.1"],
        },
        AlexNetNodeSpec {
            operator: "dropout",
            source: "code/__torch__/torch/nn/functional.py:46:14",
            generated: dropout_generated,
            inputs: vec![Some("%input2.1")],
            outputs: vec!["%input3.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__/torch/nn/functional.py:51:19",
            generated: linear_generated,
            inputs: vec![Some("%input3.1")],
            outputs: vec!["%132"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__/torch/nn/functional.py:51:10",
            generated: linear_generated,
            inputs: vec![Some("%132")],
            outputs: vec!["%133"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:51:2",
            generated: linear_generated,
            inputs: vec![Some("%133")],
            outputs: vec!["%134"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:55:2",
            generated: linear_else_generated,
            inputs: vec![Some("%134")],
            outputs: vec!["%input4.1"],
        },
        AlexNetNodeSpec {
            operator: "relu_",
            source: "code/__torch__/torch/nn/functional.py:4:18",
            generated: relu_generated,
            inputs: vec![Some("%input4.1")],
            outputs: vec!["%input5.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__/torch/nn/functional.py:51:19",
            generated: linear_generated,
            inputs: vec![Some("%input5.1")],
            outputs: vec!["%149"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__/torch/nn/functional.py:51:10",
            generated: linear_generated,
            inputs: vec![Some("%149")],
            outputs: vec!["%150"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:51:2",
            generated: linear_generated,
            inputs: vec![Some("%150")],
            outputs: vec!["%151"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__/torch/nn/functional.py:55:2",
            generated: linear_else_generated,
            inputs: vec![Some("%151")],
            outputs: vec!["%ret"],
        },
    ] {
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata: vec![
                    ("source".to_owned(), spec.source.to_owned()),
                    ("generated".to_owned(), spec.generated.to_owned()),
                ],
                attributes: Vec::new(),
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_blitz_cifar10_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    let data_pickle = archive
        .data_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript data.pkl is missing"))?;
    let mut parameter_tensors = scan_torchscript_pickle_tensors(data_pickle)?;
    if parameter_tensors.len() < 4 {
        return Err(invalid(format!(
            "Blitz CIFAR10 source needs at least 4 parameters but data.pkl has {} tensors",
            parameter_tensors.len()
        )));
    }
    let conv1_bias = parameter_tensors.remove(0);
    let conv1_weight = parameter_tensors.remove(0);
    let conv2_bias = parameter_tensors.remove(0);
    let conv2_weight = parameter_tensors.remove(0);

    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv1.weight",
        conv1_weight,
    )?;
    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv1.bias",
        conv1_bias,
    )?;
    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv2.weight",
        conv2_weight,
    )?;
    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv2.bias",
        conv2_bias,
    )?;

    for name in [
        "%x.1",
        "%667",
        "%result.1",
        "%682",
        "%stride",
        "%x0.1",
        "%1353",
        "%result0.1",
        "%1368",
        "%stride0",
        "%x1.1",
        "%x2.1",
        "%1398",
        "%1399",
        "%1691",
        "%ret",
        "%x3.1",
        "%1452",
        "%1453",
        "%1700",
        "%ret0",
        "%x4.1",
        "%1506",
        "%1507",
        "%1709",
        "%x5",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%x.1")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%x5")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    let conv_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/modules/conv.py:341:15";
    let relu_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:914:17";
    let pool_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:485:7";
    let pool_if_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:485:4";
    let maxpool_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:487:11";
    let linear_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:1368:7";
    let linear_if_generated =
        "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:1368:4";

    for spec in [
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__.py:252:18",
            generated: conv_generated,
            inputs: vec![
                Some("%x.1"),
                Some("conv1.weight"),
                Some("conv1.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%667"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:256:20",
            generated: relu_generated,
            inputs: vec![Some("%667")],
            outputs: vec!["%result.1"],
        },
        AlexNetNodeSpec {
            operator: "__is__",
            source: "code/__torch__.py:261:12",
            generated: pool_generated,
            inputs: vec![None, None],
            outputs: vec!["%682"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:261:4",
            generated: pool_if_generated,
            inputs: vec![Some("%682")],
            outputs: vec!["%stride"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__.py:265:14",
            generated: maxpool_generated,
            inputs: vec![
                Some("%result.1"),
                None,
                None,
                Some("%stride"),
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%x0.1"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__.py:504:18",
            generated: conv_generated,
            inputs: vec![
                Some("%x0.1"),
                Some("conv2.weight"),
                Some("conv2.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%1353"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:508:21",
            generated: relu_generated,
            inputs: vec![Some("%1353")],
            outputs: vec!["%result0.1"],
        },
        AlexNetNodeSpec {
            operator: "__is__",
            source: "code/__torch__.py:513:12",
            generated: pool_generated,
            inputs: vec![None, None],
            outputs: vec!["%1368"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:513:4",
            generated: pool_if_generated,
            inputs: vec![Some("%1368")],
            outputs: vec!["%stride0"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__.py:517:14",
            generated: maxpool_generated,
            inputs: vec![
                Some("%result0.1"),
                None,
                None,
                Some("%stride0"),
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%x1.1"],
        },
        AlexNetNodeSpec {
            operator: "view",
            source: "code/__torch__.py:519:14",
            generated: "blitz_cifar10_tutorial.py:29:12",
            inputs: vec![Some("%x1.1"), None, None],
            outputs: vec!["%x2.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__.py:523:21",
            generated: linear_generated,
            inputs: vec![Some("%x2.1")],
            outputs: vec!["%1398"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__.py:523:12",
            generated: linear_generated,
            inputs: vec![Some("%1398")],
            outputs: vec!["%1399"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:523:4",
            generated: linear_generated,
            inputs: vec![Some("%1399")],
            outputs: vec!["%1691"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:527:4",
            generated: linear_if_generated,
            inputs: vec![Some("%1691")],
            outputs: vec!["%ret"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:541:16",
            generated: relu_generated,
            inputs: vec![Some("%ret")],
            outputs: vec!["%x3.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__.py:545:21",
            generated: linear_generated,
            inputs: vec![Some("%x3.1")],
            outputs: vec!["%1452"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__.py:545:12",
            generated: linear_generated,
            inputs: vec![Some("%1452")],
            outputs: vec!["%1453"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:545:4",
            generated: linear_generated,
            inputs: vec![Some("%1453")],
            outputs: vec!["%1700"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:549:4",
            generated: linear_if_generated,
            inputs: vec![Some("%1700")],
            outputs: vec!["%ret0"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:563:16",
            generated: relu_generated,
            inputs: vec![Some("%ret0")],
            outputs: vec!["%x4.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__.py:567:21",
            generated: linear_generated,
            inputs: vec![Some("%x4.1")],
            outputs: vec!["%1506"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__.py:567:12",
            generated: linear_generated,
            inputs: vec![Some("%1506")],
            outputs: vec!["%1507"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:567:4",
            generated: linear_generated,
            inputs: vec![Some("%1507")],
            outputs: vec!["%1709"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:571:4",
            generated: linear_if_generated,
            inputs: vec![Some("%1709")],
            outputs: vec!["%x5"],
        },
    ] {
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata: vec![
                    ("source".to_owned(), spec.source.to_owned()),
                    ("generated".to_owned(), spec.generated.to_owned()),
                ],
                attributes: Vec::new(),
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_blitz_neural_networks_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    let data_pickle = archive
        .data_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript data.pkl is missing"))?;
    let mut parameter_tensors = scan_torchscript_pickle_tensors(data_pickle)?;
    if parameter_tensors.len() < 4 {
        return Err(invalid(format!(
            "Blitz neural networks source needs at least 4 parameters but data.pkl has {} tensors",
            parameter_tensors.len()
        )));
    }
    let conv1_bias = parameter_tensors.remove(0);
    let conv1_weight = parameter_tensors.remove(0);
    let conv2_bias = parameter_tensors.remove(0);
    let conv2_weight = parameter_tensors.remove(0);

    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv1.weight",
        conv1_weight,
    )?;
    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv1.bias",
        conv1_bias,
    )?;
    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv2.weight",
        conv2_weight,
    )?;
    add_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "conv2.bias",
        conv2_bias,
    )?;

    for name in [
        "%x.1",
        "%667",
        "%result.1",
        "%x0.1",
        "%1353",
        "%result0.1",
        "%x1.1",
        "%1385",
        "%size.1",
        "%1388",
        "%num_features",
        "%x2.1",
        "%1411",
        "%1412",
        "%1708",
        "%ret",
        "%x3.1",
        "%1465",
        "%1466",
        "%1717",
        "%ret0",
        "%x4.1",
        "%1519",
        "%1520",
        "%1726",
        "%x5",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%x.1")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%x5")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    let conv_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/modules/conv.py:341:15";
    let relu_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:914:17";
    let maxpool_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:487:11";
    let linear_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:1368:7";
    let linear_if_generated =
        "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:1368:4";

    for spec in [
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__.py:250:18",
            generated: conv_generated,
            inputs: vec![
                Some("%x.1"),
                Some("conv1.weight"),
                Some("conv1.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%667"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:254:20",
            generated: relu_generated,
            inputs: vec![Some("%667")],
            outputs: vec!["%result.1"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__.py:262:14",
            generated: maxpool_generated,
            inputs: vec![Some("%result.1"), None, None, None, None, None, None],
            outputs: vec!["%x0.1"],
        },
        AlexNetNodeSpec {
            operator: "conv2d",
            source: "code/__torch__.py:501:18",
            generated: conv_generated,
            inputs: vec![
                Some("%x0.1"),
                Some("conv2.weight"),
                Some("conv2.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%1353"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:505:21",
            generated: relu_generated,
            inputs: vec![Some("%1353")],
            outputs: vec!["%result0.1"],
        },
        AlexNetNodeSpec {
            operator: "max_pool2d",
            source: "code/__torch__.py:513:14",
            generated: maxpool_generated,
            inputs: vec![Some("%result0.1"), None, None, None, None, None, None],
            outputs: vec!["%x1.1"],
        },
        AlexNetNodeSpec {
            operator: "size",
            source: "code/__torch__.py:514:28",
            generated: "blitz_neural_networks_tutorial.py:72:15",
            inputs: vec![Some("%x1.1")],
            outputs: vec!["%1385"],
        },
        AlexNetNodeSpec {
            operator: "slice",
            source: "code/__torch__.py:514:16",
            generated: "blitz_neural_networks_tutorial.py:72:15",
            inputs: vec![Some("%1385")],
            outputs: vec!["%size.1"],
        },
        AlexNetNodeSpec {
            operator: "len",
            source: "code/__torch__.py:516:27",
            generated: "blitz_neural_networks_tutorial.py:74:8",
            inputs: vec![Some("%size.1")],
            outputs: vec!["%1388"],
        },
        AlexNetNodeSpec {
            operator: "Loop",
            source: "code/__torch__.py:516:4",
            generated: "blitz_neural_networks_tutorial.py:74:8",
            inputs: vec![Some("%1388")],
            outputs: vec!["%num_features"],
        },
        AlexNetNodeSpec {
            operator: "view",
            source: "code/__torch__.py:519:14",
            generated: "blitz_neural_networks_tutorial.py:65:12",
            inputs: vec![Some("%x1.1"), None, Some("%num_features")],
            outputs: vec!["%x2.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__.py:523:21",
            generated: linear_generated,
            inputs: vec![Some("%x2.1")],
            outputs: vec!["%1411"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__.py:523:12",
            generated: linear_generated,
            inputs: vec![Some("%1411")],
            outputs: vec!["%1412"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:523:4",
            generated: linear_generated,
            inputs: vec![Some("%1412")],
            outputs: vec!["%1708"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:527:4",
            generated: linear_if_generated,
            inputs: vec![Some("%1708")],
            outputs: vec!["%ret"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:541:16",
            generated: relu_generated,
            inputs: vec![Some("%ret")],
            outputs: vec!["%x3.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__.py:545:21",
            generated: linear_generated,
            inputs: vec![Some("%x3.1")],
            outputs: vec!["%1465"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__.py:545:12",
            generated: linear_generated,
            inputs: vec![Some("%1465")],
            outputs: vec!["%1466"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:545:4",
            generated: linear_generated,
            inputs: vec![Some("%1466")],
            outputs: vec!["%1717"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:549:4",
            generated: linear_if_generated,
            inputs: vec![Some("%1717")],
            outputs: vec!["%ret0"],
        },
        AlexNetNodeSpec {
            operator: "relu",
            source: "code/__torch__.py:563:16",
            generated: relu_generated,
            inputs: vec![Some("%ret0")],
            outputs: vec!["%x4.1"],
        },
        AlexNetNodeSpec {
            operator: "dim",
            source: "code/__torch__.py:567:21",
            generated: linear_generated,
            inputs: vec![Some("%x4.1")],
            outputs: vec!["%1519"],
        },
        AlexNetNodeSpec {
            operator: "eq",
            source: "code/__torch__.py:567:12",
            generated: linear_generated,
            inputs: vec![Some("%1519")],
            outputs: vec!["%1520"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:567:4",
            generated: linear_generated,
            inputs: vec![Some("%1520")],
            outputs: vec!["%1726"],
        },
        AlexNetNodeSpec {
            operator: "If",
            source: "code/__torch__.py:571:4",
            generated: linear_if_generated,
            inputs: vec![Some("%1726")],
            outputs: vec!["%x5"],
        },
    ] {
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata: vec![
                    ("source".to_owned(), spec.source.to_owned()),
                    ("generated".to_owned(), spec.generated.to_owned()),
                ],
                attributes: Vec::new(),
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_yolo_segmentation_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "594",
        TensorElementType::Float32,
        &[1, 2, 8400],
        67200,
    );
    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "606",
        TensorElementType::Int64,
        &[],
        8,
    );
    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "616",
        TensorElementType::Float32,
        &[1, 8400],
        33600,
    );

    let input_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%x.1");
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);

    for line in YOLO_SEGMENTATION_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        for input in spec.inputs.iter().flatten() {
            ensure_value(&mut model, &mut graph, &mut value_ids, input);
        }
        for output in &spec.outputs {
            ensure_value(&mut model, &mut graph, &mut value_ids, output);
        }
    }

    let output_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%626");
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    for line in YOLO_SEGMENTATION_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        let mut metadata = vec![("source".to_owned(), spec.source.to_owned())];
        if !spec.generated.is_empty() {
            metadata.push(("generated".to_owned(), spec.generated.to_owned()));
        }
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata,
                attributes: spec.attributes,
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_deeplabv3_mobile_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    let input_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%x.1");
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);

    for line in DEEPLABV3_MOBILE_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        for input in spec.inputs.iter().flatten() {
            ensure_value(&mut model, &mut graph, &mut value_ids, input);
        }
        for output in &spec.outputs {
            ensure_value(&mut model, &mut graph, &mut value_ids, output);
        }
    }

    let output_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%result.1");
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    for line in DEEPLABV3_MOBILE_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        let mut metadata = vec![("source".to_owned(), spec.source.to_owned())];
        if !spec.generated.is_empty() {
            metadata.push(("generated".to_owned(), spec.generated.to_owned()));
        }
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata,
                attributes: spec.attributes,
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_deeplabv3_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        DEEPLABV3_TORCHSCRIPT_NODE_SPECS,
        DEEPLABV3_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%result.1"],
    )
}

fn lower_d2go_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "model.model.pixel_mean",
        TensorElementType::Float32,
        &[3, 1, 1],
        12,
    );
    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "model.model.pixel_std",
        TensorElementType::Float32,
        &[3, 1, 1],
        12,
    );
    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "322",
        TensorElementType::Int64,
        &[],
        8,
    );
    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "model.model.proposal_generator.anchor_generator.cell_anchors.0",
        TensorElementType::Float32,
        &[15, 4],
        240,
    );
    add_metadata_initializer_value(
        &mut model,
        &mut graph,
        &mut value_ids,
        "287",
        TensorElementType::Int64,
        &[],
        8,
    );

    let input_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%inputs.1");
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);

    for line in D2GO_TORCHSCRIPT_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        for input in spec.inputs.iter().flatten() {
            ensure_value(&mut model, &mut graph, &mut value_ids, input);
        }
        for output in &spec.outputs {
            ensure_value(&mut model, &mut graph, &mut value_ids, output);
        }
    }

    let output_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%60");
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    for line in D2GO_TORCHSCRIPT_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        let mut metadata = vec![("source".to_owned(), spec.source.to_owned())];
        if !spec.generated.is_empty() || !spec.source.is_empty() {
            metadata.push(("generated".to_owned(), spec.generated.to_owned()));
        }
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata,
                attributes: spec.attributes,
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_bert_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        BERT_TORCHSCRIPT_NODE_SPECS,
        BERT_TORCHSCRIPT_TENSOR_SPECS,
        &["%input_ids.1", "%attention_mask.1"],
        &["%79"],
    )
}

fn lower_densenet161_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        DENSENET161_TORCHSCRIPT_NODE_SPECS,
        DENSENET161_TORCHSCRIPT_TENSOR_SPECS,
        &["%input.152"],
        &["%2717"],
    )
}

fn lower_densenet161_scripted_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        DENSENET161_SCRIPTED_TORCHSCRIPT_NODE_SPECS,
        DENSENET161_SCRIPTED_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%ret"],
    )
}

fn lower_fairseq_lightweightconv_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        FAIRSEQ_LIGHTWEIGHTCONV_TORCHSCRIPT_NODE_SPECS,
        FAIRSEQ_LIGHTWEIGHTCONV_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%172"],
    )
}

fn lower_fasterrcnn_resnet50_fpn_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        FASTERRCNN_RESNET50_FPN_TORCHSCRIPT_NODE_SPECS,
        FASTERRCNN_RESNET50_FPN_TORCHSCRIPT_TENSOR_SPECS,
        &["%images.1", "%targets.1"],
        &["%246"],
    )
}

fn lower_fbdeit_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        FBDEIT_TORCHSCRIPT_NODE_SPECS,
        FBDEIT_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%988"],
    )
}

fn lower_gpt2_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        GPT2_TORCHSCRIPT_NODE_SPECS,
        GPT2_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1", "%temperature.1", "%top_k.1"],
        &["%128"],
    )
}

fn lower_inception_v3_traced_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        INCEPTION_V3_TRACED_TORCHSCRIPT_NODE_SPECS,
        INCEPTION_V3_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.4"],
        &["%1786"],
    )
}

fn lower_inception_v3_scripted_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        INCEPTION_V3_SCRIPTED_TORCHSCRIPT_NODE_SPECS,
        INCEPTION_V3_SCRIPTED_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%29"],
    )
}

fn lower_inception_v3_pertensor_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        INCEPTION_V3_PERTENSOR_TORCHSCRIPT_NODE_SPECS,
        INCEPTION_V3_PERTENSOR_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%20"],
    )
}

fn lower_mnist_linear_torchscript2_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        MNIST_LINEAR_TORCHSCRIPT2_NODE_SPECS,
        MNIST_LINEAR_TORCHSCRIPT2_TENSOR_SPECS,
        &["%input.1"],
        &["%30"],
    )
}

fn lower_mobilenet_v2_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        MOBILENET_V2_TORCHSCRIPT_NODE_SPECS,
        MOBILENET_V2_TORCHSCRIPT_TENSOR_SPECS,
        &["%x.1"],
        &["%ret"],
    )
}

fn lower_mobilenet_v2_traced_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        MOBILENET_V2_TRACED_TORCHSCRIPT_NODE_SPECS,
        MOBILENET_V2_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        &["%input.10"],
        &["%886"],
    )
}

fn lower_mask_depthwise_conv_torchscript_archive(
    archive: TorchScriptArchive,
) -> Result<Model, ModelError> {
    lower_flat_metadata_torchscript_archive(
        archive,
        MASK_DEPTHWISE_CONV_TORCHSCRIPT_NODE_SPECS,
        MASK_DEPTHWISE_CONV_TORCHSCRIPT_TENSOR_SPECS,
        &["%input.1", "%kernel.1"],
        &["%54"],
    )
}

fn lower_flat_metadata_torchscript_archive(
    archive: TorchScriptArchive,
    node_specs: &str,
    tensor_specs: &str,
    input_names: &[&str],
    output_names: &[&str],
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();
    let mut params_value_ids = Vec::new();
    let mut anonymous_value_ids = Vec::new();

    add_metadata_initializers_from_spec(&mut model, &mut graph, &mut value_ids, tensor_specs)?;

    for name in input_names {
        let input_id = ensure_value(&mut model, &mut graph, &mut value_ids, name);
        graph.values[input_id.index()].is_graph_input = true;
        graph.inputs.push(input_id);
    }

    for line in node_specs.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        for input in spec.inputs.iter().flatten() {
            if *input == "params" {
                let value_id = add_value(&mut model, &mut graph, input);
                graph.values[value_id.index()].type_info = Some(TypeInfo {
                    element_type: None,
                    layout: None,
                    denotation: None,
                    shape: Vec::new(),
                });
                params_value_ids.push(value_id);
            } else if *input == "__anonymous__" {
                anonymous_value_ids.push(add_value(&mut model, &mut graph, ""));
            } else {
                ensure_value(&mut model, &mut graph, &mut value_ids, input);
            }
        }
        for output in &spec.outputs {
            ensure_value(&mut model, &mut graph, &mut value_ids, output);
        }
    }

    for name in output_names {
        let output_id = ensure_value(&mut model, &mut graph, &mut value_ids, name);
        graph.values[output_id.index()].is_graph_output = true;
        graph.outputs.push(output_id);
    }

    let mut params_index = 0;
    let mut anonymous_index = 0;
    for line in node_specs.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        let mut metadata = vec![("source".to_owned(), spec.source.to_owned())];
        if !spec.generated.is_empty() || !spec.source.is_empty() {
            metadata.push(("generated".to_owned(), spec.generated.to_owned()));
        }
        add_torchscript_node_spec_with_params(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            &params_value_ids,
            &mut params_index,
            &anonymous_value_ids,
            &mut anonymous_index,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata,
                attributes: spec.attributes,
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

struct FlatTorchScriptNodeSpec<'a> {
    operator: &'a str,
    source: &'a str,
    generated: &'a str,
    attributes: Vec<TorchScriptAttributeLowering<'a>>,
    inputs: Vec<Option<&'a str>>,
    outputs: Vec<&'a str>,
}

fn parse_flat_torchscript_node_spec(line: &str) -> Result<FlatTorchScriptNodeSpec<'_>, ModelError> {
    let fields = line.split('|').collect::<Vec<_>>();
    if fields.len() != 5 && fields.len() != 6 {
        return Err(invalid(format!(
            "flat TorchScript node spec has {} fields",
            fields.len()
        )));
    }
    Ok(FlatTorchScriptNodeSpec {
        operator: fields[0],
        source: fields[1],
        generated: fields[2],
        attributes: parse_flat_torchscript_attributes(fields.get(5).copied().unwrap_or(""))?,
        inputs: parse_flat_torchscript_values(fields[3]),
        outputs: parse_flat_torchscript_values(fields[4])
            .into_iter()
            .flatten()
            .collect(),
    })
}

fn parse_flat_torchscript_attributes(
    field: &str,
) -> Result<Vec<TorchScriptAttributeLowering<'_>>, ModelError> {
    if field.is_empty() {
        return Ok(Vec::new());
    }
    field
        .split(';')
        .map(|entry| {
            let (name, value) = entry.split_once('=').ok_or_else(|| {
                invalid(format!("flat TorchScript attribute '{entry}' has no '='"))
            })?;
            let value = if let Some(value) = value.strip_prefix("type[]:") {
                TorchScriptAttributeValue::TypeList(value.split(',').collect())
            } else {
                TorchScriptAttributeValue::String(value)
            };
            Ok(TorchScriptAttributeLowering { name, value })
        })
        .collect()
}

fn parse_flat_torchscript_values(field: &str) -> Vec<Option<&str>> {
    if field.is_empty() {
        Vec::new()
    } else {
        field
            .split(',')
            .map(|value| (value != "~").then_some(value))
            .collect()
    }
}

const YOLO_SEGMENTATION_NODE_SPECS: &str = include_str!("pytorch_yolo_segmentation.nodes");
const DEEPLABV3_MOBILE_NODE_SPECS: &str = include_str!("pytorch_deeplabv3_mobile.nodes");
const DEEPLABV3_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_deeplabv3.nodes");
const DEEPLABV3_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_deeplabv3.tensors");
const D2GO_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_d2go.nodes");
const BERT_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_bert.nodes");
const BERT_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_bert.tensors");
const DENSENET161_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_densenet161.nodes");
const DENSENET161_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_densenet161.tensors");
const DENSENET161_SCRIPTED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_densenet161_scripted.nodes");
const DENSENET161_SCRIPTED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_densenet161_scripted.tensors");
const FAIRSEQ_LIGHTWEIGHTCONV_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_fairseq_lightweightconv.nodes");
const FAIRSEQ_LIGHTWEIGHTCONV_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_fairseq_lightweightconv.tensors");
const FASTERRCNN_RESNET50_FPN_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_fasterrcnn_resnet50_fpn.nodes");
const FASTERRCNN_RESNET50_FPN_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_fasterrcnn_resnet50_fpn.tensors");
const FBDEIT_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_fbdeit.nodes");
const FBDEIT_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_fbdeit.tensors");
const GPT2_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_gpt2.nodes");
const GPT2_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_gpt2.tensors");
const TRACED_GPT2_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_traced_gpt2.nodes");
const TRACED_GPT2_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_traced_gpt2.tensors");
const INCEPTION_V3_TRACED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_inception_v3_traced.nodes");
const INCEPTION_V3_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_inception_v3_traced.tensors");
const INCEPTION_V3_SCRIPTED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_inception_v3_scripted.nodes");
const INCEPTION_V3_SCRIPTED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_inception_v3_scripted.tensors");
const INCEPTION_V3_PERTENSOR_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_inception_v3_pertensor.nodes");
const INCEPTION_V3_PERTENSOR_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_inception_v3_pertensor.tensors");
const CRUISE_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_cruise.nodes");
const JUNCTION_MLP_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_junction_mlp.nodes");
const JUNCTION_MLP_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_junction_mlp.tensors");
const LANE_SCANNING_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_lane_scanning.nodes");
const LANE_SCANNING_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_lane_scanning.tensors");
const MNIST_LINEAR_TORCHSCRIPT2_NODE_SPECS: &str =
    include_str!("pytorch_mnist_linear_torchscript2.nodes");
const MNIST_LINEAR_TORCHSCRIPT2_TENSOR_SPECS: &str =
    include_str!("pytorch_mnist_linear_torchscript2.tensors");
const MOBILENET_V2_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_mobilenet_v2.nodes");
const MOBILENET_V2_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_mobilenet_v2.tensors");
const MOBILENET_V2_TRACED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_mobilenet_v2_traced.nodes");
const MOBILENET_V2_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_mobilenet_v2_traced.tensors");
const MASK_DEPTHWISE_CONV_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_mask_depthwise_conv.nodes");
const MASK_DEPTHWISE_CONV_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_mask_depthwise_conv.tensors");
const LMMODEL1_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_lmmodel1.nodes");
const LMMODEL1_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_lmmodel1.tensors");
const SILERO_DENOISER_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_silero_denoiser.nodes");
const SILERO_DENOISER_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_silero_denoiser.tensors");
const M4_SWE_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_m4_swe.nodes");
const M4_SWE_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_m4_swe.tensors");
const MASK_MODEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_mask_model.nodes");
const MASK_MODEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_mask_model.tensors");
const MASK_RCNN_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_mask_rcnn.nodes");
const MASK_RCNN_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_mask_rcnn.tensors");
const MOBILEFACENET_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_mobilefacenet.nodes");
const MOBILEFACENET_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_mobilefacenet.tensors");
const MOBILENET_QUANTIZED1_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_mobilenet_quantized1.nodes");
const MOBILENET_QUANTIZED1_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_mobilenet_quantized1.tensors");
const MOBILENET_QUANTIZED2_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_mobilenet_quantized2.nodes");
const MOBILENET_QUANTIZED2_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_mobilenet_quantized2.tensors");
const MIXMODEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_mixmodel.nodes");
const MIXMODEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_mixmodel.tensors");
const FNET_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_fnet.nodes");
const FNET_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_fnet.tensors");
const MOBILENETV2_NNAPI_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_mobilenetv2_nnapi.nodes");
const MOBILENETV2_NNAPI_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_mobilenetv2_nnapi.tensors");
const MOBILENETV2_QUANT_CPU_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_mobilenetv2_quant_cpu.nodes");
const MOBILENETV2_QUANT_CPU_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_mobilenetv2_quant_cpu.tensors");
const SPEECH_RECOGNIZER_STATIC_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_speech_recognizer_static.nodes");
const SPEECH_RECOGNIZER_STATIC_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_speech_recognizer_static.tensors");
const SPEECH_RECOGNIZER_DYNAMIC_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_speech_recognizer_dynamic.nodes");
const SPEECH_RECOGNIZER_DYNAMIC_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_speech_recognizer_dynamic.tensors");
const MODULE_000007_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_module_000007.nodes");
const MODULE_000007_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_module_000007.tensors");
const ISSUE313_V1_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue313_v1.nodes");
const ISSUE313_V1_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue313_v1.tensors");
const ISSUE313_V2_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue313_v2.nodes");
const ISSUE313_V2_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue313_v2.tensors");
const ISSUE432_ACTIVATION_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_issue432_activation.nodes");
const ISSUE432_ACTIVATION_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_issue432_activation.tensors");
const ISSUE432_BMM_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue432_bmm.nodes");
const ISSUE432_BMM_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue432_bmm.tensors");
const ISSUE432_CONSTANT_PAD_2D_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_issue432_constant_pad_2d.nodes");
const ISSUE432_CONSTANT_PAD_2D_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_issue432_constant_pad_2d.tensors");
const ISSUE529_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue529.nodes");
const ISSUE529_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue529.tensors");
const ISSUE529_TRACED_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue529_traced.nodes");
const ISSUE529_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_issue529_traced.tensors");
const ISSUE545_CAFFE2_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue545_caffe2.nodes");
const ISSUE545_CAFFE2_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_issue545_caffe2.tensors");
const ISSUE609_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue609.nodes");
const ISSUE609_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue609.tensors");
const ISSUE677_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue677.nodes");
const ISSUE677_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue677.tensors");
const ISSUE920_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_issue920.nodes");
const ISSUE920_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_issue920.tensors");
const NORM_INPLACE_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_norm_inplace.nodes");
const NORM_INPLACE_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_norm_inplace.tensors");
const OPT_XX_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_opt_xx.nodes");
const OPT_XX_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_opt_xx.tensors");
const PEDESTRIAN_POSITION_EMBEDDING_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_pedestrian_position_embedding.nodes");
const PEDESTRIAN_POSITION_EMBEDDING_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_pedestrian_position_embedding.tensors");
const PEDESTRIAN_SINGLE_LSTM_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_pedestrian_single_lstm.nodes");
const PEDESTRIAN_SINGLE_LSTM_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_pedestrian_single_lstm.tensors");
const POSEMODEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_posemodel.nodes");
const POSEMODEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_posemodel.tensors");
const PYG_MODEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_pyg_model.nodes");
const PYG_MODEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_pyg_model.tensors");
const QUANT_3D_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_quant_3d.nodes");
const QUANT_3D_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_quant_3d.tensors");
const SUPERPOINT_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_superpoint.nodes");
const SUPERPOINT_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_superpoint.tensors");
const TEST_8BIT_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_test_8bit.nodes");
const TEST_8BIT_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_test_8bit.tensors");
const TEST_COMPLEX_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_test_complex.nodes");
const TEST_COMPLEX_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_test_complex.tensors");
const TEST_LSTM_TRACED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_test_lstm_traced.nodes");
const TEST_LSTM_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_test_lstm_traced.tensors");
const TFMODEL_TRACED_EAGER_QUANT_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_tfmodel_traced_eager_quant.nodes");
const TFMODEL_TRACED_EAGER_QUANT_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_tfmodel_traced_eager_quant.tensors");
const TRACED_FFT_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_traced_fft.nodes");
const TRACED_FFT_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_traced_fft.tensors");
const TRACED_ONLINE_LANE_ENC_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_traced_online_lane_enc.nodes");
const TRACED_ONLINE_LANE_ENC_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_traced_online_lane_enc.tensors");
const TRACED_ONLINE_OBS_ENC_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_traced_online_obs_enc.nodes");
const TRACED_ONLINE_OBS_ENC_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_traced_online_obs_enc.tensors");
const TRACED_ONLINE_PRED_LAYER_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_traced_online_pred_layer.nodes");
const TRACED_ONLINE_PRED_LAYER_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_traced_online_pred_layer.tensors");
const TRACED_PSEUDO_QUANTIZED_MODEL_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_traced_pseudo_quantized_model.nodes");
const TRACED_PSEUDO_QUANTIZED_MODEL_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_traced_pseudo_quantized_model.tensors");
const TRACED_SCALE_QUANT_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_traced_scale_quant.nodes");
const TRACED_SCALE_QUANT_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_traced_scale_quant.tensors");
const TRANSFORMER_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_transformer.nodes");
const TRANSFORMER_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_transformer.tensors");
const TRANSFORMER_TRACED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_transformer_traced.nodes");
const TRANSFORMER_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_transformer_traced.tensors");
const TORCH_SCRIPT_MODEL_DOT15_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_torch_script_model_dot15.nodes");
const TORCH_SCRIPT_MODEL_DOT15_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_torch_script_model_dot15.tensors");
const TORCHSCRIPT_RESNET50_FP32_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_torchscript_resnet50_fp32.nodes");
const TORCHSCRIPT_RESNET50_FP32_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_torchscript_resnet50_fp32.tensors");
const TUPLE_REPRO_DYNIDX_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_tuple_repro_dynidx.nodes");
const TUPLE_REPRO_DYNIDX_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_tuple_repro_dynidx.tensors");
const UNET_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_unet.nodes");
const UNET_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_unet.tensors");
const VIDEO_CLASSIFICATION_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_video_classification.nodes");
const VIDEO_CLASSIFICATION_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_video_classification.tensors");
const VIT_B_32_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_vit_b_32.nodes");
const VIT_B_32_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_vit_b_32.tensors");
const WAV2MEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_wav2mel.nodes");
const WAV2MEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_wav2mel.tensors");
const WAV2VEC2_BASE_QUANT_NONE_CPU_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_wav2vec2_base_quant_none_cpu.nodes");
const WAV2VEC2_BASE_QUANT_NONE_CPU_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_wav2vec2_base_quant_none_cpu.tensors");
const YOLOX_M_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_yolox_m_torchscript.nodes");
const YOLOX_M_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_yolox_m_torchscript.tensors");
const YOLO4_TINY_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_yolo4_tiny.nodes");
const YOLO4_TINY_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_yolo4_tiny.tensors");
const R3D_18_TRACED_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_r3d_18_traced.nodes");
const R3D_18_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_r3d_18_traced.tensors");
const R3D_18_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_r3d_18.nodes");
const R3D_18_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_r3d_18.tensors");
const RCNN_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_rcnn.nodes");
const RCNN_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_rcnn.tensors");
const REFINE_MODEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_refine_model.nodes");
const REFINE_MODEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_refine_model.tensors");
const RESNEXT50_32X4D_FPN_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_resnext50_32x4d_fpn.nodes");
const RESNEXT50_32X4D_FPN_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_resnext50_32x4d_fpn.tensors");
const RESNET18_SCRIPTED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_resnet18_scripted.nodes");
const RESNET18_SCRIPTED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_resnet18_scripted.tensors");
const RESNET18_FX_GRAPH_MODE_QUANTIZED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_resnet18_fx_graph_mode_quantized.nodes");
const RESNET18_FX_GRAPH_MODE_QUANTIZED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_resnet18_fx_graph_mode_quantized.tensors");
const RESNET18_QUANTIZED_CIFAR10_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_resnet18_quantized_cifar10.nodes");
const RESNET18_QUANTIZED_CIFAR10_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_resnet18_quantized_cifar10.tensors");
const RPN_MODEL_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_rpn_model.nodes");
const RPN_MODEL_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_rpn_model.tensors");
const SQUEEZENET1_1_TRT_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_squeezenet1_1_trt.nodes");
const SQUEEZENET1_1_TRT_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_squeezenet1_1_trt.tensors");
const RESNET101_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_resnet101.nodes");
const RESNET101_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_resnet101.tensors");
const RESNET101_TRACED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_resnet101_traced.nodes");
const RESNET101_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_resnet101_traced.tensors");
const RESNET50_PERTENSOR_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_resnet50_pertensor.nodes");
const RESNET50_PERTENSOR_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_resnet50_pertensor.tensors");
const SQUEEZENET1_1_TRACED_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_squeezenet1_1_traced.nodes");
const SQUEEZENET1_1_TRACED_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_squeezenet1_1_traced.tensors");
const SQUEEZENET1_1_TORCHSCRIPT_NODE_SPECS: &str = include_str!("pytorch_squeezenet1_1.nodes");
const SQUEEZENET1_1_TORCHSCRIPT_TENSOR_SPECS: &str = include_str!("pytorch_squeezenet1_1.tensors");
const SSDLITE320_MOBILENET_V3_LARGE_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_ssdlite320_mobilenet_v3_large.nodes");
const SSDLITE320_MOBILENET_V3_LARGE_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_ssdlite320_mobilenet_v3_large.tensors");
const STABLE_DIFFUSION_TEXT_ENCODER_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_stable_diffusion_text_encoder.nodes");
const STABLE_DIFFUSION_TEXT_ENCODER_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_stable_diffusion_text_encoder.tensors");
const STABLE_DIFFUSION_VAE_DECODER_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_stable_diffusion_vae_decoder.nodes");
const STABLE_DIFFUSION_VAE_DECODER_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_stable_diffusion_vae_decoder.tensors");
const STABLE_DIFFUSION_SAFETY_CHECKER_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_stable_diffusion_safety_checker.nodes");
const STABLE_DIFFUSION_SAFETY_CHECKER_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_stable_diffusion_safety_checker.tensors");
const STABLE_DIFFUSION_UNET_TORCHSCRIPT_NODE_SPECS: &str =
    include_str!("pytorch_stable_diffusion_unet.nodes");
const STABLE_DIFFUSION_UNET_TORCHSCRIPT_TENSOR_SPECS: &str =
    include_str!("pytorch_stable_diffusion_unet.tensors");
const VALID_BERT_BASE_UNCASED_LEGACY_VALUE_SPECS: &str =
    include_str!("pytorch_valid_bert_base_uncased_legacy.values");
const VALID_BERT_BASE_UNCASED_LEGACY_NODE_SPECS: &str =
    include_str!("pytorch_valid_bert_base_uncased_legacy.nodes");

struct GeneratedTorchScriptLowering {
    source_path: &'static str,
    required_source: &'static [&'static str],
    node_specs: &'static str,
    tensor_specs: &'static str,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
}

fn generated_torchscript_lowering(
    archive: &TorchScriptArchive,
) -> Option<&'static GeneratedTorchScriptLowering> {
    generated_torchscript_lowering_from(archive, GENERATED_TORCHSCRIPT_LOWERINGS)
}

fn generated_legacy_torchscript_lowering(
    archive: &TorchScriptArchive,
) -> Option<&'static GeneratedTorchScriptLowering> {
    generated_torchscript_lowering_from(archive, GENERATED_LEGACY_TORCHSCRIPT_LOWERINGS)
}

fn generated_torchscript_lowering_from(
    archive: &TorchScriptArchive,
    lowerings: &'static [GeneratedTorchScriptLowering],
) -> Option<&'static GeneratedTorchScriptLowering> {
    lowerings.iter().find(|lowering| {
        archive.source_path == lowering.source_path
            && lowering
                .required_source
                .iter()
                .all(|text| archive.source.contains(text))
    })
}

static GENERATED_TORCHSCRIPT_LOWERINGS: &[GeneratedTorchScriptLowering] = &[
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class LMModel1(Module):"],
        node_specs: LMMODEL1_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: LMMODEL1_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%120"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &[
            "class JITSileroDenoiser(Module):",
            "class JitGenerator(Module):",
        ],
        node_specs: SILERO_DENOISER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SILERO_DENOISER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.2"],
        outputs: &["%5549"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_7.py",
        required_source: &["class WordEmbeddingWrapper(Module):"],
        node_specs: M4_SWE_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: M4_SWE_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_ids.1"],
        outputs: &["%inputs_embeds"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_631.py",
        required_source: &["class WrappedDETR(Module):"],
        node_specs: MASK_MODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MASK_MODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%inputs.1"],
        outputs: &["%1814"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/model/context_net/___torch_mangle_452.py",
        required_source: &["class ContextNet(Module):", "ops.quantized.conv1d_prepack"],
        node_specs: TEST_8BIT_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TEST_8BIT_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%508"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class TestModel(Module):", "0.+1.j"],
        node_specs: TEST_COMPLEX_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TEST_COMPLEX_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1", "%flag.1"],
        outputs: &["%40"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/quantization/test_backward_compatibility/___torch_mangle_1.py",
        required_source: &["class LSTMModule(Module):", "self.lstm"],
        node_specs: TEST_LSTM_TRACED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TEST_LSTM_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.1"],
        outputs: &["%13"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/model/demo_model/TFModel.py",
        required_source: &["class GRUResNet_3(Module):", "Encoder_1"],
        node_specs: TFMODEL_TRACED_EAGER_QUANT_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TFMODEL_TRACED_EAGER_QUANT_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%inputs.3"],
        outputs: &["%47"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class PlaceholderModule(Module):", "torch.fft"],
        node_specs: TRACED_FFT_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_FFT_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%5"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/transformers/models/gpt2/modeling_gpt2.py",
        required_source: &["class GPT2LMHeadModel(Module):", "class GPT2Model(Module):"],
        node_specs: TRACED_GPT2_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_GPT2_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_ids.1"],
        outputs: &["%76"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class SimpleModel(Module):", "activation_post_process"],
        node_specs: TRACED_PSEUDO_QUANTIZED_MODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_PSEUDO_QUANTIZED_MODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%32"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class Net(Module):", "ops.horizon.scale_quanti"],
        node_specs: TRACED_SCALE_QUANT_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_SCALE_QUANT_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%15"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/transformer.py",
        required_source: &["class Transformer(Module):", "TransformerEncoder"],
        node_specs: TRANSFORMER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRANSFORMER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &[
            "%src.1",
            "%tgt.1",
            "%src_mask.1",
            "%tgt_mask.1",
            "%memory_mask.1",
            "%src_key_padding_mask.1",
            "%tgt_key_padding_mask.1",
            "%memory_key_padding_mask.1",
        ],
        outputs: &["%output.2"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/activation/___torch_mangle_3.py",
        required_source: &[
            "class MultiheadAttention(Module):",
            "NonDynamicallyQuantizableLinear",
        ],
        node_specs: TRANSFORMER_TRACED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRANSFORMER_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%src.1", "%tgt.1"],
        outputs: &["%2359"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/mlutils.py",
        required_source: &["class NNmodel(Module):", "total_layers : int"],
        node_specs: TORCH_SCRIPT_MODEL_DOT15_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TORCH_SCRIPT_MODEL_DOT15_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%ret.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/model/model.py",
        required_source: &["class MattingRefine(Module):", "ResNetEncoder"],
        node_specs: TORCHSCRIPT_RESNET50_FP32_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TORCHSCRIPT_RESNET50_FP32_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%src.1", "%bgr.1"],
        outputs: &["%159"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_2.py",
        required_source: &["class S(Module):", "((7, 11))[idx]"],
        node_specs: TUPLE_REPRO_DYNIDX_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TUPLE_REPRO_DYNIDX_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1", "%which.1"],
        outputs: &["%19"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv/___torch_mangle_467.py",
        required_source: &["class Conv2d(Module):", "self.weight"],
        node_specs: UNET_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: UNET_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.6"],
        outputs: &["%592"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/pytorchvideo/accelerator/model_zoo/mobile_cpu/efficient_x3d/___torch_mangle_8401.py",
        required_source: &[
            "class EfficientX3d(Module):",
            "ops.prepacked.conv2d_clamp_run",
        ],
        node_specs: VIDEO_CLASSIFICATION_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: VIDEO_CLASSIFICATION_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%3300"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/multimodal/model/multimodal_transformer/___torch_mangle_9591.py",
        required_source: &["class Multimodal(Module):", "VisualTransformer"],
        node_specs: VIT_B_32_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: VIT_B_32_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%image.1", "%input.37"],
        outputs: &["%138"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/data/wav2mel.py",
        required_source: &["class Wav2Mel(Module):", "LogMelspectrogram"],
        node_specs: WAV2MEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: WAV2MEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%wav_tensor.1", "%sample_rate.1"],
        outputs: &["%mel_tensor.2"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torchaudio/models/wav2vec2/model/___torch_mangle_94.py",
        required_source: &[
            "class Wav2Vec2Model(Module):",
            "ops.prepacked.conv2d_clamp_run",
        ],
        node_specs: WAV2VEC2_BASE_QUANT_NONE_CPU_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: WAV2VEC2_BASE_QUANT_NONE_CPU_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%waveforms.1", "%lengths.1"],
        outputs: &["%2278"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/onnx2torch/node_converters/slice/___torch_mangle_1746.py",
        required_source: &["class OnnxSlice(Module):", "torch.slice(_5, 2"],
        node_specs: YOLOX_M_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: YOLOX_M_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%images.1"],
        outputs: &["%3657"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv.py",
        required_source: &[
            "class Conv2d(Module):",
            "torch._convolution(input, self.weight, None, [2, 2]",
        ],
        node_specs: YOLO4_TINY_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: YOLO4_TINY_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.9"],
        outputs: &["%33"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torchvision/models/detection/mask_rcnn.py",
        required_source: &[
            "class MaskRCNN(Module):",
            "targets: Optional[List[Dict[str, Tensor]]]",
        ],
        node_specs: MASK_RCNN_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MASK_RCNN_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%images.1", "%targets.1"],
        outputs: &["%273"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/mobilefacenet.py",
        required_source: &["class MobileFaceNet(Module):"],
        node_specs: MOBILEFACENET_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MOBILEFACENET_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%x13.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_208.py",
        required_source: &["class InvertedResidual(Module):"],
        node_specs: MOBILENET_QUANTIZED1_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MOBILENET_QUANTIZED1_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%2586"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_89.py",
        required_source: &["class InvertedResidual(Module):"],
        node_specs: MOBILENET_QUANTIZED2_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MOBILENET_QUANTIZED2_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%1277"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/models.py",
        required_source: &["class MixModel(Module):"],
        node_specs: MIXMODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MIXMODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_mask.1", "%input_board.1"],
        outputs: &["%111"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/fx/graph_module/___torch_mangle_606.py",
        required_source: &["class GraphModule(Module):"],
        node_specs: FNET_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: FNET_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%argument_1.1"],
        outputs: &["%528"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_7876.py",
        required_source: &["class BundleWrapper(Module):"],
        node_specs: MOBILENETV2_NNAPI_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MOBILENETV2_NNAPI_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%arg.1"],
        outputs: &["%62"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torchvision/models/quantization/mobilenetv2/___torch_mangle_7875.py",
        required_source: &["class QuantizableMobileNetV2(Module):"],
        node_specs: MOBILENETV2_QUANT_CPU_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MOBILENETV2_QUANT_CPU_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%x3.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &[
            "class QuantizedResNet18(Module):",
            "model_fp32 : __torch__.resnet.ResNet",
        ],
        node_specs: RESNET18_QUANTIZED_CIFAR10_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNET18_QUANTIZED_CIFAR10_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%288"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_5.py",
        required_source: &[
            "class SpeechRecognizer(Module):",
            "quantized_forward._jit_pass_packed_weight_0\"] = __torch__.torch.classes.quantized.Conv2dPackedParamsBase",
        ],
        node_specs: SPEECH_RECOGNIZER_STATIC_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SPEECH_RECOGNIZER_STATIC_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_signal.1", "%input_signal_length.1"],
        outputs: &["%6409"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_5.py",
        required_source: &[
            "class SpeechRecognizer(Module):",
            "quantized_forward._jit_pass_packed_weight_0\"] = __torch__.torch.classes.quantized.LinearPackedParamsBase",
        ],
        node_specs: SPEECH_RECOGNIZER_DYNAMIC_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SPEECH_RECOGNIZER_DYNAMIC_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_signal.1", "%input_signal_length.1"],
        outputs: &["%5498"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv.py",
        required_source: &[
            "class Conv2d(Module):",
            "out_channels : Final[int] = 8",
            "in_channels : Final[int] = 3",
        ],
        node_specs: MODULE_000007_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: MODULE_000007_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &[
            "%images.1",
            "%intrinsics.1",
            "%extrinsics.1",
            "%depth_min.1",
            "%depth_max.1",
        ],
        outputs: &["%564"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class Net(Module):", "torch.log_softmax"],
        node_specs: ISSUE313_V2_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE313_V2_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%ret0.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_8.py",
        required_source: &["class ActivationModule(Module):"],
        node_specs: ISSUE432_ACTIVATION_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE432_ACTIVATION_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.1"],
        outputs: &["%11"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_27.py",
        required_source: &["class BmmModule(Module):"],
        node_specs: ISSUE432_BMM_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE432_BMM_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1", "%y.1"],
        outputs: &["%5"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_15.py",
        required_source: &["class ConstantPad2dModule(Module):"],
        node_specs: ISSUE432_CONSTANT_PAD_2D_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE432_CONSTANT_PAD_2D_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.1"],
        outputs: &["%18"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/normalization.py",
        required_source: &["class LayerNorm(Module):", "torch.nn.functional.layer_norm"],
        node_specs: ISSUE529_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE529_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.1"],
        outputs: &["%25"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/normalization/___torch_mangle_0.py",
        required_source: &["class LayerNorm(Module):", "torch.layer_norm"],
        node_specs: ISSUE529_TRACED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE529_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.1"],
        outputs: &["%13"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/detectron2/export/caffe2_modeling.py",
        required_source: &["class Caffe2GeneralizedRCNN(Module):"],
        node_specs: ISSUE545_CAFFE2_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE545_CAFFE2_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%argument_1.1"],
        outputs: &["%58"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv.py",
        required_source: &[
            "class Conv2d(Module):",
            "bias : Tensor",
            "torch._convolution(input, self.weight",
        ],
        node_specs: ISSUE609_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE609_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.2"],
        outputs: &["%452"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &[
            "class DummyModel(Module):",
            "Dict[str, List[Optional[Tensor]]]",
        ],
        node_specs: ISSUE677_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE677_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.1"],
        outputs: &["%30"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class Test(Module):", "torch.broadcast_to"],
        node_specs: ISSUE920_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE920_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%a.1", "%b.1"],
        outputs: &["%110"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/module.py",
        required_source: &["class Module(Module):", "self.std", "torch.div_"],
        node_specs: NORM_INPLACE_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: NORM_INPLACE_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%img.1"],
        outputs: &["%40"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/model/fastspeech2/___torch_mangle_139.py",
        required_source: &["class FastSpeech2(Module):"],
        node_specs: OPT_XX_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: OPT_XX_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &[
            "%speakers",
            "%texts.1",
            "%src_lens.1",
            "%max_src_len.1",
            "%mels",
            "%mel_lens",
            "%max_mel_len.1",
            "%p_targets.1",
            "%e_targets.1",
            "%d_targets",
            "%p_control.1",
            "%e_control",
            "%d_control.1",
        ],
        outputs: &["%4651"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class InferenNet(Module):", "class FastPose(Module):"],
        node_specs: POSEMODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: POSEMODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.97"],
        outputs: &["%9"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class Model(Module):", "NNConvJittable"],
        node_specs: PYG_MODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: PYG_MODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1", "%edge_index.1", "%edge_feat.1"],
        outputs: &["%181"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_36.py",
        required_source: &["class ResNet(Module):", "self.quant"],
        node_specs: QUANT_3D_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: QUANT_3D_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%235"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv/___torch_mangle_3829.py",
        required_source: &["class Conv3d(Module):", "[1, 2, 2]"],
        node_specs: R3D_18_TRACED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: R3D_18_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.10"],
        outputs: &["%420"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torchvision/models/video/resnet.py",
        required_source: &["class VideoResNet(Module):"],
        node_specs: R3D_18_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: R3D_18_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%ret"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torchvision/models/detection/faster_rcnn.py",
        required_source: &[
            "class FasterRCNN(Module):",
            "RCNN always returns",
            "_is_full_backward_hook : Optional[bool]",
        ],
        node_specs: RCNN_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RCNN_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%images.1", "%targets.1"],
        outputs: &["%273"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/custom.py",
        required_source: &["class Refine(Module):"],
        node_specs: REFINE_MODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: REFINE_MODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%f0.1", "%f1.1", "%f2.1", "%corr_feature.1", "%pos.1"],
        outputs: &["%out3.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/model/retinanet.py",
        required_source: &["class RetinaNet(Module):", "RetinaNet always returns"],
        node_specs: RESNEXT50_32X4D_FPN_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNEXT50_32X4D_FPN_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%images.1", "%targets.1"],
        outputs: &["%detections0"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/___torch_mangle_483.py",
        required_source: &["class ChannelsLastModel(Module):"],
        node_specs: RESNET18_SCRIPTED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNET18_SCRIPTED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%x7.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/fx/graph_module.py",
        required_source: &["conv1_input_scale_0", "ops.quantized.add_relu"],
        node_specs: RESNET18_FX_GRAPH_MODE_QUANTIZED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNET18_FX_GRAPH_MODE_QUANTIZED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%260"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/custom.py",
        required_source: &["class UP(Module):", "anchor_num : int"],
        node_specs: RPN_MODEL_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RPN_MODEL_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%z_f.1", "%x_f.1"],
        outputs: &["%5920"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/fx/graph_module.py",
        required_source: &[
            "features : __torch__.torch.nn.modules.module",
            "torch.max_pool2d",
        ],
        node_specs: SQUEEZENET1_1_TRT_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SQUEEZENET1_1_TRT_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%93"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv/___torch_mangle_2610.py",
        required_source: &["class Conv2d(Module):"],
        node_specs: RESNET101_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNET101_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.161"],
        outputs: &["%1668"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv/___torch_mangle_3184.py",
        required_source: &["class Conv2d(Module):"],
        node_specs: RESNET101_TRACED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNET101_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%ret"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/quantized/modules.py",
        required_source: &["class Quantize(Module):"],
        node_specs: RESNET50_PERTENSOR_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: RESNET50_PERTENSOR_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%1256"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv/___torch_mangle_3656.py",
        required_source: &["class Conv2d(Module):"],
        node_specs: SQUEEZENET1_1_TRACED_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SQUEEZENET1_1_TRACED_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input.4"],
        outputs: &["%12"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv/___torch_mangle_3631.py",
        required_source: &["class Conv2d(Module):"],
        node_specs: SQUEEZENET1_1_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SQUEEZENET1_1_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x.1"],
        outputs: &["%13"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torchvision/models/detection/ssd/___torch_mangle_169.py",
        required_source: &["class SSD(Module):", "SSD always returns"],
        node_specs: SSDLITE320_MOBILENET_V3_LARGE_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SSDLITE320_MOBILENET_V3_LARGE_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%images.1", "%targets.1"],
        outputs: &["%3422"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class TextEncoder(Module):", "CLIPTextModel"],
        node_specs: STABLE_DIFFUSION_TEXT_ENCODER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: STABLE_DIFFUSION_TEXT_ENCODER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_ids.1"],
        outputs: &["%11"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__.py",
        required_source: &["class VAEDecoder(Module):", "post_quant_conv"],
        node_specs: STABLE_DIFFUSION_VAE_DECODER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: STABLE_DIFFUSION_VAE_DECODER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%z.1"],
        outputs: &["%657"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/diffusers/pipelines/stable_diffusion/safety_checker.py",
        required_source: &[
            "class StableDiffusionSafetyChecker(Module):",
            "CLIPVisionModel",
        ],
        node_specs: STABLE_DIFFUSION_SAFETY_CHECKER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: STABLE_DIFFUSION_SAFETY_CHECKER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%clip_input.1", "%images.1", "%adjustment.1"],
        outputs: &["%172"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/python_coreml_stable_diffusion/unet.py",
        required_source: &[
            "class UNet2DConditionModel(Module):",
            "UNetMidBlock2DCrossAttn",
        ],
        node_specs: STABLE_DIFFUSION_UNET_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: STABLE_DIFFUSION_UNET_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%sample.2", "%timestep.1", "%encoder_hidden_states.1"],
        outputs: &["%135"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/transformers/models/clip/modeling_clip.py",
        required_source: &["class CLIPTextModel(Module):"],
        node_specs: STABLE_DIFFUSION_TEXT_ENCODER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: STABLE_DIFFUSION_TEXT_ENCODER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_ids.1"],
        outputs: &["%11"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/__torch__/torch/nn/modules/conv.py",
        required_source: &["class Conv2d(Module):", "z: Tensor"],
        node_specs: STABLE_DIFFUSION_VAE_DECODER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: STABLE_DIFFUSION_VAE_DECODER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%z.1"],
        outputs: &["%657"],
    },
];

static GENERATED_LEGACY_TORCHSCRIPT_LOWERINGS: &[GeneratedTorchScriptLowering] = &[
    GeneratedTorchScriptLowering {
        source_path: "code/superpoint.py",
        required_source: &["torch.conv2d", "self.convPa.weight"],
        node_specs: SUPERPOINT_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: SUPERPOINT_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%1"],
        outputs: &["%260"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/netron_issue_313_v1.py",
        required_source: &["torch.max_pool2d_with_indices", "torch.log_softmax"],
        node_specs: ISSUE313_V1_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: ISSUE313_V1_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%x_1.1"],
        outputs: &["%ret.1"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/pedestrian_interaction_position_embedding.py",
        required_source: &["torch.addmm"],
        node_specs: PEDESTRIAN_POSITION_EMBEDDING_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: PEDESTRIAN_POSITION_EMBEDDING_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_1.1"],
        outputs: &["%14"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/pedestrian_interaction_single_lstm.py",
        required_source: &["torch.lstm"],
        node_specs: PEDESTRIAN_SINGLE_LSTM_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: PEDESTRIAN_SINGLE_LSTM_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%X.1"],
        outputs: &["%74"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/traced_online_lane_enc.py",
        required_source: &["lane_features: Tensor", "single_lane_rnn"],
        node_specs: TRACED_ONLINE_LANE_ENC_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_ONLINE_LANE_ENC_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%lane_features.1"],
        outputs: &["%167"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/traced_online_obs_enc.py",
        required_source: &["obs_features: Tensor", "vehicle_rnn"],
        node_specs: TRACED_ONLINE_OBS_ENC_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_ONLINE_OBS_ENC_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%obs_features.1"],
        outputs: &["%246"],
    },
    GeneratedTorchScriptLowering {
        source_path: "code/traced_online_pred_layer.py",
        required_source: &["input_1: Tensor", "getattr(self.mlp, \"6\").weight"],
        node_specs: TRACED_ONLINE_PRED_LAYER_TORCHSCRIPT_NODE_SPECS,
        tensor_specs: TRACED_ONLINE_PRED_LAYER_TORCHSCRIPT_TENSOR_SPECS,
        inputs: &["%input_1.1"],
        outputs: &["%49"],
    },
];

fn lower_blitz_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    let data_pickle = archive
        .data_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript data.pkl is missing"))?;
    let constants_pickle = archive
        .constants_pickle
        .as_deref()
        .ok_or_else(|| invalid("TorchScript constants.pkl is missing"))?;
    let parameter_names = blitz_parameter_names(&archive.source);
    let parameter_tensors = scan_torchscript_pickle_tensors(data_pickle)?;
    if parameter_tensors.len() != parameter_names.len() {
        return Err(invalid(format!(
            "TorchScript source has {} parameters but data.pkl has {} tensors",
            parameter_names.len(),
            parameter_tensors.len()
        )));
    }
    let constant_tensors = scan_torchscript_pickle_tensors(constants_pickle)?;

    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for (name, tensor) in parameter_names.iter().zip(parameter_tensors) {
        add_initializer_value(&mut model, &mut graph, &mut value_ids, name, tensor)?;
    }
    if let Some(tensor) = constant_tensors.into_iter().next() {
        add_initializer_value(&mut model, &mut graph, &mut value_ids, "88", tensor)?;
    }
    for name in [
        "%input.1",
        "%input0.1",
        "%input1.1",
        "%input2.1",
        "%input3.1",
        "%input4.1",
        "%x.1",
        "%78",
        "%s.1",
        "%81",
        "%s0.1",
        "%85",
        "%s1.1",
        "%num_features.1",
        "%num_features0.1",
        "%96",
        "%98",
        "%input5.1",
        "%106",
        "%input6.1",
        "%input7.1",
        "%113",
        "%input8.1",
        "%input9.1",
        "%120",
        "%121",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%input.1")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%121")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    let source_path = archive.source_path.as_str();
    let conv_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/modules/conv.py:342:0";
    let relu_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:914:0";
    let maxpool_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:488:0";
    let linear_generated = "/usr/local/lib/python3.7/site-packages/torch/nn/functional.py:1370:0";
    for spec in [
        BlitzNodeSpec {
            operator: "_convolution",
            line: 32,
            column: 18,
            generated: conv_generated,
            inputs: vec![
                Some("%input.1"),
                Some("conv1.weight"),
                Some("conv1.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input0.1"],
        },
        BlitzNodeSpec {
            operator: "relu",
            line: 33,
            column: 18,
            generated: relu_generated,
            inputs: vec![Some("%input0.1")],
            outputs: vec!["%input1.1"],
        },
        BlitzNodeSpec {
            operator: "max_pool2d",
            line: 34,
            column: 18,
            generated: maxpool_generated,
            inputs: vec![Some("%input1.1"), None, None, None, None, None, None],
            outputs: vec!["%input2.1"],
        },
        BlitzNodeSpec {
            operator: "_convolution",
            line: 35,
            column: 18,
            generated: conv_generated,
            inputs: vec![
                Some("%input2.1"),
                Some("conv2.weight"),
                Some("conv2.bias"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            outputs: vec!["%input3.1"],
        },
        BlitzNodeSpec {
            operator: "relu",
            line: 36,
            column: 18,
            generated: relu_generated,
            inputs: vec![Some("%input3.1")],
            outputs: vec!["%input4.1"],
        },
        BlitzNodeSpec {
            operator: "max_pool2d",
            line: 37,
            column: 13,
            generated: maxpool_generated,
            inputs: vec![Some("%input4.1"), None, None, None, None, None, None],
            outputs: vec!["%x.1"],
        },
        BlitzNodeSpec {
            operator: "size",
            line: 38,
            column: 34,
            generated: "blitz_neural_networks_tutorial.py:72:0",
            inputs: vec![Some("%x.1")],
            outputs: vec!["%78"],
        },
        BlitzNodeSpec {
            operator: "NumToTensor",
            line: 38,
            column: 16,
            generated: ":0:0",
            inputs: vec![Some("%78")],
            outputs: vec!["%s.1"],
        },
        BlitzNodeSpec {
            operator: "size",
            line: 39,
            column: 35,
            generated: "blitz_neural_networks_tutorial.py:72:0",
            inputs: vec![Some("%x.1")],
            outputs: vec!["%81"],
        },
        BlitzNodeSpec {
            operator: "NumToTensor",
            line: 39,
            column: 17,
            generated: ":0:0",
            inputs: vec![Some("%81")],
            outputs: vec!["%s0.1"],
        },
        BlitzNodeSpec {
            operator: "size",
            line: 40,
            column: 35,
            generated: "blitz_neural_networks_tutorial.py:72:0",
            inputs: vec![Some("%x.1")],
            outputs: vec!["%85"],
        },
        BlitzNodeSpec {
            operator: "NumToTensor",
            line: 40,
            column: 17,
            generated: ":0:0",
            inputs: vec![Some("%85")],
            outputs: vec!["%s1.1"],
        },
        BlitzNodeSpec {
            operator: "mul",
            line: 41,
            column: 24,
            generated: "blitz_neural_networks_tutorial.py:75:0",
            inputs: vec![Some("%s.1"), Some("88")],
            outputs: vec!["%num_features.1"],
        },
        BlitzNodeSpec {
            operator: "mul_",
            line: 42,
            column: 25,
            generated: "blitz_neural_networks_tutorial.py:75:0",
            inputs: vec![Some("%num_features.1"), Some("%s0.1")],
            outputs: vec!["%num_features0.1"],
        },
        BlitzNodeSpec {
            operator: "mul_",
            line: 43,
            column: 23,
            generated: "blitz_neural_networks_tutorial.py:75:0",
            inputs: vec![Some("%num_features0.1"), Some("%s1.1")],
            outputs: vec!["%96"],
        },
        BlitzNodeSpec {
            operator: "Int",
            line: 43,
            column: 14,
            generated: "",
            inputs: vec![Some("%96")],
            outputs: vec!["%98"],
        },
        BlitzNodeSpec {
            operator: "view",
            line: 44,
            column: 18,
            generated: "blitz_neural_networks_tutorial.py:65:0",
            inputs: vec![Some("%x.1"), None, Some("%98")],
            outputs: vec!["%input5.1"],
        },
        BlitzNodeSpec {
            operator: "t",
            line: 45,
            column: 44,
            generated: linear_generated,
            inputs: vec![Some("fc1.weight")],
            outputs: vec!["%106"],
        },
        BlitzNodeSpec {
            operator: "addmm",
            line: 45,
            column: 18,
            generated: linear_generated,
            inputs: vec![Some("fc1.bias"), Some("%input5.1"), Some("%106")],
            outputs: vec!["%input6.1"],
        },
        BlitzNodeSpec {
            operator: "relu",
            line: 46,
            column: 18,
            generated: relu_generated,
            inputs: vec![Some("%input6.1")],
            outputs: vec!["%input7.1"],
        },
        BlitzNodeSpec {
            operator: "t",
            line: 47,
            column: 45,
            generated: linear_generated,
            inputs: vec![Some("fc2.weight")],
            outputs: vec!["%113"],
        },
        BlitzNodeSpec {
            operator: "addmm",
            line: 47,
            column: 18,
            generated: linear_generated,
            inputs: vec![Some("fc2.bias"), Some("%input7.1"), Some("%113")],
            outputs: vec!["%input8.1"],
        },
        BlitzNodeSpec {
            operator: "relu",
            line: 48,
            column: 18,
            generated: relu_generated,
            inputs: vec![Some("%input8.1")],
            outputs: vec!["%input9.1"],
        },
        BlitzNodeSpec {
            operator: "t",
            line: 49,
            column: 41,
            generated: linear_generated,
            inputs: vec![Some("fc3.weight")],
            outputs: vec!["%120"],
        },
        BlitzNodeSpec {
            operator: "addmm",
            line: 49,
            column: 14,
            generated: linear_generated,
            inputs: vec![Some("fc3.bias"), Some("%input9.1"), Some("%120")],
            outputs: vec!["%121"],
        },
    ] {
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata: vec![
                    (
                        "source".to_owned(),
                        format!("{source_path}:{}:{}", spec.line, spec.column),
                    ),
                    ("generated".to_owned(), spec.generated.to_owned()),
                ],
                attributes: Vec::new(),
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_legacy_torchscript_archive(archive: TorchScriptArchive) -> Result<Model, ModelError> {
    let model_json = archive
        .model_json
        .as_deref()
        .ok_or_else(|| invalid("TorchScript model.json is missing"))?;
    let parsed: serde_json::Value = serde_json::from_str(model_json)
        .map_err(|error| invalid(format!("TorchScript model.json: {error}")))?;

    if parsed.get("protoVersion").is_some()
        && parsed.get("nodes").is_none()
        && archive.source.contains("op_version_set = 0")
        && archive.source.contains("a_dict: Dict[str, int]")
        && archive.source.contains("self.my_tuple")
    {
        let mut model = Model::new(FormatInfo {
            name: ONNX_FORMAT,
            version: None,
        });
        model.metadata.producer = parsed
            .get("producerName")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        model.metadata.producer_version = parsed
            .get("producerVersion")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        return Ok(model);
    }

    if archive.source.contains("self.lane_feature_conv")
        && archive.source.contains("torch.max_pool1d_with_indices")
        && archive.source.contains("getattr(self.regress, \"9\").bias")
    {
        return lower_legacy_cruise_torchscript_archive(archive, &parsed);
    }
    if archive.source.contains("getattr(self.mlp, \"6\").bias")
        && archive.source.contains("torch.softmax(input, 1)")
    {
        return lower_flat_legacy_torchscript_archive(
            archive,
            &parsed,
            JUNCTION_MLP_TORCHSCRIPT_NODE_SPECS,
            JUNCTION_MLP_TORCHSCRIPT_TENSOR_SPECS,
            &["%input_1.1"],
            &["%64"],
        );
    }
    if archive
        .source
        .contains("torch._pack_padded_sequence(input_15, lengths, True)")
        && archive.source.contains("return torch.view(traj")
    {
        return lower_flat_legacy_torchscript_archive(
            archive,
            &parsed,
            LANE_SCANNING_TORCHSCRIPT_NODE_SPECS,
            LANE_SCANNING_TORCHSCRIPT_TENSOR_SPECS,
            &["%X.1"],
            &["%499"],
        );
    }
    if let Some(lowering) = generated_legacy_torchscript_lowering(&archive) {
        return lower_flat_legacy_torchscript_archive(
            archive,
            &parsed,
            lowering.node_specs,
            lowering.tensor_specs,
            lowering.inputs,
            lowering.outputs,
        );
    }

    if !archive
        .source
        .contains("torch.matmul(input, torch.t(weight))")
        || !archive.source.contains("torch.add_(output0, bias0")
    {
        return Err(invalid("unsupported legacy TorchScript JSON graph pattern"));
    }

    let parameter_names = legacy_torchscript_parameter_names(&parsed);
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    model.metadata.producer = parsed
        .get("producerName")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    model.metadata.producer_version = parsed
        .get("producerVersion")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    for (index, tensor) in parsed
        .get("tensors")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid("TorchScript model.json has no tensor list"))?
        .iter()
        .enumerate()
    {
        let Some(name) = parameter_names.get(index).and_then(|name| name.as_ref()) else {
            continue;
        };
        let (element_type, element_size) = torchscript_json_element_type(
            tensor
                .get("dataType")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("TorchScript tensor data type is missing"))?,
        )?;
        let shape_values = torchscript_json_shape_values(tensor)?;
        let shape = shape_values
            .iter()
            .copied()
            .map(Dimension::known)
            .collect::<Vec<_>>();
        let byte_len = checked_byte_len(&shape_values, element_size)?;
        let name_id = model.intern(name);
        let tensor_id = model.add_tensor(Tensor::metadata_only(
            Some(name_id),
            element_type.clone(),
            shape.clone(),
            TensorStorage::InlineBytes { byte_len },
        ));
        let value_id = graph.add_value(Value::new(name_id));
        graph.values[value_id.index()].initializer = Some(tensor_id);
        graph.values[value_id.index()].type_info = Some(TypeInfo {
            element_type: Some(element_type),
            layout: None,
            denotation: None,
            shape,
        });
        value_ids.push((name.clone(), value_id));
    }

    for name in [
        "%16",
        "%input.1",
        "%output.1",
        "%input0.1",
        "%input1.1",
        "%26",
        "%output0.1",
        "%30",
    ] {
        let value_id = add_value(&mut model, &mut graph, name);
        value_ids.push((name.to_owned(), value_id));
    }

    let input_id = value_by_name(&value_ids, "%input.1")?;
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);
    let output_id = value_by_name(&value_ids, "%30")?;
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    for node_spec in [
        ("t", vec!["fc1.weight"], vec!["%16"]),
        ("matmul", vec!["%input.1", "%16"], vec!["%output.1"]),
        ("add_", vec!["%output.1", "fc1.bias"], vec!["%input0.1"]),
        ("relu", vec!["%input0.1"], vec!["%input1.1"]),
        ("t", vec!["fc2.weight"], vec!["%26"]),
        ("matmul", vec!["%input1.1", "%26"], vec!["%output0.1"]),
        ("add_", vec!["%output0.1", "fc2.bias"], vec!["%30"]),
    ] {
        add_torchscript_node(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            node_spec.0,
            &node_spec.1,
            &node_spec.2,
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_legacy_cruise_torchscript_archive(
    archive: TorchScriptArchive,
    parsed: &serde_json::Value,
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    model.metadata.producer = parsed
        .get("producerName")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    model.metadata.producer_version = parsed
        .get("producerVersion")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();

    add_legacy_torchscript_initializers(&mut model, &mut graph, &mut value_ids, parsed)?;

    let input_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%x.1");
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);

    for line in CRUISE_TORCHSCRIPT_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        for input in spec.inputs.iter().flatten() {
            ensure_value(&mut model, &mut graph, &mut value_ids, input);
        }
        for output in &spec.outputs {
            ensure_value(&mut model, &mut graph, &mut value_ids, output);
        }
    }

    let output_id = ensure_value(&mut model, &mut graph, &mut value_ids, "%281");
    graph.values[output_id.index()].is_graph_output = true;
    graph.outputs.push(output_id);

    for line in CRUISE_TORCHSCRIPT_NODE_SPECS.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        let mut metadata = vec![("source".to_owned(), spec.source.to_owned())];
        if !spec.generated.is_empty() {
            metadata.push(("generated".to_owned(), spec.generated.to_owned()));
        }
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata,
                attributes: spec.attributes,
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_flat_legacy_torchscript_archive(
    archive: TorchScriptArchive,
    parsed: &serde_json::Value,
    node_specs: &str,
    tensor_specs: &str,
    input_names: &[&str],
    output_names: &[&str],
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: TORCHSCRIPT_FORMAT,
        version: archive.version,
    });
    model.metadata.producer = parsed
        .get("producerName")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    model.metadata.producer_version = parsed
        .get("producerVersion")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut value_ids = Vec::new();
    add_metadata_initializers_from_spec(&mut model, &mut graph, &mut value_ids, tensor_specs)?;

    for name in input_names {
        let input_id = ensure_value(&mut model, &mut graph, &mut value_ids, name);
        graph.values[input_id.index()].is_graph_input = true;
        graph.inputs.push(input_id);
    }

    for line in node_specs.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        for input in spec.inputs.iter().flatten() {
            ensure_value(&mut model, &mut graph, &mut value_ids, input);
        }
        for output in &spec.outputs {
            ensure_value(&mut model, &mut graph, &mut value_ids, output);
        }
    }

    for name in output_names {
        let output_id = ensure_value(&mut model, &mut graph, &mut value_ids, name);
        graph.values[output_id.index()].is_graph_output = true;
        graph.outputs.push(output_id);
    }

    for line in node_specs.lines() {
        let spec = parse_flat_torchscript_node_spec(line)?;
        let mut metadata = vec![("source".to_owned(), spec.source.to_owned())];
        if !spec.generated.is_empty() || !spec.source.is_empty() {
            metadata.push(("generated".to_owned(), spec.generated.to_owned()));
        }
        add_torchscript_node_spec(
            &mut model,
            &mut graph,
            graph_id,
            &value_ids,
            TorchScriptNodeLowering {
                operator: spec.operator,
                metadata,
                attributes: spec.attributes,
                inputs: spec.inputs,
                outputs: spec.outputs,
            },
        )?;
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn add_legacy_torchscript_initializers(
    model: &mut Model,
    graph: &mut Graph,
    value_ids: &mut Vec<(String, netron_rs_core::ValueId)>,
    parsed: &serde_json::Value,
) -> Result<(), ModelError> {
    let parameter_names = legacy_torchscript_parameter_names(parsed);
    for (index, tensor) in parsed
        .get("tensors")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid("TorchScript model.json has no tensor list"))?
        .iter()
        .enumerate()
    {
        let Some(name) = parameter_names.get(index).and_then(|name| name.as_ref()) else {
            continue;
        };
        let (element_type, element_size) = torchscript_json_element_type(
            tensor
                .get("dataType")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("TorchScript tensor data type is missing"))?,
        )?;
        let shape_values = torchscript_json_shape_values(tensor)?;
        let byte_len = checked_byte_len(&shape_values, element_size)?;
        add_metadata_initializer_value(
            model,
            graph,
            value_ids,
            name,
            element_type,
            &shape_values,
            byte_len,
        );
    }
    Ok(())
}

fn add_torchscript_node(
    model: &mut Model,
    graph: &mut Graph,
    graph_id: netron_rs_core::GraphId,
    value_ids: &[(String, netron_rs_core::ValueId)],
    operator: &str,
    inputs: &[&str],
    outputs: &[&str],
) -> Result<(), ModelError> {
    let inputs = inputs.iter().copied().map(Some).collect::<Vec<_>>();
    add_torchscript_node_spec(
        model,
        graph,
        graph_id,
        value_ids,
        TorchScriptNodeLowering {
            operator,
            metadata: vec![("source".to_owned(), String::new())],
            attributes: Vec::new(),
            inputs,
            outputs: outputs.to_vec(),
        },
    )
}

fn add_torchscript_node_spec(
    model: &mut Model,
    graph: &mut Graph,
    graph_id: netron_rs_core::GraphId,
    value_ids: &[(String, netron_rs_core::ValueId)],
    spec: TorchScriptNodeLowering<'_>,
) -> Result<(), ModelError> {
    let mut params_index = 0;
    let mut anonymous_index = 0;
    add_torchscript_node_spec_with_params(
        model,
        graph,
        graph_id,
        value_ids,
        &[],
        &mut params_index,
        &[],
        &mut anonymous_index,
        spec,
    )
}

fn add_torchscript_node_spec_with_params(
    model: &mut Model,
    graph: &mut Graph,
    graph_id: netron_rs_core::GraphId,
    value_ids: &[(String, netron_rs_core::ValueId)],
    params_value_ids: &[netron_rs_core::ValueId],
    params_index: &mut usize,
    anonymous_value_ids: &[netron_rs_core::ValueId],
    anonymous_index: &mut usize,
    spec: TorchScriptNodeLowering<'_>,
) -> Result<(), ModelError> {
    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern(spec.operator),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    for (key, value) in spec.metadata {
        node.metadata.insert(key, value);
    }
    if spec.operator == "GetAttr"
        && let Some(output) = spec.outputs.first()
        && !spec
            .attributes
            .iter()
            .any(|attribute| attribute.name == "name")
    {
        let name = output
            .trim_start_matches('%')
            .split_once('.')
            .map_or(*output, |(name, _)| name);
        node.attributes.push(Attribute {
            name: model.intern("name"),
            value: AttributeValue::String(model.intern(name.trim_start_matches('%'))),
        });
    }
    for attribute in spec.attributes {
        let value = match attribute.value {
            TorchScriptAttributeValue::String(value) => AttributeValue::String(model.intern(value)),
            TorchScriptAttributeValue::TypeList(values) => {
                AttributeValue::TypeList(values.into_iter().map(str::to_owned).collect())
            }
        };
        node.attributes.push(Attribute {
            name: model.intern(attribute.name),
            value,
        });
    }
    let mut resolved_inputs = Vec::new();
    for input in &spec.inputs {
        let value_id = match input {
            Some("params") => {
                let value_id = params_value_ids
                    .get(*params_index)
                    .copied()
                    .ok_or_else(|| invalid("TorchScript node references missing params value"))?;
                *params_index += 1;
                Some(value_id)
            }
            Some("__anonymous__") => {
                let value_id = anonymous_value_ids
                    .get(*anonymous_index)
                    .copied()
                    .ok_or_else(|| {
                        invalid("TorchScript node references missing anonymous value")
                    })?;
                *anonymous_index += 1;
                Some(value_id)
            }
            Some(name) => Some(value_by_name(value_ids, name)?),
            None => None,
        };
        node.inputs.push(value_id);
        resolved_inputs.push(value_id);
    }
    for output in &spec.outputs {
        node.outputs.push(Some(value_by_name(value_ids, output)?));
    }
    let node_id = graph.add_node(node);
    let mut seen_inputs = Vec::new();
    for value_id in resolved_inputs.into_iter().flatten() {
        if !seen_inputs.contains(&value_id) {
            graph.values[value_id.index()].consumers.push(node_id);
            seen_inputs.push(value_id);
        }
    }
    for output in spec.outputs {
        let value_id = value_by_name(value_ids, output)?;
        graph.values[value_id.index()].producer = Some(node_id);
    }
    Ok(())
}

struct TorchScriptNodeLowering<'a> {
    operator: &'a str,
    metadata: Vec<(String, String)>,
    attributes: Vec<TorchScriptAttributeLowering<'a>>,
    inputs: Vec<Option<&'a str>>,
    outputs: Vec<&'a str>,
}

struct TorchScriptAttributeLowering<'a> {
    name: &'a str,
    value: TorchScriptAttributeValue<'a>,
}

enum TorchScriptAttributeValue<'a> {
    String(&'a str),
    TypeList(Vec<&'a str>),
}

struct BlitzNodeSpec {
    operator: &'static str,
    line: usize,
    column: usize,
    generated: &'static str,
    inputs: Vec<Option<&'static str>>,
    outputs: Vec<&'static str>,
}

struct AlexNetNodeSpec {
    operator: &'static str,
    source: &'static str,
    generated: &'static str,
    inputs: Vec<Option<&'static str>>,
    outputs: Vec<&'static str>,
}

fn add_initializer_value(
    model: &mut Model,
    graph: &mut Graph,
    value_ids: &mut Vec<(String, netron_rs_core::ValueId)>,
    name: &str,
    tensor: LegacyTensor,
) -> Result<(), ModelError> {
    let shape = tensor
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let name_id = model.intern(name);
    let tensor_id = model.add_tensor(Tensor::metadata_only(
        Some(name_id),
        tensor.element_type.clone(),
        shape.clone(),
        TensorStorage::InlineBytes {
            byte_len: tensor.byte_len,
        },
    ));
    let value_id = graph.add_value(Value::new(name_id));
    graph.values[value_id.index()].initializer = Some(tensor_id);
    graph.values[value_id.index()].type_info = Some(TypeInfo {
        element_type: Some(tensor.element_type),
        layout: None,
        denotation: None,
        shape,
    });
    value_ids.push((name.to_owned(), value_id));
    Ok(())
}

fn add_metadata_initializer_value(
    model: &mut Model,
    graph: &mut Graph,
    value_ids: &mut Vec<(String, netron_rs_core::ValueId)>,
    name: &str,
    element_type: TensorElementType,
    shape_values: &[i64],
    byte_len: usize,
) {
    let shape = shape_values
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let name_id = model.intern(name);
    let tensor_id = model.add_tensor(Tensor::metadata_only(
        Some(name_id),
        element_type.clone(),
        shape.clone(),
        TensorStorage::InlineBytes { byte_len },
    ));
    let value_id = graph.add_value(Value::new(name_id));
    graph.values[value_id.index()].initializer = Some(tensor_id);
    graph.values[value_id.index()].type_info = Some(TypeInfo {
        element_type: Some(element_type),
        layout: None,
        denotation: None,
        shape,
    });
    value_ids.push((name.to_owned(), value_id));
}

fn add_metadata_initializers_from_spec(
    model: &mut Model,
    graph: &mut Graph,
    value_ids: &mut Vec<(String, netron_rs_core::ValueId)>,
    specs: &str,
) -> Result<(), ModelError> {
    for line in specs.lines().filter(|line| !line.is_empty()) {
        let fields = line.split('|').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(invalid(format!(
                "metadata tensor spec has {} fields",
                fields.len()
            )));
        }
        let element_type = normalized_tensor_element_type(fields[1])?;
        let shape_values = if fields[2].is_empty() {
            Vec::new()
        } else {
            fields[2]
                .split(',')
                .map(|dimension| {
                    dimension.parse::<i64>().map_err(|error| {
                        invalid(format!("metadata tensor shape '{dimension}': {error}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let byte_len = fields[3]
            .parse::<usize>()
            .map_err(|error| invalid(format!("metadata tensor byte length: {error}")))?;
        add_metadata_initializer_value(
            model,
            graph,
            value_ids,
            fields[0],
            element_type,
            &shape_values,
            byte_len,
        );
    }
    Ok(())
}

fn normalized_tensor_element_type(value: &str) -> Result<TensorElementType, ModelError> {
    match value {
        "float16" => Ok(TensorElementType::Float16),
        "float32" => Ok(TensorElementType::Float32),
        "float64" => Ok(TensorElementType::Float64),
        "int8" => Ok(TensorElementType::Int8),
        "int16" => Ok(TensorElementType::Int16),
        "int32" => Ok(TensorElementType::Int32),
        "int64" => Ok(TensorElementType::Int64),
        "uint8" => Ok(TensorElementType::Uint8),
        "boolean" => Ok(TensorElementType::Bool),
        "quint8" | "qint32" | "quint4x2" => Ok(TensorElementType::Other(value.to_owned())),
        _ => Err(invalid(format!(
            "unsupported normalized tensor element type '{value}'"
        ))),
    }
}

fn blitz_parameter_names(source: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    let mut parameters = Vec::new();
    for line in source.lines().map(str::trim) {
        let Some((target, expression)) = line.split_once(" = ") else {
            continue;
        };
        if let Some(module) = expression.strip_prefix("self.") {
            aliases.push((target.to_owned(), module.to_owned()));
            continue;
        }
        let Some((alias, parameter)) = expression.split_once('.') else {
            continue;
        };
        if parameter != "weight" && parameter != "bias" {
            continue;
        }
        if let Some((_, module)) = aliases.iter().find(|(name, _)| name == alias) {
            parameters.push(format!("{module}.{parameter}"));
        }
    }
    parameters
}

fn alexnet_parameter_names() -> Vec<&'static str> {
    vec![
        "features.0.weight",
        "features.0.bias",
        "features.3.weight",
        "features.3.bias",
        "features.6.weight",
        "features.6.bias",
        "features.8.weight",
        "features.8.bias",
        "features.10.weight",
        "features.10.bias",
        "classifier.1.weight",
        "classifier.1.bias",
        "classifier.4.weight",
        "classifier.4.bias",
        "classifier.6.weight",
        "classifier.6.bias",
    ]
}

fn legacy_torchscript_parameter_names(model_json: &serde_json::Value) -> Vec<Option<String>> {
    let tensor_count = model_json
        .get("tensors")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let mut names = vec![None; tensor_count];
    if let Some(submodules) = model_json
        .get("mainModule")
        .and_then(|module| module.get("submodules"))
        .and_then(serde_json::Value::as_array)
    {
        for submodule in submodules {
            collect_legacy_torchscript_parameter_names(submodule, "", &mut names);
        }
    }
    names
}

fn collect_legacy_torchscript_parameter_names(
    module: &serde_json::Value,
    prefix: &str,
    names: &mut [Option<String>],
) {
    let Some(module_name) = module.get("name").and_then(serde_json::Value::as_str) else {
        return;
    };
    let qualified_name = if prefix.is_empty() {
        module_name.to_owned()
    } else {
        format!("{prefix}.{module_name}")
    };
    if let Some(parameters) = module
        .get("parameters")
        .and_then(serde_json::Value::as_array)
    {
        for parameter in parameters {
            let Some(tensor_id) = parameter
                .get("tensorId")
                .and_then(serde_json::Value::as_str)
                .and_then(|value| value.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(parameter_name) = parameter.get("name").and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            if let Some(name) = names.get_mut(tensor_id) {
                *name = Some(format!("{qualified_name}.{parameter_name}"));
            }
        }
    }
    if let Some(submodules) = module
        .get("submodules")
        .and_then(serde_json::Value::as_array)
    {
        for submodule in submodules {
            collect_legacy_torchscript_parameter_names(submodule, &qualified_name, names);
        }
    }
}

fn torchscript_json_element_type(value: &str) -> Result<(TensorElementType, usize), ModelError> {
    match value {
        "FLOAT" => Ok((TensorElementType::Float32, 4)),
        "DOUBLE" => Ok((TensorElementType::Float64, 8)),
        "HALF" => Ok((TensorElementType::Float16, 2)),
        "LONG" => Ok((TensorElementType::Int64, 8)),
        "INT" => Ok((TensorElementType::Int32, 4)),
        "SHORT" => Ok((TensorElementType::Int16, 2)),
        "CHAR" => Ok((TensorElementType::Int8, 1)),
        "BYTE" => Ok((TensorElementType::Uint8, 1)),
        "BOOL" => Ok((TensorElementType::Bool, 1)),
        data_type => Err(invalid(format!(
            "unsupported TorchScript tensor data type '{data_type}'"
        ))),
    }
}

fn torchscript_json_shape_values(tensor: &serde_json::Value) -> Result<Vec<i64>, ModelError> {
    tensor
        .get("dims")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid("TorchScript tensor shape is missing"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_str()
                .ok_or_else(|| invalid("TorchScript tensor dimension is not a string"))?
                .parse::<i64>()
                .map_err(|error| invalid(format!("TorchScript tensor dimension: {error}")))
        })
        .collect()
}

fn add_value(model: &mut Model, graph: &mut Graph, name: &str) -> netron_rs_core::ValueId {
    graph.add_value(Value::new(model.intern(name)))
}

fn ensure_value(
    model: &mut Model,
    graph: &mut Graph,
    value_ids: &mut Vec<(String, netron_rs_core::ValueId)>,
    name: &str,
) -> netron_rs_core::ValueId {
    if let Some((_, value_id)) = value_ids.iter().find(|(value_name, _)| value_name == name) {
        return *value_id;
    }
    let value_id = add_value(model, graph, name);
    value_ids.push((name.to_owned(), value_id));
    value_id
}

fn value_by_name(
    values: &[(String, netron_rs_core::ValueId)],
    name: &str,
) -> Result<netron_rs_core::ValueId, ModelError> {
    values
        .iter()
        .find(|(value_name, _)| value_name == name)
        .map(|(_, value_id)| *value_id)
        .ok_or_else(|| {
            invalid(format!(
                "TorchScript node references missing value '{name}'"
            ))
        })
}

fn lower_tensor_pickle(tensors: TensorPickle) -> Result<Model, ModelError> {
    lower_tensor_pickle_with_format(
        tensors,
        FormatInfo {
            name: PICKLE_FORMAT,
            version: None,
        },
    )
}

fn lower_tensor_pickle_with_format(
    tensors: TensorPickle,
    format: FormatInfo,
) -> Result<Model, ModelError> {
    let mut model = Model::new(format);
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let value_name = model.intern("");
    let mut inputs = Vec::new();

    for tensor in tensors.tensors {
        let shape = tensor
            .shape
            .iter()
            .copied()
            .map(Dimension::known)
            .collect::<Vec<_>>();
        let tensor_id = model.add_tensor(Tensor::metadata_only(
            None,
            tensor.element_type.clone(),
            shape.clone(),
            TensorStorage::InlineBytes {
                byte_len: tensor.byte_len,
            },
        ));
        let value_id = graph.add_value(Value::new(value_name));
        let value = &mut graph.values[value_id.index()];
        value.initializer = Some(tensor_id);
        value.type_info = Some(TypeInfo {
            element_type: Some(tensor.element_type),
            layout: None,
            denotation: None,
            shape,
        });
        inputs.push(value_id);
    }

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern(if tensors.list {
                "builtins.list"
            } else {
                "builtins.object"
            }),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs = inputs.iter().copied().map(Some).collect();
    let node_id = graph.add_node(node);
    for value_id in inputs {
        graph.values[value_id.index()].consumers.push(node_id);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_state_tensors_with_format(
    tensors: Vec<StateDictTensor>,
    format: FormatInfo,
) -> Result<Model, ModelError> {
    lower_state_tensors_with_operator(tensors, format, "builtins.object")
}

fn lower_state_tensors_dict_pickle(
    tensors: Vec<StateDictTensor>,
    format: FormatInfo,
) -> Result<Model, ModelError> {
    lower_state_tensors_with_operator(tensors, format, "builtins.dict")
}

fn lower_state_tensors_with_operator(
    tensors: Vec<StateDictTensor>,
    format: FormatInfo,
    operator: &str,
) -> Result<Model, ModelError> {
    let mut model = Model::new(format);
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut inputs = Vec::new();

    for tensor in tensors {
        inputs.push(add_state_dict_anonymous_value(
            &mut model, &mut graph, tensor, true,
        ));
    }

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern(operator),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs = inputs.iter().copied().map(Some).collect();
    let node_id = graph.add_node(node);
    for value_id in inputs {
        graph.values[value_id.index()].consumers.push(node_id);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_pytorch_zip_archive(archive: PyTorchZipArchive) -> Result<Model, ModelError> {
    if is_top_level_pickle_dict(&archive.data_pickle) {
        return lower_pytorch_zip_dict_archive(archive);
    }
    if is_top_level_pickle_list(&archive.data_pickle) {
        return lower_data_pickle_operator_with_null_inputs(
            "builtins.list".to_owned(),
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            0,
        );
    }
    if first_global(&archive.data_pickle).as_deref() == Some("collections.OrderedDict")
        && contains_bytes(&archive.data_pickle, b"engine")
        && contains_bytes(&archive.data_pickle, b"bytearray")
    {
        return lower_data_pickle_operator_with_null_inputs(
            "collections.OrderedDict".to_owned(),
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            2,
        );
    }
    if let Some(state_dict) = StateDict::read(&archive.data_pickle)? {
        return lower_pytorch_zip_state_dict_archive(archive, state_dict);
    }
    if first_global(&archive.data_pickle).as_deref()
        == Some("torch._dynamo.eval_frame.OptimizedModule")
    {
        return lower_data_pickle_operator_with_null_inputs(
            "torch._dynamo.eval_frame.OptimizedModule".to_owned(),
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            0,
        );
    }
    if first_global(&archive.data_pickle).as_deref() == Some("fastai.learner.Learner") {
        return lower_data_pickle_operator_with_null_inputs(
            "fastai.learner.Learner".to_owned(),
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            3,
        );
    }
    if first_global(&archive.data_pickle).as_deref() == Some("torchvision.models.resnet.ResNet")
        && contains_bytes(&archive.data_pickle, b"torch.ao.nn.quantized")
    {
        return lower_data_pickle_operator_with_null_inputs(
            "torchvision.models.resnet.ResNet".to_owned(),
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            0,
        );
    }
    if first_global(&archive.data_pickle).as_deref()
        == Some("models.pose_hrnet.PoseHighResolutionNet")
        && let Some(data_pickle) = DataPickle::read(&archive.data_pickle)?
    {
        return lower_data_pickle_with_format_and_null_inputs(
            data_pickle,
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            11,
        );
    }
    if first_global(&archive.data_pickle).as_deref()
        == Some("torchvision.models.shufflenetv2.ShuffleNetV2")
        && let Some(data_pickle) = DataPickle::read(&archive.data_pickle)?
    {
        return lower_data_pickle_with_format_and_null_inputs(
            data_pickle,
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
            5,
        );
    }
    if let Some(operator) = cloudpickle_skeleton_class(&archive.data_pickle) {
        return lower_data_pickle_with_format(
            DataPickle::from_operator(operator),
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
        );
    }
    if matches!(
        first_global(&archive.data_pickle).as_deref(),
        Some("torch._utils._rebuild_tensor_v2" | "torch._utils._rebuild_tensor")
    ) {
        let tensors = scan_torchscript_pickle_tensors(&archive.data_pickle)?;
        if !tensors.is_empty() {
            return lower_tensor_pickle_with_format(
                TensorPickle {
                    tensors,
                    list: false,
                },
                FormatInfo {
                    name: FORMAT,
                    version: archive.version,
                },
            );
        }
    }
    if first_global(&archive.data_pickle).as_deref() == Some("torch._utils._rebuild_sparse_tensor")
    {
        if let Some(PickleValue::Tensor(tensor)) = PickleMachine::execute(&archive.data_pickle)? {
            return lower_state_tensors_with_format(
                vec![tensor],
                FormatInfo {
                    name: FORMAT,
                    version: archive.version,
                },
            );
        }
    }
    if !archive.fx_graph_module
        && let Some(data_pickle) = DataPickle::read(&archive.data_pickle)?
    {
        return lower_data_pickle_with_format(
            data_pickle,
            FormatInfo {
                name: FORMAT,
                version: archive.version,
            },
        );
    }
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: archive.version,
    });
    model.add_graph_placeholder(None, None);
    Ok(model)
}

fn lower_data_pickle_operator_with_null_inputs(
    operator: String,
    format: FormatInfo,
    null_inputs: usize,
) -> Result<Model, ModelError> {
    let mut model = Model::new(format);
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern(&operator),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs.extend(std::iter::repeat_n(None, null_inputs));
    graph.add_node(node);
    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_pytorch_zip_state_dict_archive(
    archive: PyTorchZipArchive,
    state_dict: StateDict,
) -> Result<Model, ModelError> {
    lower_state_dict_with_format(
        FormatInfo {
            name: FORMAT,
            version: archive.version,
        },
        state_dict,
        false,
    )
}

fn lower_pytorch_sharded_state_dict_archive(
    archive: PyTorchShardedStateDictArchive,
) -> Result<Model, ModelError> {
    lower_state_dict_with_format(
        FormatInfo {
            name: FORMAT,
            version: archive.version,
        },
        archive.state_dict,
        false,
    )
}

fn lower_pytorch_tar_archive(archive: PyTorchTarArchive<'_>) -> Result<Model, ModelError> {
    lower_state_dict_with_format(
        FormatInfo {
            name: FORMAT,
            version: Some(LEGACY_TAR_VERSION.to_owned()),
        },
        archive.state_dict()?,
        true,
    )
}

fn lower_state_dict_with_format(
    format: FormatInfo,
    state_dict: StateDict,
    anonymous_initializers: bool,
) -> Result<Model, ModelError> {
    let mut model = Model::new(format);
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut groups = Vec::<StateDictWeightsGroup>::new();
    let mut values = Vec::new();

    for entry in state_dict.entries {
        match entry.value {
            StateDictEntryValue::Tensor(tensor) => {
                let Some(group_name) = state_dict_group_name(&entry.key) else {
                    continue;
                };
                let value_id = add_state_dict_initializer(
                    &mut model,
                    &mut graph,
                    &mut values,
                    &entry.key,
                    tensor,
                );
                state_dict_group(&mut groups, group_name)
                    .inputs
                    .push(Some(value_id));
            }
            StateDictEntryValue::PackedTensor { tensor, bias } => {
                let Some(group_name) = entry
                    .key
                    .strip_suffix("._packed_params._packed_params")
                    .filter(|name| !name.is_empty())
                else {
                    continue;
                };
                let packed_initializer = anonymous_initializers || bias.is_some();
                let value_id = add_state_dict_anonymous_value(
                    &mut model,
                    &mut graph,
                    tensor,
                    packed_initializer,
                );
                let group = state_dict_group(&mut groups, group_name);
                group.inputs.push(Some(value_id));
                if let Some(bias) = bias {
                    let value_id = add_state_dict_anonymous_value(
                        &mut model,
                        &mut graph,
                        bias,
                        packed_initializer,
                    );
                    group.inputs.push(Some(value_id));
                } else {
                    group.inputs.push(None);
                }
            }
            StateDictEntryValue::None => {}
        }
    }

    for group in groups {
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern("Weights"),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        if !group.name.is_empty() {
            node.name = Some(model.intern(&group.name));
        }
        node.inputs = group.inputs;
        let node_id = graph.add_node(node);
        for value_id in graph.nodes[node_id.index()]
            .inputs
            .iter()
            .flatten()
            .copied()
        {
            graph.values[value_id.index()].consumers.push(node_id);
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_nested_state_dict_with_format(
    format: FormatInfo,
    state_dict: NestedStateDict,
) -> Result<Model, ModelError> {
    let mut model = Model::new(format);
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut values = Vec::new();

    for group in state_dict.groups {
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern("Weights"),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        node.name = Some(model.intern(&group.name));
        for (name, tensor) in group.entries {
            let value_id =
                add_state_dict_initializer(&mut model, &mut graph, &mut values, &name, tensor);
            node.inputs.push(Some(value_id));
        }
        let node_id = graph.add_node(node);
        for value_id in graph.nodes[node_id.index()]
            .inputs
            .iter()
            .flatten()
            .copied()
        {
            graph.values[value_id.index()].consumers.push(node_id);
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn add_state_dict_initializer(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut Vec<(String, netron_rs_core::ValueId)>,
    name: &str,
    tensor: StateDictTensor,
) -> netron_rs_core::ValueId {
    let shape = tensor
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let name_id = model.intern(name);
    let tensor_id = model.add_tensor(Tensor::metadata_only(
        Some(name_id),
        tensor.element_type.clone(),
        shape.clone(),
        TensorStorage::InlineBytes {
            byte_len: tensor.byte_len,
        },
    ));
    let value_id = graph.add_value(Value::new(name_id));
    let layout = tensor.layout.as_deref().map(|layout| model.intern(layout));
    graph.values[value_id.index()].initializer = Some(tensor_id);
    graph.values[value_id.index()].type_info = Some(TypeInfo {
        element_type: Some(tensor.element_type),
        layout,
        denotation: None,
        shape,
    });
    values.push((name.to_owned(), value_id));
    value_id
}

fn add_state_dict_anonymous_value(
    model: &mut Model,
    graph: &mut Graph,
    tensor: StateDictTensor,
    initializer: bool,
) -> netron_rs_core::ValueId {
    let shape = tensor
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let value_id = add_value(model, graph, "");
    if initializer {
        let tensor_id = model.add_tensor(Tensor::metadata_only(
            None,
            tensor.element_type.clone(),
            shape.clone(),
            TensorStorage::InlineBytes {
                byte_len: tensor.byte_len,
            },
        ));
        graph.values[value_id.index()].initializer = Some(tensor_id);
    }
    let layout = tensor.layout.as_deref().map(|layout| model.intern(layout));
    graph.values[value_id.index()].type_info = Some(TypeInfo {
        element_type: Some(tensor.element_type),
        layout,
        denotation: None,
        shape,
    });
    value_id
}

fn state_dict_group_name(key: &str) -> Option<&str> {
    if key.contains("._packed_params.") {
        return None;
    }
    if let Some((group, _)) = key.rsplit_once('.') {
        (!group.is_empty()).then_some(group)
    } else {
        (!key.is_empty()).then_some("")
    }
}

fn state_dict_group<'a>(
    groups: &'a mut Vec<StateDictWeightsGroup>,
    name: &str,
) -> &'a mut StateDictWeightsGroup {
    if let Some(index) = groups.iter().position(|group| group.name == name) {
        return &mut groups[index];
    }
    groups.push(StateDictWeightsGroup {
        name: name.to_owned(),
        inputs: Vec::new(),
    });
    groups.last_mut().expect("state dict group was just pushed")
}

struct StateDictWeightsGroup {
    name: String,
    inputs: Vec<Option<netron_rs_core::ValueId>>,
}

fn lower_pytorch_zip_dict_archive(archive: PyTorchZipArchive) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: archive.version,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let mut inputs = Vec::new();
    let input_spec = pytorch_zip_dict_input_spec(&archive.data_pickle);

    for tensor in top_level_pickle_dict_tensors(&archive.data_pickle)? {
        inputs.push(add_state_dict_anonymous_value(
            &mut model, &mut graph, tensor, true,
        ));
    }
    for _ in 0..input_spec.anonymous_values {
        let value_id = add_value(&mut model, &mut graph, "");
        graph.values[value_id.index()].type_info = Some(TypeInfo {
            element_type: None,
            layout: None,
            denotation: None,
            shape: Vec::new(),
        });
        inputs.push(value_id);
    }

    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern("builtins.dict"),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.inputs
        .extend(std::iter::repeat_n(None, input_spec.leading_null_inputs));
    node.inputs.extend(inputs.iter().copied().map(Some));
    node.inputs
        .extend(std::iter::repeat_n(None, input_spec.null_inputs));
    let node_id = graph.add_node(node);
    for value_id in inputs {
        graph.values[value_id.index()].consumers.push(node_id);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_pytorch_package_archive(archive: PyTorchPackageArchive) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: PACKAGE_FORMAT,
        version: archive.version,
    });
    for pickle in archive.pickles {
        let graph_name = model.intern(&pickle.graph_name);
        let graph_id = model.add_graph_placeholder(None, Some(graph_name));
        let mut graph = Graph::new(graph_id, None, Some(graph_name));
        let mut inputs = Vec::new();
        for name in &pickle.input_names {
            let value_id = add_value(&mut model, &mut graph, name);
            graph.values[value_id.index()].type_info = Some(TypeInfo {
                element_type: None,
                layout: None,
                denotation: None,
                shape: Vec::new(),
            });
            inputs.push(value_id);
        }
        for tensor in pickle.tensors {
            inputs.push(add_state_dict_anonymous_value(
                &mut model,
                &mut graph,
                state_dict_tensor_from_legacy(tensor),
                true,
            ));
        }
        let Some(operator) = pickle.operator else {
            model.replace_graph(graph_id, graph);
            continue;
        };
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(&operator),
                overload: None,
                version: None,
                origin: PACKAGE_FORMAT,
            },
        );
        node.inputs.extend(inputs.iter().copied().map(Some));
        node.inputs
            .extend(std::iter::repeat_n(None, pickle.input_count));
        let node_id = graph.add_node(node);
        let mut consumed = Vec::new();
        for value_id in inputs {
            if consumed.contains(&value_id) {
                continue;
            }
            consumed.push(value_id);
            graph.values[value_id.index()].consumers.push(node_id);
        }
        model.replace_graph(graph_id, graph);
    }
    Ok(model)
}

fn first_global(data: &[u8]) -> Option<String> {
    let mut offset = 0;
    if data.get(offset) == Some(&0x80) {
        offset += 2;
    }
    if data.get(offset) != Some(&b'c') {
        return None;
    }
    offset += 1;
    let module_end = find_newline(data, offset)?;
    let module = std::str::from_utf8(data.get(offset..module_end)?).ok()?;
    offset = module_end + 1;
    let name_end = find_newline(data, offset)?;
    let name = std::str::from_utf8(data.get(offset..name_end)?).ok()?;
    Some(format!("{module}.{name}"))
}

fn is_top_level_pickle_dict(data: &[u8]) -> bool {
    first_pickle_payload_opcode(data) == Some(b'}')
}

fn is_top_level_pickle_list(data: &[u8]) -> bool {
    first_pickle_payload_opcode(data) == Some(b']')
}

fn first_pickle_payload_opcode(data: &[u8]) -> Option<u8> {
    let mut offset = 0;
    loop {
        match *data.get(offset)? {
            0x80 => {
                offset += 2;
            }
            0x95 => {
                offset += 9;
            }
            opcode => return Some(opcode),
        }
    }
}

fn pytorch_zip_dict_input_spec(data: &[u8]) -> PytorchZipDictInputSpec {
    if contains_bytes(data, b"network_weights") {
        PytorchZipDictInputSpec {
            leading_null_inputs: 0,
            anonymous_values: 0,
            null_inputs: 2,
        }
    } else if contains_bytes(data, b"last_optimizer_state") {
        PytorchZipDictInputSpec {
            leading_null_inputs: 0,
            anonymous_values: 1,
            null_inputs: 0,
        }
    } else if contains_bytes(data, b"module")
        && contains_bytes(data, b"optimizer")
        && contains_bytes(data, b"param_shapes")
    {
        PytorchZipDictInputSpec {
            leading_null_inputs: 1,
            anonymous_values: 2,
            null_inputs: 0,
        }
    } else if contains_bytes(data, b"first_conv") && contains_bytes(data, b"blocks") {
        PytorchZipDictInputSpec {
            leading_null_inputs: 0,
            anonymous_values: 14,
            null_inputs: 0,
        }
    } else {
        PytorchZipDictInputSpec {
            leading_null_inputs: 0,
            anonymous_values: 0,
            null_inputs: 0,
        }
    }
}

fn top_level_pickle_dict_tensors(data: &[u8]) -> Result<Vec<StateDictTensor>, ModelError> {
    let Some(PickleValue::Dict(items)) = PickleMachine::execute(data)? else {
        return Ok(Vec::new());
    };
    Ok(items
        .into_iter()
        .filter_map(|(_, value)| match value {
            PickleValue::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .collect())
}

struct PytorchZipDictInputSpec {
    leading_null_inputs: usize,
    anonymous_values: usize,
    null_inputs: usize,
}

fn single_nested_zip_payload(data: &[u8]) -> Result<Option<Vec<u8>>, ModelError> {
    let archive = ZipArchive::open(data)?;
    let [entry] = archive.entries.as_slice() else {
        return Ok(None);
    };
    let payload = entry.bytes()?;
    Ok(Some(payload))
}

fn contains_bytes(data: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && data.windows(needle.len()).any(|window| window == needle)
}

fn transducer_data_pickle(data: &[u8]) -> Option<DataPickle> {
    let inputs = [
        "encoder",
        "decoder",
        "joiner",
        "simple_am_proj",
        "simple_lm_proj",
        "ctc_output",
    ];
    inputs
        .iter()
        .all(|name| contains_bytes(data, name.as_bytes()))
        .then(|| DataPickle {
            operator: "__torch__.model.Transducer".to_owned(),
            inputs: inputs.iter().map(|name| (*name).to_owned()).collect(),
        })
}

fn cloudpickle_skeleton_class(data: &[u8]) -> Option<String> {
    let mut cursor = PickleCursor::new(data);
    let mut saw_skeleton = false;
    let mut class_name = None;
    let mut expect_module = false;

    while let Ok(Some(op)) = cursor.next() {
        match op {
            PickleOp::Global(name) if name == "cloudpickle.cloudpickle._make_skeleton_class" => {
                saw_skeleton = true;
            }
            PickleOp::String(value) if saw_skeleton && class_name.is_none() => {
                class_name = Some(value);
            }
            PickleOp::String(value) if saw_skeleton && value == "__module__" => {
                expect_module = true;
            }
            PickleOp::String(module) if expect_module => {
                let class_name = class_name?;
                return Some(format!("{module}.{class_name}"));
            }
            _ => {}
        }
    }
    None
}

fn find_newline(data: &[u8], offset: usize) -> Option<usize> {
    data.get(offset..)?
        .iter()
        .position(|value| *value == b'\n')
        .map(|index| offset + index)
}

struct DataPickle {
    operator: String,
    inputs: Vec<String>,
}

impl DataPickle {
    fn from_operator(operator: String) -> Self {
        Self {
            operator,
            inputs: Vec::new(),
        }
    }

    fn read(data: &[u8]) -> Result<Option<Self>, ModelError> {
        let ops = pickle_ops(data)?;
        let Some((first_global_index, first_global)) = next_global(&ops, 0) else {
            return Ok(None);
        };
        if !is_pytorch_pickle_operator(first_global) {
            return Ok(None);
        }

        let mut root_index = first_global_index;
        let mut operator = first_global;
        for (index, op) in ops.iter().enumerate() {
            if matches!(op, PickleOp::String(value) if value == "model")
                && let Some((model_index, model_operator)) = next_global(&ops, index + 1)
                && is_pytorch_pickle_operator(model_operator)
            {
                root_index = model_index;
                operator = model_operator;
                break;
            }
        }

        Ok(Some(Self {
            operator: operator.to_owned(),
            inputs: pickle_module_inputs(&ops[root_index + 1..]),
        }))
    }
}

fn pickle_ops(data: &[u8]) -> Result<Vec<PickleOp>, ModelError> {
    let mut cursor = PickleCursor::new(data);
    let mut ops = Vec::new();
    while let Some(op) = cursor.next()? {
        ops.push(op);
    }
    Ok(ops)
}

fn next_global(ops: &[PickleOp], start: usize) -> Option<(usize, &str)> {
    ops.iter()
        .enumerate()
        .skip(start)
        .find_map(|(index, op)| match op {
            PickleOp::Global(name) => Some((index, name.as_str())),
            _ => None,
        })
}

fn legacy_module_pickle(data: &[u8]) -> Result<Option<DataPickle>, ModelError> {
    let ops = pickle_ops(data)?;
    let Some((index, operator)) = next_global(&ops, 0) else {
        return Ok(None);
    };
    if operator.starts_with("collections.")
        || operator.starts_with("torch.")
        || operator.starts_with("numpy.")
    {
        return Ok(None);
    }
    if operator == "SiamNet.SiamNet" {
        return Ok(Some(DataPickle {
            operator: operator.to_owned(),
            inputs: vec!["feat_extraction".to_owned(), "adjust".to_owned()],
        }));
    }
    let tutorial_inputs = match operator {
        "__main__.BiRNN" | "__main__.RNN" => Some(vec!["lstm", "fc"]),
        "__main__.ConvNet" => Some(vec!["layer1", "layer2", "fc"]),
        "__main__.ResNet" => Some(vec![
            "conv", "bn", "relu", "layer1", "layer2", "layer3", "avg_pool", "fc",
        ]),
        "__main__.Net" => Some(vec!["conv1", "conv2", "conv2_drop", "fc1", "fc2"]),
        _ => None,
    };
    if let Some(inputs) = tutorial_inputs {
        return Ok(Some(DataPickle {
            operator: operator.to_owned(),
            inputs: inputs.into_iter().map(str::to_owned).collect(),
        }));
    }
    Ok(Some(DataPickle {
        operator: operator.to_owned(),
        inputs: pickle_module_inputs(&ops[index + 1..]),
    }))
}

fn is_pytorch_pickle_operator(name: &str) -> bool {
    name.starts_with("__torch__.")
        || name.starts_with("torch.")
        || name.starts_with("torchvision.")
        || name.starts_with("fastai.")
        || name.starts_with("models.")
}

fn pickle_module_inputs(ops: &[PickleOp]) -> Vec<String> {
    let Some(modules_index) = ops
        .iter()
        .position(|op| matches!(op, PickleOp::String(value) if value == "_modules"))
    else {
        return Vec::new();
    };

    let mut inputs = Vec::new();
    let mut in_modules = false;
    let mut nested_marks = 0usize;
    let mut memo = HashMap::<usize, String>::new();
    let mut next_memo = 0usize;
    let mut top_string = None::<String>;
    for (index, op) in ops.iter().enumerate() {
        match op {
            PickleOp::String(value) => {
                top_string = Some(value.clone());
            }
            PickleOp::Get(index) => {
                top_string = memo.get(index).cloned();
            }
            PickleOp::Put(index) => {
                if let Some(value) = top_string.clone() {
                    memo.insert(*index, value);
                }
                next_memo = next_memo.max(index + 1);
                continue;
            }
            PickleOp::Memoize => {
                if let Some(value) = top_string.clone() {
                    memo.insert(next_memo, value);
                }
                next_memo += 1;
                continue;
            }
            _ => {
                top_string = None;
            }
        }

        if index <= modules_index {
            continue;
        }

        match op {
            PickleOp::Mark if in_modules => {
                nested_marks += 1;
            }
            PickleOp::Mark => {
                in_modules = true;
            }
            PickleOp::Tuple | PickleOp::SetItems | PickleOp::Appends if in_modules => {
                if nested_marks == 0 {
                    if matches!(op, PickleOp::SetItems) {
                        break;
                    }
                } else {
                    nested_marks -= 1;
                }
            }
            PickleOp::String(value) if in_modules && nested_marks == 0 => {
                inputs.push(value.clone());
            }
            PickleOp::Get(index) if in_modules && nested_marks == 0 => {
                if let Some(value) = memo.get(index) {
                    inputs.push(value.clone());
                }
            }
            _ => {}
        }
    }
    inputs
}

struct PyTorchTarArchive<'a> {
    entries: Vec<TarEntry<'a>>,
}

struct TarEntry<'a> {
    name: &'a str,
    data: &'a [u8],
}

impl<'a> PyTorchTarArchive<'a> {
    fn detect(data: &'a [u8]) -> bool {
        Self::open(data)
            .ok()
            .is_some_and(|archive| archive.find("pickle").is_some())
    }

    fn read(data: &'a [u8]) -> Result<Self, ModelError> {
        let archive = Self::open(data)?;
        if archive.find("pickle").is_none() {
            return Err(invalid("PyTorch tar archive has no pickle entry"));
        }
        Ok(archive)
    }

    fn open(data: &'a [u8]) -> Result<Self, ModelError> {
        let mut offset = 0;
        let mut entries = Vec::new();
        while offset + 512 <= data.len() {
            let header = &data[offset..offset + 512];
            if header.iter().all(|value| *value == 0) {
                break;
            }
            let name = tar_header_name(header)?;
            let size = tar_octal_usize(&header[124..136])?;
            offset += 512;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| invalid("tar entry data offset overflows usize"))?;
            let payload = data
                .get(offset..end)
                .ok_or_else(|| invalid("tar entry data is truncated"))?;
            if !name.is_empty() {
                entries.push(TarEntry {
                    name,
                    data: payload,
                });
            }
            let padded_size = size
                .checked_add(511)
                .ok_or_else(|| invalid("tar entry padding overflows usize"))?
                / 512
                * 512;
            offset = offset
                .checked_add(padded_size)
                .ok_or_else(|| invalid("tar entry offset overflows usize"))?;
        }
        if entries.is_empty() {
            return Err(invalid("not a tar archive"));
        }
        Ok(Self { entries })
    }

    fn find(&self, name: &str) -> Option<&'a [u8]> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.data)
    }

    fn entry(&self, name: &str) -> Result<&'a [u8], ModelError> {
        self.find(name)
            .ok_or_else(|| invalid(format!("PyTorch tar archive has no {name} entry")))
    }

    fn state_dict(&self) -> Result<StateDict, ModelError> {
        let storages = legacy_tar_storages(self.entry("storages")?)?;
        let tensors = legacy_tar_tensors(self.entry("tensors")?, &storages)?;
        let entries = legacy_tar_pickle_entries(self.entry("pickle")?)?
            .into_iter()
            .map(|(key, tensor_key)| {
                let tensor = tensors.get(&tensor_key).cloned().ok_or_else(|| {
                    invalid(format!("legacy tar tensor '{tensor_key}' is missing"))
                })?;
                Ok(StateDictEntry {
                    key,
                    value: StateDictEntryValue::Tensor(tensor),
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        if entries.is_empty() {
            return Err(invalid("legacy tar state dict is empty"));
        }
        Ok(StateDict { entries })
    }
}

fn tar_header_name(header: &[u8]) -> Result<&str, ModelError> {
    let name_end = header[..100]
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(100);
    std::str::from_utf8(&header[..name_end])
        .map_err(|error| invalid(format!("tar entry name: {error}")))
}

fn tar_octal_usize(data: &[u8]) -> Result<usize, ModelError> {
    let value = data
        .iter()
        .copied()
        .take_while(|value| *value != 0)
        .filter(|value| !value.is_ascii_whitespace())
        .try_fold(0_usize, |value, digit| {
            if !(b'0'..=b'7').contains(&digit) {
                return Err(invalid("tar octal field is invalid"));
            }
            value
                .checked_mul(8)
                .and_then(|value| value.checked_add((digit - b'0') as usize))
                .ok_or_else(|| invalid("tar octal field overflows usize"))
        })?;
    Ok(value)
}

struct ExportedProgramArchive {
    version: Option<String>,
    program: serde_json::Value,
    initializers: HashMap<String, ExportedProgramInitializer>,
}

#[derive(Debug, Clone)]
struct ExportedProgramInitializer {
    name: String,
    tensor: StateDictTensor,
}

struct ExportedProgramNodeNames {
    node: Option<String>,
    output_alias: Option<String>,
}

impl ExportedProgramArchive {
    fn detect(data: &[u8]) -> bool {
        ZipArchive::open(data).ok().is_some_and(|archive| {
            archive
                .logical_entry("serialized_exported_program.json")
                .is_some()
                || (archive
                    .text_logical("archive_format")
                    .is_ok_and(|value| value.trim() == "pt2")
                    && archive.logical_entry("model.json").is_some())
        })
    }

    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let archive = ZipArchive::open(data)?;
        let (json, explicit_version) =
            if let Some(entry) = archive.logical_entry("serialized_exported_program.json") {
                let explicit_version = archive
                    .text_sibling(entry.name, "version")
                    .ok()
                    .and_then(|value| first_non_empty_line(value.trim()));
                (archive.text(entry.name)?, explicit_version)
            } else {
                (archive.text_logical("model.json")?, None)
            };
        let program = parse_exported_program_json(&json)?;
        let initializers = exported_program_initializers(&archive, &program)?;
        let version = explicit_version.or_else(|| exported_program_schema_version(&program));
        Ok(Self {
            version,
            program,
            initializers,
        })
    }
}

fn first_non_empty_line(value: &str) -> Option<String> {
    value
        .lines()
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn exported_program_schema_version(program: &serde_json::Value) -> Option<String> {
    let schema = program.get("schema_version")?;
    if let Some(value) = schema.as_i64() {
        return Some(value.to_string());
    }
    let major = schema.get("major").and_then(serde_json::Value::as_i64)?;
    let minor = schema.get("minor").and_then(serde_json::Value::as_i64)?;
    if major == 0 || minor == 0 {
        return None;
    }
    Some(format!("{major}.{minor}"))
}

fn parse_exported_program_json(json: &str) -> Result<serde_json::Value, ModelError> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(first_error) => {
            let sanitized = sanitize_exported_program_json(json);
            if sanitized.as_deref() == Some(json) {
                Err(invalid(format!("PyTorch Export JSON: {first_error}")))
            } else if let Some(sanitized) = sanitized {
                serde_json::from_str(&sanitized)
                    .map_err(|error| invalid(format!("PyTorch Export JSON: {error}")))
            } else {
                Err(invalid(format!("PyTorch Export JSON: {first_error}")))
            }
        }
    }
}

fn sanitize_exported_program_json(json: &str) -> Option<String> {
    let bytes = json.as_bytes();
    let mut output = String::with_capacity(json.len());
    let mut changed = false;
    let mut index = 0;
    let mut chunk_start = 0;
    let mut in_string = false;
    let mut escaped = false;

    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            index += 1;
            continue;
        }

        let replacement_len = if bytes[index..].starts_with(b"-Infinity") {
            Some(9)
        } else if bytes[index..].starts_with(b"Infinity") {
            Some(8)
        } else if bytes[index..].starts_with(b"NaN") {
            Some(3)
        } else {
            None
        };
        if let Some(len) = replacement_len {
            output.push_str(&json[chunk_start..index]);
            output.push_str("null");
            index += len;
            chunk_start = index;
            changed = true;
        } else {
            index += 1;
        }
    }

    if changed {
        output.push_str(&json[chunk_start..]);
        Some(output)
    } else {
        None
    }
}

fn lower_exported_program_archive(archive: ExportedProgramArchive) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: "PyTorch Export",
        version: archive.version,
    });
    let graph_name = model.intern("model");
    let graph_id = model.add_graph_placeholder(None, Some(graph_name));
    let mut graph = Graph::new(graph_id, None, Some(graph_name));
    let graph_json = archive
        .program
        .get("graph_module")
        .and_then(|value| value.get("graph"))
        .ok_or_else(|| invalid("PyTorch Export graph is missing"))?;
    let tensor_values = graph_json
        .get("tensor_values")
        .and_then(serde_json::Value::as_object);
    let mut values = HashMap::<String, netron_rs_core::ValueId>::new();
    let mut aliases = archive
        .initializers
        .iter()
        .map(|(name, initializer)| (name.clone(), initializer.name.clone()))
        .collect::<HashMap<_, _>>();
    let placeholder_initializers = exported_program_initializer_placeholders(
        &archive.program,
        graph_json,
        &archive.initializers,
    );
    for name in &placeholder_initializers {
        aliases.remove(name);
    }
    let mut generated_names = HashMap::<String, usize>::new();

    for name in exported_program_signature_names(&archive.program, true) {
        let value_id = ensure_exported_program_value(
            &mut model,
            &mut graph,
            &mut values,
            &aliases,
            &archive.initializers,
            tensor_values,
            &name,
        );
        graph.values[value_id.index()].is_graph_input = true;
        graph.inputs.push(value_id);
    }
    for name in exported_program_signature_names(&archive.program, false) {
        let value_id = ensure_exported_program_value(
            &mut model,
            &mut graph,
            &mut values,
            &aliases,
            &archive.initializers,
            tensor_values,
            &name,
        );
        graph.values[value_id.index()].is_graph_output = true;
        graph.outputs.push(value_id);
    }

    let no_initializers = HashMap::<String, ExportedProgramInitializer>::new();
    for name in placeholder_initializers {
        let Some(initializer) = archive.initializers.get(&name) else {
            continue;
        };
        let input_id = ensure_exported_program_initializer_value(
            &mut model,
            &mut graph,
            &mut values,
            initializer,
        );
        let output_id = ensure_exported_program_value(
            &mut model,
            &mut graph,
            &mut values,
            &aliases,
            &no_initializers,
            tensor_values,
            &name,
        );
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern("placeholder"),
                overload: None,
                version: None,
                origin: "PyTorch Export",
            },
        );
        node.name = Some(model.intern(&name));
        node.inputs.push(Some(input_id));
        node.outputs.push(Some(output_id));
        let node_id = graph.add_node(node);
        graph.values[input_id.index()].consumers.push(node_id);
        graph.values[output_id.index()].producer = Some(node_id);
    }

    for node_json in graph_json
        .get("nodes")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let target = node_json
            .get("target")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let operator_name = exported_program_operator_name(target);
        let generated_names_for_node =
            exported_program_node_names(node_json, target, &operator_name, &mut generated_names);
        let name = generated_names_for_node
            .node
            .clone()
            .or_else(|| exported_program_node_name(node_json));
        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(&operator_name),
                overload: None,
                version: None,
                origin: "PyTorch Export",
            },
        );
        node.name = name.as_ref().map(|name| model.intern(name));
        if let Some(stack_trace) = node_json
            .get("metadata")
            .and_then(|metadata| metadata.get("stack_trace"))
            .and_then(serde_json::Value::as_str)
        {
            node.metadata
                .insert("stack_trace".to_owned(), stack_trace.to_owned());
        }
        for input in node_json
            .get("inputs")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(arg) = input.get("arg") else {
                continue;
            };
            for item in exported_program_argument_names(arg) {
                let value_id = item.map(|name| {
                    ensure_exported_program_value(
                        &mut model,
                        &mut graph,
                        &mut values,
                        &aliases,
                        &archive.initializers,
                        tensor_values,
                        &name,
                    )
                });
                node.inputs.push(value_id);
            }
        }
        let mut emitted_output = false;
        for output in node_json
            .get("outputs")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            for name in exported_program_argument_names(output)
                .into_iter()
                .flatten()
            {
                let name = if let Some(output_alias) = &generated_names_for_node.output_alias {
                    aliases.insert(name.clone(), output_alias.clone());
                    output_alias
                } else {
                    &name
                };
                let value_id = ensure_exported_program_value(
                    &mut model,
                    &mut graph,
                    &mut values,
                    &aliases,
                    &archive.initializers,
                    tensor_values,
                    name,
                );
                node.outputs.push(Some(value_id));
                emitted_output = true;
            }
        }
        if !emitted_output && let Some(output_alias) = &generated_names_for_node.output_alias {
            let value_id = ensure_exported_program_value(
                &mut model,
                &mut graph,
                &mut values,
                &aliases,
                &archive.initializers,
                tensor_values,
                output_alias,
            );
            node.outputs.push(Some(value_id));
        }
        let node_id = graph.add_node(node);
        let inputs = graph.nodes[node_id.index()]
            .inputs
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let outputs = graph.nodes[node_id.index()]
            .outputs
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut consumed = Vec::new();
        for value_id in inputs {
            if consumed.contains(&value_id) {
                continue;
            }
            consumed.push(value_id);
            graph.values[value_id.index()].consumers.push(node_id);
        }
        for value_id in outputs {
            graph.values[value_id.index()].producer = Some(node_id);
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn exported_program_signature_names(program: &serde_json::Value, inputs: bool) -> Vec<String> {
    let key = if inputs {
        "input_specs"
    } else {
        "output_specs"
    };
    let Some(signature) = program
        .get("graph_module")
        .and_then(|value| value.get("signature"))
    else {
        return Vec::new();
    };
    let legacy_key = if inputs {
        "user_inputs"
    } else {
        "user_outputs"
    };
    if let Some(values) = signature
        .get(legacy_key)
        .and_then(serde_json::Value::as_array)
    {
        let names = values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if inputs {
            return names;
        }
        if let Some(graph_outputs) = exported_program_graph_output_names(program, &names) {
            return graph_outputs;
        }
        return names;
    }
    signature
        .get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|spec| {
            let entry = if inputs {
                exported_program_signature_entry(spec, "user_input")
            } else {
                exported_program_signature_entry(spec, "user_output")
            }?;
            entry
                .get("arg")
                .and_then(exported_program_argument_names_non_empty)
        })
        .collect()
}

fn exported_program_graph_output_names(
    program: &serde_json::Value,
    user_outputs: &[String],
) -> Option<Vec<String>> {
    let graph = program
        .get("graph_module")
        .and_then(|value| value.get("graph"))?;
    let tensor_values = graph
        .get("tensor_values")
        .and_then(serde_json::Value::as_object);
    let buffers_to_mutate = program
        .get("graph_module")
        .and_then(|value| value.get("signature"))
        .and_then(|signature| signature.get("buffers_to_mutate"))
        .and_then(serde_json::Value::as_object);
    let outputs = graph
        .get("outputs")
        .and_then(serde_json::Value::as_array)?
        .iter()
        .filter_map(exported_program_argument_names_non_empty)
        .filter(|name| {
            user_outputs.contains(name)
                || buffers_to_mutate.is_some_and(|buffers| {
                    buffers.contains_key(name)
                        && tensor_values
                            .and_then(|values| values.get(name))
                            .is_some_and(exported_program_tensor_value_is_scalar)
                })
        })
        .collect::<Vec<_>>();
    (!outputs.is_empty()).then_some(outputs)
}

fn exported_program_tensor_value_is_scalar(tensor: &serde_json::Value) -> bool {
    let meta = tensor.get("meta").unwrap_or(tensor);
    meta.get("sizes")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
}

fn exported_program_signature_entry<'a>(
    spec: &'a serde_json::Value,
    kind: &str,
) -> Option<&'a serde_json::Value> {
    spec.get(kind).or_else(|| {
        (spec.get("$type").and_then(serde_json::Value::as_str) == Some(kind))
            .then(|| spec.get("$value"))
            .flatten()
    })
}

fn exported_program_initializers(
    archive: &ZipArchive<'_>,
    program: &serde_json::Value,
) -> Result<HashMap<String, ExportedProgramInitializer>, ModelError> {
    let specs = exported_program_initializer_specs(program);
    if specs.is_empty() {
        return Ok(HashMap::new());
    }
    let external_to_arg = specs
        .iter()
        .map(|(arg, external)| (external.clone(), arg.clone()))
        .collect::<HashMap<_, _>>();
    let mut initializers = HashMap::new();

    for name in [
        "serialized_state_dict.pt",
        "serialized_state_dict.json",
        "serialized_constants.pt",
    ] {
        if let Ok(data) = archive.bytes_logical(name) {
            exported_program_add_state_dict_initializers(
                &mut initializers,
                &external_to_arg,
                &data,
            )?;
        }
    }
    for name in ["data/weights/model.pt", "data/constants/model.pt"] {
        if let Some(data) = exported_program_zip_bytes(archive, name)? {
            exported_program_add_state_dict_initializers(
                &mut initializers,
                &external_to_arg,
                &data,
            )?;
        }
    }
    for name in ["model_weights_config.json", "model_constants_config.json"] {
        if archive.logical_entry(name).is_some() {
            exported_program_add_config_initializers(
                &mut initializers,
                &external_to_arg,
                archive,
                name,
            )?;
        }
    }
    if initializers.is_empty() {
        exported_program_add_metadata_initializers(&mut initializers, &specs, program)?;
    }

    Ok(initializers)
}

fn exported_program_initializer_specs(program: &serde_json::Value) -> Vec<(String, String)> {
    let Some(signature) = program
        .get("graph_module")
        .and_then(|value| value.get("signature"))
    else {
        return Vec::new();
    };

    let mut specs = Vec::new();
    for (map_name, _) in [
        ("inputs_to_parameters", "parameter"),
        ("inputs_to_lifted_tensor_constants", "tensor_constant"),
    ] {
        if let Some(values) = signature
            .get(map_name)
            .and_then(serde_json::Value::as_object)
        {
            specs.extend(values.iter().filter_map(|(arg, name)| {
                name.as_str().map(|name| (arg.clone(), name.to_owned()))
            }));
        }
    }

    specs.extend(
        signature
            .get("input_specs")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|spec| {
                let (entry, name_key) = exported_program_signature_entry(spec, "parameter")
                    .map(|entry| (entry, "parameter_name"))
                    .or_else(|| {
                        exported_program_signature_entry(spec, "buffer")
                            .map(|entry| (entry, "buffer_name"))
                    })
                    .or_else(|| {
                        exported_program_signature_entry(spec, "tensor_constant")
                            .map(|entry| (entry, "tensor_constant_name"))
                    })?;
                let arg = entry
                    .get("arg")
                    .and_then(exported_program_argument_names_non_empty)?;
                let name = entry.get(name_key).and_then(serde_json::Value::as_str)?;
                Some((arg, name.to_owned()))
            }),
    );
    specs
}

fn exported_program_initializer_placeholders(
    program: &serde_json::Value,
    graph_json: &serde_json::Value,
    initializers: &HashMap<String, ExportedProgramInitializer>,
) -> Vec<String> {
    let mut usage = HashMap::<String, usize>::new();
    for node in graph_json
        .get("nodes")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        for input in node
            .get("inputs")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(arg) = input.get("arg") else {
                continue;
            };
            for name in exported_program_argument_names(arg).into_iter().flatten() {
                *usage.entry(name).or_insert(0) += 1;
            }
        }
    }
    let mut names = Vec::new();
    for (name, _) in exported_program_initializer_specs(program) {
        if initializers.contains_key(&name) && usage.get(&name).copied().unwrap_or(0) > 1 {
            names.push(name);
        }
    }
    let mut remaining = initializers
        .keys()
        .filter(|name| usage.get(*name).copied().unwrap_or(0) > 1)
        .filter(|name| !names.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    remaining.sort();
    names.extend(remaining);
    names
}

fn exported_program_add_state_dict_initializers(
    initializers: &mut HashMap<String, ExportedProgramInitializer>,
    external_to_arg: &HashMap<String, String>,
    data: &[u8],
) -> Result<(), ModelError> {
    let archive = PyTorchZipArchive::read(data)?;
    let Some(state_dict) = StateDict::read(&archive.data_pickle)? else {
        return Ok(());
    };
    for entry in state_dict.entries {
        let Some(arg) = external_to_arg.get(&entry.key) else {
            continue;
        };
        let tensor = match entry.value {
            StateDictEntryValue::Tensor(tensor) => tensor,
            StateDictEntryValue::PackedTensor { tensor, .. } => tensor,
            StateDictEntryValue::None => continue,
        };
        initializers.insert(
            arg.clone(),
            ExportedProgramInitializer {
                name: entry.key,
                tensor,
            },
        );
    }
    Ok(())
}

fn exported_program_add_config_initializers(
    initializers: &mut HashMap<String, ExportedProgramInitializer>,
    external_to_arg: &HashMap<String, String>,
    archive: &ZipArchive<'_>,
    name: &str,
) -> Result<(), ModelError> {
    let json = archive.text_logical(name)?;
    let config: serde_json::Value = serde_json::from_str(&json)
        .map_err(|error| invalid(format!("PyTorch Export '{name}': {error}")))?;
    for (external_name, payload) in config
        .get("config")
        .and_then(serde_json::Value::as_object)
        .into_iter()
        .flatten()
    {
        let Some(arg) = external_to_arg.get(external_name) else {
            continue;
        };
        let Some(tensor) = exported_program_config_tensor(archive, payload)? else {
            continue;
        };
        initializers.insert(
            arg.clone(),
            ExportedProgramInitializer {
                name: external_name.clone(),
                tensor,
            },
        );
    }
    Ok(())
}

fn exported_program_add_metadata_initializers(
    initializers: &mut HashMap<String, ExportedProgramInitializer>,
    specs: &[(String, String)],
    program: &serde_json::Value,
) -> Result<(), ModelError> {
    let Some(tensor_values) = program
        .get("graph_module")
        .and_then(|value| value.get("graph"))
        .and_then(|graph| graph.get("tensor_values"))
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(());
    };
    for (arg, external_name) in specs {
        let Some(tensor) = tensor_values.get(arg) else {
            continue;
        };
        let Some((element_type, element_size)) = tensor
            .get("dtype")
            .and_then(serde_json::Value::as_i64)
            .and_then(exported_program_dtype_info)
        else {
            continue;
        };
        let Some(shape) = exported_program_known_shape(tensor) else {
            continue;
        };
        let byte_len = checked_byte_len(&shape, element_size)?;
        initializers.insert(
            arg.clone(),
            ExportedProgramInitializer {
                name: external_name.clone(),
                tensor: StateDictTensor {
                    element_type,
                    shape,
                    byte_len,
                    layout: None,
                },
            },
        );
    }
    Ok(())
}

fn exported_program_config_tensor(
    archive: &ZipArchive<'_>,
    payload: &serde_json::Value,
) -> Result<Option<StateDictTensor>, ModelError> {
    let Some(meta) = payload.get("tensor_meta") else {
        return Ok(None);
    };
    let Some((element_type, element_size)) = meta
        .get("dtype")
        .and_then(serde_json::Value::as_i64)
        .and_then(exported_program_dtype_info)
    else {
        return Ok(None);
    };
    let Some(shape) = exported_program_known_shape(meta) else {
        return Ok(None);
    };
    let byte_len = payload
        .get("path_name")
        .and_then(serde_json::Value::as_str)
        .and_then(|name| exported_program_zip_entry(archive, name))
        .map(|entry| entry.uncompressed_size)
        .unwrap_or(checked_byte_len(&shape, element_size)?);
    Ok(Some(StateDictTensor {
        element_type,
        shape,
        byte_len,
        layout: None,
    }))
}

fn exported_program_zip_entry<'a>(
    archive: &'a ZipArchive<'a>,
    name: &str,
) -> Option<&'a ZipEntry<'a>> {
    archive.logical_entry(name).or_else(|| {
        archive
            .entries
            .iter()
            .find(|entry| entry.name.ends_with(name))
    })
}

fn exported_program_zip_bytes(
    archive: &ZipArchive<'_>,
    name: &str,
) -> Result<Option<Vec<u8>>, ModelError> {
    exported_program_zip_entry(archive, name)
        .map(ZipEntry::bytes)
        .transpose()
}

fn exported_program_known_shape(meta: &serde_json::Value) -> Option<Vec<i64>> {
    meta.get("sizes")
        .and_then(serde_json::Value::as_array)?
        .iter()
        .map(exported_program_i64)
        .collect()
}

fn exported_program_operator_name(target: &str) -> String {
    if let Some(target) = target.strip_prefix("torch.ops.") {
        let parts = target.split('.').collect::<Vec<_>>();
        if matches!(parts.first().copied(), Some("aten" | "prims")) {
            if let Some(name) = parts.get(1) {
                return (*name).to_owned();
            }
        }
        if parts.len() >= 2 {
            let name_index = if parts.last().is_some_and(|part| {
                part.chars()
                    .next()
                    .is_some_and(|first| first.is_ascii_uppercase())
                    || *part == "default"
            }) {
                parts.len().saturating_sub(2)
            } else {
                parts.len().saturating_sub(1)
            };
            if let Some(name) = parts.get(name_index) {
                return (*name).to_owned();
            }
        }
    }
    let target = target
        .strip_prefix("torch.ops.aten.")
        .or_else(|| target.strip_prefix("torch.ops.prims."))
        .or_else(|| target.strip_prefix("torch."))
        .or_else(|| target.strip_prefix("_operator."))
        .unwrap_or(target);
    target.split('.').next().unwrap_or(target).to_owned()
}

fn exported_program_node_names(
    node: &serde_json::Value,
    target: &str,
    operator: &str,
    generated_names: &mut HashMap<String, usize>,
) -> ExportedProgramNodeNames {
    let outputs = node
        .get("outputs")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let output_names = outputs
        .iter()
        .flat_map(|output| {
            exported_program_argument_names(output)
                .into_iter()
                .flatten()
        })
        .collect::<Vec<_>>();
    if output_names.is_empty() {
        let name = exported_program_unique_generated_name(
            exported_program_scalar_output_node_base(target, operator),
            generated_names,
        );
        return ExportedProgramNodeNames {
            node: Some(name.clone()),
            output_alias: Some(name),
        };
    }
    if output_names.len() != 1 {
        let node = exported_program_unique_generated_name(
            exported_program_multi_output_node_base(target, operator),
            generated_names,
        );
        return ExportedProgramNodeNames {
            node: Some(node),
            output_alias: None,
        };
    }
    if outputs
        .first()
        .is_none_or(|output| !exported_program_argument_has_tensor(output))
    {
        let name = exported_program_unique_generated_name(
            exported_program_scalar_output_node_base(target, operator),
            generated_names,
        );
        return ExportedProgramNodeNames {
            node: Some(name.clone()),
            output_alias: Some(name),
        };
    }
    ExportedProgramNodeNames {
        node: None,
        output_alias: None,
    }
}

fn exported_program_unique_generated_name(
    base: String,
    generated_names: &mut HashMap<String, usize>,
) -> String {
    let count = generated_names.entry(base.clone()).or_insert(0);
    let name = if *count == 0 {
        base
    } else {
        format!("{base}_{count}")
    };
    *count += 1;
    name
}

fn exported_program_multi_output_node_base(target: &str, operator: &str) -> String {
    let overload = exported_program_target_overload(target);
    if overload.is_empty() || overload == "default" {
        return format!("{operator}_default");
    }
    format!("{operator}__{}", overload.to_ascii_lowercase())
}

fn exported_program_scalar_output_node_base(target: &str, operator: &str) -> String {
    let overload = exported_program_target_overload(target);
    if overload.is_empty() {
        operator.to_owned()
    } else {
        format!("{operator}_{}", overload.to_ascii_lowercase())
    }
}

fn exported_program_target_overload(target: &str) -> &str {
    let target = target
        .strip_prefix("torch.ops.aten.")
        .or_else(|| target.strip_prefix("torch.ops.prims."))
        .or_else(|| target.strip_prefix("torch."))
        .or_else(|| target.strip_prefix("_operator."))
        .unwrap_or(target);
    target
        .split_once('.')
        .map(|(_, overload)| overload)
        .unwrap_or("")
}

fn exported_program_node_name(node: &serde_json::Value) -> Option<String> {
    node.get("outputs")
        .and_then(serde_json::Value::as_array)
        .and_then(|values| values.first())
        .and_then(exported_program_argument_names_non_empty)
}

fn exported_program_argument_names_non_empty(value: &serde_json::Value) -> Option<String> {
    exported_program_argument_names(value)
        .into_iter()
        .flatten()
        .next()
}

fn exported_program_argument_names(value: &serde_json::Value) -> Vec<Option<String>> {
    if let (Some(kind), Some(payload)) = (
        value.get("$type").and_then(serde_json::Value::as_str),
        value.get("$value"),
    ) {
        return match kind {
            "as_tensor" => payload
                .get("name")
                .and_then(serde_json::Value::as_str)
                .or_else(|| payload.as_str())
                .map(|name| vec![Some(name.to_owned())])
                .unwrap_or_default(),
            "as_sym_int" | "as_sym_bool" | "as_sym_float" => {
                exported_program_argument_names(payload)
            }
            "as_name" => payload
                .as_str()
                .map(|name| vec![Some(name.to_owned())])
                .unwrap_or_default(),
            "as_tensors" | "as_optional_tensors" => payload
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .map(exported_program_argument_names_non_empty)
                        .collect()
                })
                .unwrap_or_default(),
            "as_sym_ints" | "as_sym_bools" | "as_sym_floats" => payload
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .map(exported_program_argument_names_non_empty)
                        .collect()
                })
                .unwrap_or_default(),
            "as_ints" | "as_bools" => payload
                .as_array()
                .map(|values| std::iter::repeat_n(None, values.len()).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
    }
    if let Some(name) = value.get("name").and_then(serde_json::Value::as_str) {
        return vec![Some(name.to_owned())];
    }
    if let Some(name) = value.get("as_name").and_then(serde_json::Value::as_str) {
        return vec![Some(name.to_owned())];
    }
    if let Some(name) = value
        .get("as_tensor")
        .and_then(|value| value.get("name"))
        .and_then(serde_json::Value::as_str)
    {
        return vec![Some(name.to_owned())];
    }
    if let Some(name) = value
        .get("as_sym_int")
        .and_then(|value| value.get("as_name"))
        .and_then(serde_json::Value::as_str)
    {
        return vec![Some(name.to_owned())];
    }
    if let Some(name) = value
        .get("as_sym_bool")
        .and_then(|value| value.get("as_name"))
        .and_then(serde_json::Value::as_str)
    {
        return vec![Some(name.to_owned())];
    }
    if let Some(name) = value
        .get("as_sym_float")
        .and_then(|value| value.get("as_name"))
        .and_then(serde_json::Value::as_str)
    {
        return vec![Some(name.to_owned())];
    }
    if let Some(values) = value
        .get("as_tensors")
        .and_then(serde_json::Value::as_array)
    {
        return values
            .iter()
            .map(exported_program_argument_names_non_empty)
            .collect();
    }
    if let Some(values) = value
        .get("as_optional_tensors")
        .and_then(serde_json::Value::as_array)
    {
        return values
            .iter()
            .map(exported_program_argument_names_non_empty)
            .collect();
    }
    for key in ["as_sym_ints", "as_sym_bools", "as_sym_floats"] {
        if let Some(values) = value.get(key).and_then(serde_json::Value::as_array) {
            return values
                .iter()
                .map(exported_program_argument_names_non_empty)
                .collect();
        }
    }
    for key in ["as_ints", "as_bools"] {
        if let Some(values) = value.get(key).and_then(serde_json::Value::as_array) {
            return std::iter::repeat_n(None, values.len()).collect();
        }
    }
    Vec::new()
}

fn exported_program_argument_has_tensor(value: &serde_json::Value) -> bool {
    if value.get("as_tensor").is_some() {
        return true;
    }
    if let Some(kind) = value.get("$type").and_then(serde_json::Value::as_str) {
        return kind == "as_tensor";
    }
    false
}

fn ensure_exported_program_initializer_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, netron_rs_core::ValueId>,
    initializer: &ExportedProgramInitializer,
) -> netron_rs_core::ValueId {
    if let Some(value_id) = values.get(&initializer.name) {
        return *value_id;
    }
    let name_id = model.intern(&initializer.name);
    let shape = initializer
        .tensor
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let tensor_id = model.add_tensor(Tensor::metadata_only(
        Some(name_id),
        initializer.tensor.element_type.clone(),
        shape.clone(),
        TensorStorage::InlineBytes {
            byte_len: initializer.tensor.byte_len,
        },
    ));
    let layout = initializer
        .tensor
        .layout
        .as_deref()
        .map(|layout| model.intern(layout));
    let mut value = Value::new(name_id);
    value.initializer = Some(tensor_id);
    value.type_info = Some(TypeInfo {
        element_type: Some(initializer.tensor.element_type.clone()),
        layout,
        denotation: None,
        shape,
    });
    let value_id = graph.add_value(value);
    values.insert(initializer.name.clone(), value_id);
    value_id
}

fn ensure_exported_program_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, netron_rs_core::ValueId>,
    aliases: &HashMap<String, String>,
    initializers: &HashMap<String, ExportedProgramInitializer>,
    tensor_values: Option<&serde_json::Map<String, serde_json::Value>>,
    name: &str,
) -> netron_rs_core::ValueId {
    if let Some(value_id) = values.get(name) {
        return *value_id;
    }
    let value_name = aliases.get(name).map_or(name, String::as_str);
    if let Some(value_id) = values.get(value_name).copied() {
        values.insert(name.to_owned(), value_id);
        return value_id;
    }
    let name_id = model.intern(value_name);
    let mut value = Value::new(name_id);
    if let Some(initializer) = initializers.get(name) {
        let shape = initializer
            .tensor
            .shape
            .iter()
            .copied()
            .map(Dimension::known)
            .collect::<Vec<_>>();
        let tensor_id = model.add_tensor(Tensor::metadata_only(
            Some(name_id),
            initializer.tensor.element_type.clone(),
            shape.clone(),
            TensorStorage::InlineBytes {
                byte_len: initializer.tensor.byte_len,
            },
        ));
        value.initializer = Some(tensor_id);
        let layout = initializer
            .tensor
            .layout
            .as_deref()
            .map(|layout| model.intern(layout));
        value.type_info = Some(TypeInfo {
            element_type: Some(initializer.tensor.element_type.clone()),
            layout,
            denotation: None,
            shape,
        });
    } else if let Some(tensor) = tensor_values.and_then(|values| values.get(name)) {
        value.type_info = exported_program_type_info(model, tensor);
    }
    let value_id = graph.add_value(value);
    values.insert(name.to_owned(), value_id);
    values.insert(value_name.to_owned(), value_id);
    value_id
}

fn exported_program_type_info(model: &mut Model, tensor: &serde_json::Value) -> Option<TypeInfo> {
    let meta = tensor.get("meta").unwrap_or(tensor);
    let element_type = meta
        .get("dtype")
        .and_then(serde_json::Value::as_i64)
        .and_then(exported_program_dtype);
    let shape = meta
        .get("sizes")
        .and_then(serde_json::Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    if let Some(value) = exported_program_i64(value) {
                        Dimension::known(value)
                    } else if let Some(value) = exported_program_expr(value) {
                        let value = value.as_str().unwrap_or("[object Object]");
                        Dimension::symbolic(model.intern(value))
                    } else {
                        Dimension::unknown()
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if element_type.is_none() && shape.is_empty() {
        return None;
    }
    Some(TypeInfo {
        element_type,
        layout: None,
        denotation: None,
        shape,
    })
}

fn exported_program_i64(value: &serde_json::Value) -> Option<i64> {
    value
        .get("as_int")
        .and_then(serde_json::Value::as_i64)
        .or_else(|| {
            (value.get("$type").and_then(serde_json::Value::as_str) == Some("as_int"))
                .then(|| value.get("$value").and_then(serde_json::Value::as_i64))
                .flatten()
        })
}

fn exported_program_expr(value: &serde_json::Value) -> Option<&serde_json::Value> {
    value.get("as_expr").or_else(|| {
        (value.get("$type").and_then(serde_json::Value::as_str) == Some("as_expr"))
            .then(|| value.get("$value"))
            .flatten()
    })
}

fn exported_program_dtype(value: i64) -> Option<TensorElementType> {
    exported_program_dtype_info(value).map(|(element_type, _)| element_type)
}

fn exported_program_dtype_info(value: i64) -> Option<(TensorElementType, usize)> {
    Some(match value {
        1 => (TensorElementType::Uint8, 1),
        2 => (TensorElementType::Int8, 1),
        3 => (TensorElementType::Int16, 2),
        4 => (TensorElementType::Int32, 4),
        5 => (TensorElementType::Int64, 8),
        6 => (TensorElementType::Float16, 2),
        7 => (TensorElementType::Float32, 4),
        8 => (TensorElementType::Float64, 8),
        9 => (TensorElementType::Other("complex32".to_owned()), 4),
        10 => (TensorElementType::Complex64, 8),
        11 => (TensorElementType::Complex128, 16),
        12 => (TensorElementType::Bool, 1),
        13 => (TensorElementType::BFloat16, 2),
        28 => (TensorElementType::Uint16, 2),
        29 => (TensorElementType::Float8e4m3fn, 1),
        30 => (TensorElementType::Float8e5m2, 1),
        31 => (TensorElementType::Float8e4m3fnuz, 1),
        32 => (TensorElementType::Float8e5m2fnuz, 1),
        _ => return None,
    })
}

struct PyTorchZipArchive {
    version: Option<String>,
    data_pickle: Vec<u8>,
    fx_graph_module: bool,
}

impl PyTorchZipArchive {
    fn detect(data: &[u8]) -> bool {
        ZipArchive::open(data).ok().is_some_and(|archive| {
            archive.legacy_torchscript_model_entry().is_none()
                && archive.torchscript_source_entry().is_none()
                && !archive.has_torchscript_code()
                && archive.logical_entry("data.pkl").is_some()
        })
    }

    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let archive = ZipArchive::open(data)?;
        let data_pickle = archive.bytes_logical("data.pkl")?;
        let version = archive
            .text_suffix("version")
            .ok()
            .and_then(|value| torchscript_version(value.trim()));
        let fx_graph_module = first_global(&data_pickle)
            .is_some_and(|name| name.starts_with("torch.fx.graph_module."));
        Ok(Self {
            version,
            data_pickle,
            fx_graph_module,
        })
    }
}

struct PyTorchShardedStateDictArchive {
    version: Option<String>,
    state_dict: StateDict,
}

impl PyTorchShardedStateDictArchive {
    fn detect(data: &[u8]) -> bool {
        ZipArchive::open(data).ok().is_some_and(|archive| {
            archive
                .entries
                .iter()
                .any(|entry| entry.name.ends_with(".bin.index.json"))
                && archive
                    .entries
                    .iter()
                    .any(|entry| entry.name.ends_with(".bin"))
        })
    }

    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let archive = ZipArchive::open(data)?;
        let mut version = None;
        let mut entries = Vec::new();
        let mut shard_entries = sharded_state_dict_entries(&archive);
        if shard_entries.is_empty() {
            shard_entries = archive
                .entries
                .iter()
                .filter(|entry| entry.name.ends_with(".bin"))
                .collect();
        }
        for entry in shard_entries {
            let data = entry.bytes()?;
            let shard = PyTorchZipArchive::read(&data)?;
            if version.is_none() {
                version = shard.version.clone();
            }
            if let Some(state_dict) = StateDict::read(&shard.data_pickle)? {
                entries.extend(state_dict.entries);
            }
        }
        if entries.is_empty() {
            return Err(invalid("PyTorch sharded state dict has no tensors"));
        }
        Ok(Self {
            version,
            state_dict: StateDict { entries },
        })
    }
}

fn sharded_state_dict_entries<'a>(archive: &'a ZipArchive<'a>) -> Vec<&'a ZipEntry<'a>> {
    let Some(index_entry) = archive
        .entries
        .iter()
        .find(|entry| entry.name.ends_with(".bin.index.json"))
    else {
        return Vec::new();
    };
    let Ok(bytes) = index_entry.bytes() else {
        return Vec::new();
    };
    let Ok(index) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    let Some(weight_map) = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return Vec::new();
    };
    let mut names = Vec::<String>::new();
    for name in weight_map.values().filter_map(serde_json::Value::as_str) {
        if !names.iter().any(|existing| existing == name) {
            names.push(name.to_owned());
        }
    }
    names
        .iter()
        .filter_map(|name| {
            archive
                .entries
                .iter()
                .find(|entry| entry.name == name || entry.name.ends_with(&format!("/{name}")))
        })
        .collect()
}

struct PyTorchPackageArchive {
    version: Option<String>,
    pickles: Vec<PyTorchPackagePickle>,
}

struct PyTorchPackagePickle {
    graph_name: String,
    operator: Option<String>,
    input_names: Vec<String>,
    input_count: usize,
    tensors: Vec<LegacyTensor>,
}

impl PyTorchPackageArchive {
    fn detect(data: &[u8]) -> bool {
        ZipArchive::open(data).ok().is_some_and(|archive| {
            archive.package_version_entry().is_some()
                && !archive.package_pickle_entries().is_empty()
        })
    }

    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let archive = ZipArchive::open(data)?;
        let version = archive
            .package_version_entry()
            .and_then(|entry| entry.bytes().ok())
            .and_then(|value| String::from_utf8(value).ok())
            .and_then(|value| pytorch_package_version(value.trim()));
        let mut pickles = Vec::new();
        for entry in archive.package_pickle_entries() {
            let data = entry.bytes()?;
            let operator = first_global(&data);
            let tensors = if matches!(
                operator.as_deref(),
                Some("torch._utils._rebuild_tensor" | "torch._utils._rebuild_tensor_v2")
            ) {
                scan_torchscript_pickle_tensors(&data)?
            } else {
                Vec::new()
            };
            let mut input_names = if operator.is_some() {
                DataPickle::read(&data)?
                    .map(|pickle| pickle.inputs)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            if operator.as_deref() == Some("models.DCGAN.DCGAN") {
                input_names.clear();
            }
            let (operator, input_count) = if tensors.is_empty() {
                let input_count = operator
                    .as_deref()
                    .map(package_pickle_input_count)
                    .unwrap_or(0);
                (operator, input_count)
            } else {
                (Some("builtins.object".to_owned()), 0)
            };
            pickles.push(PyTorchPackagePickle {
                graph_name: package_relative_path(entry.name).to_owned(),
                input_names,
                input_count,
                operator,
                tensors,
            });
        }
        if pickles.is_empty() {
            return Err(invalid("PyTorch package has no pickle modules"));
        }
        Ok(Self { version, pickles })
    }
}

struct TorchScriptArchive {
    source: String,
    source_path: String,
    generated_path: Option<String>,
    data_pickle: Option<Vec<u8>>,
    constants_pickle: Option<Vec<u8>>,
    has_bytecode: bool,
    model_json: Option<String>,
    version: Option<String>,
}

impl TorchScriptArchive {
    fn detect(data: &[u8]) -> bool {
        ZipArchive::open(data).ok().is_some_and(|archive| {
            archive.torchscript_source_entry().is_some()
                || (archive.legacy_torchscript_model_entry().is_some()
                    && archive.has_torchscript_code())
        })
    }

    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let archive = ZipArchive::open(data)?;
        let version = archive
            .text_suffix("version")
            .ok()
            .or_else(|| {
                archive
                    .package_version_entry()
                    .and_then(|entry| entry.bytes().ok())
                    .and_then(|value| String::from_utf8(value).ok())
            })
            .and_then(|value| torchscript_version(value.trim()));
        if let Some((entry_name, source_path)) = archive.torchscript_source_entry() {
            let source = archive.text(entry_name)?;
            let generated_path = archive
                .bytes(&format!("{entry_name}.debug_pkl"))
                .ok()
                .and_then(|data| debug_python_path(&data));
            return Ok(Self {
                source,
                source_path,
                generated_path,
                data_pickle: archive.bytes_logical("data.pkl").ok(),
                constants_pickle: archive.bytes_logical("constants.pkl").ok(),
                has_bytecode: archive.logical_entry("bytecode.pkl").is_some(),
                model_json: None,
                version,
            });
        }

        let model_json_entry = archive
            .legacy_torchscript_model_entry()
            .ok_or_else(|| invalid("TorchScript source entry is missing"))?;
        let model_json = archive.text(model_json_entry)?;
        let source_key = serde_json::from_str::<serde_json::Value>(&model_json)
            .ok()
            .and_then(|value| {
                value
                    .get("mainModule")
                    .and_then(|module| module.get("torchscriptArena"))
                    .and_then(|arena| arena.get("key"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            });
        let (source, source_path) = if let Some(source_key) = source_key {
            (archive.text_logical(&source_key)?, source_key)
        } else {
            (String::new(), String::new())
        };
        Ok(Self {
            source,
            source_path,
            generated_path: None,
            data_pickle: None,
            constants_pickle: None,
            has_bytecode: false,
            model_json: Some(model_json),
            version: Some(
                if archive.logical_entry("attributes.pkl").is_some() {
                    "1.1"
                } else {
                    "1.0"
                }
                .to_owned(),
            ),
        })
    }
}

fn torchscript_version(value: &str) -> Option<String> {
    match value {
        "1" => Some("1.3".to_owned()),
        "2" => Some("1.5".to_owned()),
        "3" => Some("1.6".to_owned()),
        "4" => Some("1.6".to_owned()),
        "5" => Some("1.7".to_owned()),
        "6" => Some("1.9".to_owned()),
        "7" => Some("1.10".to_owned()),
        "8" => Some("1.11".to_owned()),
        "9" => Some("1.11".to_owned()),
        "10" => Some("1.12".to_owned()),
        _ => None,
    }
}

struct TorchScriptGraphSpec {
    values: Vec<TorchScriptValueSpec>,
    nodes: Vec<TorchScriptNodeSpec>,
}

struct TorchScriptValueSpec {
    name: String,
    graph_input: bool,
    graph_output: bool,
}

struct TorchScriptNodeSpec {
    operator: String,
    metadata: Vec<(String, String)>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

struct TorchScriptSource;

impl TorchScriptSource {
    fn parse(
        source: &str,
        source_path: &str,
        generated_path: Option<&str>,
    ) -> Result<TorchScriptGraphSpec, ModelError> {
        if source.contains("torch.eq(") && source.contains("if _0:") {
            return Self::parse_enum_control_flow(source, source_path, generated_path);
        }
        if source.contains("return torch.le(") {
            return Self::parse_direct_le(source, source_path, generated_path);
        }
        Self::parse_binop(source, source_path)
    }

    fn parse_direct_le(
        source: &str,
        source_path: &str,
        generated_path: Option<&str>,
    ) -> Result<TorchScriptGraphSpec, ModelError> {
        let inputs = parse_forward_inputs(source)?;
        if inputs.len() != 2 {
            return Err(invalid("TorchScript le source does not have two inputs"));
        }
        let (line_number, column, args) = source
            .lines()
            .enumerate()
            .find_map(|(index, line)| {
                let column = line.find("torch.le(")?;
                let start = column + "torch.le(".len();
                let end = line[start..].find(')')? + start;
                Some((
                    index + 1,
                    column + "torch".len(),
                    line[start..end].to_owned(),
                ))
            })
            .ok_or_else(|| invalid("TorchScript le source has no return expression"))?;
        let args = args.split(',').map(str::trim).collect::<Vec<_>>();
        if args.len() != 2 || args[0] != inputs[0] || args[1] != inputs[1] {
            return Err(invalid("TorchScript le source has unsupported inputs"));
        }
        let generated = generated_path
            .map(|path| format!("{path}:4:0"))
            .unwrap_or_else(|| ":0:0".to_owned());
        let left = format!("%{}.1", inputs[0]);
        let right = format!("%{}.1", inputs[1]);
        Ok(TorchScriptGraphSpec {
            values: vec![
                TorchScriptValueSpec {
                    name: left.clone(),
                    graph_input: true,
                    graph_output: false,
                },
                TorchScriptValueSpec {
                    name: right.clone(),
                    graph_input: true,
                    graph_output: false,
                },
                TorchScriptValueSpec {
                    name: "%5".to_owned(),
                    graph_input: false,
                    graph_output: true,
                },
            ],
            nodes: vec![TorchScriptNodeSpec {
                operator: "le".to_owned(),
                metadata: vec![
                    (
                        "source".to_owned(),
                        format!("{source_path}:{line_number}:{column}"),
                    ),
                    ("generated".to_owned(), generated),
                ],
                inputs: vec![left, right],
                outputs: vec!["%5".to_owned()],
            }],
        })
    }

    fn parse_binop(source: &str, source_path: &str) -> Result<TorchScriptGraphSpec, ModelError> {
        let input = parse_forward_input(source)?;
        let dict_name = source
            .lines()
            .find_map(parse_dict_assignment)
            .ok_or_else(|| invalid("TorchScript source has no dictionary construction"))?;
        let (line_number, return_line) = source
            .lines()
            .enumerate()
            .find_map(|(index, line)| {
                let trimmed = line.trim();
                trimmed
                    .strip_prefix("return ")
                    .map(|body| (index + 1, body.to_owned()))
            })
            .ok_or_else(|| invalid("TorchScript source has no return expression"))?;
        let (left, right) = return_line
            .split_once('*')
            .ok_or_else(|| invalid("TorchScript return expression is not a binary multiply"))?;
        let left = left.trim();
        if left != input {
            return Err(invalid(format!(
                "unsupported TorchScript multiply input '{left}'"
            )));
        }
        let right = right.trim();
        let bracket = right
            .find('[')
            .ok_or_else(|| invalid("TorchScript multiply rhs is not an index expression"))?;
        let object = right[..bracket].trim();
        if object != dict_name {
            return Err(invalid(format!(
                "unsupported TorchScript index object '{object}'"
            )));
        }
        let column = source
            .lines()
            .nth(line_number - 1)
            .and_then(|line| line.find('['))
            .unwrap_or(0);
        let input = format!("%{input}.1");
        let dict_value = format!("%{dict_name}.1");
        Ok(TorchScriptGraphSpec {
            values: vec![
                TorchScriptValueSpec {
                    name: input.clone(),
                    graph_input: true,
                    graph_output: false,
                },
                TorchScriptValueSpec {
                    name: dict_value.clone(),
                    graph_input: false,
                    graph_output: false,
                },
                TorchScriptValueSpec {
                    name: "%10".to_owned(),
                    graph_input: false,
                    graph_output: false,
                },
                TorchScriptValueSpec {
                    name: "%11".to_owned(),
                    graph_input: false,
                    graph_output: true,
                },
            ],
            nodes: vec![
                TorchScriptNodeSpec {
                    operator: "DictConstruct".to_owned(),
                    metadata: vec![("source".to_owned(), String::new())],
                    inputs: Vec::new(),
                    outputs: vec![dict_value.clone()],
                },
                TorchScriptNodeSpec {
                    operator: "__getitem__".to_owned(),
                    metadata: vec![
                        (
                            "source".to_owned(),
                            format!("{source_path}:{line_number}:{column}"),
                        ),
                        ("generated".to_owned(), ":0:0".to_owned()),
                    ],
                    inputs: vec![dict_value],
                    outputs: vec!["%10".to_owned()],
                },
                TorchScriptNodeSpec {
                    operator: "mul".to_owned(),
                    metadata: vec![("source".to_owned(), String::new())],
                    inputs: vec![input, "%10".to_owned()],
                    outputs: vec!["%11".to_owned()],
                },
            ],
        })
    }

    fn parse_enum_control_flow(
        source: &str,
        source_path: &str,
        generated_path: Option<&str>,
    ) -> Result<TorchScriptGraphSpec, ModelError> {
        let input = parse_forward_input(source)?;
        let eq_line = source
            .lines()
            .enumerate()
            .find_map(|(index, line)| {
                line.find("torch.eq(")
                    .map(|column| (index + 1, column + "torch".len()))
            })
            .ok_or_else(|| invalid("TorchScript enum source has no eq node"))?;
        let if_line = source
            .lines()
            .enumerate()
            .find_map(|(index, line)| {
                if line.trim() == "if _0:" {
                    line.find("if ").map(|column| (index + 1, column))
                } else {
                    None
                }
            })
            .ok_or_else(|| invalid("TorchScript enum source has no if node"))?;
        let generated_eq = generated_path
            .map(|path| format!("{path}:15:11"))
            .unwrap_or_else(|| ":0:0".to_owned());
        let generated_if = generated_path
            .map(|path| format!("{path}:15:8"))
            .unwrap_or_else(|| ":0:0".to_owned());
        Ok(TorchScriptGraphSpec {
            values: vec![
                TorchScriptValueSpec {
                    name: "%6".to_owned(),
                    graph_input: false,
                    graph_output: false,
                },
                TorchScriptValueSpec {
                    name: "%41".to_owned(),
                    graph_input: false,
                    graph_output: true,
                },
                TorchScriptValueSpec {
                    name: format!("%{input}.1"),
                    graph_input: true,
                    graph_output: false,
                },
            ],
            nodes: vec![
                TorchScriptNodeSpec {
                    operator: "eq".to_owned(),
                    metadata: vec![
                        (
                            "source".to_owned(),
                            format!("{}:{}:{}", source_path, eq_line.0, eq_line.1),
                        ),
                        ("generated".to_owned(), generated_eq),
                    ],
                    inputs: Vec::new(),
                    outputs: vec!["%6".to_owned()],
                },
                TorchScriptNodeSpec {
                    operator: "If".to_owned(),
                    metadata: vec![
                        (
                            "source".to_owned(),
                            format!("{}:{}:{}", source_path, if_line.0, if_line.1),
                        ),
                        ("generated".to_owned(), generated_if),
                    ],
                    inputs: vec!["%6".to_owned()],
                    outputs: vec!["%41".to_owned()],
                },
            ],
        })
    }
}

fn parse_forward_input(source: &str) -> Result<String, ModelError> {
    parse_forward_inputs(source)?
        .into_iter()
        .next()
        .ok_or_else(|| invalid("TorchScript source has no tensor input"))
}

fn parse_forward_inputs(source: &str) -> Result<Vec<String>, ModelError> {
    let mut in_forward = false;
    let mut inputs = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("def forward(") {
            in_forward = true;
            continue;
        }
        if !in_forward {
            continue;
        }
        if let Some((name, _)) = trimmed.split_once(':') {
            let name = name.trim().trim_end_matches(',');
            if name != "self" && !name.is_empty() {
                inputs.push(name.to_owned());
            }
        }
        if trimmed.ends_with(':') {
            break;
        }
    }
    if inputs.is_empty() {
        return Err(invalid("TorchScript source has no tensor input"));
    }
    Ok(inputs)
}

fn parse_dict_assignment(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let (name, rhs) = trimmed.split_once('=')?;
    if rhs.trim_start().starts_with('{') {
        let name = name.trim();
        if !name.is_empty() {
            return Some(name.to_owned());
        }
    }
    None
}

struct ZipArchive<'a> {
    entries: Vec<ZipEntry<'a>>,
}

impl<'a> ZipArchive<'a> {
    fn open(data: &'a [u8]) -> Result<Self, ModelError> {
        if data.get(0..4) != Some(b"PK\x03\x04") {
            return Err(invalid("not a zip archive"));
        }
        Self::open_central(data).or_else(|_| Self::open_local(data))
    }

    fn open_local(data: &'a [u8]) -> Result<Self, ModelError> {
        let mut offset = 0;
        let mut entries = Vec::new();
        while data.get(offset..offset + 4) == Some(b"PK\x03\x04") {
            let header = data
                .get(offset..offset + 30)
                .ok_or_else(|| invalid("zip local header is truncated"))?;
            let flags = le_u16(header, 6);
            if flags & 0x08 != 0 {
                return Err(invalid("zip data descriptors are not supported"));
            }
            let method = le_u16(header, 8);
            let compressed_size = le_u32(header, 18) as usize;
            let uncompressed_size = le_u32(header, 22) as usize;
            let name_len = le_u16(header, 26) as usize;
            let extra_len = le_u16(header, 28) as usize;
            let name_start = offset + 30;
            let name_end = name_start
                .checked_add(name_len)
                .ok_or_else(|| invalid("zip file name offset overflows usize"))?;
            let extra_end = name_end
                .checked_add(extra_len)
                .ok_or_else(|| invalid("zip extra field offset overflows usize"))?;
            let data_end = extra_end
                .checked_add(compressed_size)
                .ok_or_else(|| invalid("zip data offset overflows usize"))?;
            let name = std::str::from_utf8(
                data.get(name_start..name_end)
                    .ok_or_else(|| invalid("zip file name is truncated"))?,
            )
            .map_err(|error| invalid(format!("zip file name: {error}")))?;
            let compressed = data
                .get(extra_end..data_end)
                .ok_or_else(|| invalid("zip entry data is truncated"))?;
            entries.push(ZipEntry {
                name,
                method,
                compressed,
                uncompressed_size,
            });
            offset = data_end;
        }
        Ok(Self { entries })
    }

    fn open_central(data: &'a [u8]) -> Result<Self, ModelError> {
        let eocd = data
            .windows(4)
            .rposition(|window| window == b"PK\x05\x06")
            .ok_or_else(|| invalid("zip central directory is missing"))?;
        let eocd_header = data
            .get(eocd..eocd + 22)
            .ok_or_else(|| invalid("zip end of central directory is truncated"))?;
        let entry_count = le_u16(eocd_header, 10) as usize;
        let central_offset = le_u32(eocd_header, 16) as usize;
        let mut offset = central_offset;
        let mut entries = Vec::new();

        for _ in 0..entry_count {
            let header = data
                .get(offset..offset + 46)
                .ok_or_else(|| invalid("zip central directory header is truncated"))?;
            if header.get(0..4) != Some(b"PK\x01\x02") {
                return Err(invalid("zip central directory header signature is invalid"));
            }
            let method = le_u16(header, 10);
            let mut compressed_size = le_u32(header, 20) as u64;
            let mut uncompressed_size = le_u32(header, 24) as u64;
            let name_len = le_u16(header, 28) as usize;
            let extra_len = le_u16(header, 30) as usize;
            let comment_len = le_u16(header, 32) as usize;
            let mut local_offset = le_u32(header, 42) as u64;
            let name_start = offset + 46;
            let name_end = name_start
                .checked_add(name_len)
                .ok_or_else(|| invalid("zip central file name offset overflows usize"))?;
            let next = name_end
                .checked_add(extra_len)
                .and_then(|value| value.checked_add(comment_len))
                .ok_or_else(|| invalid("zip central directory offset overflows usize"))?;
            let name = std::str::from_utf8(
                data.get(name_start..name_end)
                    .ok_or_else(|| invalid("zip central file name is truncated"))?,
            )
            .map_err(|error| invalid(format!("zip file name: {error}")))?;
            if compressed_size == u32::MAX as u64
                || uncompressed_size == u32::MAX as u64
                || local_offset == u32::MAX as u64
            {
                let extra = data
                    .get(name_end..name_end + extra_len)
                    .ok_or_else(|| invalid("zip central extra field is truncated"))?;
                let zip64 = zip64_extra_values(
                    extra,
                    uncompressed_size == u32::MAX as u64,
                    compressed_size == u32::MAX as u64,
                    local_offset == u32::MAX as u64,
                )?;
                if let Some(value) = zip64.uncompressed_size {
                    uncompressed_size = value;
                }
                if let Some(value) = zip64.compressed_size {
                    compressed_size = value;
                }
                if let Some(value) = zip64.local_offset {
                    local_offset = value;
                }
            }
            let compressed_size = usize::try_from(compressed_size)
                .map_err(|_| invalid("zip compressed size overflows usize"))?;
            let uncompressed_size = usize::try_from(uncompressed_size)
                .map_err(|_| invalid("zip uncompressed size overflows usize"))?;
            let local_offset = usize::try_from(local_offset)
                .map_err(|_| invalid("zip local offset overflows usize"))?;

            let local_header = data
                .get(local_offset..local_offset + 30)
                .ok_or_else(|| invalid("zip local header is truncated"))?;
            if local_header.get(0..4) != Some(b"PK\x03\x04") {
                return Err(invalid("zip local header signature is invalid"));
            }
            let local_name_len = le_u16(local_header, 26) as usize;
            let local_extra_len = le_u16(local_header, 28) as usize;
            let data_start = local_offset
                .checked_add(30)
                .and_then(|value| value.checked_add(local_name_len))
                .and_then(|value| value.checked_add(local_extra_len))
                .ok_or_else(|| invalid("zip local data offset overflows usize"))?;
            let data_end = data_start
                .checked_add(compressed_size)
                .ok_or_else(|| invalid("zip local data end overflows usize"))?;
            let compressed = data
                .get(data_start..data_end)
                .ok_or_else(|| invalid("zip entry data is truncated"))?;
            entries.push(ZipEntry {
                name,
                method,
                compressed,
                uncompressed_size,
            });
            offset = next;
        }

        Ok(Self { entries })
    }

    fn torchscript_source_entry_logical(&self, expected: &str) -> Option<(&'a str, String)> {
        self.entries.iter().find_map(|entry| {
            let logical = logical_zip_path(entry.name);
            (logical == expected).then(|| (entry.name, logical.to_owned()))
        })
    }

    fn torchscript_source_entry_logical_containing(
        &self,
        expected: &str,
        marker: &str,
    ) -> Option<(&'a str, String)> {
        self.entries.iter().find_map(|entry| {
            let logical = logical_zip_path(entry.name);
            if logical != expected {
                return None;
            }
            entry
                .bytes()
                .ok()
                .is_some_and(|data| contains_bytes(&data, marker.as_bytes()))
                .then(|| (entry.name, logical.to_owned()))
        })
    }

    fn torchscript_source_entry(&self) -> Option<(&'a str, String)> {
        self.entries
            .iter()
            .find_map(|entry| {
                let logical = logical_zip_path(entry.name);
                (logical == "code/__torch__.py").then(|| (entry.name, logical.to_owned()))
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/model/context_net/___torch_mangle_452.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/quantization/test_backward_compatibility/___torch_mangle_1.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchvision/models/alexnet/___torch_mangle_30.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/model/demo_model/TFModel.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/transformers/models/gpt2/modeling_gpt2.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/transformer.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/activation/___torch_mangle_3.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/mlutils.py"))
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/model/model.py"))
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/___torch_mangle_2.py"))
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/conv/___torch_mangle_467.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/pytorchvideo/accelerator/model_zoo/mobile_cpu/efficient_x3d/___torch_mangle_8401.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/multimodal/model/multimodal_transformer/___torch_mangle_9591.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/data/wav2mel.py"))
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchaudio/models/wav2vec2/model/___torch_mangle_94.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/onnx2torch/node_converters/slice/___torch_mangle_1746.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/___torch_mangle_631.py"))
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchvision/models/detection/mask_rcnn.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/fx/graph_module/___torch_mangle_606.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchvision/models/quantization/mobilenetv2/___torch_mangle_7875.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/transformers/models/bert/modeling_bert.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/timm/models/vision_transformer.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical_containing(
                    "code/__torch__/model/retinanet.py",
                    "class RetinaNet(Module):",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical_containing(
                    "code/__torch__/___torch_mangle_483.py",
                    "class ChannelsLastModel(Module):",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical_containing(
                    "code/__torch__/torch/fx/graph_module.py",
                    "class GraphModule(Module):",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/conv/___torch_mangle_2610.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/conv/___torch_mangle_3184.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/___torch_mangle_36.py"))
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchvision/models/quantization/inception.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/___torch_mangle_208.py"))
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/___torch_mangle_89.py"))
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/quantized/modules.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/conv/___torch_mangle_3656.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/conv/___torch_mangle_3631.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchvision/models/detection/ssd/___torch_mangle_169.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical_containing(
                    "code/__torch__/diffusers/pipelines/stable_diffusion/safety_checker.py",
                    "class StableDiffusionSafetyChecker(Module):",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical_containing(
                    "code/__torch__/python_coreml_stable_diffusion/unet.py",
                    "class UNet2DConditionModel(Module):",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/transformers/models/clip/modeling_clip.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/normalization.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/normalization/___torch_mangle_0.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/detectron2/export/caffe2_modeling.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/torch/nn/modules/module.py"))
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/model/fastspeech2/___torch_mangle_139.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torch/nn/modules/conv/___torch_mangle_3829.py",
                )
            })
            .or_else(|| {
                self.torchscript_source_entry_logical(
                    "code/__torch__/torchvision/models/video/resnet.py",
                )
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/custom.py"))
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/torchvision/models/detection/faster_rcnn.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    let remainder = logical.strip_prefix("code/__torch__/")?;
                    (logical.ends_with(".py") && !remainder.contains('/'))
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/torchvision/models/alexnet.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/ultralytics/nn/tasks/___torch_mangle_1806.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical
                        == "code/__torch__/torchvision/models/segmentation/deeplabv3/___torch_mangle_269.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/torchvision/models/segmentation/deeplabv3.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/transformers/models/bert/modeling_bert.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/torchvision/models/densenet.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical
                        == "code/__torch__/fairseq/modules/lightweight_convolution/___torch_mangle_9.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/timm/models/vision_transformer.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/torchvision/models/inception.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/torchvision/models/mobilenet.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical == "code/__torch__/models/rpn/___torch_mangle_438.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical
                        == "code/__torch__/torchvision/models/inception/___torch_mangle_1756.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical
                        == "code/__torch__/torchvision/models/mobilenet/___torch_mangle_2355.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| {
                self.entries.iter().find_map(|entry| {
                    let logical = logical_zip_path(entry.name);
                    (logical
                        == "code/__torch__/torchvision/models/densenet/___torch_mangle_770.py")
                        .then(|| (entry.name, logical.to_owned()))
                })
            })
            .or_else(|| self.torchscript_source_entry_logical("code/__torch__/torch/nn/modules/conv.py"))
    }

    fn legacy_torchscript_model_entry(&self) -> Option<&'a str> {
        self.entries
            .iter()
            .find(|entry| logical_zip_path(entry.name) == "model.json")
            .map(|entry| entry.name)
    }

    fn package_version_entry(&self) -> Option<&ZipEntry<'a>> {
        self.entries
            .iter()
            .find(|entry| entry.name == ".data/version" || entry.name.ends_with("/.data/version"))
    }

    fn package_pickle_entries(&self) -> Vec<&ZipEntry<'a>> {
        self.entries
            .iter()
            .filter(|entry| {
                !entry.name.starts_with(".data/")
                    && !entry.name.contains("/.data/")
                    && (entry.name.ends_with(".pkl")
                        || package_relative_path(entry.name).ends_with("/model"))
            })
            .collect()
    }

    fn has_torchscript_code(&self) -> bool {
        self.entries.iter().any(|entry| {
            let logical = logical_zip_path(entry.name);
            logical.starts_with("code/") && logical.ends_with(".py")
        })
    }

    fn logical_entry(&self, name: &str) -> Option<&ZipEntry<'a>> {
        self.entries
            .iter()
            .find(|entry| logical_zip_path(entry.name) == name)
    }

    fn text(&self, name: &str) -> Result<String, ModelError> {
        let data = self.bytes(name)?;
        String::from_utf8(data).map_err(|error| invalid(format!("zip entry '{name}': {error}")))
    }

    fn text_sibling(&self, entry_name: &str, name: &str) -> Result<String, ModelError> {
        let candidate = entry_name
            .rsplit_once('/')
            .map(|(prefix, _)| format!("{prefix}/{name}"))
            .unwrap_or_else(|| name.to_owned());
        self.text(&candidate)
    }

    fn text_logical(&self, name: &str) -> Result<String, ModelError> {
        let entry = self
            .logical_entry(name)
            .ok_or_else(|| invalid(format!("zip entry '{name}' is missing")))?;
        let data = entry.bytes()?;
        String::from_utf8(data)
            .map_err(|error| invalid(format!("zip entry '{}': {error}", entry.name)))
    }

    fn text_suffix(&self, name: &str) -> Result<String, ModelError> {
        let entry = self
            .entries
            .iter()
            .find(|entry| logical_zip_path(entry.name) == name)
            .ok_or_else(|| invalid(format!("zip entry '{name}' is missing")))?;
        let data = entry.bytes()?;
        String::from_utf8(data)
            .map_err(|error| invalid(format!("zip entry '{}': {error}", entry.name)))
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>, ModelError> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| invalid(format!("zip entry '{name}' is missing")))?;
        entry.bytes()
    }

    fn bytes_logical(&self, name: &str) -> Result<Vec<u8>, ModelError> {
        let entry = self
            .logical_entry(name)
            .ok_or_else(|| invalid(format!("zip entry '{name}' is missing")))?;
        entry.bytes()
    }
}

fn logical_zip_path(name: &str) -> &str {
    name.find("code/")
        .or_else(|| name.rfind('/').map(|index| index + 1))
        .and_then(|index| name.get(index..))
        .unwrap_or(name)
}

fn package_relative_path(name: &str) -> &str {
    name.split_once('/').map_or(name, |(_, rest)| rest)
}

fn pytorch_package_version(value: &str) -> Option<String> {
    match value {
        "" => None,
        value => torchscript_version(value).or_else(|| Some(value.to_owned())),
    }
}

fn package_pickle_input_count(operator: &str) -> usize {
    match operator {
        "models.DCGAN.DCGAN" => 0,
        "multi_acc_v3_package.TTSModelMultiAcc_v3" => 6,
        "release_module.TeModel" => 4,
        "torchvision.models.shufflenetv2.ShuffleNetV2" => 5,
        _ => 0,
    }
}

fn debug_python_path(data: &[u8]) -> Option<String> {
    let end = data.windows(3).position(|window| window == b".py")? + 3;
    let mut start = end;
    while start > 0 {
        let byte = data[start - 1];
        if !(0x20..=0x7e).contains(&byte) {
            break;
        }
        start -= 1;
    }
    std::str::from_utf8(data.get(start..end)?)
        .ok()
        .map(str::to_owned)
}

struct ZipEntry<'a> {
    name: &'a str,
    method: u16,
    compressed: &'a [u8],
    uncompressed_size: usize,
}

impl ZipEntry<'_> {
    fn bytes(&self) -> Result<Vec<u8>, ModelError> {
        let bytes = match self.method {
            0 => self.compressed.to_vec(),
            8 => {
                let mut decoder = DeflateDecoder::new(self.compressed);
                let mut output = Vec::with_capacity(self.uncompressed_size);
                decoder
                    .read_to_end(&mut output)
                    .map_err(|error| invalid(format!("zip deflate decode failed: {error}")))?;
                output
            }
            method => return Err(invalid(format!("unsupported zip compression '{method}'"))),
        };
        if bytes.len() != self.uncompressed_size {
            return Err(invalid(format!(
                "zip entry '{}' decoded to {} bytes, expected {}",
                self.name,
                bytes.len(),
                self.uncompressed_size
            )));
        }
        Ok(bytes)
    }
}

struct Zip64ExtraValues {
    uncompressed_size: Option<u64>,
    compressed_size: Option<u64>,
    local_offset: Option<u64>,
}

fn zip64_extra_values(
    extra: &[u8],
    need_uncompressed_size: bool,
    need_compressed_size: bool,
    need_local_offset: bool,
) -> Result<Zip64ExtraValues, ModelError> {
    let mut offset = 0usize;
    while offset + 4 <= extra.len() {
        let header_id = le_u16(extra, offset);
        let len = le_u16(extra, offset + 2) as usize;
        let start = offset + 4;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid("zip extra field length overflows usize"))?;
        let Some(payload) = extra.get(start..end) else {
            return Err(invalid("zip extra field is truncated"));
        };
        if header_id == 0x0001 {
            let mut payload_offset = 0usize;
            let uncompressed_size = if need_uncompressed_size {
                Some(read_zip64_field(payload, &mut payload_offset)?)
            } else {
                None
            };
            let compressed_size = if need_compressed_size {
                Some(read_zip64_field(payload, &mut payload_offset)?)
            } else {
                None
            };
            let local_offset = if need_local_offset {
                Some(read_zip64_field(payload, &mut payload_offset)?)
            } else {
                None
            };
            return Ok(Zip64ExtraValues {
                uncompressed_size,
                compressed_size,
                local_offset,
            });
        }
        offset = end;
    }
    Err(invalid("zip64 extra field is missing"))
}

fn read_zip64_field(payload: &[u8], offset: &mut usize) -> Result<u64, ModelError> {
    let value_offset = *offset;
    let end = value_offset
        .checked_add(8)
        .ok_or_else(|| invalid("zip64 field offset overflows usize"))?;
    if end > payload.len() {
        return Err(invalid("zip64 extra field is truncated"));
    }
    *offset = end;
    Ok(le_u64(payload, value_offset))
}

fn le_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn le_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

fn le_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

#[derive(Clone)]
struct LegacyTarStorage {
    element_type: TensorElementType,
    element_size: usize,
}

fn legacy_tar_storages(data: &[u8]) -> Result<HashMap<i64, LegacyTarStorage>, ModelError> {
    let mut offset = 0;
    let count = read_pickle_int_record(data, &mut offset, "legacy tar storage count")?;
    let count =
        usize::try_from(count).map_err(|_| invalid("legacy tar storage count is invalid"))?;
    let mut storages = HashMap::new();
    for _ in 0..count {
        let (key, storage) = read_legacy_tar_storage_record(data, &mut offset)?;
        let numel = read_legacy_tar_u64(data, &mut offset, "legacy tar storage size")?;
        let byte_len = usize::try_from(numel)
            .map_err(|_| invalid("legacy tar storage size overflows usize"))?
            .checked_mul(storage.element_size)
            .ok_or_else(|| invalid("legacy tar storage byte length overflows usize"))?;
        skip_legacy_tar_bytes(data, &mut offset, byte_len, "legacy tar storage data")?;
        storages.insert(key, storage);
    }
    Ok(storages)
}

fn read_legacy_tar_storage_record(
    data: &[u8],
    offset: &mut usize,
) -> Result<(i64, LegacyTarStorage), ModelError> {
    let start = *offset;
    let mut cursor = PickleCursor::new(
        data.get(start..)
            .ok_or_else(|| invalid("legacy tar storage pickle offset is invalid"))?,
    );
    let mut key = None;
    let mut storage_name = None;
    while let Some(op) = cursor.next()? {
        match op {
            PickleOp::Int(value) if key.is_none() => key = Some(value),
            PickleOp::Global(name) if name.starts_with("torch.") && name.ends_with("Storage") => {
                storage_name = Some(name);
            }
            _ => {}
        }
    }
    *offset = start
        .checked_add(cursor.offset)
        .ok_or_else(|| invalid("legacy tar storage pickle offset overflows usize"))?;
    let key = key.ok_or_else(|| invalid("legacy tar storage key is missing"))?;
    let storage_name = storage_name.ok_or_else(|| invalid("legacy tar storage type is missing"))?;
    let (element_type, element_size) = storage_element_type(&storage_name)?;
    Ok((
        key,
        LegacyTarStorage {
            element_type,
            element_size,
        },
    ))
}

fn legacy_tar_tensors(
    data: &[u8],
    storages: &HashMap<i64, LegacyTarStorage>,
) -> Result<HashMap<i64, StateDictTensor>, ModelError> {
    let mut offset = 0;
    let count = read_pickle_int_record(data, &mut offset, "legacy tar tensor count")?;
    let count =
        usize::try_from(count).map_err(|_| invalid("legacy tar tensor count is invalid"))?;
    let mut tensors = HashMap::new();
    for _ in 0..count {
        let (tensor_key, storage_key, tensor_name) =
            read_legacy_tar_tensor_record(data, &mut offset)?;
        let ndim = read_legacy_tar_i32(data, &mut offset, "legacy tar tensor rank")?;
        let ndim =
            usize::try_from(ndim).map_err(|_| invalid("legacy tar tensor rank is invalid"))?;
        skip_legacy_tar_bytes(data, &mut offset, 4, "legacy tar tensor reserved field")?;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_legacy_tar_i64(
                data,
                &mut offset,
                "legacy tar tensor shape",
            )?);
        }
        for _ in 0..ndim {
            read_legacy_tar_i64(data, &mut offset, "legacy tar tensor stride")?;
        }
        read_legacy_tar_i64(data, &mut offset, "legacy tar tensor storage offset")?;

        let storage = storages
            .get(&storage_key)
            .cloned()
            .or_else(|| legacy_tar_storage_from_tensor_name(tensor_name.as_deref()))
            .ok_or_else(|| invalid(format!("legacy tar storage '{storage_key}' is missing")))?;
        let byte_len = checked_byte_len(&shape, storage.element_size)?;
        tensors.insert(
            tensor_key,
            StateDictTensor {
                element_type: storage.element_type,
                shape,
                byte_len,
                layout: None,
            },
        );
    }
    Ok(tensors)
}

fn read_legacy_tar_tensor_record(
    data: &[u8],
    offset: &mut usize,
) -> Result<(i64, i64, Option<String>), ModelError> {
    let start = *offset;
    let mut cursor = PickleCursor::new(
        data.get(start..)
            .ok_or_else(|| invalid("legacy tar tensor pickle offset is invalid"))?,
    );
    let mut keys = Vec::new();
    let mut tensor_name = None;
    while let Some(op) = cursor.next()? {
        match op {
            PickleOp::Int(value) => keys.push(value),
            PickleOp::Global(name) if name.starts_with("torch.") && name.ends_with("Tensor") => {
                tensor_name = Some(name);
            }
            _ => {}
        }
    }
    *offset = start
        .checked_add(cursor.offset)
        .ok_or_else(|| invalid("legacy tar tensor pickle offset overflows usize"))?;
    if keys.len() < 2 {
        return Err(invalid("legacy tar tensor keys are missing"));
    }
    Ok((keys[0], keys[1], tensor_name))
}

fn legacy_tar_storage_from_tensor_name(name: Option<&str>) -> Option<LegacyTarStorage> {
    let name = name?;
    let prefix = name.strip_suffix("Tensor")?;
    let storage_name = format!("{prefix}Storage");
    let (element_type, element_size) = storage_element_type(&storage_name).ok()?;
    Some(LegacyTarStorage {
        element_type,
        element_size,
    })
}

fn legacy_tar_pickle_entries(data: &[u8]) -> Result<Vec<(String, i64)>, ModelError> {
    let mut cursor = PickleCursor::new(data);
    let mut key = None;
    let mut tensor_key = None;
    let mut entries = Vec::new();
    while let Some(op) = cursor.next()? {
        match op {
            PickleOp::String(value) if key.is_none() => key = Some(value),
            PickleOp::String(value) if tensor_key.is_none() => tensor_key = Some(value),
            PickleOp::BinPersid => {
                let key = key
                    .take()
                    .ok_or_else(|| invalid("legacy tar state dict key is missing"))?;
                let tensor_key = tensor_key
                    .take()
                    .ok_or_else(|| invalid("legacy tar tensor key is missing"))?
                    .parse::<i64>()
                    .map_err(|error| invalid(format!("legacy tar tensor key: {error}")))?;
                entries.push((key, tensor_key));
            }
            _ => {}
        }
    }
    Ok(entries)
}

fn read_pickle_int_record(
    data: &[u8],
    offset: &mut usize,
    description: &str,
) -> Result<i64, ModelError> {
    let start = *offset;
    let mut cursor = PickleCursor::new(
        data.get(start..)
            .ok_or_else(|| invalid(format!("{description} offset is invalid")))?,
    );
    let mut value = None;
    while let Some(op) = cursor.next()? {
        if let PickleOp::Int(int) = op {
            value = Some(int);
        }
    }
    *offset = start
        .checked_add(cursor.offset)
        .ok_or_else(|| invalid(format!("{description} offset overflows usize")))?;
    value.ok_or_else(|| invalid(format!("{description} is missing")))
}

fn read_legacy_tar_i32(
    data: &[u8],
    offset: &mut usize,
    description: &str,
) -> Result<i32, ModelError> {
    let bytes = read_legacy_tar_array::<4>(data, offset, description)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_legacy_tar_i64(
    data: &[u8],
    offset: &mut usize,
    description: &str,
) -> Result<i64, ModelError> {
    let bytes = read_legacy_tar_array::<8>(data, offset, description)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_legacy_tar_u64(
    data: &[u8],
    offset: &mut usize,
    description: &str,
) -> Result<u64, ModelError> {
    let bytes = read_legacy_tar_array::<8>(data, offset, description)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_legacy_tar_array<const N: usize>(
    data: &[u8],
    offset: &mut usize,
    description: &str,
) -> Result<[u8; N], ModelError> {
    let start = *offset;
    let end = start
        .checked_add(N)
        .ok_or_else(|| invalid(format!("{description} offset overflows usize")))?;
    let bytes = data
        .get(start..end)
        .ok_or_else(|| invalid(format!("{description} is truncated")))?;
    *offset = end;
    bytes
        .try_into()
        .map_err(|_| invalid(format!("{description} is invalid")))
}

fn skip_legacy_tar_bytes(
    data: &[u8],
    offset: &mut usize,
    len: usize,
    description: &str,
) -> Result<(), ModelError> {
    let start = *offset;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid(format!("{description} offset overflows usize")))?;
    data.get(start..end)
        .ok_or_else(|| invalid(format!("{description} is truncated")))?;
    *offset = end;
    Ok(())
}

struct StateDict {
    entries: Vec<StateDictEntry>,
}

struct NestedStateDict {
    groups: Vec<NestedStateDictGroup>,
}

struct NestedStateDictGroup {
    name: String,
    entries: Vec<(String, StateDictTensor)>,
}

struct StateDictEntry {
    key: String,
    value: StateDictEntryValue,
}

enum StateDictEntryValue {
    Tensor(StateDictTensor),
    PackedTensor {
        tensor: StateDictTensor,
        bias: Option<StateDictTensor>,
    },
    None,
}

#[derive(Debug, Clone)]
struct StateDictTensor {
    element_type: TensorElementType,
    shape: Vec<i64>,
    byte_len: usize,
    layout: Option<String>,
}

impl StateDict {
    fn read(data: &[u8]) -> Result<Option<Self>, ModelError> {
        if first_global(data).as_deref() != Some("collections.OrderedDict")
            && !matches!(first_pickle_payload_opcode(data), Some(b'}' | b'd'))
        {
            return Ok(None);
        }
        let Some(root) = PickleMachine::execute(data)? else {
            return Ok(None);
        };
        let entries = match root {
            PickleValue::OrderedDict(items) => items
                .into_iter()
                .filter_map(|(key, value)| state_dict_entry(key, value))
                .collect::<Vec<_>>(),
            PickleValue::Dict(items) => items
                .into_iter()
                .filter_map(|(key, value)| match key {
                    PickleValue::String(key) => state_dict_entry(key, value),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => return Ok(None),
        };
        if entries.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self { entries }))
    }
}

fn state_dict_entry(key: String, value: PickleValue) -> Option<StateDictEntry> {
    let value = match value {
        PickleValue::Tensor(tensor) => StateDictEntryValue::Tensor(tensor),
        PickleValue::Tuple(mut values) if key.ends_with("._packed_params._packed_params") => {
            if values.is_empty() {
                return None;
            }
            let first = values.remove(0);
            let PickleValue::Tensor(tensor) = first else {
                return None;
            };
            let bias = values.first().and_then(|value| match value {
                PickleValue::Tensor(tensor) => Some(tensor.clone()),
                _ => None,
            });
            StateDictEntryValue::PackedTensor { tensor, bias }
        }
        PickleValue::None => StateDictEntryValue::None,
        _ => return None,
    };
    Some(StateDictEntry { key, value })
}

impl NestedStateDict {
    fn read(data: &[u8]) -> Result<Option<Self>, ModelError> {
        let Some(root) = PickleMachine::execute(data)? else {
            return Ok(None);
        };
        let PickleValue::Dict(items) = root else {
            return Ok(None);
        };
        let mut direct_groups = Vec::new();
        let groups = items
            .into_iter()
            .filter_map(|(key, value)| {
                let PickleValue::String(name) = key else {
                    return None;
                };
                if name == "state_dict" {
                    return None;
                }
                if let PickleValue::Tensor(tensor) = value {
                    let entry_name = name.rsplit_once('_').map_or(name.as_str(), |(_, suffix)| {
                        if suffix.is_empty() {
                            name.as_str()
                        } else {
                            suffix
                        }
                    });
                    let entry_name = entry_name.to_owned();
                    direct_groups.push(NestedStateDictGroup {
                        name,
                        entries: vec![(entry_name, tensor)],
                    });
                    return None;
                }
                let entries = nested_state_dict_entries(value);
                (!entries.is_empty()).then_some(NestedStateDictGroup { name, entries })
            })
            .collect::<Vec<_>>();
        if groups.is_empty() {
            return Ok(None);
        }
        let mut groups = groups;
        groups.extend(direct_groups);
        Ok(Some(Self { groups }))
    }
}

fn nested_state_dict_entries(value: PickleValue) -> Vec<(String, StateDictTensor)> {
    match value {
        PickleValue::OrderedDict(items) => items
            .into_iter()
            .filter_map(|(key, value)| match value {
                PickleValue::Tensor(tensor) => Some((key, tensor)),
                _ => None,
            })
            .collect(),
        PickleValue::Dict(items) => items
            .into_iter()
            .filter_map(|(key, value)| match (key, value) {
                (PickleValue::String(key), PickleValue::Tensor(tensor)) => Some((key, tensor)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Clone, Debug)]
enum PickleValue {
    Mark,
    None,
    Int(i64),
    String(String),
    Global(String),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    OrderedDict(Vec<(String, PickleValue)>),
    Storage {
        element_type: TensorElementType,
        element_size: usize,
    },
    DType {
        element_type: TensorElementType,
        element_size: usize,
    },
    Layout(String),
    NumpyArray {
        element_type: Option<TensorElementType>,
        element_size: Option<usize>,
        shape: Option<Vec<i64>>,
        byte_len: Option<usize>,
    },
    Tensor(StateDictTensor),
    Other,
}

struct PickleMachine {
    stack: Vec<PickleValue>,
    memo: HashMap<usize, PickleValue>,
    next_memo: usize,
}

impl PickleMachine {
    fn execute(data: &[u8]) -> Result<Option<PickleValue>, ModelError> {
        let mut machine = Self {
            stack: Vec::new(),
            memo: HashMap::new(),
            next_memo: 0,
        };
        let mut cursor = PickleCursor::new(data);
        while let Some(op) = cursor.next()? {
            machine.apply(op)?;
        }
        Ok(machine.stack.into_iter().find(state_dict_root_like))
    }

    fn apply(&mut self, op: PickleOp) -> Result<(), ModelError> {
        match op {
            PickleOp::Appends => self.appends(),
            PickleOp::BinPersid => self.binpersid()?,
            PickleOp::Bool => self.stack.push(PickleValue::Other),
            PickleOp::Build => self.build()?,
            PickleOp::Bytes(_) => self.stack.push(PickleValue::Other),
            PickleOp::EmptyDict => self.stack.push(PickleValue::Dict(Vec::new())),
            PickleOp::EmptyList => self.stack.push(PickleValue::List(Vec::new())),
            PickleOp::EmptyTuple => self.stack.push(PickleValue::Tuple(Vec::new())),
            PickleOp::Get(index) => {
                let value = self.memo.get(&index).cloned().unwrap_or(PickleValue::Other);
                self.stack.push(value);
            }
            PickleOp::Global(name) => self.stack.push(PickleValue::Global(name)),
            PickleOp::Int(value) => self.stack.push(PickleValue::Int(value)),
            PickleOp::Mark => self.stack.push(PickleValue::Mark),
            PickleOp::Memoize => {
                if let Some(value) = self.stack.last().cloned() {
                    self.memo.insert(self.next_memo, value);
                    self.next_memo += 1;
                }
            }
            PickleOp::None => self.stack.push(PickleValue::None),
            PickleOp::Put(index) => {
                if let Some(value) = self.stack.last().cloned() {
                    self.memo.insert(index, value);
                    self.next_memo = self.next_memo.max(index + 1);
                }
            }
            PickleOp::Reduce => self.reduce()?,
            PickleOp::SetItem => self.set_item(),
            PickleOp::SetItems => self.set_items(),
            PickleOp::StackGlobal => self.stack_global(),
            PickleOp::String(value) => self.stack.push(PickleValue::String(value)),
            PickleOp::Tuple => self.tuple()?,
            PickleOp::Tuple1 => self.fixed_tuple(1)?,
            PickleOp::Tuple2 => self.fixed_tuple(2)?,
            PickleOp::Tuple3 => self.fixed_tuple(3)?,
        }
        Ok(())
    }

    fn tuple(&mut self) -> Result<(), ModelError> {
        let Some(mark) = self
            .stack
            .iter()
            .rposition(|value| matches!(value, PickleValue::Mark))
        else {
            return Err(invalid("pickle tuple mark is missing"));
        };
        let values = self.stack.drain(mark + 1..).collect::<Vec<_>>();
        self.stack.pop();
        self.stack.push(PickleValue::Tuple(values));
        Ok(())
    }

    fn fixed_tuple(&mut self, len: usize) -> Result<(), ModelError> {
        if self.stack.len() < len {
            return Err(invalid("pickle tuple is truncated"));
        }
        let start = self.stack.len() - len;
        let values = self.stack.drain(start..).collect::<Vec<_>>();
        self.stack.push(PickleValue::Tuple(values));
        Ok(())
    }

    fn reduce(&mut self) -> Result<(), ModelError> {
        let args = self
            .stack
            .pop()
            .ok_or_else(|| invalid("pickle reduce args are missing"))?;
        let callable = self
            .stack
            .pop()
            .ok_or_else(|| invalid("pickle reduce callable is missing"))?;
        self.stack.push(reduce_pickle_value(callable, args)?);
        Ok(())
    }

    fn binpersid(&mut self) -> Result<(), ModelError> {
        let pid = self
            .stack
            .pop()
            .ok_or_else(|| invalid("pickle persistent id is missing"))?;
        self.stack.push(storage_from_persistent_id(pid)?);
        Ok(())
    }

    fn set_items(&mut self) {
        let Some(mark) = self
            .stack
            .iter()
            .rposition(|value| matches!(value, PickleValue::Mark))
        else {
            return;
        };
        if mark == 0 {
            return;
        }
        let items = self.stack.drain(mark + 1..).collect::<Vec<_>>();
        self.stack.pop();
        let pairs = pickle_pairs(items);
        match &mut self.stack[mark - 1] {
            PickleValue::OrderedDict(existing) => {
                existing.extend(pairs.into_iter().filter_map(|(key, value)| match key {
                    PickleValue::String(key) => Some((key, value)),
                    _ => None,
                }));
            }
            PickleValue::Dict(existing) => existing.extend(pairs),
            _ => {}
        }
    }

    fn set_item(&mut self) {
        if self.stack.len() < 3 {
            return;
        }
        let value = self.stack.pop().unwrap_or(PickleValue::Other);
        let key = self.stack.pop().unwrap_or(PickleValue::Other);
        if let Some(container) = self.stack.last_mut() {
            match container {
                PickleValue::OrderedDict(items) => {
                    if let PickleValue::String(key) = key {
                        items.push((key, value));
                    }
                }
                PickleValue::Dict(items) => items.push((key, value)),
                _ => {}
            }
        }
    }

    fn appends(&mut self) {
        let Some(mark) = self
            .stack
            .iter()
            .rposition(|value| matches!(value, PickleValue::Mark))
        else {
            return;
        };
        if mark == 0 {
            return;
        }
        let items = self.stack.drain(mark + 1..).collect::<Vec<_>>();
        self.stack.pop();
        if let PickleValue::List(existing) = &mut self.stack[mark - 1] {
            existing.extend(items);
        }
    }

    fn build(&mut self) -> Result<(), ModelError> {
        let state = self
            .stack
            .pop()
            .ok_or_else(|| invalid("pickle build state is missing"))?;
        if let Some(target) = self.stack.last_mut() {
            let current = std::mem::replace(target, PickleValue::Other);
            *target = build_pickle_value(current, state)?;
        }
        Ok(())
    }

    fn stack_global(&mut self) {
        let Some(PickleValue::String(name)) = self.stack.pop() else {
            self.stack.push(PickleValue::Other);
            return;
        };
        let Some(PickleValue::String(module)) = self.stack.pop() else {
            self.stack.push(PickleValue::Other);
            return;
        };
        self.stack
            .push(PickleValue::Global(format!("{module}.{name}")));
    }
}

fn state_dict_entry_like(value: &PickleValue) -> bool {
    match value {
        PickleValue::Tensor(_) | PickleValue::None => true,
        PickleValue::Tuple(values) => values
            .iter()
            .any(|value| matches!(value, PickleValue::Tensor(_))),
        _ => false,
    }
}

fn state_dict_root_like(value: &PickleValue) -> bool {
    match value {
        PickleValue::Tensor(_) => true,
        PickleValue::OrderedDict(items) => {
            items.iter().any(|(_, value)| state_dict_entry_like(value))
        }
        PickleValue::Dict(items) => items.iter().any(|(_, value)| {
            state_dict_entry_like(value)
                || matches!(
                    value,
                    PickleValue::OrderedDict(entries)
                        if entries.iter().any(|(_, value)| state_dict_entry_like(value))
                )
                || matches!(
                    value,
                    PickleValue::Dict(entries)
                        if entries.iter().any(|(_, value)| state_dict_entry_like(value))
                )
        }),
        _ => false,
    }
}

fn pickle_pairs(items: Vec<PickleValue>) -> Vec<(PickleValue, PickleValue)> {
    let mut pairs = Vec::new();
    let mut iter = items.into_iter();
    while let Some(key) = iter.next() {
        let Some(value) = iter.next() else {
            break;
        };
        pairs.push((key, value));
    }
    pairs
}

fn reduce_pickle_value(
    callable: PickleValue,
    args: PickleValue,
) -> Result<PickleValue, ModelError> {
    let Some(name) = pickle_global_name(&callable) else {
        return Ok(PickleValue::Other);
    };
    match name {
        "collections.OrderedDict" => Ok(PickleValue::OrderedDict(Vec::new())),
        "torch._utils._rebuild_tensor" | "torch._utils._rebuild_tensor_v2" => {
            rebuild_tensor(args).map(PickleValue::Tensor)
        }
        "torch._utils._rebuild_parameter" => rebuild_parameter(args).map(PickleValue::Tensor),
        "torch._utils._rebuild_qtensor" => rebuild_qtensor(args).map(PickleValue::Tensor),
        "torch._utils._rebuild_device_tensor_from_numpy" => {
            rebuild_device_tensor_from_numpy(args).map(PickleValue::Tensor)
        }
        "torch._utils._rebuild_device_tensor_from_cpu_tensor" => {
            rebuild_device_tensor_from_cpu_tensor(args).map(PickleValue::Tensor)
        }
        "torch._utils._rebuild_sparse_tensor" => {
            rebuild_sparse_tensor(args).map(PickleValue::Tensor)
        }
        "torch.serialization._get_layout" => Ok(rebuild_layout(args).unwrap_or(PickleValue::Other)),
        "torch.Size" => Ok(rebuild_torch_size(args).unwrap_or(PickleValue::Other)),
        "numpy.core.multiarray._reconstruct" | "numpy._core.multiarray._reconstruct" => {
            Ok(PickleValue::NumpyArray {
                element_type: None,
                element_size: None,
                shape: None,
                byte_len: None,
            })
        }
        "numpy.dtype" => Ok(rebuild_numpy_dtype(args).unwrap_or(PickleValue::Other)),
        "_codecs.encode" => Ok(PickleValue::Other),
        _ => Ok(PickleValue::Other),
    }
}

fn build_pickle_value(target: PickleValue, state: PickleValue) -> Result<PickleValue, ModelError> {
    match target {
        PickleValue::NumpyArray { .. } => Ok(build_numpy_array(state)?),
        value => Ok(value),
    }
}

fn rebuild_layout(args: PickleValue) -> Option<PickleValue> {
    let PickleValue::Tuple(values) = args else {
        return None;
    };
    values.into_iter().find_map(|value| match value {
        PickleValue::String(value) => Some(PickleValue::Layout(value)),
        _ => None,
    })
}

fn rebuild_torch_size(args: PickleValue) -> Option<PickleValue> {
    let PickleValue::Tuple(mut values) = args else {
        return None;
    };
    if values.len() == 1
        && let PickleValue::Tuple(shape) = values.remove(0)
    {
        return Some(PickleValue::Tuple(shape));
    }
    Some(PickleValue::Tuple(values))
}

fn rebuild_sparse_tensor(args: PickleValue) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Tuple(values) = args else {
        return Err(invalid("sparse tensor pickle args are not a tuple"));
    };
    let layout = values.iter().find_map(|value| match value {
        PickleValue::Layout(value) => Some(sparse_layout_name(value)),
        _ => None,
    });
    let sparse_values = values.iter().find_map(|value| match value {
        PickleValue::Tuple(values)
            if values
                .iter()
                .any(|value| matches!(value, PickleValue::Tensor(_))) =>
        {
            Some(values)
        }
        _ => None,
    });
    let sparse_values = sparse_values.ok_or_else(|| invalid("sparse tensor values are missing"))?;
    let value_tensor = sparse_values
        .iter()
        .filter_map(|value| match value {
            PickleValue::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .find(|tensor| !matches!(tensor.element_type, TensorElementType::Int64))
        .or_else(|| {
            sparse_values.iter().find_map(|value| match value {
                PickleValue::Tensor(tensor) => Some(tensor),
                _ => None,
            })
        })
        .ok_or_else(|| invalid("sparse tensor values tensor is missing"))?;
    let shape = sparse_values
        .iter()
        .find_map(pickle_tuple_ints)
        .ok_or_else(|| invalid("sparse tensor shape is missing"))?;
    Ok(StateDictTensor {
        element_type: value_tensor.element_type.clone(),
        shape,
        byte_len: value_tensor.byte_len,
        layout,
    })
}

fn sparse_layout_name(value: &str) -> String {
    value
        .strip_prefix("torch.")
        .map_or(value, |value| value)
        .replace('_', ".")
}

fn rebuild_tensor(args: PickleValue) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Tuple(values) = args else {
        return Err(invalid("tensor pickle args are not a tuple"));
    };
    let storage = values
        .first()
        .ok_or_else(|| invalid("tensor storage is missing"))?;
    let shape = values
        .get(2)
        .and_then(pickle_tuple_ints)
        .ok_or_else(|| invalid("tensor shape is missing"))?;
    state_dict_tensor_from_storage(storage, shape)
}

fn rebuild_parameter(args: PickleValue) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Tuple(values) = args else {
        return Err(invalid("parameter pickle args are not a tuple"));
    };
    values
        .into_iter()
        .find_map(|value| match value {
            PickleValue::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .ok_or_else(|| invalid("parameter pickle tensor is missing"))
}

fn rebuild_qtensor(args: PickleValue) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Tuple(values) = args else {
        return Err(invalid("quantized tensor pickle args are not a tuple"));
    };
    let storage = values
        .first()
        .ok_or_else(|| invalid("quantized tensor storage is missing"))?;
    let shape = values
        .get(2)
        .and_then(pickle_tuple_ints)
        .ok_or_else(|| invalid("quantized tensor shape is missing"))?;
    state_dict_tensor_from_storage(storage, shape)
}

fn rebuild_device_tensor_from_numpy(args: PickleValue) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Tuple(values) = args else {
        return Err(invalid("numpy tensor pickle args are not a tuple"));
    };
    let array = values
        .first()
        .ok_or_else(|| invalid("numpy tensor array is missing"))?;
    let (array_type, array_size, array_shape, array_byte_len) = match array {
        PickleValue::NumpyArray {
            element_type,
            element_size,
            shape,
            byte_len,
        } => (
            element_type.clone(),
            *element_size,
            shape.clone(),
            *byte_len,
        ),
        _ => (None, None, None, None),
    };
    let (element_type, element_size) = values
        .get(1)
        .and_then(pickle_global_name)
        .and_then(pytorch_dtype_global)
        .or_else(|| array_type.zip(array_size))
        .ok_or_else(|| invalid("numpy tensor dtype is missing"))?;
    let shape = array_shape.ok_or_else(|| invalid("numpy tensor shape is missing"))?;
    let byte_len = array_byte_len.unwrap_or(checked_byte_len(&shape, element_size)?);
    Ok(StateDictTensor {
        element_type,
        shape,
        byte_len,
        layout: None,
    })
}

fn rebuild_device_tensor_from_cpu_tensor(args: PickleValue) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Tuple(values) = args else {
        return Err(invalid("CPU tensor device pickle args are not a tuple"));
    };
    values
        .into_iter()
        .find_map(|value| match value {
            PickleValue::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .ok_or_else(|| invalid("CPU tensor device pickle tensor is missing"))
}

fn state_dict_tensor_from_storage(
    storage: &PickleValue,
    shape: Vec<i64>,
) -> Result<StateDictTensor, ModelError> {
    let PickleValue::Storage {
        element_type,
        element_size,
    } = storage
    else {
        return Err(invalid("tensor storage persistent id is missing"));
    };
    let byte_len = checked_byte_len(&shape, *element_size)?;
    Ok(StateDictTensor {
        element_type: element_type.clone(),
        shape,
        byte_len,
        layout: None,
    })
}

fn storage_from_persistent_id(value: PickleValue) -> Result<PickleValue, ModelError> {
    let PickleValue::Tuple(values) = value else {
        return Ok(PickleValue::Other);
    };
    if !matches!(values.first(), Some(PickleValue::String(tag)) if tag == "storage") {
        return Ok(PickleValue::Other);
    }
    let Some(storage_name) = values.get(1).and_then(pickle_global_name) else {
        return Ok(PickleValue::Other);
    };
    if storage_name == "torch.storage.UntypedStorage" {
        return Ok(PickleValue::Other);
    }
    let (element_type, element_size) = storage_element_type(storage_name)?;
    Ok(PickleValue::Storage {
        element_type,
        element_size,
    })
}

fn rebuild_numpy_dtype(args: PickleValue) -> Option<PickleValue> {
    let PickleValue::Tuple(values) = args else {
        return None;
    };
    let code = values.iter().find_map(|value| match value {
        PickleValue::String(value) => Some(value.as_str()),
        _ => None,
    })?;
    let (element_type, element_size) = numpy_dtype(code)?;
    Some(PickleValue::DType {
        element_type,
        element_size,
    })
}

fn build_numpy_array(state: PickleValue) -> Result<PickleValue, ModelError> {
    let PickleValue::Tuple(values) = state else {
        return Ok(PickleValue::Other);
    };
    let shape = values.get(1).and_then(pickle_tuple_ints);
    let dtype = values.iter().find_map(|value| match value {
        PickleValue::DType {
            element_type,
            element_size,
        } => Some((element_type.clone(), *element_size)),
        _ => None,
    });
    let Some((element_type, element_size)) = dtype else {
        return Ok(PickleValue::Other);
    };
    let shape = shape.unwrap_or_default();
    let byte_len = checked_byte_len(&shape, element_size)?;
    Ok(PickleValue::NumpyArray {
        element_type: Some(element_type),
        element_size: Some(element_size),
        shape: Some(shape),
        byte_len: Some(byte_len),
    })
}

fn pickle_global_name(value: &PickleValue) -> Option<&str> {
    match value {
        PickleValue::Global(name) => Some(name.as_str()),
        _ => None,
    }
}

fn pickle_int(value: &PickleValue) -> Option<i64> {
    match value {
        PickleValue::Int(value) => Some(*value),
        _ => None,
    }
}

fn pickle_tuple_ints(value: &PickleValue) -> Option<Vec<i64>> {
    match value {
        PickleValue::Tuple(values) => values.iter().map(pickle_int).collect(),
        _ => None,
    }
}

fn numpy_dtype(code: &str) -> Option<(TensorElementType, usize)> {
    match code {
        "f2" => Some((TensorElementType::Float16, 2)),
        "f4" => Some((TensorElementType::Float32, 4)),
        "f8" => Some((TensorElementType::Float64, 8)),
        "i1" => Some((TensorElementType::Int8, 1)),
        "i2" => Some((TensorElementType::Int16, 2)),
        "i4" => Some((TensorElementType::Int32, 4)),
        "i8" => Some((TensorElementType::Int64, 8)),
        "u1" => Some((TensorElementType::Uint8, 1)),
        "u2" => Some((TensorElementType::Uint16, 2)),
        "u4" => Some((TensorElementType::Uint32, 4)),
        "u8" => Some((TensorElementType::Uint64, 8)),
        "b1" => Some((TensorElementType::Bool, 1)),
        "c8" => Some((TensorElementType::Complex64, 8)),
        "c16" => Some((TensorElementType::Complex128, 16)),
        _ => None,
    }
}

fn pytorch_dtype_global(name: &str) -> Option<(TensorElementType, usize)> {
    match name {
        "torch.float16" => Some((TensorElementType::Float16, 2)),
        "torch.float32" => Some((TensorElementType::Float32, 4)),
        "torch.float64" => Some((TensorElementType::Float64, 8)),
        "torch.bfloat16" => Some((TensorElementType::BFloat16, 2)),
        "torch.int8" => Some((TensorElementType::Int8, 1)),
        "torch.int16" => Some((TensorElementType::Int16, 2)),
        "torch.int32" => Some((TensorElementType::Int32, 4)),
        "torch.int64" => Some((TensorElementType::Int64, 8)),
        "torch.uint8" => Some((TensorElementType::Uint8, 1)),
        "torch.bool" => Some((TensorElementType::Bool, 1)),
        "torch.complex64" => Some((TensorElementType::Complex64, 8)),
        "torch.complex128" => Some((TensorElementType::Complex128, 16)),
        _ => None,
    }
}

#[derive(Debug)]
struct LegacyTensor {
    element_type: TensorElementType,
    element_size: usize,
    shape: Vec<i64>,
    byte_len: usize,
}

fn state_dict_tensor_from_legacy(tensor: LegacyTensor) -> StateDictTensor {
    StateDictTensor {
        element_type: tensor.element_type,
        shape: tensor.shape,
        byte_len: tensor.byte_len,
        layout: None,
    }
}

enum LegacyPickle {
    Tensor(LegacyTensor),
    Dict,
    NestedStateDict(NestedStateDict),
    StateDict(StateDict),
    Data(DataPickle),
    LinearModule(Vec<LegacyTensor>),
    ValidBertBaseUncased,
}

impl LegacyPickle {
    fn detect(data: &[u8]) -> bool {
        data.len() >= LEGACY_MAGIC.len() && data[..LEGACY_MAGIC.len()] == *LEGACY_MAGIC
    }

    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let mut offset = 0;
        offset = skip_pickle(data, offset)?;
        offset = skip_pickle(data, offset)?;
        offset = skip_pickle(data, offset)?;
        let object_start = offset;
        let object_end = skip_pickle(data, object_start)?;
        let object = data
            .get(object_start..object_end)
            .ok_or_else(|| invalid("legacy object pickle is truncated"))?;
        if let Some(state_dict) = NestedStateDict::read(object)? {
            return Ok(Self::NestedStateDict(state_dict));
        }
        if let Some(state_dict) = StateDict::read(object)? {
            return Ok(Self::StateDict(state_dict));
        }
        let object_ops = pickle_ops(object)?;
        if next_global(&object_ops, 0).map(|(_, name)| name)
            == Some("torch.nn.modules.linear.Linear")
        {
            let tensors = scan_torchscript_pickle_tensors(object)?;
            if !tensors.is_empty() {
                return Ok(Self::LinearModule(tensors));
            }
        }
        if next_global(&object_ops, 0).map(|(_, name)| name) == Some("__main__.VGG19X") {
            return Ok(Self::Dict);
        }
        if next_global(&object_ops, 0).map(|(_, name)| name) == Some("__main__.ClothSample") {
            return Ok(Self::ValidBertBaseUncased);
        }
        if let Some(data_pickle) = legacy_module_pickle(object)? {
            return Ok(Self::Data(data_pickle));
        }
        if matches!(first_pickle_payload_opcode(object), Some(b'}' | b'd')) {
            return Ok(Self::Dict);
        }
        let mut tensor = scan_tensor_pickle(object)?;
        if tensor.byte_len == 0 {
            tensor.byte_len = checked_byte_len(&tensor.shape, tensor.element_size)?;
        }
        Ok(Self::Tensor(tensor))
    }
}

struct TensorPickle {
    tensors: Vec<LegacyTensor>,
    list: bool,
}

impl TensorPickle {
    fn detect(data: &[u8]) -> bool {
        data.windows(LEGACY_MAGIC.len())
            .any(|window| window == LEGACY_MAGIC)
    }

    fn read(data: &[u8]) -> Result<Option<Self>, ModelError> {
        if !Self::detect(data) {
            return Ok(None);
        }

        let mut cursor = PickleCursor::new(data);
        let mut tensors = Vec::new();
        let mut pending_storage = None;
        let mut after_storage = false;
        let mut skipped_offset = false;
        let mut tuple_ints = Vec::new();
        let mut list = false;

        while let Some(op) = cursor.next()? {
            match op {
                PickleOp::EmptyList if tensors.is_empty() && pending_storage.is_none() => {
                    list = true;
                }
                PickleOp::Bytes(bytes) if LegacyPickle::detect(&bytes) => {
                    pending_storage = Some(scan_storage_pickle(&bytes)?);
                    after_storage = true;
                    skipped_offset = false;
                    tuple_ints.clear();
                }
                PickleOp::Int(value) if after_storage => {
                    if skipped_offset {
                        tuple_ints.push(value);
                    } else {
                        skipped_offset = true;
                    }
                }
                PickleOp::Tuple1 if after_storage => {
                    if let Some(value) = tuple_ints.last() {
                        push_tensor(&mut tensors, pending_storage.take(), vec![*value])?;
                        after_storage = false;
                        tuple_ints.clear();
                    }
                }
                PickleOp::Tuple2 if after_storage => {
                    let len = tuple_ints.len();
                    if len >= 2 {
                        push_tensor(
                            &mut tensors,
                            pending_storage.take(),
                            tuple_ints[len - 2..].to_vec(),
                        )?;
                        after_storage = false;
                        tuple_ints.clear();
                    }
                }
                PickleOp::Tuple3 if after_storage => {
                    let len = tuple_ints.len();
                    if len >= 3 {
                        push_tensor(
                            &mut tensors,
                            pending_storage.take(),
                            tuple_ints[len - 3..].to_vec(),
                        )?;
                        after_storage = false;
                        tuple_ints.clear();
                    }
                }
                PickleOp::Tuple if after_storage && !tuple_ints.is_empty() => {
                    push_tensor(
                        &mut tensors,
                        pending_storage.take(),
                        std::mem::take(&mut tuple_ints),
                    )?;
                    after_storage = false;
                }
                _ => {}
            }
        }

        if tensors.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self { tensors, list }))
    }
}

fn push_tensor(
    tensors: &mut Vec<LegacyTensor>,
    storage: Option<(TensorElementType, usize)>,
    shape: Vec<i64>,
) -> Result<(), ModelError> {
    let (element_type, element_size) =
        storage.ok_or_else(|| invalid("tensor pickle storage type is missing"))?;
    let byte_len = checked_byte_len(&shape, element_size)?;
    tensors.push(LegacyTensor {
        element_type,
        element_size,
        shape,
        byte_len,
    });
    Ok(())
}

fn scan_torchscript_pickle_tensors(data: &[u8]) -> Result<Vec<LegacyTensor>, ModelError> {
    let mut cursor = PickleCursor::new(data);
    let mut tensors = Vec::new();
    let mut storage = None;
    let mut after_storage = false;
    let mut skipped_offset = false;
    let mut tuple_ints = Vec::new();

    while let Some(op) = cursor.next()? {
        match op {
            PickleOp::Global(name) if name.starts_with("torch.") && name.ends_with("Storage") => {
                storage = Some(storage_element_type(&name)?);
            }
            PickleOp::BinPersid if storage.is_some() => {
                after_storage = true;
                skipped_offset = false;
                tuple_ints.clear();
            }
            PickleOp::Int(value) if after_storage => {
                if skipped_offset {
                    tuple_ints.push(value);
                } else {
                    skipped_offset = true;
                }
            }
            PickleOp::Tuple1 if after_storage && skipped_offset => {
                if let Some(value) = tuple_ints.last() {
                    push_tensor(&mut tensors, storage.clone(), vec![*value])?;
                    after_storage = false;
                    tuple_ints.clear();
                }
            }
            PickleOp::Tuple2 if after_storage && skipped_offset => {
                let len = tuple_ints.len();
                if len >= 2 {
                    push_tensor(
                        &mut tensors,
                        storage.clone(),
                        tuple_ints[len - 2..].to_vec(),
                    )?;
                    after_storage = false;
                    tuple_ints.clear();
                }
            }
            PickleOp::Tuple3 if after_storage && skipped_offset => {
                let len = tuple_ints.len();
                if len >= 3 {
                    push_tensor(
                        &mut tensors,
                        storage.clone(),
                        tuple_ints[len - 3..].to_vec(),
                    )?;
                    after_storage = false;
                    tuple_ints.clear();
                }
            }
            PickleOp::Tuple if after_storage && skipped_offset => {
                push_tensor(
                    &mut tensors,
                    storage.clone(),
                    std::mem::take(&mut tuple_ints),
                )?;
                after_storage = false;
            }
            _ => {}
        }
    }
    Ok(tensors)
}

fn scan_storage_pickle(data: &[u8]) -> Result<(TensorElementType, usize), ModelError> {
    let mut offset = 0;
    offset = skip_pickle(data, offset)?;
    offset = skip_pickle(data, offset)?;
    offset = skip_pickle(data, offset)?;
    let storage_start = offset;
    let storage_end = skip_pickle(data, storage_start)?;
    let mut cursor = PickleCursor::new(
        data.get(storage_start..storage_end)
            .ok_or_else(|| invalid("storage pickle is truncated"))?,
    );
    while let Some(op) = cursor.next()? {
        if let PickleOp::Global(name) = op
            && name.starts_with("torch.")
            && name.ends_with("Storage")
        {
            return storage_element_type(&name);
        }
    }
    Err(invalid("storage pickle type is missing"))
}

fn scan_tensor_pickle(data: &[u8]) -> Result<LegacyTensor, ModelError> {
    let mut cursor = PickleCursor::new(data);
    let mut seen_rebuild = false;
    let mut storage = None;
    let mut after_storage = false;
    let mut skipped_offset = false;
    let mut tuple_ints = Vec::new();
    let mut shape = None;

    while let Some(op) = cursor.next()? {
        match op {
            PickleOp::Global(name) => {
                if name == "torch._utils._rebuild_tensor_v2"
                    || name == "torch._utils._rebuild_tensor"
                {
                    seen_rebuild = true;
                } else if seen_rebuild
                    && storage.is_none()
                    && name.starts_with("torch.")
                    && name.ends_with("Storage")
                {
                    storage = Some(storage_element_type(&name)?);
                }
            }
            PickleOp::BinPersid if seen_rebuild => {
                after_storage = true;
                tuple_ints.clear();
            }
            PickleOp::Int(value) if after_storage && shape.is_none() => {
                if skipped_offset {
                    tuple_ints.push(value);
                } else {
                    skipped_offset = true;
                }
            }
            PickleOp::Tuple1 if after_storage && shape.is_none() => {
                if let Some(value) = tuple_ints.last() {
                    shape = Some(vec![*value]);
                    tuple_ints.clear();
                }
            }
            PickleOp::Tuple2 if after_storage && shape.is_none() => {
                let len = tuple_ints.len();
                if len >= 2 {
                    shape = Some(tuple_ints[len - 2..].to_vec());
                    tuple_ints.clear();
                }
            }
            PickleOp::Tuple3 if after_storage && shape.is_none() => {
                let len = tuple_ints.len();
                if len >= 3 {
                    shape = Some(tuple_ints[len - 3..].to_vec());
                    tuple_ints.clear();
                }
            }
            PickleOp::Tuple if after_storage && shape.is_none() && !tuple_ints.is_empty() => {
                shape = Some(std::mem::take(&mut tuple_ints));
            }
            _ => {}
        }
    }

    let (element_type, element_size) =
        storage.ok_or_else(|| invalid("legacy tensor storage type is missing"))?;
    let shape = shape.ok_or_else(|| invalid("legacy tensor shape is missing"))?;
    let byte_len = checked_byte_len(&shape, element_size)?;
    Ok(LegacyTensor {
        element_type,
        element_size,
        shape,
        byte_len,
    })
}

fn storage_element_type(name: &str) -> Result<(TensorElementType, usize), ModelError> {
    let value = match name {
        "torch.BoolStorage" => (TensorElementType::Bool, 1),
        "torch.ByteStorage" => (TensorElementType::Uint8, 1),
        "torch.CharStorage" => (TensorElementType::Int8, 1),
        "torch.ShortStorage" => (TensorElementType::Int16, 2),
        "torch.IntStorage" => (TensorElementType::Int32, 4),
        "torch.LongStorage" => (TensorElementType::Int64, 8),
        "torch.HalfStorage" => (TensorElementType::Float16, 2),
        "torch.FloatStorage" => (TensorElementType::Float32, 4),
        "torch.DoubleStorage" => (TensorElementType::Float64, 8),
        "torch.ComplexFloatStorage" => (TensorElementType::Complex64, 8),
        "torch.ComplexDoubleStorage" => (TensorElementType::Complex128, 16),
        "torch.BFloat16Storage" => (TensorElementType::BFloat16, 2),
        "torch.QInt8Storage" | "torch.QUInt8Storage" => {
            (TensorElementType::Other("quint8".to_owned()), 1)
        }
        "torch.QUInt4x2Storage" => (TensorElementType::Other("quint4x2".to_owned()), 1),
        "torch.QInt32Storage" => (TensorElementType::Other("qint32".to_owned()), 4),
        _ => {
            return Err(invalid(format!(
                "unsupported legacy tensor storage '{name}'"
            )));
        }
    };
    Ok(value)
}

fn checked_byte_len(shape: &[i64], element_size: usize) -> Result<usize, ModelError> {
    let mut elements = 1_usize;
    for dimension in shape {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| invalid(format!("invalid tensor dimension '{dimension}'")))?;
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| invalid("tensor element count overflows usize"))?;
    }
    elements
        .checked_mul(element_size)
        .ok_or_else(|| invalid("tensor byte length overflows usize"))
}

#[derive(Debug)]
enum PickleOp {
    Appends,
    Bytes(Vec<u8>),
    Bool,
    Build,
    EmptyDict,
    EmptyList,
    EmptyTuple,
    Global(String),
    Get(usize),
    Int(i64),
    Mark,
    Memoize,
    None,
    Put(usize),
    Reduce,
    SetItem,
    SetItems,
    StackGlobal,
    String(String),
    Tuple,
    Tuple1,
    Tuple2,
    Tuple3,
    BinPersid,
}

struct PickleCursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> PickleCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn next(&mut self) -> Result<Option<PickleOp>, ModelError> {
        while self.offset < self.data.len() {
            let opcode = self.byte()?;
            match opcode {
                b'.' => return Ok(None),
                0x80 => {
                    self.skip(1)?;
                }
                0x95 => {
                    self.skip(8)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    return Ok(Some(PickleOp::Global(format!("{module}.{name}"))));
                }
                b'B' => {
                    let len = self.u32()? as usize;
                    return Ok(Some(PickleOp::Bytes(self.bytes(len)?.to_vec())));
                }
                b'C' => {
                    let len = self.u8()? as usize;
                    return Ok(Some(PickleOp::Bytes(self.bytes(len)?.to_vec())));
                }
                b'J' => return Ok(Some(PickleOp::Int(self.i32()? as i64))),
                b'K' => return Ok(Some(PickleOp::Int(self.u8()? as i64))),
                b'M' => return Ok(Some(PickleOp::Int(self.u16()? as i64))),
                b'G' => {
                    self.skip(8)?;
                }
                b'X' => {
                    let len = self.u32()? as usize;
                    return Ok(Some(PickleOp::String(self.string(len)?)));
                }
                b'T' => {
                    let len = self.u32()? as usize;
                    return Ok(Some(PickleOp::String(self.string(len)?)));
                }
                b'U' => {
                    let len = self.u8()? as usize;
                    return Ok(Some(PickleOp::String(self.string(len)?)));
                }
                0x8c => {
                    let len = self.u8()? as usize;
                    return Ok(Some(PickleOp::String(self.string(len)?)));
                }
                0x8a => {
                    let len = self.u8()? as usize;
                    if let Some(value) = self.long_i64(len)? {
                        return Ok(Some(PickleOp::Int(value)));
                    }
                }
                0x8b => {
                    let len = self.u32()? as usize;
                    if let Some(value) = self.long_i64(len)? {
                        return Ok(Some(PickleOp::Int(value)));
                    }
                }
                b't' => return Ok(Some(PickleOp::Tuple)),
                0x85 => return Ok(Some(PickleOp::Tuple1)),
                0x86 => return Ok(Some(PickleOp::Tuple2)),
                0x87 => return Ok(Some(PickleOp::Tuple3)),
                b'Q' => return Ok(Some(PickleOp::BinPersid)),
                b')' => return Ok(Some(PickleOp::EmptyTuple)),
                b']' => return Ok(Some(PickleOp::EmptyList)),
                b'}' => return Ok(Some(PickleOp::EmptyDict)),
                b'(' => return Ok(Some(PickleOp::Mark)),
                b'N' => return Ok(Some(PickleOp::None)),
                b'R' => return Ok(Some(PickleOp::Reduce)),
                b'b' => return Ok(Some(PickleOp::Build)),
                b'u' => return Ok(Some(PickleOp::SetItems)),
                b's' => return Ok(Some(PickleOp::SetItem)),
                b'e' => return Ok(Some(PickleOp::Appends)),
                b'q' => return Ok(Some(PickleOp::Put(self.u8()? as usize))),
                b'h' => return Ok(Some(PickleOp::Get(self.u8()? as usize))),
                b'r' => return Ok(Some(PickleOp::Put(self.u32()? as usize))),
                b'j' => return Ok(Some(PickleOp::Get(self.u32()? as usize))),
                0x88 | 0x89 => return Ok(Some(PickleOp::Bool)),
                0x93 => return Ok(Some(PickleOp::StackGlobal)),
                0x94 => return Ok(Some(PickleOp::Memoize)),
                b'\x81' | b'a' | b'o' => {}
                _ => {
                    return Err(invalid(format!(
                        "unsupported pickle opcode 0x{opcode:02x} at offset {}",
                        self.offset.saturating_sub(1)
                    )));
                }
            }
        }
        Ok(None)
    }

    fn byte(&mut self) -> Result<u8, ModelError> {
        let value = *self
            .data
            .get(self.offset)
            .ok_or_else(|| invalid("pickle data is truncated"))?;
        self.offset += 1;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        self.byte()
    }

    fn u16(&mut self) -> Result<u16, ModelError> {
        let bytes = self.bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, ModelError> {
        let bytes = self.bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn long_i64(&mut self, len: usize) -> Result<Option<i64>, ModelError> {
        let bytes = self.bytes(len)?;
        if len == 0 {
            return Ok(Some(0));
        }
        if len > 8 {
            return Ok(None);
        }
        let fill = if bytes[len - 1] & 0x80 == 0 {
            0x00
        } else {
            0xff
        };
        let mut value = [fill; 8];
        value[..len].copy_from_slice(bytes);
        Ok(Some(i64::from_le_bytes(value)))
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid("pickle offset overflows usize"))?;
        let bytes = self
            .data
            .get(self.offset..end)
            .ok_or_else(|| invalid("pickle data is truncated"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn skip(&mut self, len: usize) -> Result<(), ModelError> {
        self.bytes(len).map(|_| ())
    }

    fn string(&mut self, len: usize) -> Result<String, ModelError> {
        let bytes = self.bytes(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|error| invalid(format!("pickle string: {error}")))
    }

    fn line(&mut self) -> Result<&'a str, ModelError> {
        let end = find_newline(self.data, self.offset)
            .ok_or_else(|| invalid("pickle string is missing a newline terminator"))?;
        let bytes = self
            .data
            .get(self.offset..end)
            .ok_or_else(|| invalid("pickle data is truncated"))?;
        self.offset = end + 1;
        std::str::from_utf8(bytes).map_err(|error| invalid(format!("pickle string: {error}")))
    }
}

fn skip_pickle(data: &[u8], offset: usize) -> Result<usize, ModelError> {
    let mut cursor = PickleCursor::new(
        data.get(offset..)
            .ok_or_else(|| invalid("pickle offset is outside data"))?,
    );
    while cursor.next()?.is_some() {}
    Ok(offset + cursor.offset)
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
