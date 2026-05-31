use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
    TypeInfo, Value, ValueId,
};
use serde::Deserialize;

const FORMAT: &str = "ncnn";
const PNNX_FORMAT: &str = "PNNX";
const METADATA: &str = include_str!("metadata/ncnn-metadata.json");

pub struct NcnnFormat;

impl ModelFormat for NcnnFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["param", "ncnn", "bin"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        let Some(path) = input.path else {
            return Confidence::None;
        };
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if name.ends_with(".weights.ncnn")
            || (name.ends_with(".bin") && !name.ends_with(".param.bin"))
        {
            return if sidecar_definition(path).is_some_and(|path| path.exists()) {
                Confidence::High
            } else {
                Confidence::None
            };
        }
        if name.ends_with(".param.bin") {
            return if input.data.len() >= 4
                && u32::from_le_bytes(input.data[0..4].try_into().unwrap()) == 0x0076_85dd
            {
                Confidence::High
            } else {
                Confidence::None
            };
        }
        if !(name.ends_with(".param") || name.ends_with(".cfg.ncnn")) {
            return Confidence::None;
        }
        let Ok(text) = std::str::from_utf8(
            input
                .data
                .get(..input.data.len().min(65536))
                .unwrap_or(input.data),
        ) else {
            return Confidence::None;
        };
        if parse_header(text).is_some() {
            Confidence::High
        } else {
            Confidence::None
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        parse_with_blob_data(input.data, input.path, None)
    }
}

pub(crate) fn parse_with_blob_data(
    data: &[u8],
    path: Option<&Path>,
    blob_data: Option<Vec<u8>>,
) -> Result<Model, ModelError> {
    let file_name = path
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
        .unwrap_or_default();
    let blobs = |format| {
        blob_data
            .clone()
            .map(BlobReader::from_data)
            .unwrap_or_else(|| BlobReader::open(path, format))
    };
    if file_name.ends_with(".weights.ncnn")
        || (file_name.ends_with(".bin") && !file_name.ends_with(".param.bin"))
    {
        let definition = path
            .and_then(sidecar_definition)
            .ok_or_else(|| invalid("Required ncnn model definition not found."))?;
        let definition = read_definition(&definition)?;
        return lower_model(
            definition.format,
            definition.layers,
            BlobReader::from_data(data.to_vec()),
            definition.decode_enums,
        );
    }
    if file_name.ends_with(".param.bin") {
        let layers = parse_binary_param(data)?;
        return lower_model(FORMAT, layers, blobs(FORMAT), false);
    }
    let text = std::str::from_utf8(data)
        .map_err(|error| invalid(format!("ncnn param is not UTF-8: {error}")))?;
    let format = if is_pnnx_text(text) {
        PNNX_FORMAT
    } else {
        FORMAT
    };
    let layers = parse_text_param(text)?;
    lower_model(format, layers, blobs(format), true)
}

fn lower_model(
    format: &'static str,
    layers: Vec<NcnnLayer>,
    blobs: BlobReader,
    decode_enums: bool,
) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: format,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let metadata = metadata();
    let mut values = NcnnValues::default();
    let mut layers = layers;

    for layer in &mut layers {
        if let Some(ParamValue::List(items)) = layer.params.get("30").cloned() {
            for output in &layer.outputs {
                if let Some(type_info) = shape_param_type(&items) {
                    values.set_type(output, type_info);
                }
            }
            layer.params.remove("30");
        }
    }

    let mut blobs = blobs;
    for layer in layers {
        match layer.op_type.as_str() {
            "Input" | "16" => {
                let shape = layer
                    .params
                    .values()
                    .iter()
                    .map(|value| dimension_from_param(&mut model, value))
                    .collect::<Vec<_>>();
                let type_info = TypeInfo {
                    element_type: Some(TensorElementType::Float32),
                    layout: None,
                    denotation: None,
                    shape,
                };
                let ids = layer
                    .outputs
                    .iter()
                    .map(|output| {
                        values.value(
                            &mut model,
                            &mut graph,
                            output,
                            Some(type_info.clone()),
                            None,
                        )
                    })
                    .collect::<Vec<_>>();
                for value_id in ids {
                    graph.values[value_id.index()].is_graph_input = true;
                    if !graph.inputs.contains(&value_id) {
                        graph.inputs.push(value_id);
                    }
                }
            }
            "pnnx.Input" => {
                let ids = layer
                    .outputs
                    .iter()
                    .map(|output| {
                        let type_info = route_type(&mut model, &mut layer.params.clone(), output);
                        values.value(&mut model, &mut graph, output, type_info, None)
                    })
                    .collect::<Vec<_>>();
                for value_id in ids {
                    graph.values[value_id.index()].is_graph_input = true;
                    if !graph.inputs.contains(&value_id) {
                        graph.inputs.push(value_id);
                    }
                }
            }
            "pnnx.Output" => {
                for input in &layer.inputs {
                    let type_info = route_type(&mut model, &mut layer.params.clone(), input);
                    let value_id = values.value(&mut model, &mut graph, input, type_info, None);
                    graph.values[value_id.index()].is_graph_output = true;
                    if !graph.outputs.contains(&value_id) {
                        graph.outputs.push(value_id);
                    }
                }
            }
            _ => {
                let node = lower_node(
                    &mut model,
                    &mut graph,
                    graph_id,
                    metadata,
                    &mut blobs,
                    &mut values,
                    layer,
                    format,
                    decode_enums,
                )?;
                let node_id = graph.add_node(node);
                for value_id in graph.nodes[node_id.index()]
                    .inputs
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
                {
                    if !graph.values[value_id.index()].consumers.contains(&node_id) {
                        graph.values[value_id.index()].consumers.push(node_id);
                    }
                }
                for value_id in graph.nodes[node_id.index()]
                    .outputs
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>()
                {
                    graph.values[value_id.index()].producer = Some(node_id);
                }
            }
        }
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn lower_node(
    model: &mut Model,
    graph: &mut Graph,
    graph_id: netron_rs_core::GraphId,
    metadata: &NcnnMetadata,
    blobs: &mut BlobReader,
    values: &mut NcnnValues,
    mut layer: NcnnLayer,
    format: &'static str,
    decode_enums: bool,
) -> Result<Node, ModelError> {
    let op_name = metadata.type_name(&layer.op_type);
    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: model.intern(&op_name),
            overload: None,
            version: None,
            origin: format,
        },
    );
    if !layer.name.is_empty() {
        node.name = Some(model.intern(&layer.name));
    }

    for key in layer
        .params
        .iter()
        .filter(|(key, _)| key.starts_with('$'))
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>()
    {
        layer.params.remove(&key);
    }

    for input in &layer.inputs {
        let type_info = route_type(model, &mut layer.params, input);
        let value_id = values.value(model, graph, input, type_info, None);
        node.inputs.push(Some(value_id));
    }
    for output in &layer.outputs {
        let type_info = route_type(model, &mut layer.params, output);
        let value_id = values.value(model, graph, output, type_info, None);
        node.outputs.push(Some(value_id));
    }

    add_operator_weights(
        model,
        graph,
        blobs,
        values,
        &mut node,
        &op_name,
        &mut layer.params,
    )?;

    node.attributes = layer
        .params
        .iter()
        .filter(|(key, _)| !key.starts_with('@') && !key.starts_with('#'))
        .map(|(key, value)| lower_attribute(model, metadata, &op_name, key, value, decode_enums))
        .collect();
    Ok(node)
}

fn add_operator_weights(
    model: &mut Model,
    graph: &mut Graph,
    blobs: &mut BlobReader,
    values: &mut NcnnValues,
    node: &mut Node,
    op: &str,
    params: &mut ParamMap,
) -> Result<(), ModelError> {
    let mut add = |name: &str, shape: Vec<i64>, code: i64| -> Result<(), ModelError> {
        let value_id = add_weight(model, graph, blobs, values, shape, code)?;
        node.inputs.push(Some(value_id));
        let _ = name;
        Ok(())
    };

    match op {
        "BatchNorm" => {
            let channels = params.i64("0", 0);
            add("slope", vec![channels], 1)?;
            add("mean", vec![channels], 1)?;
            add("variance", vec![channels], 1)?;
            add("bias", vec![channels], 1)?;
        }
        "InnerProduct" => {
            let num_output = params.i64("0", 0);
            let bias_term = params.i64("1", 0);
            let weight_data_size = params.i64("2", 0);
            let int8_scale_term = params.i64("8", 0);
            let input_size = if num_output > 0 {
                weight_data_size / num_output
            } else {
                0
            };
            add("weight", vec![num_output, input_size], 0)?;
            if bias_term != 0 {
                add("bias", vec![num_output], 1)?;
            }
            if int8_scale_term != 0 {
                add("weight_scales", vec![num_output], 1)?;
                add("bottom_scales", vec![1], 1)?;
            }
            params.remove("2");
        }
        "Bias" => {
            add("bias", vec![params.i64("0", 0)], 1)?;
        }
        "Embed" => {
            let num_output = params.i64("0", 0);
            let weight_data_size = params.i64("3", 0);
            let num_input = if num_output > 0 {
                weight_data_size / num_output
            } else {
                0
            };
            add("weight", vec![num_input, num_output], 0)?;
            if params.i64("2", 0) == 1 {
                add("bias", vec![num_output], 1)?;
            }
        }
        "Convolution"
        | "ConvolutionDepthWise"
        | "Deconvolution"
        | "DeconvolutionDepthWise"
        | "DeformableConv2D" => {
            let num_output = params.i64("0", 0);
            let kernel_w = params.i64("1", 0);
            let kernel_h = params.i64("11", kernel_w);
            let weight_data_size = params.i64("6", 0);
            let denom = num_output * kernel_w * kernel_h;
            let num_input = if denom > 0 {
                weight_data_size / denom
            } else {
                0
            };
            add("weight", vec![num_output, num_input, kernel_h, kernel_w], 0)?;
            if params.i64("5", 0) == 1 {
                add("bias", vec![num_output], 1)?;
            }
            let int8_scale_term = params.i64("8", 0);
            if op == "Convolution" && int8_scale_term != 0 {
                add("weight_scales", vec![num_output], 1)?;
                add("bottom_scales", vec![1], 1)?;
                if int8_scale_term > 100 {
                    add("top_scales", vec![1], 1)?;
                }
            } else if op == "ConvolutionDepthWise" {
                let group = params.i64("7", 1);
                if int8_scale_term == 1 || int8_scale_term == 101 {
                    add("weight_scales", vec![group], 1)?;
                    add("bottom_scales", vec![1], 1)?;
                } else if int8_scale_term == 2 || int8_scale_term == 102 {
                    add("weight_scales", vec![1], 1)?;
                    add("bottom_scales", vec![1], 1)?;
                }
                if int8_scale_term > 100 {
                    add("top_scales", vec![1], 1)?;
                }
            }
            params.remove("6");
        }
        "Convolution1D" | "ConvolutionDepthWise1D" => {
            let dynamic_weight = params.i64("19", 0);
            if dynamic_weight == 0 {
                let num_output = params.i64("0", 0);
                let kernel_w = params.i64("1", 0);
                let weight_data_size = params.i64("6", 0);
                let denom = num_output * kernel_w;
                let num_input = if denom > 0 {
                    weight_data_size / denom
                } else {
                    0
                };
                add("weight", vec![num_output, num_input, kernel_w], 0)?;
                if params.i64("5", 0) == 1 {
                    add("bias", vec![num_output], 1)?;
                }
                params.remove("6");
            }
            params.remove("19");
        }
        "Deconvolution1D" | "DeconvolutionDepthWise1D" => {
            let dynamic_weight = params.i64("28", 0);
            if dynamic_weight == 0 {
                let num_output = params.i64("0", 0);
                let kernel_w = params.i64("1", 0);
                let weight_data_size = params.i64("6", 0);
                let denom = num_output * kernel_w;
                let num_input = if denom > 0 {
                    weight_data_size / denom
                } else {
                    0
                };
                add("weight", vec![num_output, num_input, kernel_w], 0)?;
                if params.i64("5", 0) == 1 {
                    add("bias", vec![num_output], 1)?;
                }
                params.remove("6");
            }
            params.remove("28");
        }
        "Convolution3D"
        | "ConvolutionDepthWise3D"
        | "Deconvolution3D"
        | "DeconvolutionDepthWise3D" => {
            let num_output = params.i64("0", 0);
            let kernel_w = params.i64("1", 0);
            let kernel_h = params.i64("11", kernel_w);
            let kernel_d = params.i64("21", kernel_w);
            let weight_data_size = params.i64("6", 0);
            let denom = num_output * kernel_w * kernel_h * kernel_d;
            let num_input = if denom > 0 {
                weight_data_size / denom
            } else {
                0
            };
            add(
                "weight",
                vec![num_output, num_input, kernel_d, kernel_h, kernel_w],
                0,
            )?;
            if params.i64("5", 0) == 1 {
                add("bias", vec![num_output], 1)?;
            }
            params.remove("6");
        }
        "Quantize" => add("scale", vec![params.i64("0", 1)], 1)?,
        "Dequantize" => {
            add("scale", vec![params.i64("0", 1)], 1)?;
            add("bias", vec![params.i64("1", 0)], 1)?;
        }
        "Requantize" => {
            add("scale_in", vec![params.i64("0", 1)], 1)?;
            add("scale_out", vec![params.i64("1", 1)], 1)?;
            add("bias", vec![params.i64("2", 0)], 1)?;
        }
        "InstanceNorm" => {
            if params.i64("2", 1) == 1 {
                let channels = params.i64("0", 0);
                add("gamma", vec![channels], 1)?;
                add("beta", vec![channels], 1)?;
            }
        }
        "Scale" => {
            let size = params.i64("0", 0);
            if size != -233 {
                add("scale", vec![size], 1)?;
                if params.get("1").and_then(ParamValue::as_str) == Some("1") {
                    add("bias", vec![size], 1)?;
                }
            }
        }
        "Normalize" => add("scale", vec![params.i64("3", 0)], 1)?,
        "PReLU" => add("slope", vec![params.i64("0", 0)], 1)?,
        "Padding" => add("per_channel_pad_data", vec![params.i64("6", 0)], 1)?,
        "MemoryData" => {
            let w = params.i64("0", 0);
            let h = params.i64("1", 0);
            let d = params.i64("11", 0);
            let c = params.i64("2", 0);
            let code = params.i64("21", 1);
            let shape = if d != 0 {
                vec![w, h, d, c]
            } else if c != 0 {
                vec![w, h, c]
            } else if h != 0 {
                vec![w, h]
            } else if w != 0 {
                vec![w]
            } else {
                vec![1]
            };
            add("data", shape, code)?;
        }
        "GroupNorm" => {
            if params.i64("3", 1) == 1 {
                let channels = params.i64("1", 0);
                add("gamma", vec![channels], 1)?;
                add("beta", vec![channels], 1)?;
            }
        }
        "LayerNorm" => {
            let channels = params.i64("0", 0);
            add("gamma", vec![channels], 1)?;
            add("beta", vec![channels], 1)?;
        }
        "RNN" => {
            let num_output = params.i64("0", 0);
            let weight_data_size = params.i64("1", 0);
            let direction = params.i64("2", 0);
            let int8_scale_term = params.i64("8", 0);
            let directions = if direction == 2 { 2 } else { 1 };
            let num_input = if directions * num_output > 0 {
                weight_data_size / directions / num_output
            } else {
                0
            };
            add("weight_xc", vec![directions, num_output, num_input], 0)?;
            add("bias_c", vec![directions, num_output], 0)?;
            add("weight_hc", vec![directions, num_output, num_output], 0)?;
            if int8_scale_term != 0 {
                add("weight_xc_scales", vec![directions, num_output], 1)?;
                add("weight_hc_scales", vec![directions, num_output], 1)?;
            }
            params.remove("1");
        }
        "LSTM" => {
            let num_output = params.i64("0", 0);
            let weight_data_size = params.i64("1", 0);
            let direction = params.i64("2", 0);
            let hidden_size = params.i64("3", num_output);
            let int8_scale_term = params.i64("8", 0);
            let directions = if direction == 2 { 2 } else { 1 };
            let num_input = if directions * hidden_size > 0 {
                weight_data_size / directions / hidden_size / 4
            } else {
                0
            };
            add("weight_xc", vec![directions, 4, hidden_size, num_input], 0)?;
            add("bias_c", vec![directions, 4, hidden_size], 0)?;
            add("weight_hc", vec![directions, 4, hidden_size, num_output], 0)?;
            if num_output != hidden_size {
                add("weight_hr", vec![directions, num_output, hidden_size], 0)?;
            }
            if int8_scale_term != 0 {
                add("weight_xc_scales", vec![directions, hidden_size * 4], 1)?;
                add("weight_hc_scales", vec![directions, hidden_size * 4], 1)?;
            }
            params.remove("1");
        }
        "GRU" => {
            let num_output = params.i64("0", 0);
            let weight_data_size = params.i64("1", 0);
            let direction = params.i64("2", 0);
            let int8_scale_term = params.i64("8", 0);
            let directions = if direction == 2 { 2 } else { 1 };
            let num_input = if directions * num_output > 0 {
                weight_data_size / directions / num_output / 3
            } else {
                0
            };
            add("weight_xc", vec![directions, 3, num_output, num_input], 0)?;
            add("bias_c", vec![directions, 4, num_output], 0)?;
            add("weight_hc", vec![directions, 3, num_output, num_output], 0)?;
            if int8_scale_term != 0 {
                add("weight_xc_scales", vec![directions, num_output * 3], 1)?;
                add("weight_hc_scales", vec![directions, num_output * 3], 1)?;
            }
            params.remove("1");
        }
        "MultiHeadAttention" => {
            let embed_dim = params.i64("0", 0);
            let weight_data_size = params.i64("2", 0);
            let kdim = params.i64("3", embed_dim);
            let vdim = params.i64("4", embed_dim);
            let int8_scale_term = params.i64("18", 0);
            let qdim = if embed_dim > 0 {
                weight_data_size / embed_dim
            } else {
                0
            };
            add("weight_q", vec![embed_dim, qdim], 0)?;
            add("bias_q", vec![embed_dim], 1)?;
            add("weight_k", vec![embed_dim, kdim], 0)?;
            add("bias_k", vec![embed_dim], 1)?;
            add("weight_v", vec![embed_dim, vdim], 0)?;
            add("bias_v", vec![embed_dim], 1)?;
            add("weight_out", vec![qdim, embed_dim], 0)?;
            add("bias_out", vec![qdim], 1)?;
            if int8_scale_term != 0 {
                add("q_weight_scales", vec![embed_dim], 1)?;
                add("k_weight_scales", vec![embed_dim], 1)?;
                add("v_weight_scales", vec![embed_dim], 1)?;
                add("out_weight_scale", vec![1], 1)?;
            }
            params.remove("2");
        }
        "Gemm" => {
            let trans_a = params.i64("2", 0);
            let trans_b = params.i64("3", 0);
            let constant_a = params.i64("4", 0);
            let constant_b = params.i64("5", 0);
            let constant_c = params.i64("6", 0);
            let m = params.i64("7", 0);
            let n = params.i64("8", 0);
            let k = params.i64("9", 0);
            let c_broadcast = params.i64("10", 0);
            if constant_a == 1 {
                add("A", if trans_a == 0 { vec![k, m] } else { vec![m, k] }, 0)?;
            }
            if constant_b == 1 {
                add("B", if trans_b == 1 { vec![n, k] } else { vec![k, n] }, 0)?;
            }
            if constant_c == 1 && c_broadcast != -1 {
                let shape = match c_broadcast {
                    0 => Some(vec![1]),
                    1 => Some(vec![m]),
                    2 => Some(vec![1, m]),
                    3 => Some(vec![n, m]),
                    4 => Some(vec![n, 1]),
                    _ => None,
                };
                if let Some(shape) = shape {
                    add("C", shape, 0)?;
                }
            }
        }
        "RMSNorm" => {
            if params.i64("2", 1) == 1 {
                add("gamma", vec![params.i64("0", 0)], 1)?;
            }
        }
        _ => {}
    }

    for (key, signature) in params
        .iter()
        .filter(|(key, _)| key.starts_with('@'))
        .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_owned())))
        .collect::<Vec<_>>()
    {
        if let Some(type_info) = signature_type(model, &signature) {
            let value_id = add_weight_with_type(model, graph, values, type_info);
            node.inputs.push(Some(value_id));
            params.remove(&key);
        }
    }

    Ok(())
}

fn add_weight(
    model: &mut Model,
    graph: &mut Graph,
    blobs: &mut BlobReader,
    values: &mut NcnnValues,
    shape: Vec<i64>,
    code: i64,
) -> Result<ValueId, ModelError> {
    let element_type = blobs.load(&shape, code)?;
    let type_info = TypeInfo {
        element_type: Some(element_type.clone()),
        layout: None,
        denotation: None,
        shape: shape.iter().copied().map(Dimension::known).collect(),
    };
    let tensor = Tensor::metadata_only(
        None,
        element_type,
        type_info.shape.clone(),
        TensorStorage::Absent,
    );
    let tensor_id = model.add_tensor(tensor);
    Ok(values.value(model, graph, "", Some(type_info), Some(tensor_id)))
}

fn add_weight_with_type(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut NcnnValues,
    type_info: TypeInfo,
) -> ValueId {
    let tensor = Tensor::metadata_only(
        None,
        type_info
            .element_type
            .clone()
            .unwrap_or(TensorElementType::Unknown),
        type_info.shape.clone(),
        TensorStorage::Absent,
    );
    let tensor_id = model.add_tensor(tensor);
    values.value(model, graph, "", Some(type_info), Some(tensor_id))
}

fn lower_attribute(
    model: &mut Model,
    metadata: &NcnnMetadata,
    op: &str,
    key: &str,
    value: &ParamValue,
    decode_enums: bool,
) -> Attribute {
    let attribute = metadata.attribute(op, key);
    let name = attribute
        .map(|attribute| attribute.name.as_str())
        .unwrap_or(key);
    let value = match (
        attribute.and_then(|attribute| attribute.attribute_type.as_deref()),
        value,
    ) {
        (Some("int32"), ParamValue::Scalar(value)) => value
            .parse::<i64>()
            .map(AttributeValue::Int)
            .unwrap_or_else(|_| AttributeValue::String(model.intern(value))),
        (Some("float32"), ParamValue::Scalar(value)) => value
            .parse::<f32>()
            .map(AttributeValue::Float)
            .unwrap_or_else(|_| AttributeValue::String(model.intern(value))),
        (Some("float32[]"), ParamValue::List(values)) => AttributeValue::Floats(
            values
                .iter()
                .filter_map(|value| value.parse::<f32>().ok())
                .collect(),
        ),
        (Some(kind), ParamValue::Scalar(value)) if decode_enums => enum_value(kind, value)
            .map(|value| AttributeValue::String(model.intern(value)))
            .unwrap_or_else(|| scalar_attribute(model, value)),
        (Some(kind), ParamValue::Scalar(value)) if is_enum_kind(kind) => {
            AttributeValue::String(model.intern(value))
        }
        (Some(_), ParamValue::Scalar(value)) => scalar_attribute(model, value),
        (_, ParamValue::Scalar(value)) => scalar_attribute(model, value),
        (_, ParamValue::List(values)) if !decode_enums => AttributeValue::Ints(
            values
                .iter()
                .filter_map(|value| value.parse::<i64>().ok())
                .collect(),
        ),
        (_, ParamValue::List(values)) => {
            AttributeValue::Strings(values.iter().map(|value| model.intern(value)).collect())
        }
    };
    Attribute {
        name: model.intern(name),
        value,
    }
}

fn scalar_attribute(model: &mut Model, value: &str) -> AttributeValue {
    if value == "True" {
        return AttributeValue::Bool(true);
    }
    if value == "False" {
        return AttributeValue::Bool(false);
    }
    if let Ok(value) = value.parse::<i64>() {
        return AttributeValue::Int(value);
    }
    if let Ok(number) = value.parse::<f64>()
        && number.is_finite()
        && number.fract() == 0.0
        && number >= i64::MIN as f64
        && number <= i64::MAX as f64
    {
        return AttributeValue::Int(number as i64);
    }
    if value.len() > 3 && value.starts_with('(') && value.ends_with(')') {
        let values = value[1..value.len() - 1]
            .split(',')
            .map(|item| item.trim().parse::<i64>())
            .collect::<Result<Vec<_>, _>>();
        if let Ok(values) = values {
            return AttributeValue::Ints(values);
        }
    }
    AttributeValue::String(model.intern(value))
}

fn enum_value<'a>(kind: &str, value: &'a str) -> Option<&'a str> {
    let index = value.parse::<usize>().ok()?;
    let values: &[&str] = match kind {
        "BinaryOpType" => &[
            "Add", "Sub", "Mul", "Div", "Max", "Min", "Pow", "RSub", "RDiv",
        ],
        "CastOpType" => &["Auto", "Float32", "Float16", "Int8", "BFloat16"],
        "EltwiseType" => &["Prod", "Sum", "Max"],
        "PaddingType" => &["Constant", "Replicate", "Reflect"],
        "PoolingType" => &["Max", "Average"],
        "InterpResizeType" => &["", "Nearest", "Bilinear", "Bicubic"],
        "PermuteOrderType" => &[
            "WH WHC WHDC",
            "HW HWC HWDC",
            "WCH WDHC",
            "CWH DWHC",
            "HCW HDWC",
            "CHW DHWC",
            "WHCD",
            "HWCD",
            "WCHD",
            "CWHD",
            "HCWD",
            "CHWD",
            "WDCH",
            "DWCH",
            "WCDH",
            "CWDH",
            "DCWH",
            "CDWH",
            "HDCW",
            "DHCW",
            "HCDW",
            "CHDW",
            "DCHW",
            "CDHW",
        ],
        "ReductionOpType" => &[
            "Sum",
            "ASum",
            "SumSq",
            "Mean",
            "Max",
            "Min",
            "Prod",
            "L1",
            "L2",
            "LogSum",
            "LogSumExp",
        ],
        "UnaryOpType" => &[
            "Abs",
            "Neg",
            "Floor",
            "Ceil",
            "Square",
            "Sqrt",
            "Rsq",
            "Exp",
            "Log",
            "Sin",
            "Cos",
            "Tan",
            "ASin",
            "ACos",
            "ATan",
            "Reciprocal",
            "Tanh",
        ],
        _ => return None,
    };
    values.get(index).copied()
}

fn is_enum_kind(kind: &str) -> bool {
    matches!(
        kind,
        "BinaryOpType"
            | "CastOpType"
            | "EltwiseType"
            | "PaddingType"
            | "PoolingType"
            | "InterpResizeType"
            | "PermuteOrderType"
            | "ReductionOpType"
            | "UnaryOpType"
    )
}

fn route_type(model: &mut Model, params: &mut ParamMap, id: &str) -> Option<TypeInfo> {
    let key = format!("#{id}");
    params.remove(&key).and_then(|value| {
        value
            .as_str()
            .and_then(|signature| signature_type(model, signature))
    })
}

fn signature_type(_model: &mut Model, signature: &str) -> Option<TypeInfo> {
    let open = signature.find('(')?;
    let close = signature[open + 1..].find(')')? + open + 1;
    let shape = signature[open + 1..close]
        .split(',')
        .filter(|value| !value.is_empty())
        .filter_map(|value| value.parse::<i64>().ok())
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let data_type = &signature[close + 1..];
    let element_type = match data_type {
        "f32" => TensorElementType::Float32,
        "f16" => TensorElementType::Float16,
        value => TensorElementType::Other(value.to_owned()),
    };
    Some(TypeInfo {
        element_type: Some(element_type),
        layout: None,
        denotation: None,
        shape,
    })
}

fn shape_param_type(items: &[String]) -> Option<TypeInfo> {
    let mut values = items
        .iter()
        .filter_map(|item| item.parse::<i64>().ok())
        .collect::<Vec<_>>();
    let rank = usize::try_from(*values.first()?).ok()?;
    if rank > values.len().saturating_sub(1) {
        return None;
    }
    values.remove(0);
    Some(TypeInfo {
        element_type: Some(TensorElementType::Float32),
        layout: None,
        denotation: None,
        shape: values
            .into_iter()
            .take(rank)
            .map(Dimension::known)
            .collect(),
    })
}

fn dimension_from_param(model: &mut Model, value: &ParamValue) -> Dimension {
    let Some(value) = value.as_str() else {
        return Dimension::unknown();
    };
    value
        .parse::<i64>()
        .map(Dimension::known)
        .unwrap_or_else(|_| Dimension::symbolic(model.intern(value)))
}

#[derive(Default)]
struct NcnnValues {
    ids: HashMap<String, ValueId>,
    types: HashMap<String, TypeInfo>,
}

impl NcnnValues {
    fn set_type(&mut self, name: &str, type_info: TypeInfo) {
        self.types.insert(name.to_owned(), type_info);
    }

    fn value(
        &mut self,
        model: &mut Model,
        graph: &mut Graph,
        name: &str,
        type_info: Option<TypeInfo>,
        initializer: Option<netron_rs_core::TensorId>,
    ) -> ValueId {
        if name.is_empty() && initializer.is_some() {
            return self.add_value(model, graph, name, type_info, initializer);
        }
        if let Some(value_id) = self.ids.get(name).copied() {
            if type_info.is_some() && graph.values[value_id.index()].type_info.is_none() {
                graph.values[value_id.index()].type_info = type_info;
            }
            return value_id;
        }
        let type_info = type_info.or_else(|| self.types.get(name).cloned());
        let value_id = self.add_value(model, graph, name, type_info, initializer);
        self.ids.insert(name.to_owned(), value_id);
        value_id
    }

    fn add_value(
        &mut self,
        model: &mut Model,
        graph: &mut Graph,
        name: &str,
        type_info: Option<TypeInfo>,
        initializer: Option<netron_rs_core::TensorId>,
    ) -> ValueId {
        let mut value = Value::new(model.intern(name));
        value.type_info = type_info;
        value.initializer = initializer;
        graph.add_value(value)
    }
}

#[derive(Debug, Clone)]
struct NcnnLayer {
    op_type: String,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    params: ParamMap,
}

#[derive(Debug, Clone, Default)]
struct ParamMap {
    entries: Vec<(String, ParamValue)>,
}

impl ParamMap {
    fn insert(&mut self, key: String, value: ParamValue) {
        self.entries.push((key, value));
    }

    fn get(&self, key: &str) -> Option<&ParamValue> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value)
    }

    fn remove(&mut self, key: &str) -> Option<ParamValue> {
        let index = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == key)?;
        Some(self.entries.remove(index).1)
    }

    fn i64(&self, key: &str, default: i64) -> i64 {
        self.get(key)
            .and_then(ParamValue::as_str)
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(default)
    }

    fn iter(&self) -> impl Iterator<Item = (&String, &ParamValue)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }

    fn values(&self) -> Vec<&ParamValue> {
        self.entries.iter().map(|(_, value)| value).collect()
    }
}

#[derive(Debug, Clone)]
enum ParamValue {
    Scalar(String),
    List(Vec<String>),
}

impl ParamValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::Scalar(value) => Some(value),
            ParamValue::List(_) => None,
        }
    }
}

fn parse_text_param(text: &str) -> Result<Vec<NcnnLayer>, ModelError> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let Some(signature) = lines.first().copied() else {
        return Err(invalid("Invalid header."));
    };
    let header_index = usize::from(signature == "7767517");
    let Some(header) = lines.get(header_index).copied() else {
        return Err(invalid("Invalid header."));
    };
    let header = header.split_whitespace().collect::<Vec<_>>();
    if header.len() != 2 || !header.iter().all(|value| value.parse::<u32>().is_ok()) {
        return Err(invalid("Invalid header."));
    }
    lines.drain(..=header_index);

    let mut layers = Vec::new();
    for line in lines {
        let mut columns = line.split_whitespace().collect::<Vec<_>>();
        if columns.len() < 4 {
            continue;
        }
        let op_type = columns.remove(0).to_owned();
        let name = columns.remove(0).to_owned();
        let input_count = columns
            .remove(0)
            .parse::<usize>()
            .map_err(|_| invalid(format!("Invalid input count in '{line}'.")))?;
        let output_count = columns
            .remove(0)
            .parse::<usize>()
            .map_err(|_| invalid(format!("Invalid output count in '{line}'.")))?;
        if columns.len() < input_count + output_count {
            return Err(invalid(format!("Invalid layer arity in '{line}'.")));
        }
        let outputs_start = input_count;
        let params_start = input_count + output_count;
        let inputs = columns[..outputs_start]
            .iter()
            .map(|value| (*value).to_owned())
            .collect();
        let outputs = columns[outputs_start..params_start]
            .iter()
            .map(|value| (*value).to_owned())
            .collect();
        let mut params = ParamMap::default();
        for (index, column) in columns[params_start..].iter().enumerate() {
            let (mut key, value) = column
                .split_once('=')
                .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
                .unwrap_or_else(|| (index.to_string(), (*column).to_owned()));
            if key.parse::<i64>().is_ok_and(|key| key < 0) {
                let mut values = value
                    .split(',')
                    .map(|value| value.trim().to_owned())
                    .collect::<Vec<_>>();
                if !values.is_empty() {
                    values.remove(0);
                }
                let key_int = key.parse::<i64>().unwrap_or_default();
                key = (-(key_int + 23300)).to_string();
                params.insert(key, ParamValue::List(values));
            } else {
                params.insert(key, ParamValue::Scalar(value));
            }
        }
        layers.push(NcnnLayer {
            op_type,
            name,
            inputs,
            outputs,
            params,
        });
    }
    Ok(layers)
}

fn parse_binary_param(data: &[u8]) -> Result<Vec<NcnnLayer>, ModelError> {
    let mut reader = BinaryReader::new(data);
    if reader.i32()? != 0x0076_85dd {
        return Err(invalid("Invalid signature."));
    }
    let layer_count =
        usize::try_from(reader.i32()?).map_err(|_| invalid("Invalid binary layer count."))?;
    let _blob_count = reader.i32()?;
    let mut layers = Vec::with_capacity(layer_count);
    for index in 0..layer_count {
        let op_type = reader.i32()?.to_string();
        let name = index.to_string();
        let input_count =
            usize::try_from(reader.i32()?).map_err(|_| invalid("Invalid input count."))?;
        let output_count =
            usize::try_from(reader.i32()?).map_err(|_| invalid("Invalid output count."))?;
        let mut inputs = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            inputs.push(reader.i32()?.to_string());
        }
        let mut outputs = Vec::with_capacity(output_count);
        for _ in 0..output_count {
            outputs.push(reader.i32()?.to_string());
        }
        let mut params = ParamMap::default();
        let mut id = reader.i32()?;
        while id != -233 {
            let is_array = id <= -23300;
            if is_array {
                id = -id - 23300;
            }
            let key = id.to_string();
            if is_array {
                let length =
                    usize::try_from(reader.i32()?).map_err(|_| invalid("Invalid array length."))?;
                let mut values = Vec::with_capacity(length);
                for _ in 0..length {
                    values.push(reader.i32()?.to_string());
                }
                params.insert(key, ParamValue::List(values));
            } else {
                params.insert(key, ParamValue::Scalar(reader.i32()?.to_string()));
            }
            id = reader.i32()?;
        }
        layers.push(NcnnLayer {
            op_type,
            name,
            inputs,
            outputs,
            params,
        });
    }
    Ok(layers)
}

fn parse_header(text: &str) -> Option<()> {
    let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
    let first = lines.next()?;
    let header = if first == "7767517" {
        lines.next()?
    } else {
        first
    };
    let parts = header.split_whitespace().collect::<Vec<_>>();
    (parts.len() == 2 && parts.iter().all(|value| value.parse::<u32>().is_ok())).then_some(())
}

fn is_pnnx_text(text: &str) -> bool {
    parse_header(text).is_some()
        && text.lines().skip(1).take(32).any(|line| {
            let line = line.trim_start();
            line.starts_with("pnnx.")
                || line.starts_with("nn.")
                || line.starts_with("F.")
                || line.starts_with("torch.")
                || line.starts_with("Tensor.")
        })
}

struct BlobReader {
    data: Option<Vec<u8>>,
    position: usize,
}

impl BlobReader {
    fn from_data(data: Vec<u8>) -> Self {
        Self {
            data: Some(data),
            position: 0,
        }
    }

    fn open(path: Option<&Path>, format: &'static str) -> Self {
        let data = path
            .and_then(|path| sidecar_path(path, format))
            .and_then(|path| std::fs::read(path).ok());
        Self { data, position: 0 }
    }

    fn load(&mut self, shape: &[i64], code: i64) -> Result<TensorElementType, ModelError> {
        if self.data.is_none() {
            return Ok(TensorElementType::Other(code.to_string()));
        }
        let size = shape.iter().try_fold(1usize, |product, dimension| {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| product.checked_mul(dimension))
        });
        let Some(size) = size else {
            return Ok(TensorElementType::Other(code.to_string()));
        };
        if code == 1 {
            self.position = self.position.saturating_add(size.saturating_mul(4));
            return Ok(TensorElementType::Float32);
        }
        if code != 0 {
            return Ok(TensorElementType::Other(code.to_string()));
        }
        let flag = self.u32()?;
        match flag {
            0x0130_6b47 => {
                self.position = self.position.saturating_add(size.saturating_mul(2));
                self.align(4);
                Ok(TensorElementType::Float16)
            }
            0x000d_4b38 => {
                self.position = self.position.saturating_add(size);
                self.align(4);
                Ok(TensorElementType::Int8)
            }
            0x0002_c056 | 0x0000_0000 => {
                self.position = self.position.saturating_add(size.saturating_mul(4));
                Ok(TensorElementType::Float32)
            }
            _ => {
                self.position = self.position.saturating_add(1024 + size);
                self.align(4);
                Ok(TensorElementType::Uint8)
            }
        }
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let end = self.position.saturating_add(4);
        let Some(bytes) = self
            .data
            .as_ref()
            .and_then(|data| data.get(self.position..end))
        else {
            return Err(invalid("Unexpected end of ncnn weights."));
        };
        self.position = end;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn align(&mut self, size: usize) {
        let remainder = self.position % size;
        if remainder != 0 {
            self.position += size - remainder;
        }
    }
}

fn sidecar_path(path: &Path, format: &'static str) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    if name.ends_with(".param.bin") {
        return replace_path_suffix(path, ".param.bin", ".bin");
    }
    if name.ends_with(".param") {
        let extension = if format == PNNX_FORMAT && !name.ends_with(".pnnx.param") {
            "pnnx.bin"
        } else {
            "bin"
        };
        return Some(path.with_extension(extension));
    }
    if name.ends_with(".cfg.ncnn") {
        return replace_path_suffix(path, ".cfg.ncnn", ".weights.ncnn");
    }
    None
}

fn sidecar_definition(path: &Path) -> Option<PathBuf> {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.ends_with(".weights.ncnn") {
        return replace_path_suffix(path, ".weights.ncnn", ".cfg.ncnn");
    }
    if lower.ends_with(".bin") && !lower.ends_with(".param.bin") {
        let param = replace_path_suffix(path, ".bin", ".param")?;
        if param.exists() {
            return Some(param);
        }
        return replace_path_suffix(path, ".bin", ".param.bin");
    }
    None
}

fn replace_path_suffix(path: &Path, suffix: &str, replacement: &str) -> Option<PathBuf> {
    let text = path.to_string_lossy();
    let prefix_len = text.len().checked_sub(suffix.len())?;
    Some(PathBuf::from(format!(
        "{}{}",
        &text[..prefix_len],
        replacement
    )))
}

struct NcnnDefinition {
    format: &'static str,
    layers: Vec<NcnnLayer>,
    decode_enums: bool,
}

fn read_definition(path: &Path) -> Result<NcnnDefinition, ModelError> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let data = std::fs::read(path)
        .map_err(|error| invalid(format!("Required ncnn model definition not found: {error}")))?;
    if name.ends_with(".param.bin") {
        return Ok(NcnnDefinition {
            format: FORMAT,
            layers: parse_binary_param(&data)?,
            decode_enums: false,
        });
    }
    let text = std::str::from_utf8(&data)
        .map_err(|error| invalid(format!("ncnn param is not UTF-8: {error}")))?;
    let format = if is_pnnx_text(text) {
        PNNX_FORMAT
    } else {
        FORMAT
    };
    Ok(NcnnDefinition {
        format,
        layers: parse_text_param(text)?,
        decode_enums: true,
    })
}

fn metadata() -> &'static NcnnMetadata {
    static METADATA_CACHE: OnceLock<NcnnMetadata> = OnceLock::new();
    METADATA_CACHE.get_or_init(|| NcnnMetadata::new(METADATA))
}

#[derive(Debug)]
struct NcnnMetadata {
    names_by_identifier: HashMap<String, String>,
    attributes: HashMap<String, Vec<MetadataAttribute>>,
}

impl NcnnMetadata {
    fn new(data: &str) -> Self {
        let entries = serde_json::from_str::<Vec<MetadataType>>(data).unwrap_or_default();
        let mut names_by_identifier = HashMap::new();
        let mut attributes = HashMap::new();
        for entry in entries {
            if let Some(identifier) = entry.identifier {
                names_by_identifier.insert(identifier.to_string(), entry.name.clone());
            }
            attributes.insert(entry.name.clone(), entry.attributes);
        }
        Self {
            names_by_identifier,
            attributes,
        }
    }

    fn type_name(&self, name: &str) -> String {
        self.names_by_identifier
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    }

    fn attribute(&self, op: &str, key: &str) -> Option<&MetadataAttribute> {
        let index = key.parse::<usize>().ok()?;
        self.attributes.get(op)?.get(index)
    }
}

#[derive(Debug, Deserialize)]
struct MetadataType {
    name: String,
    #[serde(default)]
    identifier: Option<i64>,
    #[serde(default)]
    attributes: Vec<MetadataAttribute>,
}

#[derive(Debug, Deserialize)]
struct MetadataAttribute {
    name: String,
    #[serde(default, rename = "type")]
    attribute_type: Option<String>,
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}

struct BinaryReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn i32(&mut self) -> Result<i32, ModelError> {
        let end = self.position.saturating_add(4);
        let Some(bytes) = self.data.get(self.position..end) else {
            return Err(invalid("Unexpected end of binary ncnn param."));
        };
        self.position = end;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }
}
