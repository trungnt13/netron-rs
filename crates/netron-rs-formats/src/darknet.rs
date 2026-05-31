use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
    TypeInfo, Value, ValueId,
};
use serde::Deserialize;

const FORMAT: &str = "Darknet";
const METADATA: &str = include_str!("metadata/darknet-metadata.json");

pub struct DarknetFormat;

impl ModelFormat for DarknetFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["cfg", "model", "txt"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if input
            .path
            .is_some_and(|path| darknet_weights_sidecar(path).is_some_and(|path| path.exists()))
        {
            return Confidence::High;
        }
        let Ok(text) = std::str::from_utf8(
            input
                .data
                .get(..input.data.len().min(65536))
                .unwrap_or(input.data),
        ) else {
            return Confidence::None;
        };
        for line in text.lines() {
            let line = strip_ws(line);
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            return if line.starts_with('[') && line.ends_with(']') {
                Confidence::High
            } else {
                Confidence::None
            };
        }
        Confidence::None
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        if let Some(definition) = input.path.and_then(darknet_weights_sidecar) {
            let text = std::fs::read_to_string(&definition).map_err(|error| {
                invalid(format!(
                    "Required Darknet model definition not found: {error}"
                ))
            })?;
            return lower_model(parse_cfg(&text)?);
        }
        let text = std::str::from_utf8(input.data)
            .map_err(|error| invalid(format!("Darknet cfg is not UTF-8: {error}")))?;
        lower_model(parse_cfg(text)?)
    }
}

fn darknet_weights_sidecar(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    if !name.ends_with(".weights") {
        return None;
    }
    let mut sidecar = path.to_path_buf();
    sidecar.set_extension("cfg");
    Some(sidecar)
}

fn lower_model(mut sections: Vec<Section>) -> Result<Model, ModelError> {
    if sections.is_empty() {
        return Err(invalid("Config file has no sections."));
    }
    let net = sections.remove(0);
    if net.kind != "net" && net.kind != "network" {
        return Err(invalid(format!(
            "Unexpected '[{}]' section. First section must be [net] or [network].",
            net.kind
        )));
    }

    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    let metadata = metadata();
    let globals = net.options.clone();

    let mut params = NetworkParams {
        h: option_i64(&net.options, &globals, "height", 0)?,
        w: option_i64(&net.options, &globals, "width", 0)?,
        c: option_i64(&net.options, &globals, "channels", 0)?,
        inputs: 0,
    };
    params.inputs = option_i64(
        &net.options,
        &globals,
        "inputs",
        params.h * params.w * params.c,
    )?;

    let input_shape = if params.w != 0 && params.h != 0 && params.c != 0 {
        vec![params.w, params.h, params.c]
    } else {
        vec![params.inputs]
    };
    let input_id = add_value(
        &mut model,
        &mut graph,
        "input",
        Some(float_type(&input_shape)?),
        None,
    );
    graph.values[input_id.index()].is_graph_input = true;
    graph.inputs.push(input_id);

    let mut current = vec![input_id];
    let mut states = Vec::with_capacity(sections.len());
    let mut infer = true;
    for (index, mut section) in sections.into_iter().enumerate() {
        section.name = index.to_string();
        let inference_options = section.options.clone();
        let mut layer_inputs = current.clone();
        match section.kind.as_str() {
            "shortcut" => {
                let routes = section
                    .options
                    .get("from")
                    .map(|value| parse_routes(value))
                    .unwrap_or_default();
                let mut remove = true;
                for route in routes {
                    if let Some(state) = resolve_state(&states, index, route) {
                        layer_inputs.extend(state.outputs.iter().copied());
                    } else {
                        remove = false;
                    }
                }
                if remove {
                    section.options.remove("from");
                }
            }
            "sam" | "scale_channels" => {
                let from = option_i64(&section.options, &globals, "from", 0)?;
                if let Some(state) = resolve_state(&states, index, from) {
                    layer_inputs.extend(state.outputs.iter().copied());
                    section.options.remove("from");
                }
            }
            "route" => {
                layer_inputs.clear();
                let routes = section
                    .options
                    .get("layers")
                    .map(|value| parse_routes(value))
                    .unwrap_or_default();
                let mut remove = true;
                for route in routes {
                    if let Some(state) = resolve_state(&states, index, route) {
                        layer_inputs.extend(state.outputs.iter().copied());
                    } else {
                        remove = false;
                    }
                }
                if remove {
                    section.options.remove("layers");
                }
            }
            _ => {}
        }

        let build = if infer {
            match infer_layer(
                &section.kind,
                &inference_options,
                &globals,
                &params,
                &states,
                index,
            ) {
                Some(Ok(build)) => build,
                Some(Err(error)) => return Err(error),
                None => {
                    infer = false;
                    LayerBuild::unknown()
                }
            }
        } else {
            LayerBuild::unknown()
        };
        if build.breaks_inference {
            infer = false;
        }

        let mut node = Node::new(
            graph_id,
            Operator {
                domain: None,
                name: model.intern(&section.kind),
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        node.name = Some(model.intern(&section.name));
        node.inputs = layer_inputs.iter().copied().map(Some).collect();
        for weight in &build.weights {
            let weight_id = add_weight(&mut model, &mut graph, weight)?;
            node.inputs.push(Some(weight_id));
        }
        let output_id = add_value(
            &mut model,
            &mut graph,
            &section.name,
            build.output_type.clone(),
            None,
        );
        node.outputs.push(Some(output_id));
        node.attributes = section
            .options
            .iter()
            .map(|(name, value)| lower_attribute(&mut model, metadata, &section.kind, name, value))
            .collect();

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
        graph.values[output_id.index()].producer = Some(node_id);

        let state = LayerState {
            outputs: vec![output_id],
            out_w: build.out_w,
            out_h: build.out_h,
            out_c: build.out_c,
            out: build.out,
        };
        if infer && !build.breaks_inference {
            params.w = build.out_w;
            params.h = build.out_h;
            params.c = build.out_c;
            params.inputs = build.out;
        }
        current = state.outputs.clone();
        states.push(state);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn infer_layer(
    kind: &str,
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
    states: &[LayerState],
    index: usize,
) -> Option<Result<LayerBuild, ModelError>> {
    match kind {
        "conv" | "convolutional" => Some(convolutional(options, globals, params, "")),
        "deconvolutional" => Some(deconvolutional(options, globals, params)),
        "connected" => Some(connected(options, globals, params.inputs, "")),
        "local" => Some(local(options, globals, params)),
        "batchnorm" => Some(batchnorm(params)),
        "activation" => Some(same_image(params, params.inputs)),
        "max" | "maxpool" => Some(maxpool(options, globals, params)),
        "avgpool" => Some(Ok(LayerBuild::typed(
            1,
            1,
            params.c,
            params.c,
            vec![1, 1, params.c],
        ))),
        "crnn" => Some(crnn(options, globals, params)),
        "rnn" => Some(rnn(options, globals, params)),
        "gru" => Some(gru(options, globals, params)),
        "lstm" => Some(lstm(options, globals, params)),
        "conv_lstm" => Some(conv_lstm(options, globals, params)),
        "softmax" | "cost" | "detection" => Some(Ok(LayerBuild::typed(
            params.w,
            params.h,
            params.c,
            params.inputs,
            vec![params.inputs],
        ))),
        "dropout" => Some(same_image(params, params.inputs)),
        "upsample" => Some(upsample(options, globals, params)),
        "crop" => Some(crop(options, globals, params)),
        "yolo" => Some(yolo(options, globals, params, false)),
        "Gaussian_yolo" => Some(yolo(options, globals, params, true)),
        "region" => Some(region(options, globals, params)),
        "reorg" => Some(reorg(options, globals, params)),
        "route" => Some(route(options, globals, states, index)),
        "sam" | "scale_channels" => Some(from_shape(options, globals, states, index)),
        "shortcut" => Some(same_image(params, params.inputs)),
        _ => None,
    }
}

fn convolutional(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
    prefix: &str,
) -> Result<LayerBuild, ModelError> {
    let size = option_i64(options, globals, "size", 1)?;
    let filters = option_i64(options, globals, "filters", 1)
        .or_else(|_| option_i64(options, globals, "output", 1))?;
    let pad = option_i64(options, globals, "pad", 0)?;
    let padding = if pad != 0 {
        size >> 1
    } else {
        option_i64(options, globals, "padding", 0)?
    };
    let stride = option_i64(options, globals, "stride", 1)?;
    let mut stride_x = option_i64(options, globals, "stride_x", -1)?;
    let mut stride_y = option_i64(options, globals, "stride_y", -1)?;
    if stride_x < 1 {
        stride_x = stride;
    }
    if stride_y < 1 {
        stride_y = stride;
    }
    let groups = option_i64(options, globals, "groups", 1)?;
    let batch_normalize = option_i64(options, globals, "batch_normalize", 0)?;
    let out_w = ((params.w + (2 * padding) - size) / stride_x) + 1;
    let out_h = ((params.h + (2 * padding) - size) / stride_y) + 1;
    let out_c = filters;
    let mut build = LayerBuild::typed(
        out_w,
        out_h,
        out_c,
        out_w * out_h * out_c,
        vec![out_w, out_h, out_c],
    );
    build
        .weights
        .push(WeightSpec::new(format!("{prefix}biases"), vec![filters]));
    if batch_normalize != 0 && !prefix.is_empty() {
        push_batchnorm(&mut build.weights, prefix, filters);
    }
    build.weights.push(WeightSpec::new(
        format!("{prefix}weights"),
        vec![params.c / groups, filters, size, size],
    ));
    Ok(build)
}

fn deconvolutional(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let filters = option_i64(options, globals, "filters", 1)?;
    let size = option_i64(options, globals, "size", 1)?;
    let stride = option_i64(options, globals, "stride", 1)?;
    let pad = option_i64(options, globals, "pad", 0)?;
    let padding = if pad != 0 {
        size / 2
    } else {
        option_i64(options, globals, "padding", 0)?
    };
    let out_w = ((params.w - 1) * stride) + size - (2 * padding);
    let out_h = ((params.h - 1) * stride) + size - (2 * padding);
    let mut build = LayerBuild::typed(
        out_w,
        out_h,
        filters,
        out_w * out_h * filters,
        vec![out_w, out_h, filters],
    );
    build.weights.push(WeightSpec::new("biases", vec![filters]));
    build.weights.push(WeightSpec::new(
        "weights",
        vec![params.c, filters, size, size],
    ));
    Ok(build)
}

fn connected(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    inputs: i64,
    prefix: &str,
) -> Result<LayerBuild, ModelError> {
    let outputs = option_i64(options, globals, "output", 1)?;
    let batch_normalize = option_i64(options, globals, "batch_normalize", 0)?;
    let mut build = LayerBuild::typed(1, 1, outputs, outputs, vec![outputs]);
    build
        .weights
        .push(WeightSpec::new(format!("{prefix}biases"), vec![outputs]));
    if batch_normalize != 0 && !prefix.is_empty() {
        push_batchnorm(&mut build.weights, prefix, outputs);
    }
    build.weights.push(WeightSpec::new(
        format!("{prefix}weights"),
        vec![inputs, outputs],
    ));
    Ok(build)
}

fn local(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let filters = option_i64(options, globals, "filters", 1)?;
    let size = option_i64(options, globals, "size", 1)?;
    let stride = option_i64(options, globals, "stride", 1)?;
    let pad = option_i64(options, globals, "pad", 0)?;
    let out_h = ((params.h - if pad != 0 { 1 } else { size }) / stride) + 1;
    let out_w = ((params.w - if pad != 0 { 1 } else { size }) / stride) + 1;
    let out_c = filters;
    let mut build = LayerBuild::typed(
        out_w,
        out_h,
        out_c,
        out_w * out_h * out_c,
        vec![out_w, out_h, out_c],
    );
    build.weights.push(WeightSpec::new(
        "weights",
        vec![params.c, filters, size, size, out_h * out_w],
    ));
    build
        .weights
        .push(WeightSpec::new("biases", vec![out_w * out_h * out_c]));
    Ok(build)
}

fn batchnorm(params: &NetworkParams) -> Result<LayerBuild, ModelError> {
    let mut build = LayerBuild::typed(
        params.w,
        params.h,
        params.c,
        params.inputs,
        vec![params.w, params.h, params.c],
    );
    push_batchnorm(&mut build.weights, "", params.c);
    Ok(build)
}

fn maxpool(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let antialiasing = option_i64(options, globals, "antialiasing", 0)?;
    let stride = option_i64(options, globals, "stride", 1)?;
    let blur_stride_x = option_i64(options, globals, "stride_x", stride)?;
    let blur_stride_y = option_i64(options, globals, "stride_y", stride)?;
    let stride_x = if antialiasing != 0 { 1 } else { blur_stride_x };
    let stride_y = if antialiasing != 0 { 1 } else { blur_stride_y };
    let size = option_i64(options, globals, "size", stride)?;
    let padding = option_i64(options, globals, "padding", size - 1)?;
    let out_channels = option_i64(options, globals, "out_channels", 1)?;
    let maxpool_depth = option_i64(options, globals, "maxpool_depth", 0)?;
    let (mut out_w, mut out_h, mut out_c) = if maxpool_depth != 0 {
        (params.w, params.h, out_channels)
    } else {
        (
            ((params.w + padding - size) / stride_x) + 1,
            ((params.h + padding - size) / stride_y) + 1,
            params.c,
        )
    };
    if antialiasing != 0 {
        let blur_size = if antialiasing == 2 { 2 } else { 3 };
        let blur_pad = if antialiasing == 2 { 0 } else { blur_size / 3 };
        let blur = convolutional(
            &HashMap::from([
                ("filters".to_owned(), out_c.to_string()),
                ("size".to_owned(), blur_size.to_string()),
                ("stride_x".to_owned(), blur_stride_x.to_string()),
                ("stride_y".to_owned(), blur_stride_y.to_string()),
                ("padding".to_owned(), blur_pad.to_string()),
                ("groups".to_owned(), out_c.to_string()),
            ]),
            globals,
            &NetworkParams {
                h: out_h,
                w: out_w,
                c: out_c,
                inputs: out_w * out_h * out_c,
            },
            "",
        )?;
        out_w = blur.out_w;
        out_h = blur.out_h;
        out_c = blur.out_c;
    }
    Ok(LayerBuild {
        out_w,
        out_h,
        out_c,
        out: out_w * out_h * out_c,
        output_type: Some(float_type(&[out_w, out_h, out_c])?),
        weights: Vec::new(),
        breaks_inference: false,
    })
}

fn crnn(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let output_filters = option_i64(options, globals, "output", 1)?;
    let hidden_filters = option_i64(options, globals, "hidden", 1)?;
    let mut weights = Vec::new();
    let input = convolutional(
        &with_output(options, hidden_filters),
        globals,
        params,
        "input_",
    )?;
    let hidden_params = NetworkParams {
        c: hidden_filters,
        ..*params
    };
    let self_layer = convolutional(
        &with_output(options, hidden_filters),
        globals,
        &hidden_params,
        "self_",
    )?;
    let output = convolutional(
        &with_output(options, output_filters),
        globals,
        &hidden_params,
        "output_",
    )?;
    weights.extend(input.weights);
    weights.extend(self_layer.weights);
    weights.extend(output.weights);
    Ok(LayerBuild { weights, ..output })
}

fn rnn(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let outputs = option_i64(options, globals, "output", 1)?;
    let hidden = option_i64(options, globals, "hidden", 1)?;
    let mut weights = Vec::new();
    weights.extend(
        connected(
            &with_output(options, hidden),
            globals,
            params.inputs,
            "input_",
        )?
        .weights,
    );
    weights.extend(connected(&with_output(options, hidden), globals, hidden, "self_")?.weights);
    weights.extend(connected(&with_output(options, outputs), globals, hidden, "output_")?.weights);
    Ok(LayerBuild {
        weights,
        ..LayerBuild::typed(1, 1, outputs, outputs, vec![outputs])
    })
}

fn gru(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let outputs = option_i64(options, globals, "output", 1)?;
    let mut weights = Vec::new();
    for (prefix, inputs) in [
        ("input_z", params.inputs),
        ("state_z", outputs),
        ("input_r", params.inputs),
        ("state_r", outputs),
        ("input_h", params.inputs),
        ("state_h", outputs),
    ] {
        weights.extend(connected(&with_output(options, outputs), globals, inputs, prefix)?.weights);
    }
    Ok(LayerBuild {
        weights,
        ..LayerBuild::typed(1, 1, outputs, outputs, vec![outputs])
    })
}

fn lstm(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let outputs = option_i64(options, globals, "output", 1)?;
    let mut weights = Vec::new();
    for prefix in ["uf_", "ui_", "ug_", "uo_"] {
        weights.extend(
            connected(
                &with_output(options, outputs),
                globals,
                params.inputs,
                prefix,
            )?
            .weights,
        );
    }
    for prefix in ["wf_", "wi_", "wg_", "wo_"] {
        weights
            .extend(connected(&with_output(options, outputs), globals, outputs, prefix)?.weights);
    }
    Ok(LayerBuild {
        weights,
        ..LayerBuild::typed(1, 1, outputs, outputs, vec![outputs])
    })
}

fn conv_lstm(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let outputs = option_i64(options, globals, "output", 1)?;
    let bottleneck = option_i64(options, globals, "bottleneck", 0)?;
    let peephole = option_i64(options, globals, "peephole", 0)?;
    let mut weights = Vec::new();
    for prefix in ["uf_", "ui_", "ug_", "uo_"] {
        weights.extend(
            convolutional(&with_output(options, outputs), globals, params, prefix)?.weights,
        );
    }
    let state_params = NetworkParams {
        c: outputs,
        ..*params
    };
    if bottleneck != 0 {
        let params = NetworkParams {
            c: outputs * 2,
            ..*params
        };
        weights.extend(
            convolutional(&with_output(options, outputs), globals, &params, "wf_")?.weights,
        );
    } else {
        for prefix in ["wf_", "wi_", "wg_", "wo_"] {
            weights.extend(
                convolutional(
                    &with_output(options, outputs),
                    globals,
                    &state_params,
                    prefix,
                )?
                .weights,
            );
        }
    }
    if peephole != 0 {
        for prefix in ["vf_", "vi_", "vo_"] {
            weights.extend(
                convolutional(
                    &with_output(options, outputs),
                    globals,
                    &state_params,
                    prefix,
                )?
                .weights,
            );
        }
    }
    let output = convolutional(&with_output(options, outputs), globals, params, "uo_")?;
    Ok(LayerBuild {
        weights,
        output_type: output.output_type,
        out_w: output.out_w,
        out_h: output.out_h,
        out_c: outputs,
        out: output.out_w * output.out_h * outputs,
        breaks_inference: false,
    })
}

fn upsample(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let stride = option_i64(options, globals, "stride", 2)?;
    let out_w = params.w * stride;
    let out_h = params.h * stride;
    Ok(LayerBuild::typed(
        out_w,
        out_h,
        params.c,
        out_w * out_h * params.c,
        vec![out_w, out_h, params.c],
    ))
}

fn crop(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let out_h = option_i64(options, globals, "crop_height", 1)?;
    let out_w = option_i64(options, globals, "crop_width", 1)?;
    Ok(LayerBuild::typed(
        out_w,
        out_h,
        params.c,
        out_w * out_h * params.c,
        vec![out_w, out_h, params.c],
    ))
}

fn yolo(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
    gaussian: bool,
) -> Result<LayerBuild, ModelError> {
    let classes = option_i64(options, globals, "classes", 20)?;
    let num = option_i64(options, globals, "num", 1)?;
    let coords = if gaussian { 8 } else { 4 };
    let out_c = num * (classes + coords + 1);
    Ok(LayerBuild::typed(
        params.w,
        params.h,
        out_c,
        params.w * params.h * out_c,
        vec![params.w, params.h, out_c],
    ))
}

fn region(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let coords = option_i64(options, globals, "coords", 4)?;
    let classes = option_i64(options, globals, "classes", 20)?;
    let num = option_i64(options, globals, "num", 1)?;
    let tail = classes + coords + 1;
    Ok(LayerBuild::typed(
        params.w,
        params.h,
        num,
        params.h * params.w * num * tail,
        vec![params.h, params.w, num, tail],
    ))
}

fn reorg(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    params: &NetworkParams,
) -> Result<LayerBuild, ModelError> {
    let stride = option_i64(options, globals, "stride", 1)?;
    let reverse = option_i64(options, globals, "reverse", 0)?;
    let extra = option_i64(options, globals, "extra", 0)?;
    if extra != 0 {
        let out = params.h * params.w * params.c + extra;
        return Ok(LayerBuild::typed(0, 0, 0, out, vec![out]));
    }
    let (out_w, out_h, out_c) = if reverse != 0 {
        (
            params.w * stride,
            params.h * stride,
            params.c / (stride * stride),
        )
    } else {
        (
            params.w / stride,
            params.h / stride,
            params.c * stride * stride,
        )
    };
    Ok(LayerBuild::typed(
        out_w,
        out_h,
        out_c,
        out_w * out_h * out_c,
        vec![out_w, out_h, out_c],
    ))
}

fn route(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    states: &[LayerState],
    index: usize,
) -> Result<LayerBuild, ModelError> {
    let groups = option_i64(options, globals, "groups", 1)?;
    let routes = options
        .get("layers")
        .map(|value| parse_routes(value))
        .unwrap_or_default();
    let mut resolved = Vec::new();
    for route in routes {
        if let Some(state) = resolve_state(states, index, route) {
            resolved.push(state);
        }
    }
    let Some(first) = resolved.first() else {
        return Ok(LayerBuild::unknown_break());
    };
    let mut out_c = first.out_c / groups;
    for state in resolved.iter().skip(1) {
        if state.out_w != first.out_w || state.out_h != first.out_h {
            return Ok(LayerBuild::unknown_break());
        }
        out_c += state.out_c;
    }
    Ok(LayerBuild::typed(
        first.out_w,
        first.out_h,
        out_c,
        first.out_w * first.out_h * out_c,
        vec![first.out_w, first.out_h, out_c],
    ))
}

fn from_shape(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    states: &[LayerState],
    index: usize,
) -> Result<LayerBuild, ModelError> {
    let from = option_i64(options, globals, "from", 0)?;
    if let Some(state) = resolve_state(states, index, from) {
        Ok(LayerBuild::typed(
            state.out_w,
            state.out_h,
            state.out_c,
            state.out,
            vec![state.out_w, state.out_h, state.out_c],
        ))
    } else {
        Ok(LayerBuild::unknown_break())
    }
}

fn same_image(params: &NetworkParams, out: i64) -> Result<LayerBuild, ModelError> {
    Ok(LayerBuild::typed(
        params.w,
        params.h,
        params.c,
        out,
        vec![params.w, params.h, params.c],
    ))
}

fn add_weight(
    model: &mut Model,
    graph: &mut Graph,
    spec: &WeightSpec,
) -> Result<ValueId, ModelError> {
    let shape = spec
        .shape
        .iter()
        .copied()
        .map(Dimension::known)
        .collect::<Vec<_>>();
    let tensor = Tensor::metadata_only(
        None,
        TensorElementType::Float32,
        shape.clone(),
        TensorStorage::Absent,
    );
    let tensor_id = model.add_tensor(tensor);
    Ok(add_value(
        model,
        graph,
        "",
        Some(TypeInfo {
            element_type: Some(TensorElementType::Float32),
            layout: None,
            denotation: None,
            shape,
        }),
        Some(tensor_id),
    ))
}

fn add_value(
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

fn lower_attribute(
    model: &mut Model,
    metadata: &DarknetMetadata,
    section: &str,
    name: &str,
    value: &str,
) -> Attribute {
    let value = match metadata.attribute_type(section, name) {
        Some("int32") => value
            .parse::<i64>()
            .map(AttributeValue::Int)
            .unwrap_or_else(|_| AttributeValue::String(model.intern(value))),
        Some("float32") => value
            .parse::<f32>()
            .map(AttributeValue::Float)
            .unwrap_or_else(|_| AttributeValue::String(model.intern(value))),
        Some("int32[]") => {
            let values = value
                .split(',')
                .map(|item| item.trim().parse::<i64>())
                .collect::<Result<Vec<_>, _>>();
            values
                .map(AttributeValue::Ints)
                .unwrap_or_else(|_| AttributeValue::String(model.intern(value)))
        }
        _ => AttributeValue::String(model.intern(value)),
    };
    Attribute {
        name: model.intern(name),
        value,
    }
}

fn parse_cfg(text: &str) -> Result<Vec<Section>, ModelError> {
    let mut sections = Vec::new();
    let mut current: Option<Section> = None;
    for (index, content) in text.lines().enumerate() {
        let line = strip_ws(content);
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(stripped) = line.strip_prefix('[') {
            if let Some(section) = current.take() {
                sections.push(section);
            }
            let kind = stripped.strip_suffix(']').unwrap_or(stripped).to_owned();
            current = Some(Section {
                kind,
                name: String::new(),
                options: HashMap::new(),
            });
            continue;
        }
        let Some(section) = current.as_mut() else {
            return Err(invalid(format!(
                "Invalid cfg '{}' at line {}.",
                printable(content),
                index + 1
            )));
        };
        let Some((key, value)) = line.split_once('=') else {
            return Err(invalid(format!(
                "Invalid cfg '{}' at line {}.",
                printable(content),
                index + 1
            )));
        };
        section.options.insert(key.to_owned(), value.to_owned());
    }
    if let Some(section) = current {
        sections.push(section);
    }
    Ok(sections)
}

fn strip_ws(value: &str) -> String {
    value.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn printable(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_graphic() || ch == ' ' {
                ch
            } else {
                '?'
            }
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn option_i64(
    options: &HashMap<String, String>,
    globals: &HashMap<String, String>,
    key: &str,
    default: i64,
) -> Result<i64, ModelError> {
    let Some(value) = options.get(key) else {
        return Ok(default);
    };
    let value = value
        .strip_prefix('$')
        .and_then(|name| globals.get(name))
        .unwrap_or(value);
    value
        .parse::<i64>()
        .map_err(|_| invalid(format!("Invalid int option '{}'.", value)))
}

fn parse_routes(value: &str) -> Vec<i64> {
    value
        .split(',')
        .filter_map(|item| item.trim().parse::<i64>().ok())
        .collect()
}

fn resolve_state(states: &[LayerState], current: usize, route: i64) -> Option<&LayerState> {
    let index = if route < 0 {
        i64::try_from(current).ok()?.checked_add(route)?
    } else {
        route
    };
    usize::try_from(index)
        .ok()
        .and_then(|index| states.get(index))
}

fn float_type(shape: &[i64]) -> Result<TypeInfo, ModelError> {
    if shape
        .iter()
        .any(|dimension| *dimension == 0 || *dimension == i64::MIN)
    {
        return Err(invalid(format!("Invalid tensor shape '{shape:?}'.")));
    }
    Ok(TypeInfo {
        element_type: Some(TensorElementType::Float32),
        layout: None,
        denotation: None,
        shape: shape.iter().copied().map(Dimension::known).collect(),
    })
}

fn push_batchnorm(weights: &mut Vec<WeightSpec>, prefix: &str, size: i64) {
    weights.push(WeightSpec::new(format!("{prefix}scale"), vec![size]));
    weights.push(WeightSpec::new(format!("{prefix}mean"), vec![size]));
    weights.push(WeightSpec::new(format!("{prefix}variance"), vec![size]));
}

fn with_output(options: &HashMap<String, String>, output: i64) -> HashMap<String, String> {
    let mut result = options.clone();
    result.insert("filters".to_owned(), output.to_string());
    result.insert("output".to_owned(), output.to_string());
    result
}

fn metadata() -> &'static DarknetMetadata {
    static METADATA_CACHE: OnceLock<DarknetMetadata> = OnceLock::new();
    METADATA_CACHE.get_or_init(|| DarknetMetadata::new(METADATA))
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}

#[derive(Debug)]
struct Section {
    kind: String,
    name: String,
    options: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
struct NetworkParams {
    h: i64,
    w: i64,
    c: i64,
    inputs: i64,
}

#[derive(Debug, Clone)]
struct LayerState {
    outputs: Vec<ValueId>,
    out_w: i64,
    out_h: i64,
    out_c: i64,
    out: i64,
}

#[derive(Debug, Clone)]
struct LayerBuild {
    out_w: i64,
    out_h: i64,
    out_c: i64,
    out: i64,
    output_type: Option<TypeInfo>,
    weights: Vec<WeightSpec>,
    breaks_inference: bool,
}

impl LayerBuild {
    fn typed(out_w: i64, out_h: i64, out_c: i64, out: i64, shape: Vec<i64>) -> Self {
        Self {
            out_w,
            out_h,
            out_c,
            out,
            output_type: float_type(&shape).ok(),
            weights: Vec::new(),
            breaks_inference: false,
        }
    }

    fn unknown() -> Self {
        Self {
            out_w: 0,
            out_h: 0,
            out_c: 0,
            out: 0,
            output_type: None,
            weights: Vec::new(),
            breaks_inference: false,
        }
    }

    fn unknown_break() -> Self {
        Self {
            breaks_inference: true,
            ..Self::unknown()
        }
    }
}

#[derive(Debug, Clone)]
struct WeightSpec {
    #[allow(dead_code)]
    name: String,
    shape: Vec<i64>,
}

impl WeightSpec {
    fn new(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            shape,
        }
    }
}

#[derive(Debug)]
struct DarknetMetadata {
    attributes: HashMap<(String, String), String>,
}

impl DarknetMetadata {
    fn new(data: &str) -> Self {
        let entries = serde_json::from_str::<Vec<MetadataType>>(data).unwrap_or_default();
        let mut attributes = HashMap::new();
        for entry in entries {
            for attribute in entry.attributes {
                if let Some(attribute_type) = attribute.attribute_type {
                    attributes.insert((entry.name.clone(), attribute.name), attribute_type);
                }
            }
        }
        Self { attributes }
    }

    fn attribute_type(&self, section: &str, name: &str) -> Option<&str> {
        self.attributes
            .get(&(section.to_owned(), name.to_owned()))
            .map(String::as_str)
    }
}

#[derive(Debug, Deserialize)]
struct MetadataType {
    name: String,
    #[serde(default)]
    attributes: Vec<MetadataAttribute>,
}

#[derive(Debug, Deserialize)]
struct MetadataAttribute {
    name: String,
    #[serde(default, rename = "type")]
    attribute_type: Option<String>,
}
