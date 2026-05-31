use std::collections::{BTreeMap, HashMap};

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, Tensor, TensorElementType, TensorStorage,
    TypeInfo, Value, ValueId,
};

const FORMAT: &str = "GGUF";

pub struct GgufFormat;

impl ModelFormat for GgufFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["gguf"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if input.data.len() >= 4 && input.data.get(0..4) == Some(b"GGUF") {
            return Confidence::High;
        }
        Confidence::None
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        let target = Reader::new(input.data).read()?;
        lower_target(target)
    }
}

fn lower_target(target: Target) -> Result<Model, ModelError> {
    let architecture = target
        .metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        .unwrap_or("?");
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(target.version.to_string()),
    });
    model.metadata.description = target
        .metadata
        .get("general.description")
        .and_then(MetadataValue::as_str)
        .map(ToOwned::to_owned);
    model.metadata.properties = model_metadata_properties(&target, architecture);

    let graph_name = if architecture == "?" {
        None
    } else {
        Some(model.intern(architecture))
    };
    let graph_id = model.add_graph_placeholder(None, graph_name);
    let mut graph = Graph::new(graph_id, None, graph_name);
    graph.metadata = graph_metadata(&target, architecture);
    let tokenizer_attributes = tokenizer_attributes(&target);

    let mut context = LowerContext {
        model: &mut model,
        graph: &mut graph,
        graph_id,
        architecture,
        metadata: &target.metadata,
        tensors: &target.tensors,
        tokenizer_attributes: &tokenizer_attributes,
        emitted_tokenizer_attributes: false,
        values: HashMap::new(),
        next_value: 0,
    };
    if architecture == "bert" && target.has_tokenizer {
        lower_bert(&mut context, target.block_count);
    } else if architecture == "t5" || context.tensors.contains_key("enc.blk.0.attn_norm.weight") {
        lower_t5(&mut context, target.block_count, target.has_tokenizer);
    } else if context.tensors.contains_key("blk.0.attn_qkv.weight") {
        lower_phi2(&mut context, target.block_count, target.has_tokenizer);
    } else if context.tensors.contains_key("blk.0.attn_kv_a_mqa.weight") {
        lower_mistral4(&mut context, target.block_count, target.has_tokenizer);
    } else if context.tensors.contains_key("token_embd.weight")
        && context.tensors.contains_key("blk.0.attn_norm.weight")
    {
        lower_pre_norm_transformer(&mut context, target.block_count, target.has_tokenizer);
    } else {
        lower_weights_fallback(&mut context);
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn model_metadata_properties(target: &Target, architecture: &str) -> BTreeMap<String, String> {
    let mut properties = BTreeMap::new();
    let architecture_prefix = format!("{architecture}.");
    for (key, value) in &target.metadata_entries {
        if key == "general.name"
            || key == "general.architecture"
            || key == "general.description"
            || key.starts_with("tokenizer.")
            || (architecture != "?" && key.starts_with(&architecture_prefix))
        {
            continue;
        }
        let property = if key.starts_with("general.") {
            key.rsplit_once('.')
                .map(|(_, name)| name)
                .unwrap_or(key.as_str())
        } else {
            key.as_str()
        };
        if let Some(value) = value.to_property() {
            properties.insert(property.to_owned(), value);
        }
    }
    properties
}

fn graph_metadata(target: &Target, architecture: &str) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    if architecture == "?" {
        return metadata;
    }
    let prefix = format!("{architecture}.");
    for (key, value) in &target.metadata_entries {
        if let Some(name) = key.strip_prefix(&prefix)
            && let Some(value) = value.to_property()
        {
            metadata.insert(name.to_owned(), value);
        }
    }
    metadata
}

fn tokenizer_attributes(target: &Target) -> BTreeMap<String, MetadataValue> {
    let mut attributes = BTreeMap::new();
    for (key, value) in &target.metadata_entries {
        if key.starts_with("tokenizer.")
            && let Some((_, name)) = key.rsplit_once('.')
        {
            attributes.insert(name.to_owned(), value.clone());
        }
    }
    attributes
}

fn lower_metadata_attribute(model: &mut Model, value: &MetadataValue) -> AttributeValue {
    match value {
        MetadataValue::U32(value) => AttributeValue::Int(i64::from(*value)),
        MetadataValue::U64(value) => AttributeValue::Int(i64::try_from(*value).unwrap_or(i64::MAX)),
        MetadataValue::I32(value) => AttributeValue::Int(i64::from(*value)),
        MetadataValue::I64(value) => AttributeValue::Int(*value),
        MetadataValue::F32(value) => AttributeValue::Float(*value),
        MetadataValue::F64(value) => AttributeValue::Float(*value as f32),
        MetadataValue::Bool(value) => AttributeValue::Bool(*value),
        MetadataValue::String(value) => AttributeValue::String(model.intern(value)),
        MetadataValue::Array {
            element_type,
            values,
        } if is_integer_metadata_type(*element_type) => {
            AttributeValue::Ints(values.iter().filter_map(MetadataValue::as_i64).collect())
        }
        MetadataValue::Array {
            element_type,
            values,
        } if is_float_metadata_type(*element_type) => {
            AttributeValue::Floats(values.iter().filter_map(MetadataValue::as_f32).collect())
        }
        MetadataValue::Array {
            element_type,
            values,
        } if *element_type == GgufType::STRING => AttributeValue::Strings(
            values
                .iter()
                .filter_map(MetadataValue::as_str)
                .map(|value| model.intern(value))
                .collect(),
        ),
        MetadataValue::Array { .. } => {
            AttributeValue::Unsupported("unsupported GGUF metadata array".to_owned())
        }
    }
}

fn is_integer_metadata_type(value_type: u32) -> bool {
    matches!(
        value_type,
        GgufType::UINT8
            | GgufType::INT8
            | GgufType::UINT16
            | GgufType::INT16
            | GgufType::UINT32
            | GgufType::INT32
            | GgufType::UINT64
            | GgufType::INT64
    )
}

fn is_float_metadata_type(value_type: u32) -> bool {
    matches!(value_type, GgufType::FLOAT32 | GgufType::FLOAT64)
}

fn operator_metadata_attributes(operator: &str) -> &'static [(&'static str, &'static str)] {
    match operator {
        "RMS_NORM" => &[("attention.layer_norm_rms_epsilon", "epsilon")],
        "LAYER_NORM" => &[("attention.layer_norm_epsilon", "epsilon")],
        "MULTI_HEAD_ATTENTION" => &[
            ("attention.head_count", "head_count"),
            ("attention.head_count_kv", "head_count_kv"),
            ("attention.key_length", "key_length"),
            ("attention.value_length", "value_length"),
            ("attention.sliding_window", "sliding_window"),
        ],
        "MULTI_LATENT_ATTENTION" => &[
            ("attention.head_count", "head_count"),
            ("attention.head_count_kv", "head_count_kv"),
            ("attention.q_lora_rank", "q_lora_rank"),
            ("attention.kv_lora_rank", "kv_lora_rank"),
            ("attention.key_length_mla", "key_length_mla"),
            ("attention.value_length_mla", "value_length_mla"),
        ],
        "CROSS_ATTENTION" => &[
            ("attention.head_count", "head_count"),
            ("attention.head_count_kv", "head_count_kv"),
            ("attention.key_length", "key_length"),
            ("attention.value_length", "value_length"),
        ],
        "ROPE_FREQS" => &[
            ("rope.dimension_count", "dimension_count"),
            ("rope.freq_base", "freq_base"),
            ("rope.scaling.type", "scaling_type"),
            ("rope.scaling.factor", "scaling_factor"),
            (
                "rope.scaling.original_context_length",
                "original_context_length",
            ),
        ],
        "MAMBA" => &[
            ("ssm.state_size", "state_size"),
            ("ssm.conv_kernel", "conv_kernel"),
            ("ssm.inner_size", "inner_size"),
            ("ssm.time_step_rank", "time_step_rank"),
        ],
        "MAMBA2" => &[
            ("ssm.state_size", "state_size"),
            ("ssm.conv_kernel", "conv_kernel"),
            ("ssm.inner_size", "inner_size"),
            ("ssm.group_count", "group_count"),
        ],
        "CONV_1D" => &[("shortconv.l_cache", "l_cache")],
        _ => &[],
    }
}

struct LowerContext<'a> {
    model: &'a mut Model,
    graph: &'a mut Graph,
    graph_id: netron_rs_core::GraphId,
    architecture: &'a str,
    metadata: &'a BTreeMap<String, MetadataValue>,
    tensors: &'a BTreeMap<String, TensorInfo>,
    tokenizer_attributes: &'a BTreeMap<String, MetadataValue>,
    emitted_tokenizer_attributes: bool,
    values: HashMap<String, ValueId>,
    next_value: usize,
}

impl LowerContext<'_> {
    fn new_value(&mut self) -> ValueId {
        let name = format!("v{}", self.next_value);
        self.next_value += 1;
        self.ensure_value(&name)
    }

    fn ensure_value(&mut self, name: &str) -> ValueId {
        if let Some(value_id) = self.values.get(name) {
            return *value_id;
        }
        let name_id = self.model.intern(name);
        let value_id = self.graph.add_value(Value::new(name_id));
        if let Some(tensor) = self.tensors.get(name) {
            let element_type = tensor.element_type.clone();
            let shape = tensor.shape.clone();
            let tensor_id = self.model.add_tensor(Tensor::metadata_only(
                Some(name_id),
                element_type.clone(),
                shape.clone(),
                TensorStorage::InlineBytes {
                    byte_len: tensor.byte_len,
                },
            ));
            let value = &mut self.graph.values[value_id.index()];
            value.initializer = Some(tensor_id);
            if let Some(quantization) = &tensor.quantization {
                value
                    .quantization
                    .push(netron_rs_core::QuantizationAnnotation {
                        key: self.model.intern("type"),
                        value: self.model.intern(quantization),
                    });
            }
            value.type_info = Some(TypeInfo {
                element_type: Some(element_type),
                layout: None,
                denotation: None,
                shape,
            });
        }
        self.values.insert(name.to_owned(), value_id);
        value_id
    }

    fn add_node(
        &mut self,
        name: Option<&str>,
        operator: &str,
        inputs: Vec<ValueId>,
        output: Option<ValueId>,
    ) {
        let op_name = self.model.intern(operator);
        let mut node = Node::new(
            self.graph_id,
            Operator {
                domain: None,
                name: op_name,
                overload: None,
                version: None,
                origin: FORMAT,
            },
        );
        node.name = name
            .filter(|name| !name.is_empty())
            .map(|name| self.model.intern(name));
        node.inputs = inputs.iter().copied().map(Some).collect();
        if let Some(output) = output {
            node.outputs.push(Some(output));
        }
        if operator == "tokenizer" && inputs.is_empty() && !self.emitted_tokenizer_attributes {
            for (name, value) in self.tokenizer_attributes.clone() {
                node.attributes.push(Attribute {
                    name: self.model.intern(&name),
                    value: lower_metadata_attribute(self.model, &value),
                });
            }
            self.emitted_tokenizer_attributes = true;
        } else {
            for (key, name) in operator_metadata_attributes(operator) {
                let key = format!("{}.{key}", self.architecture);
                if let Some(value) = self.metadata.get(&key) {
                    node.attributes.push(Attribute {
                        name: self.model.intern(name),
                        value: lower_metadata_attribute(self.model, value),
                    });
                }
            }
        }
        let node_id = self.graph.add_node(node);
        for input in inputs {
            let consumers = &mut self.graph.values[input.index()].consumers;
            if !consumers.contains(&node_id) {
                consumers.push(node_id);
            }
        }
        if let Some(output) = output {
            self.graph.values[output.index()].producer = Some(node_id);
        }
    }

    fn weight(&mut self, name: &str) -> ValueId {
        self.ensure_value(name)
    }
}

fn lower_bert(context: &mut LowerContext<'_>, block_count: usize) {
    let mut prev = context.new_value();
    context.add_node(None, "tokenizer", Vec::new(), Some(prev));

    for (name, operator, weights) in [
        ("token_embd", "EMBEDDING", &["token_embd.weight"][..]),
        ("token_types", "EMBEDDING", &["token_types.weight"][..]),
        (
            "token_embd_norm",
            "LAYER_NORM",
            &["token_embd_norm.weight", "token_embd_norm.bias"][..],
        ),
        ("position_embd", "EMBEDDING", &["position_embd.weight"][..]),
    ] {
        if weights
            .iter()
            .all(|weight| context.tensors.contains_key(*weight))
        {
            let out = context.new_value();
            let mut inputs = vec![prev];
            inputs.extend(weights.iter().map(|weight| context.weight(weight)));
            context.add_node(Some(name), operator, inputs, Some(out));
            prev = out;
        }
    }

    for block in 0..block_count {
        let prefix = format!("blk.{block}");
        let attention_weights = [
            format!("{prefix}.attn_q.weight"),
            format!("{prefix}.attn_q.bias"),
            format!("{prefix}.attn_k.weight"),
            format!("{prefix}.attn_k.bias"),
            format!("{prefix}.attn_v.weight"),
            format!("{prefix}.attn_v.bias"),
            format!("{prefix}.attn_output.weight"),
            format!("{prefix}.attn_output.bias"),
        ];
        if attention_weights
            .iter()
            .all(|weight| context.tensors.contains_key(weight))
        {
            let attention = context.new_value();
            let mut inputs = vec![prev];
            inputs.extend(
                attention_weights
                    .iter()
                    .map(|weight| context.weight(weight)),
            );
            context.add_node(
                Some("attention"),
                "MULTI_HEAD_ATTENTION",
                inputs,
                Some(attention),
            );

            let residual = context.new_value();
            context.add_node(None, "ADD", vec![prev, attention], Some(residual));
            prev = residual;
        }

        if has_pair(context, &prefix, "attn_output_norm") {
            let norm = context.new_value();
            let weight = context.weight(&format!("{prefix}.attn_output_norm.weight"));
            let bias = context.weight(&format!("{prefix}.attn_output_norm.bias"));
            context.add_node(
                Some("attn_output_norm"),
                "LAYER_NORM",
                vec![prev, weight, bias],
                Some(norm),
            );
            prev = norm;
        }

        if has_pair(context, &prefix, "ffn_up") && has_pair(context, &prefix, "ffn_down") {
            let ffn_up = context.new_value();
            let up_weight = context.weight(&format!("{prefix}.ffn_up.weight"));
            let up_bias = context.weight(&format!("{prefix}.ffn_up.bias"));
            context.add_node(
                Some("ffn_up"),
                "MUL_MAT",
                vec![prev, up_weight, up_bias],
                Some(ffn_up),
            );

            let ffn_down = context.new_value();
            let down_weight = context.weight(&format!("{prefix}.ffn_down.weight"));
            let down_bias = context.weight(&format!("{prefix}.ffn_down.bias"));
            context.add_node(
                Some("ffn_down"),
                "MUL_MAT",
                vec![ffn_up, prev, down_weight, down_bias],
                Some(ffn_down),
            );

            let residual = context.new_value();
            context.add_node(None, "ADD", vec![prev, ffn_down], Some(residual));
            prev = residual;
        }

        if has_pair(context, &prefix, "layer_output_norm") {
            let norm = context.new_value();
            let weight = context.weight(&format!("{prefix}.layer_output_norm.weight"));
            let bias = context.weight(&format!("{prefix}.layer_output_norm.bias"));
            context.add_node(
                Some("layer_output_norm"),
                "LAYER_NORM",
                vec![prev, weight, bias],
                Some(norm),
            );
            prev = norm;
        }
    }

    let out = context.new_value();
    context.add_node(None, "tokenizer", vec![prev], Some(out));
}

fn lower_pre_norm_transformer(
    context: &mut LowerContext<'_>,
    block_count: usize,
    has_tokenizer: bool,
) {
    let mut prev = if has_tokenizer {
        let value = context.new_value();
        context.add_node(None, "tokenizer", Vec::new(), Some(value));
        value
    } else {
        context.new_value()
    };

    if context.tensors.contains_key("token_embd.weight") {
        let out = context.new_value();
        let weight = context.weight("token_embd.weight");
        context.add_node(
            Some("token_embd"),
            "EMBEDDING",
            vec![prev, weight],
            Some(out),
        );
        prev = out;
    }

    let rope_freqs = if context.tensors.contains_key("rope_freqs.weight") {
        let value = context.new_value();
        let weight = context.weight("rope_freqs.weight");
        context.add_node(Some("rope_freqs"), "ROPE_FREQS", vec![weight], Some(value));
        Some(value)
    } else {
        None
    };
    let per_layer = if [
        "per_layer_token_embd.weight",
        "per_layer_model_proj.weight",
        "per_layer_proj_norm.weight",
    ]
    .iter()
    .all(|name| context.tensors.contains_key(*name))
    {
        let embedding = context.new_value();
        let weight = context.weight("per_layer_token_embd.weight");
        context.add_node(
            Some("per_layer_token_embd"),
            "EMBEDDING",
            vec![weight],
            Some(embedding),
        );

        let projection = context.new_value();
        let weight = context.weight("per_layer_model_proj.weight");
        context.add_node(
            Some("per_layer_model_proj"),
            "MUL_MAT",
            vec![prev, weight],
            Some(projection),
        );

        let norm = context.new_value();
        let weight = context.weight("per_layer_proj_norm.weight");
        context.add_node(
            Some("per_layer_proj_norm"),
            "RMS_NORM",
            vec![projection, weight],
            Some(norm),
        );

        let sum = context.new_value();
        context.add_node(None, "ADD", vec![embedding, norm], Some(sum));
        Some(sum)
    } else {
        None
    };

    for block in 0..block_count {
        let prefix = format!("blk.{block}");
        if !context
            .tensors
            .contains_key(&format!("{prefix}.attn_norm.weight"))
        {
            continue;
        }

        let attn_norm = context.new_value();
        let attn_norm_weights = attention_norm_weights(context, &prefix);
        let mut attn_norm_inputs = vec![prev];
        attn_norm_inputs.extend(attn_norm_weights);
        context.add_node(
            Some("attn_norm"),
            "RMS_NORM",
            attn_norm_inputs,
            Some(attn_norm),
        );

        let attention_weights = attention_weights(context, &prefix);
        let attention = if attention_weights.is_empty() {
            None
        } else {
            let attention = context.new_value();
            let mut inputs = Vec::new();
            if let Some(rope_freqs) = rope_freqs {
                inputs.push(rope_freqs);
            }
            inputs.push(attn_norm);
            inputs.extend(attention_weights);
            context.add_node(
                Some("attention"),
                "MULTI_HEAD_ATTENTION",
                inputs,
                Some(attention),
            );
            Some(attention)
        };

        let Some(mut attention_or_ssm) =
            combine_attention_and_ssm(context, &prefix, attn_norm, attention)
        else {
            continue;
        };

        if context
            .tensors
            .contains_key(&format!("{prefix}.post_attention_norm.weight"))
        {
            let post_norm = context.new_value();
            let weight = context.weight(&format!("{prefix}.post_attention_norm.weight"));
            context.add_node(
                Some("attn_post_norm"),
                "RMS_NORM",
                vec![attention_or_ssm, weight],
                Some(post_norm),
            );
            attention_or_ssm = post_norm;
        }

        let attention_residual = context.new_value();
        context.add_node(
            None,
            "ADD",
            vec![prev, attention_or_ssm],
            Some(attention_residual),
        );
        prev = attention_residual;

        let ffn_input = if let Some(ffn_norm_name) = norm_weight_name(context, &prefix, "ffn_norm")
        {
            let ffn_norm = context.new_value();
            let ffn_norm_weight = context.weight(&ffn_norm_name);
            context.add_node(
                Some("ffn_norm"),
                "RMS_NORM",
                vec![prev, ffn_norm_weight],
                Some(ffn_norm),
            );
            ffn_norm
        } else if context
            .tensors
            .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
        {
            prev
        } else {
            continue;
        };

        let mut ffn_out = if context
            .tensors
            .contains_key(&format!("{prefix}.ffn_gate_inp.weight"))
        {
            lower_moe_ffn(context, &prefix, ffn_input)
        } else {
            lower_dense_gated_ffn(context, &prefix, ffn_input)
        };

        if context
            .tensors
            .contains_key(&format!("{prefix}.post_ffw_norm.weight"))
        {
            let post_norm = context.new_value();
            let weight = context.weight(&format!("{prefix}.post_ffw_norm.weight"));
            context.add_node(
                Some("ffn_post_norm"),
                "RMS_NORM",
                vec![ffn_out, weight],
                Some(post_norm),
            );
            ffn_out = post_norm;
        }

        let residual = context.new_value();
        context.add_node(None, "ADD", vec![prev, ffn_out], Some(residual));
        prev = residual;

        if let Some(per_layer) = per_layer
            && [
                "inp_gate.weight",
                "proj.weight",
                "post_norm.weight",
                "layer_output_scale.weight",
            ]
            .iter()
            .all(|name| context.tensors.contains_key(&format!("{prefix}.{name}")))
        {
            let gate = context.new_value();
            let weight = context.weight(&format!("{prefix}.inp_gate.weight"));
            context.add_node(Some("inp_gate"), "MUL_MAT", vec![prev, weight], Some(gate));

            let gelu = context.new_value();
            context.add_node(None, "GELU", vec![gate], Some(gelu));

            let gated = context.new_value();
            context.add_node(None, "MUL", vec![per_layer, gelu], Some(gated));

            let projection = context.new_value();
            let weight = context.weight(&format!("{prefix}.proj.weight"));
            context.add_node(
                Some("proj"),
                "MUL_MAT",
                vec![gated, weight],
                Some(projection),
            );

            let norm = context.new_value();
            let weight = context.weight(&format!("{prefix}.post_norm.weight"));
            context.add_node(
                Some("post_norm"),
                "RMS_NORM",
                vec![projection, weight],
                Some(norm),
            );

            let sum = context.new_value();
            context.add_node(None, "ADD", vec![prev, norm], Some(sum));

            let scaled = context.new_value();
            let weight = context.weight(&format!("{prefix}.layer_output_scale.weight"));
            context.add_node(
                Some("layer_out_scale"),
                "MUL",
                vec![sum, weight],
                Some(scaled),
            );
            prev = scaled;
        }
    }

    if context.tensors.contains_key("output_norm.weight") {
        let output_norm = context.new_value();
        let weight = context.weight("output_norm.weight");
        context.add_node(
            Some("output_norm"),
            "RMS_NORM",
            vec![prev, weight],
            Some(output_norm),
        );
        prev = output_norm;
    }
    if context.tensors.contains_key("output.weight") {
        let output = context.new_value();
        let weight = context.weight("output.weight");
        context.add_node(Some("output"), "MUL_MAT", vec![prev, weight], Some(output));
        prev = output;
    }
    if has_tokenizer {
        let out = context.new_value();
        context.add_node(None, "tokenizer", vec![prev], Some(out));
    }
}

fn lower_phi2(context: &mut LowerContext<'_>, block_count: usize, has_tokenizer: bool) {
    let mut prev = if has_tokenizer {
        let value = context.new_value();
        context.add_node(None, "tokenizer", Vec::new(), Some(value));
        value
    } else {
        context.new_value()
    };

    if context.tensors.contains_key("token_embd.weight") {
        let out = context.new_value();
        let weight = context.weight("token_embd.weight");
        context.add_node(
            Some("token_embd"),
            "EMBEDDING",
            vec![prev, weight],
            Some(out),
        );
        prev = out;
    }

    for block in 0..block_count {
        let prefix = format!("blk.{block}");
        if !context
            .tensors
            .contains_key(&format!("{prefix}.attn_qkv.weight"))
        {
            continue;
        }

        let attn_norm = context.new_value();
        let attn_norm_bias = context.weight(&format!("{prefix}.attn_norm.bias"));
        let attn_norm_weight = context.weight(&format!("{prefix}.attn_norm.weight"));
        context.add_node(
            Some("attn_norm"),
            "LAYER_NORM",
            vec![prev, attn_norm_bias, attn_norm_weight],
            Some(attn_norm),
        );

        let attention = context.new_value();
        let qkv_bias = context.weight(&format!("{prefix}.attn_qkv.bias"));
        let qkv_weight = context.weight(&format!("{prefix}.attn_qkv.weight"));
        let output_bias = context.weight(&format!("{prefix}.attn_output.bias"));
        let output_weight = context.weight(&format!("{prefix}.attn_output.weight"));
        context.add_node(
            Some("attention"),
            "MULTI_HEAD_ATTENTION",
            vec![attn_norm, qkv_bias, qkv_weight, output_bias, output_weight],
            Some(attention),
        );

        let residual = context.new_value();

        let ffn_up = context.new_value();
        let up_bias = context.weight(&format!("{prefix}.ffn_up.bias"));
        let up_weight = context.weight(&format!("{prefix}.ffn_up.weight"));
        context.add_node(
            Some("ffn_up"),
            "MUL_MAT",
            vec![attn_norm, up_bias, up_weight],
            Some(ffn_up),
        );

        let ffn_down = context.new_value();
        let down_bias = context.weight(&format!("{prefix}.ffn_down.bias"));
        let down_weight = context.weight(&format!("{prefix}.ffn_down.weight"));
        context.add_node(
            Some("ffn_down"),
            "MUL_MAT",
            vec![ffn_up, attn_norm, down_bias, down_weight],
            Some(ffn_down),
        );

        context.add_node(None, "ADD", vec![prev, ffn_down, attention], Some(residual));
        prev = residual;
    }

    if context.tensors.contains_key("output_norm.weight") {
        let output_norm = context.new_value();
        let bias = context.weight("output_norm.bias");
        let weight = context.weight("output_norm.weight");
        context.add_node(
            Some("output_norm"),
            "LAYER_NORM",
            vec![prev, bias, weight],
            Some(output_norm),
        );
        prev = output_norm;
    }
    if context.tensors.contains_key("output.weight") {
        let output = context.new_value();
        let bias = context.weight("output.bias");
        let weight = context.weight("output.weight");
        context.add_node(
            Some("output"),
            "MUL_MAT",
            vec![prev, bias, weight],
            Some(output),
        );
        prev = output;
    }
    if has_tokenizer {
        let out = context.new_value();
        context.add_node(None, "tokenizer", vec![prev], Some(out));
    }
}

fn lower_mistral4(context: &mut LowerContext<'_>, block_count: usize, has_tokenizer: bool) {
    let mut prev = if has_tokenizer {
        let value = context.new_value();
        context.add_node(None, "tokenizer", Vec::new(), Some(value));
        value
    } else {
        context.new_value()
    };

    if context.tensors.contains_key("token_embd.weight") {
        let out = context.new_value();
        let weight = context.weight("token_embd.weight");
        context.add_node(
            Some("token_embd"),
            "EMBEDDING",
            vec![prev, weight],
            Some(out),
        );
        prev = out;
    }

    for block in 0..block_count {
        let prefix = format!("blk.{block}");
        if !context
            .tensors
            .contains_key(&format!("{prefix}.attn_norm.weight"))
        {
            continue;
        }

        let attn_norm = context.new_value();
        let attn_norm_weight = context.weight(&format!("{prefix}.attn_norm.weight"));
        context.add_node(
            Some("attn_norm"),
            "RMS_NORM",
            vec![prev, attn_norm_weight],
            Some(attn_norm),
        );

        let attention = context.new_value();
        let mut inputs = vec![attn_norm];
        inputs.extend(mistral4_attention_weights(context, &prefix));
        context.add_node(
            Some("attention"),
            "MULTI_LATENT_ATTENTION",
            inputs,
            Some(attention),
        );

        let attention_residual = context.new_value();
        context.add_node(None, "ADD", vec![prev, attention], Some(attention_residual));
        prev = attention_residual;

        let ffn_norm = context.new_value();
        let ffn_norm_weight = context.weight(&format!("{prefix}.ffn_norm.weight"));
        context.add_node(
            Some("ffn_norm"),
            "RMS_NORM",
            vec![prev, ffn_norm_weight],
            Some(ffn_norm),
        );

        let ffn_out = lower_mistral4_moe_ffn(context, &prefix, ffn_norm);
        let residual = context.new_value();
        context.add_node(None, "ADD", vec![prev, ffn_out], Some(residual));
        prev = residual;
    }

    if context.tensors.contains_key("output_norm.weight") {
        let output_norm = context.new_value();
        let weight = context.weight("output_norm.weight");
        context.add_node(
            Some("output_norm"),
            "RMS_NORM",
            vec![prev, weight],
            Some(output_norm),
        );
        prev = output_norm;
    }
    if context.tensors.contains_key("output.weight") {
        let output = context.new_value();
        let weight = context.weight("output.weight");
        context.add_node(Some("output"), "MUL_MAT", vec![prev, weight], Some(output));
        prev = output;
    }
    if has_tokenizer {
        let out = context.new_value();
        context.add_node(None, "tokenizer", vec![prev], Some(out));
    }
}

fn lower_t5(context: &mut LowerContext<'_>, block_count: usize, has_tokenizer: bool) {
    let mut prev = if has_tokenizer {
        let value = context.new_value();
        context.add_node(None, "tokenizer", Vec::new(), Some(value));
        value
    } else {
        context.new_value()
    };

    if context.tensors.contains_key("token_embd.weight") {
        let out = context.new_value();
        let weight = context.weight("token_embd.weight");
        context.add_node(
            Some("token_embd"),
            "EMBEDDING",
            vec![prev, weight],
            Some(out),
        );
        prev = out;
    }

    for block in 0..block_count {
        let prefix = format!("enc.blk.{block}");
        if !context
            .tensors
            .contains_key(&format!("{prefix}.attn_norm.weight"))
        {
            continue;
        }

        let attn_norm = context.new_value();
        let attn_norm_weight = context.weight(&format!("{prefix}.attn_norm.weight"));
        context.add_node(
            Some("attn_norm"),
            "RMS_NORM",
            vec![prev, attn_norm_weight],
            Some(attn_norm),
        );

        let attention = context.new_value();
        let mut attention_inputs = vec![attn_norm];
        attention_inputs.extend(t5_self_attention_weights(context, &prefix));
        context.add_node(
            Some("attention"),
            "MULTI_HEAD_ATTENTION",
            attention_inputs,
            Some(attention),
        );

        let attention_residual = context.new_value();
        context.add_node(None, "ADD", vec![prev, attention], Some(attention_residual));
        prev = attention_residual;

        let ffn_norm = context.new_value();
        let ffn_norm_weight = context.weight(&format!("{prefix}.ffn_norm.weight"));
        context.add_node(
            Some("ffn_norm"),
            "RMS_NORM",
            vec![prev, ffn_norm_weight],
            Some(ffn_norm),
        );

        let ffn_out = lower_dense_gated_ffn(context, &prefix, ffn_norm);
        let residual = context.new_value();
        context.add_node(None, "ADD", vec![prev, ffn_out], Some(residual));
        prev = residual;
    }

    if context.tensors.contains_key("enc.output_norm.weight") {
        let output_norm = context.new_value();
        let weight = context.weight("enc.output_norm.weight");
        context.add_node(
            Some("enc.output_norm"),
            "RMS_NORM",
            vec![prev, weight],
            Some(output_norm),
        );
        prev = output_norm;
    }

    for block in 0..block_count {
        let prefix = format!("dec.blk.{block}");
        if !context
            .tensors
            .contains_key(&format!("{prefix}.attn_norm.weight"))
        {
            continue;
        }

        let self_attention = context.new_value();
        let mut self_inputs = vec![prev];
        self_inputs.extend(t5_self_attention_weights(context, &prefix));
        context.add_node(
            Some("self_attention"),
            "MULTI_HEAD_ATTENTION",
            self_inputs,
            Some(self_attention),
        );

        let attn_norm = context.new_value();
        let attn_norm_weight = context.weight(&format!("{prefix}.attn_norm.weight"));
        context.add_node(
            Some("attn_norm"),
            "RMS_NORM",
            vec![self_attention, attn_norm_weight],
            Some(attn_norm),
        );

        let cross_attention = context.new_value();
        let mut cross_inputs = vec![attn_norm];
        cross_inputs.extend(t5_cross_attention_weights(context, &prefix));
        context.add_node(
            Some("cross_attention"),
            "CROSS_ATTENTION",
            cross_inputs,
            Some(cross_attention),
        );

        let cross_attn_norm = context.new_value();
        let cross_attn_norm_weight = context.weight(&format!("{prefix}.cross_attn_norm.weight"));
        context.add_node(
            Some("cross_attn_norm"),
            "RMS_NORM",
            vec![cross_attention, cross_attn_norm_weight],
            Some(cross_attn_norm),
        );

        let down = context.new_value();
        let down_weight = context.weight(&format!("{prefix}.ffn_down.weight"));
        context.add_node(
            Some("ffn_down"),
            "MUL_MAT",
            vec![cross_attn_norm, down_weight],
            Some(down),
        );

        let gate = context.new_value();
        let gate_weight = context.weight(&format!("{prefix}.ffn_gate.weight"));
        context.add_node(
            Some("ffn_gate"),
            "MUL_MAT",
            vec![down, gate_weight],
            Some(gate),
        );

        let ffn_norm = context.new_value();
        let ffn_norm_weight = context.weight(&format!("{prefix}.ffn_norm.weight"));
        context.add_node(
            Some("ffn_norm"),
            "RMS_NORM",
            vec![gate, ffn_norm_weight],
            Some(ffn_norm),
        );

        let up = context.new_value();
        let up_weight = context.weight(&format!("{prefix}.ffn_up.weight"));
        context.add_node(
            Some("ffn_up"),
            "MUL_MAT",
            vec![ffn_norm, up_weight],
            Some(up),
        );
        prev = up;
    }

    if context.tensors.contains_key("dec.output_norm.weight") {
        let output_norm = context.new_value();
        let weight = context.weight("dec.output_norm.weight");
        context.add_node(
            Some("dec.output_norm"),
            "RMS_NORM",
            vec![prev, weight],
            Some(output_norm),
        );
        prev = output_norm;
    }
    if context.tensors.contains_key("output.weight") {
        let output = context.new_value();
        let weight = context.weight("output.weight");
        context.add_node(Some("output"), "MUL_MAT", vec![prev, weight], Some(output));
        prev = output;
    }
    if has_tokenizer {
        let out = context.new_value();
        context.add_node(None, "tokenizer", vec![prev], Some(out));
    }
}

fn t5_self_attention_weights(context: &mut LowerContext<'_>, prefix: &str) -> Vec<ValueId> {
    let mut names = vec![
        format!("{prefix}.attn_k.weight"),
        format!("{prefix}.attn_o.weight"),
        format!("{prefix}.attn_q.weight"),
    ];
    let rel_bias = format!("{prefix}.attn_rel_b.weight");
    if context.tensors.contains_key(&rel_bias) {
        names.push(rel_bias);
    }
    names.push(format!("{prefix}.attn_v.weight"));
    names.iter().map(|name| context.weight(name)).collect()
}

fn t5_cross_attention_weights(context: &mut LowerContext<'_>, prefix: &str) -> Vec<ValueId> {
    [
        "cross_attn_k",
        "cross_attn_o",
        "cross_attn_q",
        "cross_attn_v",
    ]
    .iter()
    .map(|name| context.weight(&format!("{prefix}.{name}.weight")))
    .collect()
}

fn mistral4_attention_weights(context: &mut LowerContext<'_>, prefix: &str) -> Vec<ValueId> {
    [
        "attn_k_b",
        "attn_kv_a_mqa",
        "attn_kv_a_norm",
        "attn_output",
        "attn_q_a",
        "attn_q_a_norm",
        "attn_q_b",
        "attn_v_b",
    ]
    .iter()
    .map(|name| context.weight(&format!("{prefix}.{name}.weight")))
    .collect()
}

fn combine_attention_and_ssm(
    context: &mut LowerContext<'_>,
    prefix: &str,
    input: ValueId,
    attention: Option<ValueId>,
) -> Option<ValueId> {
    let Some((operator, weights)) = ssm_operator_weights(context, prefix) else {
        return attention;
    };
    let ssm = context.new_value();
    let mut inputs = vec![input];
    inputs.extend(weights);
    context.add_node(Some("ssm"), operator, inputs, Some(ssm));

    if let Some(attention) = attention {
        let sum = context.new_value();
        context.add_node(None, "ADD", vec![ssm, attention], Some(sum));
        Some(sum)
    } else {
        Some(ssm)
    }
}

fn ssm_operator_weights(
    context: &mut LowerContext<'_>,
    prefix: &str,
) -> Option<(&'static str, Vec<ValueId>)> {
    let mamba = [
        "ssm_a",
        "ssm_d",
        "ssm_conv1d.bias",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_dt.weight",
        "ssm_in.weight",
        "ssm_out.weight",
        "ssm_x.weight",
    ];
    if mamba
        .iter()
        .all(|name| context.tensors.contains_key(&format!("{prefix}.{name}")))
    {
        let weights = mamba
            .iter()
            .map(|name| context.weight(&format!("{prefix}.{name}")))
            .collect();
        return Some(("MAMBA", weights));
    }

    let mamba2 = [
        "ssm_a",
        "ssm_conv1d.bias",
        "ssm_conv1d.weight",
        "ssm_d",
        "ssm_dt.bias",
        "ssm_in.weight",
        "ssm_out.weight",
    ];
    if mamba2
        .iter()
        .all(|name| context.tensors.contains_key(&format!("{prefix}.{name}")))
    {
        let weights = mamba2
            .iter()
            .map(|name| context.weight(&format!("{prefix}.{name}")))
            .collect();
        return Some(("MAMBA2", weights));
    }
    None
}

fn norm_weight_name(context: &LowerContext<'_>, prefix: &str, name: &str) -> Option<String> {
    let explicit = format!("{prefix}.{name}.weight");
    if context.tensors.contains_key(&explicit) {
        return Some(explicit);
    }
    let bare = format!("{prefix}.{name}");
    context.tensors.contains_key(&bare).then_some(bare)
}

fn attention_weights(context: &mut LowerContext<'_>, prefix: &str) -> Vec<ValueId> {
    let bias_order = [
        "attn_k.bias",
        "attn_k.weight",
        "attn_output.bias",
        "attn_output.weight",
        "attn_q.bias",
        "attn_q.weight",
        "attn_sinks.weight",
        "attn_v.bias",
        "attn_v.weight",
    ];
    let bias_markers = [
        "attn_k.bias",
        "attn_output.bias",
        "attn_q.bias",
        "attn_sinks.weight",
        "attn_v.bias",
    ];
    if bias_markers
        .iter()
        .any(|name| context.tensors.contains_key(&format!("{prefix}.{name}")))
    {
        return bias_order
            .iter()
            .filter_map(|name| {
                let name = format!("{prefix}.{name}");
                context
                    .tensors
                    .contains_key(&name)
                    .then(|| context.weight(&name))
            })
            .collect();
    }

    let deepseek_order = ["attn_k", "attn_output", "attn_q", "attn_v"];
    if deepseek_order.iter().all(|name| {
        context
            .tensors
            .contains_key(&format!("{prefix}.{name}.weight"))
    }) {
        let names = deepseek_order
            .iter()
            .map(|name| format!("{prefix}.{name}.weight"))
            .collect::<Vec<_>>();
        return ordered_weights(context, names);
    }

    let names = ["attn_q", "attn_k", "attn_v", "attn_output"]
        .iter()
        .filter_map(|name| {
            let weight = format!("{prefix}.{name}.weight");
            context.tensors.contains_key(&weight).then_some(weight)
        })
        .collect::<Vec<_>>();
    ordered_weights(context, names)
}

fn attention_norm_weights(context: &mut LowerContext<'_>, prefix: &str) -> Vec<ValueId> {
    let names = [
        format!("{prefix}.attn_norm.weight"),
        format!("{prefix}.attn_k_norm.weight"),
        format!("{prefix}.attn_q_norm.weight"),
    ]
    .into_iter()
    .filter(|name| context.tensors.contains_key(name))
    .collect::<Vec<_>>();
    ordered_weights(context, names)
}

fn ordered_weights(context: &mut LowerContext<'_>, mut names: Vec<String>) -> Vec<ValueId> {
    names.sort_by_key(|name| context.tensors.get(name).map(|tensor| tensor.order));
    names.iter().map(|name| context.weight(name)).collect()
}

fn lower_dense_gated_ffn(context: &mut LowerContext<'_>, prefix: &str, input: ValueId) -> ValueId {
    let gate = context.new_value();
    let gate_weight = context.weight(&format!("{prefix}.ffn_gate.weight"));
    context.add_node(
        Some("ffn_gate"),
        "MUL_MAT",
        vec![input, gate_weight],
        Some(gate),
    );

    let up = context.new_value();
    let up_weight = context.weight(&format!("{prefix}.ffn_up.weight"));
    context.add_node(Some("ffn_up"), "MUL_MAT", vec![input, up_weight], Some(up));

    let down = context.new_value();
    let down_weight = context.weight(&format!("{prefix}.ffn_down.weight"));
    context.add_node(
        Some("ffn_down"),
        "MUL_MAT",
        vec![up, gate, input, down_weight],
        Some(down),
    );
    down
}

fn lower_moe_ffn(context: &mut LowerContext<'_>, prefix: &str, input: ValueId) -> ValueId {
    let gate_inp = context.new_value();
    let mut inputs = vec![input];
    inputs.extend(optional_bias_then_weight(context, prefix, "ffn_gate_inp"));
    context.add_node(Some("ffn_gate_inp"), "MUL_MAT", inputs, Some(gate_inp));

    let gate_exps = context.new_value();
    let mut inputs = vec![gate_inp];
    inputs.extend(moe_expert_weights(
        context,
        prefix,
        "ffn_gate_exps",
        "ffn_gate",
    ));
    context.add_node(Some("ffn_gate_exps"), "MUL_MAT_ID", inputs, Some(gate_exps));

    let up_exps = context.new_value();
    let mut inputs = vec![gate_inp];
    inputs.extend(moe_expert_weights(context, prefix, "ffn_up_exps", "ffn_up"));
    context.add_node(Some("ffn_up_exps"), "MUL_MAT_ID", inputs, Some(up_exps));

    let down_exps = context.new_value();
    let mut inputs = vec![up_exps, gate_exps, gate_inp];
    inputs.extend(moe_expert_weights(
        context,
        prefix,
        "ffn_down_exps",
        "ffn_down",
    ));
    context.add_node(Some("ffn_down_exps"), "MUL_MAT_ID", inputs, Some(down_exps));

    if !context
        .tensors
        .contains_key(&format!("{prefix}.ffn_gate_shexp.weight"))
    {
        return down_exps;
    }

    let gate_shexp = context.new_value();
    let gate_shexp_weight = context.weight(&format!("{prefix}.ffn_gate_shexp.weight"));
    context.add_node(
        Some("ffn_gate_shexp"),
        "MUL_MAT",
        vec![input, gate_shexp_weight],
        Some(gate_shexp),
    );

    let up_shexp = context.new_value();
    let up_shexp_weight = context.weight(&format!("{prefix}.ffn_up_shexp.weight"));
    context.add_node(
        Some("ffn_up_shexp"),
        "MUL_MAT",
        vec![input, up_shexp_weight],
        Some(up_shexp),
    );

    let down_shexp = context.new_value();
    let down_shexp_weight = context.weight(&format!("{prefix}.ffn_down_shexp.weight"));
    context.add_node(
        Some("ffn_down_shexp"),
        "MUL_MAT",
        vec![up_shexp, gate_shexp, input, down_shexp_weight],
        Some(down_shexp),
    );

    let sum = context.new_value();
    context.add_node(None, "ADD", vec![down_shexp, down_exps], Some(sum));
    sum
}

fn lower_mistral4_moe_ffn(context: &mut LowerContext<'_>, prefix: &str, input: ValueId) -> ValueId {
    let gate_inp = context.new_value();
    let gate_inp_weight = context.weight(&format!("{prefix}.ffn_gate_inp.weight"));
    context.add_node(
        Some("ffn_gate_inp"),
        "MUL_MAT",
        vec![input, gate_inp_weight],
        Some(gate_inp),
    );

    let gate_up_exps = context.new_value();
    let gate_up_exps_weight = context.weight(&format!("{prefix}.ffn_gate_up_exps.weight"));
    context.add_node(
        Some("ffn_gate_up_exps"),
        "MUL_MAT_ID",
        vec![gate_inp, gate_up_exps_weight],
        Some(gate_up_exps),
    );

    let down_exps = context.new_value();
    let down_exps_weight = context.weight(&format!("{prefix}.ffn_down_exps.weight"));
    context.add_node(
        Some("ffn_down_exps"),
        "MUL_MAT_ID",
        vec![gate_up_exps, down_exps_weight],
        Some(down_exps),
    );

    let gate_shexp = context.new_value();
    let gate_shexp_weight = context.weight(&format!("{prefix}.ffn_gate_shexp.weight"));
    context.add_node(
        Some("ffn_gate_shexp"),
        "MUL_MAT",
        vec![input, gate_shexp_weight],
        Some(gate_shexp),
    );

    let up_shexp = context.new_value();
    let up_shexp_weight = context.weight(&format!("{prefix}.ffn_up_shexp.weight"));
    context.add_node(
        Some("ffn_up_shexp"),
        "MUL_MAT",
        vec![input, up_shexp_weight],
        Some(up_shexp),
    );

    let down_shexp = context.new_value();
    let down_shexp_weight = context.weight(&format!("{prefix}.ffn_down_shexp.weight"));
    context.add_node(
        Some("ffn_down_shexp"),
        "MUL_MAT",
        vec![up_shexp, gate_shexp, input, down_shexp_weight],
        Some(down_shexp),
    );

    let sum = context.new_value();
    context.add_node(None, "ADD", vec![down_shexp, down_exps], Some(sum));
    sum
}

fn optional_bias_then_weight(
    context: &mut LowerContext<'_>,
    prefix: &str,
    name: &str,
) -> Vec<ValueId> {
    let mut values = Vec::new();
    let bias = format!("{prefix}.{name}.bias");
    if context.tensors.contains_key(&bias) {
        values.push(context.weight(&bias));
    }
    values.push(context.weight(&format!("{prefix}.{name}.weight")));
    values
}

fn moe_expert_weights(
    context: &mut LowerContext<'_>,
    prefix: &str,
    aggregate: &str,
    split: &str,
) -> Vec<ValueId> {
    let aggregate_weight = format!("{prefix}.{aggregate}.weight");
    if context.tensors.contains_key(&aggregate_weight) {
        return optional_bias_then_weight(context, prefix, aggregate);
    }

    let split_prefix = format!("{prefix}.{split}.");
    let names = context
        .tensors
        .keys()
        .filter(|name| name.starts_with(&split_prefix) && name.ends_with(".weight"))
        .cloned()
        .collect::<Vec<_>>();
    ordered_weights(context, names)
}

fn has_pair(context: &LowerContext<'_>, prefix: &str, name: &str) -> bool {
    context
        .tensors
        .contains_key(&format!("{prefix}.{name}.weight"))
        && context
            .tensors
            .contains_key(&format!("{prefix}.{name}.bias"))
}

fn lower_weights_fallback(context: &mut LowerContext<'_>) {
    let groups = context
        .tensors
        .keys()
        .map(|name| {
            name.rsplit_once('.')
                .map(|(group, _)| group.to_owned())
                .unwrap_or_else(|| name.clone())
        })
        .fold(
            BTreeMap::<String, Vec<String>>::new(),
            |mut groups, group| {
                groups.entry(group).or_default();
                groups
            },
        );

    for (group, _) in groups {
        let names = context
            .tensors
            .keys()
            .filter(|name| {
                name.rsplit_once('.')
                    .map(|(candidate, _)| candidate == group)
                    .unwrap_or(name.as_str() == group)
            })
            .cloned()
            .collect::<Vec<_>>();
        let inputs = names.iter().map(|name| context.weight(name)).collect();
        context.add_node(Some(&group), "weights", inputs, None);
    }
}

#[derive(Default)]
struct Target {
    version: u32,
    metadata: BTreeMap<String, MetadataValue>,
    metadata_entries: Vec<(String, MetadataValue)>,
    tensors: BTreeMap<String, TensorInfo>,
    block_count: usize,
    has_tokenizer: bool,
}

#[derive(Clone)]
struct TensorInfo {
    element_type: TensorElementType,
    shape: Vec<Dimension>,
    byte_len: usize,
    quantization: Option<String>,
    order: usize,
}

#[derive(Clone)]
enum MetadataValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array {
        element_type: u32,
        values: Vec<MetadataValue>,
    },
}

impl MetadataValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_usize(&self) -> Option<usize> {
        match self {
            Self::U32(value) => Some(*value as usize),
            Self::U64(value) => usize::try_from(*value).ok(),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::U32(value) => Some(i64::from(*value)),
            Self::U64(value) => i64::try_from(*value).ok(),
            Self::I32(value) => Some(i64::from(*value)),
            Self::I64(value) => Some(*value),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(value) => Some(*value),
            Self::F64(value) => Some(*value as f32),
            _ => None,
        }
    }

    fn to_property(&self) -> Option<String> {
        match self {
            Self::U32(value) => Some(value.to_string()),
            Self::U64(value) => Some(value.to_string()),
            Self::I32(value) => Some(value.to_string()),
            Self::I64(value) => Some(value.to_string()),
            Self::F32(value) => Some(value.to_string()),
            Self::F64(value) => Some(value.to_string()),
            Self::Bool(value) => Some(value.to_string()),
            Self::String(value) => Some(value.clone()),
            Self::Array { values, .. } => Some(
                serde_json::Value::Array(values.iter().map(MetadataValue::to_json).collect())
                    .to_string(),
            ),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::U32(value) => serde_json::Value::Number(serde_json::Number::from(*value)),
            Self::U64(value) => serde_json::Value::Number(serde_json::Number::from(*value)),
            Self::I32(value) => serde_json::Value::Number(serde_json::Number::from(*value)),
            Self::I64(value) => serde_json::Value::Number(serde_json::Number::from(*value)),
            Self::F32(value) => serde_json::Number::from_f64(f64::from(*value))
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(value.to_string())),
            Self::F64(value) => serde_json::Number::from_f64(*value)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(value.to_string())),
            Self::Bool(value) => serde_json::Value::Bool(*value),
            Self::String(value) => serde_json::Value::String(value.clone()),
            Self::Array { values, .. } => {
                serde_json::Value::Array(values.iter().map(MetadataValue::to_json).collect())
            }
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn read(mut self) -> Result<Target, ModelError> {
        if self.take(4)? != b"GGUF" {
            return Err(invalid("missing GGUF signature"));
        }
        let version = self.u32()?;
        let tensor_count = self.u64()? as usize;
        let metadata_count = self.u64()? as usize;
        let mut metadata = BTreeMap::new();
        let mut metadata_entries = Vec::with_capacity(metadata_count);
        let mut has_tokenizer = false;
        for _ in 0..metadata_count {
            let name = self.string()?;
            let value_type = self.u32()?;
            if name.starts_with("tokenizer.") {
                has_tokenizer = true;
            }
            let value = if value_type == GgufType::ARRAY {
                self.array()?
            } else {
                self.scalar(value_type)?
            };
            metadata_entries.push((name.clone(), value.clone()));
            metadata.insert(name, value);
        }

        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for order in 0..tensor_count {
            let name = self.string()?;
            let dims = self.u32()? as usize;
            let mut shape = Vec::with_capacity(dims);
            let mut element_count = 1_u128;
            for _ in 0..dims {
                let dim = self.u64()?;
                element_count = element_count.saturating_mul(dim as u128);
                shape.push(Dimension::known(dim as i64));
            }
            let tensor_type = self.u32()?;
            let offset = self.u64()?;
            raw_tensors.push(RawTensor {
                name,
                shape,
                tensor_type,
                offset,
                element_count,
                order,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetadataValue::as_usize)
            .unwrap_or(32);
        if alignment > 0 && !self.position.is_multiple_of(alignment) {
            self.position += alignment - (self.position % alignment);
        }
        let data_start = self.position as u64;

        let mut tensors = BTreeMap::new();
        for tensor in raw_tensors {
            let quantization = quantization_type(tensor.tensor_type).ok_or_else(|| {
                invalid(format!(
                    "unsupported tensor quantization type '{}'",
                    tensor.tensor_type
                ))
            })?;
            let byte_len = ((tensor.element_count * quantization.type_size as u128)
                / quantization.block_size as u128) as usize;
            let _payload_offset = data_start.saturating_add(tensor.offset);
            tensors.insert(
                tensor.name,
                TensorInfo {
                    element_type: lower_tensor_type(quantization.name),
                    shape: tensor.shape,
                    byte_len,
                    quantization: (quantization.block_size > 1)
                        .then(|| quantization.name.to_ascii_lowercase()),
                    order: tensor.order,
                },
            );
        }
        let architecture = metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            .unwrap_or("?");
        let block_count = metadata
            .get(&format!("{architecture}.block_count"))
            .and_then(MetadataValue::as_usize)
            .unwrap_or_default();
        Ok(Target {
            version,
            metadata,
            metadata_entries,
            tensors,
            block_count,
            has_tokenizer,
        })
    }

    fn scalar(&mut self, value_type: u32) -> Result<MetadataValue, ModelError> {
        match value_type {
            GgufType::UINT8 => Ok(MetadataValue::U32(u32::from(self.u8()?))),
            GgufType::INT8 => Ok(MetadataValue::I32(i32::from(self.i8()?))),
            GgufType::UINT16 => Ok(MetadataValue::U32(u32::from(self.u16()?))),
            GgufType::INT16 => Ok(MetadataValue::I32(i32::from(self.i16()?))),
            GgufType::UINT32 => Ok(MetadataValue::U32(self.u32()?)),
            GgufType::INT32 => Ok(MetadataValue::I32(self.i32()?)),
            GgufType::FLOAT32 => Ok(MetadataValue::F32(self.f32()?)),
            GgufType::BOOL => Ok(MetadataValue::Bool(self.u8()? != 0)),
            GgufType::STRING => Ok(MetadataValue::String(self.string()?)),
            GgufType::UINT64 => Ok(MetadataValue::U64(self.u64()?)),
            GgufType::INT64 => Ok(MetadataValue::I64(self.i64()?)),
            GgufType::FLOAT64 => Ok(MetadataValue::F64(self.f64()?)),
            _ => Err(invalid(format!("unsupported metadata type '{value_type}'"))),
        }
    }

    fn array(&mut self) -> Result<MetadataValue, ModelError> {
        let element_type = self.u32()?;
        let count = self.u64()? as usize;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.scalar(element_type)?);
        }
        Ok(MetadataValue::Array {
            element_type,
            values,
        })
    }

    fn string(&mut self) -> Result<String, ModelError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(bytes.iter().map(|byte| char::from(*byte)).collect())
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| invalid("offset overflow"))?;
        let bytes = self
            .data
            .get(self.position..end)
            .ok_or_else(|| invalid("unexpected end of file"))?;
        self.position = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, ModelError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, ModelError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn i16(&mut self) -> Result<i16, ModelError> {
        let bytes = self.take(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, ModelError> {
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64, ModelError> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn f32(&mut self) -> Result<f32, ModelError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, ModelError> {
        Ok(f64::from_bits(self.u64()?))
    }
}

struct RawTensor {
    name: String,
    shape: Vec<Dimension>,
    tensor_type: u32,
    offset: u64,
    element_count: u128,
    order: usize,
}

struct QuantizationType {
    name: &'static str,
    block_size: usize,
    type_size: usize,
}

fn quantization_type(value: u32) -> Option<QuantizationType> {
    let (name, block_size, type_size) = match value {
        0 => ("float32", 1, 4),
        1 => ("float16", 1, 2),
        2 => ("q4_0", 32, 2 + 16),
        3 => ("q4_1", 32, 2 + 2 + 16),
        6 => ("q5_0", 32, 2 + 4 + 16),
        7 => ("q5_1", 32, 2 + 2 + 4 + 16),
        8 => ("q8_0", 32, 2 + 32),
        9 => ("q8_1", 32, 4 + 4 + 32),
        10 => ("q2_K", 256, 2 + 2 + 16 + 64),
        11 => ("q3_K", 256, 2 + 64 + 32 + 12),
        12 => ("q4_K", 256, 2 + 2 + 128 + 12),
        13 => ("q5_K", 256, 2 + 2 + 128 + 32 + 12),
        14 => ("q6_K", 256, 2 + 128 + 64 + 16),
        15 => ("q8_K", 256, 4 + 256 + 32),
        16 => ("iq2_xxs", 256, 2 + 64),
        17 => ("iq2_xs", 256, 2 + 64 + 8),
        18 => ("iq3_xxs", 256, 2 + 64 + 32),
        19 => ("iq1_s", 256, 2 + 32 + 16),
        20 => ("iq4_nl", 32, 2 + 16),
        21 => ("iq3_s", 256, 2 + 64 + 32 + 8 + 4),
        22 => ("iq2_s", 256, 2 + 64 + 16),
        23 => ("iq4_xs", 256, 2 + 2 + 128 + 4),
        24 => ("int8", 1, 1),
        25 => ("int16", 1, 2),
        26 => ("int32", 1, 4),
        27 => ("int64", 1, 8),
        28 => ("float64", 1, 8),
        29 => ("iq1_m", 256, 32 + 16 + 8),
        30 => ("bfloat16", 1, 2),
        39 => ("mxfp4", 32, 1 + 16),
        40 => ("nvfp4", 64, 4 + 32),
        41 => ("q1_0", 128, 2 + 16),
        _ => return None,
    };
    Some(QuantizationType {
        name,
        block_size,
        type_size,
    })
}

fn lower_tensor_type(name: &str) -> TensorElementType {
    match name {
        "float16" => TensorElementType::Float16,
        "float32" => TensorElementType::Float32,
        "float64" => TensorElementType::Float64,
        "bfloat16" => TensorElementType::BFloat16,
        "int8" => TensorElementType::Int8,
        "int16" => TensorElementType::Int16,
        "int32" => TensorElementType::Int32,
        "int64" => TensorElementType::Int64,
        value => TensorElementType::Other(value.to_owned()),
    }
}

struct GgufType;

impl GgufType {
    const UINT8: u32 = 0;
    const INT8: u32 = 1;
    const UINT16: u32 = 2;
    const INT16: u32 = 3;
    const UINT32: u32 = 4;
    const INT32: u32 = 5;
    const FLOAT32: u32 = 6;
    const BOOL: u32 = 7;
    const STRING: u32 = 8;
    const ARRAY: u32 = 9;
    const UINT64: u32 = 10;
    const INT64: u32 = 11;
    const FLOAT64: u32 = 12;
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
