use std::collections::{BTreeMap, HashMap};

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, FormatInfo, FormatMetadata, Graph, Model,
    ModelError, ModelFormat, ModelInput, Node, Operator, QuantizationAnnotation, Tensor,
    TensorElementType, TensorStorage, TypeInfo, Value, ValueId,
};

const FORMAT: &str = "TensorFlow Lite";

pub struct TfliteFormat;

impl ModelFormat for TfliteFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["tflite"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if input.data.len() >= 8 && input.data.get(4..8) == Some(b"TFL3") {
            return Confidence::High;
        }
        if input
            .path
            .and_then(|path| path.extension())
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("tflite"))
            && RootModel::read(input.data).is_ok_and(|model| model.version == 3)
        {
            return Confidence::Medium;
        }
        Confidence::None
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        let target = RootModel::read(input.data)?;
        lower_model(target)
    }
}

fn lower_model(target: RootModel) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(target.version.to_string()),
    });
    model.metadata.description = combined_description(
        target.description.filter(|value| !value.is_empty()),
        target.metadata.description.clone(),
    );
    model.metadata.producer_version = target.metadata.version.clone();
    model.metadata.properties = target.metadata.properties.clone();

    let operators = target
        .operator_codes
        .iter()
        .map(operator_code)
        .collect::<Vec<_>>();
    let graph_count = target.subgraphs.len();
    for (index, subgraph) in target.subgraphs.into_iter().enumerate() {
        let signatures = target
            .signatures
            .iter()
            .filter(|signature| signature.subgraph_index == index as u32)
            .collect::<Vec<_>>();
        let signature_inputs = signature_boundary_indexes(&signatures, true);
        let signature_outputs = signature_boundary_indexes(&signatures, false);
        let graph_inputs = if signature_inputs.is_empty() {
            subgraph.inputs.clone()
        } else {
            signature_inputs
        };
        let graph_outputs = if signature_outputs.is_empty() {
            subgraph.outputs.clone()
        } else {
            signature_outputs
        };
        let fallback_name = (graph_count > 1).then(|| index.to_string());
        let graph_name = subgraph
            .name
            .clone()
            .filter(|value| !value.is_empty())
            .or(fallback_name);
        let graph_name_id = graph_name.as_ref().map(|name| model.intern(name));
        let graph_id = model.add_graph_placeholder(None, graph_name_id);
        let mut graph = Graph::new(graph_id, None, graph_name_id);
        let mut context = LowerContext {
            model: &mut model,
            graph: &mut graph,
            graph_id,
            tensors: &subgraph.tensors,
            buffers: &target.buffers,
            tensor_metadata: boundary_tensor_metadata(
                &subgraph,
                target.metadata.subgraphs.get(index),
            ),
            values: HashMap::new(),
        };

        for tensor_index in subgraph.inputs.iter().chain(subgraph.outputs.iter()) {
            context.ensure_tensor_value(*tensor_index);
        }
        for index in graph_inputs {
            if let Some(value_id) = context.ensure_tensor_value(index)
                && !context.graph.inputs.contains(&value_id)
            {
                context.graph.values[value_id.index()].is_graph_input = true;
                context.graph.inputs.push(value_id);
            }
        }
        for index in graph_outputs {
            if let Some(value_id) = context.ensure_tensor_value(index)
                && !context.graph.outputs.contains(&value_id)
            {
                context.graph.values[value_id.index()].is_graph_output = true;
                context.graph.outputs.push(value_id);
            }
        }
        for operator in subgraph.operators {
            let op_info = operators
                .get(operator.opcode_index as usize)
                .cloned()
                .unwrap_or_else(|| OperatorInfo {
                    name: format!("({})", operator.opcode_index),
                    version: None,
                    custom: false,
                });
            context.add_operator(operator, op_info);
        }

        model.replace_graph(graph_id, graph);
    }
    Ok(model)
}

#[derive(Clone)]
struct OperatorInfo {
    name: String,
    version: Option<i64>,
    custom: bool,
}

fn operator_code(code: &OperatorCode) -> OperatorInfo {
    let builtin_code = i32::from(code.deprecated_builtin_code).max(code.builtin_code);
    if builtin_code == 32 {
        return OperatorInfo {
            name: code
                .custom_code
                .clone()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "Custom".to_owned()),
            version: Some(i64::from(code.version)),
            custom: true,
        };
    }
    OperatorInfo {
        name: builtin_operator_name(builtin_code)
            .map(display_builtin_operator_name)
            .unwrap_or_else(|| builtin_code.to_string()),
        version: Some(i64::from(code.version)),
        custom: false,
    }
}

fn display_builtin_operator_name(name: &str) -> String {
    let name = if name == "BATCH_MATMUL" {
        "BATCH_MAT_MUL"
    } else {
        name
    };
    let uppercase = ["2D", "LSH", "SVDF", "RNN", "L2", "LSTM"];
    name.split('_')
        .map(|part| {
            if part.is_empty() || uppercase.contains(&part) {
                part.to_owned()
            } else {
                let mut chars = part.chars();
                let Some(first) = chars.next() else {
                    return String::new();
                };
                format!(
                    "{}{}",
                    first.to_ascii_uppercase(),
                    chars.as_str().to_ascii_lowercase()
                )
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn combined_description(
    base: Option<String>,
    metadata_description: Option<String>,
) -> Option<String> {
    match (base, metadata_description.filter(|value| !value.is_empty())) {
        (Some(base), Some(metadata)) => Some(format!("{base} {metadata}")),
        (Some(base), None) => Some(base),
        (None, Some(metadata)) => Some(metadata),
        (None, None) => None,
    }
}

fn signature_boundary_indexes(signatures: &[&SignatureDefInfo], inputs: bool) -> Vec<i32> {
    let mut result = Vec::new();
    for signature in signatures {
        let maps = if inputs {
            &signature.inputs
        } else {
            &signature.outputs
        };
        for map in maps {
            let index = map.tensor_index as i32;
            if !result.contains(&index) {
                result.push(index);
            }
        }
    }
    result
}

fn boundary_tensor_metadata(
    subgraph: &SubgraphInfo,
    metadata: Option<&SubgraphMetadataInfo>,
) -> HashMap<i32, TensorMetadataInfo> {
    let mut result = HashMap::new();
    let Some(metadata) = metadata else {
        return result;
    };
    for (position, tensor_index) in subgraph.inputs.iter().enumerate() {
        if let Some(tensor_metadata) = metadata.inputs.get(position) {
            result.insert(*tensor_index, tensor_metadata.clone());
        }
    }
    for (position, tensor_index) in subgraph.outputs.iter().enumerate() {
        if let Some(tensor_metadata) = metadata.outputs.get(position) {
            result.insert(*tensor_index, tensor_metadata.clone());
        }
    }
    result
}

struct LowerContext<'a> {
    model: &'a mut Model,
    graph: &'a mut Graph,
    graph_id: netron_rs_core::GraphId,
    tensors: &'a [TensorInfo],
    buffers: &'a [BufferInfo],
    tensor_metadata: HashMap<i32, TensorMetadataInfo>,
    values: HashMap<i32, ValueId>,
}

impl LowerContext<'_> {
    fn ensure_tensor_value(&mut self, index: i32) -> Option<ValueId> {
        if index < 0 {
            return None;
        }
        if let Some(value_id) = self.values.get(&index) {
            return Some(*value_id);
        }
        let Some(tensor) = self.tensors.get(index as usize) else {
            let value_name = format!("\n{index}");
            let value_name_id = self.model.intern(&value_name);
            let value_id = self.graph.add_value(Value::new(value_name_id));
            self.values.insert(index, value_id);
            return Some(value_id);
        };
        let value_name = format!("{}\n{index}", tensor.name);
        let value_name_id = self.model.intern(&value_name);
        let mut value = Value::new(value_name_id);
        let tensor_metadata = self.tensor_metadata.get(&index).cloned();
        if let Some(metadata) = tensor_metadata.as_ref() {
            value.description = metadata.description.clone();
        }
        let denotation = tensor_metadata
            .as_ref()
            .and_then(|metadata| metadata.denotation.as_ref())
            .map(|denotation| self.model.intern(denotation));
        value.type_info = Some(TypeInfo {
            element_type: Some(lower_tensor_type(tensor.element_type)),
            layout: None,
            denotation,
            shape: tensor
                .shape_signature
                .as_ref()
                .filter(|shape| !shape.is_empty())
                .unwrap_or(&tensor.shape)
                .iter()
                .map(|dimension| Dimension::known(i64::from(*dimension)))
                .collect(),
        });
        if let Some(buffer) = self.buffers.get(tensor.buffer as usize)
            && (tensor.is_variable || buffer.byte_len > 0)
        {
            let tensor_name = self.model.intern(&tensor.name);
            let tensor_id = self.model.add_tensor(Tensor::metadata_only(
                Some(tensor_name),
                lower_tensor_type(tensor.element_type),
                tensor
                    .shape_signature
                    .as_ref()
                    .filter(|shape| !shape.is_empty())
                    .unwrap_or(&tensor.shape)
                    .iter()
                    .map(|dimension| Dimension::known(i64::from(*dimension)))
                    .collect(),
                if buffer.byte_len > 0 {
                    TensorStorage::InlineBytes {
                        byte_len: buffer.byte_len,
                    }
                } else {
                    TensorStorage::Absent
                },
            ));
            value.initializer = Some(tensor_id);
        }
        value.quantization = self.lower_quantization(&tensor.quantization);
        let value_id = self.graph.add_value(value);
        self.values.insert(index, value_id);
        Some(value_id)
    }

    fn lower_quantization(
        &mut self,
        quantization: &Option<QuantizationInfo>,
    ) -> Vec<QuantizationAnnotation> {
        let Some(quantization) = quantization else {
            return Vec::new();
        };
        if quantization.scale.is_empty()
            && quantization.zero_point.is_empty()
            && quantization.min.is_empty()
            && quantization.max.is_empty()
        {
            return Vec::new();
        }
        let mut entries = vec![
            ("type", "linear".to_owned()),
            ("dimension", quantization.quantized_dimension.to_string()),
        ];
        if !quantization.scale.is_empty() {
            entries.push(("scale", join_f32(&quantization.scale)));
        }
        if !quantization.zero_point.is_empty() {
            entries.push(("offset", join_i64(&quantization.zero_point)));
        }
        if !quantization.min.is_empty() {
            entries.push(("min", join_f32(&quantization.min)));
        }
        if !quantization.max.is_empty() {
            entries.push(("max", join_f32(&quantization.max)));
        }
        entries
            .into_iter()
            .map(|(key, value)| QuantizationAnnotation {
                key: self.model.intern(key),
                value: self.model.intern(value),
            })
            .collect()
    }

    fn add_operator(&mut self, operator: OperatorInfoNode, op_info: OperatorInfo) {
        let op_name = self.model.intern(&op_info.name);
        let mut node = Node::new(
            self.graph_id,
            Operator {
                domain: None,
                name: op_name,
                overload: None,
                version: op_info.version,
                origin: FORMAT,
            },
        );
        let mut inputs = Vec::new();
        for index in operator.inputs {
            if let Some(value_id) = self.ensure_tensor_value(index) {
                node.inputs.push(Some(value_id));
                inputs.push(value_id);
            }
        }
        let mut outputs = Vec::new();
        for index in operator.outputs {
            if let Some(value_id) = self.ensure_tensor_value(index) {
                node.outputs.push(Some(value_id));
                outputs.push(value_id);
            }
        }
        let use_fallback_options = operator.builtin_option_type == Some(1)
            && op_info.name != "Conv2D"
            && !operator.fallback_attributes.is_empty();
        let mut attributes = if use_fallback_options {
            operator.fallback_attributes
        } else {
            operator.attributes
        };
        if op_info.custom && !operator.custom_options.is_empty() {
            if operator.custom_options_format == 0 {
                match read_flexbuffer_attributes(&operator.custom_options) {
                    Some(decoded) => attributes.extend(decoded),
                    None => attributes.push(raw_custom_attribute(&operator.custom_options)),
                }
            } else {
                attributes.push(raw_custom_attribute(&operator.custom_options));
            }
        }
        node.attributes = attributes
            .into_iter()
            .filter(|attribute| attribute.name != "fused_activation_function")
            .map(|attribute| Attribute {
                name: self.model.intern(&attribute.name),
                value: match attribute.value {
                    ParsedAttributeValue::Bool(value) => AttributeValue::Bool(value),
                    ParsedAttributeValue::Float(value) => AttributeValue::Float(value),
                    ParsedAttributeValue::Int(value) => AttributeValue::Int(i64::from(value)),
                    ParsedAttributeValue::String(value) => {
                        AttributeValue::String(self.model.intern(&value))
                    }
                    ParsedAttributeValue::Ints(values) => {
                        AttributeValue::Ints(values.into_iter().map(i64::from).collect())
                    }
                    ParsedAttributeValue::Strings(values) => AttributeValue::Strings(
                        values
                            .into_iter()
                            .map(|value| self.model.intern(&value))
                            .collect(),
                    ),
                },
            })
            .collect();
        let node_id = self.graph.add_node(node);
        for input in inputs {
            let consumers = &mut self.graph.values[input.index()].consumers;
            if !consumers.contains(&node_id) {
                consumers.push(node_id);
            }
        }
        for output in outputs {
            self.graph.values[output.index()].producer = Some(node_id);
        }
    }
}

fn join_f32(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| f64::from(*value).to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn join_i64(values: &[i64]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn lower_tensor_type(value: i8) -> TensorElementType {
    match value {
        0 => TensorElementType::Float32,
        1 => TensorElementType::Float16,
        2 => TensorElementType::Int32,
        3 => TensorElementType::Uint8,
        4 => TensorElementType::Int64,
        5 => TensorElementType::String,
        6 => TensorElementType::Bool,
        7 => TensorElementType::Int16,
        8 => TensorElementType::Complex64,
        9 => TensorElementType::Int8,
        10 => TensorElementType::Float64,
        11 => TensorElementType::Complex128,
        12 => TensorElementType::Uint64,
        15 => TensorElementType::Uint32,
        16 => TensorElementType::Uint16,
        17 => TensorElementType::Int4,
        18 => TensorElementType::BFloat16,
        19 => TensorElementType::Int2,
        20 => TensorElementType::Uint4,
        value => TensorElementType::Other(value.to_string()),
    }
}

#[derive(Default)]
struct RootModel {
    version: u32,
    operator_codes: Vec<OperatorCode>,
    subgraphs: Vec<SubgraphInfo>,
    description: Option<String>,
    buffers: Vec<BufferInfo>,
    metadata: EmbeddedMetadata,
    signatures: Vec<SignatureDefInfo>,
}

impl RootModel {
    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let reader = FlatReader { data };
        let root = reader.root()?;
        let buffers = reader
            .table_vector(root, 12)?
            .into_iter()
            .map(|position| BufferInfo::read(&reader, position))
            .collect::<Result<Vec<_>, _>>()?;
        let metadata_entries = reader
            .table_vector(root, 16)?
            .into_iter()
            .map(|position| MetadataInfo::read(&reader, position))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            version: reader.u32_field(root, 4, 0),
            operator_codes: reader
                .table_vector(root, 6)?
                .into_iter()
                .map(|position| OperatorCode::read(&reader, position))
                .collect::<Result<_, _>>()?,
            subgraphs: reader
                .table_vector(root, 8)?
                .into_iter()
                .map(|position| SubgraphInfo::read(&reader, position))
                .collect::<Result<_, _>>()?,
            description: reader.string_field(root, 10)?,
            metadata: EmbeddedMetadata::read(&reader, &buffers, &metadata_entries),
            signatures: reader
                .table_vector(root, 18)?
                .into_iter()
                .map(|position| SignatureDefInfo::read(&reader, position))
                .collect::<Result<_, _>>()?,
            buffers,
        })
    }
}

#[derive(Default)]
struct OperatorCode {
    deprecated_builtin_code: i8,
    custom_code: Option<String>,
    version: i32,
    builtin_code: i32,
}

impl OperatorCode {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            deprecated_builtin_code: reader.i8_field(position, 4, 0),
            custom_code: reader.string_field(position, 6)?,
            version: reader.i32_field(position, 8, 1),
            builtin_code: reader.i32_field(position, 10, 0),
        })
    }
}

#[derive(Default)]
struct SubgraphInfo {
    tensors: Vec<TensorInfo>,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    operators: Vec<OperatorInfoNode>,
    name: Option<String>,
}

impl SubgraphInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            tensors: reader
                .table_vector(position, 4)?
                .into_iter()
                .map(|position| TensorInfo::read(reader, position))
                .collect::<Result<_, _>>()?,
            inputs: reader.i32_vector(position, 6)?,
            outputs: reader.i32_vector(position, 8)?,
            operators: reader
                .table_vector(position, 10)?
                .into_iter()
                .map(|position| OperatorInfoNode::read(reader, position))
                .collect::<Result<_, _>>()?,
            name: reader.string_field(position, 12)?,
        })
    }
}

#[derive(Default)]
struct TensorInfo {
    shape: Vec<i32>,
    element_type: i8,
    buffer: u32,
    name: String,
    quantization: Option<QuantizationInfo>,
    is_variable: bool,
    shape_signature: Option<Vec<i32>>,
}

impl TensorInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            shape: reader.i32_vector(position, 4)?,
            element_type: reader.i8_field(position, 6, 0),
            buffer: reader.u32_field(position, 8, 0),
            name: reader.string_field(position, 10)?.unwrap_or_default(),
            quantization: reader
                .table_field(position, 12)?
                .map(|position| QuantizationInfo::read(reader, position))
                .transpose()?,
            is_variable: reader.bool_field(position, 14, false),
            shape_signature: reader.i32_vector_opt(position, 18)?,
        })
    }
}

#[derive(Default)]
struct QuantizationInfo {
    min: Vec<f32>,
    max: Vec<f32>,
    scale: Vec<f32>,
    zero_point: Vec<i64>,
    quantized_dimension: i32,
}

impl QuantizationInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            min: reader.f32_vector(position, 4)?,
            max: reader.f32_vector(position, 6)?,
            scale: reader.f32_vector(position, 8)?,
            zero_point: reader.i64_vector(position, 10)?,
            quantized_dimension: reader.i32_field(position, 16, 0),
        })
    }
}

#[derive(Default)]
struct OperatorInfoNode {
    opcode_index: u32,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    builtin_option_type: Option<u8>,
    attributes: Vec<ParsedAttribute>,
    fallback_attributes: Vec<ParsedAttribute>,
    custom_options: Vec<u8>,
    custom_options_format: i8,
}

impl OperatorInfoNode {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        let builtin_options = read_builtin_options(reader, position)?;
        Ok(Self {
            opcode_index: reader.u32_field(position, 4, 0),
            inputs: reader.i32_vector(position, 6)?,
            outputs: reader.i32_vector(position, 8)?,
            builtin_option_type: builtin_options.option_type,
            attributes: builtin_options.attributes,
            fallback_attributes: builtin_options.fallback_attributes,
            custom_options: reader.u8_vector(position, 14)?,
            custom_options_format: reader.i8_field(position, 16, 0),
        })
    }
}

#[derive(Default)]
struct BuiltinOptionAttributes {
    option_type: Option<u8>,
    attributes: Vec<ParsedAttribute>,
    fallback_attributes: Vec<ParsedAttribute>,
}

#[derive(Clone)]
struct ParsedAttribute {
    name: String,
    value: ParsedAttributeValue,
}

#[derive(Clone)]
enum ParsedAttributeValue {
    Bool(bool),
    Float(f32),
    Int(i32),
    String(String),
    Ints(Vec<i32>),
    Strings(Vec<String>),
}

fn read_builtin_options(
    reader: &FlatReader<'_>,
    position: usize,
) -> Result<BuiltinOptionAttributes, ModelError> {
    let Some((option_type, table)) = reader.union_table_field(position, 10)? else {
        return Ok(BuiltinOptionAttributes::default());
    };
    let mut attributes = Vec::new();
    let mut fallback_attributes = Vec::new();
    match option_type {
        1 => {
            read_conv_2d_options(reader, table, &mut attributes);
            read_conv_2d_fallback_options(reader, table, &mut fallback_attributes);
        }
        2 => read_depthwise_conv_2d_options(reader, table, &mut attributes),
        4 => {
            let projection_type = reader.i8_field(table, 4, 0);
            if projection_type != 0 {
                attributes.push(string_attribute(
                    "type",
                    match projection_type {
                        1 => "SPARSE",
                        2 => "DENSE",
                        _ => "?",
                    },
                ));
            }
        }
        5 => read_pool_2d_options(reader, table, &mut attributes),
        6 => read_svdf_options(reader, table, &mut attributes),
        8 => read_fully_connected_options(reader, table, &mut attributes),
        9 => {
            let beta = reader.f32_field(table, 4, 0.0);
            if beta != 0.0 {
                attributes.push(float_attribute("beta", beta));
            }
        }
        10 => {
            let axis = reader.i32_field(table, 4, 0);
            if axis != 0 {
                attributes.push(int_attribute("axis", axis));
            }
        }
        14 | 69 | 71 => read_lstm_options(reader, table, &mut attributes),
        18 => read_skip_gram_options(reader, table, &mut attributes),
        23 => {
            push_non_default_int(&mut attributes, "axis", reader.i32_field(table, 4, 0), 0);
            push_non_default_int(
                &mut attributes,
                "batch_dims",
                reader.i32_field(table, 6, 0),
                0,
            );
        }
        17 => {
            let shape = reader.i32_vector(table, 4)?;
            if !shape.is_empty() {
                attributes.push(ParsedAttribute {
                    name: "new_shape".to_owned(),
                    value: ParsedAttributeValue::Ints(shape),
                });
            }
        }
        30 => {
            let dims = reader.i32_vector(table, 4)?;
            if !dims.is_empty() {
                attributes.push(ParsedAttribute {
                    name: "squeeze_dims".to_owned(),
                    value: ParsedAttributeValue::Ints(dims),
                });
            }
        }
        27 => {
            attributes.push(bool_attribute(
                "keep_dims",
                reader.bool_field(table, 4, false),
            ));
        }
        32 => read_strided_slice_options(reader, table, &mut attributes),
        35 => push_non_default_int(
            &mut attributes,
            "num_splits",
            reader.i32_field(table, 4, 0),
            0,
        ),
        37 => {
            push_non_default_tensor_type(
                &mut attributes,
                "in_data_type",
                reader.i8_field(table, 4, 0),
                0,
            );
            push_non_default_tensor_type(
                &mut attributes,
                "out_data_type",
                reader.i8_field(table, 6, 0),
                0,
            );
        }
        40 => push_non_default_tensor_type(
            &mut attributes,
            "output_type",
            reader.i8_field(table, 4, 0),
            0,
        ),
        55 => push_non_default_tensor_type(
            &mut attributes,
            "out_type",
            reader.i8_field(table, 4, 0),
            0,
        ),
        57 => push_non_default_tensor_type(
            &mut attributes,
            "output_type",
            reader.i8_field(table, 4, 0),
            0,
        ),
        49 => read_transpose_conv_options(reader, table, &mut attributes),
        59 => {
            push_non_default_int(
                &mut attributes,
                "values_count",
                reader.i32_field(table, 4, 0),
                0,
            );
            push_non_default_int(&mut attributes, "axis", reader.i32_field(table, 6, 0), 0);
        }
        64 => {
            push_non_default_int(&mut attributes, "num", reader.i32_field(table, 4, 0), 0);
            push_non_default_int(&mut attributes, "axis", reader.i32_field(table, 6, 0), 0);
        }
        74 => {
            if reader.bool_field(table, 4, false) {
                attributes.push(bool_attribute("align_corners", true));
            }
            if reader.bool_field(table, 6, false) {
                attributes.push(bool_attribute("half_pixel_centers", true));
            }
        }
        15 => {
            let new_height = reader.i32_field(table, 4, 0);
            let new_width = reader.i32_field(table, 6, 0);
            if new_height != 0 {
                attributes.push(int_attribute("new_height", new_height));
            }
            if new_width != 0 {
                attributes.push(int_attribute("new_width", new_width));
            }
            if reader.bool_field(table, 8, false) {
                attributes.push(bool_attribute("align_corners", true));
            }
            if reader.bool_field(table, 10, false) {
                attributes.push(bool_attribute("half_pixel_centers", true));
            }
        }
        79 => push_non_default_int(
            &mut attributes,
            "num_splits",
            reader.i32_field(table, 4, 0),
            0,
        ),
        92 => {
            push_non_default_int(
                &mut attributes,
                "then_subgraph_index",
                reader.i32_field(table, 4, 0),
                0,
            );
            push_non_default_int(
                &mut attributes,
                "else_subgraph_index",
                reader.i32_field(table, 6, 0),
                0,
            );
        }
        93 => {
            push_non_default_int(
                &mut attributes,
                "cond_subgraph_index",
                reader.i32_field(table, 4, 0),
                0,
            );
            push_non_default_int(
                &mut attributes,
                "body_subgraph_index",
                reader.i32_field(table, 6, 0),
                0,
            );
        }
        _ => {}
    }
    Ok(BuiltinOptionAttributes {
        option_type: Some(option_type),
        attributes,
        fallback_attributes,
    })
}

fn read_lstm_options(reader: &FlatReader<'_>, table: usize, attributes: &mut Vec<ParsedAttribute>) {
    push_non_default_float(
        attributes,
        "cell_clip",
        reader.f32_field(table, 6, 0.0),
        0.0,
    );
    push_non_default_float(
        attributes,
        "proj_clip",
        reader.f32_field(table, 8, 0.0),
        0.0,
    );
    let kernel_type = reader.i8_field(table, 10, 0);
    if kernel_type != 0 {
        attributes.push(string_attribute(
            "kernel_type",
            match kernel_type {
                1 => "BASIC",
                _ => "?",
            },
        ));
    }
    if reader.bool_field(table, 12, false) {
        attributes.push(bool_attribute("asymmetric_quantize_inputs", true));
    }
}

fn read_skip_gram_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    push_non_default_int(attributes, "ngram_size", reader.i32_field(table, 4, 0), 0);
    push_non_default_int(
        attributes,
        "max_skip_size",
        reader.i32_field(table, 6, 0),
        0,
    );
    if reader.bool_field(table, 8, false) {
        attributes.push(bool_attribute("include_all_ngrams", true));
    }
}

fn read_conv_2d_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    push_padding(attributes, reader.i8_field(table, 4, 0));
    push_present_non_zero_int(attributes, "stride_w", reader.i32_field_opt(table, 6));
    push_present_non_zero_int(attributes, "stride_h", reader.i32_field_opt(table, 8));
    push_non_default_int(
        attributes,
        "dilation_w_factor",
        reader.i32_field(table, 12, 1),
        1,
    );
    push_non_default_int(
        attributes,
        "dilation_h_factor",
        reader.i32_field(table, 14, 1),
        1,
    );
    push_non_default_tensor_type(
        attributes,
        "quantized_bias_type",
        reader.i8_field(table, 16, 0),
        0,
    );
}

fn read_conv_2d_fallback_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    attributes.push(int_attribute(
        "padding",
        i32::from(reader.i8_field(table, 4, 0)),
    ));
    attributes.push(int_attribute("stride_w", reader.i32_field(table, 6, 0)));
    attributes.push(int_attribute("stride_h", reader.i32_field(table, 8, 0)));
    attributes.push(int_attribute(
        "dilation_w_factor",
        reader.i32_field(table, 12, 1),
    ));
    attributes.push(int_attribute(
        "dilation_h_factor",
        reader.i32_field(table, 14, 1),
    ));
    attributes.push(int_attribute(
        "quantized_bias_type",
        i32::from(reader.i8_field(table, 16, 0)),
    ));
}

fn read_depthwise_conv_2d_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    push_padding(attributes, reader.i8_field(table, 4, 0));
    push_present_non_zero_int(attributes, "stride_w", reader.i32_field_opt(table, 6));
    push_present_non_zero_int(attributes, "stride_h", reader.i32_field_opt(table, 8));
    push_present_non_zero_int(
        attributes,
        "depth_multiplier",
        reader.i32_field_opt(table, 10),
    );
    push_non_default_int(
        attributes,
        "dilation_w_factor",
        reader.i32_field(table, 14, 1),
        1,
    );
    push_non_default_int(
        attributes,
        "dilation_h_factor",
        reader.i32_field(table, 16, 1),
        1,
    );
}

fn read_svdf_options(reader: &FlatReader<'_>, table: usize, attributes: &mut Vec<ParsedAttribute>) {
    push_non_default_int(attributes, "rank", reader.i32_field(table, 4, 0), 0);
    if reader.bool_field(table, 8, false) {
        attributes.push(bool_attribute("asymmetric_quantize_inputs", true));
    }
}

fn read_fully_connected_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    let weights_format = reader.i8_field(table, 6, 0);
    if weights_format != 0 {
        attributes.push(string_attribute(
            "weights_format",
            match weights_format {
                1 => "SHUFFLED4x16INT8",
                _ => "?",
            },
        ));
    }
    if reader.bool_field(table, 8, false) {
        attributes.push(bool_attribute("keep_num_dims", true));
    }
    if reader.bool_field(table, 10, false) {
        attributes.push(bool_attribute("asymmetric_quantize_inputs", true));
    }
    push_non_default_tensor_type(
        attributes,
        "quantized_bias_type",
        reader.i8_field(table, 12, 0),
        0,
    );
}

fn read_strided_slice_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    push_non_default_int(attributes, "begin_mask", reader.i32_field(table, 4, 0), 0);
    push_non_default_int(attributes, "end_mask", reader.i32_field(table, 6, 0), 0);
    push_non_default_int(
        attributes,
        "ellipsis_mask",
        reader.i32_field(table, 8, 0),
        0,
    );
    push_non_default_int(
        attributes,
        "new_axis_mask",
        reader.i32_field(table, 10, 0),
        0,
    );
    push_non_default_int(
        attributes,
        "shrink_axis_mask",
        reader.i32_field(table, 12, 0),
        0,
    );
    if reader.bool_field(table, 14, false) {
        attributes.push(bool_attribute("offset", true));
    }
}

fn read_transpose_conv_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    push_padding(attributes, reader.i8_field(table, 4, 0));
    push_non_default_int(attributes, "stride_w", reader.i32_field(table, 6, 0), 0);
    push_non_default_int(attributes, "stride_h", reader.i32_field(table, 8, 0), 0);
    push_non_default_tensor_type(
        attributes,
        "quantized_bias_type",
        reader.i8_field(table, 12, 0),
        0,
    );
}

fn read_pool_2d_options(
    reader: &FlatReader<'_>,
    table: usize,
    attributes: &mut Vec<ParsedAttribute>,
) {
    push_padding(attributes, reader.i8_field(table, 4, 0));
    push_non_default_int(attributes, "stride_w", reader.i32_field(table, 6, 0), 0);
    push_non_default_int(attributes, "stride_h", reader.i32_field(table, 8, 0), 0);
    push_non_default_int(
        attributes,
        "filter_width",
        reader.i32_field(table, 10, 0),
        0,
    );
    push_non_default_int(
        attributes,
        "filter_height",
        reader.i32_field(table, 12, 0),
        0,
    );
}

fn push_padding(attributes: &mut Vec<ParsedAttribute>, value: i8) {
    if value != 0 {
        attributes.push(ParsedAttribute {
            name: "padding".to_owned(),
            value: ParsedAttributeValue::String(
                match value {
                    0 => "SAME",
                    1 => "VALID",
                    _ => "?",
                }
                .to_owned(),
            ),
        });
    }
}

fn push_non_default_tensor_type(
    attributes: &mut Vec<ParsedAttribute>,
    name: &'static str,
    value: i8,
    default: i8,
) {
    if value != default {
        attributes.push(string_attribute(name, tensor_type_attribute_name(value)));
    }
}

fn push_non_default_int(
    attributes: &mut Vec<ParsedAttribute>,
    name: &'static str,
    value: i32,
    default: i32,
) {
    if value != default {
        attributes.push(int_attribute(name, value));
    }
}

fn push_non_default_float(
    attributes: &mut Vec<ParsedAttribute>,
    name: &'static str,
    value: f32,
    default: f32,
) {
    if value != default {
        attributes.push(float_attribute(name, value));
    }
}

fn push_present_non_zero_int(
    attributes: &mut Vec<ParsedAttribute>,
    name: &'static str,
    value: Option<i32>,
) {
    if let Some(value) = value
        && value != 0
    {
        attributes.push(int_attribute(name, value));
    }
}

fn int_attribute(name: &'static str, value: i32) -> ParsedAttribute {
    ParsedAttribute {
        name: name.to_owned(),
        value: ParsedAttributeValue::Int(value),
    }
}

fn float_attribute(name: &'static str, value: f32) -> ParsedAttribute {
    ParsedAttribute {
        name: name.to_owned(),
        value: ParsedAttributeValue::Float(value),
    }
}

fn string_attribute(name: &'static str, value: &'static str) -> ParsedAttribute {
    ParsedAttribute {
        name: name.to_owned(),
        value: ParsedAttributeValue::String(value.to_owned()),
    }
}

fn bool_attribute(name: &'static str, value: bool) -> ParsedAttribute {
    ParsedAttribute {
        name: name.to_owned(),
        value: ParsedAttributeValue::Bool(value),
    }
}

fn raw_custom_attribute(data: &[u8]) -> ParsedAttribute {
    ParsedAttribute {
        name: "custom".to_owned(),
        value: ParsedAttributeValue::Ints(data.iter().map(|value| i32::from(*value)).collect()),
    }
}

fn tensor_type_attribute_name(value: i8) -> &'static str {
    match value {
        0 => "FLOAT32",
        1 => "FLOAT16",
        2 => "INT32",
        3 => "UINT8",
        4 => "INT64",
        5 => "STRING",
        6 => "BOOL",
        7 => "INT16",
        8 => "COMPLEX64",
        9 => "INT8",
        10 => "FLOAT64",
        11 => "COMPLEX128",
        12 => "UINT64",
        13 => "RESOURCE",
        14 => "VARIANT",
        15 => "UINT32",
        16 => "UINT16",
        17 => "INT4",
        18 => "BFLOAT16",
        19 => "INT2",
        20 => "UINT4",
        _ => "?",
    }
}

#[derive(Default)]
struct MetadataInfo {
    name: Option<String>,
    buffer: u32,
}

impl MetadataInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            name: reader.string_field(position, 4)?,
            buffer: reader.u32_field(position, 6, 0),
        })
    }
}

#[derive(Default)]
struct EmbeddedMetadata {
    description: Option<String>,
    version: Option<String>,
    properties: BTreeMap<String, String>,
    subgraphs: Vec<SubgraphMetadataInfo>,
}

impl EmbeddedMetadata {
    fn read(reader: &FlatReader<'_>, buffers: &[BufferInfo], entries: &[MetadataInfo]) -> Self {
        let mut result = Self::default();
        for entry in entries {
            let Some(name) = entry.name.as_deref() else {
                continue;
            };
            let Some(data) = buffers
                .get(entry.buffer as usize)
                .and_then(|buffer| buffer.bytes(reader.data))
            else {
                continue;
            };
            match name {
                "min_runtime_version" => {}
                "TFLITE_METADATA" => {
                    if let Some(metadata) = ModelMetadataInfo::read(data) {
                        result.description = metadata.description;
                        result.version = metadata.version;
                        if let Some(author) = metadata.author {
                            result.properties.insert("author".to_owned(), author);
                        }
                        if let Some(license) = metadata.license {
                            result.properties.insert("license".to_owned(), license);
                        }
                        result.subgraphs = metadata.subgraphs;
                    }
                }
                _ => {
                    result
                        .properties
                        .insert(name.to_owned(), metadata_value_text(data));
                }
            }
        }
        result
    }
}

fn metadata_value_text(data: &[u8]) -> String {
    if data.len() < 256 && data.iter().all(|value| (32..128).contains(value)) {
        String::from_utf8_lossy(data).into_owned()
    } else {
        "?".to_owned()
    }
}

#[derive(Default)]
struct ModelMetadataInfo {
    description: Option<String>,
    version: Option<String>,
    author: Option<String>,
    license: Option<String>,
    subgraphs: Vec<SubgraphMetadataInfo>,
}

impl ModelMetadataInfo {
    fn read(data: &[u8]) -> Option<Self> {
        if data.len() < 8 || data.get(4..8) != Some(b"M001") {
            return None;
        }
        let reader = FlatReader { data };
        let root = reader.root().ok()?;
        Some(Self {
            description: reader.string_field(root, 6).ok().flatten(),
            version: reader.string_field(root, 8).ok().flatten(),
            subgraphs: reader
                .table_vector(root, 10)
                .ok()?
                .into_iter()
                .map(|position| SubgraphMetadataInfo::read(&reader, position))
                .collect::<Option<Vec<_>>>()?,
            author: reader.string_field(root, 12).ok().flatten(),
            license: reader.string_field(root, 14).ok().flatten(),
        })
    }
}

#[derive(Clone, Default)]
struct SubgraphMetadataInfo {
    inputs: Vec<TensorMetadataInfo>,
    outputs: Vec<TensorMetadataInfo>,
}

impl SubgraphMetadataInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Option<Self> {
        Some(Self {
            inputs: reader
                .table_vector(position, 8)
                .ok()?
                .into_iter()
                .map(|position| TensorMetadataInfo::read(reader, position))
                .collect::<Option<Vec<_>>>()?,
            outputs: reader
                .table_vector(position, 10)
                .ok()?
                .into_iter()
                .map(|position| TensorMetadataInfo::read(reader, position))
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

#[derive(Clone, Default)]
struct TensorMetadataInfo {
    description: Option<String>,
    denotation: Option<String>,
}

impl TensorMetadataInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Option<Self> {
        Some(Self {
            description: reader.string_field(position, 6).ok().flatten(),
            denotation: reader
                .table_field(position, 10)
                .ok()
                .flatten()
                .and_then(|position| read_content_denotation(reader, position)),
        })
    }
}

fn read_content_denotation(reader: &FlatReader<'_>, position: usize) -> Option<String> {
    let (kind, table) = reader.union_table_field(position, 4).ok().flatten()?;
    match kind {
        1 => Some("Feature".to_owned()),
        2 => {
            let color_space = reader.i8_field(table, 4, 0);
            Some(format!(
                "Image({})",
                match color_space {
                    0 => "Unknown",
                    1 => "RGB",
                    2 => "Grayscale",
                    _ => "?",
                }
            ))
        }
        3 => Some("BoundingBox".to_owned()),
        4 => {
            let sample_rate = reader.u32_field(table, 4, 0);
            let channels = reader.u32_field(table, 6, 0);
            Some(format!("Audio({sample_rate},{channels})"))
        }
        _ => None,
    }
}

#[derive(Default)]
struct SignatureDefInfo {
    inputs: Vec<TensorMapInfo>,
    outputs: Vec<TensorMapInfo>,
    subgraph_index: u32,
}

impl SignatureDefInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            inputs: reader
                .table_vector(position, 4)?
                .into_iter()
                .map(|position| TensorMapInfo::read(reader, position))
                .collect::<Result<_, _>>()?,
            outputs: reader
                .table_vector(position, 6)?
                .into_iter()
                .map(|position| TensorMapInfo::read(reader, position))
                .collect::<Result<_, _>>()?,
            subgraph_index: reader.u32_field(position, 12, 0),
        })
    }
}

#[derive(Default)]
struct TensorMapInfo {
    tensor_index: u32,
}

impl TensorMapInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        Ok(Self {
            tensor_index: reader.u32_field(position, 6, 0),
        })
    }
}

#[derive(Default)]
struct BufferInfo {
    byte_len: usize,
    data_range: Option<(usize, usize)>,
}

impl BufferInfo {
    fn read(reader: &FlatReader<'_>, position: usize) -> Result<Self, ModelError> {
        let inline_range = reader.u8_vector_range(position, 4)?;
        let external_range = if inline_range.is_none() {
            let offset = reader.u64_field(position, 6, 0) as usize;
            let size = reader.u64_field(position, 8, 0) as usize;
            (offset != 0 && size != 0 && offset.saturating_add(size) <= reader.data.len())
                .then_some((offset, size))
        } else {
            None
        };
        let data_range = inline_range.or(external_range);
        Ok(Self {
            byte_len: data_range.map(|(_, len)| len).unwrap_or_default(),
            data_range,
        })
    }

    fn bytes<'a>(&self, data: &'a [u8]) -> Option<&'a [u8]> {
        let (start, len) = self.data_range?;
        data.get(start..start + len)
    }
}

struct FlatReader<'a> {
    data: &'a [u8],
}

impl FlatReader<'_> {
    fn root(&self) -> Result<usize, ModelError> {
        if self.data.len() < 4 {
            return Err(invalid("file is too small"));
        }
        let root = self.u32(0)? as usize;
        self.require(root, 4)?;
        Ok(root)
    }

    fn table_field(&self, table: usize, slot: usize) -> Result<Option<usize>, ModelError> {
        let Some(field) = self.field(table, slot)? else {
            return Ok(None);
        };
        let offset = self.u32(field)? as usize;
        let target = field
            .checked_add(offset)
            .ok_or_else(|| invalid("flatbuffer offset overflow"))?;
        self.require(target, 4)?;
        Ok(Some(target))
    }

    fn union_table_field(
        &self,
        table: usize,
        slot: usize,
    ) -> Result<Option<(u8, usize)>, ModelError> {
        let Some(type_field) = self.field(table, slot)? else {
            return Ok(None);
        };
        let option_type = self
            .data
            .get(type_field)
            .copied()
            .ok_or_else(|| invalid("unexpected end of file"))?;
        if option_type == 0 {
            return Ok(None);
        }
        let Some(value_field) = self.field(table, slot + 2)? else {
            return Ok(None);
        };
        Ok(Some((option_type, self.indirect(value_field)?)))
    }

    fn string_field(&self, table: usize, slot: usize) -> Result<Option<String>, ModelError> {
        let Some(field) = self.field(table, slot)? else {
            return Ok(None);
        };
        let vector = self.indirect(field)?;
        let len = self.u32(vector)? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| invalid("flatbuffer string offset overflow"))?;
        self.require(start, len)?;
        Ok(Some(
            String::from_utf8_lossy(&self.data[start..start + len]).into_owned(),
        ))
    }

    fn table_vector(&self, table: usize, slot: usize) -> Result<Vec<usize>, ModelError> {
        let Some(vector) = self.vector(table, slot)? else {
            return Ok(Vec::new());
        };
        let len = self.u32(vector)? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| invalid("flatbuffer vector offset overflow"))?;
        let mut result = Vec::with_capacity(len);
        for index in 0..len {
            let element = start + index * 4;
            let offset = self.u32(element)? as usize;
            let target = element
                .checked_add(offset)
                .ok_or_else(|| invalid("flatbuffer table vector offset overflow"))?;
            self.require(target, 4)?;
            result.push(target);
        }
        Ok(result)
    }

    fn i32_vector(&self, table: usize, slot: usize) -> Result<Vec<i32>, ModelError> {
        self.i32_vector_opt(table, slot)
            .map(|value| value.unwrap_or_default())
    }

    fn i32_vector_opt(&self, table: usize, slot: usize) -> Result<Option<Vec<i32>>, ModelError> {
        let Some(vector) = self.vector(table, slot)? else {
            return Ok(None);
        };
        let len = self.u32(vector)? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| invalid("flatbuffer vector offset overflow"))?;
        self.require(start, len * 4)?;
        Ok(Some(
            (0..len)
                .map(|index| self.i32(start + index * 4))
                .collect::<Result<_, _>>()?,
        ))
    }

    fn i64_vector(&self, table: usize, slot: usize) -> Result<Vec<i64>, ModelError> {
        let Some(vector) = self.vector(table, slot)? else {
            return Ok(Vec::new());
        };
        let len = self.u32(vector)? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| invalid("flatbuffer vector offset overflow"))?;
        self.require(start, len * 8)?;
        (0..len).map(|index| self.i64(start + index * 8)).collect()
    }

    fn f32_vector(&self, table: usize, slot: usize) -> Result<Vec<f32>, ModelError> {
        let Some(vector) = self.vector(table, slot)? else {
            return Ok(Vec::new());
        };
        let len = self.u32(vector)? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| invalid("flatbuffer vector offset overflow"))?;
        self.require(start, len * 4)?;
        (0..len).map(|index| self.f32(start + index * 4)).collect()
    }

    fn u8_vector(&self, table: usize, slot: usize) -> Result<Vec<u8>, ModelError> {
        let Some((start, len)) = self.u8_vector_range(table, slot)? else {
            return Ok(Vec::new());
        };
        Ok(self.data[start..start + len].to_vec())
    }

    fn u8_vector_range(
        &self,
        table: usize,
        slot: usize,
    ) -> Result<Option<(usize, usize)>, ModelError> {
        let Some(vector) = self.vector(table, slot)? else {
            return Ok(None);
        };
        let len = self.u32(vector)? as usize;
        let start = vector
            .checked_add(4)
            .ok_or_else(|| invalid("flatbuffer vector offset overflow"))?;
        self.require(start, len)?;
        Ok(Some((start, len)))
    }

    fn vector(&self, table: usize, slot: usize) -> Result<Option<usize>, ModelError> {
        let Some(field) = self.field(table, slot)? else {
            return Ok(None);
        };
        let vector = self.indirect(field)?;
        self.require(vector, 4)?;
        Ok(Some(vector))
    }

    fn indirect(&self, position: usize) -> Result<usize, ModelError> {
        let offset = self.u32(position)? as usize;
        let target = position
            .checked_add(offset)
            .ok_or_else(|| invalid("flatbuffer offset overflow"))?;
        self.require(target, 4)?;
        Ok(target)
    }

    fn field(&self, table: usize, slot: usize) -> Result<Option<usize>, ModelError> {
        self.require(table, 4)?;
        let vtable_distance = self.i32(table)?;
        let vtable = table
            .checked_add_signed(-(vtable_distance as isize))
            .ok_or_else(|| invalid("vtable offset outside file"))?;
        self.require(vtable, 4)?;
        let vtable_len = usize::from(self.u16(vtable)?);
        if slot + 2 > vtable_len {
            return Ok(None);
        }
        let offset = usize::from(self.u16(vtable + slot)?);
        if offset == 0 {
            return Ok(None);
        }
        let field = table
            .checked_add(offset)
            .ok_or_else(|| invalid("field offset overflow"))?;
        self.require(field, 1)?;
        Ok(Some(field))
    }

    fn bool_field(&self, table: usize, slot: usize, default: bool) -> bool {
        self.bool_field_opt(table, slot).unwrap_or(default)
    }

    fn bool_field_opt(&self, table: usize, slot: usize) -> Option<bool> {
        self.field(table, slot)
            .ok()
            .flatten()
            .and_then(|position| self.data.get(position).copied())
            .map(|value| value != 0)
    }

    fn i8_field(&self, table: usize, slot: usize, default: i8) -> i8 {
        self.field(table, slot)
            .ok()
            .flatten()
            .and_then(|position| self.data.get(position).copied())
            .map(|value| value as i8)
            .unwrap_or(default)
    }

    fn i32_field(&self, table: usize, slot: usize, default: i32) -> i32 {
        self.i32_field_opt(table, slot).unwrap_or(default)
    }

    fn i32_field_opt(&self, table: usize, slot: usize) -> Option<i32> {
        self.field(table, slot)
            .ok()
            .flatten()
            .and_then(|position| self.i32(position).ok())
    }

    fn f32_field(&self, table: usize, slot: usize, default: f32) -> f32 {
        self.field(table, slot)
            .ok()
            .flatten()
            .and_then(|position| self.f32(position).ok())
            .unwrap_or(default)
    }

    fn u32_field(&self, table: usize, slot: usize, default: u32) -> u32 {
        self.field(table, slot)
            .ok()
            .flatten()
            .and_then(|position| self.u32(position).ok())
            .unwrap_or(default)
    }

    fn u64_field(&self, table: usize, slot: usize, default: u64) -> u64 {
        self.field(table, slot)
            .ok()
            .flatten()
            .and_then(|position| self.u64(position).ok())
            .unwrap_or(default)
    }

    fn require(&self, position: usize, len: usize) -> Result<(), ModelError> {
        let end = position
            .checked_add(len)
            .ok_or_else(|| invalid("flatbuffer offset overflow"))?;
        if end > self.data.len() {
            return Err(invalid("unexpected end of file"));
        }
        Ok(())
    }

    fn u16(&self, position: usize) -> Result<u16, ModelError> {
        self.require(position, 2)?;
        Ok(u16::from_le_bytes(
            self.data[position..position + 2].try_into().unwrap(),
        ))
    }

    fn u32(&self, position: usize) -> Result<u32, ModelError> {
        self.require(position, 4)?;
        Ok(u32::from_le_bytes(
            self.data[position..position + 4].try_into().unwrap(),
        ))
    }

    fn u64(&self, position: usize) -> Result<u64, ModelError> {
        self.require(position, 8)?;
        Ok(u64::from_le_bytes(
            self.data[position..position + 8].try_into().unwrap(),
        ))
    }

    fn i32(&self, position: usize) -> Result<i32, ModelError> {
        self.require(position, 4)?;
        Ok(i32::from_le_bytes(
            self.data[position..position + 4].try_into().unwrap(),
        ))
    }

    fn i64(&self, position: usize) -> Result<i64, ModelError> {
        self.require(position, 8)?;
        Ok(i64::from_le_bytes(
            self.data[position..position + 8].try_into().unwrap(),
        ))
    }

    fn f32(&self, position: usize) -> Result<f32, ModelError> {
        self.require(position, 4)?;
        Ok(f32::from_le_bytes(
            self.data[position..position + 4].try_into().unwrap(),
        ))
    }
}

fn read_flexbuffer_attributes(data: &[u8]) -> Option<Vec<ParsedAttribute>> {
    match FlexReader::read(data)? {
        FlexValue::Map(entries) => Some(
            entries
                .into_iter()
                .filter_map(|(name, value)| flex_value_to_attribute(name, value))
                .collect(),
        ),
        FlexValue::Vector(values) => Some(vec![ParsedAttribute {
            name: "custom_options".to_owned(),
            value: flex_vector_to_attribute(values)?,
        }]),
        _ => None,
    }
}

fn flex_value_to_attribute(name: String, value: FlexValue) -> Option<ParsedAttribute> {
    let value = match value {
        FlexValue::Bool(value) => ParsedAttributeValue::Bool(value),
        FlexValue::Float(value) => ParsedAttributeValue::Float(value as f32),
        FlexValue::Int(value) => ParsedAttributeValue::Int(value.try_into().ok()?),
        FlexValue::UInt(value) => ParsedAttributeValue::Int(value.try_into().ok()?),
        FlexValue::String(value) => ParsedAttributeValue::String(value),
        FlexValue::Vector(values) => {
            let ints = values
                .into_iter()
                .map(|value| match value {
                    FlexValue::Int(value) => value.try_into().ok(),
                    FlexValue::UInt(value) => value.try_into().ok(),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            ParsedAttributeValue::Ints(ints)
        }
        FlexValue::Null | FlexValue::Map(_) => return None,
    };
    Some(ParsedAttribute { name, value })
}

fn flex_vector_to_attribute(values: Vec<FlexValue>) -> Option<ParsedAttributeValue> {
    let mut ints = Vec::new();
    let mut strings = Vec::new();
    for value in values {
        match value {
            FlexValue::Int(value) => {
                ints.push(value.try_into().ok()?);
            }
            FlexValue::UInt(value) => {
                ints.push(value.try_into().ok()?);
            }
            FlexValue::String(value) => {
                strings.push(value);
            }
            _ => return None,
        }
    }
    if !ints.is_empty() && strings.is_empty() {
        Some(ParsedAttributeValue::Ints(ints))
    } else if ints.is_empty() {
        Some(ParsedAttributeValue::Strings(strings))
    } else {
        None
    }
}

enum FlexValue {
    Null,
    Bool(bool),
    Float(f64),
    Int(i64),
    UInt(u64),
    String(String),
    Vector(Vec<FlexValue>),
    Map(Vec<(String, FlexValue)>),
}

struct FlexReader<'a> {
    data: &'a [u8],
}

impl<'a> FlexReader<'a> {
    fn read(data: &'a [u8]) -> Option<FlexValue> {
        let reader = Self { data };
        let len = data.len();
        if len < 3 {
            return None;
        }
        let parent_width = usize::from(*data.last()?);
        if parent_width == 0 || parent_width > 8 {
            return None;
        }
        let packed_type = data[len - 2];
        let offset = len.checked_sub(2 + parent_width)?;
        let byte_width = 1usize.checked_shl(u32::from(packed_type & 3))?;
        reader.reference(offset, parent_width, byte_width, packed_type >> 2)
    }

    fn reference(
        &self,
        offset: usize,
        parent_width: usize,
        byte_width: usize,
        value_type: u8,
    ) -> Option<FlexValue> {
        match value_type {
            0x00 => Some(FlexValue::Null),
            0x01 => self.read_int(offset, parent_width).map(FlexValue::Int),
            0x02 => self.read_uint(offset, parent_width).map(FlexValue::UInt),
            0x03 => self.read_float(offset, parent_width).map(FlexValue::Float),
            0x04 => self
                .indirect(offset, parent_width)
                .and_then(|position| self.string(position, None))
                .map(FlexValue::String),
            0x05 => {
                let position = self.indirect(offset, parent_width)?;
                let size_position = position.checked_sub(byte_width)?;
                let size = usize::try_from(self.read_uint(size_position, byte_width)?).ok()?;
                self.string(position, Some(size)).map(FlexValue::String)
            }
            0x06 => self
                .indirect(offset, parent_width)
                .and_then(|position| self.read_int(position, byte_width))
                .map(FlexValue::Int),
            0x07 => self
                .indirect(offset, parent_width)
                .and_then(|position| self.read_uint(position, byte_width))
                .map(FlexValue::UInt),
            0x08 => self
                .indirect(offset, parent_width)
                .and_then(|position| self.read_float(position, byte_width))
                .map(FlexValue::Float),
            0x09 => self.map(offset, parent_width, byte_width),
            0x0a => self
                .indirect(offset, parent_width)
                .and_then(|position| self.vector(position, byte_width))
                .map(FlexValue::Vector),
            0x0b..=0x0f | 0x24 => {
                let position = self.indirect(offset, parent_width)?;
                let element_type = value_type.wrapping_sub(0x0b).wrapping_add(0x01);
                self.typed_vector(position, byte_width, element_type, None)
                    .map(FlexValue::Vector)
            }
            0x10..=0x18 => {
                let position = self.indirect(offset, parent_width)?;
                let size = usize::from((value_type - 0x10) / 3) + 2;
                let element_type = ((value_type - 0x10) % 3) + 0x01;
                self.typed_vector(position, byte_width, element_type, Some(size))
                    .map(FlexValue::Vector)
            }
            0x1a => self
                .read_uint(offset, parent_width)
                .map(|value| FlexValue::Bool(value != 0)),
            _ => None,
        }
    }

    fn map(&self, offset: usize, parent_width: usize, byte_width: usize) -> Option<FlexValue> {
        let offset = self.indirect(offset, parent_width)?;
        let keys_offset = offset.checked_sub(byte_width.checked_mul(3)?)?;
        let keys_vector_offset = keys_offset
            .checked_sub(usize::try_from(self.read_uint(keys_offset, byte_width)?).ok()?)?;
        let keys_byte_width =
            usize::try_from(self.read_uint(keys_offset + byte_width, byte_width)?).ok()?;
        let keys = self.typed_vector(keys_vector_offset, keys_byte_width, 0x04, None)?;
        let values = self.vector(offset, byte_width)?;
        let entries = keys
            .into_iter()
            .zip(values)
            .filter_map(|(key, value)| match key {
                FlexValue::String(key) => Some((key, value)),
                _ => None,
            })
            .collect();
        Some(FlexValue::Map(entries))
    }

    fn vector(&self, offset: usize, byte_width: usize) -> Option<Vec<FlexValue>> {
        let size_position = offset.checked_sub(byte_width)?;
        let size = usize::try_from(self.read_uint(size_position, byte_width)?).ok()?;
        let packed_type_offset = offset.checked_add(size.checked_mul(byte_width)?)?;
        (0..size)
            .map(|index| {
                let packed_type = *self.data.get(packed_type_offset + index)?;
                self.reference(
                    offset + index * byte_width,
                    byte_width,
                    1usize.checked_shl(u32::from(packed_type & 3))?,
                    packed_type >> 2,
                )
            })
            .collect()
    }

    fn typed_vector(
        &self,
        offset: usize,
        byte_width: usize,
        value_type: u8,
        size: Option<usize>,
    ) -> Option<Vec<FlexValue>> {
        let size = match size {
            Some(size) => size,
            None => {
                let size_position = offset.checked_sub(byte_width)?;
                usize::try_from(self.read_uint(size_position, byte_width)?).ok()?
            }
        };
        (0..size)
            .map(|index| self.reference(offset + index * byte_width, byte_width, 1, value_type))
            .collect()
    }

    fn indirect(&self, offset: usize, parent_width: usize) -> Option<usize> {
        offset.checked_sub(usize::try_from(self.read_uint(offset, parent_width)?).ok()?)
    }

    fn read_int(&self, offset: usize, size: usize) -> Option<i64> {
        self.require(offset, size)?;
        Some(match size {
            1 => i64::from(i8::from_le_bytes([self.data[offset]])),
            2 => i64::from(i16::from_le_bytes(
                self.data[offset..offset + 2].try_into().ok()?,
            )),
            4 => i64::from(i32::from_le_bytes(
                self.data[offset..offset + 4].try_into().ok()?,
            )),
            8 => i64::from_le_bytes(self.data[offset..offset + 8].try_into().ok()?),
            _ => return None,
        })
    }

    fn read_uint(&self, offset: usize, size: usize) -> Option<u64> {
        self.require(offset, size)?;
        Some(match size {
            1 => u64::from(self.data[offset]),
            2 => u64::from(u16::from_le_bytes(
                self.data[offset..offset + 2].try_into().ok()?,
            )),
            4 => u64::from(u32::from_le_bytes(
                self.data[offset..offset + 4].try_into().ok()?,
            )),
            8 => u64::from_le_bytes(self.data[offset..offset + 8].try_into().ok()?),
            _ => return None,
        })
    }

    fn read_float(&self, offset: usize, size: usize) -> Option<f64> {
        self.require(offset, size)?;
        Some(match size {
            4 => f64::from(f32::from_le_bytes(
                self.data[offset..offset + 4].try_into().ok()?,
            )),
            8 => f64::from_le_bytes(self.data[offset..offset + 8].try_into().ok()?),
            _ => return None,
        })
    }

    fn string(&self, offset: usize, size: Option<usize>) -> Option<String> {
        let end = match size {
            Some(size) => offset.checked_add(size)?,
            None => self.data[offset..]
                .iter()
                .position(|value| *value == 0)
                .map(|position| offset + position)
                .unwrap_or(self.data.len()),
        };
        self.require(offset, end.checked_sub(offset)?)?;
        Some(String::from_utf8_lossy(&self.data[offset..end]).into_owned())
    }

    fn require(&self, offset: usize, len: usize) -> Option<()> {
        let end = offset.checked_add(len)?;
        (end <= self.data.len()).then_some(())
    }
}

fn builtin_operator_name(value: i32) -> Option<&'static str> {
    Some(match value {
        0 => "ADD",
        1 => "AVERAGE_POOL_2D",
        2 => "CONCATENATION",
        3 => "CONV_2D",
        4 => "DEPTHWISE_CONV_2D",
        5 => "DEPTH_TO_SPACE",
        6 => "DEQUANTIZE",
        7 => "EMBEDDING_LOOKUP",
        8 => "FLOOR",
        9 => "FULLY_CONNECTED",
        10 => "HASHTABLE_LOOKUP",
        11 => "L2_NORMALIZATION",
        12 => "L2_POOL_2D",
        13 => "LOCAL_RESPONSE_NORMALIZATION",
        14 => "LOGISTIC",
        15 => "LSH_PROJECTION",
        16 => "LSTM",
        17 => "MAX_POOL_2D",
        18 => "MUL",
        19 => "RELU",
        20 => "RELU_N1_TO_1",
        21 => "RELU6",
        22 => "RESHAPE",
        23 => "RESIZE_BILINEAR",
        24 => "RNN",
        25 => "SOFTMAX",
        26 => "SPACE_TO_DEPTH",
        27 => "SVDF",
        28 => "TANH",
        29 => "CONCAT_EMBEDDINGS",
        30 => "SKIP_GRAM",
        31 => "CALL",
        32 => "CUSTOM",
        33 => "EMBEDDING_LOOKUP_SPARSE",
        34 => "PAD",
        35 => "UNIDIRECTIONAL_SEQUENCE_RNN",
        36 => "GATHER",
        37 => "BATCH_TO_SPACE_ND",
        38 => "SPACE_TO_BATCH_ND",
        39 => "TRANSPOSE",
        40 => "MEAN",
        41 => "SUB",
        42 => "DIV",
        43 => "SQUEEZE",
        44 => "UNIDIRECTIONAL_SEQUENCE_LSTM",
        45 => "STRIDED_SLICE",
        46 => "BIDIRECTIONAL_SEQUENCE_RNN",
        47 => "EXP",
        48 => "TOPK_V2",
        49 => "SPLIT",
        50 => "LOG_SOFTMAX",
        51 => "DELEGATE",
        52 => "BIDIRECTIONAL_SEQUENCE_LSTM",
        53 => "CAST",
        54 => "PRELU",
        55 => "MAXIMUM",
        56 => "ARG_MAX",
        57 => "MINIMUM",
        58 => "LESS",
        59 => "NEG",
        60 => "PADV2",
        61 => "GREATER",
        62 => "GREATER_EQUAL",
        63 => "LESS_EQUAL",
        64 => "SELECT",
        65 => "SLICE",
        66 => "SIN",
        67 => "TRANSPOSE_CONV",
        68 => "SPARSE_TO_DENSE",
        69 => "TILE",
        70 => "EXPAND_DIMS",
        71 => "EQUAL",
        72 => "NOT_EQUAL",
        73 => "LOG",
        74 => "SUM",
        75 => "SQRT",
        76 => "RSQRT",
        77 => "SHAPE",
        78 => "POW",
        79 => "ARG_MIN",
        80 => "FAKE_QUANT",
        81 => "REDUCE_PROD",
        82 => "REDUCE_MAX",
        83 => "PACK",
        84 => "LOGICAL_OR",
        85 => "ONE_HOT",
        86 => "LOGICAL_AND",
        87 => "LOGICAL_NOT",
        88 => "UNPACK",
        89 => "REDUCE_MIN",
        90 => "FLOOR_DIV",
        91 => "REDUCE_ANY",
        92 => "SQUARE",
        93 => "ZEROS_LIKE",
        94 => "FILL",
        95 => "FLOOR_MOD",
        96 => "RANGE",
        97 => "RESIZE_NEAREST_NEIGHBOR",
        98 => "LEAKY_RELU",
        99 => "SQUARED_DIFFERENCE",
        100 => "MIRROR_PAD",
        101 => "ABS",
        102 => "SPLIT_V",
        103 => "UNIQUE",
        104 => "CEIL",
        105 => "REVERSE_V2",
        106 => "ADD_N",
        107 => "GATHER_ND",
        108 => "COS",
        109 => "WHERE",
        110 => "RANK",
        111 => "ELU",
        112 => "REVERSE_SEQUENCE",
        113 => "MATRIX_DIAG",
        114 => "QUANTIZE",
        115 => "MATRIX_SET_DIAG",
        116 => "ROUND",
        117 => "HARD_SWISH",
        118 => "IF",
        119 => "WHILE",
        120 => "NON_MAX_SUPPRESSION_V4",
        121 => "NON_MAX_SUPPRESSION_V5",
        122 => "SCATTER_ND",
        123 => "SELECT_V2",
        124 => "DENSIFY",
        125 => "SEGMENT_SUM",
        126 => "BATCH_MATMUL",
        127 => "PLACEHOLDER_FOR_GREATER_OP_CODES",
        128 => "CUMSUM",
        129 => "CALL_ONCE",
        130 => "BROADCAST_TO",
        131 => "RFFT2D",
        132 => "CONV_3D",
        133 => "IMAG",
        134 => "REAL",
        135 => "COMPLEX_ABS",
        136 => "HASHTABLE",
        137 => "HASHTABLE_FIND",
        138 => "HASHTABLE_IMPORT",
        139 => "HASHTABLE_SIZE",
        140 => "REDUCE_ALL",
        141 => "CONV_3D_TRANSPOSE",
        142 => "VAR_HANDLE",
        143 => "READ_VARIABLE",
        144 => "ASSIGN_VARIABLE",
        145 => "BROADCAST_ARGS",
        146 => "RANDOM_STANDARD_NORMAL",
        147 => "BUCKETIZE",
        148 => "RANDOM_UNIFORM",
        149 => "MULTINOMIAL",
        150 => "GELU",
        151 => "DYNAMIC_UPDATE_SLICE",
        152 => "RELU_0_TO_1",
        153 => "UNSORTED_SEGMENT_PROD",
        154 => "UNSORTED_SEGMENT_MAX",
        155 => "UNSORTED_SEGMENT_SUM",
        156 => "ATAN2",
        157 => "UNSORTED_SEGMENT_MIN",
        158 => "SIGN",
        159 => "BITCAST",
        160 => "BITWISE_XOR",
        161 => "RIGHT_SHIFT",
        162 => "STABLEHLO_LOGISTIC",
        163 => "STABLEHLO_ADD",
        164 => "STABLEHLO_DIVIDE",
        165 => "STABLEHLO_MULTIPLY",
        166 => "STABLEHLO_MAXIMUM",
        167 => "STABLEHLO_RESHAPE",
        168 => "STABLEHLO_CLAMP",
        169 => "STABLEHLO_CONCATENATE",
        170 => "STABLEHLO_BROADCAST_IN_DIM",
        171 => "STABLEHLO_CONVOLUTION",
        172 => "STABLEHLO_SLICE",
        173 => "STABLEHLO_CUSTOM_CALL",
        174 => "STABLEHLO_REDUCE",
        175 => "STABLEHLO_ABS",
        176 => "STABLEHLO_AND",
        177 => "STABLEHLO_COSINE",
        178 => "STABLEHLO_EXPONENTIAL",
        179 => "STABLEHLO_FLOOR",
        180 => "STABLEHLO_LOG",
        181 => "STABLEHLO_MINIMUM",
        182 => "STABLEHLO_NEGATE",
        183 => "STABLEHLO_OR",
        184 => "STABLEHLO_POWER",
        185 => "STABLEHLO_REMAINDER",
        186 => "STABLEHLO_RSQRT",
        187 => "STABLEHLO_SELECT",
        188 => "STABLEHLO_SUBTRACT",
        189 => "STABLEHLO_TANH",
        190 => "STABLEHLO_SCATTER",
        191 => "STABLEHLO_COMPARE",
        192 => "STABLEHLO_CONVERT",
        193 => "STABLEHLO_DYNAMIC_SLICE",
        194 => "STABLEHLO_DYNAMIC_UPDATE_SLICE",
        195 => "STABLEHLO_PAD",
        196 => "STABLEHLO_IOTA",
        197 => "STABLEHLO_DOT_GENERAL",
        198 => "STABLEHLO_REDUCE_WINDOW",
        199 => "STABLEHLO_SORT",
        200 => "STABLEHLO_WHILE",
        201 => "STABLEHLO_GATHER",
        202 => "STABLEHLO_TRANSPOSE",
        203 => "DILATE",
        204 => "STABLEHLO_RNG_BIT_GENERATOR",
        205 => "REDUCE_WINDOW",
        206 => "STABLEHLO_COMPOSITE",
        207 => "STABLEHLO_SHIFT_LEFT",
        208 => "STABLEHLO_CBRT",
        209 => "STABLEHLO_CASE",
        _ => return None,
    })
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
