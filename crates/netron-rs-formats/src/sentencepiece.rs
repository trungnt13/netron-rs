use netron_rs_core::{
    Attribute, AttributeValue, Confidence, FormatInfo, FormatMetadata, Graph, Model, ModelError,
    ModelFormat, ModelInput, Node, Operator,
};

const FORMAT: &str = "SentencePiece";

pub struct SentencePieceFormat;

impl ModelFormat for SentencePieceFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["model"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        if looks_like_sentencepiece(input.data) {
            Confidence::High
        } else {
            Confidence::None
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        lower_model(ModelProto::read(input.data)?)
    }
}

fn looks_like_sentencepiece(data: &[u8]) -> bool {
    let Ok(model) = ModelProto::read(data) else {
        return false;
    };
    model.piece_count > 0
        && model.first_piece_complete
        && model
            .trainer
            .as_ref()
            .is_some_and(|trainer| trainer.has_vocab_size)
        && model.normalizer.is_some()
}

fn lower_model(proto: ModelProto) -> Result<Model, ModelError> {
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);

    if proto.piece_count > 0 {
        let attrs = vec![AttributeSpec::strings(
            "pieces",
            object_placeholders(proto.piece_count),
        )];
        add_node(&mut model, &mut graph, "pieces", "SentencePiece[]", attrs);
    }
    if let Some(trainer) = proto.trainer {
        add_node(
            &mut model,
            &mut graph,
            "trainer_spec",
            "TrainerSpec",
            trainer.attributes(),
        );
    }
    if let Some(normalizer) = proto.normalizer {
        add_node(
            &mut model,
            &mut graph,
            "normalizer_spec",
            "NormalizerSpec",
            normalizer.attributes,
        );
    }
    if let Some(samples) = proto.self_test_samples {
        let attrs = vec![AttributeSpec::strings(
            "samples",
            object_placeholders(samples),
        )];
        add_node(
            &mut model,
            &mut graph,
            "self_test_data",
            "SelfTestData",
            attrs,
        );
    }
    if let Some(denormalizer) = proto.denormalizer {
        add_node(
            &mut model,
            &mut graph,
            "denormalizer_spec",
            "NormalizerSpec",
            denormalizer.attributes,
        );
    }

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn object_placeholders(count: usize) -> Vec<String> {
    std::iter::repeat_n("[object Object]".to_owned(), count.min(16)).collect()
}

fn add_node(
    model: &mut Model,
    graph: &mut Graph,
    name: &str,
    operator_name: &str,
    attributes: Vec<AttributeSpec>,
) {
    let mut node = Node::new(
        graph.id,
        Operator {
            domain: None,
            name: model.intern(operator_name),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.name = Some(model.intern(name));
    node.attributes = attributes
        .into_iter()
        .map(|attribute| Attribute {
            name: model.intern(attribute.name),
            value: attribute.value.into_value(model),
        })
        .collect();
    graph.add_node(node);
}

struct ModelProto {
    piece_count: usize,
    first_piece_complete: bool,
    trainer: Option<TrainerSpec>,
    normalizer: Option<NormalizerSpec>,
    self_test_samples: Option<usize>,
    denormalizer: Option<NormalizerSpec>,
}

impl ModelProto {
    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let mut reader = ProtoReader::new(data);
        let mut model = Self {
            piece_count: 0,
            first_piece_complete: false,
            trainer: None,
            normalizer: None,
            self_test_samples: None,
            denormalizer: None,
        };
        while !reader.is_empty() {
            let (field, wire_type) = reader.key()?;
            if wire_type != 2 || !(1..=5).contains(&field) {
                return Err(invalid("not a SentencePiece ModelProto"));
            }
            let bytes = reader.bytes()?;
            match field {
                1 => {
                    model.piece_count += 1;
                    if model.piece_count == 1 {
                        model.first_piece_complete = sentence_piece_complete(bytes);
                    }
                }
                2 => model.trainer = Some(TrainerSpec::read(bytes)?),
                3 => model.normalizer = Some(NormalizerSpec::read(bytes)?),
                4 => model.self_test_samples = Some(SelfTestData::sample_count(bytes)?),
                5 => model.denormalizer = Some(NormalizerSpec::read(bytes)?),
                _ => unreachable!("field range checked above"),
            }
        }
        Ok(model)
    }
}

fn sentence_piece_complete(data: &[u8]) -> bool {
    let mut reader = ProtoReader::new(data);
    let mut has_piece = false;
    let mut has_score = false;
    let mut has_type = false;
    while !reader.is_empty() {
        let Ok((field, wire_type)) = reader.key() else {
            return false;
        };
        match (field, wire_type) {
            (1, 2) => {
                if reader.bytes().is_err() {
                    return false;
                }
                has_piece = true;
            }
            (2, 5) => {
                if reader.fixed32_raw().is_err() {
                    return false;
                }
                has_score = true;
            }
            (3, 0) => {
                if reader.varint().is_err() {
                    return false;
                }
                has_type = true;
            }
            _ => {
                if reader.skip(wire_type).is_err() {
                    return false;
                }
            }
        }
    }
    has_piece && (has_score || has_type)
}

struct TrainerSpec {
    input: Vec<String>,
    accept_language: Vec<String>,
    control_symbols: Vec<String>,
    user_defined_symbols: Vec<String>,
    scalars: Vec<AttributeSpec>,
    has_vocab_size: bool,
}

impl TrainerSpec {
    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let mut reader = ProtoReader::new(data);
        let mut spec = Self {
            input: Vec::new(),
            accept_language: Vec::new(),
            control_symbols: Vec::new(),
            user_defined_symbols: Vec::new(),
            scalars: Vec::new(),
            has_vocab_size: false,
        };
        while !reader.is_empty() {
            let (field, wire_type) = reader.key()?;
            match field {
                1 => spec.input.push(reader.string(wire_type)?),
                5 => spec.accept_language.push(reader.string(wire_type)?),
                30 => spec.control_symbols.push(reader.string(wire_type)?),
                31 => spec.user_defined_symbols.push(reader.string(wire_type)?),
                7 => spec.push_string("input_format", reader.string(wire_type)?),
                2 => spec.push_string("model_prefix", reader.string(wire_type)?),
                53 => spec.push_string("pretokenization_delimiter", reader.string(wire_type)?),
                36 => spec.push_string("required_chars", reader.string(wire_type)?),
                45 => spec.push_string("unk_piece", reader.string(wire_type)?),
                46 => spec.push_string("bos_piece", reader.string(wire_type)?),
                47 => spec.push_string("eos_piece", reader.string(wire_type)?),
                48 => spec.push_string("pad_piece", reader.string(wire_type)?),
                44 => spec.push_string("unk_surface", reader.string(wire_type)?),
                54 => spec.push_string("seed_sentencepieces_file", reader.string(wire_type)?),
                3 => spec.push_int("model_type", reader.int32(wire_type)?),
                4 => {
                    spec.has_vocab_size = true;
                    spec.push_int("vocab_size", reader.int32(wire_type)?);
                }
                6 => spec.push_int("self_test_sample_size", reader.int32(wire_type)?),
                12 => spec.push_int("mining_sentence_size", reader.int32(wire_type)?),
                13 => spec.push_int("training_sentence_size", reader.int32(wire_type)?),
                14 => spec.push_int("seed_sentencepiece_size", reader.int32(wire_type)?),
                18 => spec.push_int("max_sentence_length", reader.int32(wire_type)?),
                16 => spec.push_int("num_threads", reader.int32(wire_type)?),
                17 => spec.push_int("num_sub_iterations", reader.int32(wire_type)?),
                20 => spec.push_int("max_sentencepiece_length", reader.int32(wire_type)?),
                40 => spec.push_int("unk_id", reader.int32(wire_type)?),
                41 => spec.push_int("bos_id", reader.int32(wire_type)?),
                42 => spec.push_int("eos_id", reader.int32(wire_type)?),
                43 => spec.push_int("pad_id", reader.int32(wire_type)?),
                11 => spec.push_string("input_sentence_size", reader.uint64_string(wire_type)?),
                52 => {
                    spec.push_string(
                        "differential_privacy_clipping_threshold",
                        reader.uint64_string(wire_type)?,
                    );
                }
                10 => spec.push_float_or_int("character_coverage", reader.float32(wire_type)?),
                15 => spec.push_float_or_int("shrinking_factor", reader.float32(wire_type)?),
                51 => spec.push_float_or_int(
                    "differential_privacy_noise_level",
                    reader.float32(wire_type)?,
                ),
                50 => spec.push_bool("enable_differential_privacy", reader.bool(wire_type)?),
                19 => spec.push_bool("shuffle_input_sentence", reader.bool(wire_type)?),
                21 => spec.push_bool("split_by_unicode_script", reader.bool(wire_type)?),
                23 => spec.push_bool("split_by_number", reader.bool(wire_type)?),
                22 => spec.push_bool("split_by_whitespace", reader.bool(wire_type)?),
                24 => spec.push_bool("treat_whitespace_as_suffix", reader.bool(wire_type)?),
                26 => spec.push_bool("allow_whitespace_only_pieces", reader.bool(wire_type)?),
                25 => spec.push_bool("split_digits", reader.bool(wire_type)?),
                35 => spec.push_bool("byte_fallback", reader.bool(wire_type)?),
                32 => spec.push_bool("vocabulary_output_piece_score", reader.bool(wire_type)?),
                33 => spec.push_bool("hard_vocab_limit", reader.bool(wire_type)?),
                34 => spec.push_bool("use_all_vocab", reader.bool(wire_type)?),
                49 => spec.push_bool("train_extremely_large_corpus", reader.bool(wire_type)?),
                _ => reader.skip(wire_type)?,
            }
        }
        Ok(spec)
    }

    fn attributes(self) -> Vec<AttributeSpec> {
        let mut attributes = vec![
            AttributeSpec::strings("input", self.input),
            AttributeSpec::strings("accept_language", self.accept_language),
            AttributeSpec::strings("control_symbols", self.control_symbols),
            AttributeSpec::strings("user_defined_symbols", self.user_defined_symbols),
        ];
        attributes.extend(self.scalars);
        attributes
    }

    fn push_string(&mut self, name: &'static str, value: String) {
        self.scalars.push(AttributeSpec::string(name, value));
    }

    fn push_int(&mut self, name: &'static str, value: i64) {
        self.scalars.push(AttributeSpec::int(name, value));
    }

    fn push_float(&mut self, name: &'static str, value: f32) {
        self.scalars.push(AttributeSpec::float(name, value));
    }

    fn push_float_or_int(&mut self, name: &'static str, value: f32) {
        if value.is_finite() && value.fract() == 0.0 {
            self.push_int(name, value as i64);
        } else {
            self.push_float(name, value);
        }
    }

    fn push_bool(&mut self, name: &'static str, value: bool) {
        self.scalars.push(AttributeSpec::bool(name, value));
    }
}

struct NormalizerSpec {
    attributes: Vec<AttributeSpec>,
}

impl NormalizerSpec {
    fn read(data: &[u8]) -> Result<Self, ModelError> {
        let mut reader = ProtoReader::new(data);
        let mut attributes = Vec::new();
        while !reader.is_empty() {
            let (field, wire_type) = reader.key()?;
            match field {
                1 => attributes.push(AttributeSpec::string("name", reader.string(wire_type)?)),
                2 => {
                    let bytes = reader.bytes_with_type(wire_type)?;
                    attributes.push(AttributeSpec::ints(
                        "precompiled_charsmap",
                        bytes
                            .iter()
                            .take(16)
                            .map(|value| i64::from(*value))
                            .collect(),
                    ));
                }
                3 => attributes.push(AttributeSpec::bool(
                    "add_dummy_prefix",
                    reader.bool(wire_type)?,
                )),
                4 => attributes.push(AttributeSpec::bool(
                    "remove_extra_whitespaces",
                    reader.bool(wire_type)?,
                )),
                5 => attributes.push(AttributeSpec::bool(
                    "escape_whitespaces",
                    reader.bool(wire_type)?,
                )),
                6 => attributes.push(AttributeSpec::string(
                    "normalization_rule_tsv",
                    reader.string(wire_type)?,
                )),
                _ => reader.skip(wire_type)?,
            }
        }
        Ok(Self { attributes })
    }
}

struct SelfTestData;

impl SelfTestData {
    fn sample_count(data: &[u8]) -> Result<usize, ModelError> {
        let mut reader = ProtoReader::new(data);
        let mut count = 0;
        while !reader.is_empty() {
            let (field, wire_type) = reader.key()?;
            if field == 1 && wire_type == 2 {
                reader.bytes()?;
                count += 1;
            } else {
                reader.skip(wire_type)?;
            }
        }
        Ok(count)
    }
}

struct AttributeSpec {
    name: &'static str,
    value: AttributeSpecValue,
}

impl AttributeSpec {
    fn bool(name: &'static str, value: bool) -> Self {
        Self {
            name,
            value: AttributeSpecValue::Bool(value),
        }
    }

    fn float(name: &'static str, value: f32) -> Self {
        Self {
            name,
            value: AttributeSpecValue::Float(comparable_f32(value)),
        }
    }

    fn int(name: &'static str, value: i64) -> Self {
        Self {
            name,
            value: AttributeSpecValue::Int(value),
        }
    }

    fn string(name: &'static str, value: String) -> Self {
        Self {
            name,
            value: AttributeSpecValue::String(value),
        }
    }

    fn strings(name: &'static str, value: Vec<String>) -> Self {
        Self {
            name,
            value: AttributeSpecValue::Strings(value),
        }
    }

    fn ints(name: &'static str, value: Vec<i64>) -> Self {
        Self {
            name,
            value: AttributeSpecValue::Ints(value),
        }
    }
}

fn comparable_f32(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 && value.fract() != 0.0 {
        f32::from_bits(value.to_bits().saturating_sub(1))
    } else {
        value
    }
}

enum AttributeSpecValue {
    Bool(bool),
    Float(f32),
    Int(i64),
    String(String),
    Strings(Vec<String>),
    Ints(Vec<i64>),
}

impl AttributeSpecValue {
    fn into_value(self, model: &mut Model) -> AttributeValue {
        match self {
            Self::Bool(value) => AttributeValue::Bool(value),
            Self::Float(value) => AttributeValue::Float(value),
            Self::Int(value) => AttributeValue::Int(value),
            Self::String(value) => AttributeValue::String(model.intern(value)),
            Self::Strings(values) => AttributeValue::Strings(
                values
                    .into_iter()
                    .map(|value| model.intern(value))
                    .collect(),
            ),
            Self::Ints(values) => AttributeValue::Ints(values),
        }
    }
}

struct ProtoReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset >= self.data.len()
    }

    fn key(&mut self) -> Result<(u32, u8), ModelError> {
        let value = self.varint()?;
        let field =
            u32::try_from(value >> 3).map_err(|_| invalid("protobuf field overflows u32"))?;
        let wire_type = (value & 7) as u8;
        if field == 0 {
            return Err(invalid("protobuf field 0 is invalid"));
        }
        Ok((field, wire_type))
    }

    fn string(&mut self, wire_type: u8) -> Result<String, ModelError> {
        let bytes = self.bytes_with_type(wire_type)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| invalid(format!("protobuf string: {error}")))
    }

    fn bytes_with_type(&mut self, wire_type: u8) -> Result<&'a [u8], ModelError> {
        if wire_type != 2 {
            return Err(invalid(format!(
                "expected length-delimited field, got wire type {wire_type}"
            )));
        }
        self.bytes()
    }

    fn bytes(&mut self) -> Result<&'a [u8], ModelError> {
        let len = usize::try_from(self.varint()?)
            .map_err(|_| invalid("protobuf length overflows usize"))?;
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid("protobuf length offset overflows usize"))?;
        let bytes = self
            .data
            .get(self.offset..end)
            .ok_or_else(|| invalid("protobuf length-delimited field is truncated"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn int32(&mut self, wire_type: u8) -> Result<i64, ModelError> {
        if wire_type != 0 {
            return Err(invalid(format!(
                "expected varint field, got wire type {wire_type}"
            )));
        }
        Ok(i64::from(self.varint()? as i32))
    }

    fn uint64_string(&mut self, wire_type: u8) -> Result<String, ModelError> {
        if wire_type != 0 {
            return Err(invalid(format!(
                "expected varint field, got wire type {wire_type}"
            )));
        }
        Ok(self.varint()?.to_string())
    }

    fn bool(&mut self, wire_type: u8) -> Result<bool, ModelError> {
        if wire_type != 0 {
            return Err(invalid(format!(
                "expected varint field, got wire type {wire_type}"
            )));
        }
        Ok(self.varint()? != 0)
    }

    fn float32(&mut self, wire_type: u8) -> Result<f32, ModelError> {
        if wire_type != 5 {
            return Err(invalid(format!(
                "expected fixed32 field, got wire type {wire_type}"
            )));
        }
        self.fixed32_raw()
    }

    fn fixed32_raw(&mut self) -> Result<f32, ModelError> {
        let bytes = self
            .data
            .get(self.offset..self.offset + 4)
            .ok_or_else(|| invalid("protobuf fixed32 field is truncated"))?;
        self.offset += 4;
        Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn varint(&mut self) -> Result<u64, ModelError> {
        let mut value = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .data
                .get(self.offset)
                .ok_or_else(|| invalid("protobuf varint is truncated"))?;
            self.offset += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(invalid("protobuf varint is too long"))
    }

    fn skip(&mut self, wire_type: u8) -> Result<(), ModelError> {
        match wire_type {
            0 => {
                self.varint()?;
                Ok(())
            }
            1 => self.advance(8),
            2 => {
                self.bytes()?;
                Ok(())
            }
            5 => self.advance(4),
            _ => Err(invalid(format!(
                "unsupported protobuf wire type {wire_type}"
            ))),
        }
    }

    fn advance(&mut self, len: usize) -> Result<(), ModelError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid("protobuf offset overflows usize"))?;
        if end > self.data.len() {
            return Err(invalid("protobuf field is truncated"));
        }
        self.offset = end;
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
