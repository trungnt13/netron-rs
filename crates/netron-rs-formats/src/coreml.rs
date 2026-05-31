use std::collections::HashMap;

use netron_rs_core::{
    Attribute, AttributeValue, Confidence, Dimension, DimensionValue, FormatInfo, FormatMetadata,
    Graph, Model, ModelError, ModelFormat, ModelInput, Node, NodeId, Operator, Quantization,
    QuantizationAnnotation, Tensor, TensorElementType, TensorStorage, TypeInfo, Value, ValueId,
};

const FORMAT: &str = "Core ML";

pub struct CoreMlFormat;

impl ModelFormat for CoreMlFormat {
    fn metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: FORMAT,
            extensions: &["mlmodel", "pb"],
        }
    }

    fn detect(&self, input: ModelInput<'_>) -> Confidence {
        let extension = input
            .path
            .and_then(|path| path.extension())
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase());

        match CoreMlModel::decode(input.data) {
            Ok(model) if model.is_coreml_like() && extension.as_deref() == Some("mlmodel") => {
                Confidence::High
            }
            Ok(model) if model.is_coreml_like() => Confidence::Medium,
            _ => Confidence::None,
        }
    }

    fn parse(&self, input: ModelInput<'_>) -> Result<Model, ModelError> {
        let target = CoreMlModel::decode(input.data)?;
        if !target.is_coreml_like() {
            return Err(invalid("data is not a Core ML Model protobuf"));
        }
        lower_model(target)
    }
}

fn lower_model(target: CoreMlModel) -> Result<Model, ModelError> {
    let mut description = target.description;
    let mut model = Model::new(FormatInfo {
        name: FORMAT,
        version: Some(target.specification_version.to_string()),
    });
    if let Some(metadata) = description.metadata.take() {
        model.metadata.producer_version = metadata.version_string;
        model.metadata.description = metadata.short_description;
        if let Some(author) = metadata.author {
            model
                .metadata
                .properties
                .insert("author".to_owned(), author);
        }
        if let Some(license) = metadata.license {
            model
                .metadata
                .properties
                .insert("license".to_owned(), license);
        }
    }

    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);
    graph.description = Some(target.kind.display_name().to_owned());
    let mut values = HashMap::new();
    for input in &description.inputs {
        let value_id = ensure_feature_value(&mut model, &mut graph, &mut values, input);
        graph.values[value_id.index()].is_graph_input = true;
        graph.inputs.push(value_id);
    }
    for output in &description.outputs {
        let value_id = ensure_feature_value(&mut model, &mut graph, &mut values, output);
        graph.values[value_id.index()].is_graph_output = true;
        graph.outputs.push(value_id);
    }

    lower_kind_nodes(
        &mut model,
        &mut graph,
        &mut values,
        description,
        target.kind,
    )?;

    model.replace_graph(graph_id, graph);
    Ok(model)
}

fn lower_kind_nodes(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    description: ModelDescription,
    kind: ModelKind,
) -> Result<(), ModelError> {
    match kind {
        ModelKind::Pipeline(pipeline) | ModelKind::PipelineClassifier(pipeline) => {
            for child in pipeline.models {
                for input in &child.description.inputs {
                    ensure_named_value(model, graph, values, &input.name);
                }
                for output in &child.description.outputs {
                    ensure_named_value(model, graph, values, &output.name);
                }
                lower_kind_nodes(model, graph, values, child.description, child.kind)?;
            }
        }
        ModelKind::DictVectorizer(vectorizer) => {
            let mut attributes = Vec::new();
            if let Some(values) = vectorizer.string_to_index {
                attributes.push(Attribute {
                    name: model.intern("stringToIndex"),
                    value: AttributeValue::Strings(
                        values
                            .into_iter()
                            .map(|value| model.intern(value))
                            .collect(),
                    ),
                });
            }
            if let Some(values) = vectorizer.int64_to_index {
                attributes.push(Attribute {
                    name: model.intern("int64ToIndex"),
                    value: AttributeValue::Strings(
                        values
                            .into_iter()
                            .map(|value| model.intern(value))
                            .collect(),
                    ),
                });
            }
            add_simple_node(
                model,
                graph,
                values,
                "dictVectorizer",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::FeatureVectorizer(vectorizer) => {
            let attributes = (!vectorizer.input_list.is_empty()).then(|| {
                vec![object_list_attribute(
                    model,
                    "inputList",
                    vectorizer.input_list.len(),
                )]
            });
            add_simple_node(
                model,
                graph,
                values,
                "featureVectorizer",
                description.inputs.iter().map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                attributes,
            );
        }
        ModelKind::Imputer(imputer) => {
            let mut attributes = Vec::new();
            if imputer.imputed_double_array {
                attributes.push(string_attribute(
                    model,
                    "imputedDoubleArray",
                    "[object Object]",
                ));
            }
            if let Some(value) = imputer.replace_double_value {
                attributes.push(Attribute {
                    name: model.intern("replaceDoubleValue"),
                    value: AttributeValue::Float(value as f32),
                });
            }
            add_simple_node(
                model,
                graph,
                values,
                "oneHotEncoder",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::OneHotEncoder(encoder) => {
            let mut attributes = Vec::new();
            if let Some(output_sparse) = encoder.output_sparse {
                attributes.push(Attribute {
                    name: model.intern("outputSparse"),
                    value: AttributeValue::Bool(output_sparse),
                });
            }
            if encoder.int64_categories {
                attributes.push(string_attribute(
                    model,
                    "int64Categories",
                    "[object Object]",
                ));
            }
            if encoder.string_categories {
                attributes.push(string_attribute(
                    model,
                    "stringCategories",
                    "[object Object]",
                ));
            }
            add_simple_node(
                model,
                graph,
                values,
                "oneHotEncoder",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::SupportVectorClassifier(classifier) => {
            let Some(output) = description.outputs.first() else {
                return Ok(());
            };
            let label_probability = format!("{}:labelProbabilityLayerName", output.name);
            let attributes = classifier.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "supportVectorClassifier",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                std::iter::once(label_probability.as_str()),
                Some(attributes),
            );
            if let Some(labels) = classifier.class_labels {
                add_classifier_label_node(
                    model,
                    graph,
                    values,
                    &description,
                    &label_probability,
                    labels,
                );
            }
        }
        ModelKind::SupportVectorRegressor(regressor) => {
            let attributes = regressor.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "supportVectorRegressor",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::TreeEnsembleClassifier(classifier) => {
            let Some(output) = description.outputs.first() else {
                return Ok(());
            };
            let label_probability = format!("{}:labelProbabilityLayerName", output.name);
            let attributes = classifier.tree_ensemble.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "treeEnsembleClassifier",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                std::iter::once(label_probability.as_str()),
                Some(attributes),
            );
            if let Some(labels) = classifier.class_labels {
                add_classifier_label_node(
                    model,
                    graph,
                    values,
                    &description,
                    &label_probability,
                    labels,
                );
            }
        }
        ModelKind::TreeEnsembleRegressor(regressor) => {
            let attributes = regressor.tree_ensemble.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "treeEnsembleRegressor",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::NeuralNetwork(network) => {
            let preprocessing = network.preprocessing.clone();
            for layer in network.layers {
                add_neural_layer_node(model, graph, values, layer)?;
            }
            add_preprocessing_nodes(model, graph, values, &description, preprocessing);
        }
        ModelKind::MlProgram(program) => {
            program.lower(model, graph, values)?;
        }
        ModelKind::NeuralNetworkClassifier(classifier) => {
            let preprocessing = classifier.network.preprocessing.clone();
            let mut last_output = None;
            for layer in classifier.network.layers {
                if layer.outputs.len() == 1 {
                    last_output = layer.outputs.first().cloned();
                } else {
                    last_output = None;
                }
                add_neural_layer_node(model, graph, values, layer)?;
            }
            if let Some(labels) = classifier.class_labels {
                let label_probability = if classifier.label_probability_layer_name.is_empty() {
                    last_output.unwrap_or_default()
                } else {
                    classifier.label_probability_layer_name
                };
                if !label_probability.is_empty() {
                    let label_probability = split_classifier_probability_value(
                        model,
                        graph,
                        values,
                        &label_probability,
                    );
                    add_classifier_label_node(
                        model,
                        graph,
                        values,
                        &description,
                        &label_probability,
                        labels,
                    );
                }
            }
            add_preprocessing_nodes(model, graph, values, &description, preprocessing);
        }
        ModelKind::Normalizer(normalizer) => {
            let attributes = normalizer.norm_type.map(|value| {
                vec![Attribute {
                    name: model.intern("normType"),
                    value: AttributeValue::Int(i64::from(value)),
                }]
            });
            add_simple_node(
                model,
                graph,
                values,
                "normalizer",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                attributes,
            );
        }
        ModelKind::ArrayFeatureExtractor(extractor) => {
            let attributes = (!extractor.extract_index.is_empty()).then(|| {
                vec![Attribute {
                    name: model.intern("extractIndex"),
                    value: AttributeValue::Strings(
                        extractor
                            .extract_index
                            .into_iter()
                            .map(|value| model.intern(value.to_string()))
                            .collect(),
                    ),
                }]
            });
            add_simple_node(
                model,
                graph,
                values,
                "arrayFeatureExtractor",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                attributes,
            );
        }
        ModelKind::NonMaximumSuppression(nms) => {
            let attributes = nms.attributes(model);
            let input_names = [
                nms.confidence_input_feature_name.as_str(),
                nms.coordinates_input_feature_name.as_str(),
                nms.iou_threshold_input_feature_name.as_str(),
                nms.confidence_threshold_input_feature_name.as_str(),
            ];
            let output_names = [
                nms.confidence_output_feature_name.as_str(),
                nms.coordinates_output_feature_name.as_str(),
            ];
            add_simple_node(
                model,
                graph,
                values,
                "nonMaximumSuppression",
                input_names.into_iter().filter(|name| !name.is_empty()),
                output_names.into_iter().filter(|name| !name.is_empty()),
                Some(attributes),
            );
        }
        ModelKind::GlmRegressor(regressor) => {
            add_glm_regressor_node(model, graph, values, &description, regressor);
        }
        ModelKind::GlmClassifier(classifier) => {
            add_glm_classifier_node(model, graph, values, &description, classifier);
        }
        ModelKind::KNearestNeighborsClassifier(classifier) => {
            let Some(output) = description.outputs.first() else {
                return Ok(());
            };
            let label_probability = format!("{}:labelProbabilityLayerName", output.name);
            let attributes = classifier.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "kNearestNeighborsClassifier",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                std::iter::once(label_probability.as_str()),
                Some(attributes),
            );
            if let Some(labels) = classifier.class_labels {
                add_classifier_label_node(
                    model,
                    graph,
                    values,
                    &description,
                    &label_probability,
                    labels,
                );
            }
        }
        ModelKind::ItemSimilarityRecommender(recommender) => {
            let attributes = recommender.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "itemSimilarityRecommender",
                description.inputs.iter().map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::Scaler(scaler) => {
            let attributes = scaler.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "scaler",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::CustomModel(custom_model) => {
            let attributes = custom_model.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "customModel",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::LinkedModel(linked_model) => {
            let attributes = linked_model.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "linkedModel",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::WordTagger(tagger) => {
            let attributes = tagger.attributes(model);
            let output_names = [
                tagger.tokens_output_feature_name.as_str(),
                tagger.token_tags_output_feature_name.as_str(),
                tagger.token_locations_output_feature_name.as_str(),
                tagger.token_lengths_output_feature_name.as_str(),
            ];
            add_simple_node(
                model,
                graph,
                values,
                "wordTagger",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                output_names.into_iter().filter(|name| !name.is_empty()),
                Some(attributes),
            );
        }
        ModelKind::TextClassifier(classifier) => {
            let attributes = classifier.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "textClassifier",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::SoundAnalysisPreprocessing(preprocessing) => {
            let attributes = preprocessing.attributes(model);
            add_simple_node(
                model,
                graph,
                values,
                "soundAnalysisPreprocessing",
                description
                    .inputs
                    .iter()
                    .take(1)
                    .map(|input| input.name.as_str()),
                description
                    .outputs
                    .iter()
                    .take(1)
                    .map(|output| output.name.as_str()),
                Some(attributes),
            );
        }
        ModelKind::Unsupported(name) => {
            return Err(invalid(format!("unsupported Core ML model type '{name}'")));
        }
    }
    Ok(())
}

fn ensure_feature_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    feature: &FeatureDescription,
) -> ValueId {
    if let Some(value_id) = values.get(&feature.name) {
        return *value_id;
    }
    let name = model.intern(&feature.name);
    let mut value = Value::new(name);
    value.description = feature
        .description
        .clone()
        .filter(|value| !value.is_empty());
    value.type_info = feature.feature_type.as_ref().and_then(lower_feature_type);
    let value_id = graph.add_value(value);
    values.insert(feature.name.clone(), value_id);
    value_id
}

fn ensure_named_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    name: &str,
) -> ValueId {
    if let Some(value_id) = values.get(name) {
        return *value_id;
    }
    let value_id = graph.add_value(Value::new(model.intern(name)));
    values.insert(name.to_owned(), value_id);
    value_id
}

fn add_neural_layer_node(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    layer: NeuralLayer,
) -> Result<(), ModelError> {
    let operator = layer.kind.operator_name();
    if operator.is_empty() {
        return Err(invalid("unsupported Core ML neural network layer"));
    }
    let mut node = Node::new(
        graph.id,
        Operator {
            domain: None,
            name: model.intern(operator),
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    node.name = (!layer.name.is_empty()).then(|| model.intern(&layer.name));

    for input in &layer.inputs {
        let value_id = ensure_named_value(model, graph, values, input);
        node.inputs.push(Some(value_id));
    }
    let (attributes, initializers) = layer.kind.lower(model, graph);
    for initializer in initializers {
        node.inputs.push(Some(initializer));
    }
    for output in &layer.outputs {
        node.outputs
            .push(output_value(model, graph, values, output));
    }
    node.attributes = attributes;

    let inputs = node.inputs.iter().flatten().copied().collect::<Vec<_>>();
    let outputs = node.outputs.iter().flatten().copied().collect::<Vec<_>>();
    let node_id = graph.add_node(node);
    for input in inputs {
        let consumers = &mut graph.values[input.index()].consumers;
        if !consumers.contains(&node_id) {
            consumers.push(node_id);
        }
    }
    for output in outputs {
        graph.values[output.index()].producer = Some(node_id);
    }
    Ok(())
}

fn add_glm_regressor_node(
    model: &mut Model,
    graph: &mut Graph,
    values: &HashMap<String, ValueId>,
    description: &ModelDescription,
    regressor: GlmRegressor,
) {
    let Some(input) = description.inputs.first() else {
        return;
    };
    let Some(output) = description.outputs.first() else {
        return;
    };
    let name = model.intern("glmRegressor");
    let mut node = Node::new(
        graph.id,
        Operator {
            domain: None,
            name,
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    if let Some(value_id) = values.get(&input.name) {
        node.inputs.push(Some(*value_id));
    }
    if let Some(value_id) = values.get(&output.name) {
        node.outputs.push(Some(*value_id));
    }
    if !regressor.weights.is_empty() {
        let name = model.intern("weights");
        let object = model.intern("[object Object]");
        node.attributes.push(Attribute {
            name,
            value: AttributeValue::Strings(regressor.weights.iter().map(|_| object).collect()),
        });
    }
    if !regressor.offset.is_empty() {
        node.attributes.push(Attribute {
            name: model.intern("offset"),
            value: AttributeValue::Floats(
                regressor
                    .offset
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
            ),
        });
    }
    let inputs = node.inputs.iter().flatten().copied().collect::<Vec<_>>();
    let outputs = node.outputs.iter().flatten().copied().collect::<Vec<_>>();
    let node_id = graph.add_node(node);
    for input in inputs {
        graph.values[input.index()].consumers.push(node_id);
    }
    for output in outputs {
        graph.values[output.index()].producer = Some(node_id);
    }
}

fn add_glm_classifier_node(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    description: &ModelDescription,
    classifier: GlmClassifier,
) {
    let Some(output) = description.outputs.first() else {
        return;
    };
    let label_probability = format!("{}:labelProbabilityLayerName", output.name);
    let mut attributes = Vec::new();
    attributes.push(Attribute {
        name: model.intern("classEncoding"),
        value: AttributeValue::Int(i64::from(classifier.class_encoding)),
    });
    if !classifier.offset.is_empty() {
        attributes.push(Attribute {
            name: model.intern("offset"),
            value: AttributeValue::Floats(
                classifier
                    .offset
                    .iter()
                    .map(|value| *value as f32)
                    .collect(),
            ),
        });
    }
    if !classifier.weights.is_empty() {
        attributes.push(object_list_attribute(
            model,
            "weights",
            classifier.weights.len(),
        ));
    }
    add_simple_node(
        model,
        graph,
        values,
        "glmClassifier",
        description
            .inputs
            .iter()
            .take(1)
            .map(|input| input.name.as_str()),
        std::iter::once(label_probability.as_str()),
        Some(attributes),
    );
    if let Some(labels) = classifier.class_labels {
        add_classifier_label_node(
            model,
            graph,
            values,
            description,
            &label_probability,
            labels,
        );
    }
}

fn add_classifier_label_node(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    description: &ModelDescription,
    label_probability: &str,
    labels: ClassLabels,
) {
    if description.predicted_feature_name.is_empty()
        && description.predicted_probabilities_name.is_empty()
    {
        return;
    }

    let predicted_probabilities = if description.predicted_probabilities_name.is_empty() {
        "?"
    } else {
        description.predicted_probabilities_name.as_str()
    };
    let predicted_feature = if description.predicted_feature_name.is_empty() {
        "?"
    } else {
        description.predicted_feature_name.as_str()
    };
    let name = model.intern(labels.operator());
    let mut node = Node::new(
        graph.id,
        Operator {
            domain: None,
            name,
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    if let Some(input) = values.get(label_probability) {
        node.inputs.push(Some(*input));
    }
    node.outputs
        .push(output_value(model, graph, values, predicted_probabilities));
    node.outputs
        .push(output_value(model, graph, values, predicted_feature));
    let label_values = labels.values();
    node.attributes.push(Attribute {
        name: model.intern("vector"),
        value: if label_values.is_empty() {
            AttributeValue::Ints(Vec::new())
        } else {
            AttributeValue::Strings(
                label_values
                    .into_iter()
                    .map(|value| model.intern(value))
                    .collect(),
            )
        },
    });

    let inputs = node.inputs.iter().flatten().copied().collect::<Vec<_>>();
    let outputs = node.outputs.iter().flatten().copied().collect::<Vec<_>>();
    let node_id = graph.add_node(node);
    for input in inputs {
        let consumers = &mut graph.values[input.index()].consumers;
        if !consumers.contains(&node_id) {
            consumers.push(node_id);
        }
    }
    for output in outputs {
        graph.values[output.index()].producer = Some(node_id);
    }
}

fn add_simple_node<'a>(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    operator: &str,
    inputs: impl Iterator<Item = &'a str>,
    outputs: impl Iterator<Item = &'a str>,
    attributes: Option<Vec<Attribute>>,
) {
    let name = model.intern(operator);
    let mut node = Node::new(
        graph.id,
        Operator {
            domain: None,
            name,
            overload: None,
            version: None,
            origin: FORMAT,
        },
    );
    for input in inputs {
        if let Some(value_id) = values.get(input) {
            node.inputs.push(Some(*value_id));
        }
    }
    for output in outputs {
        node.outputs
            .push(output_value(model, graph, values, output));
    }
    node.attributes = attributes.unwrap_or_default();
    let inputs = node.inputs.iter().flatten().copied().collect::<Vec<_>>();
    let outputs = node.outputs.iter().flatten().copied().collect::<Vec<_>>();
    let node_id = graph.add_node(node);
    for input in inputs {
        let consumers = &mut graph.values[input.index()].consumers;
        if !consumers.contains(&node_id) {
            consumers.push(node_id);
        }
    }
    for output in outputs {
        graph.values[output.index()].producer = Some(node_id);
    }
}

fn split_classifier_probability_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    name: &str,
) -> String {
    let label_probability = format!("{name}:labelProbabilityLayerName");
    let Some(value_id) = values.get(name).copied() else {
        return label_probability;
    };
    let mut split_value = Value::new(model.intern(&label_probability));
    split_value.producer = graph.values[value_id.index()].producer;
    split_value.consumers = std::mem::take(&mut graph.values[value_id.index()].consumers);
    let split_value_id = graph.add_value(split_value);

    if let Some(producer) = graph.values[split_value_id.index()].producer {
        for output in &mut graph.nodes[producer.index()].outputs {
            if *output == Some(value_id) {
                *output = Some(split_value_id);
            }
        }
    }
    for consumer in graph.values[split_value_id.index()].consumers.clone() {
        for input in &mut graph.nodes[consumer.index()].inputs {
            if *input == Some(value_id) {
                *input = Some(split_value_id);
            }
        }
    }
    graph.values[value_id.index()].producer = None;
    values.insert(label_probability.clone(), split_value_id);
    values.insert(name.to_owned(), value_id);
    label_probability
}

fn add_preprocessing_nodes(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    description: &ModelDescription,
    preprocessing: Vec<NeuralNetworkPreprocessing>,
) {
    if preprocessing.is_empty() {
        return;
    }
    let Some(input) = description.inputs.first() else {
        return;
    };
    let Some(&input_id) = values.get(&input.name) else {
        return;
    };
    let mut current_input = input.name.clone();
    for (index, preprocessing) in preprocessing.into_iter().enumerate() {
        let output_name = format!("{}:{index}", input.name);
        let preprocessor_input = if preprocessing.feature_name.is_empty() {
            current_input.as_str()
        } else {
            preprocessing.feature_name.as_str()
        };
        let (operator, attributes) = match preprocessing.kind {
            PreprocessingKind::Scaler(scaler) => ("scaler", scaler.attributes(model)),
            PreprocessingKind::MeanImage(mean) => ("meanImage", mean.attributes(model)),
            PreprocessingKind::Unsupported => continue,
        };
        let output_id = output_value(model, graph, values, &output_name)
            .expect("preprocessing output is always materialized");
        add_simple_node(
            model,
            graph,
            values,
            operator,
            std::iter::once(preprocessor_input),
            std::iter::once(output_name.as_str()),
            Some(attributes),
        );
        let Some(node_id) = graph.values[output_id.index()].producer else {
            current_input = output_name;
            continue;
        };
        let prior_nodes = graph.nodes.len().saturating_sub(1);
        let mut consumers = Vec::new();
        for node_index in 0..prior_nodes {
            for node_input in &mut graph.nodes[node_index].inputs {
                if *node_input == Some(input_id) {
                    *node_input = Some(output_id);
                    consumers.push(NodeId::new(node_index));
                }
            }
        }
        graph.values[input_id.index()]
            .consumers
            .retain(|consumer| *consumer == node_id);
        graph.values[output_id.index()].consumers = consumers;
        current_input = output_name;
    }
}

fn output_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    name: &str,
) -> Option<ValueId> {
    let Some(existing) = values.get(name).copied() else {
        let value_id = graph.add_value(Value::new(model.intern(name)));
        values.insert(name.to_owned(), value_id);
        return Some(value_id);
    };
    if graph.values[existing.index()].is_graph_output {
        return Some(existing);
    }
    if graph.values[existing.index()].producer.is_none()
        && !graph.values[existing.index()].is_graph_input
    {
        return Some(existing);
    }
    let mut suffix = 1;
    let duplicate_name = loop {
        let candidate = format!("{name}|{suffix}");
        if !values.contains_key(&candidate) {
            break candidate;
        }
        suffix += 1;
    };
    let value_id = graph.add_value(Value::new(model.intern(&duplicate_name)));
    values.insert(name.to_owned(), value_id);
    values.insert(duplicate_name, value_id);
    Some(value_id)
}

fn object_list_attribute(model: &mut Model, name: &str, len: usize) -> Attribute {
    let object = model.intern("[object Object]");
    Attribute {
        name: model.intern(name),
        value: AttributeValue::Strings((0..len).map(|_| object).collect()),
    }
}

fn string_attribute(model: &mut Model, name: &str, value: &str) -> Attribute {
    let value = model.intern(value);
    Attribute {
        name: model.intern(name),
        value: AttributeValue::String(value),
    }
}

fn string_vector_attribute(model: &mut Model, name: &str, values: &[String]) -> Attribute {
    Attribute {
        name: model.intern(name),
        value: if values.is_empty() {
            AttributeValue::Ints(Vec::new())
        } else {
            AttributeValue::Strings(values.iter().map(|value| model.intern(value)).collect())
        },
    }
}

fn numeric_array_attribute(model: &mut Model, name: &str, values: &[f64]) -> Attribute {
    let integer_values = values
        .iter()
        .all(|value| value.is_finite() && value.fract() == 0.0);
    Attribute {
        name: model.intern(name),
        value: if integer_values {
            AttributeValue::Ints(values.iter().map(|value| *value as i64).collect())
        } else {
            AttributeValue::Floats(values.iter().map(|value| *value as f32).collect())
        },
    }
}

fn int_attribute(model: &mut Model, name: &str, value: i64) -> Attribute {
    Attribute {
        name: model.intern(name),
        value: AttributeValue::Int(value),
    }
}

fn f32_scalar_attribute(model: &mut Model, name: &str, value: f32) -> Attribute {
    Attribute {
        name: model.intern(name),
        value: if value.is_finite() && value.fract() == 0.0 {
            AttributeValue::Int(value as i64)
        } else {
            AttributeValue::Float(value)
        },
    }
}

fn uint_array_attribute(model: &mut Model, name: &str, values: &[u64]) -> Attribute {
    Attribute {
        name: model.intern(name),
        value: AttributeValue::Ints(values.iter().map(|value| *value as i64).collect()),
    }
}

fn string_array_attribute(model: &mut Model, name: &str, values: &[u64]) -> Attribute {
    Attribute {
        name: model.intern(name),
        value: if values.is_empty() {
            AttributeValue::Ints(Vec::new())
        } else {
            AttributeValue::Strings(
                values
                    .iter()
                    .map(|value| model.intern(value.to_string()))
                    .collect(),
            )
        },
    }
}

fn weight_initializer(
    model: &mut Model,
    graph: &mut Graph,
    weight: WeightParams,
    shape: Vec<i64>,
) -> ValueId {
    let element_type = weight.element_type.unwrap_or(TensorElementType::Float32);
    let storage = match weight.storage {
        WeightStorage::ElementList { len } => TensorStorage::ElementList { len },
        WeightStorage::InlineBytes { byte_len } => TensorStorage::InlineBytes { byte_len },
        WeightStorage::Absent => TensorStorage::Absent,
    };
    let mut tensor = Tensor::metadata_only(
        None,
        element_type.clone(),
        shape.iter().copied().map(Dimension::known).collect(),
        storage,
    );
    if weight.quantization.is_some() {
        tensor.quantization = Some(Quantization {
            scale: None,
            zero_point: None,
        });
    }
    let tensor_id = model.add_tensor(tensor);

    let mut value = Value::new(model.intern(""));
    value.type_info = Some(TypeInfo {
        element_type: Some(element_type),
        layout: None,
        denotation: None,
        shape: shape.into_iter().map(Dimension::known).collect(),
    });
    value.initializer = Some(tensor_id);
    if let Some(quantization) = weight.quantization {
        value.quantization = quantization.annotations(model);
    }
    graph.add_value(value)
}

fn lower_feature_type(feature_type: &FeatureType) -> Option<TypeInfo> {
    match feature_type {
        FeatureType::MultiArray(array) => Some(TypeInfo {
            element_type: Some(array_data_type(array.data_type)),
            layout: None,
            denotation: None,
            shape: array
                .shape
                .iter()
                .map(|dimension| Dimension::known(*dimension))
                .collect(),
        }),
        FeatureType::Double => Some(TypeInfo {
            element_type: Some(TensorElementType::Float64),
            layout: None,
            denotation: None,
            shape: Vec::new(),
        }),
        FeatureType::Int64 => Some(TypeInfo {
            element_type: Some(TensorElementType::Int64),
            layout: None,
            denotation: None,
            shape: Vec::new(),
        }),
        FeatureType::String => Some(TypeInfo {
            element_type: Some(TensorElementType::String),
            layout: None,
            denotation: None,
            shape: Vec::new(),
        }),
        FeatureType::Dictionary
        | FeatureType::Image
        | FeatureType::Optional
        | FeatureType::Sequence
        | FeatureType::State => Some(TypeInfo {
            element_type: None,
            layout: None,
            denotation: None,
            shape: Vec::new(),
        }),
        FeatureType::Unknown => None,
    }
}

fn array_data_type(value: i32) -> TensorElementType {
    match value {
        65_552 => TensorElementType::Float16,
        65_568 => TensorElementType::Float32,
        65_600 => TensorElementType::Float64,
        131_080 => TensorElementType::Int8,
        131_104 => TensorElementType::Int32,
        0 => TensorElementType::Other("?".to_owned()),
        value => TensorElementType::Other(value.to_string()),
    }
}

#[derive(Default)]
struct CoreMlModel {
    specification_version: u32,
    description: ModelDescription,
    kind: ModelKind,
}

impl CoreMlModel {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.specification_version = field.uint32()?,
                2 => result.description = ModelDescription::decode(field.bytes()?)?,
                300 => result.kind = ModelKind::GlmRegressor(GlmRegressor::decode(field.bytes()?)?),
                200 => {
                    result.kind =
                        ModelKind::PipelineClassifier(Pipeline::decode_classifier(field.bytes()?)?)
                }
                201 => result.kind = ModelKind::Unsupported("pipelineRegressor"),
                202 => result.kind = ModelKind::Pipeline(Pipeline::decode(field.bytes()?)?),
                301 => {
                    result.kind = ModelKind::SupportVectorRegressor(SupportVectorRegressor::decode(
                        field.bytes()?,
                    )?)
                }
                302 => {
                    result.kind = ModelKind::TreeEnsembleRegressor(TreeEnsembleRegressor::decode(
                        field.bytes()?,
                    )?)
                }
                303 => result.kind = ModelKind::Unsupported("neuralNetworkRegressor"),
                304 => result.kind = ModelKind::Unsupported("bayesianProbitRegressor"),
                400 => {
                    result.kind = ModelKind::GlmClassifier(GlmClassifier::decode(field.bytes()?)?)
                }
                401 => {
                    result.kind = ModelKind::SupportVectorClassifier(
                        SupportVectorClassifier::decode(field.bytes()?)?,
                    )
                }
                402 => {
                    result.kind = ModelKind::TreeEnsembleClassifier(TreeEnsembleClassifier::decode(
                        field.bytes()?,
                    )?)
                }
                403 => {
                    result.kind = ModelKind::NeuralNetworkClassifier(
                        NeuralNetworkClassifier::decode(field.bytes()?)?,
                    )
                }
                404 => {
                    result.kind = ModelKind::KNearestNeighborsClassifier(
                        KNearestNeighborsClassifier::decode(field.bytes()?)?,
                    )
                }
                500 => {
                    result.kind = ModelKind::NeuralNetwork(NeuralNetwork::decode(field.bytes()?)?)
                }
                501 => {
                    result.kind = ModelKind::ItemSimilarityRecommender(
                        ItemSimilarityRecommender::decode(field.bytes()?)?,
                    )
                }
                502 => result.kind = ModelKind::MlProgram(MlProgram::decode(field.bytes()?)?),
                555 => result.kind = ModelKind::CustomModel(CustomModel::decode(field.bytes()?)?),
                556 => result.kind = ModelKind::LinkedModel(LinkedModel::decode(field.bytes()?)?),
                560 => result.kind = ModelKind::Unsupported("classConfidenceThresholding"),
                600 => {
                    result.kind = ModelKind::OneHotEncoder(OneHotEncoder::decode(field.bytes()?)?)
                }
                601 => result.kind = ModelKind::Imputer(Imputer::decode(field.bytes()?)?),
                602 => {
                    result.kind =
                        ModelKind::FeatureVectorizer(FeatureVectorizer::decode(field.bytes()?)?)
                }
                603 => {
                    result.kind = ModelKind::DictVectorizer(DictVectorizer::decode(field.bytes()?)?)
                }
                604 => result.kind = ModelKind::Scaler(Scaler::decode(field.bytes()?)?),
                606 => result.kind = ModelKind::Unsupported("categoricalMapping"),
                607 => result.kind = ModelKind::Normalizer(Normalizer::decode(field.bytes()?)?),
                609 => {
                    result.kind = ModelKind::ArrayFeatureExtractor(ArrayFeatureExtractor::decode(
                        field.bytes()?,
                    )?)
                }
                610 => {
                    result.kind = ModelKind::NonMaximumSuppression(NonMaximumSuppression::decode(
                        field.bytes()?,
                    )?)
                }
                900 => result.kind = ModelKind::Unsupported("identity"),
                2000 => {
                    result.kind = ModelKind::TextClassifier(TextClassifier::decode(field.bytes()?)?)
                }
                2001 => result.kind = ModelKind::WordTagger(WordTagger::decode(field.bytes()?)?),
                2002 => result.kind = ModelKind::Unsupported("visionFeaturePrint"),
                2003 => {
                    result.kind = ModelKind::SoundAnalysisPreprocessing(
                        SoundAnalysisPreprocessing::decode(field.bytes()?)?,
                    )
                }
                2004 => result.kind = ModelKind::Unsupported("gazetteer"),
                2005 => result.kind = ModelKind::Unsupported("wordEmbedding"),
                2006 => result.kind = ModelKind::Unsupported("audioFeaturePrint"),
                3000 => result.kind = ModelKind::Unsupported("serializedModel"),
                _ => {}
            }
        }
        Ok(result)
    }

    fn is_coreml_like(&self) -> bool {
        self.specification_version > 0
            && (!self.description.inputs.is_empty()
                || !self.description.outputs.is_empty()
                || !matches!(self.kind, ModelKind::Unsupported("")))
    }
}

enum ModelKind {
    Pipeline(Pipeline),
    PipelineClassifier(Pipeline),
    FeatureVectorizer(FeatureVectorizer),
    DictVectorizer(DictVectorizer),
    Imputer(Imputer),
    OneHotEncoder(OneHotEncoder),
    SupportVectorClassifier(SupportVectorClassifier),
    SupportVectorRegressor(SupportVectorRegressor),
    TreeEnsembleClassifier(TreeEnsembleClassifier),
    TreeEnsembleRegressor(TreeEnsembleRegressor),
    NeuralNetwork(NeuralNetwork),
    MlProgram(MlProgram),
    NeuralNetworkClassifier(NeuralNetworkClassifier),
    Normalizer(Normalizer),
    ArrayFeatureExtractor(ArrayFeatureExtractor),
    GlmRegressor(GlmRegressor),
    GlmClassifier(GlmClassifier),
    KNearestNeighborsClassifier(KNearestNeighborsClassifier),
    ItemSimilarityRecommender(ItemSimilarityRecommender),
    NonMaximumSuppression(NonMaximumSuppression),
    Scaler(Scaler),
    CustomModel(CustomModel),
    LinkedModel(LinkedModel),
    WordTagger(WordTagger),
    TextClassifier(TextClassifier),
    SoundAnalysisPreprocessing(SoundAnalysisPreprocessing),
    Unsupported(&'static str),
}

impl Default for ModelKind {
    fn default() -> Self {
        Self::Unsupported("")
    }
}

impl ModelKind {
    fn display_name(&self) -> &'static str {
        match self {
            Self::Pipeline(_) => "Pipeline",
            Self::PipelineClassifier(_) => "Pipeline Classifier",
            Self::FeatureVectorizer(_) => "Feature Vectorizer",
            Self::DictVectorizer(_) => "Dictionary Vectorizer",
            Self::Imputer(_) => "Imputer",
            Self::OneHotEncoder(_) => "One Hot Encoder",
            Self::SupportVectorClassifier(_) => "Support Vector Classifier",
            Self::SupportVectorRegressor(_) => "Support Vector Regressor",
            Self::TreeEnsembleClassifier(_) => "Tree Ensemble Classifier",
            Self::TreeEnsembleRegressor(_) => "Tree Ensemble Regressor",
            Self::NeuralNetwork(_) => "Neural Network",
            Self::MlProgram(_) => "ML Program",
            Self::NeuralNetworkClassifier(_) => "Neural Network Classifier",
            Self::Normalizer(_) => "Normalizer",
            Self::ArrayFeatureExtractor(_) => "Array Feature Extractor",
            Self::GlmRegressor(_) => "Generalized Linear Regressor",
            Self::GlmClassifier(_) => "Generalized Linear Classifier",
            Self::KNearestNeighborsClassifier(_) => "Nearest Neighbors Classifier",
            Self::ItemSimilarityRecommender(_) => "Item Similarity Recommender",
            Self::NonMaximumSuppression(_) => "Non Maximum Suppression",
            Self::Scaler(_) => "Scaler",
            Self::CustomModel(_) => "customModel",
            Self::LinkedModel(_) => "Linked Model",
            Self::WordTagger(_) => "Word Tagger",
            Self::TextClassifier(_) => "Text Classifier",
            Self::SoundAnalysisPreprocessing(_) => "Sound Analysis Preprocessing",
            Self::Unsupported(name) => name,
        }
    }
}

#[derive(Default)]
struct ModelDescription {
    inputs: Vec<FeatureDescription>,
    outputs: Vec<FeatureDescription>,
    predicted_feature_name: String,
    predicted_probabilities_name: String,
    metadata: Option<Metadata>,
}

impl ModelDescription {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result
                    .inputs
                    .push(FeatureDescription::decode(field.bytes()?)?),
                10 => result
                    .outputs
                    .push(FeatureDescription::decode(field.bytes()?)?),
                11 => result.predicted_feature_name = field.string()?,
                12 => result.predicted_probabilities_name = field.string()?,
                100 => result.metadata = Some(Metadata::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct Metadata {
    short_description: Option<String>,
    version_string: Option<String>,
    author: Option<String>,
    license: Option<String>,
    user_defined: std::collections::BTreeMap<String, String>,
}

impl Metadata {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.short_description = Some(field.string()?),
                2 => result.version_string = Some(field.string()?),
                3 => result.author = Some(field.string()?),
                4 => result.license = Some(field.string()?),
                100 => {
                    let (key, value) = read_string_map_entry(field.bytes()?)?;
                    if let Some(key) = key {
                        result.user_defined.insert(key, value.unwrap_or_default());
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

fn read_string_map_entry(data: &[u8]) -> Result<(Option<String>, Option<String>), ModelError> {
    let mut key = None;
    let mut value = None;
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => key = Some(field.string()?),
            2 => value = Some(field.string()?),
            _ => {}
        }
    }
    Ok((key, value))
}

#[derive(Default)]
struct FeatureDescription {
    name: String,
    description: Option<String>,
    feature_type: Option<FeatureType>,
}

impl FeatureDescription {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.name = field.string()?,
                2 => result.description = Some(field.string()?),
                3 => result.feature_type = Some(FeatureType::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Clone)]
enum FeatureType {
    MultiArray(ArrayFeatureType),
    Double,
    Int64,
    String,
    Image,
    Dictionary,
    Sequence,
    State,
    Optional,
    Unknown,
}

impl FeatureType {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::Unknown;
        let mut optional = false;
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result = Self::Int64,
                2 => result = Self::Double,
                3 => result = Self::String,
                4 => {
                    field.bytes()?;
                    result = Self::Image;
                }
                5 => result = Self::MultiArray(ArrayFeatureType::decode(field.bytes()?)?),
                6 => {
                    field.bytes()?;
                    result = Self::Dictionary;
                }
                7 => {
                    field.bytes()?;
                    result = Self::Sequence;
                }
                8 => {
                    field.bytes()?;
                    result = Self::State;
                }
                1000 => optional = field.bool()?,
                _ => {}
            }
        }
        Ok(if optional { Self::Optional } else { result })
    }
}

#[derive(Clone, Default)]
struct ArrayFeatureType {
    shape: Vec<i64>,
    data_type: i32,
}

impl ArrayFeatureType {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.shape.extend(field.repeated_i64()?),
                2 => result.data_type = field.int32()?,
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct GlmRegressor {
    weights: Vec<Vec<f64>>,
    offset: Vec<f64>,
}

impl GlmRegressor {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.weights.push(read_double_array(field.bytes()?)?),
                2 => result.offset.extend(field.repeated_f64()?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct GlmClassifier {
    weights: Vec<Vec<f64>>,
    offset: Vec<f64>,
    class_encoding: i32,
    class_labels: Option<ClassLabels>,
}

impl GlmClassifier {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.weights.push(read_double_array(field.bytes()?)?),
                2 => result.offset.extend(field.repeated_f64()?),
                4 => result.class_encoding = field.int32()?,
                100 => {
                    result.class_labels =
                        Some(ClassLabels::String(read_string_vector(field.bytes()?)?))
                }
                101 => {
                    result.class_labels =
                        Some(ClassLabels::Int64(read_int64_vector(field.bytes()?)?))
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct KNearestNeighborsClassifier {
    nearest_neighbors_index: bool,
    number_of_neighbors: bool,
    class_labels: Option<ClassLabels>,
    default_string_label: Option<String>,
    default_int64_label: Option<i64>,
    uniform_weighting: bool,
    inverse_distance_weighting: bool,
}

impl KNearestNeighborsClassifier {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.nearest_neighbors_index = true;
                }
                3 => {
                    field.bytes()?;
                    result.number_of_neighbors = true;
                }
                100 => {
                    result.class_labels =
                        Some(ClassLabels::String(read_string_vector(field.bytes()?)?))
                }
                101 => {
                    result.class_labels =
                        Some(ClassLabels::Int64(read_int64_vector(field.bytes()?)?))
                }
                110 => result.default_string_label = Some(field.string()?),
                111 => result.default_int64_label = Some(field.varint()? as i64),
                200 => {
                    field.bytes()?;
                    result.uniform_weighting = true;
                }
                210 => {
                    field.bytes()?;
                    result.inverse_distance_weighting = true;
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if self.nearest_neighbors_index {
            attributes.push(string_attribute(
                model,
                "nearestNeighborsIndex",
                "[object Object]",
            ));
        }
        if self.number_of_neighbors {
            attributes.push(string_attribute(
                model,
                "numberOfNeighbors",
                "[object Object]",
            ));
        }
        if let Some(labels) = &self.class_labels {
            attributes.push(string_attribute(
                model,
                labels.operator(),
                "[object Object]",
            ));
        }
        if let Some(label) = &self.default_string_label {
            attributes.push(string_attribute(model, "defaultStringLabel", label));
        }
        if let Some(label) = self.default_int64_label {
            attributes.push(int_attribute(model, "defaultInt64Label", label));
        }
        if self.uniform_weighting {
            attributes.push(string_attribute(
                model,
                "uniformWeighting",
                "[object Object]",
            ));
        }
        if self.inverse_distance_weighting {
            attributes.push(string_attribute(
                model,
                "inverseDistanceWeighting",
                "[object Object]",
            ));
        }
        attributes
    }
}

#[derive(Default)]
struct Pipeline {
    models: Vec<CoreMlModel>,
}

impl Pipeline {
    fn decode_classifier(data: &[u8]) -> Result<Self, ModelError> {
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                return Self::decode(field.bytes()?);
            }
        }
        Ok(Self::default())
    }

    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.models.push(CoreMlModel::decode(field.bytes()?)?);
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct DictVectorizer {
    string_to_index: Option<Vec<String>>,
    int64_to_index: Option<Vec<String>>,
}

impl DictVectorizer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.string_to_index = Some(read_string_vector(field.bytes()?)?),
                2 => result.int64_to_index = Some(read_int64_vector(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct CustomModel {
    class_name: Option<String>,
    has_parameters: bool,
}

impl CustomModel {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                10 => result.class_name = Some(field.string()?),
                30 => {
                    field.bytes()?;
                    result.has_parameters = true;
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if let Some(class_name) = self.class_name {
            attributes.push(string_attribute(model, "className", &class_name));
        }
        attributes.push(string_attribute(model, "parameters", "[object Object]"));
        attributes
    }
}

#[derive(Default)]
struct LinkedModel {
    linked_model_file_name: bool,
    linked_model_search_path: bool,
}

impl LinkedModel {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.decode_linked_model_file(field.bytes()?)?;
            }
        }
        Ok(result)
    }

    fn decode_linked_model_file(&mut self, data: &[u8]) -> Result<(), ModelError> {
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    self.linked_model_file_name = true;
                }
                2 => {
                    field.bytes()?;
                    self.linked_model_search_path = true;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn attributes(self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if self.linked_model_file_name {
            attributes.push(string_attribute(
                model,
                "linkedModelFileName",
                "[object Object]",
            ));
        }
        if self.linked_model_search_path {
            attributes.push(string_attribute(
                model,
                "linkedModelSearchPath",
                "[object Object]",
            ));
        }
        attributes
    }
}

#[derive(Default)]
struct ItemSimilarityRecommender {
    item_string_ids: Option<Vec<String>>,
    item_int64_ids: Option<Vec<String>>,
    item_item_similarities: usize,
}

impl ItemSimilarityRecommender {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.item_item_similarities += 1;
                }
                2 => result.item_string_ids = Some(read_string_vector(field.bytes()?)?),
                3 => result.item_int64_ids = Some(read_int64_vector(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if let Some(values) = self.item_string_ids {
            attributes.push(Attribute {
                name: model.intern("itemStringIds"),
                value: AttributeValue::Strings(
                    values
                        .into_iter()
                        .map(|value| model.intern(value))
                        .collect(),
                ),
            });
        } else if let Some(values) = self.item_int64_ids {
            attributes.push(Attribute {
                name: model.intern("itemInt64Ids"),
                value: AttributeValue::Strings(
                    values
                        .into_iter()
                        .map(|value| model.intern(value))
                        .collect(),
                ),
            });
        }
        if self.item_item_similarities > 0 {
            attributes.push(object_list_attribute(
                model,
                "itemItemSimilarities",
                self.item_item_similarities,
            ));
        }
        attributes
    }
}

#[derive(Default)]
struct WordTagger {
    revision: Option<u32>,
    language: Option<String>,
    tokens_output_feature_name: String,
    token_tags_output_feature_name: String,
    token_locations_output_feature_name: String,
    token_lengths_output_feature_name: String,
    model_parameter_data: Vec<u8>,
    string_tags: Option<Vec<String>>,
}

impl WordTagger {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.revision = Some(field.uint32()?),
                10 => result.language = Some(field.string()?),
                20 => result.tokens_output_feature_name = field.string()?,
                21 => result.token_tags_output_feature_name = field.string()?,
                22 => result.token_locations_output_feature_name = field.string()?,
                23 => result.token_lengths_output_feature_name = field.string()?,
                100 => result.model_parameter_data = field.bytes()?.to_vec(),
                200 => result.string_tags = Some(read_string_vector(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if let Some(revision) = self.revision {
            attributes.push(int_attribute(model, "revision", i64::from(revision)));
        }
        if let Some(language) = &self.language {
            attributes.push(string_attribute(model, "language", language));
        }
        if !self.model_parameter_data.is_empty() {
            attributes.push(Attribute {
                name: model.intern("modelParameterData"),
                value: AttributeValue::Ints(
                    self.model_parameter_data
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
            });
        }
        if let Some(tags) = &self.string_tags {
            attributes.push(Attribute {
                name: model.intern("stringTags"),
                value: AttributeValue::Strings(tags.iter().map(|tag| model.intern(tag)).collect()),
            });
        }
        attributes
    }
}

#[derive(Default)]
struct TextClassifier {
    revision: Option<u32>,
    language: Option<String>,
    model_parameter_data: Vec<u8>,
    string_class_labels: Option<Vec<String>>,
}

impl TextClassifier {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.revision = Some(field.uint32()?),
                10 => result.language = Some(field.string()?),
                100 => result.model_parameter_data = field.bytes()?.to_vec(),
                200 => result.string_class_labels = Some(read_string_vector(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if let Some(revision) = self.revision {
            attributes.push(int_attribute(model, "revision", i64::from(revision)));
        }
        if let Some(language) = &self.language {
            attributes.push(string_attribute(model, "language", language));
        }
        if !self.model_parameter_data.is_empty() {
            attributes.push(Attribute {
                name: model.intern("modelParameterData"),
                value: AttributeValue::Ints(
                    self.model_parameter_data
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect(),
                ),
            });
        }
        if let Some(labels) = &self.string_class_labels {
            attributes.push(string_vector_attribute(model, "stringClassLabels", labels));
        }
        attributes
    }
}

#[derive(Default)]
struct SoundAnalysisPreprocessing {
    vggish: bool,
}

impl SoundAnalysisPreprocessing {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 20 {
                field.bytes()?;
                result.vggish = true;
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        if self.vggish {
            vec![string_attribute(model, "vggish", "[object Object]")]
        } else {
            Vec::new()
        }
    }
}

#[derive(Default)]
struct MlProgram {
    functions: Vec<(String, MlFunction)>,
}

impl MlProgram {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 2 {
                result
                    .functions
                    .push(read_ml_function_entry(field.bytes()?)?);
            }
        }
        Ok(result)
    }

    fn lower(
        self,
        model: &mut Model,
        graph: &mut Graph,
        values: &mut HashMap<String, ValueId>,
    ) -> Result<(), ModelError> {
        let Some((_, function)) = self
            .functions
            .iter()
            .find(|(name, _)| name == "main")
            .or_else(|| self.functions.first())
        else {
            return Ok(());
        };
        let Some((_, block)) = function
            .block_specializations
            .iter()
            .find(|(name, _)| name.starts_with("CoreML"))
            .or_else(|| function.block_specializations.first())
        else {
            return Ok(());
        };

        let mut runtime_values = HashMap::new();
        for input in &function.inputs {
            runtime_values.insert(
                input.name.clone(),
                MlRuntimeValue {
                    type_info: input.type_info.clone(),
                    producer: None,
                    const_value: None,
                },
            );
            if let Some(value_id) = values.get(&input.name).copied()
                && graph.values[value_id.index()].type_info.is_none()
            {
                graph.values[value_id.index()].type_info = input
                    .type_info
                    .as_ref()
                    .map(|value| ml_graph_type(model, value));
            }
        }
        for (index, op) in block.operations.iter().enumerate() {
            for output in &op.outputs {
                runtime_values
                    .entry(output.name.clone())
                    .or_insert(MlRuntimeValue {
                        type_info: output.type_info.clone(),
                        producer: Some(index),
                        const_value: None,
                    });
            }
        }

        let mut runtime_ops = block
            .operations
            .iter()
            .map(MlRuntimeOperation::from_operation)
            .collect::<Vec<_>>();
        for (index, op) in block.operations.iter().enumerate() {
            if op.op_type == "const"
                && op.inputs.is_empty()
                && op.outputs.len() == 1
                && let Some(value) = op
                    .attributes
                    .iter()
                    .find(|(name, _)| name == "val")
                    .and_then(|(_, value)| value.converted())
                    .filter(MlConvertedValue::is_js_truthy)
            {
                runtime_ops[index].delete = true;
                if let Some(output) = op.outputs.first()
                    && let Some(runtime_value) = runtime_values.get_mut(&output.name)
                {
                    runtime_value.const_value = Some(value);
                }
            }
        }

        for op in &block.operations {
            for (_, input) in &op.inputs {
                let resolved = input
                    .bindings
                    .iter()
                    .map(|binding| resolve_ml_binding(binding, &runtime_values))
                    .collect::<Vec<_>>();
                if resolved.len() > 1
                    && resolved.iter().any(|value| value.is_const)
                    && !resolved
                        .iter()
                        .all(|value| matches!(value.const_value, Some(MlConvertedValue::Tensor(_))))
                {
                    for value in resolved {
                        if let Some(name) = value.name
                            && let Some(runtime_value) = runtime_values.get_mut(&name)
                        {
                            if let Some(producer) = runtime_value.producer {
                                runtime_ops[producer].delete = false;
                            }
                            runtime_value.const_value = None;
                        }
                    }
                }
            }
        }
        for op in &block.operations {
            for (name, input) in &op.inputs {
                if !ml_kept_const_input(&op.op_type, name) {
                    continue;
                }
                for binding in &input.bindings {
                    if let Some(name) = binding.name.as_deref()
                        && let Some(runtime_value) = runtime_values.get_mut(name)
                        && runtime_value.const_value.is_some()
                    {
                        if let Some(producer) = runtime_value.producer {
                            runtime_ops[producer].delete = false;
                        }
                        runtime_value.const_value = None;
                    }
                }
            }
        }

        for (index, op) in block.operations.iter().enumerate() {
            if runtime_ops[index].delete {
                continue;
            }
            let mut node = Node::new(
                graph.id,
                Operator {
                    domain: None,
                    name: model.intern(&op.op_type),
                    overload: None,
                    version: None,
                    origin: FORMAT,
                },
            );
            node.attributes = runtime_ops[index]
                .attributes
                .iter()
                .filter_map(|(name, value)| ml_attribute(model, name, value.clone()))
                .collect();
            let mut ordered_inputs = op.inputs.iter().collect::<Vec<_>>();
            ordered_inputs.sort_by_key(|(name, _)| ml_input_order(&op.op_type, name));
            for (name, input) in ordered_inputs {
                let resolved = input
                    .bindings
                    .iter()
                    .map(|binding| resolve_ml_binding(binding, &runtime_values))
                    .collect::<Vec<_>>();
                if resolved.iter().all(|value| {
                    value.const_value.is_none()
                        || matches!(value.const_value, Some(MlConvertedValue::Tensor(_)))
                }) {
                    for value in resolved {
                        if let Some(name) = value.name {
                            let value_id = ensure_ml_program_value(
                                model,
                                graph,
                                values,
                                &mut runtime_values,
                                &name,
                            );
                            node.inputs.push(Some(value_id));
                        } else if let Some(MlConvertedValue::Tensor(tensor)) = value.const_value {
                            let value_id =
                                add_anonymous_ml_tensor_value(model, graph, values, tensor);
                            node.inputs.push(Some(value_id));
                        }
                    }
                } else if let Some(value) = combine_ml_values(
                    resolved
                        .iter()
                        .filter_map(|value| value.const_value.clone())
                        .collect(),
                ) {
                    if let Some(attribute) = ml_attribute(model, name, value) {
                        node.attributes.push(attribute);
                    }
                }
            }
            for output in &op.outputs {
                let value_id = ensure_ml_program_value(
                    model,
                    graph,
                    values,
                    &mut runtime_values,
                    &output.name,
                );
                node.outputs.push(Some(value_id));
            }

            let inputs = node.inputs.iter().flatten().copied().collect::<Vec<_>>();
            let outputs = node.outputs.iter().flatten().copied().collect::<Vec<_>>();
            let node_id = graph.add_node(node);
            for input in inputs {
                let consumers = &mut graph.values[input.index()].consumers;
                if !consumers.contains(&node_id) {
                    consumers.push(node_id);
                }
            }
            for output in outputs {
                graph.values[output.index()].producer = Some(node_id);
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct MlFunction {
    inputs: Vec<MlNamedValueType>,
    block_specializations: Vec<(String, MlBlock)>,
}

impl MlFunction {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result
                    .inputs
                    .push(MlNamedValueType::decode(field.bytes()?)?),
                3 => result
                    .block_specializations
                    .push(read_ml_block_entry(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlBlock {
    operations: Vec<MlOperation>,
}

impl MlBlock {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 3 {
                result.operations.push(MlOperation::decode(field.bytes()?)?);
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlOperation {
    op_type: String,
    inputs: Vec<(String, MlArgument)>,
    outputs: Vec<MlNamedValueType>,
    attributes: Vec<(String, MlValue)>,
}

impl MlOperation {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.op_type = field.string()?,
                2 => result.inputs.push(read_ml_argument_entry(field.bytes()?)?),
                3 => result
                    .outputs
                    .push(MlNamedValueType::decode(field.bytes()?)?),
                5 => result.attributes.push(read_ml_value_entry(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlArgument {
    bindings: Vec<MlBinding>,
}

impl MlArgument {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.bindings.push(MlBinding::decode(field.bytes()?)?);
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlBinding {
    name: Option<String>,
    value: Option<MlValue>,
}

impl MlBinding {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.name = Some(field.string()?),
                2 => result.value = Some(MlValue::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlNamedValueType {
    name: String,
    type_info: Option<TypeInfo>,
}

impl MlNamedValueType {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.name = field.string()?,
                2 => result.type_info = MlValueType::decode(field.bytes()?)?.type_info,
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlValueType {
    type_info: Option<TypeInfo>,
}

impl MlValueType {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.type_info = Some(MlTensorType::decode(field.bytes()?)?.type_info());
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct MlTensorType {
    data_type: i32,
    dimensions: Vec<Dimension>,
}

impl MlTensorType {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.data_type = field.int32()?,
                3 => result.dimensions.push(read_ml_dimension(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn type_info(self) -> TypeInfo {
        TypeInfo {
            element_type: Some(ml_data_type(self.data_type)),
            layout: None,
            denotation: None,
            shape: self.dimensions,
        }
    }
}

#[derive(Clone)]
struct MlValue {
    type_info: Option<TypeInfo>,
    kind: Option<MlValueKind>,
}

impl MlValue {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self {
            type_info: None,
            kind: None,
        };
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                2 => result.type_info = MlValueType::decode(field.bytes()?)?.type_info,
                3 => {
                    result.kind = Some(MlValueKind::Immediate(MlImmediateValue::decode(
                        field.bytes()?,
                    )?))
                }
                5 => {
                    field.bytes()?;
                    result.kind = Some(MlValueKind::BlobFile);
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn converted(&self) -> Option<MlConvertedValue> {
        match self.kind.as_ref()? {
            MlValueKind::Immediate(value) => value.converted(self.type_info.as_ref()),
            MlValueKind::BlobFile => Some(MlConvertedValue::Tensor(MlTensorSpec {
                type_info: self.type_info.clone(),
            })),
        }
    }
}

#[derive(Clone)]
enum MlValueKind {
    Immediate(MlImmediateValue),
    BlobFile,
}

#[derive(Clone, Default)]
struct MlImmediateValue {
    tensor: Option<MlTensorLiteral>,
}

impl MlImmediateValue {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.tensor = Some(MlTensorLiteral::decode(field.bytes()?)?);
            }
        }
        Ok(result)
    }

    fn converted(&self, type_info: Option<&TypeInfo>) -> Option<MlConvertedValue> {
        self.tensor
            .as_ref()
            .and_then(|tensor| tensor.converted(type_info))
    }
}

#[derive(Clone, Default)]
struct MlTensorLiteral {
    value: Option<MlTensorLiteralValue>,
}

impl MlTensorLiteral {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    result.value = Some(MlTensorLiteralValue::Floats(read_ml_floats(
                        field.bytes()?,
                    )?))
                }
                2 => result.value = Some(MlTensorLiteralValue::Ints(read_ml_ints(field.bytes()?)?)),
                3 => {
                    result.value = Some(MlTensorLiteralValue::Bools(read_ml_bools(field.bytes()?)?))
                }
                4 => {
                    result.value = Some(MlTensorLiteralValue::Strings(read_ml_strings(
                        field.bytes()?,
                    )?))
                }
                5 => result.value = Some(MlTensorLiteralValue::Ints(read_ml_ints(field.bytes()?)?)),
                6 => {
                    result.value = Some(MlTensorLiteralValue::Floats(read_ml_doubles(
                        field.bytes()?,
                    )?))
                }
                7 => {
                    result.value = Some(MlTensorLiteralValue::Bytes(read_ml_bytes(field.bytes()?)?))
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn converted(&self, type_info: Option<&TypeInfo>) -> Option<MlConvertedValue> {
        let scalar = type_info.is_some_and(|type_info| type_info.shape.is_empty());
        match self.value.as_ref()? {
            MlTensorLiteralValue::Floats(values) if scalar => {
                values.first().copied().map(MlConvertedValue::Float)
            }
            MlTensorLiteralValue::Floats(values) => Some(MlConvertedValue::Floats(values.clone())),
            MlTensorLiteralValue::Ints(values) if scalar => {
                values.first().copied().map(MlConvertedValue::Int)
            }
            MlTensorLiteralValue::Ints(values) => Some(MlConvertedValue::Ints(values.clone())),
            MlTensorLiteralValue::Bools(values) if scalar => {
                values.first().copied().map(MlConvertedValue::Bool)
            }
            MlTensorLiteralValue::Bools(values) => Some(MlConvertedValue::Bools(values.clone())),
            MlTensorLiteralValue::Strings(values) if scalar => {
                values.first().cloned().map(MlConvertedValue::String)
            }
            MlTensorLiteralValue::Strings(values) => {
                Some(MlConvertedValue::Strings(values.clone()))
            }
            MlTensorLiteralValue::Bytes(values) if scalar => values
                .first()
                .copied()
                .map(|value| MlConvertedValue::Int(i64::from(value))),
            MlTensorLiteralValue::Bytes(values) => Some(MlConvertedValue::Bytes(values.len())),
        }
    }
}

#[derive(Clone)]
enum MlTensorLiteralValue {
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Bools(Vec<bool>),
    Strings(Vec<String>),
    Bytes(Vec<u8>),
}

#[derive(Clone)]
enum MlConvertedValue {
    Bool(bool),
    Float(f32),
    Int(i64),
    String(String),
    Bytes(usize),
    Bools(Vec<bool>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensor(MlTensorSpec),
}

impl MlConvertedValue {
    fn is_js_truthy(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            Self::Float(value) => *value != 0.0 && !value.is_nan(),
            Self::Int(value) => *value != 0,
            Self::String(value) => !value.is_empty(),
            Self::Bytes(_)
            | Self::Bools(_)
            | Self::Floats(_)
            | Self::Ints(_)
            | Self::Strings(_)
            | Self::Tensor(_) => true,
        }
    }
}

#[derive(Clone)]
struct MlTensorSpec {
    type_info: Option<TypeInfo>,
}

struct MlRuntimeValue {
    type_info: Option<TypeInfo>,
    producer: Option<usize>,
    const_value: Option<MlConvertedValue>,
}

struct MlRuntimeOperation {
    delete: bool,
    attributes: Vec<(String, MlConvertedValue)>,
}

impl MlRuntimeOperation {
    fn from_operation(op: &MlOperation) -> Self {
        Self {
            delete: false,
            attributes: op
                .attributes
                .iter()
                .filter_map(|(name, value)| value.converted().map(|value| (name.clone(), value)))
                .collect(),
        }
    }
}

struct MlResolvedBinding {
    name: Option<String>,
    const_value: Option<MlConvertedValue>,
    is_const: bool,
}

fn resolve_ml_binding(
    binding: &MlBinding,
    runtime_values: &HashMap<String, MlRuntimeValue>,
) -> MlResolvedBinding {
    if let Some(name) = binding.name.as_deref() {
        let const_value = runtime_values
            .get(name)
            .and_then(|value| value.const_value.clone());
        MlResolvedBinding {
            name: Some(name.to_owned()),
            is_const: const_value.is_some(),
            const_value,
        }
    } else {
        let const_value = binding.value.as_ref().and_then(MlValue::converted);
        MlResolvedBinding {
            name: None,
            is_const: const_value.is_some(),
            const_value,
        }
    }
}

fn ensure_ml_program_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    runtime_values: &mut HashMap<String, MlRuntimeValue>,
    name: &str,
) -> ValueId {
    let value_id = if let Some(value_id) = values.get(name).copied() {
        value_id
    } else {
        let value_id = graph.add_value(Value::new(model.intern(name)));
        values.insert(name.to_owned(), value_id);
        value_id
    };
    if let Some(runtime_value) = runtime_values.get(name) {
        if let Some(type_info) = &runtime_value.type_info {
            let keep_existing_graph_input = graph.values[value_id.index()].is_graph_input
                && graph.values[value_id.index()].type_info.is_some();
            if !keep_existing_graph_input {
                graph.values[value_id.index()].type_info = Some(ml_graph_type(model, type_info));
            }
        }
        if graph.values[value_id.index()].initializer.is_none()
            && let Some(MlConvertedValue::Tensor(tensor)) = &runtime_value.const_value
        {
            graph.values[value_id.index()].initializer = Some(add_ml_tensor(model, tensor.clone()));
        }
    }
    value_id
}

fn add_anonymous_ml_tensor_value(
    model: &mut Model,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
    tensor: MlTensorSpec,
) -> ValueId {
    let name = format!(":tensor:{}", graph.values.len());
    let mut value = Value::new(model.intern(&name));
    value.type_info = tensor
        .type_info
        .as_ref()
        .map(|value| ml_graph_type(model, value));
    value.initializer = Some(add_ml_tensor(model, tensor));
    let value_id = graph.add_value(value);
    values.insert(name, value_id);
    value_id
}

fn add_ml_tensor(model: &mut Model, tensor: MlTensorSpec) -> netron_rs_core::TensorId {
    let type_info = tensor.type_info.unwrap_or(TypeInfo {
        element_type: Some(TensorElementType::Unknown),
        layout: None,
        denotation: None,
        shape: Vec::new(),
    });
    let type_info = ml_graph_type(model, &type_info);
    model.add_tensor(Tensor::metadata_only(
        None,
        type_info.element_type.unwrap_or(TensorElementType::Unknown),
        type_info.shape,
        TensorStorage::Absent,
    ))
}

fn ml_graph_type(model: &mut Model, value: &TypeInfo) -> TypeInfo {
    TypeInfo {
        element_type: value.element_type.clone(),
        layout: value.layout,
        denotation: value.denotation,
        shape: value
            .shape
            .iter()
            .map(|dimension| match dimension.value {
                DimensionValue::Unknown => Dimension::symbolic(model.intern("?")),
                _ => dimension.clone(),
            })
            .collect(),
    }
}

fn ml_attribute(model: &mut Model, name: &str, value: MlConvertedValue) -> Option<Attribute> {
    let value = match value {
        MlConvertedValue::Bool(value) => AttributeValue::Bool(value),
        MlConvertedValue::Float(value) => f32_scalar_attribute(model, name, value).value,
        MlConvertedValue::Int(value) => AttributeValue::Int(value),
        MlConvertedValue::String(value) => AttributeValue::String(model.intern(value)),
        MlConvertedValue::Bytes(byte_len) => AttributeValue::Bytes { byte_len },
        MlConvertedValue::Bools(values) => AttributeValue::Strings(
            values
                .into_iter()
                .map(|value| model.intern(value.to_string()))
                .collect(),
        ),
        MlConvertedValue::Floats(values) => AttributeValue::Floats(values),
        MlConvertedValue::Ints(values) => AttributeValue::Ints(values),
        MlConvertedValue::Strings(values) => AttributeValue::Strings(
            values
                .into_iter()
                .map(|value| model.intern(value))
                .collect(),
        ),
        MlConvertedValue::Tensor(tensor) => AttributeValue::Tensor(add_ml_tensor(model, tensor)),
    };
    Some(Attribute {
        name: model.intern(name),
        value,
    })
}

fn combine_ml_values(values: Vec<MlConvertedValue>) -> Option<MlConvertedValue> {
    if values.len() == 1 {
        return values.into_iter().next();
    }
    if values
        .iter()
        .all(|value| matches!(value, MlConvertedValue::Int(_)))
    {
        return Some(MlConvertedValue::Ints(
            values
                .into_iter()
                .filter_map(|value| match value {
                    MlConvertedValue::Int(value) => Some(value),
                    _ => None,
                })
                .collect(),
        ));
    }
    if values
        .iter()
        .all(|value| matches!(value, MlConvertedValue::Float(_)))
    {
        return Some(MlConvertedValue::Floats(
            values
                .into_iter()
                .filter_map(|value| match value {
                    MlConvertedValue::Float(value) => Some(value),
                    _ => None,
                })
                .collect(),
        ));
    }
    if values
        .iter()
        .all(|value| matches!(value, MlConvertedValue::String(_)))
    {
        return Some(MlConvertedValue::Strings(
            values
                .into_iter()
                .filter_map(|value| match value {
                    MlConvertedValue::String(value) => Some(value),
                    _ => None,
                })
                .collect(),
        ));
    }
    None
}

fn ml_kept_const_input(op_type: &str, name: &str) -> bool {
    matches!(
        (op_type, name),
        ("pad", "constant_val")
            | ("reduce_mean", "keep_dims")
            | ("concat", "interleave")
            | ("max_pool", "ceil_mode")
    )
}

fn ml_input_order(op_type: &str, name: &str) -> usize {
    match op_type {
        "conv" | "linear" => match name {
            "x" => 0,
            "weight" => 1,
            "bias" => 2,
            _ => 10,
        },
        "pad" => match name {
            "constant_val" => 0,
            "x" => 1,
            _ => 10,
        },
        "reshape" => match name {
            "x" => 0,
            "shape" => 1,
            _ => 10,
        },
        "transpose" | "sigmoid" | "softmax" => match name {
            "x" => 0,
            _ => 10,
        },
        _ => 10,
    }
}

fn read_ml_function_entry(data: &[u8]) -> Result<(String, MlFunction), ModelError> {
    let mut key = String::new();
    let mut value = MlFunction::default();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => key = field.string()?,
            2 => value = MlFunction::decode(field.bytes()?)?,
            _ => {}
        }
    }
    Ok((key, value))
}

fn read_ml_block_entry(data: &[u8]) -> Result<(String, MlBlock), ModelError> {
    let mut key = String::new();
    let mut value = MlBlock::default();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => key = field.string()?,
            2 => value = MlBlock::decode(field.bytes()?)?,
            _ => {}
        }
    }
    Ok((key, value))
}

fn read_ml_argument_entry(data: &[u8]) -> Result<(String, MlArgument), ModelError> {
    let mut key = String::new();
    let mut value = MlArgument::default();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => key = field.string()?,
            2 => value = MlArgument::decode(field.bytes()?)?,
            _ => {}
        }
    }
    Ok((key, value))
}

fn read_ml_value_entry(data: &[u8]) -> Result<(String, MlValue), ModelError> {
    let mut key = String::new();
    let mut value = None;
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => key = field.string()?,
            2 => value = Some(MlValue::decode(field.bytes()?)?),
            _ => {}
        }
    }
    Ok((
        key,
        value.unwrap_or(MlValue {
            type_info: None,
            kind: None,
        }),
    ))
}

fn read_ml_dimension(data: &[u8]) -> Result<Dimension, ModelError> {
    let mut result = Dimension::unknown();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => {
                let mut inner = PbReader::new(field.bytes()?);
                while let Some(inner_field) = inner.next_field()? {
                    if inner_field.number == 1 {
                        result = Dimension::known(inner_field.varint()? as i64);
                    }
                }
            }
            2 => {
                field.bytes()?;
                result = Dimension::unknown();
            }
            _ => {}
        }
    }
    Ok(result)
}

fn read_ml_floats(data: &[u8]) -> Result<Vec<f32>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(field.repeated_f32()?);
        }
    }
    Ok(values)
}

fn read_ml_doubles(data: &[u8]) -> Result<Vec<f32>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(field.repeated_f64()?.into_iter().map(|value| value as f32));
        }
    }
    Ok(values)
}

fn read_ml_ints(data: &[u8]) -> Result<Vec<i64>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(field.repeated_i64()?);
        }
    }
    Ok(values)
}

fn read_ml_bools(data: &[u8]) -> Result<Vec<bool>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(field.repeated_u64()?.into_iter().map(|value| value != 0));
        }
    }
    Ok(values)
}

fn read_ml_strings(data: &[u8]) -> Result<Vec<String>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.push(field.string()?);
        }
    }
    Ok(values)
}

fn read_ml_bytes(data: &[u8]) -> Result<Vec<u8>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(field.bytes()?);
        }
    }
    Ok(values)
}

fn ml_data_type(value: i32) -> TensorElementType {
    match value {
        1 => TensorElementType::Bool,
        2 => TensorElementType::String,
        10 => TensorElementType::Float16,
        11 => TensorElementType::Float32,
        12 => TensorElementType::Float64,
        13 => TensorElementType::BFloat16,
        21 => TensorElementType::Int8,
        22 => TensorElementType::Int16,
        23 => TensorElementType::Int32,
        24 => TensorElementType::Int64,
        25 => TensorElementType::Int4,
        31 => TensorElementType::Uint8,
        32 => TensorElementType::Uint16,
        33 => TensorElementType::Uint32,
        34 => TensorElementType::Uint64,
        35 => TensorElementType::Uint4,
        36 => TensorElementType::Uint2,
        37 => TensorElementType::Other("uint1".to_owned()),
        38 => TensorElementType::Other("uint6".to_owned()),
        39 => TensorElementType::Other("uint3".to_owned()),
        40 => TensorElementType::Float8e4m3fn,
        41 => TensorElementType::Float8e5m2,
        _ => TensorElementType::Unknown,
    }
}

#[derive(Default)]
struct FeatureVectorizer {
    input_list: Vec<()>,
}

impl FeatureVectorizer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                field.bytes()?;
                result.input_list.push(());
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct Imputer {
    imputed_double_array: bool,
    replace_double_value: Option<f64>,
}

impl Imputer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                4 => {
                    field.bytes()?;
                    result.imputed_double_array = true;
                }
                11 => result.replace_double_value = Some(field.f64()?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct OneHotEncoder {
    string_categories: bool,
    int64_categories: bool,
    output_sparse: Option<bool>,
}

impl OneHotEncoder {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.string_categories = true;
                }
                2 => {
                    field.bytes()?;
                    result.int64_categories = true;
                }
                10 => result.output_sparse = Some(field.bool()?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct SupportVectorClassifier {
    kernel: bool,
    dense_support_vectors: bool,
    sparse_support_vectors: bool,
    coefficients_count: usize,
    number_of_support_vectors_per_class: Vec<i64>,
    rho: Vec<f64>,
    prob_a: Vec<f64>,
    prob_b: Vec<f64>,
    class_labels: Option<ClassLabels>,
}

impl SupportVectorClassifier {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.kernel = true;
                }
                2 => result
                    .number_of_support_vectors_per_class
                    .extend(field.repeated_i64()?),
                3 => {
                    field.bytes()?;
                    result.sparse_support_vectors = true;
                }
                4 => {
                    field.bytes()?;
                    result.dense_support_vectors = true;
                }
                5 => {
                    field.bytes()?;
                    result.coefficients_count += 1;
                }
                6 => result.rho.extend(field.repeated_f64()?),
                7 => result.prob_a.extend(field.repeated_f64()?),
                8 => result.prob_b.extend(field.repeated_f64()?),
                100 => {
                    result.class_labels =
                        Some(ClassLabels::String(read_string_vector(field.bytes()?)?));
                }
                101 => {
                    result.class_labels =
                        Some(ClassLabels::Int64(read_int64_vector(field.bytes()?)?));
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if self.coefficients_count > 0 {
            attributes.push(object_list_attribute(
                model,
                "coefficients",
                self.coefficients_count,
            ));
        }
        if self.dense_support_vectors {
            attributes.push(string_attribute(
                model,
                "denseSupportVectors",
                "[object Object]",
            ));
        }
        if self.kernel {
            attributes.push(string_attribute(model, "kernel", "[object Object]"));
        }
        attributes.push(Attribute {
            name: model.intern("numberOfSupportVectorsPerClass"),
            value: AttributeValue::Ints(self.number_of_support_vectors_per_class.clone()),
        });
        attributes.push(numeric_array_attribute(model, "probA", &self.prob_a));
        attributes.push(numeric_array_attribute(model, "probB", &self.prob_b));
        attributes.push(numeric_array_attribute(model, "rho", &self.rho));
        if let Some(support_vectors) = self.support_vectors() {
            attributes.push(string_attribute(model, "supportVectors", support_vectors));
        }
        attributes
    }

    fn support_vectors(&self) -> Option<&'static str> {
        if self.sparse_support_vectors {
            Some("sparseSupportVectors")
        } else if self.dense_support_vectors {
            Some("denseSupportVectors")
        } else {
            None
        }
    }
}

#[derive(Default)]
struct SupportVectorRegressor {
    kernel: bool,
    dense_support_vectors: bool,
    sparse_support_vectors: bool,
    coefficients: bool,
    rho: Option<f64>,
}

impl SupportVectorRegressor {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.kernel = true;
                }
                2 => {
                    field.bytes()?;
                    result.sparse_support_vectors = true;
                }
                3 => {
                    field.bytes()?;
                    result.dense_support_vectors = true;
                }
                4 => {
                    field.bytes()?;
                    result.coefficients = true;
                }
                5 => result.rho = Some(field.f64()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if self.coefficients {
            attributes.push(string_attribute(model, "coefficients", "[object Object]"));
        }
        if self.kernel {
            attributes.push(string_attribute(model, "kernel", "[object Object]"));
        }
        if let Some(rho) = self.rho {
            attributes.push(Attribute {
                name: model.intern("rho"),
                value: AttributeValue::Float(rho as f32),
            });
        }
        if let Some(support_vectors) = self.support_vectors() {
            attributes.push(string_attribute(model, "supportVectors", support_vectors));
        }
        attributes
    }

    fn support_vectors(&self) -> Option<&'static str> {
        if self.sparse_support_vectors {
            Some("sparseSupportVectors")
        } else if self.dense_support_vectors {
            Some("denseSupportVectors")
        } else {
            None
        }
    }
}

#[derive(Default)]
struct TreeEnsembleClassifier {
    tree_ensemble: TreeEnsembleParameters,
    class_labels: Option<ClassLabels>,
}

impl TreeEnsembleClassifier {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.tree_ensemble = TreeEnsembleParameters::decode(field.bytes()?)?,
                100 => {
                    result.class_labels =
                        Some(ClassLabels::String(read_string_vector(field.bytes()?)?));
                }
                101 => {
                    result.class_labels =
                        Some(ClassLabels::Int64(read_int64_vector(field.bytes()?)?));
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct TreeEnsembleRegressor {
    tree_ensemble: TreeEnsembleParameters,
}

impl TreeEnsembleRegressor {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.tree_ensemble = TreeEnsembleParameters::decode(field.bytes()?)?;
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct TreeEnsembleParameters {
    nodes_count: usize,
    num_prediction_dimensions: Option<u64>,
    base_prediction_value: Vec<f64>,
}

impl TreeEnsembleParameters {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.nodes_count += 1;
                }
                2 => result.num_prediction_dimensions = Some(field.varint()?),
                3 => result.base_prediction_value.extend(field.repeated_f64()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if self.nodes_count > 0 {
            attributes.push(object_list_attribute(model, "nodes", self.nodes_count));
        }
        if !self.base_prediction_value.is_empty() {
            attributes.push(numeric_array_attribute(
                model,
                "basePredictionValue",
                &self.base_prediction_value,
            ));
        }
        if let Some(value) = self.num_prediction_dimensions {
            attributes.push(string_attribute(
                model,
                "numPredictionDimensions",
                &value.to_string(),
            ));
        }
        attributes
    }
}

enum ClassLabels {
    String(Vec<String>),
    Int64(Vec<String>),
}

impl ClassLabels {
    fn operator(&self) -> &'static str {
        match self {
            Self::String(_) => "stringClassLabels",
            Self::Int64(_) => "int64ClassLabels",
        }
    }

    fn values(self) -> Vec<String> {
        match self {
            Self::String(values) | Self::Int64(values) => values,
        }
    }
}

#[derive(Default)]
struct NeuralNetwork {
    layers: Vec<NeuralLayer>,
    preprocessing: Vec<NeuralNetworkPreprocessing>,
}

impl NeuralNetwork {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.layers.push(NeuralLayer::decode(field.bytes()?)?),
                2 => result
                    .preprocessing
                    .push(NeuralNetworkPreprocessing::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct NeuralNetworkClassifier {
    network: NeuralNetwork,
    class_labels: Option<ClassLabels>,
    label_probability_layer_name: String,
}

impl NeuralNetworkClassifier {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result
                    .network
                    .layers
                    .push(NeuralLayer::decode(field.bytes()?)?),
                2 => result
                    .network
                    .preprocessing
                    .push(NeuralNetworkPreprocessing::decode(field.bytes()?)?),
                100 => {
                    result.class_labels =
                        Some(ClassLabels::String(read_string_vector(field.bytes()?)?));
                }
                101 => {
                    result.class_labels =
                        Some(ClassLabels::Int64(read_int64_vector(field.bytes()?)?));
                }
                200 => result.label_probability_layer_name = field.string()?,
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct NeuralLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    kind: NeuralLayerKind,
}

impl NeuralLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.name = field.string()?,
                2 => result.inputs.push(field.string()?),
                3 => result.outputs.push(field.string()?),
                100 => {
                    result.kind =
                        NeuralLayerKind::Convolution(ConvolutionLayer::decode(field.bytes()?)?)
                }
                120 => {
                    result.kind = NeuralLayerKind::Pooling(PoolingLayer::decode(field.bytes()?)?)
                }
                130 => {
                    result.kind =
                        NeuralLayerKind::Activation(ActivationLayer::decode(field.bytes()?)?)
                }
                140 => {
                    result.kind =
                        NeuralLayerKind::InnerProduct(InnerProductLayer::decode(field.bytes()?)?)
                }
                150 => {
                    result.kind =
                        NeuralLayerKind::Embedding(EmbeddingLayer::decode(field.bytes()?)?)
                }
                160 => {
                    result.kind =
                        NeuralLayerKind::Batchnorm(BatchnormLayer::decode(field.bytes()?)?)
                }
                175 => {
                    field.bytes()?;
                    result.kind = NeuralLayerKind::Simple("softmax");
                }
                180 => result.kind = NeuralLayerKind::Lrn(LrnLayer::decode(field.bytes()?)?),
                190 => result.kind = NeuralLayerKind::Crop(CropLayer::decode(field.bytes()?)?),
                200 => {
                    result.kind = NeuralLayerKind::Padding(PaddingLayer::decode(field.bytes()?)?)
                }
                210 => {
                    result.kind = NeuralLayerKind::Upsample(UpsampleLayer::decode(field.bytes()?)?)
                }
                220 => result.kind = NeuralLayerKind::Unary(UnaryLayer::decode(field.bytes()?)?),
                230 => result.kind = NeuralLayerKind::Add(AddLayer::decode(field.bytes()?)?),
                231 => {
                    result.kind = NeuralLayerKind::Multiply(MultiplyLayer::decode(field.bytes()?)?)
                }
                245 => result.kind = NeuralLayerKind::Scale(ScaleLayer::decode(field.bytes()?)?),
                290 => {
                    result.kind =
                        NeuralLayerKind::LoadConstant(LoadConstantLayer::decode(field.bytes()?)?)
                }
                300 => {
                    result.kind = NeuralLayerKind::Reshape(ReshapeLayer::decode(field.bytes()?)?)
                }
                301 => {
                    result.kind = NeuralLayerKind::Flatten(FlattenLayer::decode(field.bytes()?)?)
                }
                310 => {
                    result.kind = NeuralLayerKind::Permute(PermuteLayer::decode(field.bytes()?)?)
                }
                320 => {
                    field.bytes()?;
                    result.kind = NeuralLayerKind::Simple("concat");
                }
                350 => result.kind = NeuralLayerKind::Slice(SliceLayer::decode(field.bytes()?)?),
                410 => result.kind = NeuralLayerKind::Gru(GruLayer::decode(field.bytes()?)?),
                420 => {
                    result.kind = NeuralLayerKind::UniDirectionalLstm(
                        UniDirectionalLstmLayer::decode(field.bytes()?)?,
                    )
                }
                430 => {
                    result.kind = NeuralLayerKind::BiDirectionalLstm(
                        BiDirectionalLstmLayer::decode(field.bytes()?)?,
                    )
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
enum NeuralLayerKind {
    Activation(ActivationLayer),
    Add(AddLayer),
    Batchnorm(BatchnormLayer),
    BiDirectionalLstm(BiDirectionalLstmLayer),
    Convolution(ConvolutionLayer),
    Crop(CropLayer),
    Embedding(EmbeddingLayer),
    Flatten(FlattenLayer),
    Gru(GruLayer),
    InnerProduct(InnerProductLayer),
    LoadConstant(LoadConstantLayer),
    Lrn(LrnLayer),
    Multiply(MultiplyLayer),
    Padding(PaddingLayer),
    Permute(PermuteLayer),
    Pooling(PoolingLayer),
    Reshape(ReshapeLayer),
    Scale(ScaleLayer),
    Slice(SliceLayer),
    Simple(&'static str),
    UniDirectionalLstm(UniDirectionalLstmLayer),
    Unary(UnaryLayer),
    Upsample(UpsampleLayer),
    #[default]
    Unsupported,
}

impl NeuralLayerKind {
    fn operator_name(&self) -> &'static str {
        match self {
            Self::Activation(_) => "activation",
            Self::Add(_) => "add",
            Self::Batchnorm(_) => "batchnorm",
            Self::BiDirectionalLstm(_) => "biDirectionalLSTM",
            Self::Convolution(_) => "convolution",
            Self::Crop(_) => "crop",
            Self::Embedding(_) => "embedding",
            Self::Flatten(_) => "flatten",
            Self::Gru(_) => "gru",
            Self::InnerProduct(_) => "innerProduct",
            Self::LoadConstant(_) => "loadConstant",
            Self::Lrn(_) => "lrn",
            Self::Multiply(_) => "multiply",
            Self::Padding(_) => "padding",
            Self::Permute(_) => "permute",
            Self::Pooling(_) => "pooling",
            Self::Reshape(_) => "reshape",
            Self::Scale(_) => "scale",
            Self::Slice(_) => "slice",
            Self::Simple(name) => name,
            Self::UniDirectionalLstm(_) => "uniDirectionalLSTM",
            Self::Unary(_) => "unary",
            Self::Upsample(_) => "upsample",
            Self::Unsupported => "",
        }
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        match self {
            Self::Activation(layer) => layer.lower(model, graph),
            Self::Add(layer) => layer.lower(model, graph),
            Self::Batchnorm(layer) => layer.lower(model, graph),
            Self::BiDirectionalLstm(layer) => layer.lower(model, graph),
            Self::Convolution(layer) => layer.lower(model, graph),
            Self::Crop(layer) => layer.lower(model, graph),
            Self::Embedding(layer) => layer.lower(model, graph),
            Self::Flatten(layer) => layer.lower(model, graph),
            Self::Gru(layer) => layer.lower(model, graph),
            Self::InnerProduct(layer) => layer.lower(model, graph),
            Self::LoadConstant(layer) => layer.lower(model, graph),
            Self::Lrn(layer) => layer.lower(model, graph),
            Self::Multiply(layer) => layer.lower(model, graph),
            Self::Padding(layer) => layer.lower(model, graph),
            Self::Permute(layer) => layer.lower(model, graph),
            Self::Pooling(layer) => layer.lower(model, graph),
            Self::Reshape(layer) => layer.lower(model, graph),
            Self::Scale(layer) => layer.lower(model, graph),
            Self::Slice(layer) => layer.lower(model, graph),
            Self::Simple(_) => (Vec::new(), Vec::new()),
            Self::UniDirectionalLstm(layer) => layer.lower(model, graph),
            Self::Unary(layer) => layer.lower(model, graph),
            Self::Upsample(layer) => layer.lower(model, graph),
            Self::Unsupported => (Vec::new(), Vec::new()),
        }
    }
}

#[derive(Clone)]
struct NeuralNetworkPreprocessing {
    feature_name: String,
    kind: PreprocessingKind,
}

impl Default for NeuralNetworkPreprocessing {
    fn default() -> Self {
        Self {
            feature_name: String::new(),
            kind: PreprocessingKind::Unsupported,
        }
    }
}

impl NeuralNetworkPreprocessing {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.feature_name = field.string()?,
                10 => result.kind = PreprocessingKind::Scaler(ImageScaler::decode(field.bytes()?)?),
                11 => {
                    result.kind = PreprocessingKind::MeanImage(MeanImage::decode(field.bytes()?)?)
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Clone)]
enum PreprocessingKind {
    Scaler(ImageScaler),
    MeanImage(MeanImage),
    Unsupported,
}

#[derive(Clone, Default)]
struct ImageScaler {
    channel_scale: Option<f32>,
    blue_bias: Option<f32>,
    green_bias: Option<f32>,
    red_bias: Option<f32>,
    gray_bias: Option<f32>,
}

impl ImageScaler {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                10 => result.channel_scale = Some(field.f32()?),
                20 => result.blue_bias = Some(field.f32()?),
                21 => result.green_bias = Some(field.f32()?),
                22 => result.red_bias = Some(field.f32()?),
                30 => result.gray_bias = Some(field.f32()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if let Some(value) = self.channel_scale {
            attributes.push(f32_scalar_attribute(model, "channelScale", value));
        }
        if let Some(value) = self.blue_bias {
            attributes.push(f32_scalar_attribute(model, "blueBias", value));
        }
        if let Some(value) = self.green_bias {
            attributes.push(f32_scalar_attribute(model, "greenBias", value));
        }
        if let Some(value) = self.red_bias {
            attributes.push(f32_scalar_attribute(model, "redBias", value));
        }
        if let Some(value) = self.gray_bias {
            attributes.push(f32_scalar_attribute(model, "grayBias", value));
        }
        attributes
    }
}

#[derive(Clone, Default)]
struct MeanImage {
    values: Vec<f32>,
}

impl MeanImage {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.values.extend(field.repeated_f32()?);
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        vec![Attribute {
            name: model.intern("meanImage"),
            value: if self
                .values
                .iter()
                .all(|value| value.is_finite() && value.fract() == 0.0)
            {
                AttributeValue::Ints(self.values.iter().map(|value| *value as i64).collect())
            } else {
                AttributeValue::Floats(self.values.clone())
            },
        }]
    }
}

#[derive(Default)]
struct AddLayer {
    alpha: Option<f32>,
}

impl AddLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.alpha = Some(field.f32()?);
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let attributes = self
            .alpha
            .map(|value| vec![f32_scalar_attribute(model, "alpha", value)])
            .unwrap_or_default();
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct MultiplyLayer {
    alpha: Option<f32>,
}

impl MultiplyLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.alpha = Some(field.f32()?);
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let attributes = self
            .alpha
            .map(|value| vec![f32_scalar_attribute(model, "alpha", value)])
            .unwrap_or_default();
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct BatchnormLayer {
    channels: Option<u64>,
    compute_mean_var: bool,
    instance_normalization: bool,
    epsilon: Option<f32>,
    gamma: Option<WeightParams>,
    beta: Option<WeightParams>,
    mean: Option<WeightParams>,
    variance: Option<WeightParams>,
}

impl BatchnormLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.channels = Some(field.varint()?),
                5 => result.compute_mean_var = field.bool()?,
                6 => result.instance_normalization = field.bool()?,
                10 => result.epsilon = Some(field.f32()?),
                15 => result.gamma = Some(WeightParams::decode(field.bytes()?)?),
                16 => result.beta = Some(WeightParams::decode(field.bytes()?)?),
                17 => result.mean = Some(WeightParams::decode(field.bytes()?)?),
                18 => result.variance = Some(WeightParams::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        let channels = self.channels.unwrap_or_default();
        if let Some(channels) = self.channels {
            attributes.push(string_attribute(model, "channels", &channels.to_string()));
        }
        if self.compute_mean_var {
            attributes.push(Attribute {
                name: model.intern("computeMeanVar"),
                value: AttributeValue::Bool(true),
            });
        }
        if self.instance_normalization {
            attributes.push(Attribute {
                name: model.intern("instanceNormalization"),
                value: AttributeValue::Bool(true),
            });
        }
        if let Some(epsilon) = self.epsilon {
            attributes.push(Attribute {
                name: model.intern("epsilon"),
                value: AttributeValue::Float(epsilon),
            });
        }

        let mut initializers = Vec::new();
        let shape = vec![channels as i64];
        push_lstm_weight(model, graph, &mut initializers, self.gamma, shape.clone());
        push_lstm_weight(model, graph, &mut initializers, self.beta, shape.clone());
        push_lstm_weight(model, graph, &mut initializers, self.mean, shape.clone());
        push_lstm_weight(model, graph, &mut initializers, self.variance, shape);
        (attributes, initializers)
    }
}

#[derive(Default)]
struct ConvolutionLayer {
    output_channels: u64,
    kernel_channels: u64,
    n_groups: Option<u64>,
    kernel_size: Vec<u64>,
    stride: Vec<u64>,
    dilation_factor: Vec<u64>,
    valid: bool,
    same: bool,
    is_deconvolution: bool,
    has_bias: bool,
    weights: Option<WeightParams>,
    bias: Option<WeightParams>,
    output_shape: Vec<u64>,
}

impl ConvolutionLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.output_channels = field.varint()?,
                2 => result.kernel_channels = field.varint()?,
                10 => result.n_groups = Some(field.varint()?),
                20 => result.kernel_size.extend(field.repeated_u64()?),
                30 => result.stride.extend(field.repeated_u64()?),
                40 => result.dilation_factor.extend(field.repeated_u64()?),
                50 => {
                    field.bytes()?;
                    result.valid = true;
                }
                51 => {
                    field.bytes()?;
                    result.same = true;
                }
                60 => result.is_deconvolution = field.bool()?,
                70 => result.has_bias = field.bool()?,
                90 => result.weights = Some(WeightParams::decode(field.bytes()?)?),
                91 => result.bias = Some(WeightParams::decode(field.bytes()?)?),
                100 => result.output_shape.extend(field.repeated_u64()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![
            uint_array_attribute(model, "kernelSize", &self.kernel_size),
            uint_array_attribute(model, "stride", &self.stride),
            uint_array_attribute(model, "dilationFactor", &self.dilation_factor),
            uint_array_attribute(model, "outputShape", &self.output_shape),
            int_attribute(model, "outputChannels", self.output_channels as i64),
            int_attribute(model, "kernelChannels", self.kernel_channels as i64),
        ];
        if let Some(n_groups) = self.n_groups {
            attributes.push(int_attribute(model, "nGroups", n_groups as i64));
        }
        if self.valid {
            attributes.push(string_attribute(model, "valid", "[object Object]"));
        }
        if self.same {
            attributes.push(string_attribute(model, "same", "[object Object]"));
        }
        if self.is_deconvolution {
            attributes.push(Attribute {
                name: model.intern("isDeconvolution"),
                value: AttributeValue::Bool(true),
            });
        }
        if self.has_bias {
            attributes.push(Attribute {
                name: model.intern("hasBias"),
                value: AttributeValue::Bool(true),
            });
        }

        let kernel_height = self.kernel_size.first().copied().unwrap_or_default() as i64;
        let kernel_width = self.kernel_size.get(1).copied().unwrap_or_default() as i64;
        let mut weight_shape = vec![
            self.output_channels as i64,
            self.kernel_channels as i64,
            kernel_height,
            kernel_width,
        ];
        if self.is_deconvolution {
            let groups = self.n_groups.unwrap_or(1).max(1);
            weight_shape[0] = self.kernel_channels as i64;
            weight_shape[1] = (self.output_channels / groups) as i64;
        }

        let mut initializers = Vec::new();
        if let Some(weights) = self.weights {
            initializers.push(weight_initializer(model, graph, weights, weight_shape));
        }
        if self.has_bias
            && let Some(bias) = self.bias
        {
            initializers.push(weight_initializer(
                model,
                graph,
                bias,
                vec![self.output_channels as i64],
            ));
        }
        (attributes, initializers)
    }
}

#[derive(Default)]
struct ActivationLayer {
    kind: Option<&'static str>,
}

impl ActivationLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            let kind = match field.number {
                5 => "linear",
                10 => "ReLU",
                15 => "leakyReLU",
                20 => "thresholdedReLU",
                25 => "PReLU",
                30 => "tanh",
                31 => "scaledTanh",
                40 => "sigmoid",
                41 => "sigmoidHard",
                50 => "ELU",
                60 => "softsign",
                70 => "softplus",
                71 => "parametricSoftplus",
                _ => {
                    continue;
                }
            };
            field.bytes()?;
            result.kind = Some(kind);
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let attributes = self
            .kind
            .map(|kind| vec![string_attribute(model, kind, "[object Object]")])
            .unwrap_or_default();
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct PoolingLayer {
    pooling_type: Option<i32>,
    kernel_size: Vec<u64>,
    stride: Vec<u64>,
    valid: bool,
    same: bool,
    include_last_pixel: bool,
    avg_pool_exclude_padding: bool,
    global_pooling: bool,
}

impl PoolingLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.pooling_type = Some(field.int32()?),
                10 => result.kernel_size.extend(field.repeated_u64()?),
                20 => result.stride.extend(field.repeated_u64()?),
                30 => {
                    field.bytes()?;
                    result.valid = true;
                }
                31 => {
                    field.bytes()?;
                    result.same = true;
                }
                32 => {
                    field.bytes()?;
                    result.include_last_pixel = true;
                }
                50 => result.avg_pool_exclude_padding = field.bool()?,
                60 => result.global_pooling = field.bool()?,
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![
            uint_array_attribute(model, "kernelSize", &self.kernel_size),
            uint_array_attribute(model, "stride", &self.stride),
        ];
        if let Some(pooling_type) = self.pooling_type {
            let value = match pooling_type {
                0 => "MAX",
                1 => "AVERAGE",
                2 => "L2",
                _ => "?",
            };
            attributes.push(string_attribute(model, "type", value));
        }
        if self.valid {
            attributes.push(string_attribute(model, "valid", "[object Object]"));
        }
        if self.same {
            attributes.push(string_attribute(model, "same", "[object Object]"));
        }
        if self.include_last_pixel {
            attributes.push(string_attribute(
                model,
                "includeLastPixel",
                "[object Object]",
            ));
        }
        if self.avg_pool_exclude_padding {
            attributes.push(Attribute {
                name: model.intern("avgPoolExcludePadding"),
                value: AttributeValue::Bool(true),
            });
        }
        if self.global_pooling {
            attributes.push(Attribute {
                name: model.intern("globalPooling"),
                value: AttributeValue::Bool(true),
            });
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct CropLayer {
    crop_amounts: bool,
    offset: Vec<u64>,
}

impl CropLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.crop_amounts = true;
                }
                5 => result.offset.extend(field.repeated_u64()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![if self.offset.is_empty() {
            uint_array_attribute(model, "offset", &self.offset)
        } else {
            string_array_attribute(model, "offset", &self.offset)
        }];
        if self.crop_amounts {
            attributes.push(string_attribute(model, "cropAmounts", "[object Object]"));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct PaddingLayer {
    constant: bool,
    reflection: bool,
    replication: bool,
    padding_amounts: bool,
}

impl PaddingLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.constant = true;
                }
                2 => {
                    field.bytes()?;
                    result.reflection = true;
                }
                3 => {
                    field.bytes()?;
                    result.replication = true;
                }
                10 => {
                    field.bytes()?;
                    result.padding_amounts = true;
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if self.constant {
            attributes.push(string_attribute(model, "constant", "[object Object]"));
        }
        if self.reflection {
            attributes.push(string_attribute(model, "reflection", "[object Object]"));
        }
        if self.replication {
            attributes.push(string_attribute(model, "replication", "[object Object]"));
        }
        if self.padding_amounts {
            attributes.push(string_attribute(model, "paddingAmounts", "[object Object]"));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct LrnLayer {
    alpha: Option<f32>,
    beta: Option<f32>,
    local_size: Option<u64>,
    k: Option<f32>,
}

impl LrnLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.alpha = Some(field.f32()?),
                2 => result.beta = Some(field.f32()?),
                3 => result.local_size = Some(field.varint()?),
                4 => result.k = Some(field.f32()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if let Some(value) = self.alpha {
            attributes.push(f32_scalar_attribute(model, "alpha", value));
        }
        if let Some(value) = self.beta {
            attributes.push(f32_scalar_attribute(model, "beta", value));
        }
        if let Some(value) = self.local_size {
            attributes.push(string_attribute(model, "localSize", &value.to_string()));
        }
        if let Some(value) = self.k {
            attributes.push(f32_scalar_attribute(model, "k", value));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct PermuteLayer {
    axis: Vec<u64>,
}

impl PermuteLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.axis.extend(field.repeated_u64()?);
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        (
            vec![string_array_attribute(model, "axis", &self.axis)],
            Vec::new(),
        )
    }
}

#[derive(Default)]
struct ScaleLayer {
    shape_scale: Vec<u64>,
    scale: Option<WeightParams>,
    has_bias: bool,
    shape_bias: Vec<u64>,
    bias: Option<WeightParams>,
}

impl ScaleLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.shape_scale.extend(field.repeated_u64()?),
                2 => result.scale = Some(WeightParams::decode(field.bytes()?)?),
                3 => result.has_bias = field.bool()?,
                4 => result.shape_bias.extend(field.repeated_u64()?),
                5 => result.bias = Some(WeightParams::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![
            string_array_attribute(model, "shapeScale", &self.shape_scale),
            string_array_attribute(model, "shapeBias", &self.shape_bias),
        ];
        if self.has_bias {
            attributes.push(Attribute {
                name: model.intern("hasBias"),
                value: AttributeValue::Bool(true),
            });
        }
        let mut initializers = Vec::new();
        let scale_shape = self
            .shape_scale
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>();
        let bias_shape = self
            .shape_bias
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>();
        push_lstm_weight(model, graph, &mut initializers, self.scale, scale_shape);
        if self.has_bias {
            push_lstm_weight(model, graph, &mut initializers, self.bias, bias_shape);
        }
        (attributes, initializers)
    }
}

#[derive(Default)]
struct LoadConstantLayer {
    shape: Vec<u64>,
    data: Option<WeightParams>,
}

impl LoadConstantLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.shape.extend(field.repeated_u64()?),
                2 => result.data = Some(WeightParams::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let attributes = vec![string_array_attribute(model, "shape", &self.shape)];
        let mut initializers = Vec::new();
        let shape = self
            .shape
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>();
        push_lstm_weight(model, graph, &mut initializers, self.data, shape);
        (attributes, initializers)
    }
}

#[derive(Default)]
struct UnaryLayer {
    op_type: Option<i32>,
    alpha: Option<f32>,
    epsilon: Option<f32>,
    shift: Option<f32>,
    scale: Option<f32>,
}

impl UnaryLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.op_type = Some(field.int32()?),
                2 => result.alpha = Some(field.f32()?),
                3 => result.epsilon = Some(field.f32()?),
                4 => result.shift = Some(field.f32()?),
                5 => result.scale = Some(field.f32()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if let Some(op_type) = self.op_type {
            let value = match op_type {
                0 => "SQRT",
                1 => "RSQRT",
                2 => "INVERSE",
                3 => "POWER",
                4 => "EXP",
                5 => "LOG",
                6 => "ABS",
                7 => "THRESHOLD",
                _ => "?",
            };
            attributes.push(string_attribute(model, "type", value));
        }
        if let Some(value) = self.alpha {
            attributes.push(f32_scalar_attribute(model, "alpha", value));
        }
        if let Some(value) = self.epsilon {
            attributes.push(f32_scalar_attribute(model, "epsilon", value));
        }
        if let Some(value) = self.shift {
            attributes.push(f32_scalar_attribute(model, "shift", value));
        }
        if let Some(value) = self.scale {
            attributes.push(f32_scalar_attribute(model, "scale", value));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct UpsampleLayer {
    scaling_factor: Vec<u64>,
    fractional_scaling_factor: Vec<f32>,
    mode: Option<i32>,
    linear_upsample_mode: Option<i32>,
}

impl UpsampleLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.scaling_factor.extend(field.repeated_u64()?),
                5 => result.mode = Some(field.int32()?),
                6 => result.linear_upsample_mode = Some(field.int32()?),
                7 => result
                    .fractional_scaling_factor
                    .extend(field.repeated_f32()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![
            string_array_attribute(model, "scalingFactor", &self.scaling_factor),
            Attribute {
                name: model.intern("fractionalScalingFactor"),
                value: if self.fractional_scaling_factor.is_empty() {
                    AttributeValue::Ints(Vec::new())
                } else {
                    AttributeValue::Floats(self.fractional_scaling_factor)
                },
            },
        ];
        if let Some(value) = self.mode {
            attributes.push(int_attribute(model, "mode", i64::from(value)));
        }
        if let Some(value) = self.linear_upsample_mode {
            attributes.push(int_attribute(model, "linearUpsampleMode", i64::from(value)));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct FlattenLayer {
    mode: Option<i32>,
}

impl FlattenLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.mode = Some(field.int32()?);
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let attributes = self
            .mode
            .map(|mode| {
                let value = match mode {
                    0 => "CHANNEL_FIRST",
                    1 => "CHANNEL_LAST",
                    _ => "?",
                };
                vec![string_attribute(model, "mode", value)]
            })
            .unwrap_or_default();
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct ReshapeLayer {
    target_shape: Vec<i64>,
    mode: Option<i32>,
}

impl ReshapeLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.target_shape.extend(field.repeated_i64()?),
                2 => result.mode = Some(field.int32()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if !self.target_shape.is_empty() {
            attributes.push(Attribute {
                name: model.intern("targetShape"),
                value: AttributeValue::Strings(
                    self.target_shape
                        .iter()
                        .map(|value| model.intern(value.to_string()))
                        .collect(),
                ),
            });
        }
        if let Some(mode) = self.mode {
            let value = match mode {
                0 => "CHANNEL_FIRST",
                1 => "CHANNEL_LAST",
                _ => "?",
            };
            attributes.push(string_attribute(model, "mode", value));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct SliceLayer {
    start_index: Option<i64>,
    end_index: Option<i64>,
    stride: Option<u64>,
    axis: Option<i32>,
}

impl SliceLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.start_index = Some(field.varint()? as i64),
                2 => result.end_index = Some(field.varint()? as i64),
                3 => result.stride = Some(field.varint()?),
                4 => result.axis = Some(field.int32()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, _graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if let Some(value) = self.start_index {
            attributes.push(string_attribute(model, "startIndex", &value.to_string()));
        }
        if let Some(value) = self.end_index {
            attributes.push(string_attribute(model, "endIndex", &value.to_string()));
        }
        if let Some(value) = self.stride {
            attributes.push(string_attribute(model, "stride", &value.to_string()));
        }
        if let Some(value) = self.axis {
            attributes.push(int_attribute(model, "axis", i64::from(value)));
        }
        (attributes, Vec::new())
    }
}

#[derive(Default)]
struct GruLayer {
    input_vector_size: u64,
    output_vector_size: u64,
    activations_count: usize,
    sequence_output: bool,
    has_bias_vectors: bool,
    update_gate_weight_matrix: Option<WeightParams>,
    reset_gate_weight_matrix: Option<WeightParams>,
    output_gate_weight_matrix: Option<WeightParams>,
    update_gate_recursion_matrix: Option<WeightParams>,
    reset_gate_recursion_matrix: Option<WeightParams>,
    output_gate_recursion_matrix: Option<WeightParams>,
    update_gate_bias_vector: Option<WeightParams>,
    reset_gate_bias_vector: Option<WeightParams>,
    output_gate_bias_vector: Option<WeightParams>,
    reverse_input: bool,
}

impl GruLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.input_vector_size = field.varint()?,
                2 => result.output_vector_size = field.varint()?,
                10 => {
                    field.bytes()?;
                    result.activations_count += 1;
                }
                15 => result.sequence_output = field.bool()?,
                20 => result.has_bias_vectors = field.bool()?,
                30 => {
                    result.update_gate_weight_matrix = Some(WeightParams::decode(field.bytes()?)?)
                }
                31 => result.reset_gate_weight_matrix = Some(WeightParams::decode(field.bytes()?)?),
                32 => {
                    result.output_gate_weight_matrix = Some(WeightParams::decode(field.bytes()?)?)
                }
                50 => {
                    result.update_gate_recursion_matrix =
                        Some(WeightParams::decode(field.bytes()?)?)
                }
                51 => {
                    result.reset_gate_recursion_matrix = Some(WeightParams::decode(field.bytes()?)?)
                }
                52 => {
                    result.output_gate_recursion_matrix =
                        Some(WeightParams::decode(field.bytes()?)?)
                }
                70 => result.update_gate_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                71 => result.reset_gate_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                72 => result.output_gate_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                100 => result.reverse_input = field.bool()?,
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if self.activations_count > 0 {
            attributes.push(object_list_attribute(
                model,
                "activations",
                self.activations_count,
            ));
        }
        attributes.push(string_attribute(
            model,
            "inputVectorSize",
            &self.input_vector_size.to_string(),
        ));
        attributes.push(string_attribute(
            model,
            "outputVectorSize",
            &self.output_vector_size.to_string(),
        ));
        if self.sequence_output {
            attributes.push(Attribute {
                name: model.intern("sequenceOutput"),
                value: AttributeValue::Bool(true),
            });
        }
        if self.has_bias_vectors {
            attributes.push(Attribute {
                name: model.intern("hasBiasVectors"),
                value: AttributeValue::Bool(true),
            });
        }
        if self.reverse_input {
            attributes.push(Attribute {
                name: model.intern("reverseInput"),
                value: AttributeValue::Bool(true),
            });
        }

        let h = self.output_vector_size as i64;
        let x = self.input_vector_size as i64;
        let mut initializers = Vec::new();
        push_lstm_weight(
            model,
            graph,
            &mut initializers,
            self.update_gate_weight_matrix,
            vec![h, x],
        );
        push_lstm_weight(
            model,
            graph,
            &mut initializers,
            self.reset_gate_weight_matrix,
            vec![h, x],
        );
        push_lstm_weight(
            model,
            graph,
            &mut initializers,
            self.output_gate_weight_matrix,
            vec![h, x],
        );
        push_lstm_weight(
            model,
            graph,
            &mut initializers,
            self.update_gate_recursion_matrix,
            vec![h, h],
        );
        push_lstm_weight(
            model,
            graph,
            &mut initializers,
            self.reset_gate_recursion_matrix,
            vec![h, h],
        );
        push_lstm_weight(
            model,
            graph,
            &mut initializers,
            self.output_gate_recursion_matrix,
            vec![h, h],
        );
        if self.has_bias_vectors {
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                self.update_gate_bias_vector,
                vec![h],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                self.reset_gate_bias_vector,
                vec![h],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                self.output_gate_bias_vector,
                vec![h],
            );
        }
        (attributes, initializers)
    }
}

#[derive(Default)]
struct LstmParams {
    has_bias_vectors: bool,
    has_peephole_vectors: bool,
}

impl LstmParams {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                20 => result.has_bias_vectors = field.bool()?,
                40 => result.has_peephole_vectors = field.bool()?,
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct LstmWeightParams {
    input_gate_weight_matrix: Option<WeightParams>,
    forget_gate_weight_matrix: Option<WeightParams>,
    block_input_weight_matrix: Option<WeightParams>,
    output_gate_weight_matrix: Option<WeightParams>,
    input_gate_recursion_matrix: Option<WeightParams>,
    forget_gate_recursion_matrix: Option<WeightParams>,
    block_input_recursion_matrix: Option<WeightParams>,
    output_gate_recursion_matrix: Option<WeightParams>,
    input_gate_bias_vector: Option<WeightParams>,
    forget_gate_bias_vector: Option<WeightParams>,
    block_input_bias_vector: Option<WeightParams>,
    output_gate_bias_vector: Option<WeightParams>,
    input_gate_peephole_vector: Option<WeightParams>,
    forget_gate_peephole_vector: Option<WeightParams>,
    output_gate_peephole_vector: Option<WeightParams>,
}

impl LstmWeightParams {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.input_gate_weight_matrix = Some(WeightParams::decode(field.bytes()?)?),
                2 => result.forget_gate_weight_matrix = Some(WeightParams::decode(field.bytes()?)?),
                3 => result.block_input_weight_matrix = Some(WeightParams::decode(field.bytes()?)?),
                4 => result.output_gate_weight_matrix = Some(WeightParams::decode(field.bytes()?)?),
                20 => {
                    result.input_gate_recursion_matrix = Some(WeightParams::decode(field.bytes()?)?)
                }
                21 => {
                    result.forget_gate_recursion_matrix =
                        Some(WeightParams::decode(field.bytes()?)?)
                }
                22 => {
                    result.block_input_recursion_matrix =
                        Some(WeightParams::decode(field.bytes()?)?)
                }
                23 => {
                    result.output_gate_recursion_matrix =
                        Some(WeightParams::decode(field.bytes()?)?)
                }
                40 => result.input_gate_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                41 => result.forget_gate_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                42 => result.block_input_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                43 => result.output_gate_bias_vector = Some(WeightParams::decode(field.bytes()?)?),
                60 => {
                    result.input_gate_peephole_vector = Some(WeightParams::decode(field.bytes()?)?)
                }
                61 => {
                    result.forget_gate_peephole_vector = Some(WeightParams::decode(field.bytes()?)?)
                }
                62 => {
                    result.output_gate_peephole_vector = Some(WeightParams::decode(field.bytes()?)?)
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct UniDirectionalLstmLayer {
    input_vector_size: u64,
    output_vector_size: u64,
    activations_count: usize,
    params: Option<LstmParams>,
    weight_params: Option<LstmWeightParams>,
    reverse_input: bool,
}

impl UniDirectionalLstmLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.input_vector_size = field.varint()?,
                2 => result.output_vector_size = field.varint()?,
                10 => {
                    field.bytes()?;
                    result.activations_count += 1;
                }
                15 => result.params = Some(LstmParams::decode(field.bytes()?)?),
                20 => result.weight_params = Some(LstmWeightParams::decode(field.bytes()?)?),
                100 => result.reverse_input = field.bool()?,
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if self.activations_count > 0 {
            attributes.push(object_list_attribute(
                model,
                "activations",
                self.activations_count,
            ));
        }
        attributes.push(string_attribute(
            model,
            "inputVectorSize",
            &self.input_vector_size.to_string(),
        ));
        attributes.push(string_attribute(
            model,
            "outputVectorSize",
            &self.output_vector_size.to_string(),
        ));
        if self.params.is_some() {
            attributes.push(string_attribute(model, "params", "[object Object]"));
        }
        if self.reverse_input {
            attributes.push(Attribute {
                name: model.intern("reverseInput"),
                value: AttributeValue::Bool(true),
            });
        }

        let mut initializers = Vec::new();
        if let Some(weights) = self.weight_params {
            let h = self.output_vector_size as i64;
            let x = self.input_vector_size as i64;
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.input_gate_weight_matrix,
                vec![h, x],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.forget_gate_weight_matrix,
                vec![h, x],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.block_input_weight_matrix,
                vec![h, x],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.output_gate_weight_matrix,
                vec![h, x],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.input_gate_recursion_matrix,
                vec![h, h],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.forget_gate_recursion_matrix,
                vec![h, h],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.block_input_recursion_matrix,
                vec![h, h],
            );
            push_lstm_weight(
                model,
                graph,
                &mut initializers,
                weights.output_gate_recursion_matrix,
                vec![h, h],
            );

            let params = self.params.unwrap_or_default();
            if params.has_bias_vectors {
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.input_gate_bias_vector,
                    vec![h],
                );
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.forget_gate_bias_vector,
                    vec![h],
                );
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.block_input_bias_vector,
                    vec![h],
                );
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.output_gate_bias_vector,
                    vec![h],
                );
            }
            if params.has_peephole_vectors {
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.input_gate_peephole_vector,
                    vec![h],
                );
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.forget_gate_peephole_vector,
                    vec![h],
                );
                push_lstm_weight(
                    model,
                    graph,
                    &mut initializers,
                    weights.output_gate_peephole_vector,
                    vec![h],
                );
            }
        }
        (attributes, initializers)
    }
}

#[derive(Default)]
struct BiDirectionalLstmLayer {
    input_vector_size: u64,
    output_vector_size: u64,
    activations_forward_count: usize,
    activations_backward_count: usize,
    params: Option<LstmParams>,
    weight_params: Vec<LstmWeightParams>,
}

impl BiDirectionalLstmLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.input_vector_size = field.varint()?,
                2 => result.output_vector_size = field.varint()?,
                10 => {
                    field.bytes()?;
                    result.activations_forward_count += 1;
                }
                11 => {
                    field.bytes()?;
                    result.activations_backward_count += 1;
                }
                15 => result.params = Some(LstmParams::decode(field.bytes()?)?),
                20 => result
                    .weight_params
                    .push(LstmWeightParams::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = Vec::new();
        if self.activations_forward_count > 0 {
            attributes.push(object_list_attribute(
                model,
                "activationsForwardLSTM",
                self.activations_forward_count,
            ));
        }
        if self.activations_backward_count > 0 {
            attributes.push(object_list_attribute(
                model,
                "activationsBackwardLSTM",
                self.activations_backward_count,
            ));
        }
        attributes.push(string_attribute(
            model,
            "inputVectorSize",
            &self.input_vector_size.to_string(),
        ));
        attributes.push(string_attribute(
            model,
            "outputVectorSize",
            &self.output_vector_size.to_string(),
        ));
        if self.params.is_some() {
            attributes.push(string_attribute(model, "params", "[object Object]"));
        }

        let params = self.params.unwrap_or_default();
        let mut initializers = Vec::new();
        for weights in self.weight_params {
            push_lstm_weight_params(
                model,
                graph,
                &mut initializers,
                weights,
                &params,
                self.output_vector_size as i64,
                self.input_vector_size as i64,
            );
        }
        (attributes, initializers)
    }
}

fn push_lstm_weight_params(
    model: &mut Model,
    graph: &mut Graph,
    initializers: &mut Vec<ValueId>,
    weights: LstmWeightParams,
    params: &LstmParams,
    h: i64,
    x: i64,
) {
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.input_gate_weight_matrix,
        vec![h, x],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.forget_gate_weight_matrix,
        vec![h, x],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.block_input_weight_matrix,
        vec![h, x],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.output_gate_weight_matrix,
        vec![h, x],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.input_gate_recursion_matrix,
        vec![h, h],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.forget_gate_recursion_matrix,
        vec![h, h],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.block_input_recursion_matrix,
        vec![h, h],
    );
    push_lstm_weight(
        model,
        graph,
        initializers,
        weights.output_gate_recursion_matrix,
        vec![h, h],
    );
    if params.has_bias_vectors {
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.input_gate_bias_vector,
            vec![h],
        );
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.forget_gate_bias_vector,
            vec![h],
        );
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.block_input_bias_vector,
            vec![h],
        );
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.output_gate_bias_vector,
            vec![h],
        );
    }
    if params.has_peephole_vectors {
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.input_gate_peephole_vector,
            vec![h],
        );
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.forget_gate_peephole_vector,
            vec![h],
        );
        push_lstm_weight(
            model,
            graph,
            initializers,
            weights.output_gate_peephole_vector,
            vec![h],
        );
    }
}

fn push_lstm_weight(
    model: &mut Model,
    graph: &mut Graph,
    initializers: &mut Vec<ValueId>,
    weight: Option<WeightParams>,
    shape: Vec<i64>,
) {
    if let Some(weight) = weight {
        initializers.push(weight_initializer(model, graph, weight, shape));
    }
}

#[derive(Default)]
struct EmbeddingLayer {
    input_dim: u64,
    output_channels: u64,
    has_bias: bool,
    weights: Option<WeightParams>,
    bias: Option<WeightParams>,
}

impl EmbeddingLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.input_dim = field.varint()?,
                2 => result.output_channels = field.varint()?,
                10 => result.has_bias = field.bool()?,
                20 => result.weights = Some(WeightParams::decode(field.bytes()?)?),
                21 => result.bias = Some(WeightParams::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![
            string_attribute(model, "inputDim", &self.input_dim.to_string()),
            string_attribute(model, "outputChannels", &self.output_channels.to_string()),
        ];
        if self.has_bias {
            attributes.push(Attribute {
                name: model.intern("hasBias"),
                value: AttributeValue::Bool(true),
            });
        }

        let mut initializers = Vec::new();
        if let Some(weights) = self.weights {
            initializers.push(weight_initializer(
                model,
                graph,
                weights,
                vec![self.input_dim as i64, self.output_channels as i64],
            ));
        }
        if self.has_bias
            && let Some(bias) = self.bias
        {
            initializers.push(weight_initializer(
                model,
                graph,
                bias,
                vec![self.output_channels as i64],
            ));
        }
        (attributes, initializers)
    }
}

#[derive(Default)]
struct InnerProductLayer {
    input_channels: u64,
    output_channels: u64,
    has_bias: bool,
    weights: Option<WeightParams>,
    bias: Option<WeightParams>,
}

impl InnerProductLayer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.input_channels = field.varint()?,
                2 => result.output_channels = field.varint()?,
                10 => result.has_bias = field.bool()?,
                20 => result.weights = Some(WeightParams::decode(field.bytes()?)?),
                21 => result.bias = Some(WeightParams::decode(field.bytes()?)?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn lower(self, model: &mut Model, graph: &mut Graph) -> (Vec<Attribute>, Vec<ValueId>) {
        let mut attributes = vec![
            int_attribute(model, "inputChannels", self.input_channels as i64),
            int_attribute(model, "outputChannels", self.output_channels as i64),
        ];
        if self.has_bias {
            attributes.push(Attribute {
                name: model.intern("hasBias"),
                value: AttributeValue::Bool(true),
            });
        }

        let mut initializers = Vec::new();
        if let Some(weights) = self.weights {
            initializers.push(weight_initializer(
                model,
                graph,
                weights,
                vec![self.output_channels as i64, self.input_channels as i64],
            ));
        }
        if self.has_bias
            && let Some(bias) = self.bias
        {
            initializers.push(weight_initializer(
                model,
                graph,
                bias,
                vec![self.output_channels as i64],
            ));
        }
        (attributes, initializers)
    }
}

#[derive(Default)]
struct WeightParams {
    element_type: Option<TensorElementType>,
    storage: WeightStorage,
    quantization: Option<WeightQuantization>,
}

impl WeightParams {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    let values = field.repeated_f32()?;
                    result.element_type = Some(TensorElementType::Float32);
                    result.storage = WeightStorage::ElementList { len: values.len() };
                }
                2 => {
                    let bytes = field.bytes()?;
                    result.element_type = Some(TensorElementType::Float16);
                    result.storage = WeightStorage::InlineBytes {
                        byte_len: bytes.len(),
                    };
                }
                30 => {
                    let bytes = field.bytes()?;
                    result.element_type = Some(TensorElementType::Float32);
                    result.storage = WeightStorage::InlineBytes {
                        byte_len: bytes.len(),
                    };
                }
                31 => {
                    let bytes = field.bytes()?;
                    result.element_type = Some(TensorElementType::Int8);
                    result.storage = WeightStorage::InlineBytes {
                        byte_len: bytes.len(),
                    };
                }
                40 => {
                    let quantization = WeightQuantization::decode(field.bytes()?)?;
                    if let Some(bits) = quantization.number_of_bits {
                        result.element_type = Some(match bits {
                            1 => TensorElementType::Other("uint1".to_owned()),
                            2 => TensorElementType::Uint2,
                            4 => TensorElementType::Uint4,
                            8 => TensorElementType::Uint8,
                            _ => TensorElementType::Other(format!("uint{bits}")),
                        });
                    }
                    result.quantization = Some(quantization);
                }
                _ => {}
            }
        }
        Ok(result)
    }
}

struct WeightQuantization {
    number_of_bits: Option<u64>,
    kind: Option<&'static str>,
    scale: Vec<f32>,
    bias: Vec<f32>,
}

impl WeightQuantization {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self {
            number_of_bits: None,
            kind: None,
            scale: Vec::new(),
            bias: Vec::new(),
        };
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.number_of_bits = Some(field.varint()?),
                101 => {
                    result.kind = Some("linear");
                    let (scale, bias) = read_linear_quantization(field.bytes()?)?;
                    result.scale = scale;
                    result.bias = bias;
                }
                102 => {
                    field.bytes()?;
                }
                _ => {}
            }
        }
        Ok(result)
    }

    fn annotations(self, model: &mut Model) -> Vec<QuantizationAnnotation> {
        let mut annotations = Vec::new();
        if let Some(kind) = self.kind {
            annotations.push(QuantizationAnnotation {
                key: model.intern("type"),
                value: model.intern(kind),
            });
        }
        if !self.scale.is_empty() {
            annotations.push(QuantizationAnnotation {
                key: model.intern("scale"),
                value: model.intern(join_f32_list(&self.scale)),
            });
        }
        if !self.bias.is_empty() {
            annotations.push(QuantizationAnnotation {
                key: model.intern("bias"),
                value: model.intern(join_f32_list(&self.bias)),
            });
        }
        annotations
    }
}

fn read_linear_quantization(data: &[u8]) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
    let mut scale = Vec::new();
    let mut bias = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        match field.number {
            1 => scale.extend(field.repeated_f32()?),
            2 => bias.extend(field.repeated_f32()?),
            _ => {}
        }
    }
    Ok((scale, bias))
}

fn join_f32_list(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Default)]
enum WeightStorage {
    #[default]
    Absent,
    ElementList {
        len: usize,
    },
    InlineBytes {
        byte_len: usize,
    },
}

#[derive(Default)]
struct Normalizer {
    norm_type: Option<i32>,
}

impl Normalizer {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.norm_type = Some(field.int32()?);
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct ArrayFeatureExtractor {
    extract_index: Vec<u64>,
}

impl ArrayFeatureExtractor {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            if field.number == 1 {
                result.extract_index.extend(field.repeated_u64()?);
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct NonMaximumSuppression {
    pick_top: bool,
    string_class_labels: Option<Vec<String>>,
    int64_class_labels: Option<Vec<String>>,
    iou_threshold: Option<f64>,
    confidence_threshold: Option<f64>,
    confidence_input_feature_name: String,
    coordinates_input_feature_name: String,
    iou_threshold_input_feature_name: String,
    confidence_threshold_input_feature_name: String,
    confidence_output_feature_name: String,
    coordinates_output_feature_name: String,
}

impl NonMaximumSuppression {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => {
                    field.bytes()?;
                    result.pick_top = true;
                }
                100 => result.string_class_labels = Some(read_string_vector(field.bytes()?)?),
                101 => result.int64_class_labels = Some(read_int64_vector(field.bytes()?)?),
                110 => result.iou_threshold = Some(field.f64()?),
                111 => result.confidence_threshold = Some(field.f64()?),
                200 => result.confidence_input_feature_name = field.string()?,
                201 => result.coordinates_input_feature_name = field.string()?,
                202 => result.iou_threshold_input_feature_name = field.string()?,
                203 => result.confidence_threshold_input_feature_name = field.string()?,
                210 => result.confidence_output_feature_name = field.string()?,
                211 => result.coordinates_output_feature_name = field.string()?,
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if self.pick_top {
            attributes.push(string_attribute(model, "pickTop", "[object Object]"));
        }
        if let Some(labels) = &self.string_class_labels {
            attributes.push(string_vector_attribute(model, "stringClassLabels", labels));
        }
        if let Some(labels) = &self.int64_class_labels {
            attributes.push(string_vector_attribute(model, "int64ClassLabels", labels));
        }
        if let Some(value) = self.iou_threshold {
            attributes.push(Attribute {
                name: model.intern("iouThreshold"),
                value: AttributeValue::Float(value as f32),
            });
        }
        if let Some(value) = self.confidence_threshold {
            attributes.push(Attribute {
                name: model.intern("confidenceThreshold"),
                value: AttributeValue::Float(value as f32),
            });
        }
        attributes
    }
}

#[derive(Default)]
struct Scaler {
    shift_value: Vec<f64>,
    scale_value: Vec<f64>,
}

impl Scaler {
    fn decode(data: &[u8]) -> Result<Self, ModelError> {
        let mut result = Self::default();
        let mut reader = PbReader::new(data);
        while let Some(field) = reader.next_field()? {
            match field.number {
                1 => result.shift_value.extend(field.repeated_f64()?),
                2 => result.scale_value.extend(field.repeated_f64()?),
                _ => {}
            }
        }
        Ok(result)
    }

    fn attributes(&self, model: &mut Model) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        if !self.shift_value.is_empty() {
            attributes.push(numeric_array_attribute(
                model,
                "shiftValue",
                &self.shift_value,
            ));
        }
        if !self.scale_value.is_empty() {
            attributes.push(numeric_array_attribute(
                model,
                "scaleValue",
                &self.scale_value,
            ));
        }
        attributes
    }
}

fn read_string_vector(data: &[u8]) -> Result<Vec<String>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.push(field.string()?);
        }
    }
    Ok(values)
}

fn read_int64_vector(data: &[u8]) -> Result<Vec<String>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(
                field
                    .repeated_i64()?
                    .into_iter()
                    .map(|value| value.to_string()),
            );
        }
    }
    Ok(values)
}

fn read_double_array(data: &[u8]) -> Result<Vec<f64>, ModelError> {
    let mut values = Vec::new();
    let mut reader = PbReader::new(data);
    while let Some(field) = reader.next_field()? {
        if field.number == 1 {
            values.extend(field.repeated_f64()?);
        }
    }
    Ok(values)
}

#[derive(Clone, Copy)]
struct Field<'a> {
    number: u32,
    value: FieldValue<'a>,
}

impl<'a> Field<'a> {
    fn bytes(self) -> Result<&'a [u8], ModelError> {
        match self.value {
            FieldValue::Bytes(value) => Ok(value),
            _ => Err(invalid("expected length-delimited field")),
        }
    }

    fn string(self) -> Result<String, ModelError> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|error| invalid(format!("string field is not UTF-8: {error}")))
    }

    fn uint32(self) -> Result<u32, ModelError> {
        Ok(self.varint()? as u32)
    }

    fn int32(self) -> Result<i32, ModelError> {
        Ok(self.varint()? as i32)
    }

    fn bool(self) -> Result<bool, ModelError> {
        Ok(self.varint()? != 0)
    }

    fn repeated_u64(self) -> Result<Vec<u64>, ModelError> {
        match self.value {
            FieldValue::Varint(value) => Ok(vec![value]),
            FieldValue::Bytes(bytes) => {
                let mut reader = ScalarReader::new(bytes);
                let mut values = Vec::new();
                while !reader.is_empty() {
                    values.push(reader.varint()?);
                }
                Ok(values)
            }
            _ => Err(invalid("expected repeated uint64 field")),
        }
    }

    fn repeated_i64(self) -> Result<Vec<i64>, ModelError> {
        match self.value {
            FieldValue::Varint(value) => Ok(vec![value as i64]),
            FieldValue::Bytes(bytes) => {
                let mut reader = ScalarReader::new(bytes);
                let mut values = Vec::new();
                while !reader.is_empty() {
                    values.push(reader.varint()? as i64);
                }
                Ok(values)
            }
            _ => Err(invalid("expected repeated int64 field")),
        }
    }

    fn repeated_f64(self) -> Result<Vec<f64>, ModelError> {
        match self.value {
            FieldValue::Fixed64(value) => Ok(vec![f64::from_bits(value)]),
            FieldValue::Bytes(bytes) => {
                if bytes.len() % 8 != 0 {
                    return Err(invalid("packed double field has non-8 byte length"));
                }
                Ok(bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_bits(u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]))
                    })
                    .collect())
            }
            _ => Err(invalid("expected repeated double field")),
        }
    }

    fn repeated_f32(self) -> Result<Vec<f32>, ModelError> {
        match self.value {
            FieldValue::Fixed32(value) => Ok(vec![f32::from_bits(value)]),
            FieldValue::Bytes(bytes) => {
                if bytes.len() % 4 != 0 {
                    return Err(invalid("packed float field has non-4 byte length"));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    })
                    .collect())
            }
            _ => Err(invalid("expected repeated float field")),
        }
    }

    fn f64(self) -> Result<f64, ModelError> {
        match self.value {
            FieldValue::Fixed64(value) => Ok(f64::from_bits(value)),
            _ => Err(invalid("expected fixed64 field")),
        }
    }

    fn f32(self) -> Result<f32, ModelError> {
        match self.value {
            FieldValue::Fixed32(value) => Ok(f32::from_bits(value)),
            _ => Err(invalid("expected fixed32 field")),
        }
    }

    fn varint(self) -> Result<u64, ModelError> {
        match self.value {
            FieldValue::Varint(value) => Ok(value),
            _ => Err(invalid("expected varint field")),
        }
    }
}

#[derive(Clone, Copy)]
enum FieldValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

struct PbReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> PbReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn next_field(&mut self) -> Result<Option<Field<'a>>, ModelError> {
        if self.position == self.data.len() {
            return Ok(None);
        }

        let key = self.varint()?;
        let number = (key >> 3) as u32;
        let wire_type = (key & 0x07) as u8;
        if number == 0 {
            return Err(invalid("field number 0 is invalid"));
        }

        let value = match wire_type {
            0 => FieldValue::Varint(self.varint()?),
            1 => {
                let bytes = self.take(8)?;
                FieldValue::Fixed64(u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            }
            2 => {
                let len = self.varint()? as usize;
                FieldValue::Bytes(self.take(len)?)
            }
            5 => {
                let bytes = self.take(4)?;
                FieldValue::Fixed32(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            _ => {
                return Err(invalid(format!(
                    "unsupported protobuf wire type {wire_type}"
                )));
            }
        };

        Ok(Some(Field { number, value }))
    }

    fn varint(&mut self) -> Result<u64, ModelError> {
        let mut result = 0_u64;
        for shift in (0..64).step_by(7) {
            let Some(byte) = self.data.get(self.position).copied() else {
                return Err(invalid("unexpected end of protobuf varint"));
            };
            self.position += 1;
            result |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(invalid("protobuf varint is too long"))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ModelError> {
        let Some(end) = self.position.checked_add(len) else {
            return Err(invalid("protobuf length overflow"));
        };
        if end > self.data.len() {
            return Err(invalid("unexpected end of protobuf field"));
        }
        let value = &self.data[self.position..end];
        self.position = end;
        Ok(value)
    }
}

struct ScalarReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> ScalarReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn is_empty(&self) -> bool {
        self.position == self.data.len()
    }

    fn varint(&mut self) -> Result<u64, ModelError> {
        let mut result = 0_u64;
        for shift in (0..64).step_by(7) {
            let Some(byte) = self.data.get(self.position).copied() else {
                return Err(invalid("unexpected end of packed varint"));
            };
            self.position += 1;
            result |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(invalid("packed varint is too long"))
    }
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: FORMAT,
        message: message.into(),
    }
}
