use netron_rs_core::{
    FormatInfo, Graph, Model, ModelError, Node, Operator, StringId, TensorId, TensorStorage, Value,
};

#[test]
fn valid_graph_with_reciprocal_edges_passes() {
    let model = valid_model();

    model.validate().expect("valid model passes invariants");
}

#[test]
fn node_input_must_be_listed_as_value_consumer() {
    let mut model = valid_model();
    model.graphs[0].values[0].consumers.clear();

    assert_invariant_contains(
        model.validate(),
        "input value 0 does not list the node as a consumer",
    );
}

#[test]
fn node_output_must_be_listed_as_value_producer() {
    let mut model = valid_model();
    model.graphs[0].values[1].producer = None;

    assert_invariant_contains(
        model.validate(),
        "output value 1 does not list the node as producer",
    );
}

#[test]
fn string_ids_must_resolve_before_normalization() {
    let mut model = valid_model();
    model.graphs[0].nodes[0].operator.name = StringId::new(99);

    assert_invariant_contains(
        model.validate(),
        "operator name string id 99 does not resolve",
    );
}

#[test]
fn subgraph_parent_links_are_reciprocal() {
    let mut model = Model::new(FormatInfo {
        name: "test",
        version: None,
    });
    let parent = model.add_graph_placeholder(None, None);
    let child = model.add_graph_placeholder(Some(parent), None);
    model.replace_graph(parent, Graph::new(parent, None, None));
    model.replace_graph(child, Graph::new(child, Some(parent), None));

    assert_invariant_contains(model.validate(), "parent 0 does not list it as a subgraph");
}

#[test]
fn tensor_references_must_resolve() {
    let mut model = valid_model();
    model.graphs[0].values[0].initializer = Some(TensorId::new(7));

    assert_invariant_contains(model.validate(), "initializer 7 does not resolve");
}

#[test]
fn graph_inputs_must_match_value_flags() {
    let mut model = valid_model();
    model.graphs[0].values[0].is_graph_input = false;

    assert_invariant_contains(
        model.validate(),
        "input value 0 is not marked as graph input",
    );
}

#[test]
fn graph_outputs_must_match_value_flags() {
    let mut model = valid_model();
    model.graphs[0].values[1].is_graph_output = false;

    assert_invariant_contains(
        model.validate(),
        "output value 1 is not marked as graph output",
    );
}

#[test]
fn hidden_graph_output_flags_are_allowed() {
    let mut model = valid_model();
    model.graphs[0].outputs.clear();

    model
        .validate()
        .expect("hidden graph output flag is allowed");
}

fn valid_model() -> Model {
    let mut model = Model::new(FormatInfo {
        name: "test",
        version: None,
    });
    let graph_id = model.add_graph_placeholder(None, None);
    let mut graph = Graph::new(graph_id, None, None);

    let input_name = model.intern("input");
    let output_name = model.intern("output");
    let input = graph.add_value(Value::new(input_name));
    let output = graph.add_value(Value::new(output_name));
    graph.inputs.push(input);
    graph.outputs.push(output);
    graph.values[input.index()].is_graph_input = true;
    graph.values[output.index()].is_graph_output = true;

    let op_name = model.intern("Identity");
    let mut node = Node::new(
        graph_id,
        Operator {
            domain: None,
            name: op_name,
            overload: None,
            version: None,
            origin: "test",
        },
    );
    node.inputs.push(Some(input));
    node.outputs.push(Some(output));
    let node_id = graph.add_node(node);
    graph.values[input.index()].consumers.push(node_id);
    graph.values[output.index()].producer = Some(node_id);

    let tensor_name = model.intern("unused");
    model.add_tensor(netron_rs_core::Tensor::metadata_only(
        Some(tensor_name),
        netron_rs_core::TensorElementType::Float32,
        Vec::new(),
        TensorStorage::Absent,
    ));

    model.replace_graph(graph_id, graph);
    model
}

fn assert_invariant_contains(result: Result<(), ModelError>, expected: &str) {
    match result {
        Err(ModelError::Invariant(message)) if message.contains(expected) => {}
        other => panic!("expected invariant containing {expected:?}, got {other:?}"),
    }
}
