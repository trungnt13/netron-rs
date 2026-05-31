#!/usr/bin/env node

import * as child_process from 'child_process';
import * as path from 'path';
import * as url from 'url';

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..');

const main = async () => {
    const args = parseArgs(process.argv.slice(2));
    if (args.files.length === 0) {
        throw new Error('usage: node tools/golden-diff.mjs [--rust-bin <path>] <model>...');
    }

    const results = [];
    for (const file of args.files) {
        const target = path.resolve(file);
        const oldModel = await oldNetron(target);
        const rustModel = await rustNetron(target, args.rustBin);
        const oldSummary = summarizeOld(oldModel, target);
        const rustSummary = summarizeRust(rustModel, target);
        const differences = diff(oldSummary, rustSummary);
        const result = {
            file: target,
            ok: differences.length === 0,
            differences: args.compact ? differences.map(compactDifference) : differences
        };
        if (!args.compact) {
            result.old = oldSummary;
            result.rust = rustSummary;
        }
        results.push(result);
    }

    const ok = results.every((result) => result.ok);
    console.log(JSON.stringify({ ok, results }, null, 2));
    if (!ok) {
        process.exitCode = 1;
    }
};

const parseArgs = (args) => {
    const parsed = { rustBin: null, compact: false, files: [] };
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg === '--compact') {
            parsed.compact = true;
        } else if (arg === '--rust-bin') {
            parsed.rustBin = args[++i] || null;
            if (!parsed.rustBin) {
                throw new Error('--rust-bin requires a path');
            }
        } else {
            parsed.files.push(arg);
        }
    }
    return parsed;
};

const compactDifference = (difference) => ({
    path: difference.path,
    old: preview(difference.old),
    rust: preview(difference.rust)
});

const preview = (value) => {
    const text = JSON.stringify(value);
    return text.length <= 500 ? value : `${text.slice(0, 500)}...`;
};

const oldNetron = async (file) => {
    const output = await exec(process.execPath, [path.join(root, 'tools', 'netron-normalize.mjs'), file], path.dirname(root));
    return JSON.parse(output);
};

const rustNetron = async (file, rustBin) => {
    const output = rustBin ?
        await exec(rustBin, ['parse', file], path.dirname(root)) :
        await exec('cargo', ['run', '--quiet', '--manifest-path', path.join(root, 'Cargo.toml'), '--', 'parse', file], path.dirname(root));
    return JSON.parse(output);
};

const exec = (command, args, cwd) => new Promise((resolve, reject) => {
    child_process.execFile(command, args, { cwd, maxBuffer: 256 * 1024 * 1024 }, (error, stdout, stderr) => {
        if (error) {
            error.message = `${error.message}\n${stderr}`;
            reject(error);
        } else {
            resolve(stdout);
        }
    });
});

const summarizeOld = (model, file) => {
    const options = {
        tflite: /^TensorFlow Lite\b/.test(model.format?.name || ''),
        coreml: /^Core ML\b/.test(model.format?.name || ''),
        darknet: /^Darknet\b/.test(model.format?.name || ''),
        lightgbm: /^LightGBM\b/.test(model.format?.name || ''),
        mlir: /^MLIR\b/.test(model.format?.name || ''),
        ncnn: /^(ncnn|PNNX)\b/.test(model.format?.name || ''),
        xgboost: /^XGBoost\b/.test(model.format?.name || ''),
        message: isMessageFile(file),
        sentencepiece: /^SentencePiece\b/.test(model.format?.name || ''),
        dot: /^DOT\b/.test(model.format?.name || ''),
        anonymousValues: /^(Core ML|Darknet|ncnn|PNNX|PyTorch|TorchScript|NumPy Array)\b/.test(model.format?.name || '')
    };
    return {
        format: normalizeFormat(model.format.name),
        format_version: normalizeFormatVersion(model.format.name),
        metadata: summarizeMetadata(model.metadata),
        graph_count: model.graphs.length,
        function_count: model.functions.length,
        tensor_count: model.tensors.length,
        graphs: model.graphs.map((graph) => summarizeOldGraph(graph, options)),
        functions: model.functions.map((func) => ({
            name: func.name,
            graph: summarizeOldFunctionGraph(func.graph, options)
        }))
    };
};

const summarizeRust = (model, file) => {
    const options = {
        message: isMessageFile(file)
    };
    const graphs = (model.graphs || []).filter((graph) => graph.parent === null || graph.parent === undefined);
    const functions = model.functions || [];
    const tensorIds = visibleRustTensorIds(graphs, functions);
    return {
        format: normalizeFormat(model.format.name),
        format_version: model.format.version || null,
        metadata: summarizeMetadata(model.metadata, { combineProducerVersion: true }),
        graph_count: graphs.length,
        function_count: functions.length,
        tensor_count: tensorIds.size,
        graphs: graphs.map((graph) => summarizeRustGraph(graph, options)),
        functions: functions.map((func) => ({
            name: func.name,
            graph: summarizeRustFunction(func, options)
        }))
    };
};

const isMessageFile = (file) => ['.message', '.maxviz'].includes(path.extname(file || '').toLowerCase());

const summarizeRustFunction = (func, options = {}) => {
    const values = new Map((func.values || []).map((value) => [value.name, {
        name: value.name,
        type: value.type ? {
            data_type: value.type.element_type || null,
            layout: value.type.layout || null,
            denotation: value.type.denotation || null,
            shape: (value.type.shape || []).map((dimension) => normalizeDimension(dimension.value ?? null))
        } : null,
        initializer: value.initializer !== null && value.initializer !== undefined
    }]));
    const ensureValue = (name) => {
        if (name && !values.has(name)) {
            values.set(name, { name, type: null, initializer: false });
        }
    };
    for (const value of [...(func.inputs || []), ...(func.outputs || [])]) {
        ensureValue(value);
    }
    for (const node of func.nodes || []) {
        for (const value of flattenNullable(node.inputs || [])) {
            ensureValue(value);
        }
        for (const value of flattenNullable(node.outputs || [])) {
            ensureValue(value);
        }
    }
    return {
        name: func.name,
        inputs: func.inputs || [],
        outputs: func.outputs || [],
        values: Array.from(values.values()).sort(byName),
        nodes: (func.nodes || []).map((node) => summarizeRustFunctionNode(node, options))
    };
};

const summarizeOldGraph = (graph, options) => {
    const values = new Map((graph.values || []).map((value) => [value.id, value]));
    const valueName = (id) => {
        const value = values.get(id);
        return value ? normalizeValueName(value.name, options) : null;
    };
    return {
        name: graph.name || null,
        description: graph.description || null,
        metadata: normalizeObject(graph.metadata),
        inputs: graph.inputs || [],
        outputs: graph.outputs || [],
        values: (graph.values || []).map((value) => ({
            name: normalizeValueName(value.name, options),
            description: value.description || null,
            metadata: normalizeObject(value.metadata),
            type: value.type ? {
                data_type: value.type.data_type || null,
                layout: value.type.layout || null,
                denotation: value.type.denotation || null,
                shape: (value.type.shape || []).map(normalizeDimension)
            } : null,
            producer: value.producer ?? null,
            consumers: value.consumers || [],
            initializer: value.initializer !== null && value.initializer !== undefined,
            quantization: normalizeQuantization(value.quantization),
            graph_input: !!value.graph_input,
            graph_output: !!value.graph_output
        })).sort(byName),
        nodes: (graph.nodes || []).map((node) => summarizeOldNode(node, valueName, options))
    };
};

const summarizeOldFunctionGraph = (graph, options) => {
    const values = new Map((graph.values || []).map((value) => [value.id, value]));
    const valueName = (id) => {
        const value = values.get(id);
        return value ? normalizeValueName(value.name, options) : null;
    };
    return {
        name: graph.name || null,
        inputs: graph.inputs || [],
        outputs: graph.outputs || [],
        values: (graph.values || []).map((value) => ({
            name: normalizeValueName(value.name, options),
            type: value.type ? {
                data_type: value.type.data_type || null,
                layout: value.type.layout || null,
                denotation: value.type.denotation || null,
                shape: (value.type.shape || []).map(normalizeDimension)
            } : null,
            initializer: value.initializer !== null && value.initializer !== undefined
        })).sort(byName),
        nodes: (graph.nodes || []).map((node) => summarizeOldFunctionNode(node, valueName, options))
    };
};

const summarizeRustGraph = (graph, options = {}) => {
    const visible = visibleRustValueNames(graph);
    return {
        name: graph.name || null,
        description: graph.description || null,
        metadata: normalizeObject(graph.metadata),
        inputs: graph.inputs || [],
        outputs: graph.outputs || [],
        values: (graph.values || []).filter((value) => visible.has(value.name)).map(summarizeRustValue).sort(byName),
        nodes: (graph.nodes || []).map((node) => summarizeRustNode(node, options))
    };
};

const summarizeRustValue = (value) => ({
    name: value.name,
    description: value.description || null,
    metadata: normalizeObject(value.metadata),
    type: value.type ? {
        data_type: value.type.element_type || null,
        layout: value.type.layout || null,
        denotation: value.type.denotation || null,
        shape: (value.type.shape || []).map((dimension) => normalizeDimension(dimension.value ?? null))
    } : null,
    producer: value.producer ?? null,
    consumers: value.consumers || [],
    initializer: value.initializer !== null && value.initializer !== undefined,
    quantization: normalizeQuantization(value.quantization),
    graph_input: !!value.graph_input,
    graph_output: !!value.graph_output
});

const summarizeOldNode = (node, valueName, options = {}) => ({
    name: node.name || null,
    description: null,
    metadata: normalizeObject(node.metadata),
    operator: summarizeOperator(node.operator),
    inputs: flattenIds(node.inputs || [], valueName),
    outputs: flattenIds(node.outputs || [], valueName),
    attributes: summarizeOldAttributes(node.attributes || [], options)
});

const summarizeRustNode = (node, options = {}) => ({
    name: node.name || null,
    description: null,
    metadata: normalizeObject(node.metadata),
    operator: summarizeOperator(node.operator),
    inputs: flattenNullable(node.inputs || []),
    outputs: flattenNullable(node.outputs || []),
    attributes: summarizeRustAttributes(node.attributes || [], options)
});

const summarizeOldFunctionNode = (node, valueName, options = {}) => ({
    ...summarizeOldNode(node, valueName, options),
    attributes: summarizeOldAttributes(node.attributes || [], options).filter(isComparableFunctionAttribute)
});

const summarizeRustFunctionNode = (node, options = {}) => ({
    ...summarizeRustNode(node, options),
    attributes: summarizeRustAttributes(node.attributes || [], options).filter(isComparableFunctionAttribute)
});

const visibleRustValueNames = (graph) => {
    const names = new Set([...(graph.inputs || []), ...(graph.outputs || [])]);
    for (const node of graph.nodes || []) {
        for (const value of flattenNullable(node.inputs || [])) {
            if (value !== null && value !== undefined) {
                names.add(value);
            }
        }
        for (const value of flattenNullable(node.outputs || [])) {
            if (value !== null && value !== undefined) {
                names.add(value);
            }
        }
    }
    return names;
};

const visibleRustTensorIds = (graphs, functions) => {
    const ids = new Set();
    for (const graph of graphs) {
        const visible = visibleRustValueNames(graph);
        for (const value of graph.values || []) {
            if (visible.has(value.name) && value.initializer !== null && value.initializer !== undefined) {
                ids.add(value.initializer);
            }
        }
        for (const node of graph.nodes || []) {
            collectAttributeTensorIds(node.attributes || [], ids);
        }
    }
    for (const func of functions) {
        for (const value of func.values || []) {
            if (value.initializer !== null && value.initializer !== undefined) {
                ids.add(value.initializer);
            }
        }
        for (const node of func.nodes || []) {
            collectAttributeTensorIds(node.attributes || [], ids);
        }
    }
    return ids;
};

const collectAttributeTensorIds = (attributes, ids) => {
    for (const attribute of attributes || []) {
        const value = attribute.value || {};
        if (value.kind === 'tensor' && value.value !== null && value.value !== undefined) {
            ids.add(value.value);
        } else if (value.kind === 'tensors' && Array.isArray(value.value)) {
            for (const id of value.value) {
                ids.add(id);
            }
        }
    }
};

const diff = (oldSummary, rustSummary) => {
    canonicalizeLegacyAttributePairs(oldSummary, rustSummary);
    const differences = [];
    compare(differences, 'format', oldSummary.format, rustSummary.format);
    compare(differences, 'format_version', oldSummary.format_version, rustSummary.format_version);
    compare(differences, 'metadata', oldSummary.metadata, rustSummary.metadata);
    compare(differences, 'graph_count', oldSummary.graph_count, rustSummary.graph_count);
    compare(differences, 'function_count', oldSummary.function_count, rustSummary.function_count);
    compare(differences, 'tensor_count', oldSummary.tensor_count, rustSummary.tensor_count);
    compare(differences, 'graphs', oldSummary.graphs, rustSummary.graphs);
    compare(differences, 'functions', oldSummary.functions, rustSummary.functions);
    return differences;
};

const legacyScalarAttributeNames = new Set(['axis', 'broadcast', 'group']);

const canonicalizeLegacyAttributePairs = (oldSummary, rustSummary) => {
    for (const [oldGraph, rustGraph] of paired(oldSummary.graphs, rustSummary.graphs)) {
        canonicalizeLegacyNodeAttributes(oldGraph?.nodes || [], rustGraph?.nodes || []);
    }
    for (const [oldFunction, rustFunction] of paired(oldSummary.functions, rustSummary.functions)) {
        canonicalizeLegacyNodeAttributes(oldFunction?.graph?.nodes || [], rustFunction?.graph?.nodes || []);
    }
};

const canonicalizeLegacyNodeAttributes = (oldNodes, rustNodes) => {
    for (const [oldNode, rustNode] of paired(oldNodes, rustNodes)) {
        const oldAttributes = new Map((oldNode?.attributes || []).map((attribute) => [attribute.name, attribute]));
        for (const rustAttribute of rustNode?.attributes || []) {
            const oldAttribute = oldAttributes.get(rustAttribute.name);
            if (!oldAttribute || !legacyScalarAttributeNames.has(rustAttribute.name)) {
                continue;
            }
            if (oldAttribute.kind === 'float' && oldAttribute.value === '0' && rustAttribute.kind === 'int') {
                oldAttribute.kind = 'legacy_scalar';
                oldAttribute.value = null;
                rustAttribute.kind = 'legacy_scalar';
                rustAttribute.value = null;
            }
        }
    }
};

const paired = (left, right) => {
    const length = Math.max(left?.length || 0, right?.length || 0);
    return Array.from({ length }, (_, index) => [left?.[index], right?.[index]]);
};

const compare = (differences, path, oldValue, rustValue) => {
    if (JSON.stringify(oldValue) !== JSON.stringify(rustValue)) {
        differences.push({ path, old: oldValue, rust: rustValue });
    }
};

const flattenIds = (items, valueName) => items.flatMap((item) => Array.isArray(item) ? item.map(valueName) : []);

const flattenNullable = (items) => items.flatMap((item) => Array.isArray(item) ? item : [item]);

const normalizeValueName = (name, options = {}) => options.anonymousValues && (name === null || name === undefined) ? '' : name;

const normalizeFormat = (format) => (format || '').split(/\s+/)[0];

const normalizeFormatVersion = (format) => {
    const match = String(format || '').match(/\bv(\d+(?:\.\d+)*)\b/i);
    return match ? match[1] : null;
};

const normalizeDimension = (dimension) => dimension === null || dimension === undefined ? null : String(dimension);

const byName = (a, b) => String(a.name).localeCompare(String(b.name));

const byAttribute = (a, b) => String(a.name).localeCompare(String(b.name)) || String(a.kind).localeCompare(String(b.kind));

const byKey = (a, b) => String(a.key).localeCompare(String(b.key));

const summarizeMetadata = (metadata, options = {}) => ({
    producer: normalizeProducer(metadata, options),
    domain: metadata?.domain || null,
    description: metadata?.description || null,
    properties: normalizeObject(metadata?.properties)
});

const normalizeProducer = (metadata, options) => {
    const producer = metadata?.producer || null;
    const version = metadata?.producer_version || null;
    if (options.combineProducerVersion && producer && version) {
        return producerFamily(`${producer} ${version}`);
    }
    return producerFamily(producer);
};

const producerFamily = (producer) => producer ? String(producer).split(/\s+/)[0] : null;

const normalizeObject = (value) => Object.fromEntries(Object.entries(value || {})
    .map(([key, entry]) => [key, normalizeObjectValue(entry)])
    .sort(([a], [b]) => a.localeCompare(b)));

const normalizeObjectValue = (value) => {
    if (value === null || value === undefined) {
        return null;
    }
    if (typeof value === 'string') {
        const trimmed = value.trim();
        if ((trimmed.startsWith('[') && trimmed.endsWith(']')) || (trimmed.startsWith('{') && trimmed.endsWith('}'))) {
            try {
                return normalizeObjectValue(JSON.parse(trimmed));
            } catch {
                // Fall through and compare it as a scalar string.
            }
        }
        return scalarMetadataText(value);
    }
    if (typeof value === 'number') {
        return scalarMetadataText(value);
    }
    if (typeof value === 'bigint' || typeof value === 'boolean') {
        return String(value);
    }
    if (Array.isArray(value)) {
        return value.map(normalizeObjectValue);
    }
    if (typeof value === 'object') {
        return normalizeObject(value);
    }
    return String(value);
};

const scalarMetadataText = (value) => {
    const text = String(value);
    if (text.trim() === '') {
        return text;
    }
    const number = Number(text);
    return Number.isFinite(number) ? Number(number.toPrecision(7)).toString() : text;
};

const summarizeOperator = (operator) => ({
    domain: normalizeOperatorDomain(operator?.domain),
    name: operator?.name || null,
    overload: operator?.overload || null
});

const normalizeOperatorDomain = (domain) => {
    if (domain === '' || domain === 'ai.onnx') {
        return null;
    }
    return domain || null;
};

const summarizeOldAttributes = (attributes, options = {}) => attributes
    .filter((attribute) => !options.tflite || attribute.visible !== false)
    .map((attribute) => {
    const kind = normalizeOldAttributeKind(attribute, options);
    if (Array.isArray(attribute.ids)) {
        return {
            name: attribute.name || null,
            kind,
            count: attribute.ids.length
        };
    }
    if (attribute.ref !== undefined) {
        if (options.mlir && normalizeAttributeKind(attribute.kind) === 'reference') {
            return {
                name: attribute.name || null,
                kind: 'reference',
                value: attribute.ref === null || attribute.ref === undefined ? null : String(attribute.ref)
            };
        }
        return {
            name: attribute.name || null,
            kind,
            count: attribute.ref === null || attribute.ref === undefined ? 0 : 1
        };
    }
    if (kind === 'bytes') {
        return {
            name: attribute.name || null,
            kind: 'bytes'
        };
    }
    return {
        name: attribute.name || null,
        kind,
        value: normalizeOldAttributePayload(attribute, kind, options)
    };
}).sort(byAttribute);

const normalizeOldAttributeKind = (attribute, options = {}) => {
    const value = attribute.value;
    const normalizedKind = normalizeAttributeKind(attribute.kind);
    if (options.coreml && Array.isArray(value) && normalizedKind === 'int' && value.every(isIntegerLike)) {
        return 'ints';
    }
    if (options.darknet && normalizedKind === 'int' && !isIntegerLike(value)) {
        return 'string';
    }
    if (options.darknet && normalizedKind === 'float' && Number.isNaN(Number(value))) {
        return 'string';
    }
    if (options.darknet && normalizedKind === 'ints' && !Array.isArray(value)) {
        return 'string';
    }
    if (options.lightgbm && attribute.kind === 'object[]') {
        return 'strings';
    }
    if (options.xgboost && attribute.kind === 'object') {
        return 'string';
    }
    if (options.xgboost && attribute.kind === 'object[]') {
        return 'strings';
    }
    if (options.message) {
        if (attribute.kind === 'boolean') {
            return 'int';
        }
        if (['SymInt', 'SymInt?', 'Scalar'].includes(attribute.kind)) {
            return 'string';
        }
        if ((attribute.kind === 'Tensor?' || attribute.kind === 'int64?') && (value === null || isEmptyArray(value))) {
            return 'null';
        }
    }
    if (!(options.tflite || options.coreml || options.darknet || options.lightgbm || options.mlir || options.ncnn || options.xgboost || options.message || options.sentencepiece || options.dot) || attribute.kind !== 'attribute') {
        return normalizedKind;
    }
    if (options.sentencepiece && Array.isArray(value) && value.length === 0) {
        if (['input', 'accept_language', 'control_symbols', 'user_defined_symbols', 'pieces', 'samples'].includes(attribute.name)) {
            return 'strings';
        }
        if (attribute.name === 'precompiled_charsmap') {
            return 'ints';
        }
    }
    if (typeof value === 'boolean') {
        return 'boolean';
    }
    if (Number.isInteger(value)) {
        return 'int';
    }
    if (typeof value === 'number') {
        return 'float';
    }
    if (typeof value === 'string') {
        return 'string';
    }
    if (options.coreml && value === null) {
        return 'float';
    }
    if (Array.isArray(value)) {
        if (value.every((item) => Number.isInteger(item))) {
            return 'ints';
        }
        if (value.every((item) => typeof item === 'number')) {
            return 'floats';
        }
        return 'strings';
    }
    return normalizeAttributeKind(attribute.kind);
};

const normalizeOldAttributePayload = (attribute, kind, options = {}) => {
    if (options.message && isEmptyArray(attribute.value)) {
        if ((attribute.kind === 'boolean' || attribute.kind === 'int64') && kind === 'int') {
            return '0';
        }
        if (['SymInt', 'SymInt?', 'Scalar'].includes(attribute.kind) && kind === 'string') {
            return '0';
        }
        if ((attribute.kind === 'Tensor?' || attribute.kind === 'int64?') && kind === 'null') {
            return null;
        }
    }
    return normalizeAttributePayloadForOptions(kind, attribute.value, options);
};

const summarizeRustAttributes = (attributes, options = {}) => attributes.map((attribute) => {
    const value = attribute.value || {};
    const kind = normalizeAttributeKind(value.kind);
    if (kind === 'tensor' || kind === 'graph') {
        return {
            name: attribute.name || null,
            kind,
            count: value.value === null || value.value === undefined ? 0 : 1
        };
    }
    if (kind === 'tensors' || kind === 'graphs') {
        return {
            name: attribute.name || null,
            kind,
            count: Array.isArray(value.value) ? value.value.length : 0
        };
    }
    if (kind === 'bytes') {
        return {
            name: attribute.name || null,
            kind
        };
    }
    return {
        name: attribute.name || null,
        kind,
        value: normalizeAttributePayloadForOptions(kind, value.value, options)
    };
}).sort(byAttribute);

const isComparableFunctionAttribute = (attribute) => {
    if (attribute.kind === 'graph' || attribute.kind === 'graphs') {
        return false;
    }
    if (attribute.kind === 'unsupported' && typeof attribute.value === 'string' && attribute.value.includes('graph attribute')) {
        return false;
    }
    return true;
};

const normalizeAttributeKind = (kind) => {
    switch (kind) {
        case 'bool':
        case 'boolean':
            return 'boolean';
        case 'ActivationFunctionType':
        case 'CombinerType':
        case 'FullyConnectedOptionsWeightsFormat':
        case 'FlattenLayerParams.FlattenOrder':
        case 'LSHProjectionType':
        case 'LSTMKernelType':
        case 'MirrorPadMode':
        case 'Padding':
        case 'BinaryOpType':
        case 'CastOpType':
        case 'EltwiseType':
        case 'PaddingType':
        case 'PoolingLayerParams.PoolingType':
        case 'PoolingType':
        case 'InterpResizeType':
        case 'PermuteOrderType':
        case 'ReductionOpType':
        case 'UnaryOpType':
        case 'ReduceWindowFunction':
        case 'SamePadding':
        case 'StablehloComparisonType':
        case 'TensorType':
        case 'UnaryFunctionLayerParams.Operation':
        case 'ValidCompletePadding':
        case 'ValidPadding':
            return 'string';
        case 'byte[]':
        case 'bytes':
            return 'bytes';
        case 'float32':
        case 'float64':
        case 'float':
            return 'float';
        case 'float32[]':
        case 'float64[]':
        case 'floats':
            return 'floats';
        case 'uint8':
        case 'int8':
        case 'uint16':
        case 'int16':
        case 'uint32':
        case 'int32':
        case 'int64':
        case 'uint64':
        case 'int':
            return 'int';
        case 'uint8[]':
        case 'int8[]':
        case 'uint16[]':
        case 'int16[]':
        case 'uint32[]':
        case 'int32[]':
        case 'int64[]':
        case 'uint64[]':
        case 'ints':
            return 'ints';
        case 'string[]':
        case 'strings':
            return 'strings';
        case 'tensor[]':
        case 'tensors':
            return 'tensors';
        case 'graph[]':
        case 'graphs':
            return 'graphs';
        case 'DataType':
        case 'function':
        case 'type':
            return kind === 'function' ? 'reference' : 'type';
        default:
            return kind || null;
    }
};

const normalizeAttributePayload = (kind, value) => {
    if (value === null || value === undefined) {
        return null;
    }
    if (kind === 'int') {
        return scalarText(value);
    }
    if (kind === 'float') {
        return floatText(value);
    }
    if (kind === 'ints') {
        return Array.isArray(value) ? value.slice(0, 16).map(scalarText) : [];
    }
    if (kind === 'floats') {
        return Array.isArray(value) ? value.slice(0, 16).map(floatText) : [];
    }
    if (Array.isArray(value)) {
        return value.slice(0, 16).map((item) => item === null || item === undefined ? null : String(item));
    }
    if (typeof value === 'object') {
        return normalizeObject(value);
    }
    return String(value);
};

const normalizeAttributePayloadForOptions = (kind, value, options = {}) => {
    const payload = normalizeAttributePayload(kind, value);
    return options.message ? normalizeUnsafeIntegerPayload(payload) : payload;
};

const normalizeUnsafeIntegerPayload = (value) => {
    if (Array.isArray(value)) {
        return value.map(normalizeUnsafeIntegerPayload);
    }
    if (typeof value === 'string' && /^-?\d+$/.test(value)) {
        const number = Number(value);
        if (Number.isFinite(number) && Math.abs(number) > Number.MAX_SAFE_INTEGER) {
            return String(number);
        }
    }
    return value;
};

const isEmptyArray = (value) => Array.isArray(value) && value.length === 0;

const isIntegerLike = (value) => {
    if (Number.isInteger(value)) {
        return true;
    }
    if (typeof value === 'bigint') {
        return true;
    }
    return typeof value === 'string' && /^-?\d+$/.test(value);
};

const scalarText = (value) => {
    if (value === null || value === undefined) {
        return null;
    }
    return String(value);
};

const floatText = (value) => {
    if (value === null || value === undefined) {
        return null;
    }
    const number = Number(value);
    return Number.isFinite(number) ? Number(number.toPrecision(4)).toString() : String(value);
};

const normalizeQuantization = (value) => {
    if (!value) {
        return [];
    }
    if (Array.isArray(value)) {
        return value.flatMap((entry) => {
            const normalized = normalizeQuantizationValue(entry.value);
            return normalized === null ? [] : [{
                key: String(entry.key),
                value: normalized
            }];
        }).sort(byKey);
    }
    if (typeof value === 'object') {
        return Object.entries(value).flatMap(([key, entry]) => {
            const normalized = normalizeQuantizationValue(entry);
            return normalized === null ? [] : [{
                key: String(key),
                value: normalized
            }];
        }).sort(byKey);
    }
    return [{ key: 'value', value: String(value) }];
};

const normalizeQuantizationValue = (value) => {
    if (value === null || value === undefined) {
        return null;
    }
    if (Array.isArray(value)) {
        return value.length === 0 ? null : value.slice(0, 16).map(normalizeQuantizationScalar).join(',');
    }
    if (typeof value === 'object') {
        const entries = Object.entries(value);
        if (entries.length === 0) {
            return null;
        }
        if (entries.every(([key]) => /^-?\d+$/.test(key))) {
            return entries
                .sort(([a], [b]) => Number(a) - Number(b))
                .slice(0, 16)
                .map(([, entry]) => normalizeQuantizationScalar(entry))
                .join(',');
        }
        return JSON.stringify(normalizeObject(value));
    }
    return normalizeQuantizationScalar(value);
};

const normalizeQuantizationScalar = (value) => {
    if (typeof value === 'string' && value.includes(',')) {
        const parts = value.split(',');
        if (parts.every((part) => part !== '' && Number.isFinite(Number(part)))) {
            return parts.slice(0, 16).map((part) => normalizeQuantizationScalar(part)).join(',');
        }
    }
    const number = Number(value);
    return value !== '' && Number.isFinite(number) ?
        Number(number.toPrecision(3)).toString() :
        String(value);
};

main().catch((error) => {
    console.error(`${error.name}: ${error.message}`);
    process.exit(1);
});
