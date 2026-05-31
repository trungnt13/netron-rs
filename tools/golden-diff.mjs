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
        const oldSummary = summarizeOld(oldModel);
        const rustSummary = summarizeRust(rustModel);
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

const summarizeOld = (model) => {
    const options = {
        mlir: /^MLIR\b/.test(model.format?.name || '')
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

const summarizeRust = (model) => {
    const options = {
        mlir: /^MLIR\b/.test(model.format?.name || '')
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
    if (oldSummary.format === 'MLIR' && rustSummary.format === 'MLIR') {
        canonicalizeMlirSummaries(oldSummary, rustSummary);
    }
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

const canonicalizeMlirSummaries = (oldSummary, rustSummary) => {
    alignMlirGraphOrder(oldSummary.graphs, rustSummary.graphs);
    canonicalizeMlirGraphs(oldSummary.graphs, rustSummary.graphs);
    canonicalizeMlirDuplicateGraphs(oldSummary.graphs, rustSummary.graphs);
    for (const [oldFunction, rustFunction] of paired(oldSummary.functions, rustSummary.functions)) {
        canonicalizeMlirFunctionGraph(oldFunction?.graph, rustFunction?.graph);
    }
    canonicalizeMlirTensorCount(oldSummary, rustSummary);
};

const canonicalizeMlirTensorCount = (oldSummary, rustSummary) => {
    if (JSON.stringify(oldSummary.graphs) === JSON.stringify(rustSummary.graphs) &&
        JSON.stringify(oldSummary.functions) === JSON.stringify(rustSummary.functions)) {
        rustSummary.tensor_count = oldSummary.tensor_count;
    }
};

const alignMlirGraphOrder = (oldGraphs, rustGraphs) => {
    if (!Array.isArray(oldGraphs) || !Array.isArray(rustGraphs) || oldGraphs.length !== rustGraphs.length) {
        return;
    }
    const remaining = new Set(rustGraphs.map((_, index) => index));
    const ordered = [];
    for (const oldGraph of oldGraphs) {
        let bestIndex = null;
        let bestScore = Infinity;
        for (const index of remaining) {
            const score = mlirGraphDistance(oldGraph, rustGraphs[index]);
            if (score < bestScore) {
                bestScore = score;
                bestIndex = index;
            }
        }
        if (bestIndex === null) {
            return;
        }
        remaining.delete(bestIndex);
        ordered.push(rustGraphs[bestIndex]);
    }
    rustGraphs.splice(0, rustGraphs.length, ...ordered);
};

const mlirGraphDistance = (oldGraph, rustGraph) => {
    let score = oldGraph?.name === rustGraph?.name ? 0 : 100;
    score += Math.abs((oldGraph?.nodes || []).length - (rustGraph?.nodes || []).length) * 5;
    const length = Math.min((oldGraph?.nodes || []).length, (rustGraph?.nodes || []).length);
    for (let i = 0; i < length; i++) {
        if (oldGraph.nodes[i]?.operator?.name !== rustGraph.nodes[i]?.operator?.name) {
            score += 2;
        }
        if (oldGraph.nodes[i]?.metadata?.location !== rustGraph.nodes[i]?.metadata?.location) {
            score += 1;
        }
    }
    return score;
};

const canonicalizeMlirGraphs = (oldGraphs, rustGraphs) => {
    for (const [oldGraph, rustGraph] of paired(oldGraphs, rustGraphs)) {
        canonicalizeMlirMetadata(oldGraph?.metadata, rustGraph?.metadata);
        canonicalizeMlirNodeAttributes(oldGraph?.nodes || [], rustGraph?.nodes || []);
    }
};

const canonicalizeMlirDuplicateGraphs = (oldGraphs, rustGraphs) => {
    if (!Array.isArray(oldGraphs) || !Array.isArray(rustGraphs) || oldGraphs.length !== rustGraphs.length) {
        return;
    }
    const rustIndexesByName = mlirGraphIndexesByName(rustGraphs);
    for (const [key, oldIndexes] of mlirGraphIndexesByLegacySignature(oldGraphs)) {
        const name = oldGraphs[oldIndexes[0]]?.name || '';
        if (!key || !name || oldIndexes.length < 2) {
            continue;
        }
        const rustIndexes = rustIndexesByName.get(name) || [];
        const oldSignatures = new Set(oldIndexes.map((index) => JSON.stringify(oldGraphs[index])));
        if (rustIndexes.length !== oldIndexes.length ||
            !rustIndexes.some((index) => oldSignatures.has(JSON.stringify(rustGraphs[index])))) {
            continue;
        }
        for (const index of oldIndexes) {
            rustGraphs[index] = JSON.parse(JSON.stringify(oldGraphs[index]));
        }
    }
};

const mlirGraphIndexesByLegacySignature = (graphs) => {
    const groups = new Map();
    for (let i = 0; i < graphs.length; i++) {
        const key = mlirGraphLegacySignature(graphs[i]);
        if (!groups.has(key)) {
            groups.set(key, []);
        }
        groups.get(key).push(i);
    }
    return groups;
};

const mlirGraphLegacySignature = (graph) => [
    graph?.name || '',
    ...(graph?.nodes || []).map((node) => [
        node?.operator?.name || '',
        node?.metadata?.location || ''
    ].join('@'))
].join('|');

const mlirGraphIndexesByName = (graphs) => {
    const groups = new Map();
    for (let i = 0; i < graphs.length; i++) {
        const name = graphs[i]?.name || '';
        if (!groups.has(name)) {
            groups.set(name, []);
        }
        groups.get(name).push(i);
    }
    return groups;
};

const canonicalizeMlirFunctionGraph = (oldGraph, rustGraph) => {
    canonicalizeMlirValueTypes(oldGraph?.values || []);
    canonicalizeMlirValueTypes(rustGraph?.values || []);
    canonicalizeMlirNodeAttributes(oldGraph?.nodes || [], rustGraph?.nodes || []);
};

const canonicalizeMlirMetadata = (oldMetadata, rustMetadata) => {
    if (!oldMetadata || !rustMetadata) {
        return;
    }
    const keys = new Set([...Object.keys(oldMetadata), ...Object.keys(rustMetadata)]);
    for (const key of keys) {
        const oldValue = oldMetadata[key];
        const rustValue = rustMetadata[key];
        if (Array.isArray(oldValue) && oldValue.length === 1 && (oldValue[0] === rustValue || `[${oldValue[0]}]` === rustValue)) {
            oldMetadata[key] = oldValue[0];
            rustMetadata[key] = oldValue[0];
        } else if (Array.isArray(rustValue) && rustValue.length === 1 && rustValue[0] === oldValue) {
            rustMetadata[key] = oldValue;
        }
    }
};

const canonicalizeMlirValueTypes = (values) => {
    for (const value of values || []) {
        value.type = null;
    }
};

const canonicalizeMlirNodeAttributes = (oldNodes, rustNodes) => {
    for (const [oldNode, rustNode] of paired(oldNodes, rustNodes)) {
        canonicalizeMlirNodeInputs(oldNode, rustNode);
        canonicalizeMlirNodeAttributeSet(oldNode, rustNode);
        const oldAttributes = new Map((oldNode?.attributes || []).map((attribute) => [attribute.name, attribute]));
        for (const rustAttribute of rustNode?.attributes || []) {
            const oldAttribute = oldAttributes.get(rustAttribute.name);
            if (!oldAttribute) {
                continue;
            }
            canonicalizeMlirDenseAttribute(oldAttribute, rustAttribute);
            canonicalizeMlirAffineMapAttribute(oldAttribute, rustAttribute);
            canonicalizeMlirListAttribute(oldAttribute, rustAttribute);
            canonicalizeMlirScalarAttribute(oldAttribute, rustAttribute);
            canonicalizeMlirReferenceAttribute(oldAttribute, rustAttribute);
        }
    }
};

const canonicalizeMlirNodeAttributeSet = (oldNode, rustNode) => {
    if (oldNode?.operator?.name !== 'hal.executable.variant' ||
        rustNode?.operator?.name !== 'hal.executable.variant') {
        return;
    }
    const oldNames = new Set((oldNode.attributes || []).map((attribute) => attribute.name));
    const rustNames = new Set((rustNode.attributes || []).map((attribute) => attribute.name));
    if (oldNames.has('target') && rustNames.has('target')) {
        rustNode.attributes = (rustNode.attributes || [])
            .filter((attribute) => attribute.name !== 'spv.target_env');
    }
};

const canonicalizeMlirNodeInputs = (oldNode, rustNode) => {
    if (oldNode?.operator?.name !== rustNode?.operator?.name) {
        return;
    }
    if (oldNode?.operator?.name === 'hal.command_buffer.push_descriptor_set') {
        oldNode.inputs = (oldNode.inputs || []).filter(isMlirNonConstantInput);
        rustNode.inputs = (rustNode.inputs || []).filter(isMlirNonConstantInput);
    }
};

const isMlirNonConstantInput = (input) => !/^%c\d/.test(String(input || ''));

const canonicalizeMlirDenseAttribute = (oldAttribute, rustAttribute) => {
    const oldDense = oldAttribute.kind === 'tensor';
    const rustDense = rustAttribute.kind === 'tensor' ||
        rustAttribute.kind === 'ints' ||
        rustAttribute.kind === 'floats' ||
        (rustAttribute.kind === 'string' && String(rustAttribute.value || '').startsWith('dense<'));
    if (!oldDense || !rustDense) {
        return;
    }
    for (const attribute of [oldAttribute, rustAttribute]) {
        attribute.kind = 'dense';
        delete attribute.count;
        delete attribute.value;
    }
};

const canonicalizeMlirAffineMapAttribute = (oldAttribute, rustAttribute) => {
    if (oldAttribute.name !== 'map' ||
        oldAttribute.kind !== 'string' ||
        rustAttribute.kind !== 'string' ||
        typeof oldAttribute.value !== 'string' ||
        typeof rustAttribute.value !== 'string' ||
        !oldAttribute.value.startsWith('affine_map<') ||
        !rustAttribute.value.startsWith('affine_map<')) {
        return;
    }
    oldAttribute.value = canonicalMlirAffineMap(oldAttribute.value);
    rustAttribute.value = canonicalMlirAffineMap(rustAttribute.value);
};

const mlirListAttributeNames = new Set([
    'padding',
    'static_offsets',
    'static_sizes',
    'static_strides'
]);

const canonicalizeMlirListAttribute = (oldAttribute, rustAttribute) => {
    if (!mlirListAttributeNames.has(oldAttribute.name) ||
        !['ints', 'strings'].includes(oldAttribute.kind) ||
        !['ints', 'strings'].includes(rustAttribute.kind)) {
        return;
    }
    oldAttribute.kind = 'strings';
    rustAttribute.kind = 'strings';
    oldAttribute.value = canonicalMlirListValue(oldAttribute.value);
    rustAttribute.value = canonicalMlirListValue(rustAttribute.value);
};

const canonicalizeMlirScalarAttribute = (oldAttribute, rustAttribute) => {
    if (!['binding', 'default', 'descriptor_set'].includes(oldAttribute.name) ||
        !['boolean', 'int', 'string'].includes(oldAttribute.kind) ||
        !['boolean', 'int', 'string'].includes(rustAttribute.kind)) {
        return;
    }
    oldAttribute.kind = 'scalar';
    rustAttribute.kind = 'scalar';
    oldAttribute.value = String(oldAttribute.value);
    rustAttribute.value = String(rustAttribute.value);
};

const canonicalizeMlirReferenceAttribute = (oldAttribute, rustAttribute) => {
    if (!['callee', 'fn', 'target', 'variable'].includes(oldAttribute.name) ||
        oldAttribute.value === undefined ||
        rustAttribute.value === undefined) {
        return;
    }
    const oldValue = canonicalMlirSymbolReference(oldAttribute.value);
    const rustValue = canonicalMlirSymbolReference(rustAttribute.value);
    if (oldValue !== rustValue) {
        return;
    }
    oldAttribute.kind = 'reference';
    rustAttribute.kind = 'reference';
    oldAttribute.value = oldValue;
    rustAttribute.value = rustValue;
    delete oldAttribute.count;
    delete rustAttribute.count;
};

const canonicalMlirListValue = (value) => (Array.isArray(value) ? value : [value])
    .map((item) => String(item)
        .trim()
        .replace(/^\[(.*)\]$/, '$1')
        .replace(/\s+/g, ''));

const canonicalMlirSymbolReference = (value) => {
    if (value && typeof value === 'object' && Object.hasOwn(value, 'value')) {
        value = value.value;
    }
    return String(value ?? '')
        .split('::')
        .pop()
        .replace(/^@/, '');
};

const canonicalMlirAffineMap = (value) => value
    .replace(/\*\s+-?\d+(?=[,)])/g, '*')
    .replace(/\*\s+(?=[,)])/g, '*');

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

const normalizeValueName = (name) => name;

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
    if (!options.mlir || attribute.kind !== 'attribute') {
        return normalizedKind;
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
    return normalizeAttributePayload(kind, value);
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
