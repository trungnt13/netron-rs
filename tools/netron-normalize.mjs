#!/usr/bin/env node

import * as fs from 'fs/promises';
import * as path from 'path';

import * as mock from '../../netron/test/mock.js';
import * as node from '../../netron/source/node.js';
import * as view from '../../netron/source/view.js';

const main = async () => {
    const [file] = process.argv.slice(2);
    if (!file) {
        throw new Error('usage: node tools/netron-normalize.mjs <model>');
    }

    const target = path.resolve(file);
    const host = new mock.Host({ zoom: 'none', serial: true });
    const service = new view.ModelFactoryService(host);
    if (typeof service.import === 'function') {
        await service.import();
    }
    const context = await openContext(host, target);
    const model = await service.open(context);
    console.log(JSON.stringify(normalizeModel(model, host.errors), jsonReplacer, 2));
};

const openContext = async (host, target) => {
    const stat = await fs.stat(target);
    const basename = path.basename(target);
    if (stat.isFile()) {
        const stream = new node.FileStream(target, 0, stat.size, stat.mtimeMs);
        return new mock.Context(host, path.dirname(target), basename, stream, new Map());
    }
    if (stat.isDirectory()) {
        const entries = new Map();
        const walk = async (dir) => {
            const names = await fs.readdir(dir);
            names.sort();
            for (const name of names) {
                const pathname = path.join(dir, name);
                const stat = await fs.stat(pathname);
                if (stat.isDirectory()) {
                    await walk(pathname);
                } else if (stat.isFile()) {
                    const stream = new node.FileStream(pathname, 0, stat.size, stat.mtimeMs);
                    const key = pathname.split(path.sep).join(path.posix.sep);
                    entries.set(key, stream);
                }
            }
        };
        await walk(target);
        return new mock.Context(host, target, basename, null, entries);
    }
    throw new Error(`Unsupported path target: ${target}`);
};

const jsonReplacer = (_key, value) => typeof value === 'bigint' ? value.toString() : value;

const normalizeModel = (model, errors) => {
    const tensors = new TensorTable();
    const graphs = (model.modules || []).map((graph, index) => normalizeGraph(graph, tensors, index));
    const functions = (model.functions || []).map((func, index) => ({
        id: index,
        name: func.name || func.identifier || null,
        graph: normalizeGraph(func, tensors, index, null)
    }));
    return {
        format: {
            name: typeof model.format === 'string' ? model.format : model.format?.name || null
        },
        metadata: {
            producer: model.producer || null,
            producer_version: model.version || null,
            domain: model.domain || null,
            description: model.description || null,
            properties: toObject(model.metadata)
        },
        graphs,
        functions,
        tensors: tensors.toJSON(),
        errors: errors.map((error) => ({ name: error.name, message: error.message }))
    };
};

class TensorTable {

    constructor() {
        this._ids = new Map();
    }

    id(tensor) {
        if (!tensor || typeof tensor !== 'object') {
            return null;
        }
        if (!this._ids.has(tensor)) {
            this._ids.set(tensor, this._ids.size);
        }
        return this._ids.get(tensor);
    }

    toJSON() {
        return Array.from(this._ids.keys()).map((tensor) => ({
            id: this.id(tensor),
            name: tensor.name || null,
            description: tensor.description || null,
            metadata: toObject(tensor.metadata),
            type: normalizeType(tensor.type),
            location: tensor.location || null,
            encoding: tensor.encoding || null,
            layout: tensor.layout || null,
            quantized: !!tensor.quantization,
            loaded: typeof tensor.peek === 'function' ? tensor.peek() : null
        }));
    }
}

const normalizeGraph = (graph, tensors, graphId, parentId = null) => {
    const values = new ValueTable(tensors);
    const nodes = [];
    const inputs = new Set();
    const outputs = new Set();

    for (const node of graph.nodes || []) {
        const id = nodes.length;
        const inputArguments = node.inputs || [];
        const outputArguments = node.outputs || [];

        for (const argument of inputArguments) {
            for (const value of argumentValues(argument)) {
                const record = values.get(value);
                if (record && !record.consumers.includes(id)) {
                    record.consumers.push(id);
                }
            }
        }
        for (const argument of outputArguments) {
            for (const value of argumentValues(argument)) {
                const record = values.get(value);
                if (record) {
                    record.producer = id;
                }
            }
        }

        nodes.push({
            id,
            name: node.name || null,
            description: node.description || null,
            metadata: toObject(node.metadata),
            operator: normalizeOperator(node.type),
            attributes: (node.attributes || []).map((attribute) => normalizeAttribute(attribute, tensors)).filter(Boolean),
            inputs: inputArguments.map((argument) => argumentValues(argument).map((value) => values.get(value)?.id ?? null)),
            outputs: outputArguments.map((argument) => argumentValues(argument).map((value) => values.get(value)?.id ?? null))
        });
    }

    for (const signature of signatures(graph)) {
        for (const argument of signature.inputs || []) {
            for (const value of argumentValues(argument)) {
                const record = values.get(value);
                if (record) {
                    record.graph_input = true;
                    if (record.name) {
                        inputs.add(record.name);
                    }
                }
            }
        }
        for (const argument of signature.outputs || []) {
            for (const value of argumentValues(argument)) {
                const record = values.get(value);
                if (record) {
                    record.graph_output = true;
                    if (record.name) {
                        outputs.add(record.name);
                    }
                }
            }
        }
    }

    return {
        id: graphId,
        parent: parentId,
        name: graph.name || null,
        description: graph.description || null,
        metadata: toObject(graph.metadata),
        inputs: Array.from(inputs),
        outputs: Array.from(outputs),
        values: values.toJSON(),
        nodes
    };
};

class ValueTable {

    constructor(tensors) {
        this._ids = new Map();
        this._tensors = tensors;
    }

    get(value) {
        if (!value || typeof value !== 'object') {
            return null;
        }
        if (!this._ids.has(value)) {
            this._ids.set(value, {
                id: this._ids.size,
                name: value.name || null,
                description: value.description || null,
                metadata: toObject(value.metadata),
                type: normalizeType(value.type),
                producer: null,
                consumers: [],
                initializer: this._tensors.id(value.initializer || null),
                quantization: normalizeQuantization(value.quantization),
                graph_input: false,
                graph_output: false
            });
        }
        return this._ids.get(value);
    }

    toJSON() {
        return Array.from(this._ids.values());
    }
}

const signatures = (graph) => Array.isArray(graph.signatures) && graph.signatures.length > 0 ? graph.signatures : [graph];

const argumentValues = (argument) => argument && Array.isArray(argument.value) ? argument.value : [];

const normalizeOperator = (type) => type ? {
    domain: type.module || type.domain || null,
    name: type.name || type.type || null,
    overload: type.overload || null,
    version: type.version ?? null
} : {};

const normalizeAttribute = (attribute, tensors) => {
    if (!attribute) {
        return null;
    }
    const name = attribute.name || null;
    const kind = attribute.type || 'attribute';
    if (kind === 'tensor' && attribute.value) {
        return { name, kind, ids: [tensors.id(attribute.value)].filter((id) => id !== null) };
    }
    if (kind === 'tensor[]' && Array.isArray(attribute.value)) {
        return { name, kind, ids: attribute.value.map((tensor) => tensors.id(tensor)).filter((id) => id !== null) };
    }
    if ((kind === 'graph' || kind === 'function') && attribute.value && typeof attribute.value === 'object') {
        return { name, kind, ref: attribute.value.name || null };
    }
    return {
        name,
        kind,
        visible: attribute.visible !== undefined ? attribute.visible : null,
        value: summarize(attribute.value)
    };
};

const normalizeType = (type) => {
    if (!type) {
        return null;
    }
    return {
        text: type.toString ? type.toString() : String(type),
        data_type: type.dataType || null,
        layout: type.layout || null,
        denotation: type.denotation || null,
        shape: Array.isArray(type.shape?.dimensions) ? type.shape.dimensions.map((dimension) => primitive(dimension)) : []
    };
};

const normalizeQuantization = (quantization) => {
    if (!quantization) {
        return null;
    }
    if (quantization.value instanceof Map) {
        return Object.fromEntries(Array.from(quantization.value.entries()).sort(([a], [b]) => a.localeCompare(b)));
    }
    return primitive(quantization.value || quantization);
};

const toObject = (metadata) => {
    const result = {};
    for (const item of metadata || []) {
        if (item && item.name) {
            result[item.name] = primitive(item.value);
        }
    }
    return result;
};

const summarize = (value) => {
    if (value === null || value === undefined || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        return value;
    }
    if (typeof value === 'bigint') {
        return value.toString();
    }
    if (Array.isArray(value)) {
        return value.slice(0, 16).map((item) => summarize(item));
    }
    return value.toString ? value.toString() : Object.prototype.toString.call(value);
};

const primitive = (value) => {
    if (typeof value === 'bigint') {
        return value.toString();
    }
    return value === undefined ? null : value;
};

main().catch((error) => {
    console.error(`${error.name}: ${error.message}`);
    process.exit(1);
});
