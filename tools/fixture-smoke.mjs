#!/usr/bin/env node

import * as child_process from 'child_process';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..');

const defaultFixtures = [
    {
        name: 'mlirbc-model',
        path: 'tests/fixtures/mlir/model.mlirbc',
        format: 'MLIR',
        minFunctions: 1,
        requireDecodedBytecodeProperties: true,
        requireDecodedBytecodeSummaryEntries: true,
        maxSummaryJsonBytes: 64 * 1024
    },
    {
        name: 'mlirbc-sd-clip-tank',
        path: 'tests/fixtures/mlir/sd-clip-tank.mlirbc',
        format: 'MLIR',
        minFunctions: 2,
        requireDecodedBytecodeProperties: true,
        requireDecodedBytecodeSummaryEntries: true,
        maxSummaryJsonBytes: 64 * 1024
    },
    {
        name: 'onnx-external-chain',
        path: 'tests/fixtures/onnx/external-chain.onnx',
        format: 'ONNX',
        minNodes: 1024,
        externalTensors: 1,
        externalPayload: 'tests/fixtures/onnx/external-chain.bin',
        maxSummaryJsonBytes: 16 * 1024
    }
];

const requiredNonNegative = [
    'session_open_ms_last',
    'summary_ms_last',
    'search_ms_last',
    'tensor_metadata_ms_last',
    'session_layout_ms_last',
    'parse_and_json_ms_last',
    'time_to_first_graph_ms_last'
];
const requiredPositive = ['summary_json_bytes_last', 'peak_rss_kb_last'];

const main = async () => {
    const options = parseArgs(process.argv.slice(2));
    const manifestFixtures = [];
    for (const manifest of options.manifests) {
        manifestFixtures.push(...await loadManifest(manifest));
    }
    const fixtures = [
        ...(options.includeDefaultFixtures ? defaultFixtures : []),
        ...manifestFixtures
    ];
    if (fixtures.length === 0) {
        throw new Error('No fixtures selected.');
    }
    const results = [];
    for (const fixture of fixtures) {
        results.push(await checkFixture(fixture, options));
    }
    const failures = results.flatMap((result) => result.failures.map((failure) => ({
        fixture: result.name,
        failure
    })));
    const report = {
        ok: failures.length === 0,
        thresholds: fixtures.map(({
            name,
            maxSummaryJsonBytes,
            maxPeakRssKb,
            minBytes,
            minFunctions,
            minNodes,
            minTensors,
            minValues,
            externalTensors,
            requireDecodedBytecodeProperties,
            requireDecodedBytecodeSummaryEntries
        }) => ({
            name,
            max_summary_json_bytes: maxSummaryJsonBytes,
            max_peak_rss_kb: maxPeakRssKb ?? null,
            min_bytes: minBytes ?? null,
            min_functions: minFunctions ?? null,
            min_nodes: minNodes ?? null,
            min_tensors: minTensors ?? null,
            min_values: minValues ?? null,
            external_tensors: externalTensors ?? null,
            require_decoded_bytecode_properties: requireDecodedBytecodeProperties === true,
            require_decoded_bytecode_summary_entries: requireDecodedBytecodeSummaryEntries === true
        })),
        measurement: {
            iterations: options.iterations
        },
        results,
        failures
    };
    const output = JSON.stringify(report, null, 2);
    if (options.output) {
        await fs.mkdir(path.dirname(options.output), { recursive: true });
        await fs.writeFile(options.output, `${output}\n`);
    }
    console.log(output);
    if (!report.ok) {
        process.exit(1);
    }
};

const parseArgs = (args) => {
    const options = {
        includeDefaultFixtures: true,
        iterations: 1,
        manifests: [],
        output: null,
        rustBin: null
    };
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        switch (arg) {
            case '--iterations':
                options.iterations = positiveInt(value(args, ++index, arg), arg);
                break;
            case '--manifest':
                options.manifests.push(path.resolve(value(args, ++index, arg)));
                break;
            case '--no-default-fixtures':
                options.includeDefaultFixtures = false;
                break;
            case '--output':
                options.output = path.resolve(value(args, ++index, arg));
                break;
            case '--rust-bin':
                options.rustBin = path.resolve(value(args, ++index, arg));
                break;
            default:
                throw new Error(`Unknown option: ${arg}`);
        }
    }
    return options;
};

const loadManifest = async (manifestPath) => {
    const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    const entries = Array.isArray(manifest) ? manifest : manifest.fixtures;
    if (!Array.isArray(entries)) {
        throw new Error(`${manifestPath} must be an array or an object with a fixtures array`);
    }
    const baseDir = path.dirname(manifestPath);
    return entries.map((entry, index) => normalizeManifestFixture(entry, index, baseDir, manifestPath));
};

const normalizeManifestFixture = (entry, index, baseDir, manifestPath) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new Error(`${manifestPath} fixture ${index} must be an object`);
    }
    const fixture = {
        ...entry,
        name: stringField(entry, 'name', manifestPath, index),
        path: stringField(entry, 'path', manifestPath, index),
        format: stringField(entry, 'format', manifestPath, index),
        baseDir
    };
    for (const field of [
        'externalTensors',
        'maxPeakRssKb',
        'maxSummaryJsonBytes',
        'minBytes',
        'minFunctions',
        'minNodes',
        'minTensors',
        'minValues'
    ]) {
        if (fixture[field] !== undefined) {
            fixture[field] = nonNegativeInt(fixture[field], `${manifestPath} fixture ${fixture.name} ${field}`);
        }
    }
    if (fixture.externalPayload !== undefined && typeof fixture.externalPayload !== 'string') {
        throw new Error(`${manifestPath} fixture ${fixture.name} externalPayload must be a string`);
    }
    if (fixture.requireDecodedBytecodeProperties !== undefined && typeof fixture.requireDecodedBytecodeProperties !== 'boolean') {
        throw new Error(`${manifestPath} fixture ${fixture.name} requireDecodedBytecodeProperties must be a boolean`);
    }
    if (fixture.requireDecodedBytecodeSummaryEntries !== undefined && typeof fixture.requireDecodedBytecodeSummaryEntries !== 'boolean') {
        throw new Error(`${manifestPath} fixture ${fixture.name} requireDecodedBytecodeSummaryEntries must be a boolean`);
    }
    return fixture;
};

const checkFixture = async (fixture, options) => {
    const target = resolveFixturePath(fixture.path, fixture.baseDir);
    const bench = await runBench(target, options);
    const failures = [];
    for (const key of requiredNonNegative) {
        if (!Number.isFinite(bench[key]) || bench[key] < 0) {
            failures.push(`${key} must be present and non-negative`);
        }
    }
    for (const key of requiredPositive) {
        if (!Number.isFinite(bench[key]) || bench[key] <= 0) {
            failures.push(`${key} must be present and positive`);
        }
    }
    const stats = bench.stats || {};
    if (fixture.minBytes !== undefined && (bench.bytes || 0) < fixture.minBytes) {
        failures.push(`bytes must be >= ${fixture.minBytes}`);
    }
    if (stats.format !== fixture.format) {
        failures.push(`format must be ${fixture.format}`);
    }
    if (fixture.minFunctions !== undefined && (stats.functions || 0) < fixture.minFunctions) {
        failures.push(`functions must be >= ${fixture.minFunctions}`);
    }
    if (fixture.minNodes !== undefined && (stats.nodes || 0) < fixture.minNodes) {
        failures.push(`nodes must be >= ${fixture.minNodes}`);
    }
    if (fixture.minTensors !== undefined && (stats.tensors || 0) < fixture.minTensors) {
        failures.push(`tensors must be >= ${fixture.minTensors}`);
    }
    if (fixture.minValues !== undefined && (stats.values || 0) < fixture.minValues) {
        failures.push(`values must be >= ${fixture.minValues}`);
    }
    if (fixture.externalTensors !== undefined && stats.tensor_external !== fixture.externalTensors) {
        failures.push(`tensor_external must be ${fixture.externalTensors}`);
    }
    if (fixture.maxSummaryJsonBytes !== undefined && (bench.summary_json_bytes_last || 0) > fixture.maxSummaryJsonBytes) {
        failures.push(`summary_json_bytes_last must be <= ${fixture.maxSummaryJsonBytes}`);
    }
    if (fixture.maxPeakRssKb !== undefined && (bench.peak_rss_kb_last || 0) > fixture.maxPeakRssKb) {
        failures.push(`peak_rss_kb_last must be <= ${fixture.maxPeakRssKb}`);
    }
    if (fixture.externalPayload) {
        const payload = await fs.stat(resolveFixturePath(fixture.externalPayload, fixture.baseDir));
        if (payload.size <= bench.bytes) {
            failures.push('external payload must be larger than protobuf');
        }
    }
    const bytecodeProperties = fixture.requireDecodedBytecodeProperties ?
        await checkDecodedBytecodeProperties(target, options) :
        null;
    if (bytecodeProperties && bytecodeProperties.opaque_nodes > 0) {
        failures.push(`bytecode property nodes must expose decoded metadata; ${bytecodeProperties.opaque_nodes} opaque`);
    }
    const bytecodeSummary = fixture.requireDecodedBytecodeSummaryEntries ?
        await checkDecodedBytecodeSummaryEntries(target, options) :
        null;
    if (bytecodeSummary && bytecodeSummary.opaque_custom_entries > 0) {
        failures.push(`bytecode summary custom attributes/types must expose decoded assembly; ${bytecodeSummary.opaque_custom_entries} opaque`);
    }
    return {
        name: fixture.name,
        file: displayPath(target),
        ok: failures.length === 0,
        failures,
        bytes: bench.bytes,
        stats,
        bytecode_properties: bytecodeProperties,
        bytecode_summary: bytecodeSummary,
        summary_json_bytes_last: bench.summary_json_bytes_last,
        peak_rss_kb_last: bench.peak_rss_kb_last,
        session_open_ms_last: bench.session_open_ms_last,
        session_layout_ms_last: bench.session_layout_ms_last
    };
};

const runBench = async (target, options) => {
    const args = options.rustBin ?
        ['bench', target, String(options.iterations)] :
        ['run', '--locked', '--', 'bench', target, String(options.iterations)];
    const output = await execFile(options.rustBin || 'cargo', args);
    return JSON.parse(output);
};

const checkDecodedBytecodeProperties = async (target, options) => {
    const args = options.rustBin ?
        ['parse', target] :
        ['run', '--locked', '--', 'parse', target];
    const output = await execFile(options.rustBin || 'cargo', args);
    const model = JSON.parse(output);
    const nodes = (model.functions || []).flatMap((fn) => fn.nodes || []);
    const propertyNodes = nodes.filter((node) => node.metadata && node.metadata['bytecode.properties'] !== undefined);
    const decodedNodes = propertyNodes.filter((node) =>
        Object.keys(node.metadata).some((key) => key.startsWith('bytecode.property.')));
    return {
        property_nodes: propertyNodes.length,
        decoded_nodes: decodedNodes.length,
        opaque_nodes: propertyNodes.length - decodedNodes.length
    };
};

const checkDecodedBytecodeSummaryEntries = async (target, options) => {
    const args = options.rustBin ?
        ['summary', '--json', target] :
        ['run', '--locked', '--', 'summary', '--json', target];
    const output = await execFile(options.rustBin || 'cargo', args);
    const envelope = JSON.parse(output);
    const bytecode = envelope?.data?.mlir?.bytecode || {};
    const attributes = checkDecodedBytecodeSummaryList(bytecode.attributes);
    const types = checkDecodedBytecodeSummaryList(bytecode.types);
    return {
        attributes,
        types,
        custom_entries: attributes.custom_entries + types.custom_entries,
        decoded_custom_entries: attributes.decoded_custom_entries + types.decoded_custom_entries,
        opaque_custom_entries: attributes.opaque_custom_entries + types.opaque_custom_entries
    };
};

const checkDecodedBytecodeSummaryList = (entries) => {
    const items = Array.isArray(entries) ? entries : [];
    const customEntries = items.filter((entry) => entry && entry.has_custom_encoding === true);
    const decodedCustomEntries = customEntries.filter((entry) =>
        typeof entry.assembly === 'string' && entry.assembly.length > 0);
    return {
        entries: items.length,
        custom_entries: customEntries.length,
        decoded_custom_entries: decodedCustomEntries.length,
        opaque_custom_entries: customEntries.length - decodedCustomEntries.length
    };
};

const positiveInt = (input, flag) => {
    const result = Number.parseInt(input, 10);
    if (!Number.isInteger(result) || result < 1) {
        throw new Error(`${flag} must be a positive integer`);
    }
    return result;
};

const nonNegativeInt = (input, field) => {
    const result = Number.parseInt(input, 10);
    if (!Number.isInteger(result) || result < 0) {
        throw new Error(`${field} must be a non-negative integer`);
    }
    return result;
};

const stringField = (input, field, manifestPath, index) => {
    const result = input[field];
    if (typeof result !== 'string' || result.length === 0) {
        throw new Error(`${manifestPath} fixture ${index} ${field} must be a non-empty string`);
    }
    return result;
};

const resolveFixturePath = (fixturePath, baseDir = root) =>
    path.resolve(path.isAbsolute(fixturePath) ? fixturePath : path.join(baseDir || root, fixturePath));

const displayPath = (fixturePath) => {
    const relative = path.relative(root, fixturePath);
    return relative && !relative.startsWith('..') && !path.isAbsolute(relative) ? relative : fixturePath;
};

const value = (args, index, flag) => {
    const result = args[index];
    if (!result) {
        throw new Error(`${flag} requires a value`);
    }
    return result;
};

const execFile = (command, args) => new Promise((resolve, reject) => {
    child_process.execFile(command, args, { cwd: root, maxBuffer: 64 * 1024 * 1024 }, (error, stdout, stderr) => {
        if (error) {
            error.message = `${error.message}\n${stderr}`;
            reject(error);
            return;
        }
        resolve(stdout);
    });
});

main().catch((error) => {
    console.error(`${error.name}: ${error.message}`);
    process.exit(1);
});
