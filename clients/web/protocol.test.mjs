import assert from 'node:assert/strict';

import { ProjectionClient, initialLayoutHandle, overviewRows, summaryLists } from './protocol.mjs';

const calls = [];
const transport = {
    async request(method, params) {
        calls.push({ method, params });
        if (method === 'open') {
            return ok(method, { session: 7, summary: summary() });
        }
        if (method === 'search') {
            return ok(method, [{ kind: 'node', handle: { kind: 'node', graph: 0, node: 4 }, id: 4 }]);
        }
        if (method === 'detail') {
            return ok(method, { title: 'node', fields: { operator: 'Add' } });
        }
        if (method === 'layout') {
            return ok(method, { limit_used: params.max_nodes, scope: params.handle, graph: { nodes: [], edges: [] } });
        }
        if (method === 'mlir.symbols') {
            return ok(method, [{ handle: { kind: 'mlir_symbol', symbol: 0 }, name: '@main' }]);
        }
        if (method === 'onnx.tensor') {
            return ok(method, {
                handle: { kind: 'tensor', tensor: params.tensor },
                name: 'weight',
                element_type: 'float32',
                shape: ['3'],
                storage: 'inline_bytes',
            });
        }
        return ok(method, {});
    },
};

const client = new ProjectionClient(transport);
const opened = await client.open('/tmp/model.onnx');
assert.equal(opened.session, 7);
assert.deepEqual(calls.map((call) => call.method), ['open']);
assert.equal(client.requestLog.includes('export'), false);
assert.equal(overviewRows(opened.summary).some(([label]) => label === 'Graphs'), true);
assert.equal(summaryLists(opened.summary)[0][0], 'Graphs');

await client.search('add', 1);
await client.detail({ kind: 'node', graph: 0, node: 4 });
const layout = await client.layout(initialLayoutHandle(opened.summary), 25);
assert.equal(layout.limit_used, 25);
const symbols = await client.mlirSymbols(10);
assert.equal(symbols[0].name, '@main');
const tensor = await client.onnxTensor(0, 10);
assert.equal(tensor.storage, 'inline_bytes');
assert.deepEqual(calls.map((call) => call.method), ['open', 'search', 'detail', 'layout', 'mlir.symbols', 'onnx.tensor']);
assert.deepEqual(calls.at(-1).params, { session: 7, tensor: 0, limit: 10 });
assert.equal(client.requestLog.includes('export'), false);

function ok(command, data) {
    return { schema_version: 1, status: 'ok', command, data };
}

function summary() {
    return {
        format: 'onnx',
        graphs: 1,
        functions: 0,
        nodes: 1,
        values: 2,
        tensors: 1,
        external_data: 0,
        onnx: {
            graph_summaries: [{ handle: { kind: 'graph', graph: 0 }, name: 'main' }],
            histograms: { operator_types: [{ key: 'Add', count: 1 }], storage_kinds: [] },
        },
    };
}
