import assert from 'node:assert/strict';

import { HttpTransport, ProjectionClient, canOpenLocation, formatLocation, handleLabel, initialLayoutHandle, mlirNavigationRows, overviewRows, summaryLists } from './protocol.mjs';

const calls = [];
const transport = {
    async request(method, params) {
        calls.push({ method, params });
        if (method === 'open') {
            return ok(method, { session: 7, summary: summary() });
        }
        if (method === 'search') {
            if (Object.hasOwn(params, 'cursor')) {
                return ok(method, {
                    api_version: 1,
                    session_id: params.session,
                    format: 'onnx',
                    query: params.query,
                    next_cursor: null,
                    limit_used: params.limit,
                    total_count: 1,
                    truncated: false,
                    omitted_count: 0,
                    results: [{ kind: 'node', handle: { kind: 'node', graph: 0, node: 4 }, id: 4 }],
                });
            }
            return ok(method, [{ kind: 'node', handle: { kind: 'node', graph: 0, node: 4 }, id: 4 }]);
        }
        if (method === 'detail') {
            return ok(method, { title: 'node', fields: { operator: 'Add' } });
        }
        if (method === 'layout') {
            return ok(method, { limit_used: params.max_nodes, scope: params.handle, graph: { nodes: [], edges: [] } });
        }
        if (method === 'slice') {
            return ok(method, { limit_used: params.max_nodes, scope: params.handle, entities: [], edges: [] });
        }
        if (method === 'mlir.symbols') {
            return ok(method, [{ handle: { kind: 'mlir_symbol', symbol: 0 }, name: '@main' }]);
        }
        if (method === 'mlir.symbol.tree') {
            return ok(method, {
                api_version: 1,
                limit_used: params.limit,
                truncated: false,
                omitted_count: 0,
                roots: [{ id: 'mlir_tree:symbols', kind: 'group', label: 'Symbols', children: [] }],
            });
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
const mlirLists = summaryLists({
    mlir: {
        modules: [],
        functions: [],
        regions: [],
        blocks: [],
        resources: [{ name: 'blob', kind: 'rodata', handle: { kind: 'mlir_resource', resource: 0 } }],
        histograms: {
            operations: [{ key: 'func.func', count: 2 }],
            dialects: [{ key: 'func', count: 2 }],
        },
        bytecode: {
            sections: [{ id: 1, len: 64 }],
        },
    },
});
assert.deepEqual(mlirLists.find(([label]) => label === 'Resources'), ['Resources', [{ name: 'blob', kind: 'rodata', handle: { kind: 'mlir_resource', resource: 0 } }]]);
assert.deepEqual(mlirLists.find(([label]) => label === 'Operations'), ['Operations', [{ key: 'func.func', count: 2 }]]);
assert.deepEqual(mlirLists.find(([label]) => label === 'Dialects'), ['Dialects', [{ key: 'func', count: 2 }]]);
assert.deepEqual(mlirLists.find(([label]) => label === 'Bytecode Sections'), ['Bytecode Sections', [{ key: 'section 1', count: 64 }]]);
const mlirNavigation = mlirNavigationRows({
    mlir: {
        modules: [
            {
                scope_id: 'module:0',
                name: '@module',
                operation_count: 3,
                region_count: 1,
                block_count: 1,
                handle: { kind: 'mlir_module', module: 0 },
            },
        ],
        functions: [
            {
                scope_id: 'function:0',
                name: '@main',
                operation_count: 2,
                region_count: 1,
                block_count: 1,
                handle: { kind: 'mlir_function', function: 0 },
            },
        ],
        regions: [
            { scope_id: 'module:0', block_count: 1, handle: { kind: 'mlir_region', scope: 'module:0', region: 0 } },
            { scope_id: 'function:0', block_count: 1, handle: { kind: 'mlir_region', scope: 'function:0', region: 0 } },
        ],
        blocks: [
            {
                scope_id: 'module:0',
                region: 0,
                operation_count: 3,
                value_count: 1,
                handle: { kind: 'mlir_block', scope: 'module:0', block: 0 },
            },
            {
                scope_id: 'function:0',
                region: 0,
                operation_count: 2,
                value_count: 1,
                handle: { kind: 'mlir_block', scope: 'function:0', block: 0 },
            },
        ],
    },
}, [{ handle: { kind: 'mlir_symbol', symbol: 0 }, name: '@main' }]);
assert.deepEqual(
    mlirNavigation.map((row) => [row.kind, row.depth, row.label]),
    [
        ['group', 0, 'Modules'],
        ['scope', 1, '@module'],
        ['region', 2, 'module:0 region 0'],
        ['block', 3, 'module:0 block 0'],
        ['group', 0, 'Functions'],
        ['scope', 1, '@main'],
        ['region', 2, 'function:0 region 0'],
        ['block', 3, 'function:0 block 0'],
        ['group', 0, 'Symbols'],
        ['symbol', 1, '@main'],
    ],
);
assert.equal(handleLabel({ kind: 'mlir_operation', scope: 'function:0', operation: 3 }), 'function:0 operation 3');
assert.equal(formatLocation({ kind: 'source', file: 'kernel.mlir', line: 12, column: 4 }), 'kernel.mlir:12:4');
assert.equal(formatLocation({ kind: 'source', file: 'kernel.mlir', line: 12, column: 4, end_line: 13, end_column: 2 }), 'kernel.mlir:12:4 to 13:2');
assert.equal(formatLocation({ kind: 'source', file: 'kernel.mlir', line: 12, column: 4, end_line: 12, end_column: 9 }), 'kernel.mlir:12:4 to 9');
assert.equal(formatLocation({ kind: 'source', line: 12, column: 4 }), '12:4');
assert.equal(formatLocation({ kind: 'bytecode', index: 42, region: 3, block: 7 }), 'bytecode location 42 · region 3 · block 7');
assert.equal(canOpenLocation({ kind: 'source', line: 12 }), true);
assert.equal(canOpenLocation({ kind: 'bytecode', index: 42 }), false);

const jumpCalls = [];
const jumpClient = new ProjectionClient({
    openLocation(location) {
        jumpCalls.push(location);
        return true;
    },
    async request(method) {
        return ok(method, {});
    },
});
assert.equal(jumpClient.openLocation({ kind: 'source', line: 12 }), true);
assert.deepEqual(jumpCalls, [{ kind: 'source', line: 12 }]);
assert.equal(new ProjectionClient({ async request(method) { return ok(method, {}); } }).openLocation({ kind: 'source', line: 12 }), false);

await client.search('add', 1);
const searchPage = await client.searchPage('add', 1, null);
assert.equal(searchPage.results.length, 1);
assert.deepEqual(calls.at(-1).params, { session: 7, query: 'add', limit: 1, cursor: null });
await client.detail({ kind: 'node', graph: 0, node: 4 });
const layout = await client.layout(initialLayoutHandle(opened.summary), 25);
assert.equal(layout.limit_used, 25);
assert.deepEqual(calls.at(-1).params, { session: 7, handle: initialLayoutHandle(opened.summary), max_nodes: 25, collapse: 'none' });
await client.slice(initialLayoutHandle(opened.summary), 12, 'structural');
assert.deepEqual(calls.at(-1).params, { session: 7, handle: initialLayoutHandle(opened.summary), max_nodes: 12, collapse: 'structural' });
await client.layout(initialLayoutHandle(opened.summary), 13, 'structural');
assert.deepEqual(calls.at(-1).params, { session: 7, handle: initialLayoutHandle(opened.summary), max_nodes: 13, collapse: 'structural' });
const symbols = await client.mlirSymbols(10);
assert.equal(symbols[0].name, '@main');
const symbolTree = await client.mlirSymbolTree(11);
assert.equal(symbolTree.roots[0].label, 'Symbols');
assert.deepEqual(calls.at(-1).params, { session: 7, limit: 11 });
const tensor = await client.onnxTensor(0, 10);
assert.equal(tensor.storage, 'inline_bytes');
assert.deepEqual(calls.map((call) => call.method), ['open', 'search', 'search', 'detail', 'layout', 'slice', 'layout', 'mlir.symbols', 'mlir.symbol.tree', 'onnx.tensor']);
assert.deepEqual(calls.at(-1).params, { session: 7, tensor: 0, limit: 10 });
assert.equal(client.requestLog.includes('export'), false);

const originalFetch = globalThis.fetch;
const httpCalls = [];
globalThis.fetch = async (url, options) => {
    httpCalls.push({ url, options });
    return {
        ok: true,
        status: 200,
        async json() {
            return ok('summary', { format: 'onnx' });
        },
    };
};
const http = new HttpTransport('http://127.0.0.1:4567/', 'secret-token');
const httpResponse = await http.request('summary', { session: 7 });
assert.equal(httpResponse.data.format, 'onnx');
assert.equal(httpCalls[0].url, 'http://127.0.0.1:4567');
assert.equal(httpCalls[0].options.headers.Authorization, 'Bearer secret-token');
assert.deepEqual(JSON.parse(httpCalls[0].options.body), { id: '1', method: 'summary', params: { session: 7 } });

let resolveProjectionFetch;
globalThis.fetch = async (url, options) => {
    httpCalls.push({ url, options });
    const body = JSON.parse(options.body);
    if (body.method === 'layout' || body.method === 'slice') {
        return await new Promise((resolve) => {
            resolveProjectionFetch = () => resolve({
                ok: true,
                status: 200,
                async json() {
                    return ok(body.method, { graph: { nodes: [], edges: [] }, entities: [] });
                },
            });
        });
    }
    return {
        ok: true,
        status: 200,
        async json() {
            return ok(body.method, { canceled: true });
        },
    };
};
const cancelHttp = new HttpTransport('http://127.0.0.1:4567/', 'secret-token');
const pendingLayout = cancelHttp.request('layout', { session: 7, handle: { kind: 'graph', graph: 0 } });
cancelHttp.cancelPending(7);
const layoutBody = JSON.parse(httpCalls.at(-2).options.body);
const cancelBody = JSON.parse(httpCalls.at(-1).options.body);
assert.equal(layoutBody.method, 'layout');
assert.equal(layoutBody.params.cancel_token, layoutBody.id);
assert.deepEqual(cancelBody, {
    id: `cancel:${layoutBody.id}`,
    method: 'cancel',
    params: { cancel_token: layoutBody.id, session: 7 },
});
resolveProjectionFetch();
await pendingLayout;
const pendingSlice = cancelHttp.request('slice', { session: 7, handle: { kind: 'graph', graph: 0 } });
cancelHttp.cancelPending(7);
const sliceBody = JSON.parse(httpCalls.at(-2).options.body);
const cancelSliceBody = JSON.parse(httpCalls.at(-1).options.body);
assert.equal(sliceBody.method, 'slice');
assert.equal(sliceBody.params.cancel_token, sliceBody.id);
assert.deepEqual(cancelSliceBody, {
    id: `cancel:${sliceBody.id}`,
    method: 'cancel',
    params: { cancel_token: sliceBody.id, session: 7 },
});
resolveProjectionFetch();
await pendingSlice;
globalThis.fetch = originalFetch;

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
