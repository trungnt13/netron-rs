import { ProjectionClient, initialLayoutHandle, overviewRows, summaryLists } from './protocol.mjs';

const state = {
    client: null,
    summary: null,
    session: null,
    selectedHandle: null,
};

const nodes = {};

window.addEventListener('DOMContentLoaded', () => {
    for (const id of [
        'status',
        'overview',
        'lists',
        'search',
        'results',
        'detail',
        'layout',
        'diagnostics',
        'request-log',
    ]) {
        nodes[id] = document.getElementById(id);
    }
    nodes.search.addEventListener('input', debounce(runSearch, 160));
    document.getElementById('layout-button').addEventListener('click', requestLayout);
    document.getElementById('diagnostics-button').addEventListener('click', requestDiagnostics);

    const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : null;
    const transport = vscode ? new VsCodeTransport(vscode) : new DemoTransport();
    state.client = new ProjectionClient(transport);
    if (vscode) {
        vscode.postMessage({ type: 'ready' });
    } else {
        openDemo();
    }
});

window.addEventListener('message', (event) => {
    if (event.data?.type === 'ready') {
        receiveOpen(event.data.response);
    }
});

function receiveOpen(response) {
    if (response.status !== 'ok') {
        setStatus(response.error?.message || 'Open failed');
        return;
    }
    const data = response.data;
    state.session = data.session;
    state.client.session = data.session;
    state.client.requestLog.push('open');
    renderSummary(data.summary);
    setStatus(`Session ${data.session}`);
}

async function openDemo() {
    const data = await state.client.open('demo.onnx');
    renderSummary(data.summary);
    setStatus('Demo session');
}

async function runSearch() {
    const query = nodes.search.value.trim();
    if (!query || !state.client.session) {
        renderVirtualList(nodes.results, []);
        return;
    }
    try {
        const results = await state.client.search(query, 80);
        renderVirtualList(nodes.results, results, (entry) => searchRow(entry), (entry) => {
            state.selectedHandle = entry.handle;
            requestDetail(entry.handle);
        });
        renderRequestLog();
    } catch (error) {
        setStatus(error.message);
    }
}

async function requestDetail(handle) {
    try {
        const detail = await state.client.detail(handle);
        nodes.detail.replaceChildren(renderDetail(detail));
        renderRequestLog();
    } catch (error) {
        setStatus(error.message);
    }
}

async function requestLayout() {
    const handle = state.selectedHandle || initialLayoutHandle(state.summary);
    if (!handle) {
        setStatus('No layout handle');
        return;
    }
    try {
        const layout = await state.client.layout(handle, 500);
        drawLayout(layout);
        renderRequestLog();
    } catch (error) {
        setStatus(error.message);
    }
}

async function requestDiagnostics() {
    try {
        const diagnostics = await state.client.diagnostics(100);
        renderVirtualList(nodes.diagnostics, diagnostics.diagnostics || [], (entry) => {
            const source = entry.source ? ` · ${entry.source}` : '';
            return `${entry.kind}: ${entry.code}${source}`;
        });
        renderRequestLog();
    } catch (error) {
        setStatus(error.message);
    }
}

function renderSummary(summary) {
    state.summary = summary;
    state.selectedHandle = initialLayoutHandle(summary);
    nodes.overview.replaceChildren(...overviewRows(summary).map(([label, value]) => metric(label, value)));
    nodes.lists.replaceChildren(...summaryLists(summary).map(([title, rows]) => {
        const section = document.createElement('section');
        const heading = document.createElement('h2');
        heading.textContent = title;
        const list = document.createElement('div');
        list.className = 'virtual-list';
        renderVirtualList(list, rows, compactRow, (row) => {
            if (row.handle) {
                state.selectedHandle = row.handle;
                requestDetail(row.handle);
            }
        });
        section.append(heading, list);
        return section;
    }));
    drawEmptyLayout();
    renderRequestLog();
}

function metric(label, value) {
    const item = document.createElement('div');
    item.className = 'metric';
    const name = document.createElement('span');
    name.textContent = label;
    const count = document.createElement('strong');
    count.textContent = String(value);
    item.append(name, count);
    return item;
}

function renderVirtualList(container, rows, label = compactRow, onClick = null) {
    const rowHeight = 28;
    const viewport = container.clientHeight || 220;
    const start = Math.floor(container.scrollTop / rowHeight);
    const visible = Math.ceil(viewport / rowHeight) + 6;
    const end = Math.min(rows.length, start + visible);
    const spacer = document.createElement('div');
    spacer.style.height = `${rows.length * rowHeight}px`;
    spacer.className = 'virtual-spacer';
    for (let index = start; index < end; index++) {
        const row = document.createElement('button');
        row.className = 'list-row';
        row.style.transform = `translateY(${index * rowHeight}px)`;
        row.textContent = label(rows[index]);
        if (onClick) {
            row.addEventListener('click', () => onClick(rows[index]));
        }
        spacer.append(row);
    }
    container.onscroll = () => renderVirtualList(container, rows, label, onClick);
    container.replaceChildren(spacer);
}

function compactRow(row) {
    if (typeof row === 'string') {
        return row;
    }
    return row.name || row.key || row.scope_id || row.operator || row.title || JSON.stringify(row.handle || row);
}

function searchRow(entry) {
    const title = entry.name || entry.operator || `${entry.kind} ${entry.id}`;
    return `${entry.kind} · ${title}`;
}

function renderDetail(detail) {
    const fragment = document.createDocumentFragment();
    const title = document.createElement('h2');
    title.textContent = detail.title;
    fragment.append(title);
    for (const [name, value] of Object.entries(detail.fields || {})) {
        const row = document.createElement('div');
        row.className = 'detail-row';
        const key = document.createElement('span');
        key.textContent = name;
        const text = document.createElement('code');
        text.textContent = value;
        row.append(key, text);
        fragment.append(row);
    }
    return fragment;
}

function drawEmptyLayout() {
    const context = nodes.layout.getContext('2d');
    context.clearRect(0, 0, nodes.layout.width, nodes.layout.height);
    context.fillStyle = '#4b5563';
    context.fillText('No layout requested', 24, 32);
}

function drawLayout(response) {
    const graph = response.graph || { nodes: [], edges: [] };
    const context = nodes.layout.getContext('2d');
    context.clearRect(0, 0, nodes.layout.width, nodes.layout.height);
    context.strokeStyle = '#9ca3af';
    context.fillStyle = '#111827';
    for (const edge of graph.edges || []) {
        const from = findLayoutNode(graph.nodes, edge.from);
        const to = findLayoutNode(graph.nodes, edge.to);
        if (!from || !to) {
            continue;
        }
        context.beginPath();
        context.moveTo(from.x + 120, from.y + 22);
        context.lineTo(to.x, to.y + 22);
        context.stroke();
    }
    for (const item of graph.nodes || []) {
        context.fillStyle = item.kind === 'operator' ? '#dbeafe' : '#ecfdf5';
        context.strokeStyle = '#1f2937';
        context.fillRect(item.x, item.y, 112, 42);
        context.strokeRect(item.x, item.y, 112, 42);
        context.fillStyle = '#111827';
        context.fillText(item.label || item.operator || item.id, item.x + 8, item.y + 25, 96);
    }
    setStatus(response.truncated ? `Layout truncated at ${response.limit_used}` : `Layout ${graph.nodes?.length || 0} nodes`);
}

function findLayoutNode(nodes, id) {
    const item = nodes.find((node) => node.id === id);
    if (!item) {
        return null;
    }
    return {
        x: Math.max(16, Math.round(item.position?.x ?? item.x ?? 16)),
        y: Math.max(16, Math.round(item.position?.y ?? item.y ?? 16)),
    };
}

function renderRequestLog() {
    nodes['request-log'].textContent = state.client.requestLog.join(' → ');
}

function setStatus(message) {
    nodes.status.textContent = message;
}

function debounce(callback, delay) {
    let timer = 0;
    return () => {
        clearTimeout(timer);
        timer = setTimeout(callback, delay);
    };
}

class VsCodeTransport {
    constructor(vscode) {
        this.vscode = vscode;
        this.nextId = 1;
        this.pending = new Map();
        window.addEventListener('message', (event) => {
            const message = event.data;
            if (!message || !this.pending.has(message.id)) {
                return;
            }
            this.pending.get(message.id)(message.response);
            this.pending.delete(message.id);
        });
    }

    request(method, params) {
        const id = this.nextId++;
        return new Promise((resolve) => {
            this.pending.set(id, resolve);
            this.vscode.postMessage({ id, method, params });
        });
    }
}

class DemoTransport {
    constructor() {
        this.session = 1;
    }

    async request(method, params) {
        if (method === 'open') {
            return ok('open', {
                session: this.session,
                summary: demoSummary(),
            });
        }
        if (method === 'search') {
            return ok('search', [
                {
                    kind: 'node',
                    handle: { kind: 'node', graph: 0, node: 0 },
                    graph: 0,
                    id: 0,
                    name: params.query,
                    operator: 'MatMul',
                },
            ]);
        }
        if (method === 'detail') {
            return ok('detail', {
                title: 'MatMul',
                handle: params.handle,
                fields: { operator: 'MatMul', inputs: 'x, w', outputs: 'y' },
                related: [],
            });
        }
        if (method === 'diagnostics') {
            return ok('diagnostics', { diagnostics: [], truncated: false });
        }
        if (method === 'layout') {
            return ok('layout', demoLayout());
        }
        return ok(method, {});
    }
}

function ok(command, data) {
    return { schema_version: 1, status: 'ok', command, data };
}

function demoSummary() {
    return {
        format: 'onnx',
        source_format_name: 'ONNX',
        graphs: 1,
        functions: 0,
        nodes: 3,
        values: 4,
        tensors: 1,
        diagnostics: 0,
        external_data: 0,
        onnx: {
            initializer_count: 1,
            graph_summaries: [{ name: 'main', node_count: 3, value_count: 4, tensor_count: 1, handle: { kind: 'graph', graph: 0 } }],
            histograms: {
                operator_types: [{ key: 'MatMul', count: 2 }, { key: 'Add', count: 1 }],
                storage_kinds: [{ key: 'inline_bytes', count: 1 }],
            },
        },
    };
}

function demoLayout() {
    return {
        scope: { kind: 'graph', graph: 0 },
        limit_used: 500,
        truncated: false,
        graph: {
            nodes: [
                { id: 'x', label: 'x', kind: 'value', x: 32, y: 80 },
                { id: 'matmul', label: 'MatMul', kind: 'operator', x: 200, y: 80 },
                { id: 'y', label: 'y', kind: 'value', x: 380, y: 80 },
            ],
            edges: [{ from: 'x', to: 'matmul' }, { from: 'matmul', to: 'y' }],
        },
    };
}
