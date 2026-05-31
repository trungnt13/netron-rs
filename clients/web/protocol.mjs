export class ProjectionClient {
    constructor(transport) {
        this.transport = transport;
        this.session = null;
        this.requestLog = [];
    }

    async open(path) {
        const data = await this.request('open', { path });
        this.session = data.session;
        return data;
    }

    async summary() {
        return this.request('summary', { session: this.requireSession() });
    }

    async diagnostics(limit = 64) {
        return this.request('diagnostics', { session: this.requireSession(), limit });
    }

    async search(query, limit = 50) {
        return this.request('search', { session: this.requireSession(), query, limit });
    }

    async detail(handle) {
        return this.request('detail', { session: this.requireSession(), handle });
    }

    async slice(handle, maxNodes = 500) {
        return this.request('slice', { session: this.requireSession(), handle, max_nodes: maxNodes });
    }

    async layout(handle, maxNodes = 500) {
        return this.request('layout', { session: this.requireSession(), handle, max_nodes: maxNodes });
    }

    async request(method, params) {
        this.requestLog.push(method);
        const response = await this.transport.request(method, params);
        if (!response || response.status !== 'ok') {
            const message = response?.error?.message || `request failed: ${method}`;
            throw new Error(message);
        }
        return response.data;
    }

    requireSession() {
        if (!this.session) {
            throw new Error('session is not open');
        }
        return this.session;
    }
}

export function overviewRows(summary) {
    const rows = [
        ['Format', summary.format || summary.source_format_name || 'unknown'],
        ['Graphs', summary.graphs ?? 0],
        ['Functions', summary.functions ?? 0],
        ['Nodes', summary.nodes ?? 0],
        ['Values', summary.values ?? 0],
        ['Tensors', summary.tensors ?? 0],
        ['Diagnostics', summary.diagnostics ?? 0],
    ];
    if (summary.onnx) {
        rows.push(['Initializers', summary.onnx.initializer_count ?? 0]);
        rows.push(['External Data', summary.external_data ?? 0]);
    }
    if (summary.mlir) {
        rows.push(['Modules', summary.mlir.module_count ?? 0]);
        rows.push(['Regions', summary.mlir.region_count ?? 0]);
        rows.push(['Blocks', summary.mlir.block_count ?? 0]);
        rows.push(['Dialects', summary.mlir.dialects?.length ?? 0]);
    }
    return rows;
}

export function initialLayoutHandle(summary) {
    if (summary.format === 'onnx' && summary.graphs > 0) {
        return { kind: 'graph', graph: 0 };
    }
    const functionHandle = summary.mlir?.functions?.[0]?.handle;
    if (functionHandle) {
        return functionHandle;
    }
    const moduleHandle = summary.mlir?.modules?.[0]?.handle;
    return moduleHandle || null;
}

export function summaryLists(summary) {
    if (summary.onnx) {
        return [
            ['Graphs', summary.onnx.graph_summaries || []],
            ['Operators', summary.onnx.histograms?.operator_types || []],
            ['Tensors', summary.onnx.histograms?.storage_kinds || []],
        ];
    }
    if (summary.mlir) {
        return [
            ['Modules', summary.mlir.modules || []],
            ['Functions', summary.mlir.functions || []],
            ['Dialects', (summary.mlir.dialects || []).map((name) => ({ key: name, count: 1 }))],
            ['Resources', summary.mlir.resources || []],
        ];
    }
    return [];
}
