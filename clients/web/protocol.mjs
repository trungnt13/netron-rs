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

    async searchPage(query, limit = 50, cursor = null) {
        return this.request('search', { session: this.requireSession(), query, limit, cursor });
    }

    async detail(handle) {
        return this.request('detail', { session: this.requireSession(), handle });
    }

    async slice(handle, maxNodes = 500, collapse = 'none') {
        return this.request('slice', { session: this.requireSession(), handle, max_nodes: maxNodes, collapse });
    }

    async layout(handle, maxNodes = 500, collapse = 'none') {
        return this.request('layout', { session: this.requireSession(), handle, max_nodes: maxNodes, collapse });
    }

    async mlirSymbols(limit = 200) {
        return this.request('mlir.symbols', { session: this.requireSession(), limit });
    }

    async mlirSymbolTree(limit = 200) {
        return this.request('mlir.symbol.tree', { session: this.requireSession(), limit });
    }

    async onnxTensor(tensor, limit = 200) {
        return this.request('onnx.tensor', { session: this.requireSession(), tensor, limit });
    }

    cancelPending() {
        if (typeof this.transport.cancelPending === 'function') {
            this.transport.cancelPending(this.session);
        }
    }

    openLocation(location) {
        if (typeof this.transport.openLocation !== 'function') {
            return false;
        }
        return this.transport.openLocation(location) !== false;
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

export class HttpTransport {
    constructor(endpoint, token) {
        this.endpoint = endpoint.replace(/\/+$/, '');
        this.token = token;
        this.nextId = 1;
        this.pending = new Map();
    }

    async request(method, params) {
        const id = String(this.nextId++);
        const cancelable = method === 'layout' || method === 'slice';
        const requestParams = cancelable ? { ...params, cancel_token: id } : params;
        if (cancelable) {
            this.pending.set(id, method);
        }
        try {
            return await this.#post(id, method, requestParams);
        } finally {
            this.pending.delete(id);
        }
    }

    cancelPending(session = null) {
        const pending = [...this.pending.keys()];
        for (const cancelToken of pending) {
            const params = { cancel_token: cancelToken };
            if (session !== null && session !== undefined) {
                params.session = session;
            }
            this.#post(`cancel:${cancelToken}`, 'cancel', params).catch(() => {});
        }
    }

    async #post(id, method, params) {
        const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ id, method, params }),
        });
        const payload = await response.json();
        if (!response.ok && payload?.status !== 'error') {
            throw new Error(`HTTP ${response.status}`);
        }
        return payload;
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
            ['Domains', summary.onnx.histograms?.domains || []],
        ];
    }
    if (summary.mlir) {
        const lists = [
            ['Modules', summary.mlir.modules || []],
            ['Functions', summary.mlir.functions || []],
            ['Regions', summary.mlir.regions || []],
            ['Blocks', summary.mlir.blocks || []],
            ['Resources', summary.mlir.resources || []],
            ['Operations', summary.mlir.histograms?.operations || []],
            ['Dialects', summary.mlir.histograms?.dialects || (summary.mlir.dialects || []).map((name) => ({ key: name, count: 1 }))],
        ];
        if (summary.mlir.bytecode?.sections?.length) {
            lists.push([
                'Bytecode Sections',
                summary.mlir.bytecode.sections.map((section) => ({
                    key: `section ${section.id}`,
                    count: section.len,
                })),
            ]);
        }
        return lists;
    }
    return [];
}

export function mlirNavigationRows(summary, symbols = []) {
    const mlir = summary?.mlir;
    if (!mlir) {
        return [];
    }
    const regionsByScope = groupBy(mlir.regions || [], (region) => region.scope_id || '');
    const blocksByScopeRegion = groupBy(mlir.blocks || [], (block) => `${block.scope_id || ''}:${block.region ?? 0}`);
    const rows = [];
    addMlirScopeGroup(rows, 'Modules', mlir.modules || [], regionsByScope, blocksByScopeRegion);
    addMlirScopeGroup(rows, 'Functions', mlir.functions || [], regionsByScope, blocksByScopeRegion);
    const symbolRows = (Array.isArray(symbols) ? symbols : []).map((symbol) => ({
        kind: 'symbol',
        depth: 1,
        label: symbol.name || handleLabel(symbol.handle),
        detail: '',
        handle: symbol.handle,
    }));
    if (symbolRows.length) {
        rows.push({ kind: 'group', depth: 0, label: 'Symbols', detail: `${symbolRows.length}`, handle: null });
        rows.push(...symbolRows);
    }
    return rows;
}

export function formatLocation(location) {
    if (!location || typeof location !== 'object') {
        return 'location';
    }
    if (location.kind === 'bytecode') {
        const index = location.index ?? location.raw ?? 'unknown';
        const parts = [`bytecode location ${index}`];
        if (Number.isInteger(location.region)) {
            parts.push(`region ${location.region}`);
        }
        if (Number.isInteger(location.block)) {
            parts.push(`block ${location.block}`);
        }
        return parts.join(' · ');
    }
    const file = typeof location.file === 'string' ? location.file : '';
    const line = Number.isInteger(location.line) ? location.line : null;
    const column = Number.isInteger(location.column) ? location.column : null;
    const range = formatLocationRange(location, line);
    if (file && line !== null && column !== null) {
        return `${file}:${line}:${column}${range}`;
    }
    if (file && line !== null) {
        return `${file}:${line}${range}`;
    }
    if (file) {
        return file;
    }
    if (line !== null && column !== null) {
        return `${line}:${column}${range}`;
    }
    if (line !== null) {
        return `line ${line}${range}`;
    }
    return location.raw || location.kind || 'location';
}

function formatLocationRange(location, line) {
    const endLine = Number.isInteger(location.end_line) ? location.end_line : null;
    const endColumn = Number.isInteger(location.end_column) ? location.end_column : null;
    if (endLine !== null && endColumn !== null) {
        return endLine === line ? ` to ${endColumn}` : ` to ${endLine}:${endColumn}`;
    }
    if (endLine !== null) {
        return ` to ${endLine}`;
    }
    if (endColumn !== null) {
        return ` to ${endColumn}`;
    }
    return '';
}

export function canOpenLocation(location) {
    return Boolean(location && location.kind === 'source' && Number.isInteger(location.line) && location.line > 0);
}

export function handleLabel(handle) {
    if (!handle || typeof handle !== 'object') {
        return 'handle';
    }
    switch (handle.kind) {
        case 'graph':
            return `graph ${handle.graph}`;
        case 'node':
            return `graph ${handle.graph} node ${handle.node}`;
        case 'value':
            return `graph ${handle.graph} value ${handle.value}`;
        case 'tensor':
            return `tensor ${handle.tensor}`;
        case 'function':
            return `function ${handle.function}`;
        case 'onnx_repeated_block':
            return `graph ${handle.graph} repeated block ${handle.group}`;
        case 'mlir_module':
            return `module ${handle.module}`;
        case 'mlir_function':
            return `function ${handle.function}`;
        case 'mlir_operation':
            return `${handle.scope} operation ${handle.operation}`;
        case 'mlir_value':
            return `${handle.scope} value ${handle.value}`;
        case 'mlir_region':
            return `${handle.scope} region ${handle.region}`;
        case 'mlir_block':
            return `${handle.scope} block ${handle.block}`;
        case 'mlir_symbol':
            return `symbol ${handle.symbol}`;
        case 'mlir_dialect':
            return `dialect ${handle.dialect}`;
        case 'mlir_attribute':
            return `${handle.scope} attribute ${handle.attribute}`;
        case 'mlir_resource':
            return `resource ${handle.resource}`;
        default:
            return handle.kind || 'handle';
    }
}

function addMlirScopeGroup(rows, title, scopes, regionsByScope, blocksByScopeRegion) {
    if (!scopes.length) {
        return;
    }
    rows.push({ kind: 'group', depth: 0, label: title, detail: `${scopes.length}`, handle: null });
    for (const scope of scopes) {
        rows.push({
            kind: 'scope',
            depth: 1,
            label: scope.name || scope.scope_id || handleLabel(scope.handle),
            detail: `${scope.operation_count ?? 0} ops · ${scope.region_count ?? 0} regions · ${scope.block_count ?? 0} blocks`,
            handle: scope.handle,
        });
        const regions = regionsByScope.get(scope.scope_id || '') || [];
        for (const region of regions) {
            rows.push({
                kind: 'region',
                depth: 2,
                label: handleLabel(region.handle),
                detail: `${region.block_count ?? 0} blocks`,
                handle: region.handle,
            });
            const key = `${region.scope_id || ''}:${region.handle?.region ?? 0}`;
            const blocks = blocksByScopeRegion.get(key) || [];
            for (const block of blocks) {
                rows.push({
                    kind: 'block',
                    depth: 3,
                    label: handleLabel(block.handle),
                    detail: `${block.operation_count ?? 0} ops · ${block.value_count ?? 0} values`,
                    handle: block.handle,
                });
            }
        }
    }
}

function groupBy(values, keyFor) {
    const groups = new Map();
    for (const value of values) {
        const key = keyFor(value);
        const group = groups.get(key);
        if (group) {
            group.push(value);
        } else {
            groups.set(key, [value]);
        }
    }
    return groups;
}
