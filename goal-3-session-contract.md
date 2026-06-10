# Goal 3 Session Contract (v1)

This document defines the shared indexed-session API contract used by CLI, service,
and future web/VS Code clients.

## Version

- `session_api_version`: `1`
- Stable JSON objects use snake_case keys.

## Transports

- `serve --stdio` reads one JSON request per line and writes one JSON response
  per line. Requests are correlated by `id`; responses may arrive out of order
  because cancelable work can run concurrently with later requests.
- `serve --http [addr]` or `serve --http --listen <addr>` starts a local HTTP
  transport for the same request/response objects. On startup it writes one JSON
  line containing `data.address` and an opaque `data.token`.
- HTTP requests use `POST` with the service JSON request as the body and either
  `Authorization: Bearer <token>` or `X-Netron-Token: <token>`.
- HTTP accepts concurrent connections. Requests for different sessions may run
  concurrently; requests sharing one session serialize on that session so
  session-local caches remain consistent.
- Active stdio and HTTP layout and slice requests can be canceled cooperatively
  with `cancel`.

## Core Surface

- `ModelSource`
  - `kind`
    - `file`
      - `path`: source file path
      - `base_dir`: parent directory used for resource resolution
    - `memory`
      - `name`: optional source label
  - `byte_len`: byte count of input blob
  - `content_identity`: optional opaque content hash/etag
  - `allow_unsafe_paths`: explicit trust flag for external-path/resource reads
- `ModelSession`
  - `open(data, source) -> ModelSession`
  - `id() -> u64`
  - `format() -> FormatIndex`
  - `summary(limits) -> SessionSummary`
  - `diagnostics(limits) -> DiagnosticsResponse`
  - `search(query, limits) -> SearchEntry[]` for legacy bounded callers
  - `search_page(query, cursor?, limits) -> SearchResponse`
  - `tensor_metadata(limits) -> TensorMetadata[]`
  - `detail(handle, limits) -> EntityDetail?`
  - `slice(handle, limits, options?) -> SliceResponse?`
  - `layout(handle, limits, options?) -> LayoutResponse?`
  - `export(limits) -> SessionExport`

## Service Methods

- `open`
- `close`
- `summary`
- `diagnostics`
- `search`
- `detail`
- `slice`
- `layout`
- `cancel`
- `export`
- `mlir.symbols`
- `mlir.symbol.tree`
- `onnx.tensor`

`cancel` accepts `{ "cancel_token": "<token>" }` plus an optional `session`.
It returns `{ "cancel_token": "<token>", "canceled": bool }` and echoes
`session` when provided. `canceled: true` means the token was active and has
been signaled; the target request may still complete normally if it already
finished before observing the signal.

`export` returns bounded normalized JSON lowered from the session-owned parsed
model. The service also echoes the transport-level `session` id for compatibility.

## Format and Handle Types

- `FormatIndex`
  - `onnx`
  - `mlir`
  - `unknown`
- Typed handles are JSON objects tagged by `kind`:
  - `{ "kind": "graph", "graph": 0 }`
  - `{ "kind": "node", "graph": 0, "node": 3 }`
  - `{ "kind": "value", "graph": 0, "value": 7 }`
  - `{ "kind": "tensor", "tensor": 2 }`
  - `{ "kind": "function", "function": 1 }`
  - `{ "kind": "onnx_repeated_block", "graph": 0, "group": 0 }`
  - `{ "kind": "metadata", "owner": "model", "key": "license" }`
  - `{ "kind": "operator_set", "domain": "ai.onnx", "version": 18 }`
  - `{ "kind": "diagnostic", "diagnostic": 0 }`
  - `{ "kind": "mlir_module", "module": 0 }`
  - `{ "kind": "mlir_function", "function": 1 }`
  - `{ "kind": "mlir_operation", "scope": "function:0", "operation": 3 }`
  - `{ "kind": "mlir_value", "scope": "function:0", "value": 7 }`
  - `{ "kind": "mlir_region", "scope": "function:0", "region": 0 }`
  - `{ "kind": "mlir_block", "scope": "function:0", "block": 0 }`
  - `{ "kind": "mlir_symbol", "symbol": 0 }`
  - `{ "kind": "mlir_dialect", "dialect": "arith" }`
  - `{ "kind": "mlir_attribute", "scope": "function:0", "attribute": 2 }`
  - `{ "kind": "mlir_resource", "resource": 0 }`

## Shared Limits

- Search
  - `max_results`
  - default: `25`
  - hard max: `500`
- Diagnostics
  - `max_entries`
  - default: `64`
  - hard max: `500`
- Layout/Slice/Preview/Detail/Export
  - defaults: `500 / 500 / 200 / 100 / 1000`
  - hard maxs: `5000 / 5000 / 5000 / 1000 / 10000`

## Projection Options

- `collapse`
  - `none`: default; return the flat bounded projection.
  - `structural`: return expandable structural groups where the format has a
    concrete hierarchy or safe repeated-block heuristic. The current
    implementation applies this to MLIR module/function/region/block scopes and
    ONNX graph repeated connected motifs up to eight operators wide, including
    linear, interleaved, order-insensitive branch/merge, and shared-boundary
    motifs. Broader ONNX repeated-pattern detection and general subgraph isomorphism are
    intentionally outside this v1 contract.
- `depth`
  - ONNX node slices only.
  - default: `1`
  - hard max: `16`
- `cancel_token`
  - Optional service parameter for cancelable requests.
  - Explicit `cancel_token` must be a non-empty string. If omitted, the service
    can use a string, number, or boolean request `id` as the cancellation key.
  - Current cooperative cancellation is implemented for stdio and HTTP layout
    and slice responses.

## Responses

- `SessionSummary`
  - `api_version`
  - `session_id`
  - `source`
  - `format`
  - `source_format_name`
  - `byte_len`
  - `graphs`, `functions`, `nodes`, `values`, `tensors`
  - `initializers`, `subgraphs`, `sparse_tensors`, `metadata`, `opsets`
  - `external_data`, `mlir_resources`
  - `onnx`: optional ONNX-specific summary block with producer fields,
    per-kind counts, opsets, metadata keys, graph summaries, and bounded
    histograms for operator type, domain, dtype, storage kind, shape rank,
    fan-in, and fan-out
  - `mlir`: optional MLIR-specific summary block with module/function/operation,
    region/block/value, symbol/dialect/attribute/resource counts and bounded
    region/block/scope/resource summaries plus operation/dialect/fan-in/fan-out
    histograms; block summaries include their parent `region` index. MLIR
    bytecode sessions include a bounded `bytecode` block with version, producer,
    section summaries, parser counts, and synthetic region/block/value topology
    where the bytecode reader can recover raw indexes.
- `DiagnosticsResponse`
  - `api_version`
  - `session_id`
  - `diagnostics` (list; format-specific diagnostics may include handles)
  - `truncated`
- `SearchEntry`
  - `kind`, `handle`, `graph`, `id`, `name`, `operator`, `origin`
- `SearchResponse`
  - `api_version`, `session_id`, `format`, `query`
  - optional numeric `cursor` and `next_cursor` offsets tied to the same
    immutable session, query, ranking, and limit
  - `limit_used`, `total_count`, `truncated`, `omitted_count`
  - `results`: bounded list of `SearchEntry`
- `SessionExport`
  - `api_version`, `session_id`, `format`
  - `limit_used`, `truncated`, `omitted_count`
  - `normalized`: normalized model object whose arrays are recursively bounded
    while lowering; omitted children are not constructed and then discarded
- `TensorMetadata`
  - `handle`, `name`, `element_type`, `shape`, `storage`
  - optional `byte_len`, `element_count`, `external_data`, sparse tensor links,
    and metadata
- `EntityDetail`
  - `handle`, `title`, bounded string `fields`, and bounded related handles;
    ONNX and MLIR details share this envelope
  - optional `locations` list for operation source metadata. Textual MLIR rows
    use `kind: "source"`, `raw`, optional `file`, and optional 1-based `line`
    and `column`, plus optional 1-based `end_line`/`end_column` when a decoded
    source range is available; bytecode rows use `kind: "bytecode"`, `raw`,
    optional raw `index`, and optional `region`/`block`. Existing metadata
    remains present in `fields` for compatibility.
- `SliceResponse`
  - `api_version`, `session_id`, `format`, `scope`, `collapse`, `cache_key`
  - `limit_used`, `truncated`, `omitted_count`, `warnings`
  - bounded `entities`, `edges`, and `boundaries`
  - optional `collapsed_groups` list with group id, kind, handle, label,
    item/omitted counts, and an `expand` handle. The `expand` handle is intended
    for a follow-up projection request using `collapse: "none"`.
- `LayoutResponse`
  - `api_version`, `session_id`, `format`, `scope`, `collapse`, `cache_key`
  - `limit_used`, `truncated`, `omitted_count`, `warnings`
  - bounded `graph` layout payload with layout stats
  - optional `collapsed_groups` list matching the slice projection contract

`cache_key` values identify session-local projection cache entries for slice and
layout responses. Different scopes, limits, collapse modes, and ONNX node depths
must produce different keys.

## Error Mapping

- Parse and validation failures use `netron-rs-core::ModelError` and map to:
  - `UnsupportedFormat`
  - `InvalidData`
  - `AccessDenied`
  - `Invariant`
- Canceled service requests return a normal error envelope with
  `error.code = "canceled"` and the original request id.
