# ONNX and MLIR Scalable Viewer Architecture for netron-rs

Date: 2026-05-31

Topic: Architecture design for making `netron-rs` a scalable backend for ONNX and MLIR model/IR inspection while preserving Netron-like portability through web, VS Code, CLI, and service clients.

## Executive Summary

The research should no longer be framed as "raw ONNX only." ONNX and MLIR should be treated as the first two serious indexed backends for `netron-rs`, with a shared service protocol above them and format-specific indexes below.

Recommended shape:

- `OnnxIndex`: graph/tensor/value/function/external-data index for `.onnx` and ONNX protobuf content.
- `MlirIndex`: operation/value/block/region/symbol/dialect/resource index for `.mlir` text and eventually `.mlirbc` bytecode.
- Shared `ModelSession`: one open file, one cached index, projection APIs for summary, search, detail, layout, and metadata.
- Existing `Model`: keep as the compatibility inspection/render IR and parity artifact, but do not force every UI request through full `parse -> Model -> normalized JSON`.

This makes `netron-rs` complementary to Netron's web-first viewer. Netron is easy to embed because the frontend runs in a browser-like shell, but large models suffer when the UI builds and lays out the whole graph as SVG. `netron-rs` can keep portability by serving web/VS Code clients while moving parse, indexing, search, tensor/resource metadata, and graph/region slicing into Rust.

ONNX and MLIR need different internal models. ONNX is primarily a typed tensor graph with initializers, functions, subgraphs, and external data. MLIR is a compiler IR with operations, values, blocks, regions, symbols, dialects, attributes, types, and optionally bytecode sections. A good architecture should share the viewer/service contract without pretending MLIR is just ONNX with different operator names.

The current codebase already points in this direction: ONNX has metadata-only tensor lowering, MLIR text and bytecode are registered Rust formats, focused MLIR tests exist, and indexed `ModelSession` projections now cover summaries, search, detail, slices, layout, diagnostics, and tensor/resource metadata. The immediate gap is deepening the remaining bytecode and frontend views instead of routing large-file workflows through another eager normalizer.

## Scope and Method

Scope:

- ONNX `.onnx` / protobuf model files.
- MLIR `.mlir` textual files now, with `.mlirbc` bytecode as an explicit support target.
- Model/IR inspection, graph or region browsing, search, layout, tensor/resource metadata, and parity with Netron output.
- No execution, no ONNX Runtime integration, no MLIR pass pipeline, no code generation.

Evidence used:

- Local `netron-rs` source under `/Users/trungnt13/codes/rewrite-netron/netron-rs`.
- Local upstream Netron source under `/Users/trungnt13/codes/rewrite-netron/netron`.
- Repo-local `GOALS.md` and session contract notes.
- Official ONNX IR and external data documentation.
- Official MLIR language, bytecode, and builtin dialect documentation.
- Official protobuf wire encoding documentation.
- Official VS Code webview, custom editor, extension host, and remote extension documentation.

Subagent note: one ONNX scaling pass completed in the earlier research and supported the `OnnxIndex` recommendation. A new MLIR-focused local inventory subagent was dispatched for this update, but the report was completed from direct source inspection before its result arrived. Several earlier subagents returned only workspace-instruction acknowledgements, so the final synthesis relies primarily on direct code reading and primary docs.

## Key Findings

### 1. The existing parse boundary is still eager

The core parser trait exposes `parse(input) -> Result<Model, ModelError>`, so every format adapter is currently shaped around materializing the canonical `Model` before clients can ask summary questions. See [format.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/format.rs:25).

The registry includes both ONNX and MLIR: `MLIR` is registered at [lib.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/lib.rs:29), ONNX at [lib.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/lib.rs:32), and both are included in the format list at [lib.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/lib.rs:46).

The scalable viewer should keep `parse -> Model` for compatibility and parity, but add indexed open paths:

- `open_onnx_index(input) -> OnnxIndex`
- `open_mlir_index(input) -> MlirIndex`
- `session.open(input) -> SessionSummary`

### 2. ONNX already has strong metadata-first foundations

ONNX detection and parsing currently decode ONNX protobuf variants and lower a `ModelProto`, `GraphProto`, or `TensorProto` into `Model`. See [onnx.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:23) and [onnx.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:57).

The current ONNX lowering builds model metadata, opsets, functions, graph initializers, values, nodes, producer/consumer links, and subgraphs. ONNX graph attributes are lowered as subgraphs. See [onnx.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:91) and [onnx.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:1440).

Tensor lowering is already metadata-oriented. It records external data descriptors, inline byte length, element-list length, sparse structure, or absent payload rather than eagerly materializing tensor bytes. See [onnx.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:1460) and [model.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/model.rs:748).

This makes ONNX the best candidate for the first production-grade summary-first index.

### 3. MLIR support exists, but it is currently a pragmatic textual adapter

`MlirFormat` supports textual `.mlir` and MLIR bytecode `.mlirbc` inputs. Text detection samples the first 64 KiB and looks for module/function/tensor/memref/dialect markers; bytecode detection recognizes the MLIR bytecode magic and records a bounded structured bytecode summary.

The adapter lowers parsed modules into `Model.graphs` and parsed functions into `Model.functions`. See [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:104).

The parser handles MLIR text with modules, aliases, function capture, top-level module operations, operation parsing, attributes, dense constants, call references, block arguments, and type parsing. Representative paths are [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:420), [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:795), [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:1085), and [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:2648).

Current project goals explicitly keep the MLIR adapter pragmatic: the backend should inspect ONNX and MLIR without running MLIR passes, converting frameworks, or requiring full-IR serialization/layout on open.

### 4. MLIR is already an active support priority in this repo

`GOALS.md` defines the project around ONNX and MLIR only: indexed `ModelSession` views, bounded summary/search/detail/tensor-resource/slice/layout projections, and `parse -> Model` retained only for compatibility, golden parity, and small exports.

This makes MLIR a first-class indexed backend beside ONNX, including text support today and bytecode as the remaining deeper-indexing target.

### 5. MLIR bytecode is a gap relative to Netron

Upstream Netron registers MLIR for `.mlir`, `.mlir.txt`, `.mlirbc`, and `.txt`. See [view.js](/Users/trungnt13/codes/rewrite-netron/netron/source/view.js:7296).

Netron's MLIR loader recognizes the MLIR bytecode magic signature, parses textual MLIR with a parser, and parses bytecode with a bytecode reader. See [mlir.js](/Users/trungnt13/codes/rewrite-netron/netron/source/mlir.js:10) and [mlir.js](/Users/trungnt13/codes/rewrite-netron/netron/source/mlir.js:56).

By contrast, current `netron-rs` MLIR support is `.mlir` text only by extension and UTF-8 parsing. See [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:18) and [mlir.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:47).

For ONNX + MLIR support, `.mlirbc` should be planned explicitly, but it should not block a text-first `MlirIndex`.

### 6. Current normalized JSON is useful for parity, not for large-model UI

The normalized JSON path serializes all graphs, functions, and tensors, and each graph contains full values and nodes. See [normalize.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/normalize.rs:18) and [normalize.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/normalize.rs:101).

This is valuable for golden parity against Netron and for small artifacts, but it should not be the frontend protocol for large ONNX or MLIR files. The frontend should request projections: summaries, search results, selected details, graph/region slices, layout responses, and tensor/resource metadata.

### 7. Netron's portability lesson still applies

Netron's graph update path builds a full `view.Graph`, adds the active graph, creates SVG elements, measures nodes, lays out the graph, and restores the canvas. See [view.js](/Users/trungnt13/codes/rewrite-netron/netron/source/view.js:451).

The SVG canvas is built in the browser, all values and graph content are built into it, measurement waits for fonts and animation frames, and the full canvas uses `getBBox()` for sizing. See [view.js](/Users/trungnt13/codes/rewrite-netron/netron/source/view.js:2319).

The grapher builds SVG nodes/edges for all entries and constructs full `nodes`/`edges` arrays for Dagre, with a large-graph warning path. See [grapher.js](/Users/trungnt13/codes/rewrite-netron/netron/source/grapher.js:166) and [grapher.js](/Users/trungnt13/codes/rewrite-netron/netron/source/grapher.js:224).

For ONNX and MLIR, `netron-rs` should keep Netron's portability boundary but move scale-sensitive parsing, indexing, search, slicing, and layout into Rust.

## Format Constraints

### ONNX constraints

Official ONNX IR documentation describes a model as metadata plus a graph, where graphs contain inputs, outputs, nodes, initializers, optional value information, and related metadata. Source: [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html).

ONNX has nested graph-valued attributes for control-flow style operators; current `netron-rs` already records ONNX graph attributes as subgraphs. See [onnx.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:1440).

ONNX external data lets large tensors live outside the `.onnx` protobuf. The viewer must preserve external-data descriptors and avoid loading payload bytes unless requested. Source: [ONNX External Data](https://onnx.ai/onnx/repo-docs/ExternalData.html).

Protocol buffers encode embedded messages as length-delimited records, and repeated fields can appear multiple times. Source: [Protocol Buffers Encoding](https://protobuf.dev/programming-guides/encoding/). This makes a streaming ONNX index plausible later, but cross-reference tables are still required for producers, consumers, functions, graph boundaries, and initializers.

### MLIR constraints

Official MLIR docs define MLIR as operations and values, with operations contained in blocks and blocks contained in regions. Operations may themselves contain regions, which enables hierarchy. Source: [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/).

MLIR operations are ordered within blocks, blocks are ordered within regions, and values are produced by exactly one operation or block argument. Regions can have SSACFG or graph-like semantics, and graph regions treat operations as nodes and values as multi-edges. Source: [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/).

MLIR has an open dialect system. Dialects define operations, attributes, and types, and multiple dialects can coexist in a single module. Source: [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/).

The builtin `module` operation is a top-level container with a single graph region and a single block; it can contain any operations, has no terminator, and is isolated from above. Source: [MLIR Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/).

MLIR bytecode has a four-byte magic number `4D 4C EF 52`, then version, producer string, and sections. The bytecode format includes string, dialect, attribute/type, resource, and IR sections. It also states that sections support delayed processing, lazy-loading, and out-of-order processing. Source: [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/).

Implication: MLIR support should not be graph-only. A viewer backend needs operation, value, block, region, symbol, dialect, attribute/type, and resource summaries.

## Recommended Architecture

### Layer 1: Shared session

Introduce a shared indexed session abstraction:

```rust
pub enum FormatIndex {
    Onnx(OnnxIndex),
    Mlir(MlirIndex),
}

pub struct ModelSession {
    id: SessionId,
    source: ModelSource,
    index: FormatIndex,
    search_cache: SearchCache,
    layout_cache: LayoutCache,
}
```

`ModelSource` should preserve:

- memory-mapped file or byte buffer;
- canonical file path when available;
- base directory for ONNX external data and MLIR resource references;
- file size and content hash or mtime stamp;
- trust/security flags for local, remote, uploaded, or virtual files.

The existing CLI already mmap-opens local files before parsing; preserve that for local sessions. See [main.rs](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-cli/src/main.rs:167).

### Layer 2: OnnxIndex

`OnnxIndex` should answer ONNX-specific discovery questions without lowering or serializing the full model:

```rust
pub struct OnnxIndex {
    header: OnnxModelHeader,
    graphs: Vec<OnnxGraphSummary>,
    functions: Vec<OnnxFunctionSummary>,
    nodes: OnnxNodeTable,
    values: OnnxValueTable,
    tensors: OnnxTensorTable,
    strings: StringInterner,
    diagnostics: Vec<Diagnostic>,
}
```

It should expose:

- model metadata: IR version, producer, domain, model version, opsets, metadata props;
- graph summaries: parent graph, name, inputs, outputs, node/value/tensor counts, subgraphs;
- node summaries: stable id, graph id, domain, op type, name, inputs, outputs, lightweight attributes;
- value summaries: name, type/shape, producer, consumer count, initializer, graph boundary flags;
- tensor summaries: dtype, shape, storage kind, inline byte length, element count, external-data descriptor, sparse links;
- function summaries: domain, name, overload, input/output count, node count;
- histograms: operator, domain, dtype, tensor storage, shape rank, fan-in/fan-out;
- diagnostics: duplicate names, unresolved references, malformed tensor metadata, missing external data.

First implementation can build `OnnxIndex` from decoded protobuf if that is fastest to ship. The API should still be index/projection-first, so a later streaming protobuf scanner can replace the internals.

### Layer 3: MlirIndex

`MlirIndex` should be the MLIR equivalent, but it should reflect MLIR structure instead of copying ONNX names:

```rust
pub struct MlirIndex {
    header: MlirHeader,
    modules: Vec<MlirModuleSummary>,
    functions: Vec<MlirFunctionSummary>,
    operations: MlirOperationTable,
    values: MlirValueTable,
    blocks: MlirBlockTable,
    regions: MlirRegionTable,
    symbols: MlirSymbolTable,
    dialects: MlirDialectTable,
    resources: MlirResourceTable,
    strings: StringInterner,
    diagnostics: Vec<Diagnostic>,
}
```

It should expose:

- file kind: text or bytecode;
- bytecode version and producer when available;
- module tree and symbol paths;
- function summaries with visibility, inputs, outputs, block count, op count;
- operation summaries with dialect, op name, result ids, operand ids, attribute keys, location, regions;
- block summaries with arguments, terminator, predecessor/successor hints where parseable;
- region summaries with kind if known or inferred, entry block, block count, op count;
- value summaries with type text, parsed tensor/memref/vector shape when available, defining op or block argument, users;
- dialect inventory and op histograms;
- attribute/type/resource summaries, including dense resource metadata without eager payload materialization;
- diagnostics for unsupported syntax, unresolved symbols, malformed regions, and fallback parsing.

Text-first `MlirIndex` can start from the existing parser. Bytecode support should become a second parser path that fills the same index shape.

### Layer 4: Controlled lowering to Model

Keep `Model` as the normalized compatibility IR:

- `parse -> Model` remains for current format registry behavior.
- `to_normalized_json()` remains for small models, golden parity, CLI debugging, and exported artifacts.
- New viewer clients use `FormatIndex` projections.

Both `OnnxIndex` and `MlirIndex` should provide bounded lowering:

```rust
pub trait IndexedFormat {
    fn summary(&self) -> ModelSummary;
    fn search(&self, request: SearchRequest) -> SearchResponse;
    fn detail(&self, handle: EntityHandle) -> DetailResponse;
    fn slice(&self, request: SliceRequest) -> SliceResponse;
    fn layout(&self, request: LayoutRequest) -> LayoutResponse;
    fn lower_to_model(&self, limits: LowerLimits) -> Result<Model>;
}
```

The design point is consistent client behavior, not identical internals.

### Layer 5: Search

Build search directly on each index:

- ONNX search: graph names, node names, op types, domains, value names, tensor names, function names, metadata keys, external-data locations.
- MLIR search: module names, symbol names, function names, operation names, dialect names, SSA values, block labels, locations, attribute keys, type strings, resource names.

Return typed handles:

```json
{
  "results": [
    { "format": "mlir", "kind": "operation", "id": 1842, "name": "stablehlo.dot_general", "scope": "@main" },
    { "format": "onnx", "kind": "node", "graph": 0, "id": 955, "operator": "MatMul" }
  ],
  "next_cursor": null,
  "omitted": 0
}
```

### Layer 6: Slices and layout

Layout should be query-driven for both formats.

ONNX slice examples:

- selected node neighborhood;
- selected value producer/consumer fanout;
- graph input-to-output boundary path;
- function body;
- collapsed repeated block.

MLIR slice examples:

- selected operation neighborhood within a block or graph region;
- selected function body;
- selected region under an operation;
- block-level SSACFG view;
- module tree and symbol graph;
- dialect-filtered operation view.

Collapse modes should distinguish concrete structural grouping from heuristics.
`collapse:structural` can rely on MLIR modules, functions, regions, and blocks;
ONNX repeated-block collapse needs a separate pattern detector.

The layout response should always include:

- visible nodes/ops/blocks;
- visible edges or value flows;
- boundary markers for omitted upstream/downstream/parent/child content;
- `omitted_nodes`, `omitted_edges`, `omitted_regions`, or `omitted_blocks`;
- warnings for high-degree values, cyclic graph regions, huge dense constants, and incomplete parsing.

### Layer 7: Rendering

Use the same frontend rendering strategy for ONNX and MLIR:

- Overview first, no full graph layout on open.
- Canvas/WebGL for large graph or region surfaces.
- SVG/HTML only for selected details, small focused views, sidebars, and tooltips.
- Virtualized lists for search, operations, values, tensors, attributes, resources, and diagnostics.
- Collapse by default when repeated ONNX blocks or MLIR module/function/region hierarchies are present.

MLIR should get extra views that ONNX does not need:

- module/symbol tree;
- region/block navigation;
- operation location/source table with editor jumps for textual source coordinates;
- deeper bytecode operation/value/region/block detail beyond the overview
  section/resource tables.

## Service and CLI Design

Add a backend crate such as `netron-rs-service` or `netron-rs-server`.

Recommended transports:

- `serve --stdio`: default for VS Code and desktop integrations.
- `serve --http 127.0.0.1:0`: browser/web app and debug mode, with random local port and session token.
- `serve --http --listen ...`: remote/team mode only when explicitly configured.

Recommended methods:

```text
session.open(path | bytes_handle) -> session_id, ModelSummary
session.close(session_id)
model.summary(session_id)
model.diagnostics(session_id)
search(session_id, query, scope, limit, cursor)
entity.detail(session_id, handle)
slice(session_id, request)
layout(session_id, request)
export.normalized(session_id, limits?)

onnx.graph.list(session_id)
onnx.graph.summary(session_id, graph_id)
onnx.tensor.metadata(session_id, tensor_id)
# optional future payload path, outside the current metadata-only GOALS.md scope:
onnx.tensor.preview(session_id, tensor_id, slice, max_bytes)

mlir.module.list(session_id)
mlir.symbol.tree(session_id)
mlir.function.summary(session_id, function_id)
mlir.region.summary(session_id, region_id)
mlir.block.summary(session_id, block_id)
mlir.resource.metadata(session_id, resource_id)
```

CLI should become a client of the same operations:

```text
netron-rs summary model.onnx
netron-rs summary module.mlir
netron-rs search model.onnx MatMul --limit 50
netron-rs search module.mlir stablehlo.dot_general --limit 50
netron-rs layout model.onnx --node 1842 --depth 2 --max-nodes 500
netron-rs layout module.mlir --function @main --region 0 --max-ops 500
netron-rs mlir symbols module.mlir
netron-rs onnx tensor model.onnx --tensor 42
netron-rs serve --stdio
```

## VS Code and Web Deployment

VS Code custom editors are designed for alternative views of resources, and readonly custom editors fit `.onnx`, `.mlir`, and `.mlirbc` previews. Source: [VS Code Custom Editor API](https://code.visualstudio.com/api/extension-guides/custom-editors).

Webviews communicate with the extension through message passing and require CSP/resource care. Source: [VS Code Webview API](https://code.visualstudio.com/api/extension-guides/webview).

VS Code has local, web, and remote extension hosts. Workspace extensions run where the workspace is located, which is important for large local/remote ONNX files, external tensor data, MLIR files, and MLIR bytecode. Source: [VS Code Extension Host](https://code.visualstudio.com/api/advanced-topics/extension-host) and [Remote Extensions](https://code.visualstudio.com/api/advanced-topics/remote-extensions).

Recommended extension architecture:

1. Register custom readonly editors for `.onnx`, `.mlir`, and `.mlirbc`.
2. Start `netron-rs serve --stdio` in the workspace extension host.
3. Use `session.open` to create an indexed session near the file.
4. Render the webview with only the model/IR summary.
5. Use message passing for search, detail, slice, layout, tensor metadata, and MLIR region navigation.
6. Use direct HTTP only as a controlled optimization, not the default assumption.

## Implementation Milestones

### M0: Stabilize current MLIR baseline

- Keep repo-local `GOALS.md`, this architecture note, and
  `goal-3-session-contract.md` aligned with the implemented session contract.
- Run `cargo test --workspace` and `cargo fmt --all -- --check` after indexed
  session changes.
- Keep focused MLIR parser/bytecode fixture tests for corpus regressions.
- Treat source links in this document as orientation aids; current build/test
  status is established by validation, not by older research-line references.

### M1: Shared index/session protocol

- Define `ModelSession`, `FormatIndex`, typed handles, common summary, search, detail, slice, layout, and diagnostics schemas.
- Keep existing `parse -> Model` behavior unchanged.
- Add CLI summary commands that use the new index path.

### M2: OnnxIndex production path

- Build `OnnxIndex` from the existing ONNX decode path.
- Expose model summary, graph summary, search, tensor metadata, diagnostics, and selected node/value details.
- Preserve external-data descriptors and never read tensor payloads on open.

### M3: MlirIndex text path

- Build `MlirIndex` from the existing textual parser.
- Index modules, functions, operations, SSA values, block arguments, regions, symbols, dialects, attributes, types, locations, and dense/resource metadata.
- Use the current MLIR parity corpus as regression coverage.

### M4: Bounded slices and layout

- Add ONNX graph neighborhood layout.
- Add MLIR function/region/block layout.
- Return omitted-count statistics and boundary markers.
- Cache layout by session id, format, scope, collapse state, and limits.
- Support cooperative cancellation for long-running layout requests. The current
  implementation covers active HTTP layout requests; stdio preemption and slice
  cancellation require additional transport/projection work.

### M5: MLIR bytecode

- Detect `.mlirbc` by magic bytes `4D 4C EF 52`.
- Parse bytecode header, dialect section, string section, attr/type/resource metadata, and IR summaries into `MlirIndex`.
- Preserve lazy resource handling.
- Defer full dialect-specific decoding where necessary, but expose diagnostics.

### M6: Frontend integration

- Build overview-first web UI.
- Use canvas/WebGL for large surfaces.
- Add ONNX graph/tensor views.
- Add MLIR module/symbol tree, region/block navigation, operation detail views,
  and source-location rows.
- Integrate through VS Code custom readonly editors.

## Validation Plan

ONNX validation:

- Small ONNX parity fixtures.
- Medium transformer/CNN models with thousands of nodes.
- Large ONNX models with external data.
- Current repo-local smoke coverage includes `tests/fixtures/onnx/external-chain.onnx`,
  a 1,024-node graph with a 1 MiB external tensor payload; broader real-model
  coverage is still required for full validation.
- Metrics: open-to-summary latency, peak RSS, summary response size, search latency, tensor metadata latency, layout-slice latency.

MLIR validation:

- Current 10-file MLIR corpus slice from `GOALS.md`.
- Additional textual MLIR fixtures for nested modules, generic `"func.func"` syntax, SSACFG regions, graph regions, dense constants, resource-like dense payloads, and ONNX dialect constants.
- Later `.mlirbc` fixtures for header/section parsing and resource metadata.
- Metrics: open-to-summary latency, operation count, region/block count, histogram latency, search latency, and layout latency.

Frontend validation:

- Opening ONNX or MLIR should not transfer full normalized JSON.
- Opening should not create full SVG DOM for every node/op.
- Large files should land on summary/overview first.
- Search and detail views should remain responsive with pagination.
- Layout should be bounded and cancellable. Current HTTP layout requests are
  cooperatively cancellable; stdio and slice cancellation remain follow-up work
  if those paths need preemption.

Parity validation:

- Keep normalized JSON golden diff for small and medium corpus files.
- Do not require large-file frontend protocol to match normalized JSON shape.
- Treat parity as "same inspectable meaning" for MLIR while keeping large-file
  frontend protocol bounded and indexed, matching the repo-local `GOALS.md`.

## Risks and Open Questions

ONNX full protobuf decode may still be the bottleneck for huge models. The first index builder can be eager internally, but the service API should leave room for a streaming protobuf scanner.

MLIR textual parsing is intentionally pragmatic. It can be fast and corpus-driven, but complex nested regions, generic operation syntax, symbol scoping, and dialect-specific custom assembly will keep producing edge cases unless the index model explicitly represents regions and symbols.

MLIR bytecode support is currently a gap relative to Netron. The bytecode format is friendly to lazy section processing, but dialect-specific attribute/type encodings can require dialect-specific handling.

ONNX block collapsing is heuristic because raw ONNX has no mandatory module hierarchy. MLIR has stronger structural hierarchy through modules, functions, regions, blocks, and symbols, so its collapse model should rely on those first.

VS Code webview networking differs across desktop, remote, and browser-based environments. Message passing through the extension should be the default; direct localhost HTTP should be optional.

External ONNX data and MLIR resources are security boundaries. The backend must not become a generic filesystem reader through malicious model metadata.

## Confidence Assessment

High confidence:

- ONNX and MLIR are both registered in `netron-rs` today.
- Current ONNX lowering is metadata-first for tensor payloads.
- Current MLIR support is textual `.mlir` only, while Netron also supports `.mlirbc`.
- The existing normalized JSON path is whole-model and should not be the large-file UI protocol.
- Netron's current graph viewer builds and lays out a full SVG graph.

Medium confidence:

- `OnnxIndex` built from decoded protobuf will deliver a useful first large-model win before a streaming parser is necessary.
- `MlirIndex` built from the existing textual parser will be sufficient for an overview/search/detail first UI if region and symbol summaries are added.
- A shared service API can support both formats without forcing a shared internal graph model.

Low confidence until measured:

- Exact open-to-summary latency for very large ONNX external-data models.
- Exact open-to-summary latency for very large MLIR modules or bytecode files.
- Whether the current pragmatic MLIR parser can support broad dialect coverage without eventually needing a richer parser or bytecode reader.

## Source Map

Local code:

- [core format trait](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/format.rs:25)
- [format registry with ONNX and MLIR](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/lib.rs:46)
- [ONNX detection and parse path](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:23)
- [ONNX model lowering](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:91)
- [ONNX subgraph lowering](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:1440)
- [ONNX tensor lowering](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/onnx.rs:1460)
- [Model IR](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/model.rs:5)
- [Value and Tensor IR](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/model.rs:607)
- [normalized JSON](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-core/src/normalize.rs:18)
- [CLI commands](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-cli/src/main.rs:19)
- [query index](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-query/src/lib.rs:31)
- [layout options](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-layout/src/lib.rs:13)
- [MLIR format adapter](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:14)
- [MLIR text parser](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:420)
- [MLIR operation parsing](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:1085)
- [MLIR type parsing](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/src/mlir.rs:2648)
- [MLIR focused tests](/Users/trungnt13/codes/rewrite-netron/netron-rs/crates/netron-rs-formats/tests/mlir.rs:4)
- [project goal](/Users/trungnt13/codes/rewrite-netron/netron-rs/GOALS.md:1)
- [session contract](/Users/trungnt13/codes/rewrite-netron/netron-rs/goal-3-session-contract.md:1)
- [Netron MLIR registration](/Users/trungnt13/codes/rewrite-netron/netron/source/view.js:7296)
- [Netron MLIR text and bytecode loader](/Users/trungnt13/codes/rewrite-netron/netron/source/mlir.js:10)
- [Netron graph update pipeline](/Users/trungnt13/codes/rewrite-netron/netron/source/view.js:451)
- [Netron SVG canvas build/measure/restore](/Users/trungnt13/codes/rewrite-netron/netron/source/view.js:2319)
- [Netron grapher SVG build/layout](/Users/trungnt13/codes/rewrite-netron/netron/source/grapher.js:166)

External sources:

- [ONNX IR Specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX External Data](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [Protocol Buffers Encoding](https://protobuf.dev/programming-guides/encoding/)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/)
- [MLIR Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/)
- [VS Code Webview API](https://code.visualstudio.com/api/extension-guides/webview)
- [VS Code Custom Editor API](https://code.visualstudio.com/api/extension-guides/custom-editors)
- [VS Code Extension Host](https://code.visualstudio.com/api/advanced-topics/extension-host)
- [VS Code Remote Extensions](https://code.visualstudio.com/api/advanced-topics/remote-extensions)
