# netron-rs Agent Instructions

This file applies to the whole repository.

## Project Mission

`netron-rs` is a Rust workspace for inspecting neural-network model graphs. The
active scope is ONNX and MLIR only. Keep the backend small, fast, and
inspection-oriented.

The intended product shape is an indexed Rust backend serving bounded
projections to CLI, web, and VS Code clients: summary, diagnostics, search,
detail, tensor/resource metadata, graph/region slices, and layout. Normalized
JSON remains useful for compatibility, golden parity, debugging, and explicit
small exports, not as the initial frontend payload.

## Current Architecture

- `crates/netron-rs-core`: canonical IR (`Model`, `Graph`, `Node`, `Value`,
  `Tensor`), validation, string interning, IDs, errors, and normalized JSON.
- `crates/netron-rs-formats`: format detection, parsing, lowering, archive
  handling, ONNX external-data safety, and MLIR resource safety. Current
  registered formats are ONNX and MLIR, with limited `.onnx.zip` wrapper
  support.
- `crates/netron-rs-query`: `ModelSession`, `FormatIndex`, typed handles,
  summaries, diagnostics, search, detail, slice/layout projections, MLIR symbol
  APIs, ONNX tensor metadata, limits, and projection caches.
- `crates/netron-rs-layout`: bounded graph layout and cooperative cancellation.
- `crates/netron-rs-cli`: command surface plus `serve --stdio` and `serve
  --http` transports.
- `clients/web`: dependency-free shared webview UI and service protocol client.
- `clients/vscode`: readonly VS Code custom editor that starts
  `netron-rs serve --stdio` and proxies webview requests.
- `tools`: fixture smoke, parity, corpus, and performance helper scripts.

Important current caveat: the client protocol is projection-oriented, but
`ModelSession::open()` still parses the full file into `Model`, then builds a
`FormatIndex`, and stores both. Do not claim the backend is fully lazy or
streaming unless you implement and verify that change.

## Invariants

- Keep ONNX and MLIR first-class but distinct. ONNX is graph/tensor/function
  oriented; MLIR is operation/value/block/region/symbol/dialect/resource
  oriented. Do not force MLIR concepts into ONNX-only terminology.
- Preserve `Model::validate()` invariants whenever editing lowering, IDs,
  producer/consumer links, subgraphs, functions, tensors, or string IDs.
- Keep service JSON stable, snake_case, and `session_api_version` compatible
  unless deliberately changing the contract.
- Respect `SessionLimits` and hard caps. New summary/detail/layout/search fields
  must remain bounded.
- Do not make initial client open request `export` or full normalized JSON.
  Clients should open once, render the returned summary, then request search,
  detail, diagnostics, symbols, tensors, slices, and layout on user action.
- Preserve cancellation behavior for expensive `slice` and `layout` work.
- Keep `parse -> Model` and `to_normalized_json()` behavior available unless
  the user explicitly removes compatibility paths.
- Do not add broad dependencies or generated code without a clear benefit.

## Development Commands

Default validation for Rust changes:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --locked
```
