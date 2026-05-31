# Goal 3 Session Contract (v1)

This document defines the shared indexed-session API contract used by CLI, service,
and future web/VS Code clients.

## Version

- `session_api_version`: `1`
- Stable JSON objects use snake_case keys.

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

## Responses

- `SessionSummary`
  - `api_version`
  - `session_id`
  - `source`
  - `format`
  - `source_format_name`
  - `byte_len`
  - `graphs`, `functions`, `nodes`, `values`, `tensors`
  - `external_data`, `mlir_resources`
- `DiagnosticsResponse`
  - `api_version`
  - `session_id`
  - `diagnostics` (list)
  - `truncated`

## Error Mapping

- Parse and validation failures use `netron-rs-core::ModelError` and map to:
  - `UnsupportedFormat`
  - `InvalidData`
  - `AccessDenied`
  - `Invariant`
