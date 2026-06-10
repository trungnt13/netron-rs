# netron-rs

Rust workspace for parsing, normalizing, querying, and laying out neural-network model graphs.

This repository is an early Rust implementation inspired by Netron's model inspection workflows. It currently contains:

- `netron-rs-core`: shared model graph data structures and normalization.
- `netron-rs-formats`: parsers and lowerers for supported model formats.
- `netron-rs-layout`: graph layout utilities.
- `netron-rs-query`: search utilities for normalized graph data.
- `netron-rs`: command-line interface.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --locked
node tools/fixture-smoke.mjs --output artifacts/ci/fixture-smoke.json
```

`fixture-smoke.mjs` always runs the repo-local CI fixtures by default. The
checked-in `.mlirbc` fixtures also require every operation with a bytecode
property blob to expose decoded `bytecode.property.*` metadata in the parsed
model, and every custom bytecode attribute/type entry exposed by `summary
--json` to expose decoded assembly. To add larger local corpus anchors without
making CI depend on private files, pass one or more manifests:

```sh
node tools/fixture-smoke.mjs --manifest /path/to/local-corpus.json
```

Manifest paths are resolved relative to the manifest file, unless absolute:

```json
{
  "fixtures": [
    {
      "name": "resnet18-v2-7",
      "path": "/path/to/resnet18-v2-7.onnx",
      "format": "ONNX",
      "minBytes": 40000000,
      "minNodes": 60,
      "minTensors": 90,
      "maxSummaryJsonBytes": 4096,
      "requireDecodedBytecodeProperties": false,
      "requireDecodedBytecodeSummaryEntries": false
    }
  ]
}
```

The workspace requires the Rust toolchain declared in `rust-toolchain.toml`.

Install the Git pre-commit hook locally with:

```sh
pre-commit install
```

Run the hook suite manually with:

```sh
pre-commit run --all-files
```
