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
