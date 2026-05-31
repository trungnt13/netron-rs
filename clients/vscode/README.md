# Netron RS VS Code Client

This folder contains a lightweight, readonly VS Code custom editor for ONNX and MLIR
files backed by `netron-rs serve --stdio`.

- Custom editor view type: `netron.overview`
- Supported extensions: `*.onnx`, `*.mlir`, `*.mlirbc`
- Transport: persistent stdio `netron-rs serve --stdio`

## Behavior

- Files are opened through `open` first and then `summary` is requested.
- Search, detail, diagnostics, slice, layout, MLIR symbols, and ONNX tensor metadata
  are requested only on user action.
- Cancel or newer actions invalidate pending webview responses before they render.
- No `export` (full normalized JSON) request is made on initial open.

## Configuration

Set `netronRsViewer.binaryPath` if `netron-rs` is not on `PATH`:

```json
{
  "netronRsViewer.binaryPath": "/absolute/path/to/netron-rs"
}
```

## Development

Run the syntax check from `netron-rs/clients/vscode`:

```bash
npm run check
```
