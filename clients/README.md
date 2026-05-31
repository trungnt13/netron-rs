# netron-rs Clients

The clients use the indexed service contract instead of normalized whole-model
JSON on open.

- `web/` contains dependency-free webview assets. The first payload is the
  `open` response with a session id and summary. Search, detail, diagnostics,
  slice, layout, MLIR symbols, and ONNX tensor metadata are requested only after
  user actions.
- `vscode/` registers a readonly custom editor for `.onnx`, `.mlir`, and
  `.mlirbc`. The extension host starts `netron-rs serve --stdio` and proxies
  webview messages to the JSON Lines service.

Validation:

```sh
npm --prefix clients/vscode run check
```
