# netron-rs Web Assets

This folder contains the dependency-free webview UI shared by the standalone
demo page and the VS Code custom editor.

The UI is overview-first:

- initial load receives only the service `open` response with session id and
  summary;
- search, detail, diagnostics, slice, layout, MLIR symbols, and ONNX tensor
  metadata are requested after user actions;
- the request log is visible so smoke tests can prove no `export` request happens
  before an explicit export workflow exists.

`index.html` can be opened directly for a demo transport. In VS Code, the
extension injects the same `styles.css` and `viewer.mjs` files into a webview and
proxies messages to `netron-rs serve --stdio`.
