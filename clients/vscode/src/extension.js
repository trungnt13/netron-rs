const path = require('node:path');
const vscode = require('vscode');

const { NetronServiceClient } = require('./netronService');

const VIEW_TYPE = 'netron.overview';

function activate(context) {
  const service = new NetronServiceClient(context, () =>
    vscode.workspace.getConfiguration('netronRsViewer').get('binaryPath', ''),
  );
  context.subscriptions.push(service);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      VIEW_TYPE,
      new NetronOverviewProvider(context, service),
      {
        supportsMultipleEditorsPerDocument: false,
        webviewOptions: { retainContextWhenHidden: true },
      },
    ),
  );
}

function deactivate() {}

class NetronOverviewProvider {
  constructor(context, service) {
    this.context = context;
    this.service = service;
  }

  async openCustomDocument(uri) {
    return {
      uri,
      dispose() {},
    };
  }

  async resolveCustomEditor(document, panel) {
    const mediaRoot = vscode.Uri.file(path.join(this.context.extensionPath, '..', 'web'));
    panel.webview.options = {
      enableScripts: true,
      localResourceRoots: [mediaRoot],
    };
    panel.webview.html = this.html(panel.webview, mediaRoot);
    const opened = this.service.open(document.uri.fsPath);

    panel.webview.onDidReceiveMessage(async (message) => {
      const openedData = await opened;
      if (message?.type === 'ready') {
        panel.webview.postMessage({
          type: 'ready',
          response: ok('open', openedData),
        });
        return;
      }
      if (message?.type === 'openLocation') {
        await this.openLocation(document, message.location);
        return;
      }
      if (!message || typeof message.method !== 'string') {
        return;
      }
      const params = { ...(message.params || {}) };
      if (!Object.prototype.hasOwnProperty.call(params, 'session')) {
        params.session = openedData.session;
      }
      try {
        const data = await this.service.request(message.method, params);
        panel.webview.postMessage({ id: message.id, response: ok(message.method, data) });
      } catch (error) {
        panel.webview.postMessage({
          id: message.id,
          response: fail(message.method, error),
        });
      }
    });

    panel.onDidDispose(() => {
      opened
        .then((data) => this.service.close(data.session))
        .catch(() => {});
    });
  }

  async openLocation(document, location) {
    if (!location || !Number.isInteger(location.line) || location.line < 1) {
      return;
    }
    const targetUri = this.locationUri(document, location.file);
    try {
      const targetDocument = await vscode.workspace.openTextDocument(targetUri);
      const editor = await vscode.window.showTextDocument(targetDocument, { preview: true });
      const line = Math.min(Math.max(location.line - 1, 0), Math.max(targetDocument.lineCount - 1, 0));
      const rawColumn = Number.isInteger(location.column) && location.column > 0 ? location.column - 1 : 0;
      const column = Math.min(Math.max(rawColumn, 0), targetDocument.lineAt(line).text.length);
      const position = new vscode.Position(line, column);
      const end = this.locationEndPosition(targetDocument, location, position);
      editor.selection = new vscode.Selection(position, end);
      editor.revealRange(new vscode.Range(position, end), vscode.TextEditorRevealType.InCenter);
    } catch (error) {
      const label = location.file || path.basename(document.uri.fsPath);
      vscode.window.showWarningMessage(`Unable to open MLIR location ${label}:${location.line}: ${error.message}`);
    }
  }

  locationUri(document, file) {
    if (typeof file !== 'string' || file.length === 0) {
      return document.uri;
    }
    if (path.isAbsolute(file)) {
      return vscode.Uri.file(file);
    }
    return vscode.Uri.file(path.join(path.dirname(document.uri.fsPath), file));
  }

  locationEndPosition(document, location, start) {
    const hasEndLine = Number.isInteger(location.end_line) && location.end_line > 0;
    const hasEndColumn = Number.isInteger(location.end_column) && location.end_column > 0;
    if (!hasEndLine && !hasEndColumn) {
      return start;
    }
    const line = hasEndLine
      ? Math.min(Math.max(location.end_line - 1, 0), Math.max(document.lineCount - 1, 0))
      : start.line;
    const rawColumn = hasEndColumn ? location.end_column - 1 : 0;
    const column = Math.min(Math.max(rawColumn, 0), document.lineAt(line).text.length);
    const end = new vscode.Position(line, column);
    return end.isBefore(start) ? start : end;
  }

  html(webview, mediaRoot) {
    const nonce = makeNonce();
    const style = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, 'styles.css'));
    const script = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, 'viewer.mjs'));
    const csp = [
      "default-src 'none'",
      `style-src ${webview.cspSource}`,
      `script-src 'nonce-${nonce}' ${webview.cspSource}`,
      `img-src ${webview.cspSource} data:`,
    ].join('; ');
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="${style}">
<title>netron-rs</title>
</head>
<body>
<main class="shell">
<header class="topbar">
<div><h1>netron-rs</h1><p id="status">Opening indexed session</p></div>
<div class="actions"><button id="diagnostics-button" type="button">Diagnostics</button><button id="symbols-button" type="button">Symbols</button><button id="tensors-button" type="button">Tensors</button><button id="layout-button" type="button">Layout</button><button id="cancel-button" type="button">Cancel</button></div>
</header>
<section id="overview" class="overview" aria-label="Overview"></section>
<section class="workbench">
<aside class="panel"><label class="search"><span>Search</span><input id="search" type="search" autocomplete="off"></label><div id="results" class="virtual-list" aria-label="Search results"></div></aside>
<section class="panel main-panel"><canvas id="layout" width="960" height="540" aria-label="Bounded layout"></canvas></section>
<aside class="panel"><div id="detail" class="detail" aria-label="Detail"></div><h2>Metadata</h2><div id="metadata" class="virtual-list small" aria-label="Metadata"></div><h2>Diagnostics</h2><div id="diagnostics" class="virtual-list small" aria-label="Diagnostics"></div></aside>
</section>
<section id="lists" class="lists" aria-label="Indexed lists"></section>
<footer><code id="request-log"></code></footer>
</main>
<script nonce="${nonce}" type="module" src="${script}"></script>
</body>
</html>`;
  }
}

function ok(command, data) {
  return { schema_version: 1, status: 'ok', command, data };
}

function fail(command, error) {
  return {
    schema_version: 1,
    status: 'error',
    command,
    error: {
      code: 'internal_error',
      message: error.message || String(error),
    },
  };
}

function makeNonce() {
  return Array.from({ length: 24 }, () => Math.floor(Math.random() * 36).toString(36)).join('');
}

module.exports = { activate, deactivate };
