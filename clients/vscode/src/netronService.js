const cp = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');

const DEFAULT_REQUEST_LIMIT = 5000;

class NetronServiceClient {
  #context;
  #serviceProcess = null;
  #stdoutReader = null;
  #requestId = 0;
  #starting = null;
  #pending = new Map();
  #requestLog = [];
  #getBinaryPath;

  constructor(context, getBinaryPath) {
    this.#context = context;
    this.#getBinaryPath = getBinaryPath;
  }

  async open(pathValue) {
    return this.request('open', { path: pathValue });
  }

  async close(session) {
    return this.request('close', { session });
  }

  async summary(session, limit) {
    return this.request('summary', this.#limitParams(session, limit));
  }

  async diagnostics(session, limit) {
    return this.request('diagnostics', this.#limitParams(session, limit));
  }

  async search(session, query, limit) {
    return this.request('search', {
      ...this.#limitParams(session, limit),
      query,
    });
  }

  async searchPage(session, query, limit, cursor = null) {
    return this.request('search', {
      ...this.#limitParams(session, limit),
      query,
      cursor,
    });
  }

  async detail(session, handle) {
    return this.request('detail', {
      session,
      handle,
    });
  }

  async slice(session, handle, maxNodes, collapse = 'none') {
    return this.request('slice', {
      ...this.#limitParams(session, maxNodes, true),
      handle,
      max_nodes: maxNodes,
      collapse,
    });
  }

  async layout(session, handle, maxNodes, collapse = 'none') {
    return this.request('layout', {
      ...this.#limitParams(session, maxNodes, true),
      handle,
      max_nodes: maxNodes,
      collapse,
    });
  }

  async mlirSymbols(session, limit) {
    return this.request('mlir.symbols', this.#limitParams(session, limit));
  }

  async onnxTensor(session, tensor, limit) {
    const payload = { session, tensor };
    if (typeof limit === 'number' && Number.isFinite(limit) && limit > 0) {
      payload.limit = limit;
    }
    return this.request('onnx.tensor', payload);
  }

  async request(method, params = {}) {
    await this.#ensureStarted();
    return this.#postRequest(method, params);
  }

  snapshotLog() {
    return this.#requestLog.slice(-20);
  }

  dispose() {
    if (this.#stdoutReader) {
      this.#stdoutReader.close();
      this.#stdoutReader = null;
    }
    for (const pending of this.#pending.values()) {
      pending.reject(new Error('service terminated'));
    }
    this.#pending.clear();
    if (this.#serviceProcess && !this.#serviceProcess.killed) {
      this.#serviceProcess.kill();
    }
    this.#serviceProcess = null;
    this.#starting = null;
  }

  #limitParams(session, limit, projection) {
    const params = { session };
    if (typeof limit === 'number' && Number.isFinite(limit) && limit > 0) {
      params.limit = Math.min(limit, DEFAULT_REQUEST_LIMIT);
    } else if (projection) {
      params.max_nodes = DEFAULT_REQUEST_LIMIT;
    }
    return params;
  }

  async #ensureStarted() {
    if (this.#serviceProcess) {
      return;
    }
    if (this.#starting) {
      await this.#starting;
      return;
    }
    this.#starting = this.#startService();
    try {
      await this.#starting;
    } finally {
      this.#starting = null;
    }
  }

  async #startService() {
    const binary = this.#resolveBinaryPath();
    if (!binary) {
      throw new Error(
        'netron-rs binary not found. Set netronRsViewer.binaryPath or build netron-rs.',
      );
    }

    await new Promise((resolve, reject) => {
      const process = cp.spawn(binary, ['serve', '--stdio'], {
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      let resolved = false;

      const cleanup = () => {
        process.removeListener('error', onError);
        process.removeListener('spawn', onSpawn);
      };
      const onError = (error) => {
        if (resolved) {
          return;
        }
        resolved = true;
        cleanup();
        reject(error);
      };
      const onSpawn = () => {
        if (resolved) {
          return;
        }
        resolved = true;
        cleanup();
        this.#serviceProcess = process;
        this.#serviceProcess.stderr?.on('data', (chunk) => {
          this.#logInternal(`${chunk}`);
        });
        this.#stdoutReader = readline.createInterface({
          input: this.#serviceProcess.stdout,
        });
        this.#stdoutReader.on('line', (line) => {
          this.#handleResponseLine(line);
        });
        this.#stdoutReader.on('close', () => {
          this.#serviceProcess = null;
        });
        process.once('exit', (code, signal) => {
          const active = this.#serviceProcess === process;
          this.#serviceProcess = null;
          if (!active) {
            return;
          }
          const cause = signal
            ? `service exited with ${signal}`
            : `service exited with code ${code}`;
          for (const pending of this.#pending.values()) {
            pending.reject(new Error(cause));
          }
          this.#pending.clear();
        });
        resolve();
      };

      process.once('error', onError);
      process.once('spawn', onSpawn);
    });
  }

  #resolveBinaryPath() {
    const configured = this.#getBinaryPath?.();
    const extensionRoot = path.resolve(this.#context.extensionUri.fsPath, '..', '..');
    const configuredBinary = configured?.trim() || '';
    if (configuredBinary) {
      if (this.#isExecutable(configuredBinary)) {
        return configuredBinary;
      }
      return null;
    }
    const candidates = [
      path.join(extensionRoot, 'target', 'release', 'netron-rs'),
      path.join(extensionRoot, 'target', 'release', 'netron-rs.exe'),
      path.join(extensionRoot, 'target', 'debug', 'netron-rs'),
      path.join(extensionRoot, 'target', 'debug', 'netron-rs.exe'),
    ].filter(Boolean);
    const fallback = 'netron-rs';

    for (const candidate of candidates) {
      if (this.#isExecutable(candidate)) {
        return candidate;
      }
    }
    return fallback;
  }

  #isExecutable(candidate) {
    try {
      return fs.statSync(candidate).isFile();
    } catch (_error) {
      return false;
    }
  }

  #postRequest(method, params) {
    const id = String(++this.#requestId);
    const envelope = {
      id,
      method,
      params,
    };
    const payload = `${JSON.stringify(envelope)}\n`;

    this.#requestLog.push({
      id,
      at: new Date().toISOString(),
      method,
      params: Object.keys(params),
    });
    if (this.#requestLog.length > 200) {
      this.#requestLog = this.#requestLog.slice(-200);
    }

    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      try {
        this.#serviceProcess.stdin.write(payload);
      } catch (error) {
        this.#pending.delete(id);
        reject(error);
      }
    });
  }

  #handleResponseLine(line) {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    try {
      const response = JSON.parse(trimmed);
      const request = this.#pending.get(String(response.id));
      if (!request) {
        return;
      }
      this.#pending.delete(String(response.id));
      if (response.status === 'ok') {
        request.resolve(response.data);
        return;
      }
      const code = response.error?.code || 'error';
      const message = response.error?.message || 'service request failed';
      request.reject(new Error(`[${code}] ${message}`));
    } catch (error) {
      for (const entry of this.#pending.values()) {
        entry.reject(error);
      }
      this.#pending.clear();
    }
  }

  #logInternal(message) {
    this.#requestLog.push({
      at: new Date().toISOString(),
      message,
    });
    if (this.#requestLog.length > 200) {
      this.#requestLog = this.#requestLog.slice(-200);
    }
  }
}

module.exports = { NetronServiceClient };
