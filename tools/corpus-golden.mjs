#!/usr/bin/env node

import * as child_process from 'child_process';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as url from 'url';

import { Target } from '../../netron/test/worker.js';

const repo = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..', '..');
const root = path.join(repo, 'netron-rs');

const main = async () => {
    const options = parseArgs(process.argv.slice(2));
    const content = await fs.readFile(path.join(repo, 'netron', 'test', 'models.json'), 'utf-8');
    const models = JSON.parse(content);
    const entries = select(models, options);
    if (entries.length === 0) {
        throw new Error('No matching corpus entries.');
    }

    const results = [];
    for (const entry of entries) {
        const prepared = prepare(entry);
        const target = new Target(prepared);
        if (options.verbose) {
            target.on('status', (_sender, message) => {
                if (message?.name === 'name') {
                    console.error(message.target);
                }
            });
        }
        await target.download();
        const files = target.targets
            .filter((file) => options.extensions.has(path.extname(file).toLowerCase()))
            .map((file) => path.join(target.folder, file));
        for (const file of files) {
            const result = await golden(file, options.rustBin);
            results.push({
                target: entry.target,
                source: entry.source || null,
                file,
                ...result
            });
        }
    }

    const ok = results.every((result) => result.ok);
    console.log(JSON.stringify({ ok, selected: entries.length, results }, null, 2));
    if (!ok) {
        process.exit(1);
    }
};

const parseArgs = (args) => {
    const options = {
        type: 'onnx',
        limit: 5,
        tags: new Set(),
        extensions: new Set(['.onnx', '.pb']),
        rustBin: null,
        verbose: false,
        patterns: []
    };
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        switch (arg) {
            case '--type':
                options.type = value(args, ++i, arg);
                break;
            case '--limit':
                options.limit = Number.parseInt(value(args, ++i, arg), 10);
                if (!Number.isInteger(options.limit) || options.limit < 1) {
                    throw new Error('--limit must be a positive integer');
                }
                break;
            case '--tag':
                options.tags.add(value(args, ++i, arg));
                break;
            case '--extensions':
                options.extensions = new Set(value(args, ++i, arg).split(',').map((item) => item.trim().toLowerCase()).filter(Boolean));
                break;
            case '--rust-bin':
                options.rustBin = value(args, ++i, arg);
                break;
            case '--verbose':
                options.verbose = true;
                break;
            default:
                options.patterns.push(arg);
                break;
        }
    }
    return options;
};

const value = (args, index, flag) => {
    const result = args[index];
    if (!result) {
        throw new Error(`${flag} requires a value`);
    }
    return result;
};

const select = (models, options) => models
    .filter((entry) => entry.type === options.type)
    .filter((entry) => targetFiles(entry).some((file) => options.extensions.has(path.extname(file).toLowerCase())))
    .filter((entry) => {
        if (options.tags.size === 0) {
            return true;
        }
        const tags = new Set((entry.tags || '').split(',').map((tag) => tag.trim()).filter(Boolean));
        return Array.from(options.tags).every((tag) => tags.has(tag));
    })
    .filter((entry) => {
        if (options.patterns.length === 0) {
            return true;
        }
        const haystack = `${entry.target}\n${entry.source || ''}\n${entry.tags || ''}`;
        return options.patterns.some((pattern) => haystack.includes(pattern));
    })
    .slice(0, options.limit);

const prepare = (entry) => ({
    ...entry,
    targets: targetFiles(entry),
    tags: (entry.tags || '').split(',').map((tag) => tag.trim()).filter(Boolean)
});

const targetFiles = (entry) => entry.target.split(',').map((file) => file.trim()).filter(Boolean);

const golden = async (file, rustBin) => {
    const args = [path.join(root, 'tools', 'golden-diff.mjs'), '--compact'];
    if (rustBin) {
        args.push('--rust-bin', rustBin);
    }
    args.push(file);
    const result = await exec(process.execPath, args, repo);
    let parsed = null;
    if (result.stdout.trim()) {
        try {
            parsed = JSON.parse(result.stdout);
        } catch (error) {
            return {
                ok: false,
                status: result.status,
                differences: [{
                    path: 'golden-diff',
                    old: null,
                    rust: `invalid JSON output: ${error.message}`
                }],
                old: null,
                rust: null,
                stderr: result.stderr.trim()
            };
        }
    }
    return {
        ok: result.status === 0 && parsed?.ok === true,
        status: result.status,
        differences: parsed?.results?.[0]?.differences || [],
        old: parsed?.results?.[0]?.old || null,
        rust: parsed?.results?.[0]?.rust || null,
        stderr: result.stderr.trim()
    };
};

const exec = (command, args, cwd) => new Promise((resolve) => {
    child_process.execFile(command, args, { cwd, maxBuffer: 256 * 1024 * 1024 }, (error, stdout, stderr) => {
        resolve({ status: error ? error.code || 1 : 0, stdout, stderr });
    });
});

main().catch((error) => {
    console.error(`${error.name}: ${error.message}`);
    process.exit(1);
});
