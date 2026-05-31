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
    const rustBin = await resolveRustBin(options);
    const files = await resolveFiles(options);
    if (files.length === 0) {
        throw new Error('No model files selected.');
    }

    const loc = await lineCounts();
    const results = [];
    for (const file of files) {
        const target = path.resolve(file);
        const stat = await fs.stat(target);
        const oldRuns = [];
        const rustRuns = [];
        for (let index = 0; index < options.warmups; index++) {
            await timedExec(process.execPath, [path.join(root, 'tools', 'netron-normalize.mjs'), target], repo);
            await timedExec(rustBin, ['parse', target], repo);
        }
        for (let index = 0; index < options.runs; index++) {
            oldRuns.push(await timedExec(process.execPath, [path.join(root, 'tools', 'netron-normalize.mjs'), target], repo));
            rustRuns.push(await timedExec(rustBin, ['parse', target], repo));
        }
        const bench = JSON.parse(await execFile(rustBin, ['bench', target, String(options.iterations)], repo));
        const oldSummary = summarizeRuns(oldRuns);
        const rustSummary = summarizeRuns(rustRuns);
        results.push({
            file: target,
            bytes: stat.size,
            old_netron_normalize: oldSummary,
            rust_netron_parse: rustSummary,
            rust_bench: bench,
            speedup: {
                wall_mean: oldSummary.wall_ms_mean / rustSummary.wall_ms_mean,
                wall_min: oldSummary.wall_ms_min / rustSummary.wall_ms_min,
                backend_parse_and_json_last: oldSummary.wall_ms_mean / bench.parse_and_json_ms_last,
                backend_time_to_first_graph_last: oldSummary.wall_ms_mean / bench.time_to_first_graph_ms_last,
                rss_peak: oldSummary.peak_rss_kb_max > 0 && rustSummary.peak_rss_kb_max > 0 ?
                    oldSummary.peak_rss_kb_max / rustSummary.peak_rss_kb_max :
                    null
            }
        });
    }

    const ratios = results.map((result) => result.speedup.wall_mean).filter(Number.isFinite);
    const summary = {
        files: results.length,
        wall_speedup_mean: ratios.reduce((sum, value) => sum + value, 0) / ratios.length,
        wall_speedup_min: Math.min(...ratios),
        loc_ratio_old_to_rust: loc.old.lines / loc.rust.lines,
        passes_speedup_2x: ratios.length > 0 && ratios.every((value) => value >= options.minSpeedup),
        passes_loc_2x: loc.old.lines >= options.minLocRatio * loc.rust.lines
    };

    const report = {
        ok: summary.passes_speedup_2x && summary.passes_loc_2x,
        thresholds: {
            min_speedup: options.minSpeedup,
            min_loc_ratio_old_to_rust: options.minLocRatio
        },
        measurement: {
            runs: options.runs,
            warmups: options.warmups,
            bench_iterations: options.iterations
        },
        rust_bin: rustBin,
        loc,
        summary,
        results
    };
    const output = JSON.stringify(report, null, 2);
    if (options.output) {
        await fs.mkdir(path.dirname(options.output), { recursive: true });
        await fs.writeFile(options.output, `${output}\n`);
    }
    console.log(output);
    if (options.enforce && (!summary.passes_speedup_2x || !summary.passes_loc_2x)) {
        process.exit(1);
    }
};

const parseArgs = (args) => {
    const options = {
        type: 'onnx',
        limit: 1,
        extensions: new Set(['.onnx', '.pb']),
        patterns: [],
        files: [],
        rustBin: null,
        runs: 3,
        warmups: 1,
        iterations: 10,
        build: true,
        enforce: false,
        output: null,
        minSpeedup: 2,
        minLocRatio: 2
    };
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        switch (arg) {
            case '--type':
                options.type = value(args, ++index, arg);
                break;
            case '--limit':
                options.limit = positiveInt(value(args, ++index, arg), arg);
                break;
            case '--extensions':
                options.extensions = new Set(value(args, ++index, arg).split(',').map((item) => item.trim().toLowerCase()).filter(Boolean));
                break;
            case '--rust-bin':
                options.rustBin = path.resolve(value(args, ++index, arg));
                options.build = false;
                break;
            case '--runs':
                options.runs = positiveInt(value(args, ++index, arg), arg);
                break;
            case '--warmups':
                options.warmups = nonNegativeInt(value(args, ++index, arg), arg);
                break;
            case '--iterations':
                options.iterations = positiveInt(value(args, ++index, arg), arg);
                break;
            case '--min-speedup':
                options.minSpeedup = positiveNumber(value(args, ++index, arg), arg);
                break;
            case '--min-loc-ratio':
                options.minLocRatio = positiveNumber(value(args, ++index, arg), arg);
                break;
            case '--no-build':
                options.build = false;
                break;
            case '--enforce':
                options.enforce = true;
                break;
            case '--output':
                options.output = path.resolve(value(args, ++index, arg));
                break;
            default:
                if (arg.startsWith('--')) {
                    throw new Error(`Unknown option: ${arg}`);
                }
                options.files.push(arg);
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

const positiveInt = (input, flag) => {
    const result = Number.parseInt(input, 10);
    if (!Number.isInteger(result) || result < 1) {
        throw new Error(`${flag} must be a positive integer`);
    }
    return result;
};

const nonNegativeInt = (input, flag) => {
    const result = Number.parseInt(input, 10);
    if (!Number.isInteger(result) || result < 0) {
        throw new Error(`${flag} must be a non-negative integer`);
    }
    return result;
};

const positiveNumber = (input, flag) => {
    const result = Number.parseFloat(input);
    if (!Number.isFinite(result) || result <= 0) {
        throw new Error(`${flag} must be a positive number`);
    }
    return result;
};

const resolveRustBin = async (options) => {
    if (options.rustBin) {
        return options.rustBin;
    }
    if (options.build) {
        await execFile('cargo', ['build', '--release', '--manifest-path', path.join(root, 'Cargo.toml')], repo);
    }
    const name = process.platform === 'win32' ? 'netron-rs.exe' : 'netron-rs';
    return path.join(root, 'target', 'release', name);
};

const resolveFiles = async (options) => {
    if (options.files.length > 0) {
        return options.files;
    }

    const content = await fs.readFile(path.join(repo, 'netron', 'test', 'models.json'), 'utf-8');
    const models = JSON.parse(content);
    const entries = models
        .filter((entry) => entry.type === options.type)
        .filter((entry) => targetFiles(entry).some((file) => options.extensions.has(path.extname(file).toLowerCase())))
        .slice(0, options.limit);
    const files = [];
    for (const entry of entries) {
        const prepared = {
            ...entry,
            targets: targetFiles(entry),
            tags: (entry.tags || '').split(',').map((tag) => tag.trim()).filter(Boolean)
        };
        const target = new Target(prepared);
        await target.download();
        files.push(...target.targets
            .filter((file) => options.extensions.has(path.extname(file).toLowerCase()))
            .map((file) => path.join(target.folder, file)));
    }
    return files;
};

const targetFiles = (entry) => entry.target.split(',').map((file) => file.trim()).filter(Boolean);

const timedExec = async (command, args, cwd) => {
    const timed = await timedExecWithSystemTime(command, args, cwd);
    if (timed) {
        return timed;
    }
    return timedExecWithSampling(command, args, cwd);
};

const timedExecWithSystemTime = async (command, args, cwd) => {
    if (process.platform === 'win32') {
        return null;
    }
    const timeArgs = process.platform === 'darwin' ?
        ['-l', command, ...args] :
        ['-v', command, ...args];
    const start = performance.now();
    const child = child_process.spawn('/usr/bin/time', timeArgs, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    const status = await new Promise((resolve, reject) => {
        child.on('error', (error) => {
            if (error.code === 'ENOENT') {
                resolve(null);
            } else {
                reject(error);
            }
        });
        child.on('close', (code) => resolve(code ?? 1));
    });
    if (status === null) {
        return null;
    }
    const wallMs = performance.now() - start;
    const stderrText = Buffer.concat(stderr).toString('utf-8');
    if (status !== 0) {
        const unsupported = stderrText.includes('illegal option') || stderrText.includes('invalid option');
        if (unsupported) {
            return null;
        }
        throw new Error(`${command} ${args.join(' ')} failed with status ${status}\n${stderrText}`);
    }
    return {
        wall_ms: wallMs,
        peak_rss_kb: parsePeakRssKb(stderrText),
        rss_samples: null,
        stdout_bytes: Buffer.concat(stdout).byteLength
    };
};

const parsePeakRssKb = (stderr) => {
    const darwin = stderr.match(/^\s*(\d+)\s+maximum resident set size/m);
    if (darwin) {
        return Math.ceil(Number.parseInt(darwin[1], 10) / 1024);
    }
    const linux = stderr.match(/Maximum resident set size \(kbytes\):\s*(\d+)/);
    if (linux) {
        return Number.parseInt(linux[1], 10);
    }
    return 0;
};

const timedExecWithSampling = async (command, args, cwd) => {
    const start = performance.now();
    const child = child_process.spawn(command, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    let peakRssKb = 0;
    const samples = [];
    const timer = setInterval(async () => {
        try {
            const rss = await processRssKb(child.pid);
            if (rss > peakRssKb) {
                peakRssKb = rss;
            }
            samples.push(rss);
        } catch {
            clearInterval(timer);
        }
    }, 5);
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    const status = await new Promise((resolve) => {
        child.on('close', (code) => resolve(code ?? 1));
    });
    clearInterval(timer);
    const wallMs = performance.now() - start;
    if (status !== 0) {
        throw new Error(`${command} ${args.join(' ')} failed with status ${status}\n${Buffer.concat(stderr).toString('utf-8')}`);
    }
    if (peakRssKb === 0) {
        peakRssKb = await selfRssKb();
    }
    return {
        wall_ms: wallMs,
        peak_rss_kb: peakRssKb,
        rss_samples: samples.length,
        stdout_bytes: Buffer.concat(stdout).byteLength
    };
};

const processRssKb = async (pid) => {
    if (!pid) {
        return 0;
    }
    const output = await execFile('ps', ['-o', 'rss=', '-p', String(pid)], repo);
    const value = Number.parseInt(output.trim(), 10);
    return Number.isFinite(value) ? value : 0;
};

const selfRssKb = async () => {
    try {
        return await processRssKb(process.pid);
    } catch {
        return 0;
    }
};

const summarizeRuns = (runs) => ({
    runs: runs.length,
    wall_ms_min: min(runs.map((run) => run.wall_ms)),
    wall_ms_mean: mean(runs.map((run) => run.wall_ms)),
    wall_ms_max: max(runs.map((run) => run.wall_ms)),
    peak_rss_kb_max: max(runs.map((run) => run.peak_rss_kb)),
    stdout_bytes_min: min(runs.map((run) => run.stdout_bytes))
});

const min = (values) => values.reduce((result, value) => Math.min(result, value), Number.POSITIVE_INFINITY);
const max = (values) => values.reduce((result, value) => Math.max(result, value), Number.NEGATIVE_INFINITY);
const mean = (values) => values.reduce((result, value) => result + value, 0) / values.length;

const lineCounts = async () => ({
    old: await countLines(path.join(repo, 'netron'), [
        'source',
        'test',
        'tools'
    ], new Set(['.js', '.mjs', '.py', '.css', '.html', '.json', '.proto', '.fbs'])),
    rust: await countLines(root, [
        'crates',
        'tools'
    ], new Set(['.rs', '.mjs', '.toml']))
});

const countLines = async (base, roots, extensions) => {
    const files = [];
    for (const entry of roots) {
        await walk(path.join(base, entry), extensions, files);
    }
    let lines = 0;
    for (const file of files) {
        const content = await fs.readFile(file, 'utf-8');
        lines += content.split(/\r?\n/).filter((line) => {
            const trimmed = line.trim();
            return trimmed && !trimmed.startsWith('//') && !trimmed.startsWith('#');
        }).length;
    }
    return { files: files.length, lines };
};

const walk = async (folder, extensions, files) => {
    const entries = await fs.readdir(folder, { withFileTypes: true });
    for (const entry of entries) {
        if (entry.name === 'node_modules' || entry.name === 'target' || entry.name.startsWith('.')) {
            continue;
        }
        const full = path.join(folder, entry.name);
        if (entry.isDirectory()) {
            await walk(full, extensions, files);
        } else if (entry.isFile() && extensions.has(path.extname(entry.name).toLowerCase())) {
            files.push(full);
        }
    }
};

const execFile = (command, args, cwd) => new Promise((resolve, reject) => {
    child_process.execFile(command, args, { cwd, maxBuffer: 256 * 1024 * 1024 }, (error, stdout, stderr) => {
        if (error) {
            error.message = `${error.message}\n${stderr}`;
            reject(error);
        } else {
            resolve(stdout);
        }
    });
});

main().catch((error) => {
    console.error(`${error.name}: ${error.message}`);
    process.exit(1);
});
