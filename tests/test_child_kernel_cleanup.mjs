import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { EventEmitter } from "node:events";
import { appendFileSync, existsSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { runChildAgent } from "../extensions/child-agent.ts";
import { RlmHostBridge } from "../extensions/rlm-host.ts";
import { kernelStartupCode } from "../extensions/rlm.ts";
import { alive, events, waitFor } from "./fixtures/child-agent.mjs";

const ipythonRoot = resolve(process.env.PI_IPYTHON_ROOT ?? fileURLToPath(new URL("../../pi-ipython", import.meta.url)));
const librlmRoot = resolve(process.env.RLM_LIBRLM_ROOT ?? join(homedir(), "Dev", "librlm"));
if (!existsSync(join(ipythonRoot, "extensions", "kernel-runtime.ts"))) {
	throw new Error(`pi-ipython checkout not found at ${ipythonRoot}; set PI_IPYTHON_ROOT`);
}
const { KernelRuntime } = await import(pathToFileURL(join(ipythonRoot, "extensions", "kernel-runtime.ts")).href);
const exec = promisify(execFile);

function command(directory) {
	return { command: process.execPath, args: ["--no-warnings", fileURLToPath(import.meta.url), "--kernel-child", directory] };
}

function log(directory, event) {
	appendFileSync(join(directory, "events.jsonl"), `${JSON.stringify(event)}\n`);
}

async function execute(program, args, options) {
	try { return { ...await exec(program, args, options), code: 0 }; }
	catch (error) { return { code: error.code ?? 1, stderr: error.stderr ?? error.message, stdout: error.stdout ?? "" }; }
}

async function kernelChild(directory) {
	for await (const _chunk of process.stdin) {}
	const depth = Number(process.env.PI_RLM_DEPTH);
	log(directory, { type: "agent", depth, pid: process.pid });
	const bus = new EventEmitter();
	const host = new RlmHostBridge(librlmRoot, depth, command(directory));
	host.setConfig({ cwd: directory, model: { provider: "test", id: "fixture" }, thinkingLevel: "off" });
	await host.ensureStarted();
	log(directory, { type: "host", depth, socket: host.environment.RLM_HOST_SOCKET });
	bus.on("ipython:kernel-starting", (event) => {
		Object.assign(event.env, host.environment);
		event.startupCode.push(kernelStartupCode(librlmRoot));
	});
	const kernel = new KernelRuntime({ events: bus, exec: execute });
	let closing;
	const close = () => closing ??= (async () => {
		await kernel.shutdown();
		await host.shutdown();
		log(directory, { type: "stopped", depth, pid: process.pid });
	})();
	const terminate = () => {
		void close().then(() => process.exit(143), (error) => {
			console.error(error);
			process.exit(1);
		});
	};
	process.on("SIGTERM", terminate);
	const code = `import json, os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
fd = os.open(${JSON.stringify(join(directory, "events.jsonl"))}, os.O_WRONLY | os.O_APPEND)
os.write(fd, (json.dumps({"type": "kernel", "depth": ${depth}, "pid": os.getpid(), "pgid": os.getpgrp(), "bridge_pid": os.getppid(), "bridge_pgid": os.getpgid(os.getppid())}) + "\\n").encode())
os.close(fd)
${depth === 1 ? 'h = await rlm.spawn("Block with an active kernel.")\nawait rlm.gather([h])\n' : ""}time.sleep(60)`;
	try {
		await kernel.execute("blocking-kernel", code, directory, undefined, () => {}, () => {});
	} catch (error) {
		if (!closing) throw error;
	} finally {
		await close();
		process.off("SIGTERM", terminate);
	}
}

async function testCleanup() {
	const directory = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-child-kernel-test-")));
	const controller = new AbortController();
	let timeout;
	const completion = runChildAgent({
		model: { provider: "test", id: "fixture" }, thinkingLevel: "off",
		cwd: directory, task: "Block with an active child kernel and grandchild.", context: null,
		signal: controller.signal, depth: 0, librlmRoot,
	}, command(directory));
	try {
		const kernels = await waitFor(() => {
			const ready = events(directory).filter((event) => event.type === "kernel");
			return ready.length === 2 && ready;
		}, "active child and grandchild kernels");
		const agents = events(directory).filter((event) => event.type === "agent");
		assert.deepEqual(kernels.map((event) => event.depth).sort(), [1, 2]);
		assert.deepEqual(agents.map((event) => event.depth).sort(), [1, 2]);
		const pids = [...agents.map((event) => event.pid), ...kernels.flatMap((event) => [event.pid, event.bridge_pid])];
		assert.equal(new Set(pids).size, 6);
		assert.ok(pids.every(alive));
		controller.abort();
		const result = await Promise.race([
			completion,
			new Promise((_resolve, reject) => { timeout = setTimeout(() => reject(new Error("Child cleanup did not settle")), 20_000); }),
		]);
		assert.ok(result.error);
		await waitFor(() => pids.every((pid) => !alive(pid)), "all child agents, bridges and kernels to stop");
		assert.deepEqual(events(directory).filter((event) => event.type === "stopped").map((event) => event.depth).sort(), [1, 2]);
		const hosts = events(directory).filter((event) => event.type === "host");
		assert.equal(hosts.length, 2);
		assert.ok(hosts.every((event) => !existsSync(dirname(event.socket))), "child host directories are removed");
	} finally {
		clearTimeout(timeout);
		controller.abort();
		const observed = events(directory);
		for (const event of observed) {
			const groups = event.type === "agent" ? [event.pid] : event.type === "kernel" ? [event.pgid, event.bridge_pgid] : [];
			for (const pgid of groups) {
				try { process.kill(-pgid, "SIGKILL"); } catch {}
			}
		}
		await completion;
		for (const event of observed.filter((event) => event.type === "host")) {
			rmSync(dirname(event.socket), { recursive: true, force: true });
		}
		rmSync(directory, { recursive: true, force: true });
	}
}

if (process.argv[2] === "--kernel-child") await kernelChild(process.argv[3]);
else {
	await testCleanup();
	console.log("child kernels: cancellation reaps active child and grandchild kernels, agents and hosts");
}
