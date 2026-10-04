import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { EventEmitter } from "node:events";
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { runChildAgent } from "../extensions/child-agent.ts";
import { RlmHostBridge } from "../extensions/rlm-host.ts";
import { kernelStartupCode } from "../extensions/rlm.ts";
import { alive, events, running, waitFor } from "./fixtures/child-agent.mjs";

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
	appendFileSync(join(directory, "events.jsonl"), `${JSON.stringify({ at: Date.now(), ...event })}\n`);
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
		log(directory, { type: "term", depth, pid: process.pid });
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
		const started = performance.now();
		controller.abort();
		const result = await Promise.race([
			completion,
			new Promise((_resolve, reject) => { timeout = setTimeout(() => reject(new Error("Child cleanup did not settle")), 20_000); }),
		]);
		const elapsed = performance.now() - started;
		const directAgent = agents.find((event) => event.depth === 1);
		const survivors = pids.filter((pid) => pid === directAgent.pid ? alive(pid) : running(pid));
		console.log(`active child and grandchild cleanup: ${Math.round(elapsed)} ms, live PIDs at settlement: ${JSON.stringify(survivors)}`);
		assert.ok(result.error);
		assert.deepEqual(survivors.filter((pid) => agents.some((event) => event.pid === pid)), [], "child agents are dead when runChildAgent settles");
		await waitFor(() => kernels.every((event) => !running(event.pid) && !running(event.bridge_pid)), "child bridges and kernels to stop after runChildAgent settles", 5_000);
		const stopped = events(directory).filter((event) => event.type === "stopped");
		assert.deepEqual(stopped.map((event) => event.depth).sort(), [1, 2]);
		for (const event of stopped) {
			const term = events(directory).find((candidate) => candidate.type === "term" && candidate.depth === event.depth);
			console.log(`active kernel cleanup at depth ${event.depth}: ${event.at - term.at} ms`);
		}
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

function checkpointFixture(directory) {
	const code = `import json, os, time
with open(${JSON.stringify(join(directory, "events.jsonl"))}, "a") as record:
    record.write(json.dumps({"type": "kernel", "pid": os.getpid(), "pgid": os.getpgrp(), "bridge_pid": os.getppid(), "bridge_pgid": os.getpgid(os.getppid()), "socket": os.environ["RLM_HOST_SOCKET"]}) + "\\n")
class SlowCheckpoint:
    def __getstate__(self):
        with open(${JSON.stringify(join(directory, "events.jsonl"))}, "a") as record:
            record.write(json.dumps({"type": "save-start", "pid": os.getpid()}) + "\\n")
        time.sleep(60)
        with open(${JSON.stringify(join(directory, "events.jsonl"))}, "a") as record:
            record.write(json.dumps({"type": "save-complete", "pid": os.getpid()}) + "\\n")
        return {}
pending_checkpoint = SlowCheckpoint()`;
	return `import { appendFileSync, readFileSync } from "node:fs";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { Type } from "typebox";

export default function (pi) {
    const file = ${JSON.stringify(join(directory, "events.jsonl"))};
    const log = (event) => appendFileSync(file, JSON.stringify(event) + "\\n");
    pi.on("session_start", () => log({ type: "agent", pid: process.pid }));
    process.on("SIGTERM", () => log({ type: "term", pid: process.pid }));
    pi.registerTool({
        name: "pending_checkpoint", label: "Pending checkpoint", description: "Save a slow checkpoint.",
        parameters: Type.Object({}),
        async execute(_id, _args, signal, _update, ctx) {
            const outcome = await ctx.executeTool("ipython", { code: ${JSON.stringify(code)} });
            if (outcome.isError) throw new Error(JSON.stringify(outcome.result));
            log({ type: "cell-complete", checkpoint: outcome.result.details.checkpoint });
            const deadline = Date.now() + 10000;
            while (!readFileSync(file, "utf8").trim().split("\\n").filter(Boolean).map(JSON.parse).some((event) => event.type === "save-start")) {
                if (Date.now() >= deadline) throw new Error("Checkpoint save did not start");
                await new Promise((resolve) => setTimeout(resolve, 10));
            }
            log({ type: "ready" });
            await new Promise((resolve) => {
                if (signal.aborted) resolve();
                else signal.addEventListener("abort", resolve, { once: true });
            });
            return { content: [{ type: "text", text: "cancelled" }], details: {} };
        },
    });
    pi.registerProvider("kernel-test", {
        api: "kernel-cleanup-test", apiKey: "fixture", baseUrl: "http://localhost",
        models: [{ id: "fixture", name: "Fixture", reasoning: false, input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 64000, maxTokens: 4096 }],
        streamSimple(model) {
            const stream = createAssistantMessageEventStream();
            const message = { role: "assistant", api: model.api, provider: model.provider, model: model.id,
                timestamp: Date.now(), stopReason: "pending", content: [],
                usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
            stream.push({ type: "start", partial: message });
            message.content.push({ type: "toolCall", id: "checkpoint", name: "pending_checkpoint", arguments: {} });
            stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
            stream.push({ type: "toolcall_end", contentIndex: 0, toolCall: message.content[0], partial: message });
            message.stopReason = "toolUse";
            stream.push({ type: "done", reason: "toolUse", message });
            stream.end(message);
            return stream;
        },
    });
}
`;
}

async function testPendingCheckpoint(depth) {
	const directory = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-pending-checkpoint-test-")));
	const agentDirectory = join(directory, "agent");
	mkdirSync(join(agentDirectory, "extensions"), { recursive: true });
	writeFileSync(join(agentDirectory, "extensions", "checkpoint.ts"), checkpointFixture(directory));
	writeFileSync(join(agentDirectory, "settings.json"), JSON.stringify({
		packages: [ipythonRoot, resolve(fileURLToPath(new URL("..", import.meta.url)))],
	}));
	const environment = {
		PI_CODING_AGENT_DIR: agentDirectory,
		XDG_CACHE_HOME: join(directory, "cache"),
		PI_IPYTHON_PERSISTENCE: undefined,
	};
	const saved = Object.fromEntries(Object.keys(environment).map((key) => [key, process.env[key]]));
	for (const [key, value] of Object.entries(environment)) {
		if (value === undefined) delete process.env[key]; else process.env[key] = value;
	}
	const controller = new AbortController();
	let timeout;
	const completion = runChildAgent({
		model: { provider: "kernel-test", id: "fixture" }, thinkingLevel: "off",
		cwd: directory, task: "Create a pending IPython checkpoint.", context: null,
		signal: controller.signal, depth, librlmRoot,
	}, {
		command: process.execPath,
		args: [fileURLToPath(new URL("../node_modules/@earendil-works/pi-coding-agent/dist/cli.js", import.meta.url)), "--offline"],
	});
	try {
		let settled;
		void completion.then((result) => { settled = result; });
		await waitFor(() => {
			if (settled) throw new Error(`Real Pi exited before checkpoint readiness: ${settled.error}`);
			return events(directory).some((event) => event.type === "ready");
		}, "real Pi with a pending checkpoint");
		const observed = events(directory);
		const kernel = observed.find((event) => event.type === "kernel");
		const agent = observed.find((event) => event.type === "agent");
		const cell = observed.find((event) => event.type === "cell-complete");
		assert.equal(typeof cell.checkpoint, "string", "default persistence queues a real checkpoint");
		assert.ok(observed.some((event) => event.type === "save-start" && event.pid === kernel.pid));
		assert.ok(!observed.some((event) => event.type === "save-complete"), "checkpoint save is still pending");
		const pids = [agent.pid, kernel.pid, kernel.bridge_pid];
		assert.equal(new Set(pids).size, 3);
		assert.ok(pids.every(alive));
		const started = performance.now();
		controller.abort();
		const result = await Promise.race([
			completion,
			new Promise((_resolve, reject) => { timeout = setTimeout(() => reject(new Error("Pending checkpoint cleanup did not settle")), 20_000); }),
		]);
		const survivors = pids.filter((pid) => pid === agent.pid ? alive(pid) : running(pid));
		const elapsed = performance.now() - started;
		console.log(`pending checkpoint cleanup at depth ${depth + 1}: ${Math.round(elapsed)} ms, agent ${agent.pid}, kernel ${kernel.pid}, bridge ${kernel.bridge_pid}, live PIDs at settlement: ${JSON.stringify(survivors)}`);
		assert.ok(result.error);
		assert.equal(alive(agent.pid), false, "real Pi is dead when runChildAgent settles");
		await waitFor(() => !running(kernel.pid) && !running(kernel.bridge_pid), "checkpoint kernel and bridge to stop after runChildAgent settles", 5_000);
		assert.ok(!existsSync(dirname(kernel.socket)), "child host directory is removed");
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
		try {
			await completion;
			const kernels = observed.filter((event) => event.type === "kernel");
			await waitFor(() => kernels.every((event) => !running(event.pid) && !running(event.bridge_pid)), "leaked checkpoint processes to stop after test cleanup");
			for (const kernel of kernels) rmSync(dirname(kernel.socket), { recursive: true, force: true });
			console.log(`checkpoint test cleanup at depth ${depth + 1}: kernel and bridge processes have stopped`);
		} finally {
			for (const [key, value] of Object.entries(saved)) {
				if (value === undefined) delete process.env[key]; else process.env[key] = value;
			}
			rmSync(directory, { recursive: true, force: true });
		}
	}
}

if (process.argv[2] === "--kernel-child") await kernelChild(process.argv[3]);
else {
	await testCleanup();
	console.log("child kernels: cancellation reaps active child and grandchild kernels, agents and hosts");
	const failures = [];
	for (const depth of [1, 0]) {
		try { await testPendingCheckpoint(depth); }
		catch (error) { failures.push(error); }
	}
	if (failures.length) throw new AggregateError(failures, "Pending checkpoints leak detached kernels at child settlement");
}
