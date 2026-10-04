import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { EventEmitter } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { alive, command, events, waitFor } from "./fixtures/child-agent.mjs";

// Node cannot strip TypeScript types under node_modules.
const ipythonRoot = resolve(process.env.PI_IPYTHON_ROOT ?? fileURLToPath(new URL("../../pi-ipython", import.meta.url)));
if (!existsSync(join(ipythonRoot, "extensions", "kernel-runtime.ts"))) {
	throw new Error(`pi-ipython checkout not found at ${ipythonRoot}; set PI_IPYTHON_ROOT`);
}
process.env.RLM_LIBRLM_ROOT ??= join(homedir(), "Dev", "librlm");
const { KernelRuntime } = await import(pathToFileURL(join(ipythonRoot, "extensions", "kernel-runtime.ts")).href);
const { default: rlmExtension } = await import("../extensions/rlm.ts");
const exec = promisify(execFile);
const savedDepth = process.env.PI_RLM_DEPTH;

function session(depth) {
	if (depth === undefined) delete process.env.PI_RLM_DEPTH; else process.env.PI_RLM_DEPTH = depth;
	const bus = new EventEmitter();
	const handlers = new Map();
	const pi = {
		events: { emit: (channel, data) => bus.emit(channel, data), on: (channel, handler) => bus.on(channel, handler) },
		on: (name, handler) => handlers.set(name, [...(handlers.get(name) ?? []), handler]),
		getThinkingLevel: () => "off",
		async exec(program, args, options) {
			try { return { ...await exec(program, args, options), code: 0 }; }
			catch (error) { return { code: error.code ?? 1, stderr: error.stderr ?? error.message, stdout: error.stdout ?? "" }; }
		},
	};
	const fire = async (name, event, ctx) => {
		let result;
		for (const handler of handlers.get(name) ?? []) result = (await handler(event, ctx)) ?? result;
		return result;
	};
	const cwd = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-kernel-")));
	const ctx = { cwd, model: { provider: "test", id: "fixture" }, hasUI: false };
	rlmExtension(pi, command(cwd));
	const kernel = new KernelRuntime(pi);
	let index = 0;
	return {
		cwd, fire, ctx,
		async cell(code, signal) {
			const id = `call-${++index}`;
			await fire("tool_execution_start", { toolName: "ipython", toolCallId: id, args: { code } }, ctx);
			const { result } = await kernel.execute(id, code, cwd, signal, () => {}, () => {});
			const event = {
				toolName: "ipython", toolCallId: id, input: { code }, isError: result.status !== "ok",
				content: [{ type: "text", text: result.output }], details: { status: result.status },
			};
			return { result, hook: (await fire("tool_result", event, ctx)) ?? {} };
		},
		async close() {
			await kernel.shutdown();
			await fire("session_shutdown", {}, ctx);
			assert.ok(events(cwd).filter((event) => event.type === "start").every((event) => !alive(event.pid)));
			rmSync(cwd, { recursive: true, force: true });
		},
	};
}

try {
	for (const depth of ["-1", "1.5", "garbage", "", "9007199254740992"]) {
		process.env.PI_RLM_DEPTH = depth;
		assert.throws(() => rlmExtension({}), /PI_RLM_DEPTH/);
	}
	const root = session(undefined);
	try {
		// Depth is read once, before either the prompt or host is used.
		process.env.PI_RLM_DEPTH = "9";
		await root.fire("session_start", { reason: "startup" }, root.ctx);
		const promptEvent = { systemPromptOptions: { selectedTools: ["ipython"], sections: {} } };
		await root.fire("before_agent_start", promptEvent, root.ctx);
		assert.match(promptEvent.systemPromptOptions.sections.rlm, /rlm\.spawn/);
		const noIpython = { systemPromptOptions: { selectedTools: ["bash"], sections: {} } };
		await root.fire("before_agent_start", noIpython, root.ctx);
		assert.equal(noIpython.systemPromptOptions.sections.rlm, undefined);

		const spawned = await root.cell("h = await rlm.spawn('first')\nimport asyncio\nawait asyncio.sleep(0.2)");
		assert.equal(spawned.result.status, "ok", spawned.result.output);
		assert.deepEqual(spawned.hook.details.children, { spawned: 1, completed: 1 });
		assert.equal(spawned.hook.usage.totalTokens, 7);
		assert.equal(events(root.cwd)[0].depth, "1");
		const gathered = await root.cell("[r] = await rlm.gather([h])\nprint(r.status, type(r.text).__name__, r.text)");
		assert.equal(gathered.result.output.trim(), "ok str done:first");
		assert.equal(gathered.hook.usage, undefined);

		const childCwd = join(root.cwd, "kernel-cwd");
		mkdirSync(childCwd);
		const moved = await root.cell(`import os\nos.chdir(${JSON.stringify(childCwd)})\nhc = await rlm.spawn('moved')\n[rc] = await rlm.gather([hc])\nprint(rc.status)\nos.chdir(${JSON.stringify(root.cwd)})`);
		assert.equal(moved.result.output.trim(), "ok");
		assert.equal(events(root.cwd).find((event) => event.task === "moved").cwd, childCwd);

		const final = await root.cell("await rlm.final({'answer': 42})");
		assert.equal(final.result.status, "ok");
		assert.match(final.result.output, /^\[RLM final\]\n/);
		assert.deepEqual(final.hook.details.final, { answer: 42 });
		assert.equal(final.hook.terminate, undefined);

		const failed = await root.cell("h2 = await rlm.spawn('block-failed-cell')\nawait asyncio.sleep(0.2)\nraise ValueError('fixture')");
		assert.equal(failed.result.status, "error");
		const failedChild = events(root.cwd).find((event) => event.task === "block-failed-cell" && event.type === "start");
		await waitFor(() => !alive(failedChild.pid), "failed-cell child termination");
		assert.equal(failed.hook.details.final, undefined);

		const abort = new AbortController();
		const blocking = root.cell("ha = await rlm.spawn('block-ignore-gather')\nawait rlm.gather([ha])", abort.signal);
		const blockedChild = await waitFor(() => events(root.cwd).find((event) => event.task === "block-ignore-gather" && event.type === "start"), "blocking gather");
		abort.abort();
		const interrupted = await blocking;
		assert.equal(interrupted.result.status, "error", interrupted.result.output);
		await waitFor(() => !alive(blockedChild.pid), "aborted gather child termination");
		assert.equal((await root.cell("print('rlm' in globals())")).result.output.trim(), "True");
	} finally { await root.close(); }

	const limited = session("2");
	try {
		process.env.PI_RLM_DEPTH = "0";
		await limited.fire("session_start", { reason: "startup" }, limited.ctx);
		const promptEvent = { systemPromptOptions: { selectedTools: ["ipython"], sections: {} } };
		await limited.fire("before_agent_start", promptEvent, limited.ctx);
		assert.equal(promptEvent.systemPromptOptions.sections.rlm, undefined);
		const rejected = await limited.cell("h = await rlm.spawn('too deep')\n[r] = await rlm.gather([h])\nprint(r.status, r.error)");
		assert.equal(rejected.result.status, "ok", rejected.result.output);
		assert.match(rejected.result.output.trim(), /^error .*depth limit \(2\)/);
		assert.equal(events(limited.cwd).length, 0);
	} finally { await limited.close(); }
} finally {
	if (savedDepth === undefined) delete process.env.PI_RLM_DEPTH; else process.env.PI_RLM_DEPTH = savedDepth;
}
console.log("kernel: prompt, cross-cell handles, usage, final, cwd, cancellation and depth limit passed");
