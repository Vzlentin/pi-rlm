import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { EventEmitter } from "node:events";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

// pi-ipython is a sibling checkout: Node cannot strip TypeScript types under node_modules.
const ipythonRoot = resolve(process.env.PI_IPYTHON_ROOT ?? fileURLToPath(new URL("../../pi-ipython", import.meta.url)));
if (!existsSync(join(ipythonRoot, "extensions", "kernel-runtime.ts"))) {
	throw new Error(`pi-ipython checkout not found at ${ipythonRoot}; set PI_IPYTHON_ROOT`);
}
process.env.RLM_LIBRLM_ROOT ??= join(homedir(), "Dev", "librlm");
const { KernelRuntime } = await import(pathToFileURL(join(ipythonRoot, "extensions", "kernel-runtime.ts")).href);
const { default: rlmExtension } = await import("../extensions/rlm.ts");

const exec = promisify(execFile);
const bus = new EventEmitter();
const handlers = new Map();
const pi = {
	events: { emit: (channel, data) => bus.emit(channel, data), on: (channel, handler) => bus.on(channel, handler) },
	on: (name, handler) => handlers.set(name, [...(handlers.get(name) ?? []), handler]),
	getThinkingLevel: () => "off",
	async exec(command, args, options) {
		try {
			return { ...await exec(command, args, options), code: 0 };
		} catch (error) {
			return { code: error.code ?? 1, stderr: error.stderr ?? error.message, stdout: error.stdout ?? "" };
		}
	},
};
const fire = async (name, event, ctx) => {
	let result;
	for (const handler of handlers.get(name) ?? []) result = (await handler(event, ctx)) ?? result;
	return result;
};

const usage = (tokens) => ({
	input: tokens, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: tokens,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});
const aborted = [];
const modelRegistry = {
	streamSimple(_model, context, { signal }) {
		const prompt = context.messages[0].content[0].text;
		return {
			result: () => new Promise((resolveResult) => {
				const answer = (stopReason, text) => resolveResult({
					role: "assistant", content: [{ type: "text", text }], usage: usage(stopReason === "stop" ? 7 : 0),
					stopReason, errorMessage: stopReason === "stop" ? undefined : "aborted",
				});
				if (!prompt.startsWith("block")) return answer("stop", `done:${prompt}`);
				signal.addEventListener("abort", () => {
					aborted.push(prompt);
					answer("aborted", "");
				}, { once: true });
			}),
		};
	},
};
const cwd = mkdtempSync(join(tmpdir(), "pi-rlm-kernel-"));
const ctx = { cwd, model: { id: "fixture", reasoning: false }, modelRegistry, hasUI: false };

rlmExtension(pi);
await fire("session_start", { reason: "startup" }, ctx);
const kernel = new KernelRuntime(pi);
let index = 0;
async function cell(code) {
	const id = `call-${++index}`;
	await fire("tool_execution_start", { toolName: "ipython", toolCallId: id, args: { code } }, ctx);
	const { result } = await kernel.execute(id, code, cwd, undefined, () => {}, () => {});
	const event = {
		toolName: "ipython", toolCallId: id, input: { code }, isError: result.status !== "ok",
		content: [{ type: "text", text: result.output }], details: { status: result.status },
	};
	return { result, hook: (await fire("tool_result", event, ctx)) ?? {} };
}

const promptEvent = { systemPromptOptions: { selectedTools: ["ipython"], sections: {} } };
await fire("before_agent_start", promptEvent, ctx);
assert.match(promptEvent.systemPromptOptions.sections.rlm, /rlm\.spawn/);
const noIpython = { systemPromptOptions: { selectedTools: ["bash"], sections: {} } };
await fire("before_agent_start", noIpython, ctx);
assert.equal(noIpython.systemPromptOptions.sections.rlm, undefined);

try {
	// A handle spawned in one cell is gathered in the next; usage lands on the call where the child settled.
	const spawned = await cell("h = await rlm.spawn('first')\nimport asyncio\nawait asyncio.sleep(0.2)");
	assert.equal(spawned.result.status, "ok", spawned.result.output);
	assert.deepEqual(spawned.hook.details.children, { spawned: 1, completed: 1 });
	assert.equal(spawned.hook.usage.totalTokens, 7);
	const gathered = await cell("[r] = await rlm.gather([h])\nprint(r.status, r.text)");
	assert.equal(gathered.result.output.trim(), "ok done:first");
	assert.equal(gathered.hook.usage, undefined);

	// rlm.final prints its value and leaves the turn to the model.
	const final = await cell("await rlm.final({'answer': 42})");
	assert.equal(final.result.status, "ok");
	assert.match(final.result.output, /^\[RLM final\]\n/);
	assert.deepEqual(final.hook.details.final, { answer: 42 });
	assert.equal(final.hook.terminate, undefined);

	// A failed cell cancels its children through the closed sockets.
	const failed = await cell("h2 = await rlm.spawn('block me')\nimport asyncio\nawait asyncio.sleep(0.2)\nraise ValueError('fixture')");
	assert.equal(failed.result.status, "error");
	const deadline = Date.now() + 10_000;
	while (!aborted.length && Date.now() < deadline) await new Promise((r) => setTimeout(r, 50));
	assert.deepEqual(aborted, ["block me"]);
	assert.equal(failed.hook.details.final, undefined);
	assert.equal((await cell("print('rlm' in globals())")).result.output.trim(), "True");
} finally {
	await kernel.shutdown();
	await fire("session_shutdown", {}, ctx);
	rmSync(cwd, { recursive: true, force: true });
}
console.log("kernel: prompt section, cross-cell handles, usage, final and failed-cell cancellation passed");
