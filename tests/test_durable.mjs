import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { existsSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import * as ai from "@earendil-works/pi-ai";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import * as durable from "@earendil-works/pi-durable";
import { MemoryStorage } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";

// Node cannot strip TypeScript types under node_modules.
const ipythonRoot = resolve(process.env.PI_IPYTHON_ROOT ?? fileURLToPath(new URL("../../pi-ipython", import.meta.url)));
if (!existsSync(join(ipythonRoot, "durable", "index.ts"))) {
	throw new Error(`pi-ipython checkout with durable/index.ts not found at ${ipythonRoot}; set PI_IPYTHON_ROOT`);
}
process.env.RLM_LIBRLM_ROOT ??= fileURLToPath(new URL("../../librlm", import.meta.url));
const { default: ipython } = await import(pathToFileURL(join(ipythonRoot, "durable", "index.ts")).href);
const { default: rlm } = await import("../durable/index.ts");

const context = BACKGROUND_CONTEXT;
const root = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-durable-")));
const faux = fauxProvider({ models: [{ id: "model" }] });
const models = createModels();
models.setProvider(faux.provider);

/** Python that waits until the faux model has received the request of child `name`. */
const awaitStarted = (name) =>
	`import asyncio, os\nwhile not os.path.exists(${JSON.stringify(join(root, `${name}.started`))}):\n    await asyncio.sleep(0.02)`;

// Each root conversation runs the cells listed under its input. Children are named by their task.
const CELLS = {
	same: ["h = await rlm.spawn('fast')\n[r] = await rlm.gather([h])\nprint(r.status, r.text)"],
	cross: ["h = await rlm.spawn('after first cell')\nprint('spawned')", "[r] = await rlm.gather([h])\nprint(r.status, r.text)"],
	cancel: ["h = await rlm.spawn('hang cancel')\nawait rlm.gather([h])"],
	fail: [`h = await rlm.spawn('hang fail')\n${awaitStarted("fail")}\nraise ValueError('fixture')`],
	nested: ["h = await rlm.spawn('agent')\n[r] = await rlm.gather([h])\nprint(r.status, r.text)"],
	restart: ["h = await rlm.spawn('hang restart')\nprint('spawned')"],
};
const AGENT_CELL = `g = await rlm.spawn('hang grandchild')\n${awaitStarted("grandchild")}\nprint('spawned')`;
const started = new Set();
const aborted = new Set();
let firstCellSeen;
const firstCell = new Promise((resolve) => (firstCellSeen = resolve));

const textOf = (message) =>
	typeof message.content === "string"
		? message.content
		: message.content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("");
const call = (code) => fauxAssistantMessage(fauxToolCall("ipython", { code }), { stopReason: "toolUse" });
const abortOf = (signal) =>
	new Promise((resolve) => {
		if (signal?.aborted) resolve();
		signal?.addEventListener("abort", resolve, { once: true });
	});

faux.setResponses(Array.from({ length: 100 }, () => async (request, options) => {
	const messages = request.messages.filter((message) => message.role !== "system");
	const first = textOf(messages[0]);
	const results = messages.filter((message) => message.role === "toolResult").length;
	if (Object.hasOwn(CELLS, first)) {
		if (first === "cross" && results === 1) firstCellSeen();
		return results < CELLS[first].length ? call(CELLS[first][results]) : fauxAssistantMessage("done");
	}
	if (first === "agent") return results === 0 ? call(AGENT_CELL) : fauxAssistantMessage("teal");
	if (first === "fast") return fauxAssistantMessage("teal");
	if (first === "after first cell") {
		// Answers only once the parent's model has the first cell's result, so a held turn never settles.
		await Promise.race([firstCell, abortOf(options?.signal)]);
		return fauxAssistantMessage("teal");
	}
	const name = first.replace(/^hang /, "");
	started.add(name);
	writeFileSync(join(root, `${name}.started`), "");
	await abortOf(options?.signal);
	aborted.add(name);
	return fauxAssistantMessage("", { stopReason: "aborted" });
}));

/** A host with pi-ipython and pi-rlm on one bus. */
async function openHost(storage = new MemoryStorage()) {
	const events = new EventEmitter();
	const packages = [ipython({ durable, ai, events }), rlm({ durable, ai, events })];
	const registry = durable.createRegistry();
	for (const extension of packages.flatMap((item) => item.extensions)) registry.install(extension);
	const harness = await durable.Harness.open(storage, { models, registry }, context);
	const conversation = await harness.root(context, { agent: { model: { provider: "faux", modelId: "model" }, cwd: root } });
	return {
		harness,
		conversation,
		read: (change) => conversation.commit(change, context),
		async close() {
			try {
				await harness.close(context);
			} finally {
				for (const item of packages.reverse()) await item.close();
			}
		},
	};
}

async function within(promise, what, ms = 10_000) {
	let timer;
	const timeout = new Promise((_, reject) => {
		timer = setTimeout(() => reject(new Error(`timed out: ${what}`)), ms);
	});
	try {
		return await Promise.race([promise, timeout]);
	} finally {
		clearTimeout(timer);
	}
}

async function until(check, what, ms = 10_000) {
	const end = Date.now() + ms;
	while (!(await check())) {
		if (Date.now() > end) throw new Error(`timed out: ${what}`);
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
}

async function run(host, input) {
	const submission = await host.conversation.submit({ type: "input", content: input, requestId: input }, context);
	const settled = await within(submission.wait(context), `the ${input} turn`);
	assert.equal(settled.status, "done", settled.reason);
	const page = await host.conversation.entries({}, 50, undefined, context);
	return page.items
		.filter((item) => durable.ToolResultEntry.is(item))
		.map((entry) => entry.model[0].content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n").trim())
		.reverse();
}

try {
	let host = await openHost();
	try {
		assert.deepEqual(await run(host, "same"), ["ok teal"], "rlm.gather returns the child's answer");
		const conversations = await host.read((tx) => tx.scanConversations({}, 10));
		const [child] = conversations.items.filter((item) => item.owner !== undefined);
		const owner = await host.read((tx) => tx.task(child.owner.taskId));
		assert.equal(owner.background, true, "a background task of the conversation owns the child, not the cell's call");
	} finally {
		await host.close();
	}

	host = await openHost();
	try {
		assert.deepEqual(await run(host, "cross"), ["spawned", "ok teal"], "a later cell gathers a child that outlived its cell");
	} finally {
		await host.close();
	}

	host = await openHost();
	try {
		await host.conversation.submit({ type: "input", content: "cancel", requestId: "cancel" }, context);
		await until(() => started.has("cancel"), "the child starts");
		await host.conversation.abort(context);
		await until(() => aborted.has("cancel"), "cancelling the cell aborts the child's model request");
	} finally {
		await host.close();
	}

	host = await openHost();
	try {
		assert.match((await run(host, "fail"))[0], /ValueError: fixture/);
		await until(() => aborted.has("fail"), "a failed cell aborts the child it spawned");
	} finally {
		await host.close();
	}

	host = await openHost();
	try {
		assert.deepEqual(await run(host, "nested"), ["ok teal"]);
		await until(() => aborted.has("grandchild"), "a child's answer stops the grandchild it left running");
	} finally {
		await host.close();
	}

	// The turn ends with the child still running, then the process stops. Nothing can gather the child after a restart.
	const storage = join(root, "restart.sqlite");
	host = await openHost(await openNodeSqliteStorage(storage));
	try {
		assert.deepEqual(await run(host, "restart"), ["spawned"]);
		await until(() => started.has("restart"), "the child starts");
		await within(host.conversation.waitForIdle(context), "the conversation is idle while its child runs");
	} finally {
		await host.close();
	}
	host = await openHost(await openNodeSqliteStorage(storage));
	try {
		host.harness.resume();
		const live = async () => (await host.read((tx) => tx.scanTasks({}, 100))).items.filter((task) => task.state.status !== "terminal");
		await until(async () => (await live()).length === 0, "a restart stops the children of the last process");
	} finally {
		await host.close();
	}

	console.log("durable: background-owned child, cross-cell gather, cancellation, failed cell, depth 2 and restart passed");
} finally {
	rmSync(root, { recursive: true, force: true });
}
