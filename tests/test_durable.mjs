import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { existsSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import * as ai from "@earendil-works/pi-ai";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import * as durable from "@earendil-works/pi-durable";
import { MemoryStorage } from "@earendil-works/pi-durable";

// Node cannot strip TypeScript types under node_modules.
const ipythonRoot = resolve(process.env.PI_IPYTHON_ROOT ?? fileURLToPath(new URL("../../pi-ipython", import.meta.url)));
if (!existsSync(join(ipythonRoot, "durable", "index.ts"))) {
	throw new Error(`pi-ipython checkout with durable/index.ts not found at ${ipythonRoot}; set PI_IPYTHON_ROOT`);
}
process.env.RLM_LIBRLM_ROOT ??= join(homedir(), "Dev", "librlm");
const { default: ipython } = await import(pathToFileURL(join(ipythonRoot, "durable", "index.ts")).href);
const { default: rlm } = await import("../durable/index.ts");

const context = BACKGROUND_CONTEXT;
const root = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-durable-")));
const faux = fauxProvider({ models: [{ id: "model" }] });
const models = createModels();
models.setProvider(faux.provider);

/** A host with pi-ipython and pi-rlm on one bus; `taskIds` gets the owning task of every ipython cell. */
async function openHost() {
	const events = new EventEmitter();
	const taskIds = [];
	events.on("ipython:cell-start", ({ api }) => taskIds.push(api.taskId));
	const packages = [ipython({ durable, ai, events }), rlm({ durable, ai, events })];
	const registry = durable.createRegistry();
	for (const extension of packages.flatMap((item) => item.extensions)) registry.install(extension);
	const harness = await durable.Harness.open(new MemoryStorage(), { models, registry }, context);
	const conversation = await harness.root(context, { agent: { model: { provider: "faux", modelId: "model" }, cwd: root } });
	return {
		conversation,
		taskIds,
		async close() {
			try {
				await harness.close(context);
			} finally {
				for (const item of packages.reverse()) await item.close();
			}
		},
	};
}

function toolResultText(entries) {
	const entry = entries.find((item) => durable.ToolResultEntry.is(item));
	assert.ok(entry, "the turn has a tool result");
	return entry.model[0].content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("\n");
}

try {
	let host = await openHost();
	try {
		const code = "h = await rlm.spawn('name the colour')\n[r] = await rlm.gather([h])\nprint(r.status, r.text)";
		faux.setResponses([
			fauxAssistantMessage(fauxToolCall("ipython", { code }), { stopReason: "toolUse" }),
			fauxAssistantMessage("child says teal"),
			fauxAssistantMessage("done"),
		]);
		const submission = await host.conversation.submit({ type: "input", content: "spawn", requestId: "spawn" }, context);
		const settled = await submission.wait(context);
		assert.equal(settled.status, "done", settled.reason);
		const page = await host.conversation.entries({}, 50, undefined, context);
		assert.match(toolResultText(page.items), /^ok child says teal$/m, "rlm.gather returns the child's answer");
		assert.equal(host.taskIds.length, 1);
		const owned = await host.conversation.commit((tx) => tx.scanConversations({ ownerTaskId: host.taskIds[0] }, 10), context);
		assert.equal(owned.items.length, 1, "the child conversation is owned by the cell's call");
	} finally {
		await host.close();
	}

	host = await openHost();
	try {
		let started;
		const childStarted = new Promise((resolve) => (started = resolve));
		let childAborted = false;
		const code = "h = await rlm.spawn('wait forever')\nawait rlm.gather([h])";
		faux.setResponses([
			fauxAssistantMessage(fauxToolCall("ipython", { code }), { stopReason: "toolUse" }),
			async (_context, options) => {
				started();
				await new Promise((resolve) => {
					if (options?.signal?.aborted) resolve();
					options?.signal?.addEventListener("abort", resolve, { once: true });
				});
				childAborted = true;
				return fauxAssistantMessage("", { stopReason: "aborted" });
			},
		]);
		await host.conversation.submit({ type: "input", content: "cancel", requestId: "cancel" }, context);
		await childStarted;
		await host.conversation.abort(context);
		assert.equal(childAborted, true, "cancelling the cell aborts the child's model request");
	} finally {
		await host.close();
	}

	host = await openHost();
	try {
		let childAborted = false;
		const code = "h = await rlm.spawn('x')\nimport asyncio\nawait asyncio.sleep(0.2)\nraise ValueError('fixture')";
		faux.setResponses([
			fauxAssistantMessage(fauxToolCall("ipython", { code }), { stopReason: "toolUse" }),
			async (_context, options) => {
				await new Promise((resolve) => {
					if (options?.signal?.aborted) resolve();
					options?.signal?.addEventListener("abort", resolve, { once: true });
				});
				childAborted = true;
				return fauxAssistantMessage("", { stopReason: "aborted" });
			},
			fauxAssistantMessage("done"),
		]);
		const submission = await host.conversation.submit({ type: "input", content: "fail", requestId: "fail" }, context);
		let timer;
		const timeout = new Promise((_, reject) => {
			timer = setTimeout(() => reject(new Error("the failed cell did not settle")), 10_000);
		});
		try {
			await Promise.race([submission.wait(context), timeout]);
		} finally {
			clearTimeout(timer);
		}
		assert.equal(childAborted, true, "a failed cell aborts the child it spawned");
	} finally {
		await host.close();
	}

	console.log("durable: task-owned child, gather answer, cancellation and failed-cell cancellation passed");
} finally {
	rmSync(root, { recursive: true, force: true });
}
