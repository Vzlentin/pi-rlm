/**
 * pi-rlm for hosts that run agents on pi-durable, such as the workflows engine.
 *
 * pi-ipython's durable adapter emits its cell and kernel events on the host's bus. Each conversation gets its own host
 * bridge, and each child is a conversation owned by the ipython call that spawned it, so cancelling the cell stops it.
 * Durable hosts cancel all children at the end of each cell, so they do not support cross-cell handles.
 */
import type { Context } from "@earendil-works/chord";
import type * as Ai from "@earendil-works/pi-ai";
import type * as Durable from "@earendil-works/pi-durable";
import type { ConversationId, ToolExecutionApi } from "@earendil-works/pi-durable";
import { CHILD_ROLE, childPrompt, type ChildAgentResult } from "../extensions/child-agent.ts";
import { loadRlmPrompt, resolveLibrlm, rlmSection, syncLibrlm, type RlmPrompt } from "../extensions/librlm.ts";
import { emptyUsage, MAX_RLM_DEPTH, RlmHostBridge, type ChildRequest } from "../extensions/rlm-host.ts";
import { KERNEL_STARTING_EVENT, kernelStartupCode } from "../extensions/rlm.ts";

export interface DurableHost {
	readonly durable: typeof Durable;
	readonly ai: typeof Ai;
	readonly events: { on(channel: string, handler: (data: any) => void): unknown };
}

interface CellStartEvent {
	conversationId: ConversationId;
	api: ToolExecutionApi;
	context: Context;
}

interface KernelStartingEvent {
	conversationId: ConversationId;
	env: Record<string, string>;
	startupCode: string[];
	waitFor(promise: Promise<unknown>): void;
}

interface Cell {
	api: ToolExecutionApi;
	context: Context;
}

interface ConversationState {
	host: RlmHostBridge;
	cell?: Cell;
}

export default function rlm({ durable, events }: DurableHost) {
	const location = resolveLibrlm();
	let librlm: Promise<RlmPrompt> | undefined;
	const prepare = () => librlm ??= syncLibrlm(location, console.error).then(() => loadRlmPrompt(location.root));
	const states = new Map<string, ConversationState>();
	// Child conversation ID to RLM depth; roots are absent, at depth 0. This is lost on restart, which is accepted:
	// reopening interrupts the ipython call, which is not replay-safe, and that aborts the children it owns.
	const depths = new Map<string, number>();
	const depthOf = (id: ConversationId) => depths.get(String(id)) ?? 0;

	function state(id: ConversationId): ConversationState {
		const key = String(id);
		let found = states.get(key);
		if (!found) {
			const depth = depthOf(id);
			found = {
				host: new RlmHostBridge(location.root, depth, undefined, (request) => {
					const cell = states.get(key)?.cell;
					if (!cell) throw new Error("no running ipython cell");
					return (signal) => runChild(cell, depth, request, signal);
				}),
			};
			states.set(key, found);
		}
		return found;
	}

	async function runChild({ api, context }: Cell, depth: number, request: ChildRequest, signal: AbortSignal): Promise<ChildAgentResult> {
		// A task-owned conversation copies its owner's agent: model, thinking level, extensions and cwd.
		const child = await api.commit(async (tx) => {
			const created = await tx.createConversation({ ownership: { kind: "task", taskId: api.taskId } });
			await durable.configure(tx, created.id, { cwd: request.cwd, instructions: CHILD_ROLE });
			return created.id;
		}, context);
		depths.set(String(child), depth + 1);
		const handle = await api.conversation(child, context);
		if (!handle) throw new Error("The child conversation is missing");
		const abort = () => void handle.abort(context).catch(() => {});
		signal.addEventListener("abort", abort, { once: true });
		try {
			if (signal.aborted) throw new Error("Child cancelled before startup");
			const submission = await handle.submit({ type: "input", content: childPrompt(request.task, request.context) }, context);
			const settled = await submission.wait(context);
			if (settled.status !== "done" || settled.type !== "input") {
				throw new Error(`Child was not answered: ${settled.reason ?? "unknown reason"}`);
			}
			const { answer: answerId } = settled;
			const answer = await api.commit((tx) => tx.entry(durable.AssistantEntry, answerId), context);
			const message = answer?.model?.[0];
			if (message?.role !== "assistant") throw new Error("Child produced no final assistant message");
			if (message.stopReason !== "stop") {
				throw new Error(message.errorMessage || `Child response was incomplete (${message.stopReason})`);
			}
			// pi-durable already adds the child's spend to the conversation's usage.
			return {
				text: message.content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join(""),
				usage: emptyUsage(),
			};
		} finally {
			signal.removeEventListener("abort", abort);
		}
	}

	events.on("ipython:cell-start", ({ conversationId, api, context }: CellStartEvent) => {
		state(conversationId).cell = { api, context };
	});

	events.on("ipython:cell-end", ({ conversationId }: { conversationId: ConversationId }) => {
		const current = states.get(String(conversationId));
		if (!current) return;
		current.cell = undefined;
		current.host.cancelChildren();
	});

	events.on(KERNEL_STARTING_EVENT, (event: KernelStartingEvent) => {
		event.startupCode.push(kernelStartupCode(location.root));
		event.waitFor((async () => {
			await prepare();
			const { host } = state(event.conversationId);
			await host.ensureStarted();
			Object.assign(event.env, host.environment);
		})());
	});

	const section = durable.section("rlm", async (input) => {
		if (depthOf(input.conversationId) >= MAX_RLM_DEPTH) return undefined;
		if (!input.agent.tools.some((tool) => tool.name === "ipython")) return undefined;
		return rlmSection(await prepare());
	});

	return {
		extensions: [durable.defineExtension({ name: "pi-rlm", sections: [section] })],
		/** Stops every host bridge and the children it still runs. */
		async close(): Promise<void> {
			const hosts = [...states.values()].map((current) => current.host);
			states.clear();
			await Promise.allSettled(hosts.map((host) => host.shutdown()));
		},
	};
}
