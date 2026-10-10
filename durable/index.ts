/**
 * pi-rlm for hosts that run agents on pi-durable, such as the workflows engine.
 *
 * pi-ipython's durable adapter emits its cell and kernel events on the host's bus. Each conversation gets its own host
 * bridge and one background owner task, which owns the child conversations and runs them. A running child does not
 * hold the cell's call or the turn open, and a later cell can gather it, as in Pi. When a cell fails or is cancelled,
 * the kernel cancels the children it spawned and the bridge aborts their conversations.
 *
 * The owner task stays open between cells because librlm can start a queued child after its cell has ended, and the
 * cell's call cannot create or run anything once it has ended. Costs: one open task per conversation, and until
 * pi-durable has `abandonOnRestart`, a normal stop ends it as failed so that pi-durable aborts the children it owns.
 */
import { randomUUID } from "node:crypto";
import type { Context } from "@earendil-works/chord";
import type * as Ai from "@earendil-works/pi-ai";
import type * as Durable from "@earendil-works/pi-durable";
import type { ConversationId, TaskRuntime, ToolExecutionApi } from "@earendil-works/pi-durable";
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

type OwnerInput = { key: string };
type OwnerState = { phase: "serve" };

/** A running owner task invocation. Its runtime stays usable between cells, until the owner stops. */
interface Owner {
	runtime: TaskRuntime<OwnerInput, OwnerState, null, object>;
	context: Context;
}

interface OwnerWaiter {
	started(owner: Owner): void;
	ended(): void;
	stop?: () => void;
}

interface ConversationState {
	host: RlmHostBridge;
	cell?: Cell;
	owner?: Promise<Owner>;
}

export default function rlm({ durable, events }: DurableHost) {
	const location = resolveLibrlm();
	let librlm: Promise<RlmPrompt> | undefined;
	const prepare = () => librlm ??= syncLibrlm(location, console.error).then(() => loadRlmPrompt(location.root));
	const states = new Map<string, ConversationState>();
	// Child conversation ID to RLM depth; roots are absent, at depth 0. Lost on restart, like the kernels and handles.
	const depths = new Map<string, number>();
	const depthOf = (id: ConversationId) => depths.get(String(id)) ?? 0;
	// Owner key to this process's waiter. After a restart an owner task finds none, so it fails and aborts its children.
	const waiters = new Map<string, OwnerWaiter>();

	const OwnerTask = durable.defineTask<OwnerInput, OwnerState, null>({
		name: "pi-rlm.children",
		version: 1,
		initial: () => ({ phase: "serve" }),
		phases: {
			serve: async (task, runtime, context) => {
				const waiter = waiters.get(task.input.key);
				try {
					if (waiter) {
						await new Promise<void>((resolve, reject) => {
							const signal = context.abortSignal;
							if (signal?.aborted) return reject(signal.reason);
							signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
							waiter.stop = resolve;
							waiter.started({ runtime, context });
						});
					}
					// A failed outcome aborts the child conversations this task still owns.
					await runtime.commit(
						() => ({ status: "terminal", outcome: { status: "failed", error: { message: "RLM children stopped" } } }),
						context,
					);
				} finally {
					waiter?.ended();
				}
			},
		},
		abort: (_task, runtime, context) =>
			runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), context),
	});

	function state(id: ConversationId): ConversationState {
		const key = String(id);
		let found = states.get(key);
		if (!found) {
			const depth = depthOf(id);
			const current: ConversationState = {
				host: new RlmHostBridge(location.root, depth, undefined, (request) => {
					const owner = current.owner;
					if (!owner) throw new Error("RLM children have no owner task in this conversation");
					return (signal) => owner.then((running) => runChild(running, depth, request, signal));
				}),
			};
			states.set(key, found = current);
		}
		return found;
	}

	/** Starts the conversation's owner task with the running cell's call, unless one already runs. */
	function ensureOwner(current: ConversationState): Promise<Owner> | undefined {
		if (current.owner) return current.owner;
		const cell = current.cell;
		if (!cell) return undefined;
		const key = randomUUID();
		const owner: Promise<Owner> = new Promise<Owner>((resolve, reject) => {
			const end = (error: unknown) => {
				waiters.delete(key);
				if (current.owner === owner) current.owner = undefined;
				reject(error);
			};
			waiters.set(key, { started: resolve, ended: () => end(new Error("The RLM owner task ended")) });
			cell.api
				.createTask(OwnerTask, { key }, { ownership: { kind: "conversation" }, background: true }, cell.context)
				.catch(end);
		});
		owner.catch(() => {});
		current.owner = owner;
		return owner;
	}

	async function runChild({ runtime, context }: Owner, depth: number, request: ChildRequest, signal: AbortSignal): Promise<ChildAgentResult> {
		// A task-owned conversation copies its owner's agent: model, thinking level, extensions and cwd.
		let child: ConversationId | undefined;
		await runtime.commit(async (tx) => {
			const created = await tx.createConversation({ ownership: { kind: "task", taskId: runtime.taskId } });
			await durable.configure(tx, created.id, { cwd: request.cwd, instructions: CHILD_ROLE });
			child = created.id;
			return undefined;
		}, context);
		depths.set(String(child), depth + 1);
		const handle = await runtime.conversation(child!, context);
		if (!handle) throw new Error("The child conversation is missing");
		// `background` also reaches the child's own owner task, so the child's children stop too.
		const abort = () => void handle.abort(context, { background: true }).catch(() => {});
		signal.addEventListener("abort", abort, { once: true });
		try {
			if (signal.aborted) throw new Error("Child cancelled before startup");
			const submission = await handle.submit({ type: "input", content: childPrompt(request.task, request.context) }, context);
			const settled = await submission.wait(context);
			if (settled.status !== "done" || settled.type !== "input") {
				throw new Error(`Child was not answered: ${settled.reason ?? "unknown reason"}`);
			}
			const { answer: answerId } = settled;
			let message: Ai.Message | undefined;
			await runtime.commit(async (tx) => {
				message = (await tx.entry(durable.AssistantEntry, answerId))?.model?.[0];
				return undefined;
			}, context);
			if (message?.role !== "assistant") throw new Error("Child produced no final assistant message");
			if (message.stopReason !== "stop") {
				throw new Error(message.errorMessage || `Child response was incomplete (${message.stopReason})`);
			}
			// pi-durable records the child's spend in the child conversation's own usage.
			return {
				text: message.content.flatMap((part) => (part.type === "text" ? [part.text] : [])).join(""),
				usage: emptyUsage(),
			};
		} finally {
			signal.removeEventListener("abort", abort);
			// A Pi child exits after its answer, which stops its own children. Do the same here.
			if (!signal.aborted) abort();
		}
	}

	events.on("ipython:cell-start", ({ conversationId, api, context }: CellStartEvent) => {
		const current = state(conversationId);
		current.cell = { api, context };
		void ensureOwner(current);
	});

	events.on("ipython:cell-end", ({ conversationId }: { conversationId: ConversationId }) => {
		const current = states.get(String(conversationId));
		if (current) current.cell = undefined;
	});

	events.on(KERNEL_STARTING_EVENT, (event: KernelStartingEvent) => {
		event.startupCode.push(kernelStartupCode(location.root));
		event.waitFor((async () => {
			await prepare();
			const current = state(event.conversationId);
			await current.host.ensureStarted();
			// The cell still runs here, so its call can create the owner task.
			await ensureOwner(current);
			Object.assign(event.env, current.host.environment);
		})());
	});

	const section = durable.section("rlm", async (input) => {
		if (depthOf(input.conversationId) >= MAX_RLM_DEPTH) return undefined;
		if (!input.agent.tools.some((tool) => tool.name === "ipython")) return undefined;
		return rlmSection(await prepare());
	});

	return {
		extensions: [durable.defineExtension({ name: "pi-rlm", sections: [section], tasks: [OwnerTask] })],
		/** Stops every host bridge and the children it still runs, then the owner tasks. */
		async close(): Promise<void> {
			const hosts = [...states.values()].map((current) => current.host);
			states.clear();
			await Promise.allSettled(hosts.map((host) => host.shutdown()));
			for (const waiter of waiters.values()) waiter.stop?.();
		},
	};
}
