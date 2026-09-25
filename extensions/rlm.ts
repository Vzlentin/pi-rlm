import type { Usage } from "@earendil-works/pi-ai";
import { formatSize, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { createChildCompleter } from "./child-completion.ts";
import { loadRlmPrompt, resolveLibrlm, syncLibrlm, type RlmPrompt } from "./librlm.ts";
import {
	addUsage,
	emptyUsage,
	HOST_PROTOCOL_VERSION,
	MAX_CHILDREN_RUNNING,
	MAX_CHILD_REQUEST_BYTES,
	MAX_CHILD_TEXT_BYTES,
	MAX_LIVE_HANDLES,
	RlmHostBridge,
} from "./rlm-host.ts";

/** pi-ipython's hook, emitted on `pi.events` before every kernel start. */
export const KERNEL_STARTING_EVENT = "ipython:kernel-starting";
interface KernelStartingEvent {
	env: Record<string, string>;
	startupCode: string[];
	waitFor(promise: Promise<unknown>): void;
}

export const FINAL_MARKER = "[RLM final]";
const STATUS_KEY = "pi-rlm";

export function kernelStartupCode(root: string): string {
	return `def _pi_rlm_setup():
    import json, sys
    root = ${JSON.stringify(root)}
    if root not in sys.path:
        sys.path.append(root)
    import rlm.ipython_extension as ext
    if ext.HOST_PROTOCOL_VERSION != ${HOST_PROTOCOL_VERSION}:
        raise RuntimeError(
            f"librlm host protocol {ext.HOST_PROTOCOL_VERSION} is not supported by pi-rlm "
            f"(expects ${HOST_PROTOCOL_VERSION}); update pi-rlm or librlm"
        )
    shell = get_ipython()
    shell.run_line_magic("load_ext", "rlm.ipython_extension")

    def show_final(result):
        final = ext.last_summary.final
        if result.success and final.is_present:
            print(${JSON.stringify(FINAL_MARKER)} + "\\n" + json.dumps(final.value, ensure_ascii=False, indent=2))

    shell.events.register("post_run_cell", show_final)
_pi_rlm_setup()
del _pi_rlm_setup`;
}

/** The value printed by the kernel's final hook at the end of a cell's output. */
export function finalFromOutput(text: string): { value: unknown } | undefined {
	const index = text.lastIndexOf(`${FINAL_MARKER}\n`);
	if (index < 0 || (index > 0 && text[index - 1] !== "\n")) return undefined;
	try {
		return { value: JSON.parse(text.slice(index + FINAL_MARKER.length + 1)) };
	} catch {
		return undefined;
	}
}

function hasUsage(usage: Usage): boolean {
	return usage.totalTokens > 0 || usage.cost.total > 0;
}

export default function rlmExtension(pi: ExtensionAPI) {
	const location = resolveLibrlm();
	let latestCtx: ExtensionContext | undefined;
	let host: RlmHostBridge | undefined;
	let librlm: Promise<RlmPrompt> | undefined;
	const warn = (message: string) => {
		if (latestCtx?.hasUI) latestCtx.ui.notify(message, "warning");
		else console.error(message);
	};
	const prepare = () => librlm ??= syncLibrlm(location, warn).then(() => loadRlmPrompt(location.root));
	const getHost = (ctx: ExtensionContext) => {
		if (host) return host;
		host = new RlmHostBridge(createChildCompleter(ctx.modelRegistry));
		host.onActivity = (activity, running) => {
			const status = running > 0 || activity.spawned > activity.completed
				? `RLM children: ${running} running, ${activity.completed}/${activity.spawned} done`
				: undefined;
			if (latestCtx?.hasUI) latestCtx.ui.setStatus(STATUS_KEY, status);
		};
		return host;
	};

	pi.on("session_start", (_event, ctx) => {
		latestCtx = ctx;
		// Clone or update in the background; the prompt and kernel start await it.
		prepare().catch(() => {});
	});

	pi.events.on(KERNEL_STARTING_EVENT, (data) => {
		const event = data as KernelStartingEvent;
		event.startupCode.push(kernelStartupCode(location.root));
		event.waitFor((async () => {
			await prepare();
			if (!latestCtx) throw new Error("pi-rlm has no session context for child calls");
			const bridge = getHost(latestCtx);
			await bridge.ensureStarted();
			Object.assign(event.env, bridge.environment);
		})());
	});

	pi.on("before_agent_start", async (event, ctx) => {
		latestCtx = ctx;
		if (!event.systemPromptOptions.selectedTools.includes("ipython")) return;
		const prompt = await prepare();
		event.systemPromptOptions.sections.rlm = [
			prompt.rlmApi,
			prompt.rlmGuidance,
			`Child calls are limited to ${MAX_CHILDREN_RUNNING} concurrent/${MAX_LIVE_HANDLES} live handles, ${formatSize(MAX_CHILD_REQUEST_BYTES)} input, ${formatSize(MAX_CHILD_TEXT_BYTES)} returned text, and a 5-minute deadline. rlm.final prints its value at the end of the cell's output; it does not end the turn.`,
		].join("\n\n");
	});

	pi.on("tool_execution_start", (event, ctx) => {
		if (event.toolName !== "ipython") return;
		latestCtx = ctx;
		getHost(ctx).setConfig({ cwd: ctx.cwd, model: ctx.model, thinkingLevel: pi.getThinkingLevel() });
	});

	pi.on("tool_result", (event, ctx) => {
		if (event.toolName !== "ipython") return;
		const activity = host?.takeActivity() ?? { spawned: 0, completed: 0, usage: emptyUsage() };
		if (host && host.running === 0 && ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, undefined);
		const text = event.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		const final = event.isError ? undefined : finalFromOutput(text);
		const details: Record<string, unknown> = {
			...(event.details as Record<string, unknown> | undefined),
			children: { spawned: activity.spawned, completed: activity.completed },
		};
		if (final) details.final = final.value;
		if (!hasUsage(activity.usage)) return { details };
		details.nestedUsage = activity.usage;
		const usage = emptyUsage();
		if (event.usage) addUsage(usage, event.usage);
		addUsage(usage, activity.usage);
		return { details, usage };
	});

	pi.on("session_shutdown", async () => {
		const bridge = host;
		host = undefined;
		await bridge?.shutdown();
	});
}
