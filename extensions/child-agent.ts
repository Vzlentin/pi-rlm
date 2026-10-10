import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import type { AssistantMessage, Usage } from "@earendil-works/pi-ai";
import type { ChildConfig } from "./rlm-host.ts";
import { addUsage, emptyUsage, MAX_RLM_DEPTH } from "./rlm-host.ts";

export interface ChildAgentCommand {
	command: string;
	args: string[];
}

interface ChildAgentRequest {
	model: NonNullable<ChildConfig["model"]>;
	thinkingLevel: ChildConfig["thinkingLevel"];
	cwd: string;
	task: string;
	context: string | null;
	signal: AbortSignal;
	depth: number;
	librlmRoot: string;
}

export interface ChildAgentResult {
	text: string;
	usage: Usage;
	error?: string;
}

export const CHILD_ROLE = "Complete only your assigned task. Return a complete final answer.";
export const TERMINATION_GRACE_MS = 5_000;
const DIAGNOSTIC_LIMIT = 16_384;

export function childPrompt(task: string, context: string | null): string {
	return context === null ? task : `${task}\n\n<context>\n${context}\n</context>`;
}

export async function runChildAgent(
	request: ChildAgentRequest,
	command: ChildAgentCommand = { command: "pi", args: [] },
): Promise<ChildAgentResult> {
	const usage = emptyUsage();
	if (request.signal.aborted) return { text: "", usage, error: "Child cancelled before startup" };
	const env: NodeJS.ProcessEnv = { ...process.env, PI_RLM_DEPTH: String(request.depth + 1), RLM_LIBRLM_ROOT: request.librlmRoot };
	for (const key of Object.keys(env)) {
		if (key.startsWith("RLM_HOST_")) delete env[key];
	}
	const child = spawn(command.command, [
		...command.args,
		"--mode", "json", "--no-session",
		"--model", `${request.model.provider}/${request.model.id}`,
		"--thinking", request.thinkingLevel,
		"--append-system-prompt", CHILD_ROLE,
		"--",
	], { cwd: request.cwd, env, detached: true, stdio: ["pipe", "pipe", "pipe"] });
	const decoder = new StringDecoder("utf8");
	let pending = "";
	let diagnostics = "";
	let failure: string | undefined;
	let lastAssistant: AssistantMessage | undefined;
	let cleanup: Promise<void> | undefined;
	const signalGroup = (signal: NodeJS.Signals) => {
		if (!child.pid) return;
		try { process.kill(-child.pid, signal); } catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ESRCH") failure ??= String(error);
		}
	};
	const terminate = () => {
		if (cleanup) return;
		cleanup = new Promise<void>((resolve) => {
			// Leave time for Pi's detached kernels and deeper children to stop first.
			const grace = TERMINATION_GRACE_MS * Math.max(1, MAX_RLM_DEPTH - request.depth);
			signalGroup("SIGTERM");
			const force = setTimeout(() => {
				signalGroup("SIGKILL");
				clearInterval(poll);
				resolve();
			}, grace);
			const poll = setInterval(() => {
				if (child.pid) {
					try { process.kill(-child.pid, 0); return; } catch {}
				}
				clearTimeout(force);
				clearInterval(poll);
				resolve();
			}, 20);
		});
	};
	const parseRecord = (line: string) => {
		if (!line.trim()) return;
		try {
			const event = JSON.parse(line);
			if (event.type === "compaction_end" && event.result?.usage) addUsage(usage, event.result.usage);
			if (event.type !== "message_end" || !event.message) return;
			if (event.message.usage) addUsage(usage, event.message.usage);
			if (event.message.role === "assistant") lastAssistant = event.message;
		} catch (error) {
			failure ??= `Invalid child JSONL record: ${String(error).slice(0, DIAGNOSTIC_LIMIT)}`;
			terminate();
		}
	};
	child.stdout.on("data", (chunk: Buffer) => {
		pending += decoder.write(chunk);
		let newline: number;
		while ((newline = pending.indexOf("\n")) >= 0) {
			parseRecord(pending.slice(0, newline));
			pending = pending.slice(newline + 1);
		}
	});
	child.stderr.setEncoding("utf8");
	child.stderr.on("data", (chunk: string) => {
		diagnostics = (diagnostics + chunk).slice(-DIAGNOSTIC_LIMIT);
	});
	child.stdin.on("error", (error) => {
		failure ??= `Child stdin failed: ${error.message}`;
		terminate();
	});
	child.on("error", (error) => { failure ??= `Child process failed: ${error.message}`; });
	child.once("exit", terminate);
	const closed = new Promise<void>((resolve) => {
		child.once("close", (code, signal) => {
			pending += decoder.end();
			if (pending) failure ??= "Child JSONL stream ended without LF";
			if (code !== 0) failure ??= `Child process exited with ${signal ?? code}`;
			resolve();
		});
	});
	request.signal.addEventListener("abort", terminate, { once: true });
	if (request.signal.aborted) terminate();
	child.stdin.end(childPrompt(request.task, request.context));
	await closed;
	terminate();
	await cleanup;
	request.signal.removeEventListener("abort", terminate);
	if (request.signal.aborted) failure ??= "Child was cancelled";
	if (!lastAssistant) failure ??= "Child produced no final assistant message";
	else if (lastAssistant.stopReason !== "stop") {
		failure ??= lastAssistant.errorMessage || `Child response was incomplete (${lastAssistant.stopReason})`;
	}
	if (lastAssistant && !Array.isArray(lastAssistant.content)) failure ??= "Child final message has no content";
	if (failure) return { text: "", usage, error: diagnostics.trim() ? `${failure}\n${diagnostics.trim()}` : failure };
	return {
		text: lastAssistant!.content.filter((part) => part.type === "text").map((part) => part.text).join(""),
		usage,
	};
}
