import { randomBytes, timingSafeEqual } from "node:crypto";
import { statSync } from "node:fs";
import { chmod, mkdtemp, rm } from "node:fs/promises";
import { createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Usage } from "@earendil-works/pi-ai";
import { formatSize, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { runChildAgent, type ChildAgentCommand } from "./child-agent.ts";

export const MAX_CHILDREN_RUNNING = 4;
export const MAX_LIVE_HANDLES = 16;
export const MAX_CHILD_REQUEST_BYTES = 1024 * 1024;
export const MAX_CHILD_TEXT_BYTES = 256 * 1024;
const MAX_HOST_REQUEST_BYTES = MAX_CHILD_REQUEST_BYTES + 64 * 1024;
export const MAX_HOST_RESPONSE_BYTES = 5 * 1024 * 1024;
const CHILD_DEADLINE_MS = 5 * 60_000;
const HOST_REQUEST_LINE_TIMEOUT_MS = 10_000;
export const HOST_PROTOCOL_VERSION = 2;

export const MAX_RLM_DEPTH = 2;

type ActiveModel = NonNullable<ExtensionContext["model"]>;
type ThinkingLevel = NonNullable<ExtensionContext["thinkingLevel"]>;

export interface ChildConfig {
	cwd: string;
	model?: ActiveModel;
	thinkingLevel: ThinkingLevel;
}

interface ChildResult {
	status: "ok" | "error" | "cancelled" | "timeout";
	text: string | null;
	error: string | null;
	usage: Usage;
	elapsed_ms: number;
	truncated: boolean;
}

interface ChildRecord {
	controller: AbortController;
	promise: Promise<ChildResult>;
	timedOut: boolean;
}

/** Child work observed since the last `takeActivity()`. */
export interface ChildActivity {
	spawned: number;
	completed: number;
	usage: Usage;
}

interface HostRequest {
	version: 2;
	auth: string;
	id: string;
	execution_id: string;
	op: "complete";
	task: string;
	context: string | null;
	cwd: string;
}

interface HostResponse {
	version: 2;
	id: string;
	ok: boolean;
	result?: unknown;
	error?: string;
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

export function emptyUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

export function addUsage(target: Usage, source: Usage): void {
	target.input += source.input;
	target.output += source.output;
	target.cacheRead += source.cacheRead;
	target.cacheWrite += source.cacheWrite;
	target.totalTokens += source.totalTokens;
	target.cost.input += source.cost.input;
	target.cost.output += source.cost.output;
	target.cost.cacheRead += source.cost.cacheRead;
	target.cost.cacheWrite += source.cost.cacheWrite;
	target.cost.total += source.cost.total;
	if (source.cacheWrite1h !== undefined) target.cacheWrite1h = (target.cacheWrite1h ?? 0) + source.cacheWrite1h;
	if (source.reasoning !== undefined) target.reasoning = (target.reasoning ?? 0) + source.reasoning;
}
function utf8Bytes(value: string): number {
	return Buffer.byteLength(value, "utf8");
}

function truncateUtf8(value: string, maxBytes: number): { text: string; truncated: boolean } {
	const encoded = Buffer.from(value, "utf8");
	if (encoded.byteLength <= maxBytes) return { text: value, truncated: false };
	const notice = "\n[child response truncated at 256 KiB]";
	const budget = Math.max(0, maxBytes - Buffer.byteLength(notice));
	return { text: encoded.subarray(0, budget).toString("utf8") + notice, truncated: true };
}

function boundedError(error: unknown): string {
	return truncateUtf8(errorText(error), 16 * 1024).text;
}

class Semaphore {
	private running = 0;
	private readonly limit: number;
	private readonly waiters: Array<{
		resolve: (release: () => void) => void;
		reject: (error: Error) => void;
		signal: AbortSignal;
		onAbort: () => void;
	}> = [];

	constructor(limit: number) {
		this.limit = limit;
	}

	acquire(signal: AbortSignal): Promise<() => void> {
		if (signal.aborted) return Promise.reject(new Error("Child cancelled before admission"));
		if (this.running < this.limit) {
			this.running += 1;
			return Promise.resolve(this.releaseFunction());
		}
		return new Promise((resolve, reject) => {
			const waiter = {
				resolve,
				reject,
				signal,
				onAbort: () => {
					const index = this.waiters.indexOf(waiter);
					if (index >= 0) this.waiters.splice(index, 1);
					reject(new Error("Child cancelled before admission"));
				},
			};
			signal.addEventListener("abort", waiter.onAbort, { once: true });
			this.waiters.push(waiter);
		});
	}

	private releaseFunction(): () => void {
		let released = false;
		return () => {
			if (released) return;
			released = true;
			while (this.waiters.length > 0) {
				const waiter = this.waiters.shift()!;
				waiter.signal.removeEventListener("abort", waiter.onAbort);
				if (waiter.signal.aborted) continue;
				waiter.resolve(this.releaseFunction());
				return;
			}
			this.running -= 1;
		};
	}
}

export class RlmHostBridge {
	private server?: Server;
	private socketDirectory?: string;
	private socketPath?: string;
	private readonly authToken = randomBytes(32).toString("hex");
	private readonly children = new Set<ChildRecord>();
	private readonly sockets = new Set<Socket>();
	private readonly limiter = new Semaphore(MAX_CHILDREN_RUNNING);
	private starting?: Promise<void>;
	private disposed = false;
	private config?: ChildConfig;
	private activity: ChildActivity = { spawned: 0, completed: 0, usage: emptyUsage() };
	/** Called whenever a child starts or settles. */
	onActivity?: (activity: ChildActivity, running: number) => void;

	private readonly librlmRoot: string;
	private readonly depth: number;
	private readonly command?: ChildAgentCommand;

	constructor(librlmRoot: string, depth = 0, command?: ChildAgentCommand) {
		this.librlmRoot = librlmRoot;
		this.depth = depth;
		this.command = command;
	}

	/** Children use the config of the running (or most recent) ipython call. */
	setConfig(config: ChildConfig): void {
		this.config = config;
	}

	get running(): number {
		return this.children.size;
	}

	takeActivity(): ChildActivity {
		const activity = this.activity;
		this.activity = { spawned: 0, completed: 0, usage: emptyUsage() };
		return activity;
	}

	get environment(): Record<string, string> {
		if (!this.socketPath) throw new Error("RLM host bridge has not started");
		return { RLM_HOST_SOCKET: this.socketPath, RLM_HOST_TOKEN: this.authToken };
	}

	async ensureStarted(): Promise<void> {
		if (this.disposed) throw new Error("RLM host bridge is shutting down");
		if (this.server) return;
		if (!this.starting) {
			const start = this.start();
			const tracked = start.finally(() => {
				if (this.starting === tracked) this.starting = undefined;
			});
			this.starting = tracked;
		}
		await this.starting;
	}

	private async start(): Promise<void> {
		const directory = await mkdtemp(join(tmpdir(), "pi-rlm-host-"));
		const socketPath = join(directory, "host.sock");
		const server = createServer((socket) => this.handleConnection(socket));
		server.on("error", () => {});
		try {
			await chmod(directory, 0o700);
			await new Promise<void>((resolve, reject) => {
				const onError = (error: Error) => {
					server.off("listening", onListening);
					reject(error);
				};
				const onListening = () => {
					server.off("error", onError);
					resolve();
				};
				server.once("error", onError);
				server.once("listening", onListening);
				server.listen(socketPath);
			});
			await chmod(socketPath, 0o600);
			this.socketDirectory = directory;
			this.socketPath = socketPath;
			this.server = server;
		} catch (error) {
			await new Promise<void>((resolve) => server.close(() => resolve())).catch(() => {});
			await rm(directory, { recursive: true, force: true }).catch(() => {});
			throw error;
		}
	}

	private handleConnection(socket: Socket): void {
		let input = Buffer.alloc(0);
		let handled = false;
		const requester = new AbortController();
		const disconnect = () => requester.abort(new Error("Child requester disconnected"));
		this.sockets.add(socket);
		socket.once("end", disconnect);
		socket.once("close", () => {
			this.sockets.delete(socket);
			disconnect();
		});
		socket.on("error", () => {});
		socket.setTimeout(HOST_REQUEST_LINE_TIMEOUT_MS, () => socket.destroy());
		socket.on("data", (chunk: Buffer) => {
			if (handled) return;
			input = Buffer.concat([input, chunk]);
			if (input.byteLength > MAX_HOST_REQUEST_BYTES) {
				handled = true;
				socket.end(`${JSON.stringify(this.failure("", "Host request exceeded the size limit"))}\n`);
				return;
			}
			const newline = input.indexOf(0x0a);
			if (newline < 0) return;
			handled = true;
			socket.setTimeout(0);
			socket.pause();
			const line = input.subarray(0, newline).toString("utf8");
			void this.processLine(line, requester.signal)
				.then((response) => socket.end(`${JSON.stringify(response)}\n`))
				.catch((error) => socket.end(`${JSON.stringify(this.failure("", boundedError(error)))}\n`));
		});
	}

	private async processLine(line: string, requesterSignal: AbortSignal): Promise<HostResponse> {
		let raw: unknown;
		try {
			raw = JSON.parse(line);
		} catch {
			return this.failure("", "Invalid JSON host request");
		}
		if (!raw || typeof raw !== "object") return this.failure("", "Host request must be an object");
		const candidate = raw as Record<string, unknown>;
		const id = typeof candidate.id === "string" ? candidate.id : "";
		if (candidate.version !== HOST_PROTOCOL_VERSION) return this.failure(id, "Unsupported host protocol version");
		if (typeof candidate.auth !== "string" || !this.authMatches(candidate.auth)) {
			return this.failure(id, "Host authentication failed");
		}
		try {
			return await this.dispatch(candidate as unknown as HostRequest, requesterSignal);
		} catch (error) {
			return this.failure(id, boundedError(error));
		}
	}

	private authMatches(candidate: string): boolean {
		const left = Buffer.from(candidate);
		const right = Buffer.from(this.authToken);
		return left.byteLength === right.byteLength && timingSafeEqual(left, right);
	}

	private async dispatch(request: HostRequest, requesterSignal: AbortSignal): Promise<HostResponse> {
		if (typeof request.id !== "string" || typeof request.execution_id !== "string") {
			throw new Error("Host request requires string id and execution_id");
		}
		if (request.op !== "complete") throw new Error("Unknown host operation");
		if (this.depth >= MAX_RLM_DEPTH) throw new Error(`RLM child depth limit (${MAX_RLM_DEPTH}) reached`);
		if (typeof request.task !== "string" || (request.context !== null && typeof request.context !== "string")) {
			throw new Error("complete requires a string task and optional string context");
		}
		if (typeof request.cwd !== "string" || !request.cwd.startsWith("/")) {
			throw new Error("complete requires the kernel's absolute cwd");
		}
		let cwdIsDirectory = false;
		try {
			cwdIsDirectory = statSync(request.cwd).isDirectory();
		} catch {}
		if (!cwdIsDirectory) throw new Error("The kernel cwd is not an accessible directory");
		if (!request.task.trim()) throw new Error("child task must not be empty");
		const requestBytes = utf8Bytes(request.task) + (request.context === null ? 0 : utf8Bytes(request.context));
		if (requestBytes > MAX_CHILD_REQUEST_BYTES) {
			throw new Error(`Child task and context exceed ${formatSize(MAX_CHILD_REQUEST_BYTES)}`);
		}
		if (this.children.size >= MAX_LIVE_HANDLES) {
			throw new Error(`At most ${MAX_LIVE_HANDLES} child completions may be active`);
		}
		const config = this.config;
		if (!config?.model) throw new Error("No active model is available for child creation");

		const record: ChildRecord = {
			controller: new AbortController(),
			promise: undefined as unknown as Promise<ChildResult>,
			timedOut: false,
		};
		record.promise = this.runChild(record, config, request);
		this.children.add(record);
		this.activity.spawned += 1;
		this.onActivity?.(this.activity, this.children.size);
		const cancelOnDisconnect = () => record.controller.abort(new Error("Child requester disconnected"));
		requesterSignal.addEventListener("abort", cancelOnDisconnect, { once: true });
		if (requesterSignal.aborted) cancelOnDisconnect();
		try {
			const result = await record.promise;
			const response = this.success(request.id, result);
			if (utf8Bytes(JSON.stringify(response)) + 1 > MAX_HOST_RESPONSE_BYTES) {
				throw new Error(`Child response exceeds ${formatSize(MAX_HOST_RESPONSE_BYTES)}`);
			}
			return response;
		} finally {
			requesterSignal.removeEventListener("abort", cancelOnDisconnect);
			this.children.delete(record);
			this.activity.completed += 1;
			addUsage(this.activity.usage, (await record.promise).usage);
			this.onActivity?.(this.activity, this.children.size);
		}
	}

	private async runChild(
		record: ChildRecord,
		config: ChildConfig,
		request: HostRequest,
	): Promise<ChildResult> {
		const started = Date.now();
		let release: (() => void) | undefined;
		let usage = emptyUsage();
		const deadline = setTimeout(() => {
			record.timedOut = true;
			record.controller.abort(new Error("Child deadline exceeded"));
		}, CHILD_DEADLINE_MS);
		try {
			release = await this.limiter.acquire(record.controller.signal);
			if (record.controller.signal.aborted) throw new Error("Child cancelled before startup");
			if (!config.model) throw new Error("No active model is available for child creation");
			const response = await runChildAgent({
				model: config.model,
				thinkingLevel: config.thinkingLevel,
				signal: record.controller.signal,
				cwd: request.cwd,
				task: request.task,
				context: request.context,
				depth: this.depth,
				librlmRoot: this.librlmRoot,
			}, this.command);
			usage = response.usage;
			if (response.error) throw new Error(response.error);
			const bounded = truncateUtf8(response.text, MAX_CHILD_TEXT_BYTES);
			return {
				status: "ok",
				text: bounded.text,
				error: null,
				usage,
				elapsed_ms: Date.now() - started,
				truncated: bounded.truncated,
			};
		} catch (error) {
			const status = record.timedOut ? "timeout" : record.controller.signal.aborted ? "cancelled" : "error";
			return {
				status,
				text: null,
				error: boundedError(error),
				usage,
				elapsed_ms: Date.now() - started,
				truncated: false,
			};
		} finally {
			clearTimeout(deadline);
			release?.();
		}
	}

	private success(id: string, result: unknown): HostResponse {
		return { version: HOST_PROTOCOL_VERSION, id, ok: true, result };
	}

	private failure(id: string, error: string): HostResponse {
		return { version: HOST_PROTOCOL_VERSION, id, ok: false, error };
	}

	async shutdown(): Promise<void> {
		if (this.disposed) return;
		this.disposed = true;
		await this.starting?.catch(() => {});
		for (const socket of this.sockets) socket.destroy();
		this.sockets.clear();
		const server = this.server;
		this.server = undefined;
		if (server) await new Promise<void>((resolve) => server.close(() => resolve())).catch(() => {});
		const records = [...this.children];
		for (const record of records) record.controller.abort(new Error("RLM host bridge shutting down"));
		await Promise.allSettled(records.map((record) => record.promise));
		this.children.clear();
		if (this.socketDirectory) await rm(this.socketDirectory, { recursive: true, force: true }).catch(() => {});
		this.socketDirectory = undefined;
		this.socketPath = undefined;
	}
}
