import { execFile } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { promisify } from "node:util";
import { MAX_CHILDREN_RUNNING, MAX_CHILD_REQUEST_BYTES, MAX_CHILD_TEXT_BYTES, MAX_LIVE_HANDLES } from "./rlm-host.ts";

const run = promisify(execFile);
export const LIBRLM_REPOSITORY = "https://github.com/Vzlentin/librlm";
const LIBRLM_PIN = "fe4c01adcd5c223d49fcee8c420b2d1574aca6c6";
const CLONE_TIMEOUT_MS = 120_000;
const FETCH_TIMEOUT_MS = 15_000;

export interface LibrlmLocation {
	root: string;
	/** A clone owned by pi-rlm, kept at its pinned librlm commit. */
	managed: boolean;
}

export interface RlmPrompt {
	rlmApi: string;
	rlmGuidance: string;
}

// Never resolve relative to the extension: Git installs live in Pi's cache.
export function resolveLibrlm(env = process.env, home = homedir()): LibrlmLocation {
	const configured = env.RLM_LIBRLM_ROOT;
	if (configured !== undefined) {
		const path = configured.startsWith("~/") ? join(home, configured.slice(2)) : configured;
		if (!isAbsolute(path)) throw new Error("RLM_LIBRLM_ROOT must be an absolute checkout path");
		return { root: resolve(path), managed: false };
	}
	const data = env.XDG_DATA_HOME && isAbsolute(env.XDG_DATA_HOME) ? env.XDG_DATA_HOME : join(home, ".local", "share");
	return { root: join(data, "pi-rlm", "librlm"), managed: true };
}

/** Keep managed checkouts at the pin without network access when HEAD already matches. */
export async function syncLibrlm(
	location: LibrlmLocation,
	warn: (message: string) => void,
	repository = LIBRLM_REPOSITORY,
	pin = LIBRLM_PIN,
): Promise<void> {
	if (!location.managed) return;
	try {
		if (!existsSync(join(location.root, ".git"))) {
			await mkdir(dirname(location.root), { recursive: true });
			await run("git", ["clone", "--quiet", repository, location.root], { timeout: CLONE_TIMEOUT_MS });
		} else {
			const { stdout } = await run("git", ["-C", location.root, "rev-parse", "HEAD"], { timeout: FETCH_TIMEOUT_MS });
			if (stdout.trim() === pin) return;
			await run("git", ["-C", location.root, "fetch", "--quiet", "origin", pin], { timeout: FETCH_TIMEOUT_MS });
		}
		await run("git", ["-C", location.root, "checkout", "--quiet", "--detach", pin], { timeout: FETCH_TIMEOUT_MS });
	} catch (error) {
		const message = `pi-rlm could not synchronize librlm at ${location.root} to ${pin}: ${(error as Error).message}`;
		warn(message);
		throw new Error(message, { cause: error });
	}
}

export function loadRlmPrompt(root: string): RlmPrompt {
	if (!existsSync(join(root, "rlm", "ipython_extension.py"))) {
		throw new Error(`librlm with rlm/ipython_extension.py is missing at ${root}; set RLM_LIBRLM_ROOT to its checkout`);
	}
	const path = join(root, "rlm", "prompts", "ipython.json");
	let value: Record<string, unknown>;
	try {
		value = JSON.parse(readFileSync(path, "utf8"));
	} catch (error) {
		throw new Error(`Cannot load RLM instructions at ${path}`, { cause: error });
	}
	if (value.schema !== "librlm.ipython-prompt.v1" ||
		![value.rlmApi, value.rlmGuidance].every((part) => typeof part === "string" && part.length > 0)) {
		throw new Error(`Invalid RLM instructions at ${path}: pi-rlm needs schema librlm.ipython-prompt.v1 with rlmApi and rlmGuidance`);
	}
	return { rlmApi: value.rlmApi as string, rlmGuidance: value.rlmGuidance as string };
}

/** The `rlm` system prompt section. */
export function rlmSection(prompt: RlmPrompt): string {
	return [
		prompt.rlmApi,
		prompt.rlmGuidance,
		`Child calls are limited to ${MAX_CHILDREN_RUNNING} concurrent/${MAX_LIVE_HANDLES} live handles, ${MAX_CHILD_REQUEST_BYTES / 1024 ** 2} MiB input, ${MAX_CHILD_TEXT_BYTES / 1024} KiB returned text, and a 5-minute deadline. rlm.final prints its value at the end of the cell's output; it does not end the turn.`,
	].join("\n\n");
}
