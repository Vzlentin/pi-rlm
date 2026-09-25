import { execFile } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);
export const LIBRLM_REPOSITORY = "https://github.com/Vzlentin/librlm";
const CLONE_TIMEOUT_MS = 120_000;
const PULL_TIMEOUT_MS = 15_000;

export interface LibrlmLocation {
	root: string;
	/** A clone owned by pi-rlm, updated from librlm main. */
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

/** Clone the managed checkout on first use, otherwise fast-forward it; a failed pull keeps the clone. */
export async function syncLibrlm(location: LibrlmLocation, warn: (message: string) => void): Promise<void> {
	if (!location.managed) return;
	if (!existsSync(join(location.root, ".git"))) {
		await mkdir(dirname(location.root), { recursive: true });
		try {
			await run("git", ["clone", "--quiet", LIBRLM_REPOSITORY, location.root], { timeout: CLONE_TIMEOUT_MS });
		} catch (error) {
			throw new Error(`Cannot clone librlm into ${location.root}: ${(error as Error).message}`);
		}
		return;
	}
	try {
		await run("git", ["-C", location.root, "pull", "--ff-only", "--quiet"], { timeout: PULL_TIMEOUT_MS });
	} catch (error) {
		warn(`pi-rlm could not update librlm at ${location.root}; using the existing clone. ${(error as Error).message}`);
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
