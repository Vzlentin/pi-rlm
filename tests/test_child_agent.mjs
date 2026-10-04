import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { runChildAgent } from "../extensions/child-agent.ts";
import { alive, command, events, processInfo, reapZombies, running, waitFor } from "./fixtures/child-agent.mjs";

const directory = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-child-test-")));
const fixture = command(directory);
const request = {
	model: { provider: "test-provider", id: "fixture" }, thinkingLevel: "high",
	cwd: directory, task: "--assigned task", context: "explicit café context", signal: new AbortController().signal,
	depth: 1, librlmRoot: "/resolved/librlm",
};
const inherited = { RLM_HOST_SOCKET: "parent-socket", RLM_HOST_TOKEN: "parent-token", RLM_HOST_EXTRA: "parent-extra" };
const saved = Object.fromEntries(Object.keys(inherited).map((key) => [key, process.env[key]]));

try {
	Object.assign(process.env, inherited);
	const result = await runChildAgent(request, fixture);
	assert.equal(result.text, "done:--assigned task");
	assert.equal(result.error, undefined);
	assert.equal(result.usage.totalTokens, 7);
	const started = events(directory)[0];
	assert.equal(started.cwd, directory);
	assert.equal(started.prompt, `${request.task}\n\n<context>\n${request.context}\n</context>`);
	assert.equal(started.depth, "2");
	assert.equal(started.root, request.librlmRoot);
	assert.deepEqual(started.hostEnv, []);
	assert.deepEqual(started.args.slice(0, 8), [
		"--mode", "json", "--no-session", "--model", "test-provider/fixture", "--thinking", "high", "--append-system-prompt",
	]);
	assert.match(started.args[8], /assigned task/);
	assert.deepEqual(started.args.slice(9), ["--"]);
	for (const [key, value] of Object.entries(inherited)) assert.equal(process.env[key], value);

	for (const task of ["multi", "split"]) {
		const answer = await runChildAgent({ ...request, task, context: null }, fixture);
		assert.equal(answer.error, undefined);
		assert.equal(answer.text, "café😀\u2028line\u2029end");
		assert.equal(answer.usage.totalTokens, 12);
		assert.equal(answer.usage.reasoning, 3);
		assert.ok(Math.abs(answer.usage.cost.total - 0.12) < 1e-12);
		assert.equal(events(directory).find((event) => event.task === task).prompt, task);
	}
	const compacted = await runChildAgent({ ...request, task: "compaction" }, fixture);
	assert.equal(compacted.error, undefined);
	assert.equal(compacted.usage.totalTokens, 23);
	for (const task of ["@not-a-file", "x".repeat(1024 * 1024 - 1)]) {
		const answer = await runChildAgent({ ...request, task, context: null }, fixture);
		assert.equal(answer.error, undefined);
		assert.equal(answer.text, `done:${task}`);
	}
	for (const task of ["orphan", "orphan-pipes"]) {
		const answer = await runChildAgent({ ...request, task }, fixture);
		assert.equal(answer.error, undefined);
		const child = events(directory).find((event) => event.type === "start" && event.task === task);
		const descendant = events(directory).find((event) => event.type === "descendant" && event.parent === child.pid);
		assert.equal(running(descendant.pid), false);
	}
	if (process.platform === "linux") {
		let timeout;
		const completion = runChildAgent({ ...request, task: "orphan-zombie", depth: 0 }, fixture);
		try {
			const reaper = await waitFor(() => events(directory).find((event) => event.type === "zombie-reaper"), "zombie fixture");
			await waitFor(() => processInfo(reaper.zombie)?.state === "Z" && !alive(reaper.parent), "direct child exit with a zombie descendant");
			assert.equal(processInfo(reaper.zombie).pgid, reaper.parent);
			assert.equal(alive(reaper.zombie), true);
			assert.equal(running(reaper.zombie), false);
			const answer = await Promise.race([
				completion,
				new Promise((_resolve, reject) => { timeout = setTimeout(() => reject(new Error("Zombie group cleanup did not settle")), 15_000); }),
			]);
			assert.equal(answer.error, undefined);
			assert.equal(answer.text, "done:orphan-zombie");
			assert.equal(processInfo(reaper.zombie)?.state, "Z", "completion does not wait for the external reaper");
		} finally {
			clearTimeout(timeout);
			await reapZombies(directory);
			await completion;
		}
	}
	for (const task of ["error", "length", "toolUse", "aborted", "pending", "deferred", "nonzero", "unterminated", "missing", "invalid", "bad-content"]) {
		const failed = await runChildAgent({ ...request, task }, fixture);
		assert.ok(failed.error, task);
		assert.equal(failed.text, "");
		assert.equal(failed.usage.totalTokens, task === "missing" ? 3 : task === "invalid" ? 5 : task === "unterminated" ? 0 : 7);
		if (task === "nonzero") {
			assert.match(failed.error, /diagnostic-tail$/);
			assert.ok(failed.error.length < 17_000);
		}
	}
	const promptFile = join(directory, "pi-input.txt");
	const savedAgentDirectory = process.env.PI_CODING_AGENT_DIR;
	const savedPromptFile = process.env.PI_RLM_TEST_PROMPT_FILE;
	try {
		process.env.PI_CODING_AGENT_DIR = directory;
		process.env.PI_RLM_TEST_PROMPT_FILE = promptFile;
		const realPi = {
			command: process.execPath,
			args: [fileURLToPath(new URL("../node_modules/@earendil-works/pi-coding-agent/dist/cli.js", import.meta.url)),
				"--offline", "-ne", "-e", fixture.args[0], "-ns", "-nc", "-nt"],
		};
		for (const [task, context] of [["@not-a-file", "end-of-data"], ["x".repeat(1024 * 1024 - 1), null]]) {
			const handled = await runChildAgent({ ...request, model: { provider: "openai", id: "gpt-4o-mini" }, task, context }, realPi);
			assert.match(handled.error, /no final assistant message/);
			assert.equal(readFileSync(promptFile, "utf8"), context === null ? task : `${task}\n\n<context>\n${context}\n</context>`);
		}
	} finally {
		if (savedAgentDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = savedAgentDirectory;
		if (savedPromptFile === undefined) delete process.env.PI_RLM_TEST_PROMPT_FILE; else process.env.PI_RLM_TEST_PROMPT_FILE = savedPromptFile;
	}
	const missingCommand = await runChildAgent(request, { command: join(directory, "missing"), args: [] });
	assert.match(missingCommand.error, /ENOENT/);
	assert.equal(missingCommand.usage.totalTokens, 0);
	const cancelled = new AbortController();
	cancelled.abort();
	const count = events(directory).length;
	assert.ok((await runChildAgent({ ...request, signal: cancelled.signal }, fixture)).error);
	assert.equal(events(directory).length, count);

	for (const task of ["block-graceful", "block-ignore", "block-group"]) {
		const controller = new AbortController();
		const completion = runChildAgent({ ...request, task, signal: controller.signal }, fixture);
		const child = await waitFor(() => events(directory).find((event) => event.type === "start" && event.task === task), task);
		const descendant = task === "block-group" ? await waitFor(() => events(directory).find((event) => event.type === "descendant" && event.parent === child.pid), "descendant") : undefined;
		controller.abort();
		const failed = await completion;
		assert.ok(failed.error);
		assert.equal(failed.usage.totalTokens, 2);
		assert.equal(alive(child.pid), false);
		assert.ok(events(directory).some((event) => event.type === "term" && event.task === task));
		if (task === "block-graceful") assert.ok(events(directory).some((event) => event.type === "terminated" && event.task === task));
		if (descendant) assert.equal(running(descendant.pid), false);
	}
} finally {
	for (const [key, value] of Object.entries(saved)) {
		if (value === undefined) delete process.env[key]; else process.env[key] = value;
	}
	rmSync(directory, { recursive: true, force: true });
}
