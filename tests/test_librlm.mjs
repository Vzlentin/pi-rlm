import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadRlmPrompt, resolveLibrlm, syncLibrlm } from "../extensions/librlm.ts";

const root = mkdtempSync(join(tmpdir(), "pi-rlm-librlm-"));
const git = (...args) => execFileSync("git", args, { stdio: "pipe", encoding: "utf8" });
try {
	const home = join(root, "home");
	assert.deepEqual(resolveLibrlm({}, home), { root: join(home, ".local", "share", "pi-rlm", "librlm"), managed: true });
	assert.deepEqual(resolveLibrlm({ XDG_DATA_HOME: "/data" }, home), { root: "/data/pi-rlm/librlm", managed: true });
	assert.deepEqual(resolveLibrlm({ RLM_LIBRLM_ROOT: "~/src/librlm" }, home), { root: join(home, "src", "librlm"), managed: false });
	for (const override of ["", "relative/librlm"]) {
		assert.throws(() => resolveLibrlm({ RLM_LIBRLM_ROOT: override }, home), /must be an absolute checkout path/);
	}

	const prompt = { schema: "librlm.ipython-prompt.v1", rlmApi: "fixture API", rlmGuidance: "fixture guidance" };
	const origin = join(root, "origin");
	mkdirSync(join(origin, "rlm", "prompts"), { recursive: true });
	writeFileSync(join(origin, "rlm", "ipython_extension.py"), "");
	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify(prompt));
	git("-C", origin, "init", "-q", "-b", "main");
	git("-C", origin, "-c", "user.email=t@example.test", "-c", "user.name=t", "commit", "-qam", "one", "--allow-empty");
	git("-C", origin, "add", ".");
	git("-C", origin, "-c", "user.email=t@example.test", "-c", "user.name=t", "commit", "-qm", "two");

	// A managed clone fast-forwards from main; a failed pull warns and keeps the clone.
	const managed = { root: join(root, "managed"), managed: true };
	git("clone", "-q", origin, managed.root);
	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify({ ...prompt, rlmApi: "updated API" }));
	git("-C", origin, "-c", "user.email=t@example.test", "-c", "user.name=t", "commit", "-qam", "three");
	const warnings = [];
	await syncLibrlm(managed, (message) => warnings.push(message));
	assert.deepEqual(warnings, []);
	assert.equal(loadRlmPrompt(managed.root).rlmApi, "updated API");
	git("-C", managed.root, "remote", "set-url", "origin", join(root, "missing"));
	await syncLibrlm(managed, (message) => warnings.push(message));
	assert.equal(warnings.length, 1);
	assert.match(warnings[0], /using the existing clone/);
	assert.deepEqual(loadRlmPrompt(managed.root), { rlmApi: "updated API", rlmGuidance: "fixture guidance" });

	// An override is never touched by git.
	await syncLibrlm({ root: join(root, "missing"), managed: false }, () => assert.fail("no warning expected"));
	assert.throws(() => loadRlmPrompt(join(root, "missing")), /ipython_extension\.py is missing/);
	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify({ schema: prompt.schema, api: "no rlm keys" }));
	assert.throws(() => loadRlmPrompt(origin), /needs schema librlm\.ipython-prompt\.v1 with rlmApi and rlmGuidance/);
} finally {
	rmSync(root, { recursive: true, force: true });
}
console.log("librlm: location, managed pull, failed pull, override and invalid prompt passed");
