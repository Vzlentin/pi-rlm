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
	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify({ ...prompt, rlmApi: "old API" }));
	git("-C", origin, "init", "-q", "-b", "main");
	git("-C", origin, "add", ".");
	git("-C", origin, "-c", "user.email=t@example.test", "-c", "user.name=t", "commit", "-qm", "one");
	const managed = { root: join(root, "managed"), managed: true };
	git("clone", "-q", origin, managed.root);

	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify(prompt));
	git("-C", origin, "-c", "user.email=t@example.test", "-c", "user.name=t", "commit", "-qam", "two");
	const pin = git("-C", origin, "rev-parse", "HEAD").trim();
	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify({ ...prompt, rlmApi: "updated API" }));
	git("-C", origin, "-c", "user.email=t@example.test", "-c", "user.name=t", "commit", "-qam", "three");

	const warnings = [];
	const warn = (message) => warnings.push(message);
	await syncLibrlm(managed, warn, origin, pin);
	assert.deepEqual(warnings, []);
	assert.equal(git("-C", managed.root, "rev-parse", "HEAD").trim(), pin);
	assert.throws(() => git("-C", managed.root, "symbolic-ref", "--quiet", "HEAD"));
	assert.deepEqual(loadRlmPrompt(managed.root), { rlmApi: prompt.rlmApi, rlmGuidance: prompt.rlmGuidance });

	const fresh = { root: join(root, "fresh"), managed: true };
	await syncLibrlm(fresh, warn, origin, pin);
	assert.deepEqual(warnings, []);
	assert.equal(git("-C", fresh.root, "rev-parse", "HEAD").trim(), pin);
	assert.throws(() => git("-C", fresh.root, "symbolic-ref", "--quiet", "HEAD"));

	git("-C", managed.root, "remote", "remove", "origin");
	await syncLibrlm(managed, warn, origin, pin);
	assert.deepEqual(warnings, []);

	const unavailablePin = "0".repeat(40);
	for (const location of [fresh, { root: join(root, "unavailable"), managed: true }]) {
		warnings.length = 0;
		await assert.rejects(syncLibrlm(location, warn, origin, unavailablePin), (error) => {
			assert.ok(error.message.includes(location.root));
			assert.ok(error.message.includes(unavailablePin));
			return true;
		});
		assert.equal(warnings.length, 1);
		assert.ok(warnings[0].includes(location.root));
		assert.ok(warnings[0].includes(unavailablePin));
	}

	// An override is never touched by git.
	await syncLibrlm({ root: join(root, "missing"), managed: false }, () => assert.fail("no warning expected"));
	assert.throws(() => loadRlmPrompt(join(root, "missing")), /ipython_extension\.py is missing/);
	writeFileSync(join(origin, "rlm", "prompts", "ipython.json"), JSON.stringify({ schema: prompt.schema, api: "no rlm keys" }));
	assert.throws(() => loadRlmPrompt(origin), /needs schema librlm\.ipython-prompt\.v1 with rlmApi and rlmGuidance/);
} finally {
	rmSync(root, { recursive: true, force: true });
}
console.log("librlm: location, managed pin, offline reuse, unavailable pin, override and invalid prompt passed");
