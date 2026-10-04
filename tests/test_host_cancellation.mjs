import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { RlmHostBridge } from "../extensions/rlm-host.ts";
import { alive, command, events, waitFor } from "./fixtures/child-agent.mjs";

const directory = realpathSync(mkdtempSync(join(tmpdir(), "pi-rlm-host-test-")));
const bridge = new RlmHostBridge(directory, 0, command(directory));
const sockets = [];
const starts = () => events(directory).filter((event) => event.type === "start");

async function openRequest(task, overrides = {}) {
	const { RLM_HOST_SOCKET: socketPath, RLM_HOST_TOKEN: auth } = bridge.environment;
	const socket = net.createConnection(socketPath);
	sockets.push(socket);
	await once(socket, "connect");
	socket.write(`${JSON.stringify({
		version: 2, auth, id: task, execution_id: "execution", op: "complete",
		task, context: null, cwd: directory, ...overrides,
	})}\n`);
	return socket;
}
async function response(task, overrides = {}) {
	const socket = await openRequest(task, overrides);
	let text = "";
	for await (const chunk of socket) text += chunk.toString();
	return JSON.parse(text);
}

try {
	await bridge.ensureStarted();
	bridge.setConfig({ cwd: process.cwd(), model: { provider: "test", id: "fixture" }, thinkingLevel: "off" });
	const running = await Promise.all([0, 1, 2, 3].map((index) => openRequest(`block-ignore-${index}`)));
	await waitFor(() => starts().length === 4, "four admitted child requests");
	const queued = await openRequest("block-queued");
	await waitFor(() => bridge.running === 5, "the queued fifth child record");
	assert.equal(starts().length, 4);

	const disconnected = starts().find((event) => event.task === "block-ignore-0");
	running[0].destroy();
	await waitFor(() => events(directory).some((event) => event.type === "term"), "disconnect SIGTERM");
	assert.equal(starts().length, 4, "admission stays held until the child exits");
	assert.equal(alive(disconnected.pid), true);
	await waitFor(() => starts().length === 5, "disconnect releases admission after SIGKILL");
	assert.equal(alive(disconnected.pid), false);
	assert.ok(starts().every((event) => event.cwd === directory));

	const neverStarted = await openRequest("block-never-started");
	await waitFor(() => bridge.running === 5, "second queued request");
	neverStarted.destroy();
	await waitFor(() => bridge.running === 4, "queued disconnect");
	assert.equal(starts().length, 5);
	queued.destroy();
	for (const socket of running) socket.destroy();
	await waitFor(() => bridge.running === 0, "all disconnected children settle");
	assert.ok(starts().every((event) => !alive(event.pid)));
	const activity = bridge.takeActivity();
	assert.equal(activity.spawned, 6);
	assert.equal(activity.completed, 6);
	assert.equal(activity.usage.totalTokens, 10);

	const large = await response("large", { context: "explicit context" });
	assert.equal(large.ok, true);
	assert.equal(large.result.status, "ok");
	assert.equal(large.result.truncated, true);
	assert.ok(Buffer.byteLength(large.result.text) <= 256 * 1024);
	const failed = await response("error");
	assert.equal(failed.result.status, "error");
	assert.equal(failed.result.usage.totalTokens, 7);
	assert.equal((await response("bad-auth", { auth: "wrong" })).ok, false);
	assert.equal((await response("bad-cwd", { cwd: "/missing-pi-rlm-directory" })).ok, false);
	assert.equal((await response("oversized", { context: "x".repeat(1024 * 1024) })).ok, false);

	await openRequest("block-ignore-shutdown");
	await waitFor(() => starts().some((event) => event.task === "block-ignore-shutdown"), "shutdown child");
	await bridge.shutdown();
	assert.equal(bridge.running, 0);
	assert.ok(starts().every((event) => !alive(event.pid)));
} finally {
	for (const socket of sockets) socket.destroy();
	await bridge.shutdown();
	rmSync(directory, { recursive: true, force: true });
}
