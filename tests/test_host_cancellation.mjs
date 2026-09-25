import assert from "node:assert/strict";
import { once } from "node:events";
import net from "node:net";

import { RlmHostBridge } from "../extensions/rlm-host.ts";

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitFor(predicate, label) {
	const deadline = Date.now() + 5_000;
	while (!predicate()) {
		if (Date.now() >= deadline) throw new Error(`Timed out waiting for ${label}`);
		await delay(10);
	}
}

let starts = 0;
const signals = [];
const contexts = [];

async function completeChild(request) {
	starts += 1;
	signals.push(request.signal);
	contexts.push(request.context);
	return new Promise((_resolve, reject) => {
		const abort = () => reject(request.signal.reason ?? new Error("aborted"));
		request.signal.addEventListener("abort", abort, { once: true });
		if (request.signal.aborted) abort();
	});
}

const bridge = new RlmHostBridge(completeChild);
const sockets = [];

async function openRequest(index) {
	const { RLM_HOST_SOCKET: socketPath, RLM_HOST_TOKEN: auth } = bridge.environment;
	const socket = net.createConnection(socketPath);
	sockets.push(socket);
	await once(socket, "connect");
	socket.write(
		`${JSON.stringify({
			version: 2,
			auth,
			id: `request-${index}`,
			execution_id: "execution",
			op: "complete",
			task: `task-${index}`,
			context: null,
			cwd: process.cwd(),
		})}\n`,
	);
	return socket;
}

try {
	await bridge.ensureStarted();
	bridge.setConfig({ cwd: process.cwd(), model: {}, thinkingLevel: "off" });

	const running = await Promise.all([0, 1, 2, 3].map(openRequest));
	await waitFor(() => starts === 4, "four admitted child requests");

	await openRequest(4);
	await waitFor(() => bridge.children.size === 5, "the queued fifth child record");
	assert.equal(starts, 4);

	running[0].destroy();
	await waitFor(() => starts === 5, "request disconnect to release one child slot");
	assert.equal(signals[0].aborted, true);
} finally {
	for (const socket of sockets) socket.destroy();
	await waitFor(() => bridge.running === 0, "every disconnected child to settle");
	await bridge.shutdown();
}

assert.equal(bridge.children.size, 0);
assert.equal(signals.length, 5);
assert.ok(signals.every((signal) => signal.aborted));
assert.ok(contexts.every((context) => context.tools?.length === 0));
assert.ok(contexts.every((context) => context.messages.length === 1));
assert.deepEqual(
	(({ spawned, completed }) => ({ spawned, completed }))(bridge.takeActivity()),
	{ spawned: 5, completed: 5 },
);
