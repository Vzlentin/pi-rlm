import { spawn } from "node:child_process";
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const command = (directory) => ({
	command: process.execPath,
	args: [fileURLToPath(import.meta.url), directory],
});
export const events = (directory) => {
	const path = join(directory, "events.jsonl");
	return existsSync(path) ? readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse) : [];
};
export const alive = (pid) => {
	try { process.kill(pid, 0); return true; } catch { return false; }
};
export const processInfo = (pid) => {
	try {
		const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
		const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
		return { state: fields[0], pgid: Number(fields[2]) };
	} catch (error) {
		if (error.code === "ENOENT") return undefined;
		throw error;
	}
};
export const running = (pid) => {
	if (process.platform !== "linux") return alive(pid);
	const info = processInfo(pid);
	return info !== undefined && info.state !== "Z" && info.state !== "X";
};
export async function reapZombies(directory) {
	for (const event of events(directory).filter((event) => event.type === "zombie-reaper")) {
		writeFileSync(join(directory, `reap-${event.zombie}`), "");
		await waitFor(() => !processInfo(event.zombie) && !running(event.pid), "fixture zombie reaping");
	}
}
export async function waitFor(predicate, label) {
	const deadline = Date.now() + 20_000;
	while (Date.now() < deadline) {
		const result = predicate();
		if (result) return result;
		await new Promise((resolve) => setTimeout(resolve, 10));
	}
	throw new Error(`Timed out waiting for ${label}`);
}

export default function captureInput(pi) {
	pi.on("input", (event) => {
		writeFileSync(process.env.PI_RLM_TEST_PROMPT_FILE, event.text);
		return { action: "handled" };
	});
}

const usage = (tokens) => ({
	input: tokens, output: 0, reasoning: 1, cacheRead: 0, cacheWrite: 0, totalTokens: tokens,
	cost: { input: tokens / 100, output: 0, cacheRead: 0, cacheWrite: 0, total: tokens / 100 },
});
const message = (role, tokens, stopReason = "stop", text = "") => ({
	type: "message_end",
	message: {
		role, content: [{ type: "thinking", thinking: "not the answer" }, { type: "text", text }],
		usage: usage(tokens), stopReason,
		...(stopReason === "error" ? { errorMessage: "fixture failed" } : {}),
	},
});

async function main() {
	const directory = process.argv[2];
	const log = (value) => appendFileSync(join(directory, "events.jsonl"), `${JSON.stringify({ pid: process.pid, ...value })}\n`);
	if (process.argv[3] === "--descendant") {
		process.on("SIGTERM", () => log({ type: "descendant-term" }));
		log({ type: "descendant", parent: process.ppid });
		setInterval(() => {}, 1_000);
		return;
	}
	const args = process.argv.slice(3);
	let task = "";
	process.on("SIGTERM", () => {
		log({ type: "term", task });
		if (!task.includes("ignore") && task !== "block-group") {
			setTimeout(() => { log({ type: "terminated", task }); process.exit(0); }, 50);
		}
	});
	process.stdin.setEncoding("utf8");
	let prompt = "";
	for await (const chunk of process.stdin) prompt += chunk;
	task = prompt.split("\n\n<context>\n")[0];
	log({
		type: "start", task, args, cwd: process.cwd(), prompt,
		depth: process.env.PI_RLM_DEPTH, root: process.env.RLM_LIBRLM_ROOT,
		hostEnv: Object.keys(process.env).filter((key) => key.startsWith("RLM_HOST_")),
	});
	const emit = (event) => process.stdout.write(`${JSON.stringify(event)}\n`);
	if (task.startsWith("block")) {
		emit(message("assistant", 2, "toolUse", "working"));
		if (task === "block-group") spawn(process.execPath, [fileURLToPath(import.meta.url), directory, "--descendant"], { stdio: "inherit" });
		setInterval(() => {}, 1_000);
		return;
	}
	if (["multi", "split", "invalid", "compaction"].includes(task)) {
		emit(message("assistant", 2, "toolUse", "earlier"));
		emit(message("toolResult", 3));
		emit({ type: "message_update", usage: usage(999) });
	}
	if (task === "compaction") emit({ type: "compaction_end", result: { usage: usage(11) }, aborted: false });
	if (task === "orphan" || task === "orphan-pipes") {
		spawn(process.execPath, [fileURLToPath(import.meta.url), directory, "--descendant"], { stdio: task === "orphan" ? "ignore" : "inherit" });
		await waitFor(() => events(directory).some((event) => event.type === "descendant" && event.parent === process.pid), "orphan startup");
	}
	if (task === "orphan-zombie") {
		// A parent outside the Pi group keeps the zombie until the test releases it.
		const reaper = spawn("python3", ["-c", `import json, os, sys, time
from pathlib import Path
os.setpgid(0, 0)
directory = Path(sys.argv[1])
group = int(sys.argv[2])
pid = os.fork()
if pid == 0:
    os.setpgid(0, group)
    os._exit(0)
with (directory / "events.jsonl").open("a") as output:
    output.write(json.dumps({"type": "zombie-reaper", "pid": os.getpid(), "zombie": pid, "parent": group}) + "\\n")
while not (directory / f"reap-{pid}").exists():
    time.sleep(0.01)
os.waitpid(pid, 0)
`, directory, String(process.pid)], { stdio: "ignore" });
		reaper.unref();
		await waitFor(() => {
			const event = events(directory).find((event) => event.type === "zombie-reaper" && event.parent === process.pid);
			const info = event && processInfo(event.zombie);
			return info?.state === "Z" && info.pgid === process.pid;
		}, "zombie in the child group");
	}
	if (task === "missing") { emit(message("toolResult", 3)); return; }
	if (task === "invalid") { process.stdout.write("not JSON\n"); return; }
	const text = ["multi", "split"].includes(task) ? "café😀\u2028line\u2029end" : task === "large" ? "x".repeat(300_000) : `done:${task}`;
	const stopReason = ["error", "length", "toolUse", "aborted", "pending", "deferred"].includes(task) ? task : "stop";
	const final = message("assistant", 7, stopReason, text);
	if (task === "bad-content") final.message.content = null;
	const payload = Buffer.from(JSON.stringify(final) + (task === "unterminated" ? "" : "\n"));
	if (task === "split") {
		const split = payload.indexOf(Buffer.from("😀")) + 1;
		process.stdout.write(payload.subarray(0, split));
		await new Promise((resolve) => setTimeout(resolve, 20));
		process.stdout.write(payload.subarray(split));
	} else process.stdout.write(payload);
	if (["orphan", "orphan-pipes", "orphan-zombie"].includes(task)) process.stdout.write("", () => process.exit(0));
	if (task === "nonzero") {
		process.stderr.write("x".repeat(30_000) + "diagnostic-tail");
		process.exitCode = 3;
	}
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
