from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
IPYTHON_ROOT = Path(os.environ.get("PI_IPYTHON_ROOT", ROOT.parent / "pi-ipython")).resolve()
BRIDGE = IPYTHON_ROOT / "extensions" / "bridge.py"
MODEL = os.environ.get("PI_RLM_TEST_MODEL")
PROVIDER_EXTENSION = os.environ.get("PI_RLM_TEST_PROVIDER_EXTENSION")
THINKING = os.environ.get("PI_RLM_TEST_THINKING", "low")
TIMEOUT = int(os.environ.get("PI_RLM_TEST_TIMEOUT", "300"))
INCLUDE_LARGE = os.environ.get("PI_RLM_TEST_INCLUDE_LARGE") == "1"
INTEGRATION_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    global INTEGRATION_DIRECTORY
    if not MODEL:
        return
    INTEGRATION_DIRECTORY = tempfile.TemporaryDirectory(prefix="pi-rlm-integration-")
    unittest.addModuleCleanup(INTEGRATION_DIRECTORY.cleanup)
    directory = Path(INTEGRATION_DIRECTORY.name)
    agent_directory = directory / "agent"
    agent_directory.mkdir()
    (directory / "workspace").mkdir()
    source = Path(os.environ.get(
        "PI_RLM_TEST_AGENT_DIR", os.environ.get("PI_CODING_AGENT_DIR", str(Path.home() / ".pi" / "agent"))
    )).expanduser().resolve()
    for name in ("auth.json", "models.json"):
        path = source / name
        if path.is_file():
            (agent_directory / name).symlink_to(path)
    extensions = [f"-builtin:{name}" for name in ("mcp", "llama.cpp", "codemode", "tool-search")]
    if PROVIDER_EXTENSION:
        extensions.append(str(Path(PROVIDER_EXTENSION).expanduser().resolve()))
    (agent_directory / "settings.json").write_text(json.dumps({
        "packages": [str(IPYTHON_ROOT), str(ROOT)],
        "extensions": extensions,
        "defaultTools": ["ipython"],
        "defaultProjectTrust": "never",
        "cacheWarming": "off",
    }))


def test_directory() -> Path:
    if INTEGRATION_DIRECTORY is None:
        raise RuntimeError("Integration settings are not initialized")
    return Path(INTEGRATION_DIRECTORY.name)


def test_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(test_directory() / "agent")
    env["PI_RLM_DEPTH"] = "0"
    env["PI_IPYTHON_PERSISTENCE"] = "0"
    return env


def base_command(mode: str) -> list[str]:
    if not MODEL:
        raise RuntimeError("Set PI_RLM_TEST_MODEL, for example openai-codex/gpt-6.1-sol")
    return [
        "pi",
        "--mode",
        mode,
        "--no-session",
        "-ns",
        "-nc",
        "-nbt",
        "--model",
        MODEL,
        "--thinking",
        THINKING,
        "--tools",
        "ipython",
    ]


def decode_jsonl(payload: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in payload.splitlines():
        if raw:
            events.append(json.loads(raw))
    return events


def run_print(prompt: str, timeout: int = TIMEOUT) -> list[dict[str, Any]]:
    result = subprocess.run(
        [*base_command("json"), "-p", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=test_env(),
        cwd=test_directory() / "workspace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0 or result.stderr:
        raise AssertionError(
            f"Pi failed with {result.returncode}:\n{result.stderr.decode(errors='replace')}"
        )
    return decode_jsonl(result.stdout)


def tool_end(events: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [event for event in events if event.get("type") == "tool_execution_end"]
    if len(matches) != 1:
        raise AssertionError(f"Expected one tool result, got {len(matches)}")
    return matches[0]


def host_directories() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("pi-rlm-host-*"))


def bridge_processes() -> set[str]:
    result = subprocess.run(
        ["ps", "-axww", "-o", "pid=,command="], stdout=subprocess.PIPE, text=True, check=True
    )
    return {line.strip() for line in result.stdout.splitlines() if str(BRIDGE) in line}


class RpcPi:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            base_command("rpc"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=test_env(),
            cwd=test_directory() / "workspace",
            bufsize=0,
        )
        self.buffer = b""
        self.command_index = 0

    def send(self, value: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(value).encode() + b"\n")
        self.process.stdin.flush()

    def read_event(self, deadline: float) -> dict[str, Any]:
        assert self.process.stdout is not None
        while time.monotonic() < deadline:
            if b"\n" in self.buffer:
                raw, self.buffer = self.buffer.split(b"\n", 1)
                if raw:
                    return json.loads(raw)
                continue
            ready, _, _ = select.select(
                [self.process.stdout], [], [], min(0.2, deadline - time.monotonic())
            )
            if not ready:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                break
            self.buffer += chunk
        stderr = b""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise AssertionError(
            f"Timed out waiting for Pi RPC event; exit={self.process.poll()} "
            f"stderr={stderr.decode(errors='replace')}"
        )

    def collect_until(
        self, predicate: Callable[[dict[str, Any], list[dict[str, Any]]], bool]
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + TIMEOUT
        events: list[dict[str, Any]] = []
        while True:
            event = self.read_event(deadline)
            events.append(event)
            if predicate(event, events):
                return events

    def prompt(self, message: str) -> list[dict[str, Any]]:
        self.command_index += 1
        command_id = f"prompt-{self.command_index}"
        self.send({"id": command_id, "type": "prompt", "message": message})
        events = self.collect_until(lambda event, _events: event.get("type") == "agent_settled")
        responses = [
            event
            for event in events
            if event.get("type") == "response" and event.get("id") == command_id
        ]
        if len(responses) != 1 or responses[0].get("success") is not True:
            raise AssertionError(f"Prompt was not accepted: {responses}")
        return events

    def abort_after_progress(self, prompt: str, marker: str) -> list[dict[str, Any]]:
        self.command_index += 1
        prompt_id = f"prompt-{self.command_index}"
        abort_id = f"abort-{self.command_index}"
        self.send({"id": prompt_id, "type": "prompt", "message": prompt})
        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + TIMEOUT
        while True:
            event = self.read_event(deadline)
            events.append(event)
            if event.get("type") == "tool_execution_update" and marker in json.dumps(event):
                break
        self.send({"id": abort_id, "type": "abort"})
        abort_ok = False
        settled = False
        while not (abort_ok and settled):
            event = self.read_event(deadline)
            events.append(event)
            if event.get("type") == "response" and event.get("id") == abort_id:
                abort_ok = event.get("success") is True
            if event.get("type") == "agent_settled":
                settled = True
        return events

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait()
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if code != 0 or stderr:
            raise AssertionError(
                f"Pi RPC shutdown failed with {code}: {stderr.decode(errors='replace')}"
            )


@unittest.skipUnless(MODEL, "Set PI_RLM_TEST_MODEL to run model-backed acceptance tests")
class Slice2AcceptanceTests(unittest.TestCase):
    def assert_cleanup(self, before: tuple[set[Path], set[str]]) -> None:
        time.sleep(1)
        directories, processes = before
        self.assertEqual(host_directories() - directories, set())
        self.assertEqual(bridge_processes() - processes, set())

    def test_parallel_final_and_usage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pi-rlm-child-kernel-") as directory:
            child_path = Path(directory) / "child.json"
            grandchild_path = Path(directory) / "grandchild.json"
            grandchild_code = f'''import json
import os
from pathlib import Path
h=await rlm.spawn("Reply with exactly DEPTH_LIMIT_BYPASSED.")
[blocked]=await rlm.gather([h])
Path({str(grandchild_path)!r}).write_text(json.dumps({{"kernel_pid":os.getpid(),"depth":int(os.environ["PI_RLM_DEPTH"]),"blocked":{{"status":blocked["status"],"error":blocked["error"]}}}}))
await rlm.final("GRANDCHILD")'''
            grandchild_task = (
                "Call ipython exactly once with this exact code. Then reply with only GRANDCHILD: "
                + grandchild_code
            )
            child_code = f'''import json
import os
from pathlib import Path
h=await rlm.spawn({grandchild_task!r})
child=(await rlm.gather([h]))[0]
Path({str(child_path)!r}).write_text(json.dumps({{"kernel_pid":os.getpid(),"depth":int(os.environ["PI_RLM_DEPTH"]),"grandchild_status":child["status"],"grandchild_text":child["text"]}}))
await rlm.final("ALPHA")'''
            child_task = (
                "Call ipython exactly once with this exact code. Then reply with only ALPHA: "
                + child_code
            )
            prompt = f'''Call ipython exactly once with this exact code and do nothing else: import asyncio
import os
print("\\n".join(f"line-{{i}}" for i in range(12)))
await asyncio.sleep(0.2)
hs=[await rlm.spawn({child_task!r}),await rlm.spawn("Reply with exactly BETA.")]
rs=await rlm.gather(hs)
await rlm.final({{"kernel_pid":os.getpid(),"statuses":[r["status"] for r in rs],"texts":[r["text"] for r in rs]}})'''
            events = run_print(prompt)
            end = tool_end(events)
            result = end["result"]
            self.assertFalse(end["isError"])
            self.assertNotIn("terminate", result)
            self.assertEqual(result["details"]["final"]["statuses"], ["ok", "ok"])
            self.assertEqual(
                [text.strip() for text in result["details"]["final"]["texts"]],
                ["ALPHA", "BETA"],
            )
            self.assertEqual(result["details"]["children"], {"spawned": 2, "completed": 2})
            self.assertGreater(result["details"]["nestedUsage"]["totalTokens"], 0)
            self.assertEqual(result["usage"], result["details"]["nestedUsage"])
            text = result["content"][0]["text"]
            self.assertIn("line-11", text)
            self.assertIn("[RLM final]", text)
            child = json.loads(child_path.read_text())
            grandchild = json.loads(grandchild_path.read_text())
            self.assertEqual(child["depth"], 1)
            self.assertEqual(child["grandchild_status"], "ok")
            self.assertEqual(child["grandchild_text"].strip(), "GRANDCHILD")
            self.assertEqual(grandchild["depth"], 2)
            self.assertEqual(grandchild["blocked"]["status"], "error")
            self.assertIn("depth limit", grandchild["blocked"]["error"].lower())
            self.assertEqual(len({
                result["details"]["final"]["kernel_pid"],
                child["kernel_pid"],
                grandchild["kernel_pid"],
            }), 3)

    def test_failed_cell_preserves_child_usage_once(self) -> None:
        rpc = RpcPi()
        try:
            events = rpc.prompt(
                '''Call ipython exactly once with this exact code. The error is intentional; do not retry or fix it: h=await rlm.spawn("Reply with exactly ALPHA.")
child=(await rlm.gather([h]))[0]
raise ValueError("intentional failure after gather")'''
            )
            end = tool_end(events)
            result = end["result"]
            self.assertTrue(end["isError"])
            self.assertIn("intentional failure after gather", result["content"][0]["text"])
            self.assertIn("usage", result)
            self.assertGreater(result["usage"]["totalTokens"], 0)
            self.assertEqual(result["details"]["status"], "error")
            self.assertEqual(result["details"]["nestedUsage"], result["usage"])
            messages = [
                event["message"]
                for event in events
                if event.get("type") == "message_end"
                and event["message"].get("role") == "toolResult"
            ]
            self.assertEqual(len(messages), 1)
            self.assertTrue(messages[0]["isError"])
            self.assertEqual(messages[0]["usage"], result["usage"])

            recovered = tool_end(rpc.prompt(
                '''Call ipython exactly once with this exact code and do nothing else: await rlm.final(child)'''
            ))["result"]
            self.assertEqual(recovered["details"]["final"]["status"], "ok")
            self.assertEqual(recovered["details"]["final"]["usage"], result["usage"])
            self.assertFalse(recovered.get("usage"))
            self.assertFalse(recovered["details"]["kernelReset"])
        finally:
            rpc.close()

    def test_atomic_concurrent_gather(self) -> None:
        prompt = """Call ipython exactly once with this exact code and do nothing else: import asyncio
h=await rlm.spawn("Reply with exactly ALPHA.")
async def one_gather():
    try:
        value=await rlm.gather([h])
        return {"kind":"ok","text":value[0]["text"],"usage":value[0]["usage"]}
    except Exception as e:
        return {"kind":"error","type":type(e).__name__}
a,b=await asyncio.gather(one_gather(),one_gather())
await rlm.final({"attempts":[a,b]})"""
        result = tool_end(run_print(prompt))["result"]
        attempts = result["details"]["final"]["attempts"]
        self.assertEqual(sorted(item["kind"] for item in attempts), ["error", "ok"])
        successful = next(item for item in attempts if item["kind"] == "ok")
        self.assertEqual(result["details"]["nestedUsage"], successful["usage"])
        self.assertEqual(result["details"]["children"], {"spawned": 1, "completed": 1})

    def test_failed_admission_preserves_sibling(self) -> None:
        prompt = """Call ipython exactly once with this exact code and do nothing else: good=await rlm.spawn("Reply with exactly SURVIVED.")
try:
    await rlm.spawn("oversized", context="x"*(1024*1024))
except Exception as e:
    failed={"type":type(e).__name__,"message":str(e)}
result=(await rlm.gather([good]))[0]
await rlm.final({"failed":failed,"sibling":{"status":result["status"],"text":result["text"]}})"""
        final = tool_end(run_print(prompt))["result"]["details"]["final"]
        self.assertEqual(final["failed"]["type"], "RLMHostError")
        self.assertEqual(final["sibling"]["status"], "ok")
        self.assertEqual(final["sibling"]["text"].strip(), "SURVIVED")

    def test_handle_release_cancels_child_and_cleans_up(self) -> None:
        before = host_directories(), bridge_processes()
        rpc = RpcPi()
        try:
            released = rpc.prompt(
                '''Call ipython exactly once with this exact code and do nothing else: import asyncio
h=await rlm.spawn("Write 20,000 numbered lines. Do not summarize or stop early.")
await asyncio.sleep(0.1)
await h.release()
await rlm.final({"released": True})'''
            )
            result = tool_end(released)["result"]
            self.assertEqual(result["details"]["final"], {"released": True})
            self.assertEqual(result["details"]["children"]["spawned"], 1)
            recovered = rpc.prompt(
                '''Call ipython exactly once with this exact code and do nothing else: await rlm.final({"recovered": True})'''
            )
            self.assertEqual(
                tool_end(recovered)["result"]["details"]["final"],
                {"recovered": True},
            )
        finally:
            rpc.close()
        self.assert_cleanup(before)

    def test_active_cancellation_recovers_and_cleans_up(self) -> None:
        before = host_directories(), bridge_processes()
        rpc = RpcPi()
        try:
            cancelled = rpc.abort_after_progress(
                """Call ipython exactly once with this exact code and do nothing else: h=await rlm.spawn("Write 20,000 numbered lines. Do not summarize or stop early.")
print("CHILD-" + "SPAWNED", flush=True)
await rlm.gather([h])""",
                "CHILD-SPAWNED",
            )
            errors = [
                event
                for event in cancelled
                if event.get("type") == "tool_execution_end" and event.get("isError")
            ]
            self.assertEqual(len(errors), 1)
            recovered = rpc.prompt(
                """Call ipython exactly once with this exact code and do nothing else: await rlm.final({"recovered": True})"""
            )
            result = tool_end(recovered)["result"]
            self.assertEqual(result["details"]["final"], {"recovered": True})
        finally:
            rpc.close()
        self.assert_cleanup(before)

    def test_startup_cancellation_recovers_and_cleans_up(self) -> None:
        before = host_directories(), bridge_processes()
        rpc = RpcPi()
        try:
            rpc.abort_after_progress(
                """Call ipython exactly once with this exact code and do nothing else: import asyncio
await asyncio.sleep(60)""",
                "Starting IPython kernel",
            )
            recovered = rpc.prompt(
                """Call ipython exactly once with this exact code and do nothing else: await rlm.final({"recovered": True})"""
            )
            self.assertEqual(tool_end(recovered)["result"]["details"]["final"], {"recovered": True})
        finally:
            rpc.close()
        self.assert_cleanup(before)

    def test_cross_cell_handle_survives_background_gather(self) -> None:
        rpc = RpcPi()
        try:
            rpc.prompt(
                '''Call ipython exactly once with this exact code and do nothing else: import asyncio
h=await rlm.spawn("Reply with exactly CROSSCELL.")
background=asyncio.create_task(rlm.gather([h]))
"spawned"'''
            )
            events = rpc.prompt(
                """Call ipython exactly once with this exact code and do nothing else: result=(await rlm.gather([h]))[0]
await rlm.final({"status":result["status"],"text":result["text"]})"""
            )
            final = tool_end(events)["result"]["details"]["final"]
            self.assertEqual(final["status"], "ok")
            self.assertEqual(final["text"].strip(), "CROSSCELL")
        finally:
            rpc.close()

    @unittest.skipUnless(
        INCLUDE_LARGE, "Set PI_RLM_TEST_INCLUDE_LARGE=1 for the costly large-context case"
    )
    def test_large_context_stays_out_of_root_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pi-rlm-test-") as directory:
            path = Path(directory) / "context.txt"
            with path.open("w") as output:
                for index in range(12_000):
                    label = "ALPHA" if index % 3 else "BETA"
                    output.write(f"{index:05d}|{label}|payload-{index * index:012d}\n")
            code = f"""from pathlib import Path
raw=Path({str(path)!r}).read_text()
lines=raw.splitlines()
chunks=["\\n".join(lines[:6000]),"\\n".join(lines[6000:])]
hs=[await rlm.spawn("Count the records in this context. Reply with only the integer.",context=chunk) for chunk in chunks]
rs=await rlm.gather(hs)
await rlm.final({{"bytes_loaded":len(raw.encode()),"records":len(lines),"statuses":[r["status"] for r in rs],"answers":[r["text"] for r in rs]}})"""
            events = run_print(
                f"Call ipython exactly once with this exact code and do nothing else: {code}"
            )
            final = tool_end(events)["result"]["details"]["final"]
            self.assertEqual(final["bytes_loaded"], 392_000)
            self.assertEqual(final["records"], 12_000)
            self.assertEqual(final["statuses"], ["ok", "ok"])
            self.assertEqual([answer.strip() for answer in final["answers"]], ["6000", "6000"])


if __name__ == "__main__":
    unittest.main()
