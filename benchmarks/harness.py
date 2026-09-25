"""Shared process, event, usage, and provenance helpers for model-backed evals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extensions" / "rlm.ts"
IPYTHON_EXTENSION = (
    Path(os.environ.get("PI_IPYTHON_ROOT", ROOT.parent / "pi-ipython")) / "extensions" / "ipython.ts"
)
USAGE_FIELDS = ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
COST_FIELDS = ("input", "output", "cacheRead", "cacheWrite", "total")


@dataclass
class PiRun:
    stdout: bytes
    stderr: str
    elapsed_seconds: float
    returncode: int
    timed_out: bool


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def base_command(args: Any) -> list[str]:
    return [
        "pi",
        "--mode",
        "json",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-context-files",
        "--no-prompt-templates",
        "--no-themes",
        "--no-builtin-tools",
        "--no-approve",
        "--extension",
        str(IPYTHON_EXTENSION),
        "--extension",
        str(EXTENSION),
        "--model",
        args.model,
        "--thinking",
        args.thinking,
        "--tools",
        "ipython",
    ]


def decode_events(payload: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Pi emitted invalid JSON: {raw_line[:500]!r}") from error
        if isinstance(event, dict):
            events.append(event)
    return events


def empty_usage() -> dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {field: 0.0 for field in COST_FIELDS},
    }


def add_usage(total: dict[str, Any], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for field in USAGE_FIELDS:
        value = usage.get(field)
        if isinstance(value, (int, float)):
            total[field] += value
    cost = usage.get("cost")
    if isinstance(cost, dict):
        for field in COST_FIELDS:
            value = cost.get(field)
            if isinstance(value, (int, float)):
                total["cost"][field] += value


def usage_from_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    root = empty_usage()
    nested = empty_usage()
    for event in events:
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                add_usage(root, message.get("usage"))
        if event.get("type") == "tool_execution_end" and event.get("toolName") == "ipython":
            result = event.get("result")
            details = result.get("details") if isinstance(result, dict) else None
            if isinstance(details, dict):
                add_usage(nested, details.get("nestedUsage"))
    total = empty_usage()
    add_usage(total, root)
    add_usage(total, nested)
    return {"root": root, "nested": nested, "total": total}


def final_answer_from_events(events: Iterable[dict[str, Any]]) -> str:
    finals: list[Any] = []
    for event in events:
        if event.get("type") != "tool_execution_end" or event.get("toolName") != "ipython":
            continue
        if event.get("isError"):
            continue
        result = event.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        if isinstance(details, dict) and "final" in details:
            finals.append(details["final"])
    if not finals:
        raise RuntimeError("Expected a successful RLM final result, got 0")
    final = finals[-1]
    if not isinstance(final, dict) or "answer" not in final:
        raise RuntimeError('RLM final result must be an object containing "answer"')
    answer = final["answer"]
    if not isinstance(answer, (str, int, float)):
        raise RuntimeError("RLM final answer must be a string or number")
    return str(answer)


def run_pi(args: Any, prompt: str, cwd: Path) -> PiRun:
    env = dict(os.environ)
    if args.agent_dir:
        env["PI_CODING_AGENT_DIR"] = args.agent_dir
    started = time.monotonic()
    process = subprocess.Popen(
        [*base_command(args), "--print", prompt],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return PiRun(
        stdout=stdout,
        stderr=stderr.decode(errors="replace"),
        elapsed_seconds=time.monotonic() - started,
        returncode=process.returncode,
        timed_out=timed_out,
    )


def git_metadata() -> dict[str, Any]:
    def command(*argv: str) -> bytes:
        return subprocess.check_output(argv, cwd=ROOT, stderr=subprocess.DEVNULL)

    try:
        commit = command("git", "rev-parse", "HEAD").decode().strip()
        status = command("git", "status", "--porcelain=v1")
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def source_hashes(relative_paths: Iterable[str]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for relative_path in relative_paths:
        path = ROOT / relative_path
        hashes[relative_path] = sha256_bytes(path.read_bytes()) if path.is_file() else None
    return hashes


def dependency_versions(distributions: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def pi_version() -> str | None:
    try:
        return subprocess.check_output(["pi", "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_record(output: Any, record: dict[str, Any]) -> None:
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()
