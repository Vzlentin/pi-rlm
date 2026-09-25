#!/usr/bin/env python3
"""Run the Pi IPython RLM against the paper's LongBench-v2 CodeQA split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any, Iterable

from benchmarks.harness import (
    ROOT,
    add_usage,
    decode_events,
    dependency_versions,
    empty_usage,
    final_answer_from_events,
    git_metadata,
    pi_version,
    run_pi,
    sha256_bytes,
    source_hashes,
    usage_from_events,
    write_record,
)

DATASET_ID = "THUDM/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_SPLIT = "train"
DOMAIN = "Code Repository Understanding"
SUB_DOMAIN = "Code repo QA"
RLM_PAPER_URL = "https://arxiv.org/abs/2512.24601"
LONG_BENCH_URL = "https://github.com/THUDM/LongBench"
HASHED_INPUTS = (
    "benchmarks/harness.py",
    "benchmarks/longbench.py",
    "benchmarks/requirements.txt",
    "extensions/rlm.ts",
    "extensions/rlm-host.ts",
    "extensions/librlm.ts",
    "package-lock.json",
)


def load_cases(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Stream the 50 CodeQA cases used by the RLM paper."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The benchmark needs its pinned direct dependencies. Run it through `npm run eval:longbench`."
        ) from error

    stream = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
        streaming=True,
    )
    if args.shuffle_buffer:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    selected = 0
    for case in stream:
        if case.get("domain") != DOMAIN or case.get("sub_domain") != SUB_DOMAIN:
            continue
        yield dict(case)
        selected += 1
        if selected >= args.count:
            return
    raise RuntimeError(f"Requested {args.count} CodeQA cases but found {selected}.")


def context_bytes(case: dict[str, Any]) -> bytes:
    context = case.get("context", "")
    if not isinstance(context, str):
        raise RuntimeError(f"LongBench case {case.get('_id')} has a non-string context")
    return context.encode("utf-8")


def write_context(path: Path, case: dict[str, Any]) -> bytes:
    payload = context_bytes(case)
    path.write_bytes(payload)
    path.chmod(0o600)
    return payload


def benchmark_prompt(case: dict[str, Any]) -> str:
    return f'''Answer one LongBench-v2 CodeQA question with the ipython tool and its RLM children.

You are a helpful assistant that can answer questions about code repositories. You must answer the given question: {case["question"]} based on the stored context. Answer with exactly one number choice using only the choices provided:

0: {case["choice_A"]}
1: {case["choice_B"]}
2: {case["choice_C"]}
3: {case["choice_D"]}

The choices are indexed from 0 to 3.

The code repository is stored in `context.txt` in the working directory. Load it inside IPython. Keep the repository inside IPython and explicit `rlm.spawn(..., context=...)` calls rather than printing it into the parent conversation. Use Python to inspect and divide the repository, and use focused children where language understanding is needed.

When you have the answer, report it with:
await rlm.final({{"answer": "<0, 1, 2, or 3>"}})

The `answer` value must contain one number only, with no explanation.'''


def parse_choice(answer: str) -> str | None:
    """Accept a bare choice index or a terse answer wrapper, but not prose."""
    match = re.fullmatch(
        r"\s*(?:(?:answer|choice|option)\s*[:=]?\s*)?[\[(]?([0-3])[\])]?[.]?\s*",
        answer,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def multiple_choice_score(case: dict[str, Any], answer: str) -> float:
    gold_letter = str(case.get("answer", "")).strip().upper()
    gold_index = {letter: str(index) for index, letter in enumerate("ABCD")}.get(gold_letter)
    return 1.0 if parse_choice(answer) == gold_index else 0.0


def run_benchmark(args: argparse.Namespace) -> int:
    cases = load_cases(args)
    if args.dry_run:
        for index, case in enumerate(cases, 1):
            print(
                f"{index:02d} id={case.get('_id')} difficulty={case.get('difficulty')} "
                f"length={case.get('length')} answer={case.get('answer')}"
            )
        return 0

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_path = output_path.with_suffix("").with_name(output_path.stem + ".artifacts")
    scores: list[float] = []
    failures = 0
    usage_total = empty_usage()

    with output_path.open("x", encoding="utf-8") as output:
        artifacts_path.mkdir()
        write_record(
            output,
            {
                "type": "run",
                "schema_version": 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "benchmark": {
                    "name": "LongBench-v2 CodeQA",
                    "dataset": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "split": DATASET_SPLIT,
                    "domain": DOMAIN,
                    "sub_domain": SUB_DOMAIN,
                    "rlm_paper": RLM_PAPER_URL,
                    "benchmark_source": LONG_BENCH_URL,
                },
                "config": {
                    "model": args.model,
                    "thinking": args.thinking,
                    "count": args.count,
                    "seed": args.seed,
                    "shuffle_buffer": args.shuffle_buffer,
                    "timeout_seconds": args.timeout,
                },
                "repository": git_metadata(),
                "source_sha256": source_hashes(HASHED_INPUTS),
                "python_version": platform.python_version(),
                "dependency_versions": dependency_versions(
                    ("datasets", "huggingface-hub", "pyarrow")
                ),
                "pi_version": pi_version(),
            },
        )

        with tempfile.TemporaryDirectory(prefix="pi-rlm-longbench-") as temp_directory:
            for index, case in enumerate(cases, 1):
                case_directory = Path(temp_directory) / f"case-{index:03d}"
                case_directory.mkdir()
                context = write_context(case_directory / "context.txt", case)
                prompt = benchmark_prompt(case)
                prompt_bytes = prompt.encode("utf-8")
                print(f"[{index}/{args.count}] CodeQA id={case.get('_id')}...", flush=True)

                answer: str | None = None
                error: str | None = None
                stderr = ""
                elapsed = 0.0
                returncode: int | None = None
                timed_out = False
                usage = {"root": empty_usage(), "nested": empty_usage(), "total": empty_usage()}
                raw_relative = f"case-{index:03d}.jsonl"
                stderr_relative = f"case-{index:03d}.stderr.txt"
                try:
                    run = run_pi(args, prompt, case_directory)
                    stderr = run.stderr
                    elapsed = run.elapsed_seconds
                    returncode = run.returncode
                    timed_out = run.timed_out
                    (artifacts_path / raw_relative).write_bytes(run.stdout)
                    (artifacts_path / stderr_relative).write_text(stderr, encoding="utf-8")
                    events = decode_events(run.stdout)
                    usage = usage_from_events(events)
                    if timed_out:
                        raise RuntimeError(f"Pi exceeded the {args.timeout}s case timeout")
                    if returncode != 0:
                        raise RuntimeError(f"Pi exited with {returncode}: {stderr.strip()[-4000:]}")
                    answer = final_answer_from_events(events)
                except Exception as exception:
                    error = str(exception)
                    failures += 1

                score = multiple_choice_score(case, answer or "") if error is None else 0.0
                scores.append(score)
                add_usage(usage_total, usage["total"])
                write_record(
                    output,
                    {
                        "type": "case",
                        "index": index,
                        "id": case.get("_id"),
                        "domain": case.get("domain"),
                        "sub_domain": case.get("sub_domain"),
                        "difficulty": case.get("difficulty"),
                        "length": case.get("length"),
                        "context_chars": len(str(case.get("context", ""))),
                        "context_bytes": len(context),
                        "context_sha256": sha256_bytes(context),
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt_bytes),
                        "question": case.get("question"),
                        "choices": {letter: case.get(f"choice_{letter}") for letter in "ABCD"},
                        "gold": str(case.get("answer", "")),
                        "answer": answer,
                        "parsed_answer": parse_choice(answer or ""),
                        "score": score,
                        "elapsed_seconds": round(elapsed, 3),
                        "usage": usage,
                        "returncode": returncode,
                        "timed_out": timed_out,
                        "raw_events": str(artifacts_path.name + "/" + raw_relative),
                        "stderr_file": str(artifacts_path.name + "/" + stderr_relative),
                        "error": error,
                    },
                )
                print(f"  score={score:.0f} answer={answer!r}" + (f" error={error}" if error else ""))
                shutil.rmtree(case_directory)

        summary = {
            "type": "summary",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "cases": len(scores),
            "failures": failures,
            "accuracy": sum(scores) / len(scores),
            "correct": int(sum(scores)),
            "usage": usage_total,
        }
        write_record(output, summary)

    print(
        f"accuracy={summary['accuracy']:.4f} "
        f"correct={summary['correct']}/{summary['cases']} failures={failures}"
    )
    print(f"results={output_path}")
    return 0 if failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the Pi IPython RLM against the paper's 50-case LongBench-v2 CodeQA split."
    )
    parser.add_argument("--model", default=os.environ.get("PI_RLM_BENCH_MODEL"))
    parser.add_argument("--thinking", default=os.environ.get("PI_RLM_BENCH_THINKING", "low"))
    parser.add_argument("--agent-dir", default=os.environ.get("PI_RLM_BENCH_AGENT_DIR"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Streaming shuffle buffer. The paper profile keeps dataset order.",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Per-case timeout in seconds.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / f"longbench-codeqa-{timestamp}.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select and print cases without calling Pi.")
    args = parser.parse_args(argv)
    if not args.model and not args.dry_run:
        parser.error("--model or PI_RLM_BENCH_MODEL is required")
    if not 1 <= args.count <= 50:
        parser.error("--count must be between 1 and 50")
    if args.shuffle_buffer < 0:
        parser.error("--shuffle-buffer cannot be negative")
    return args


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
