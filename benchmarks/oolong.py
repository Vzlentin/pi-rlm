#!/usr/bin/env python3
"""Run pi-ipython with pi-rlm against the RLM OOLONG eval."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
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
DATASET_ID = "oolongbench/oolong-synth"
DATASET_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
RLM_SOURCE_COMMIT = "854e688fbba9d8f8989e3da9989812e4b6dfe270"
RLM_SOURCE_URL = (
    "https://github.com/alexzhang13/rlm/blob/"
    f"{RLM_SOURCE_COMMIT}/training/environments/oolong/oolong/env.py"
)
COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")
HASHED_INPUTS = (
    "benchmarks/harness.py",
    "benchmarks/oolong.py",
    "benchmarks/requirements.txt",
    "extensions/rlm.ts",
    "extensions/rlm-host.ts",
    "extensions/librlm.ts",
    "package-lock.json",
)


def find_comparison_phrase(output: str) -> str | None:
    """Return the final comparison phrase, matching the upstream RLM scorer."""
    output_lower = output.lower()
    hits = [(output_lower.rfind(phrase), phrase) for phrase in COMPARISON_PHRASES if phrase in output_lower]
    return max(hits)[1] if hits else None


def attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Parse one model answer using the upstream RLM scorer's rules."""
    comparison = find_comparison_phrase(answer)
    if comparison is not None:
        return comparison, "high"
    if not answer.strip():
        return "", "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, "low"
        return answer.split()[-1], "low"
    candidate = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    if len(candidate) < 20:
        return candidate, "vhigh"
    return candidate, "med"


def synth_score(datapoint: dict[str, Any], output: str) -> float:
    """Score one OOLONG synth answer using alexzhang13/rlm's rubric."""
    if not output.strip():
        return 0.0
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    trimmed, _confidence = attempt_answer_parse(output)
    gold_string = str(gold)
    if trimmed == gold_string or trimmed.lower() == gold_string.lower():
        return 1.0

    answer_type = datapoint.get("answer_type", "")
    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except Exception:
            return 0.0
    if answer_type == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except Exception:
            return 0.0

    comparison_answers = [phrase.lower() for phrase in COMPARISON_PHRASES]
    if gold_string and gold_string.lower() not in comparison_answers:
        if gold_string.lower() in output.lower():
            return 1.0
    return 0.0


def load_cases(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Stream the bounded OOLONG profile without retaining its huge contexts."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The benchmark needs its pinned direct dependencies. Run it through `npm run eval:oolong`."
        ) from error

    filters: list[tuple[str, str, Any]] = [
        ("context_len", ">=", args.min_context),
        ("context_len", "<=", args.max_context),
    ]
    if args.dataset:
        filters.append(("dataset", "=", args.dataset))
    if args.exclude_numeric:
        filters.append(("answer_type", "!=", "ANSWER_TYPE.NUMERIC"))
    stream = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split=args.split,
        streaming=True,
        filters=filters,
    )
    if args.shuffle_buffer:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    selected = 0
    for case in stream:
        if args.dataset and case.get("dataset") != args.dataset:
            continue
        context_length = int(case.get("context_len", 0))
        if not args.min_context <= context_length <= args.max_context:
            continue
        if args.exclude_numeric and case.get("answer_type") == "ANSWER_TYPE.NUMERIC":
            continue
        yield dict(case)
        selected += 1
        if selected >= args.count:
            return
    raise RuntimeError(
        f"Requested {args.count} cases but found {selected} for split={args.split!r}, "
        f"dataset={args.dataset!r}, context={args.min_context}..{args.max_context}."
    )


def context_bytes(case: dict[str, Any]) -> bytes:
    context = case.get("context_window_text", case.get("context", ""))
    if not isinstance(context, str):
        raise RuntimeError(f"OOLONG case {case.get('id')} has a non-string context")
    return context.encode("utf-8")


def write_context(path: Path, case: dict[str, Any]) -> bytes:
    """Write only the long context. The question is exposed in the root prompt."""
    payload = context_bytes(case)
    path.write_bytes(payload)
    path.chmod(0o600)
    return payload


def benchmark_prompt(case: dict[str, Any]) -> str:
    question = str(case["question"])
    return f'''Solve one OOLONG long-context QA case with the ipython tool and its RLM children.

Question: {question}

The long context is stored in `context.txt` in the working directory. Load it inside IPython. Keep the context inside IPython and explicit `rlm.spawn(..., context=...)` calls rather than printing it into the parent conversation. Use Python to inspect and divide the context, and use focused children where language understanding is needed.

When you have the answer, report it with:
await rlm.final({{"answer": "<answer only>"}})

The `answer` value must contain only the benchmark answer, with no explanation.'''


def run_benchmark(args: argparse.Namespace) -> int:
    cases = load_cases(args)
    if args.dry_run:
        for index, case in enumerate(cases, 1):
            print(
                f"{index:02d} id={case.get('id')} dataset={case.get('dataset')} "
                f"context_len={case.get('context_len')} answer_type={case.get('answer_type')}"
            )
        return 0

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_path = output_path.with_suffix("").with_name(output_path.stem + ".artifacts")
    run_started = datetime.now(timezone.utc).isoformat()
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
                "started_at": run_started,
                "benchmark": {
                    "name": "OOLONG-synth",
                    "dataset": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "rlm_eval_source": RLM_SOURCE_URL,
                    "rlm_eval_commit": RLM_SOURCE_COMMIT,
                },
                "config": {
                    "model": args.model,
                    "thinking": args.thinking,
                    "split": args.split,
                    "dataset_name": args.dataset,
                    "min_context": args.min_context,
                    "max_context": args.max_context,
                    "count": args.count,
                    "seed": args.seed,
                    "shuffle_buffer": args.shuffle_buffer,
                    "exclude_numeric": args.exclude_numeric,
                    "timeout_seconds": args.timeout,
                },
                "repository": git_metadata(),
                "source_sha256": source_hashes(HASHED_INPUTS),
                "python_version": platform.python_version(),
                "dependency_versions": dependency_versions(
                    ("datasets", "python-dateutil", "huggingface-hub", "pyarrow")
                ),
                "pi_version": pi_version(),
            },
        )

        with tempfile.TemporaryDirectory(prefix="pi-rlm-oolong-") as temp_directory:
            for index, case in enumerate(cases, 1):
                case_directory = Path(temp_directory) / f"case-{index:03d}"
                case_directory.mkdir()
                context = write_context(case_directory / "context.txt", case)
                prompt = benchmark_prompt(case)
                prompt_bytes = prompt.encode("utf-8")
                print(f"[{index}/{args.count}] OOLONG id={case.get('id')}...", flush=True)

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

                score = synth_score(case, answer or "") if error is None else 0.0
                scores.append(score)
                add_usage(usage_total, usage["total"])
                write_record(
                    output,
                    {
                        "type": "case",
                        "index": index,
                        "id": case.get("id"),
                        "dataset": case.get("dataset"),
                        "context_len": case.get("context_len"),
                        "context_bytes": len(context),
                        "context_sha256": sha256_bytes(context),
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt_bytes),
                        "task_group": case.get("task_group"),
                        "task": case.get("task"),
                        "answer_type": case.get("answer_type"),
                        "question": case.get("question"),
                        "gold": str(case.get("answer", "")),
                        "answer": answer,
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
                print(f"  score={score:.4f} answer={answer!r}" + (f" error={error}" if error else ""))
                shutil.rmtree(case_directory)

        summary = {
            "type": "summary",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "cases": len(scores),
            "failures": failures,
            "mean_score": sum(scores) / len(scores),
            "fully_correct": sum(score == 1.0 for score in scores),
            "usage": usage_total,
        }
        write_record(output, summary)

    print(
        f"mean_score={summary['mean_score']:.4f} "
        f"fully_correct={summary['fully_correct']}/{summary['cases']} failures={failures}"
    )
    print(f"results={output_path}")
    return 0 if failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the Pi IPython RLM against the 400-case OOLONG held-out profile."
    )
    parser.add_argument("--model", default=os.environ.get("PI_RLM_BENCH_MODEL"))
    parser.add_argument("--thinking", default=os.environ.get("PI_RLM_BENCH_THINKING", "low"))
    parser.add_argument("--agent-dir", default=os.environ.get("PI_RLM_BENCH_AGENT_DIR"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--dataset",
        help="Optional OOLONG source dataset filter. The 400-case default uses all test datasets.",
    )
    parser.add_argument("--min-context", type=int, default=131_072)
    parser.add_argument("--max-context", type=int, default=131_072)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Streaming shuffle buffer. The default profile keeps dataset order.",
    )
    parser.add_argument("--exclude-numeric", action="store_true")
    parser.add_argument("--timeout", type=int, default=900, help="Per-case timeout in seconds.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / f"oolong-{timestamp}.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select and print cases without calling Pi.")
    args = parser.parse_args(argv)
    if not args.model and not args.dry_run:
        parser.error("--model or PI_RLM_BENCH_MODEL is required")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.min_context > args.max_context:
        parser.error("--min-context cannot exceed --max-context")
    if args.shuffle_buffer < 0:
        parser.error("--shuffle-buffer cannot be negative")
    return args


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
