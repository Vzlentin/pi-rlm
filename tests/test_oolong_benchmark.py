from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from benchmarks.oolong import (
    attempt_answer_parse,
    benchmark_prompt,
    final_answer_from_events,
    parse_args,
    synth_score,
    usage_from_events,
    write_context,
)


class OolongBenchmarkTests(unittest.TestCase):
    def test_default_profile_uses_all_400_held_out_cases(self) -> None:
        args = parse_args(["--dry-run"])
        self.assertEqual(args.split, "test")
        self.assertEqual(args.count, 400)
        self.assertIsNone(args.dataset)
        self.assertEqual(args.min_context, 131_072)
        self.assertEqual(args.max_context, 131_072)
        self.assertEqual(args.shuffle_buffer, 0)

    def test_upstream_answer_parser(self) -> None:
        self.assertEqual(attempt_answer_parse("Answer: **human being**")[0], "human being")
        self.assertEqual(
            attempt_answer_parse("The first is less common than the second.")[0],
            "less common than",
        )
        self.assertEqual(attempt_answer_parse("")[0], "")

    def test_upstream_scorer(self) -> None:
        self.assertEqual(synth_score({"answer": "['ENTITY']"}, "entity"), 1.0)
        self.assertEqual(
            synth_score(
                {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"},
                "12",
            ),
            0.75**2,
        )
        self.assertEqual(
            synth_score({"answer": "['abbreviation']"}, "The answer is abbreviation."),
            1.0,
        )
        self.assertEqual(synth_score({"answer": "['entity']"}, ""), 0.0)

    def test_context_file_does_not_expose_gold_and_prompt_does_not_expose_context(self) -> None:
        case = {
            "question": "Which category is most common?",
            "context_window_text": "LONG_CONTEXT_SENTINEL",
            "answer": "['SECRET_GOLD']",
            "answer_type": "ANSWER_TYPE.CATEGORY",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.txt"
            write_context(path, case)
            model_context = path.read_text()
        prompt = benchmark_prompt(case)
        self.assertEqual(model_context, "LONG_CONTEXT_SENTINEL")
        self.assertNotIn("SECRET_GOLD", model_context)
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertNotIn("LONG_CONTEXT_SENTINEL", prompt)
        self.assertIn(case["question"], prompt)
        self.assertIn("`context.txt`", prompt)

    def test_extracts_the_successful_final_and_separates_usage(self) -> None:
        usage = {
            "input": 2,
            "output": 3,
            "cacheRead": 4,
            "cacheWrite": 5,
            "totalTokens": 14,
            "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.3},
        }
        events = [
            {"type": "message_end", "message": {"role": "assistant", "usage": usage}},
            {
                "type": "tool_execution_end",
                "toolName": "ipython",
                "isError": False,
                "result": {
                    "details": {
                        "final": {"answer": "entity"},
                        "nestedUsage": usage,
                    },
                },
            },
        ]
        self.assertEqual(final_answer_from_events(events), "entity")
        measured = usage_from_events(events)
        self.assertEqual(measured["root"]["totalTokens"], 14)
        self.assertEqual(measured["nested"]["totalTokens"], 14)
        self.assertEqual(measured["total"]["totalTokens"], 28)
        self.assertAlmostEqual(measured["total"]["cost"]["total"], 0.6)

    def test_rejects_missing_and_error_finals_and_keeps_the_last(self) -> None:
        successful = {
            "type": "tool_execution_end",
            "toolName": "ipython",
            "isError": False,
            "result": {"details": {"final": {"answer": "entity"}}},
        }
        failed = {
            **successful,
            "isError": True,
        }
        with self.assertRaisesRegex(RuntimeError, "got 0"):
            final_answer_from_events([])
        with self.assertRaisesRegex(RuntimeError, "got 0"):
            final_answer_from_events([failed])
        later = {**successful, "result": {"details": {"final": {"answer": "location"}}}}
        self.assertEqual(final_answer_from_events([successful, later]), "location")


if __name__ == "__main__":
    unittest.main()
