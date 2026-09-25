from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from benchmarks.longbench import (
    benchmark_prompt,
    multiple_choice_score,
    parse_args,
    parse_choice,
    write_context,
)


class LongBenchBenchmarkTests(unittest.TestCase):
    def test_paper_profile_uses_all_50_codeqa_cases(self) -> None:
        args = parse_args(["--dry-run"])
        self.assertEqual(args.count, 50)
        self.assertEqual(args.shuffle_buffer, 0)

    def test_choice_parser_accepts_terse_wrappers_only(self) -> None:
        self.assertEqual(parse_choice("1"), "1")
        self.assertEqual(parse_choice("Answer: (2)"), "2")
        self.assertEqual(parse_choice("option 3."), "3")
        self.assertIsNone(parse_choice("The answer is 1 because the method accepts h."))
        self.assertIsNone(parse_choice(""))

    def test_multiple_choice_scorer(self) -> None:
        case = {"answer": "B"}
        self.assertEqual(multiple_choice_score(case, "1"), 1.0)
        self.assertEqual(multiple_choice_score(case, "Answer: 1"), 1.0)
        self.assertEqual(multiple_choice_score(case, "0"), 0.0)

    def test_context_file_and_prompt_do_not_expose_gold(self) -> None:
        case = {
            "_id": "case-1",
            "context": "LONG_REPOSITORY_SENTINEL",
            "question": "Which option is correct?",
            "choice_A": "alpha",
            "choice_B": "beta",
            "choice_C": "gamma",
            "choice_D": "delta",
            "answer": "SECRET_GOLD",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.txt"
            write_context(path, case)
            model_context = path.read_text()
        prompt = benchmark_prompt(case)
        self.assertEqual(model_context, "LONG_REPOSITORY_SENTINEL")
        self.assertNotIn("SECRET_GOLD", model_context)
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertNotIn("LONG_REPOSITORY_SENTINEL", prompt)
        self.assertIn(case["question"], prompt)
        self.assertIn("0: alpha", prompt)
        self.assertIn("`context.txt`", prompt)


if __name__ == "__main__":
    unittest.main()
