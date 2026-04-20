from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_summarizer.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_summarizer_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
evaluate_summarizer = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(evaluate_summarizer)


class EvaluateSummarizerTest(unittest.TestCase):
    def test_raw_example_schema_uses_dialogue_and_summary(self) -> None:
        example = {
            "example_id": "raw-1",
            "dialogue": "raw source",
            "summary": "raw summary",
        }

        self.assertEqual(evaluate_summarizer.get_dialogue_text(example), "raw source")
        self.assertEqual(evaluate_summarizer.get_reference_summary(example), "raw summary")
        self.assertEqual(evaluate_summarizer.get_reference_summary_full(example), "raw summary")

    def test_processed_example_schema_prefers_processed_target_fields(self) -> None:
        example = {
            "example_id": "proc-1",
            "input_text": "processed source",
            "target_text": "short target",
            "target_text_full": "full target",
        }

        self.assertEqual(evaluate_summarizer.get_dialogue_text(example), "processed source")
        self.assertEqual(evaluate_summarizer.get_reference_summary(example), "short target")
        self.assertEqual(evaluate_summarizer.get_reference_summary_full(example), "full target")


if __name__ == "__main__":
    unittest.main()
