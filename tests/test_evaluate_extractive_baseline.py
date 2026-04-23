from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_extractive_baseline.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_extractive_baseline_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
evaluate_extractive_baseline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(evaluate_extractive_baseline)


class EvaluateExtractiveBaselineTest(unittest.TestCase):
    def test_build_prediction_uses_dialogue_for_raw_examples(self) -> None:
        example = {
            "example_id": "raw-1",
            "dialogue": "Sentence one. Sentence two. Sentence three.",
            "summary": "reference",
        }

        prediction = evaluate_extractive_baseline.build_prediction(
            example,
            strategy="lead",
            num_sentences=2,
        )

        self.assertEqual(prediction, "Sentence one. Sentence two.")

    def test_build_prediction_uses_input_text_for_processed_examples(self) -> None:
        example = {
            "example_id": "proc-1",
            "input_text": "Alpha. Beta. Gamma.",
            "target_text": "reference",
        }

        prediction = evaluate_extractive_baseline.build_prediction(
            example,
            strategy="lead",
            num_sentences=1,
        )

        self.assertEqual(prediction, "Alpha.")


if __name__ == "__main__":
    unittest.main()
