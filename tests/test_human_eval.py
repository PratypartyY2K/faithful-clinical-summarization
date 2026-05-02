from __future__ import annotations

import unittest
from unittest.mock import patch

from src.evaluation.human_eval import (
    aggregate_annotations_by_example,
    build_annotation_rows,
    evaluate_human_annotations,
    normalize_human_label,
)


class HumanEvalTest(unittest.TestCase):
    def test_build_annotation_rows_flattens_claim_scores(self) -> None:
        rows = build_annotation_rows(
            [
                {
                    "example_id": "ex-1",
                    "dialogue": "source",
                    "generated_summary": "summary",
                    "reference_summary": "reference",
                    "claim_scores": [
                        {
                            "claim": "Claim one.",
                            "predicted_label_name": "entailment",
                            "supported_probability": 0.93,
                        }
                    ],
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["claim_index"], 1)
        self.assertEqual(rows[0]["human_label"], "")

    def test_normalize_human_label_maps_supported_and_unsupported_forms(self) -> None:
        self.assertEqual(normalize_human_label("Entailment"), "supported")
        self.assertEqual(normalize_human_label("unsupported"), "neutral")
        self.assertEqual(normalize_human_label("contradiction"), "contradiction")

    def test_aggregate_annotations_by_example_computes_support_rate(self) -> None:
        summary = aggregate_annotations_by_example(
            [
                {"example_id": "ex-1", "human_label": "supported", "generated_summary": "g", "reference_summary": "r"},
                {"example_id": "ex-1", "human_label": "contradiction", "generated_summary": "g", "reference_summary": "r"},
            ]
        )
        self.assertEqual(summary["ex-1"]["human_claim_support_rate"], 0.5)

    @patch("src.evaluation.human_eval.compute_text_overlap_metrics", return_value={"rouge": {}, "bertscore": {}})
    @patch(
        "src.evaluation.human_eval.compute_example_level_overlap_metrics",
        return_value=[
            {"rouge1": 0.9, "rougeL": 0.8, "bertscore_f1": 0.91},
            {"rouge1": 0.3, "rougeL": 0.2, "bertscore_f1": 0.4},
        ],
    )
    def test_evaluate_human_annotations_returns_comparison_metrics(self, _overlap_rows, _aggregate_overlap) -> None:
        metrics = evaluate_human_annotations(
            [
                {
                    "example_id": "ex-1",
                    "human_label": "supported",
                    "verifier_predicted_label_name": "entailment",
                    "generated_summary": "gen1",
                    "reference_summary": "ref1",
                },
                {
                    "example_id": "ex-1",
                    "human_label": "contradiction",
                    "verifier_predicted_label_name": "contradiction",
                    "generated_summary": "gen1",
                    "reference_summary": "ref1",
                },
                {
                    "example_id": "ex-2",
                    "human_label": "unsupported",
                    "verifier_predicted_label_name": "entailment",
                    "generated_summary": "gen2",
                    "reference_summary": "ref2",
                },
            ]
        )
        self.assertEqual(metrics["num_annotated_claims"], 3)
        self.assertIn("human_vs_verifier_correlation", metrics)
        self.assertIn("overlap_vs_human_correlation", metrics)


if __name__ == "__main__":
    unittest.main()
