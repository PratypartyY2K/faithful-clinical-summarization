from __future__ import annotations

import unittest

from src.evaluation.pipeline_metrics import (
    build_evaluation_summary,
    build_qualitative_error_analysis,
    summarize_claim_labels,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_claim_label_summary_breaks_out_nli_rates(self) -> None:
        scores = [
            {"predicted_label_name": "entailment"},
            {"predicted_label_name": "contradiction"},
            {"predicted_label_name": "neutral"},
            {"predicted_label_name": "entailment"},
        ]
        summary = summarize_claim_labels(scores)
        self.assertEqual(summary["support_rate"], 0.5)
        self.assertEqual(summary["contradiction_rate"], 0.25)
        self.assertEqual(summary["neutral_or_unsupported_rate"], 0.25)

    def test_qualitative_analysis_collects_unsupported_examples(self) -> None:
        reports = [
            {
                "example_id": "encounter-0001",
                "generated_summary": "summary",
                "reference_summary": "ref",
                "claim_scores": [
                    {
                        "claim": "Continue lisinopril at 10 mg daily.",
                        "predicted_label_name": "entailment",
                        "supported_probability": 0.91,
                    },
                    {
                        "claim": "The patient denied hypertension.",
                        "predicted_label_name": "contradiction",
                        "supported_probability": 0.02,
                    },
                ],
            }
        ]
        analysis = build_qualitative_error_analysis(reports)
        self.assertEqual(analysis["error_type_counts"]["contradiction"], 1)
        self.assertEqual(len(analysis["representative_examples"]), 1)

    def test_evaluation_summary_surfaces_high_level_sections(self) -> None:
        summary = build_evaluation_summary(
            generation_metrics={"factscore": 0.8, "avg_claim_support_rate": 0.8, "avg_contradiction_rate": 0.1, "avg_neutral_or_unsupported_rate": 0.1, "rouge": {}, "bertscore": {}},
            verifier_metrics={"accuracy": 0.9},
            qualitative_error_analysis={"error_type_counts": {"contradiction": 2}},
        )
        self.assertIn("faithfulness_summary", summary)
        self.assertEqual(summary["error_summary"]["contradiction"], 2)


if __name__ == "__main__":
    unittest.main()
