from __future__ import annotations

import unittest

from src.preprocessing.io import build_summarization_rows, keep_narrative_target, take_first_sentences


class PreprocessingIOTest(unittest.TestCase):
    def test_take_first_sentences_keeps_requested_prefix(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"

        shortened = take_first_sentences(text, max_sentences=2)

        self.assertEqual(shortened, "First sentence. Second sentence!")

    def test_build_summarization_rows_preserves_full_target_and_shortens_debug_target(self) -> None:
        rows = build_summarization_rows(
            [
                {
                    "example_id": "ex-1",
                    "dialogue": "source text",
                    "summary": "Sentence one. Sentence two. Sentence three.",
                }
            ],
            target_sentence_limit=2,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_text"], "Sentence one. Sentence two.")
        self.assertEqual(rows[0]["target_text_full"], "Sentence one. Sentence two. Sentence three.")
        self.assertEqual(rows[0]["target_sentence_limit"], 2)

    def test_keep_narrative_target_rejects_structured_problem_list(self) -> None:
        text = "# CHF: Improved. # ARF: Improved. # Pain: Controlled."

        keep = keep_narrative_target(
            text,
            narrative_only=True,
            min_target_words=3,
            max_target_words=100,
            min_target_sentences=2,
            max_structured_markers=0,
        )

        self.assertFalse(keep)

    def test_build_summarization_rows_filters_non_narrative_targets(self) -> None:
        rows = build_summarization_rows(
            [
                {
                    "example_id": "keep-me",
                    "dialogue": "source",
                    "summary": "Patient underwent surgery without complication. She recovered well on the floor and was discharged home.",
                },
                {
                    "example_id": "drop-me",
                    "dialogue": "source",
                    "summary": "# CHF: Stable. # Pain: Controlled.",
                },
            ],
            narrative_only=True,
            min_target_words=5,
            max_target_words=50,
            min_target_sentences=2,
            max_structured_markers=0,
        )

        self.assertEqual([row["example_id"] for row in rows], ["keep-me"])


if __name__ == "__main__":
    unittest.main()
