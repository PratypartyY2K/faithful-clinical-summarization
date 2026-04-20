from __future__ import annotations

import unittest

from src.preprocessing.io import build_summarization_rows, take_first_sentences


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


if __name__ == "__main__":
    unittest.main()
