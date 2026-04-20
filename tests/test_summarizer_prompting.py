from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.modeling.tokenizer_utils import configure_generation_tokenizer


class SummarizerPromptingTest(unittest.TestCase):
    def test_decoder_only_generation_uses_left_padding_and_fills_pad_token(self) -> None:
        tokenizer = SimpleNamespace(padding_side="right", pad_token=None, eos_token="</s>", unk_token="<unk>")
        model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))

        configured = configure_generation_tokenizer(tokenizer, model)

        self.assertIs(configured, tokenizer)
        self.assertEqual(configured.padding_side, "left")
        self.assertEqual(configured.pad_token, "</s>")

    def test_encoder_decoder_generation_leaves_padding_unchanged(self) -> None:
        tokenizer = SimpleNamespace(padding_side="right", pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
        model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True))

        configured = configure_generation_tokenizer(tokenizer, model)

        self.assertIs(configured, tokenizer)
        self.assertEqual(configured.padding_side, "right")
        self.assertEqual(configured.pad_token, "<pad>")


if __name__ == "__main__":
    unittest.main()
