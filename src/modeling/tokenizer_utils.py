"""Tokenizer helpers for summarization inference."""

from __future__ import annotations


def configure_generation_tokenizer(tokenizer, model):
    if getattr(model.config, "is_encoder_decoder", False):
        return tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer
