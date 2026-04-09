"""Shared model loading and generate-then-verify inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.preprocessing.claim_extractor import split_into_claims


VerifierBundle = Tuple[object, AutoModelForSequenceClassification]
SummarizerBundle = Tuple[object, AutoModelForSeq2SeqLM]


def load_summarizer(model_dir: Path) -> SummarizerBundle:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    return tokenizer, model


def load_verifier(model_dir: Path) -> VerifierBundle:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return tokenizer, model


def generate_summary(
    dialogue: str,
    summarizer_tokenizer,
    summarizer_model,
    max_new_tokens: int = 96,
) -> str:
    prompt = f"Summarize this clinical dialogue:\n{dialogue}"
    encoded = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        generated = summarizer_model.generate(**encoded, max_new_tokens=max_new_tokens)
    return summarizer_tokenizer.decode(generated[0], skip_special_tokens=True)


def score_claims(
    dialogue: str,
    claims: List[str],
    verifier_tokenizer,
    verifier_model,
) -> List[Dict[str, object]]:
    scores: List[Dict[str, object]] = []
    label_lookup = {0: "unsupported", 1: "supported"}
    for claim in claims:
        encoded = verifier_tokenizer(
            dialogue,
            claim,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = verifier_model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_label = int(torch.argmax(probabilities).item())
        scores.append(
            {
                "claim": claim,
                "supported_probability": round(float(probabilities[min(1, len(probabilities) - 1)]), 4),
                "predicted_label": predicted_label,
                "predicted_label_name": label_lookup.get(predicted_label, str(predicted_label)),
            }
        )
    return scores


def build_pipeline_report(
    example: Dict[str, object],
    summarizer_tokenizer,
    summarizer_model,
    verifier_tokenizer,
    verifier_model,
    max_new_tokens: int = 96,
) -> Dict[str, object]:
    summary = generate_summary(
        dialogue=str(example["dialogue"]),
        summarizer_tokenizer=summarizer_tokenizer,
        summarizer_model=summarizer_model,
        max_new_tokens=max_new_tokens,
    )
    claims = split_into_claims(summary)
    claim_scores = score_claims(
        dialogue=str(example["dialogue"]),
        claims=claims,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
    )
    return {
        "example_id": example["example_id"],
        "dialogue": example["dialogue"],
        "reference_summary": example.get("summary"),
        "generated_summary": summary,
        "claim_scores": claim_scores,
    }
