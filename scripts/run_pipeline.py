#!/usr/bin/env python3
"""Run a lightweight generate-and-verify pipeline on one dialogue."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def read_first_example(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def split_into_claims(summary: str) -> List[str]:
    raw_claims = re.split(r"(?<=[.!?])\s+", summary.strip())
    return [claim.strip() for claim in raw_claims if claim.strip()]


def score_claims(
    dialogue: str,
    claims: List[str],
    verifier_tokenizer,
    verifier_model,
) -> List[Dict[str, object]]:
    scores: List[Dict[str, object]] = []
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
        scores.append(
            {
                "claim": claim,
                "supported_probability": round(float(probabilities[1]), 4),
                "predicted_label": int(torch.argmax(probabilities).item()),
            }
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path("data/dummy/raw/test.jsonl"))
    parser.add_argument("--summarizer-dir", type=Path, default=Path("artifacts/summarizer"))
    parser.add_argument("--verifier-dir", type=Path, default=Path("artifacts/verifier"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-file", type=Path, default=Path("artifacts/pipeline_report.json"))
    args = parser.parse_args()

    example = read_first_example(args.input_file)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(str(args.summarizer_dir))
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(str(args.summarizer_dir))
    verifier_tokenizer = AutoTokenizer.from_pretrained(str(args.verifier_dir))
    verifier_model = AutoModelForSequenceClassification.from_pretrained(str(args.verifier_dir))

    prompt = f"Summarize this clinical dialogue:\n{example['dialogue']}"
    encoded = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True)
    generated = summarizer_model.generate(**encoded, max_new_tokens=args.max_new_tokens)
    summary = summarizer_tokenizer.decode(generated[0], skip_special_tokens=True)

    claims = split_into_claims(summary)
    claim_scores = score_claims(
        dialogue=example["dialogue"],
        claims=claims,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
    )
    report = {
        "example_id": example["example_id"],
        "dialogue": example["dialogue"],
        "generated_summary": summary,
        "claim_scores": claim_scores,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
