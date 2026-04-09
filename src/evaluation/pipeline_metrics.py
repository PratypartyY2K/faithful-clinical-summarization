"""Evaluation utilities for the synthetic faithful summarization pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.modeling.pipeline import build_pipeline_report, score_claims
from src.preprocessing.io import read_jsonl


def compute_claim_support_rate(claim_scores: Iterable[Dict[str, object]]) -> float:
    scores = list(claim_scores)
    if not scores:
        return 0.0
    supported = sum(1 for row in scores if int(row["predicted_label"]) == 1)
    return round(supported / len(scores), 4)


def evaluate_generated_reports(reports: List[Dict[str, object]]) -> Dict[str, object]:
    rouge = evaluate.load("rouge")
    predictions = [str(report["generated_summary"]) for report in reports]
    references = [str(report.get("reference_summary") or "") for report in reports]
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    claim_rates = [compute_claim_support_rate(report["claim_scores"]) for report in reports]
    avg_claim_count = (
        round(sum(len(report["claim_scores"]) for report in reports) / len(reports), 4)
        if reports
        else 0.0
    )
    return {
        "num_examples": len(reports),
        "avg_claim_count": avg_claim_count,
        "avg_claim_support_rate": round(sum(claim_rates) / len(claim_rates), 4) if claim_rates else 0.0,
        "rouge": {key: round(value, 4) for key, value in rouge_scores.items()},
    }


def evaluate_verifier_dataset(
    verifier_examples: List[Dict[str, object]],
    verifier_tokenizer,
    verifier_model,
) -> Dict[str, float]:
    predicted_labels: List[int] = []
    gold_labels: List[int] = []
    for example in verifier_examples:
        score = score_claims(
            dialogue=str(example["dialogue"]),
            claims=[str(example["claim"])],
            verifier_tokenizer=verifier_tokenizer,
            verifier_model=verifier_model,
        )[0]
        predicted_labels.append(int(score["predicted_label"]))
        gold_labels.append(int(example["label"]))

    if not gold_labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    return {
        "accuracy": round(float(accuracy_score(gold_labels, predicted_labels)), 4),
        "precision": round(float(precision_score(gold_labels, predicted_labels, zero_division=0)), 4),
        "recall": round(float(recall_score(gold_labels, predicted_labels, zero_division=0)), 4),
        "f1": round(float(f1_score(gold_labels, predicted_labels, zero_division=0)), 4),
    }


def run_full_evaluation(
    raw_examples_path: Path,
    verifier_examples_path: Path,
    summarizer_tokenizer,
    summarizer_model,
    verifier_tokenizer,
    verifier_model,
    max_new_tokens: int = 96,
) -> Dict[str, object]:
    raw_examples = read_jsonl(raw_examples_path)
    verifier_examples = read_jsonl(verifier_examples_path)
    reports = [
        build_pipeline_report(
            example=example,
            summarizer_tokenizer=summarizer_tokenizer,
            summarizer_model=summarizer_model,
            verifier_tokenizer=verifier_tokenizer,
            verifier_model=verifier_model,
            max_new_tokens=max_new_tokens,
        )
        for example in raw_examples
    ]
    return {
        "generation_metrics": evaluate_generated_reports(reports),
        "verifier_metrics": evaluate_verifier_dataset(
            verifier_examples=verifier_examples,
            verifier_tokenizer=verifier_tokenizer,
            verifier_model=verifier_model,
        ),
        "example_reports": reports,
    }


def write_evaluation_report(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
