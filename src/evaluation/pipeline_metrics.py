"""Evaluation utilities for the synthetic faithful summarization pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.modeling.pipeline import build_pipeline_report, score_claims
from src.preprocessing.io import read_jsonl


SUPPORTED_LABEL_NAMES = {"supported", "entailment", "entailed"}
CONTRADICTION_LABEL_NAMES = {"contradiction", "contradicted"}
NEUTRAL_LABEL_NAMES = {"neutral", "unsupported"}


def is_supported_claim(score: Dict[str, object]) -> bool:
    return str(score.get("predicted_label_name", "")).lower() in SUPPORTED_LABEL_NAMES


def compute_claim_support_rate(claim_scores: Iterable[Dict[str, object]]) -> float:
    scores = list(claim_scores)
    if not scores:
        return 0.0
    supported = sum(1 for row in scores if is_supported_claim(row))
    return round(supported / len(scores), 4)


def compute_text_overlap_metrics(predictions: List[str], references: List[str]) -> Dict[str, object]:
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bertscore = evaluate.load("bertscore")
    bertscore_scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_summary = {
        "precision": round(sum(bertscore_scores["precision"]) / len(bertscore_scores["precision"]), 4),
        "recall": round(sum(bertscore_scores["recall"]) / len(bertscore_scores["recall"]), 4),
        "f1": round(sum(bertscore_scores["f1"]) / len(bertscore_scores["f1"]), 4),
    }
    return {
        "rouge": {key: round(value, 4) for key, value in rouge_scores.items()},
        "bertscore": bertscore_summary,
    }


def evaluate_generated_reports(reports: List[Dict[str, object]]) -> Dict[str, object]:
    predictions = [str(report["generated_summary"]) for report in reports]
    references = [str(report.get("reference_summary") or "") for report in reports]
    overlap_metrics = compute_text_overlap_metrics(predictions=predictions, references=references)
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
        "factscore": round(sum(claim_rates) / len(claim_rates), 4) if claim_rates else 0.0,
        **overlap_metrics,
    }


def evaluate_verifier_dataset(
    verifier_examples: List[Dict[str, object]],
    verifier_tokenizer,
    verifier_model,
    batch_size: int = 16,
) -> Dict[str, float]:
    predicted_labels: List[int] = []
    gold_labels: List[int] = []
    for example in verifier_examples:
        score = score_claims(
            dialogue=str(example["dialogue"]),
            claims=[str(example["claim"])],
            verifier_tokenizer=verifier_tokenizer,
            verifier_model=verifier_model,
            batch_size=batch_size,
        )[0]
        predicted_labels.append(int(score["predicted_label"]))
        gold_labels.append(int(example["label"]))

    if not gold_labels:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    average = "binary" if len(set(gold_labels)) <= 2 else "macro"
    return {
        "accuracy": round(float(accuracy_score(gold_labels, predicted_labels)), 4),
        "macro_f1": round(float(f1_score(gold_labels, predicted_labels, average="macro", zero_division=0)), 4),
        "precision": round(float(precision_score(gold_labels, predicted_labels, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(gold_labels, predicted_labels, average=average, zero_division=0)), 4),
        "f1": round(float(f1_score(gold_labels, predicted_labels, average=average, zero_division=0)), 4),
    }


def classify_error_category(claim: str) -> str:
    normalized = claim.lower()
    if any(keyword in normalized for keyword in ("mg", "daily", "nightly", "twice", "dose", "puffs")):
        return "dosage_or_frequency"
    if any(keyword in normalized for keyword in ("metformin", "lisinopril", "atorvastatin", "nitrofurantoin", "sumatriptan", "albuterol")):
        return "medication"
    if any(keyword in normalized for keyword in ("a1c", "ldl", "urinalysis", "blood pressure", "oxygen saturation", "exam")):
        return "lab_or_finding"
    if any(keyword in normalized for keyword in ("follow up", "repeat", "log", "review", "return if")):
        return "follow_up"
    if any(keyword in normalized for keyword in ("diabetes", "hypertension", "asthma", "migraine", "infection", "hyperlipidemia")):
        return "diagnosis"
    if any(keyword in normalized for keyword in ("fatigue", "headache", "wheezing", "burning", "symptom", "nausea")):
        return "symptom"
    return "other"


def build_qualitative_error_analysis(reports: List[Dict[str, object]], max_examples: int = 5) -> Dict[str, object]:
    label_buckets = {
        "contradiction": 0,
        "neutral_or_unsupported": 0,
    }
    category_counts: Dict[str, int] = {}
    examples: List[Dict[str, object]] = []
    for report in reports:
        unsupported_claims = []
        for score in report["claim_scores"]:
            if is_supported_claim(score):
                continue
            label_name = str(score.get("predicted_label_name", "")).lower()
            if label_name in CONTRADICTION_LABEL_NAMES:
                label_buckets["contradiction"] += 1
            else:
                label_buckets["neutral_or_unsupported"] += 1
            category = classify_error_category(str(score["claim"]))
            category_counts[category] = category_counts.get(category, 0) + 1
            unsupported_claims.append(
                {
                    "claim": score["claim"],
                    "predicted_label_name": score["predicted_label_name"],
                    "supported_probability": score["supported_probability"],
                    "category": category,
                }
            )
        if unsupported_claims and len(examples) < max_examples:
            examples.append(
                {
                    "example_id": report["example_id"],
                    "generated_summary": report["generated_summary"],
                    "reference_summary": report.get("reference_summary"),
                    "unsupported_claims": unsupported_claims,
                }
            )
    top_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "error_type_counts": label_buckets,
        "claim_category_counts": dict(top_categories),
        "representative_examples": examples,
    }


def run_full_evaluation(
    raw_examples_path: Path,
    verifier_examples_path: Path,
    summarizer_tokenizer,
    summarizer_model,
    verifier_tokenizer,
    verifier_model,
    max_new_tokens: int = 96,
    verifier_batch_size: int = 16,
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
            verifier_batch_size=verifier_batch_size,
        )
        for example in raw_examples
    ]
    return {
        "generation_metrics": evaluate_generated_reports(reports),
        "verifier_metrics": evaluate_verifier_dataset(
            verifier_examples=verifier_examples,
            verifier_tokenizer=verifier_tokenizer,
            verifier_model=verifier_model,
            batch_size=verifier_batch_size,
        ),
        "qualitative_error_analysis": build_qualitative_error_analysis(reports),
        "example_reports": reports,
    }


def write_evaluation_report(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
