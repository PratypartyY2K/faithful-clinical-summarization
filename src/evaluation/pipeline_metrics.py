"""Evaluation utilities for the faithful clinical summarization pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.preprocessing.io import read_jsonl


SUPPORTED_LABEL_NAMES = {"supported", "entailment", "entailed"}
CONTRADICTION_LABEL_NAMES = {"contradiction", "contradicted"}
NEUTRAL_LABEL_NAMES = {"neutral", "unsupported"}


def is_supported_claim(score: Dict[str, object]) -> bool:
    return str(score.get("predicted_label_name", "")).lower() in SUPPORTED_LABEL_NAMES


def normalize_score_label(score: Dict[str, object]) -> str:
    return str(score.get("predicted_label_name", "")).lower()


def compute_claim_support_rate(claim_scores: Iterable[Dict[str, object]]) -> float:
    scores = list(claim_scores)
    if not scores:
        return 0.0
    supported = sum(1 for row in scores if is_supported_claim(row))
    return round(supported / len(scores), 4)


def summarize_claim_labels(claim_scores: Iterable[Dict[str, object]]) -> Dict[str, float]:
    scores = list(claim_scores)
    total = len(scores)
    if total == 0:
        return {
            "support_rate": 0.0,
            "contradiction_rate": 0.0,
            "neutral_or_unsupported_rate": 0.0,
        }
    contradiction_count = 0
    neutral_count = 0
    supported_count = 0
    for score in scores:
        label_name = normalize_score_label(score)
        if label_name in CONTRADICTION_LABEL_NAMES:
            contradiction_count += 1
        elif label_name in SUPPORTED_LABEL_NAMES:
            supported_count += 1
        else:
            neutral_count += 1
    return {
        "support_rate": round(supported_count / total, 4),
        "contradiction_rate": round(contradiction_count / total, 4),
        "neutral_or_unsupported_rate": round(neutral_count / total, 4),
    }


def compute_text_overlap_metrics(predictions: List[str], references: List[str]) -> Dict[str, object]:
    import evaluate

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
    label_summaries = [summarize_claim_labels(report["claim_scores"]) for report in reports]
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
        "avg_contradiction_rate": round(
            sum(summary["contradiction_rate"] for summary in label_summaries) / len(label_summaries),
            4,
        ) if label_summaries else 0.0,
        "avg_neutral_or_unsupported_rate": round(
            sum(summary["neutral_or_unsupported_rate"] for summary in label_summaries) / len(label_summaries),
            4,
        ) if label_summaries else 0.0,
        **overlap_metrics,
    }


def evaluate_verifier_dataset(
    verifier_examples: List[Dict[str, object]],
    verifier_tokenizer,
    verifier_model,
    batch_size: int = 16,
) -> Dict[str, float]:
    from src.modeling.pipeline import score_claims

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    if any(keyword in normalized for keyword in ("mg", "mcg", "daily", "nightly", "twice", "dose", "tablet", "capsule", "units")):
        return "dosage_or_frequency"
    if any(keyword in normalized for keyword in ("medication", "drug", "prescribed", "started", "stopped", "continued")):
        return "medication"
    if any(keyword in normalized for keyword in ("lab", "exam", "imaging", "scan", "x-ray", "mri", "ct", "blood pressure", "heart rate", "oxygen saturation")):
        return "lab_or_finding"
    if any(keyword in normalized for keyword in ("follow up", "repeat", "log", "review", "return if")):
        return "follow_up"
    if any(keyword in normalized for keyword in ("diagnosis", "history of", "assessment", "impression", "syndrome", "disease", "infection")):
        return "diagnosis"
    if any(keyword in normalized for keyword in ("pain", "fever", "nausea", "vomiting", "headache", "fatigue", "cough", "shortness of breath", "symptom")):
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


def build_evaluation_summary(
    generation_metrics: Dict[str, object],
    verifier_metrics: Dict[str, object],
    qualitative_error_analysis: Dict[str, object],
) -> Dict[str, object]:
    return {
        "faithfulness_summary": {
            "factscore": generation_metrics.get("factscore", 0.0),
            "avg_claim_support_rate": generation_metrics.get("avg_claim_support_rate", 0.0),
            "avg_contradiction_rate": generation_metrics.get("avg_contradiction_rate", 0.0),
            "avg_neutral_or_unsupported_rate": generation_metrics.get("avg_neutral_or_unsupported_rate", 0.0),
        },
        "overlap_summary": {
            "rouge": generation_metrics.get("rouge", {}),
            "bertscore": generation_metrics.get("bertscore", {}),
        },
        "verifier_summary": verifier_metrics,
        "error_summary": qualitative_error_analysis.get("error_type_counts", {}),
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
    claim_extractor_backend: str = "heuristic",
    claim_extractor_model: str = "gpt-4.1-mini",
) -> Dict[str, object]:
    from src.modeling.pipeline import build_pipeline_report

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
            claim_extractor_backend=claim_extractor_backend,
            claim_extractor_model=claim_extractor_model,
        )
        for example in raw_examples
    ]
    generation_metrics = evaluate_generated_reports(reports)
    verifier_metrics = evaluate_verifier_dataset(
        verifier_examples=verifier_examples,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
        batch_size=verifier_batch_size,
    )
    qualitative_error_analysis = build_qualitative_error_analysis(reports)
    return {
        "generation_metrics": generation_metrics,
        "verifier_metrics": verifier_metrics,
        "qualitative_error_analysis": qualitative_error_analysis,
        "summary": build_evaluation_summary(
            generation_metrics=generation_metrics,
            verifier_metrics=verifier_metrics,
            qualitative_error_analysis=qualitative_error_analysis,
        ),
        "example_reports": reports,
    }


def write_evaluation_report(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
