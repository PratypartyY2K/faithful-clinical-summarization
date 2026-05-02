"""Utilities for manual claim-faithfulness annotation and analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

from src.evaluation.pipeline_metrics import (
    compute_example_level_overlap_metrics,
    compute_text_overlap_metrics,
    safe_correlation,
)
from src.preprocessing.io import read_jsonl


SUPPORTED_HUMAN_LABELS = {"supported", "support", "entailment", "entailed"}
CONTRADICTION_HUMAN_LABELS = {"contradiction", "contradicted"}
UNSUPPORTED_HUMAN_LABELS = {"neutral", "unsupported", "not enough information", "unknown"}


def normalize_human_label(label: str) -> str:
    normalized = " ".join(label.strip().lower().split())
    if normalized in SUPPORTED_HUMAN_LABELS:
        return "supported"
    if normalized in CONTRADICTION_HUMAN_LABELS:
        return "contradiction"
    if normalized in UNSUPPORTED_HUMAN_LABELS:
        return "neutral"
    return normalized


def is_human_supported(label: str) -> bool:
    return normalize_human_label(label) == "supported"


def read_annotation_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_annotation_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = [
            "example_id",
            "claim_index",
            "claim",
            "verifier_predicted_label_name",
            "verifier_supported_probability",
            "human_label",
            "human_notes",
            "generated_summary",
            "reference_summary",
            "dialogue",
        ]
    else:
        fieldnames = [str(key) for key in rows[0].keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_annotation_rows(example_reports: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for report in example_reports:
        for claim_index, score in enumerate(report.get("claim_scores", []), start=1):
            rows.append(
                {
                    "example_id": report["example_id"],
                    "claim_index": claim_index,
                    "claim": score["claim"],
                    "verifier_predicted_label_name": score.get("predicted_label_name", ""),
                    "verifier_supported_probability": score.get("supported_probability", ""),
                    "human_label": "",
                    "human_notes": "",
                    "generated_summary": report.get("generated_summary", ""),
                    "reference_summary": report.get("reference_summary", ""),
                    "dialogue": report.get("dialogue", ""),
                }
            )
    return rows


def aggregate_annotations_by_example(annotation_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in annotation_rows:
        human_label = normalize_human_label(str(row.get("human_label", "")))
        if not human_label:
            continue
        example_id = str(row["example_id"])
        summary = grouped.setdefault(
            example_id,
            {
                "example_id": example_id,
                "supported_count": 0,
                "unsupported_count": 0,
                "total_count": 0,
                "generated_summary": row.get("generated_summary", ""),
                "reference_summary": row.get("reference_summary", ""),
            },
        )
        summary["total_count"] += 1
        if human_label == "supported":
            summary["supported_count"] += 1
        else:
            summary["unsupported_count"] += 1
    for example_summary in grouped.values():
        total = int(example_summary["total_count"])
        supported = int(example_summary["supported_count"])
        example_summary["human_claim_support_rate"] = round(supported / total, 4) if total else 0.0
    return grouped


def evaluate_human_annotations(annotation_rows: List[Dict[str, str]]) -> Dict[str, object]:
    filtered_rows = [row for row in annotation_rows if str(row.get("human_label", "")).strip()]
    if not filtered_rows:
        return {
            "num_annotated_claims": 0,
            "claim_level_binary_accuracy": 0.0,
            "claim_level_three_way_accuracy": 0.0,
            "human_vs_verifier_correlation": 0.0,
            "overlap_vs_human_correlation": {
                "bertscore_f1_vs_human_claim_support_rate": 0.0,
                "rouge1_vs_human_claim_support_rate": 0.0,
                "rougeL_vs_human_claim_support_rate": 0.0,
            },
            "summary_level_examples": 0,
            "high_overlap_low_human_support_examples": [],
        }

    binary_correct = 0
    three_way_correct = 0
    verifier_support_rates: Dict[str, List[int]] = {}
    example_rows = aggregate_annotations_by_example(filtered_rows)

    for row in filtered_rows:
        human_label = normalize_human_label(str(row["human_label"]))
        verifier_label = normalize_human_label(str(row.get("verifier_predicted_label_name", "")))
        human_supported = int(human_label == "supported")
        verifier_supported = int(verifier_label == "supported")
        if human_supported == verifier_supported:
            binary_correct += 1
        if human_label == verifier_label:
            three_way_correct += 1
        verifier_support_rates.setdefault(str(row["example_id"]), []).append(verifier_supported)

    example_ids = sorted(example_rows.keys())
    predictions = [str(example_rows[example_id]["generated_summary"]) for example_id in example_ids]
    references = [str(example_rows[example_id]["reference_summary"]) for example_id in example_ids]
    overlap_rows = compute_example_level_overlap_metrics(predictions=predictions, references=references)
    human_support_rates = [float(example_rows[example_id]["human_claim_support_rate"]) for example_id in example_ids]
    verifier_support_summary_rates = []
    for example_id in example_ids:
        support_values = verifier_support_rates.get(example_id, [])
        verifier_support_summary_rates.append(
            round(sum(support_values) / len(support_values), 4) if support_values else 0.0
        )
    disagreement_examples = []
    for example_id, overlap in zip(example_ids, overlap_rows):
        support_rate = float(example_rows[example_id]["human_claim_support_rate"])
        if overlap["bertscore_f1"] >= 0.8 and support_rate <= 0.5:
            disagreement_examples.append(
                {
                    "example_id": example_id,
                    "bertscore_f1": overlap["bertscore_f1"],
                    "rouge1": overlap["rouge1"],
                    "rougeL": overlap["rougeL"],
                    "human_claim_support_rate": support_rate,
                    "generated_summary": example_rows[example_id]["generated_summary"],
                    "reference_summary": example_rows[example_id]["reference_summary"],
                }
            )
    disagreement_examples.sort(key=lambda row: (row["human_claim_support_rate"], -row["bertscore_f1"]))

    return {
        "num_annotated_claims": len(filtered_rows),
        "claim_level_binary_accuracy": round(binary_correct / len(filtered_rows), 4),
        "claim_level_three_way_accuracy": round(three_way_correct / len(filtered_rows), 4),
        "human_vs_verifier_correlation": safe_correlation(verifier_support_summary_rates, human_support_rates),
        "overlap_vs_human_correlation": {
            "bertscore_f1_vs_human_claim_support_rate": safe_correlation(
                [float(row["bertscore_f1"]) for row in overlap_rows],
                human_support_rates,
            ),
            "rouge1_vs_human_claim_support_rate": safe_correlation(
                [float(row["rouge1"]) for row in overlap_rows],
                human_support_rates,
            ),
            "rougeL_vs_human_claim_support_rate": safe_correlation(
                [float(row["rougeL"]) for row in overlap_rows],
                human_support_rates,
            ),
        },
        "summary_level_examples": len(example_ids),
        "high_overlap_low_human_support_examples": disagreement_examples[:5],
        "aggregate_overlap_metrics": compute_text_overlap_metrics(predictions=predictions, references=references),
    }


def load_example_reports(path: Path) -> List[Dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "example_reports" in payload:
        return list(payload["example_reports"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported report payload in {path}")
