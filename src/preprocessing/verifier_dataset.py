"""Weakly supervised verifier dataset generation for clinical claims."""

from __future__ import annotations

import random
import re
from typing import Dict, List, Sequence

from src.preprocessing.claim_extractor import extract_claims


LABEL_NAME_TO_ID = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
}

NEGATION_PATTERNS = (
    (re.compile(r"\bdenies\b", re.IGNORECASE), "reports"),
    (re.compile(r"\breports\b", re.IGNORECASE), "denies"),
    (re.compile(r"\bwithout\b", re.IGNORECASE), "with"),
    (re.compile(r"\bwith\b", re.IGNORECASE), "without"),
    (re.compile(r"\bno\b", re.IGNORECASE), "possible"),
    (re.compile(r"\bincreased\b", re.IGNORECASE), "decreased"),
    (re.compile(r"\bdecreased\b", re.IGNORECASE), "increased"),
    (re.compile(r"\bimproved\b", re.IGNORECASE), "worsened"),
    (re.compile(r"\bworsened\b", re.IGNORECASE), "improved"),
    (re.compile(r"\bcontinued\b", re.IGNORECASE), "stopped"),
    (re.compile(r"\bstopped\b", re.IGNORECASE), "continued"),
    (re.compile(r"\bcontinue\b", re.IGNORECASE), "stop"),
    (re.compile(r"\bstop\b", re.IGNORECASE), "continue"),
    (re.compile(r"\bpositive\b", re.IGNORECASE), "negative"),
    (re.compile(r"\bnegative\b", re.IGNORECASE), "positive"),
    (re.compile(r"\bnormal\b", re.IGNORECASE), "abnormal"),
    (re.compile(r"\babnormal\b", re.IGNORECASE), "normal"),
    (re.compile(r"\bleft\b", re.IGNORECASE), "right"),
    (re.compile(r"\bright\b", re.IGNORECASE), "left"),
    (re.compile(r"\badmitted\b", re.IGNORECASE), "discharged"),
    (re.compile(r"\bdischarged\b", re.IGNORECASE), "admitted"),
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def normalize_claim(claim: str) -> str:
    cleaned = " ".join(claim.strip().split())
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def generate_contradiction_claim(claim: str) -> str:
    normalized = normalize_claim(claim)
    for pattern, replacement in NEGATION_PATTERNS:
        if pattern.search(normalized):
            return normalize_claim(pattern.sub(replacement, normalized, count=1))
    match = NUMBER_PATTERN.search(normalized)
    if match:
        original = match.group(0)
        if "." in original:
            replacement = str(round(float(original) * 2, 2)).rstrip("0").rstrip(".")
        else:
            replacement = str(int(original) + 1)
        return normalize_claim(normalized[: match.start()] + replacement + normalized[match.end() :])
    lowered = normalized[:-1].lower() if normalized.endswith((".", "!", "?")) else normalized.lower()
    return normalize_claim(f"It is false that {lowered}")


def build_claim_row(
    *,
    example: Dict[str, object],
    claim: str,
    label_name: str,
    claim_source: str,
) -> Dict[str, object]:
    return {
        "example_id": example["example_id"],
        "dialogue": example["dialogue"],
        "claim": normalize_claim(claim),
        "label": LABEL_NAME_TO_ID[label_name],
        "label_name": label_name,
        "claim_source": claim_source,
    }


def build_verifier_rows_from_examples(
    examples: Sequence[Dict[str, object]],
    claim_extractor_backend: str = "heuristic",
    claim_extractor_model: str = "gpt-4.1-mini",
    max_claims_per_example: int | None = None,
    seed: int = 13,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    example_claims: List[List[str]] = []

    for example in examples:
        claims = extract_claims(
            str(example["summary"]),
            backend=claim_extractor_backend,
            llm_model=claim_extractor_model,
        )
        if max_claims_per_example is not None and max_claims_per_example > 0:
            claims = claims[:max_claims_per_example]
        example_claims.append(claims)

    for index, example in enumerate(examples):
        claims = example_claims[index]
        if not claims:
            continue
        for claim in claims:
            rows.append(
                build_claim_row(
                    example=example,
                    claim=claim,
                    label_name="entailment",
                    claim_source="reference_summary",
                )
            )
            rows.append(
                build_claim_row(
                    example=example,
                    claim=generate_contradiction_claim(claim),
                    label_name="contradiction",
                    claim_source="synthetic_contradiction",
                )
            )

        neutral_candidates = [
            candidate
            for other_index, other_claims in enumerate(example_claims)
            if other_index != index
            for candidate in other_claims
        ]
        if not neutral_candidates:
            continue
        neutral_sample_size = min(len(claims), len(neutral_candidates))
        for neutral_claim in rng.sample(neutral_candidates, k=neutral_sample_size):
            rows.append(
                build_claim_row(
                    example=example,
                    claim=neutral_claim,
                    label_name="neutral",
                    claim_source="cross_example_reference",
                )
            )
    return rows
