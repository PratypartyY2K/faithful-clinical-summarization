"""Heuristic claim extraction for synthetic and future clinical summaries."""

from __future__ import annotations

import re
from typing import List


CLAUSE_SPLIT_PATTERN = re.compile(r"\s*;\s*|\s+\band\b\s+(?=[a-z])", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
PREFIX_PATTERNS = (
    re.compile(r"^\s*plan\s+is\s+to\s+", re.IGNORECASE),
    re.compile(r"^\s*plan:\s*", re.IGNORECASE),
    re.compile(r"^\s*assessment:\s*", re.IGNORECASE),
)


def normalize_claim_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    for pattern in PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip(" ,;:")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def split_into_claims(summary: str) -> List[str]:
    claims: List[str] = []
    for sentence in SENTENCE_SPLIT_PATTERN.split(summary.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        fragments = CLAUSE_SPLIT_PATTERN.split(sentence)
        for fragment in fragments:
            claim = normalize_claim_text(fragment)
            if claim and claim not in claims:
                claims.append(claim)
    return claims
