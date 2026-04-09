"""Heuristic claim extraction for synthetic and future clinical summaries."""

from __future__ import annotations

import re
from typing import List


CLAUSE_SPLIT_PATTERN = re.compile(r"\s*;\s*|\s+\band\b\s+(?=(?:to|with|repeat|return|continue|follow)\b)", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
PREFIX_PATTERNS = (
    re.compile(r"^\s*plan\s+is\s+to\s+", re.IGNORECASE),
    re.compile(r"^\s*plan:\s*", re.IGNORECASE),
    re.compile(r"^\s*assessment:\s*", re.IGNORECASE),
)
LEADING_ARTICLE_PATTERN = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def split_sentence_on_colon(sentence: str) -> List[str]:
    if ":" not in sentence:
        return [sentence]
    prefix, suffix = sentence.split(":", 1)
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix or not suffix:
        return [sentence]
    lowered_prefix = prefix.lower()
    if lowered_prefix in {"relevant finding", "finding", "assessment", "impression"}:
        return [f"{prefix}: {suffix}"]
    if lowered_prefix == "plan":
        return [suffix]
    return [sentence]


def rewrite_sentence(sentence: str) -> List[str]:
    stripped = sentence.strip()
    lowered = stripped.lower()
    if " with report of " in lowered:
        prefix, suffix = re.split(r"\s+with\s+report\s+of\s+", stripped, maxsplit=1, flags=re.IGNORECASE)
        return [prefix.strip(), f"Reported {suffix.strip()}"]
    if lowered.startswith("plan is to continue ") and " and " in lowered:
        prefix, suffix = re.split(r"\s+\band\b\s+", stripped, maxsplit=1, flags=re.IGNORECASE)
        return [prefix.strip(), suffix.strip()]
    return [stripped]


def normalize_claim_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    for pattern in PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"^\s*report of\s+", "Reported ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*continue\s+", "Continue ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*repeat\s+", "Repeat ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*return if\s+", "Return if ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*follow up\s+", "Follow up ", cleaned, flags=re.IGNORECASE)
    if ":" in cleaned:
        left, right = cleaned.split(":", 1)
        cleaned = f"{left.strip()}: {right.strip()}"
    cleaned = cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned
    cleaned = cleaned.strip(" ,;:")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def split_into_claims(summary: str) -> List[str]:
    claims: List[str] = []
    seen_normalized = set()
    for sentence in SENTENCE_SPLIT_PATTERN.split(summary.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        for rewritten in rewrite_sentence(sentence):
            for colon_split in split_sentence_on_colon(rewritten):
                fragments = CLAUSE_SPLIT_PATTERN.split(colon_split)
                for fragment in fragments:
                    claim = normalize_claim_text(fragment)
                    normalized_key = LEADING_ARTICLE_PATTERN.sub("", claim).lower()
                    if claim and normalized_key not in seen_normalized:
                        seen_normalized.add(normalized_key)
                        claims.append(claim)
    return claims


def extract_claims(
    summary: str,
    backend: str = "heuristic",
    llm_model: str = "gpt-4.1-mini",
) -> List[str]:
    if backend == "heuristic":
        return split_into_claims(summary)
    if backend == "llm":
        from src.preprocessing.llm_claim_extractor import extract_claims_with_openai

        return extract_claims_with_openai(summary, model=llm_model)
    raise ValueError(f"Unsupported claim extraction backend: {backend}")
