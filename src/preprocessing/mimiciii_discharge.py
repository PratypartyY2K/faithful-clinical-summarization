"""Utilities for converting MIMIC-III discharge summaries into raw examples."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Tuple


INLINE_HEADING_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z /-]{1,80}):\s*(.*)$")
UPPERCASE_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z /-]{1,80}$")
NON_ALPHANUMERIC_RUN_PATTERN = re.compile(r"[_*=-]{3,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
DEIDENTIFIED_SPAN_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")
DICTATION_METADATA_PATTERN = re.compile(
    r"(dictated by:|medquist|job#:|d:\s|\bt:\s|md number|telephone/fax)",
    re.IGNORECASE,
)
ENUMERATED_LIST_PATTERN = re.compile(r"(?:^|\s)\d+\.\s+")
SIGNATURE_ARTIFACT_PATTERN = re.compile(
    r"\bdr\.\s*,?\s*(?:\d{1,3}-\d{1,3})?\b|\bm\.d\.\b",
    re.IGNORECASE,
)

SOURCE_SECTION_CANDIDATES = (
    "chief complaint",
    "history of present illness",
    "past medical history",
    "past surgical history",
    "social history",
    "family history",
    "physical exam",
    "pertinent results",
    "pertinent laboratory data",
    "labs on admission",
    "admission labs",
    "hospital course",
)

TARGET_SECTION_CANDIDATES = (
    "brief hospital course",
    "discharge diagnosis",
    "discharge diagnoses",
    "final diagnoses",
    "discharge condition",
    "discharge disposition",
    "follow-up plans",
    "follow up plans",
    "followup instructions",
)
NARRATIVE_TARGET_SECTION_NAMES = {"brief hospital course"}
SHORT_STATUS_VALUES = {
    "good",
    "fair",
    "stable",
    "improved",
    "guarded",
    "expired",
    "deceased",
    "home",
    "none",
    "n/a",
    "not applicable",
    "extended care",
    "rehabilitation",
    "to rehabilitation",
}


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text.replace("\x00", " ")).strip()


def normalize_section_name(name: str) -> str:
    cleaned = normalize_whitespace(name).strip(":")
    return cleaned.lower()


def strip_deidentified_spans(text: str) -> str:
    return DEIDENTIFIED_SPAN_PATTERN.sub(" ", text)


def strip_trailing_dictation_metadata(text: str) -> str:
    lines: List[str] = []
    for raw_line in text.splitlines():
        if DICTATION_METADATA_PATTERN.search(raw_line):
            break
        lines.append(raw_line)
    return "\n".join(lines)


def clean_note_text(text: str) -> str:
    cleaned = strip_deidentified_spans(text.replace("\x00", " "))
    cleaned = strip_trailing_dictation_metadata(cleaned)
    cleaned = cleaned.replace("..", ".")
    return cleaned


def clean_section_text(text: str) -> str:
    cleaned = SIGNATURE_ARTIFACT_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return normalize_whitespace(cleaned)


def is_heading_line(line: str) -> bool:
    candidate = normalize_whitespace(line)
    if not candidate or len(candidate) > 80:
        return False
    if NON_ALPHANUMERIC_RUN_PATTERN.search(candidate):
        return False
    return bool(UPPERCASE_HEADING_PATTERN.fullmatch(candidate))


def parse_note_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current_section = "preamble"
    sections[current_section] = []

    for raw_line in clean_note_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline_heading = INLINE_HEADING_PATTERN.match(line)
        if inline_heading:
            heading = normalize_section_name(inline_heading.group(1))
            current_section = heading
            sections.setdefault(current_section, [])
            remainder = inline_heading.group(2).strip()
            if remainder:
                sections[current_section].append(remainder)
            continue

        if is_heading_line(line):
            current_section = normalize_section_name(line)
            sections.setdefault(current_section, [])
            continue

        sections.setdefault(current_section, []).append(line)

    return {
        name: clean_section_text(" ".join(lines))
        for name, lines in sections.items()
        if clean_section_text(" ".join(lines))
    }


def is_target_section_usable(name: str, text: str) -> bool:
    normalized_name = normalize_section_name(name)
    normalized_text = normalize_whitespace(text).lower()
    if not normalized_text:
        return False
    if normalized_name in {"discharge medications", "discharge medictions", "medications on discharge", "discharge instructions"}:
        return False
    if normalized_text.startswith("please return to the office"):
        return False
    if ENUMERATED_LIST_PATTERN.search(text) and len(text) > 600:
        return False
    return True


def looks_like_narrative_summary(text: str) -> bool:
    normalized = normalize_whitespace(text).lower().strip(" .")
    if not normalized:
        return False
    if normalized in SHORT_STATUS_VALUES:
        return False
    if "\n\n" not in text and len(normalized.split()) < 12:
        return False
    sentence_count = sum(text.count(mark) for mark in ".!?")
    if sentence_count >= 2:
        return True
    if len(normalized.split()) >= 25 and not ENUMERATED_LIST_PATTERN.search(text):
        return True
    return False


def collect_sections(sections: Dict[str, str], section_names: Iterable[str]) -> Tuple[str, List[str]]:
    collected_names: List[str] = []
    collected_text: List[str] = []
    for name in section_names:
        value = sections.get(name)
        if value:
            collected_names.append(name)
            collected_text.append(value)
    return "\n\n".join(collected_text), collected_names


def collect_target_sections(sections: Dict[str, str], section_names: Iterable[str]) -> Tuple[str, List[str]]:
    preferred_names = [name for name in section_names if normalize_section_name(name) in NARRATIVE_TARGET_SECTION_NAMES]
    preferred_text: List[str] = []
    preferred_sections: List[str] = []
    for name in preferred_names:
        value = sections.get(name)
        if value and is_target_section_usable(name, value) and looks_like_narrative_summary(value):
            preferred_sections.append(name)
            preferred_text.append(value)

    if preferred_text:
        for name in section_names:
            normalized_name = normalize_section_name(name)
            if normalized_name in NARRATIVE_TARGET_SECTION_NAMES:
                continue
            value = sections.get(name)
            if value and is_target_section_usable(name, value) and looks_like_narrative_summary(value):
                preferred_sections.append(name)
                preferred_text.append(value)
        return "\n\n".join(preferred_text), preferred_sections
    return "", []


def stable_split_name(identifier: str, train_fraction: float, validation_fraction: float) -> str:
    digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + validation_fraction:
        return "validation"
    return "test"


def build_raw_example(
    row: Dict[str, str],
    min_source_chars: int = 400,
    min_target_chars: int = 120,
) -> Dict[str, object] | None:
    text = row.get("TEXT", "")
    if not text.strip():
        return None

    sections = parse_note_sections(text)
    source_text, source_sections = collect_sections(sections, SOURCE_SECTION_CANDIDATES)
    target_text, target_sections = collect_target_sections(sections, TARGET_SECTION_CANDIDATES)

    if len(source_text) < min_source_chars or len(target_text) < min_target_chars:
        return None

    subject_id = (row.get("SUBJECT_ID") or "").strip()
    hadm_id = (row.get("HADM_ID") or "").strip()
    chartdate = (row.get("CHARTDATE") or "").strip()
    example_id = f"mimiciii-{hadm_id or subject_id or 'unknown'}"

    return {
        "example_id": example_id,
        "dialogue": source_text,
        "summary": target_text,
        "claims": [],
        "metadata": {
            "dataset": "mimiciii",
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "chartdate": chartdate,
            "category": row.get("CATEGORY", ""),
            "description": row.get("DESCRIPTION", ""),
            "source_sections": source_sections,
            "target_sections": target_sections,
        },
    }
