"""Shared JSONL helpers and dataset transforms."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from src.preprocessing.verifier_dataset import build_verifier_rows_from_examples


SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
STRUCTURED_MARKER_PATTERN = re.compile(r"(?:(?<=\s)|^)(?:#|\d+\)|\d+\.)")


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_first_jsonl_row(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON rows found in {path}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def take_first_sentences(text: str, max_sentences: int | None = None) -> str:
    normalized = text.strip()
    if max_sentences is None or max_sentences <= 0 or not normalized:
        return normalized
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_PATTERN.split(normalized) if sentence.strip()]
    if len(sentences) <= max_sentences:
        return normalized
    return " ".join(sentences[:max_sentences]).strip()


def count_words(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def count_sentences(text: str) -> int:
    return len([sentence for sentence in SENTENCE_BOUNDARY_PATTERN.split(text.strip()) if sentence.strip()])


def count_structured_markers(text: str) -> int:
    return len(STRUCTURED_MARKER_PATTERN.findall(text))


def keep_narrative_target(
    text: str,
    narrative_only: bool = False,
    min_target_words: int | None = None,
    max_target_words: int | None = None,
    min_target_sentences: int | None = None,
    max_structured_markers: int | None = None,
) -> bool:
    if not narrative_only:
        return True
    target_words = count_words(text)
    target_sentences = count_sentences(text)
    structured_markers = count_structured_markers(text)
    if min_target_words is not None and target_words < min_target_words:
        return False
    if max_target_words is not None and target_words > max_target_words:
        return False
    if min_target_sentences is not None and target_sentences < min_target_sentences:
        return False
    if max_structured_markers is not None and structured_markers > max_structured_markers:
        return False
    return True


def build_summarization_rows(
    examples: List[Dict[str, object]],
    target_sentence_limit: int | None = None,
    narrative_only: bool = False,
    min_target_words: int | None = None,
    max_target_words: int | None = None,
    min_target_sentences: int | None = None,
    max_structured_markers: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for example in examples:
        full_target_text = str(example["summary"])
        if not keep_narrative_target(
            full_target_text,
            narrative_only=narrative_only,
            min_target_words=min_target_words,
            max_target_words=max_target_words,
            min_target_sentences=min_target_sentences,
            max_structured_markers=max_structured_markers,
        ):
            continue
        target_text = take_first_sentences(full_target_text, max_sentences=target_sentence_limit)
        rows.append(
            {
                "example_id": example["example_id"],
                "input_text": example["dialogue"],
                "target_text": target_text,
                "target_text_full": full_target_text,
                "target_sentence_limit": target_sentence_limit,
                "narrative_only": narrative_only,
            }
        )
    return rows


def build_verifier_rows(
    examples: List[Dict[str, object]],
    claim_extractor_backend: str = "heuristic",
    claim_extractor_model: str = "gpt-4.1-mini",
    max_claims_per_example: int | None = None,
    seed: int = 13,
) -> List[Dict[str, object]]:
    return build_verifier_rows_from_examples(
        examples,
        claim_extractor_backend=claim_extractor_backend,
        claim_extractor_model=claim_extractor_model,
        max_claims_per_example=max_claims_per_example,
        seed=seed,
    )


def process_dataset_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    target_sentence_limit: int | None = None,
    narrative_only: bool = False,
    min_target_words: int | None = None,
    max_target_words: int | None = None,
    min_target_sentences: int | None = None,
    max_structured_markers: int | None = None,
    verifier_claim_extractor_backend: str = "heuristic",
    verifier_claim_extractor_model: str = "gpt-4.1-mini",
    verifier_max_claims_per_example: int | None = None,
    verifier_seed: int = 13,
) -> None:
    examples = read_jsonl(input_dir / f"{split}.jsonl")
    write_jsonl(
        output_dir / "summarization" / f"{split}.jsonl",
        build_summarization_rows(
            examples,
            target_sentence_limit=target_sentence_limit,
            narrative_only=narrative_only,
            min_target_words=min_target_words,
            max_target_words=max_target_words,
            min_target_sentences=min_target_sentences,
            max_structured_markers=max_structured_markers,
        ),
    )
    write_jsonl(
        output_dir / "verifier" / f"{split}.jsonl",
        build_verifier_rows(
            examples,
            claim_extractor_backend=verifier_claim_extractor_backend,
            claim_extractor_model=verifier_claim_extractor_model,
            max_claims_per_example=verifier_max_claims_per_example,
            seed=verifier_seed,
        ),
    )
