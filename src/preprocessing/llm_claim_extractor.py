"""Optional LLM-backed medical fact extraction."""

from __future__ import annotations

import json
import os
from typing import List, Sequence


SYSTEM_PROMPT = """You extract independent medical facts from clinical text.

Return JSON only.
Output schema:
{
  "claims": ["fact 1", "fact 2"]
}

Rules:
- Extract only facts explicitly stated in the input.
- Split combined statements into independent facts whenever possible.
- Preserve medical meaning and key details such as medication names, dosages, frequencies, diagnoses, labs, symptoms, and follow-up instructions.
- Do not infer missing facts.
- Do not include duplicates.
- Keep each claim short and self-contained.
"""


def build_user_prompt(text: str) -> str:
    return (
        "Extract all independent medical facts from this text into JSON.\n\n"
        f"Text:\n{text}\n"
    )


def normalize_llm_claims(claims: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for claim in claims:
        cleaned = " ".join(str(claim).strip().split())
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned += "."
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def extract_claims_with_openai(
    text: str,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
) -> List[str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The OpenAI Python package is required for the llm claim extractor. "
            "Install dependencies from requirements.txt."
        ) from exc

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("Expected a JSON object with a 'claims' list.")
    return normalize_llm_claims(claims)
