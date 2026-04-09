#!/usr/bin/env python3
"""Create a synthetic clinical dialogue dataset for local pipeline development."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from src.config.cli import parse_args_with_optional_config


PATIENT_PROFILES: List[Dict[str, str]] = [
    {
        "condition": "type 2 diabetes",
        "symptom": "fatigue after meals",
        "medication": "metformin",
        "dosage": "500 mg twice daily",
        "lab": "A1c 8.1%",
        "follow_up": "repeat A1c in 3 months",
        "specialty": "primary care",
    },
    {
        "condition": "hypertension",
        "symptom": "morning headaches",
        "medication": "lisinopril",
        "dosage": "10 mg daily",
        "lab": "blood pressure 152/94",
        "follow_up": "home blood pressure log for 2 weeks",
        "specialty": "internal medicine",
    },
    {
        "condition": "asthma",
        "symptom": "wheezing at night",
        "medication": "albuterol inhaler",
        "dosage": "2 puffs as needed",
        "lab": "oxygen saturation 98%",
        "follow_up": "pulmonary follow-up in 1 month",
        "specialty": "pulmonology",
    },
    {
        "condition": "migraine",
        "symptom": "throbbing headache with nausea",
        "medication": "sumatriptan",
        "dosage": "50 mg at onset",
        "lab": "neurologic exam normal",
        "follow_up": "headache diary review in 6 weeks",
        "specialty": "neurology",
    },
    {
        "condition": "urinary tract infection",
        "symptom": "burning with urination",
        "medication": "nitrofurantoin",
        "dosage": "100 mg twice daily for 5 days",
        "lab": "urinalysis positive for leukocytes",
        "follow_up": "return if fever or flank pain develops",
        "specialty": "urgent care",
    },
    {
        "condition": "hyperlipidemia",
        "symptom": "no new symptoms",
        "medication": "atorvastatin",
        "dosage": "20 mg nightly",
        "lab": "LDL 168 mg/dL",
        "follow_up": "repeat lipid panel in 8 weeks",
        "specialty": "cardiology",
    },
]


def build_dialogue(profile: Dict[str, str], encounter_id: int) -> str:
    return "\n".join(
        [
            f"Doctor: This is follow-up visit {encounter_id}. What brings you in today?",
            f"Patient: I have been dealing with {profile['symptom']} and wanted to check on my {profile['condition']}.",
            f"Doctor: I reviewed your chart. Your latest result shows {profile['lab']}.",
            f"Patient: I have been taking {profile['medication']}, but I am not sure it is enough.",
            f"Doctor: We will continue {profile['medication']} at {profile['dosage']}.",
            "Doctor: Please watch for worsening symptoms and keep up with hydration, diet, and rest.",
            f"Doctor: I want you to follow up with {profile['specialty']} and plan to {profile['follow_up']}.",
            "Patient: Understood, I will follow the plan.",
        ]
    )


def build_summary(profile: Dict[str, str]) -> str:
    return (
        f"Patient seen for {profile['condition']} with report of {profile['symptom']}. "
        f"Relevant finding: {profile['lab']}. "
        f"Plan is to continue {profile['medication']} at {profile['dosage']} and {profile['follow_up']}."
    )


def build_claims(profile: Dict[str, str], label_schema: str) -> List[Dict[str, object]]:
    entailed_claims = [
        f"The patient has {profile['condition']}.",
        f"The patient reported {profile['symptom']}.",
        f"The clinician documented {profile['lab']}.",
        f"The plan includes {profile['medication']} at {profile['dosage']}.",
        f"The patient was advised to {profile['follow_up']}.",
    ]
    contradicted_claims = [
        f"The patient denied having {profile['condition']}.",
        f"The clinician stopped {profile['medication']}.",
        f"The patient denied {profile['symptom']}.",
    ]
    neutral_claims = [
        "The patient was admitted to the ICU.",
        "The patient lives alone.",
        "The patient missed work this week.",
    ]

    claims: List[Dict[str, object]] = []
    if label_schema == "binary":
        for claim in entailed_claims:
            claims.append({"claim": claim, "label": 1, "label_name": "supported"})
        for claim in contradicted_claims + neutral_claims:
            claims.append({"claim": claim, "label": 0, "label_name": "unsupported"})
        return claims

    label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
    for claim in entailed_claims:
        claims.append({"claim": claim, "label": label_map["entailment"], "label_name": "entailment"})
    for claim in contradicted_claims:
        claims.append({"claim": claim, "label": label_map["contradiction"], "label_name": "contradiction"})
    for claim in neutral_claims:
        claims.append({"claim": claim, "label": label_map["neutral"], "label_name": "neutral"})
    return claims


def create_example(profile: Dict[str, str], encounter_id: int, label_schema: str) -> Dict[str, object]:
    return {
        "example_id": f"encounter-{encounter_id:04d}",
        "dialogue": build_dialogue(profile, encounter_id),
        "summary": build_summary(profile),
        "claims": build_claims(profile, label_schema=label_schema),
        "metadata": {
            "specialty": profile["specialty"],
            "condition": profile["condition"],
        },
    }


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dummy/raw"))
    parser.add_argument("--train-size", type=int, default=24)
    parser.add_argument("--val-size", type=int, default=6)
    parser.add_argument("--test-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-schema", choices=("binary", "nli"), default="nli")
    args = parse_args_with_optional_config(parser)

    random.seed(args.seed)
    total_size = args.train_size + args.val_size + args.test_size
    profiles = [random.choice(PATIENT_PROFILES) for _ in range(total_size)]
    examples = [
        create_example(profile, index + 1, label_schema=args.label_schema)
        for index, profile in enumerate(profiles)
    ]

    train_end = args.train_size
    val_end = train_end + args.val_size

    write_jsonl(args.output_dir / "train.jsonl", examples[:train_end])
    write_jsonl(args.output_dir / "validation.jsonl", examples[train_end:val_end])
    write_jsonl(args.output_dir / "test.jsonl", examples[val_end:])

    manifest = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "label_schema": args.label_schema,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote dummy dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
