"""Shared model loading and generate-then-verify inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.modeling.tokenizer_utils import configure_generation_tokenizer
from src.preprocessing.claim_extractor import extract_claims


VerifierBundle = Tuple[object, AutoModelForSequenceClassification]
SummarizerBundle = Tuple[object, object]


def build_summary_prompt(dialogue: str) -> str:
    return (
        "You are a clinical documentation assistant.\n"
        "Write a concise faithful clinical summary grounded only in the source dialogue.\n\n"
        f"Dialogue:\n{dialogue}\n\n"
        "Summary:\n"
    )


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def normalize_label_name(label_name: str) -> str:
    return label_name.strip().lower().replace("_", " ")


def get_label_lookup(model) -> Dict[int, str]:
    id2label = getattr(model.config, "id2label", None) or {}
    if id2label:
        return {int(label_id): normalize_label_name(label_name) for label_id, label_name in id2label.items()}
    num_labels = int(getattr(model.config, "num_labels", 2))
    if num_labels == 3:
        return {0: "contradiction", 1: "neutral", 2: "entailment"}
    return {0: "unsupported", 1: "supported"}


def get_support_label_ids(model) -> List[int]:
    return [
        label_id
        for label_id, label_name in get_label_lookup(model).items()
        if label_name in {"supported", "entailment", "entailed"}
    ]


def load_summarizer(model_dir: Path) -> SummarizerBundle:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(str(model_dir))
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
        except (ValueError, OSError):
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    tokenizer = configure_generation_tokenizer(tokenizer, model)
    return tokenizer, model


def load_verifier(model_dir: Path) -> VerifierBundle:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return tokenizer, model


def generate_summary(
    dialogue: str,
    summarizer_tokenizer,
    summarizer_model,
    max_new_tokens: int = 96,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> str:
    prompt = build_summary_prompt(dialogue)
    device = get_model_device(summarizer_model)
    summarizer_model.eval()
    encoded = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = summarizer_model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
    if getattr(summarizer_model.config, "is_encoder_decoder", False):
        decoded_tokens = generated[0]
    else:
        prompt_length = encoded["input_ids"].shape[-1]
        decoded_tokens = generated[0][prompt_length:]
    return summarizer_tokenizer.decode(decoded_tokens, skip_special_tokens=True).strip()


def generate_summaries_batch(
    dialogues: Iterable[str],
    summarizer_tokenizer,
    summarizer_model,
    max_new_tokens: int = 96,
    batch_size: int = 4,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> List[str]:
    prompts = [build_summary_prompt(dialogue) for dialogue in dialogues]
    outputs: List[str] = []
    device = get_model_device(summarizer_model)
    summarizer_model.eval()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = summarizer_tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = summarizer_model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
        if getattr(summarizer_model.config, "is_encoder_decoder", False):
            decoded_batch = summarizer_tokenizer.batch_decode(generated, skip_special_tokens=True)
        else:
            input_length = encoded["input_ids"].shape[-1]
            decoded_batch = []
            for row in generated:
                decoded_tokens = row[int(input_length) :]
                decoded_batch.append(
                    summarizer_tokenizer.decode(decoded_tokens, skip_special_tokens=True).strip()
                )
        outputs.extend(text.strip() for text in decoded_batch)
    return outputs


def score_claims_batched(
    dialogue: str,
    claims: List[str],
    verifier_tokenizer,
    verifier_model,
    batch_size: int = 8,
) -> List[Dict[str, object]]:
    scores: List[Dict[str, object]] = []
    if not claims:
        return scores
    label_lookup = get_label_lookup(verifier_model)
    support_label_ids = get_support_label_ids(verifier_model)
    device = get_model_device(verifier_model)
    for start in range(0, len(claims), batch_size):
        batch_claims = claims[start : start + batch_size]
        encoded = verifier_tokenizer(
            [dialogue] * len(batch_claims),
            batch_claims,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = verifier_model(**encoded).logits
        batch_probabilities = torch.softmax(logits, dim=-1).cpu()
        batch_predictions = torch.argmax(batch_probabilities, dim=-1).cpu().tolist()
        for claim, probabilities, predicted_label in zip(
            batch_claims,
            batch_probabilities,
            batch_predictions,
        ):
            supported_probability = 0.0
            if support_label_ids:
                supported_probability = float(sum(float(probabilities[label_id]) for label_id in support_label_ids))
            label_probabilities = {
                label_lookup.get(label_id, str(label_id)): round(float(probabilities[label_id]), 4)
                for label_id in range(len(probabilities))
            }
            scores.append(
                {
                    "claim": claim,
                    "supported_probability": round(supported_probability, 4),
                    "predicted_label": int(predicted_label),
                    "predicted_label_name": label_lookup.get(int(predicted_label), str(predicted_label)),
                    "label_probabilities": label_probabilities,
                }
            )
    return scores


def score_claims(
    dialogue: str,
    claims: List[str],
    verifier_tokenizer,
    verifier_model,
    batch_size: int = 8,
) -> List[Dict[str, object]]:
    return score_claims_batched(
        dialogue=dialogue,
        claims=claims,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
        batch_size=batch_size,
    )


def build_pipeline_report(
    example: Dict[str, object],
    summarizer_tokenizer,
    summarizer_model,
    verifier_tokenizer,
    verifier_model,
    max_new_tokens: int = 96,
    verifier_batch_size: int = 8,
    claim_extractor_backend: str = "heuristic",
    claim_extractor_model: str = "gpt-4.1-mini",
    num_beams: int = 1,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> Dict[str, object]:
    summary = generate_summary(
        dialogue=str(example["dialogue"]),
        summarizer_tokenizer=summarizer_tokenizer,
        summarizer_model=summarizer_model,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
    )
    claims = extract_claims(
        summary,
        backend=claim_extractor_backend,
        llm_model=claim_extractor_model,
    )
    claim_scores = score_claims(
        dialogue=str(example["dialogue"]),
        claims=claims,
        verifier_tokenizer=verifier_tokenizer,
        verifier_model=verifier_model,
        batch_size=verifier_batch_size,
    )
    return {
        "example_id": example["example_id"],
        "dialogue": example["dialogue"],
        "reference_summary": example.get("summary"),
        "generated_summary": summary,
        "claim_extractor_backend": claim_extractor_backend,
        "claim_scores": claim_scores,
    }
