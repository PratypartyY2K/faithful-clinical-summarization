---
base_model: mistralai/Mistral-7B-Instruct-v0.2
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
- clinical-summarization
---

# Mistral 7B QLoRA Smoke Run

This directory contains a PEFT adapter produced from a smoke-test fine-tuning run for the MIMIC-III discharge-summary summarization pipeline.

## What This Artifact Is

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Fine-tuning method: LoRA / QLoRA
- Task: clinical discharge-summary generation from MIMIC-III-derived source sections
- Intended use: smoke-test and debugging artifact, not a final project model release

## Repository Context

The current project workflow is:

1. ingest MIMIC-III discharge summaries into `data/mimiciii/raw/*.jsonl`
2. prepare processed summarization splits under `data/mimiciii/processed/summarization/*.jsonl`
3. train a summarizer with `scripts/train_summarizer.py`
4. evaluate with `scripts/evaluate_summarizer.py`

This adapter was created during the Mistral QLoRA migration after earlier seq2seq baselines were found to be inadequate for the current task formulation.

It is useful for:
- confirming that the causal QLoRA path works end to end
- quick regression checks after preprocessing or generation changes
- verifying that evaluation still runs before launching a full-data training job

It is not the current best model for reporting final project results.

## Files

- `adapter_model.safetensors`: LoRA adapter weights
- `adapter_config.json`: PEFT adapter config
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: tokenizer assets
- `checkpoint-817/`: trainer checkpoint snapshot from the smoke run

## Notes

- This is an intermediate experiment artifact.
- The smoke run was mainly a pipeline-validation checkpoint, not a quality-optimized model.
- The model card metadata here is intentionally brief and local to this repository.
- For the current project status and recommended commands, see the top-level [README.md](../../../README.md).
