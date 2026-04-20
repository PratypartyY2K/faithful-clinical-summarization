# Faithful Clinical Summarization via Atomic Claim Verification

This repository is now set up for the official MIMIC-III workflow. The synthetic bootstrap dataset and its helper code have been removed.

The current codebase supports:
- summarization training with a seq2seq baseline or optional PEFT/QLoRA causal LM path
- MIMIC-III discharge-summary ingestion from `NOTEEVENTS.csv`
- summarizer-only evaluation with ROUGE and BERTScore
- optional short-target debug preprocessing for overfit checks
- claim-level verification code paths and pipeline evaluation scaffolding for future use
- heuristic atomic claim extraction, with an optional OpenAI-backed LLM extractor
- evaluation helpers for ROUGE, BERTScore, FactScore-style support, and qualitative error summaries

## Current Status

The repository is currently a summarizer-first MIMIC-III pipeline.

What works now:
- ingesting MIMIC-III discharge summaries into `data/mimiciii/raw/*.jsonl`
- preparing summarization splits under `data/mimiciii/processed/summarization/*.jsonl`
- training summarization models
- running summarizer-only evaluation with overlap metrics

What is not active yet on the current MIMIC pipeline:
- claim-label generation for verifier training
- non-empty verifier splits
- meaningful end-to-end generate-and-verify evaluation on MIMIC examples

The ingestion step currently writes `claims: []` for each raw example. As a result, processed verifier splits are empty by design until a claim-labeling pipeline is added.

## Expected Data Layout

The preprocessing and training scripts assume data lives under `data/mimiciii/`.

Raw split files consumed by `scripts/prepare_datasets.py`:

```text
data/mimiciii/raw/
  train.jsonl
  validation.jsonl
  test.jsonl
```

Each raw example is expected to contain:
- `example_id`
- `dialogue`
- `summary`
- `claims`

For the current MIMIC ingestion path, `claims` is intentionally an empty list.

When a future claim-labeling stage is added, each `claims` entry should contain:
- `claim`
- `label`
- optional `label_name`

Processed outputs written by `scripts/prepare_datasets.py`:

```text
data/mimiciii/processed/
  summarization/
    train.jsonl
    validation.jsonl
    test.jsonl
  verifier/
    train.jsonl
    validation.jsonl
    test.jsonl
```

## Pipeline Overview

1. Ingest MIMIC-III discharge summaries from `NOTEEVENTS.csv` into raw JSONL splits.
2. Convert raw examples into processed summarization splits.
3. Train either a seq2seq baseline or a PEFT/QLoRA causal summarizer.
4. Evaluate summarizer generations with ROUGE and BERTScore.
5. Add claim-labeling and verifier training later to enable full faithfulness evaluation.

## Core Scripts

`scripts/prepare_datasets.py`
Transforms raw clinical examples into task-specific splits under `data/mimiciii/processed/`.

`scripts/ingest_mimiciii_notes.py`
Reads `NOTEEVENTS.csv`, extracts structured discharge-summary source/target pairs, and writes `data/mimiciii/raw/`.

`scripts/train_summarizer.py`
Trains a summarizer from `data/mimiciii/processed/summarization/`.

`scripts/evaluate_summarizer.py`
Runs summarizer-only generation and overlap evaluation without requiring a verifier.
It accepts either raw examples with `dialogue`/`summary` fields or processed summarization rows with `input_text`/`target_text`.

`scripts/train_verifier.py`
Trains a claim verifier from `data/mimiciii/processed/verifier/`.
This is not currently usable on the MIMIC pipeline because verifier splits are empty until claim labels are added.

`scripts/run_pipeline.py`
Loads trained models and runs generate-and-verify inference on one input example.
This remains scaffolding for the future verifier-enabled pipeline.

`scripts/evaluate_pipeline.py`
Runs the trained pipeline over the test split and writes aggregate evaluation reports.
This remains scaffolding for the future verifier-enabled pipeline.

## Config-Driven Runs

All main scripts support `--config <json-file>`. Presets are stored under `configs/`.

Prepare processed splits:

```bash
python3 scripts/ingest_mimiciii_notes.py
python3 scripts/prepare_datasets.py
```

Train the `flan-t5-small` baseline:

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
```

Train the `flan-t5-base` baseline:

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_base.json
```

Train the longer-source `flan-t5-base` baseline:

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_base_long_source.json
```

Run summarizer-only evaluation on raw test examples:

```bash
python3 scripts/evaluate_summarizer.py --summarizer-dir artifacts/summarizer/flan_t5_base_long_source
```

Prepare a short-target debug dataset for overfit checks:

```bash
python3 scripts/prepare_datasets.py --target-sentence-limit 2
```

Verifier and full pipeline commands remain available in the codebase, but they are not currently active on MIMIC data because the verifier splits are empty. When a claim-labeling stage exists, the verifier flow will look like:

```bash
python3 scripts/train_verifier.py --config configs/verifier/deberta_v3_large.json
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline.json
```

Additional preset notes are in [configs/README.md](configs/README.md).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/ingest_mimiciii_notes.py
python3 scripts/prepare_datasets.py
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_base_long_source.json
python3 scripts/evaluate_summarizer.py --summarizer-dir artifacts/summarizer/flan_t5_base_long_source
```

## Testing

Run the lightweight test suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

These tests cover:
- config loading
- claim extraction behavior
- summarizer evaluation schema handling
- evaluation summary helpers
- MIMIC-III discharge-summary ingestion
- preprocessing helpers for debug target shortening

## Expected Outputs

Depending on the configured paths, artifact directories may include:
- `generation_metrics.json`
- `verifier_metrics.json`
- `run_metadata.json`
- `evaluation_report.json`
- `evaluation_run_metadata.json`

The ingestion step currently creates summarization-ready examples with `claims: []`. That is sufficient for summarizer training. Verifier training still needs a separate claim-labeling step.

For the initial MIMIC-III summarizer, the ingestion defaults intentionally exclude discharge medication and instruction sections from the target summary because they introduce long boilerplate lists that degraded generation quality.

For overfit/debug runs, `scripts/prepare_datasets.py --target-sentence-limit N` can create shortened summarization targets while preserving the full original target in `target_text_full`.

## Limitations

- The current atomic claim extractor is heuristic, not parser-based or model-based.
- An optional OpenAI-backed claim extractor is available, but it requires API access and adds runtime cost.
- The FactScore-style metric here is an engineering proxy built from claim support predictions, not a full reproduction of the original FActScore framework.
- Model downloads require internet access the first time they are used.
- The Llama-3 QLoRA path assumes a CUDA-capable environment.
- The current verifier and end-to-end faithfulness evaluation path are not yet active on MIMIC-III because claim annotations are not generated during ingestion.
