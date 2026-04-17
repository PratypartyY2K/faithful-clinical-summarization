# Faithful Clinical Summarization via Atomic Claim Verification

This repository is now set up for the official MIMIC-III workflow. The synthetic bootstrap dataset and its helper code have been removed.

The current codebase supports:
- summarization training with a seq2seq baseline or optional PEFT/QLoRA causal LM path
- claim-level verification with a DeBERTa-style verifier
- heuristic atomic claim extraction, with an optional OpenAI-backed LLM extractor
- evaluation with ROUGE, BERTScore, a FactScore-style support rate, and qualitative error summaries

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

Each `claims` entry should contain:
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

1. Convert raw MIMIC-III-derived examples into summarization and verifier splits.
2. Train either a seq2seq baseline or a PEFT/QLoRA causal summarizer.
3. Train a verifier on dialogue-claim pairs with binary or NLI-style labels.
4. Generate a summary, decompose it into atomic claims, and score each claim against the source.
5. Evaluate ROUGE, BERTScore, FactScore-style support, label breakdowns, and qualitative error patterns.

## Core Scripts

`scripts/prepare_datasets.py`
Transforms raw clinical examples into task-specific splits under `data/mimiciii/processed/`.

`scripts/train_summarizer.py`
Trains a summarizer from `data/mimiciii/processed/summarization/`.

`scripts/train_verifier.py`
Trains a claim verifier from `data/mimiciii/processed/verifier/`.

`scripts/run_pipeline.py`
Loads trained models and runs generate-and-verify inference on one input example.

`scripts/evaluate_pipeline.py`
Runs the trained pipeline over the test split and writes aggregate evaluation reports.

## Config-Driven Runs

All main scripts support `--config <json-file>`. Presets are stored under `configs/`.

Prepare processed splits:

```bash
python3 scripts/prepare_datasets.py
```

Train the `flan-t5-small` baseline:

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
```

Train the DeBERTa verifier:

```bash
python3 scripts/train_verifier.py --config configs/verifier/deberta_v3_large.json
```

Run end-to-end evaluation:

```bash
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline.json
```

Run evaluation with LLM-based claim extraction:

```bash
export OPENAI_API_KEY=...
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline_llm_claims.json
```

Additional preset notes are in [configs/README.md](configs/README.md).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/prepare_datasets.py
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
python3 scripts/train_verifier.py --config configs/verifier/deberta_v3_large.json
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline.json
```

## Testing

Run the lightweight test suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

These tests cover:
- config loading
- claim extraction behavior
- evaluation summary helpers

## Expected Outputs

Depending on the configured paths, artifact directories may include:
- `generation_metrics.json`
- `verifier_metrics.json`
- `run_metadata.json`
- `evaluation_report.json`
- `evaluation_run_metadata.json`

## Limitations

- The current atomic claim extractor is heuristic, not parser-based or model-based.
- An optional OpenAI-backed claim extractor is available, but it requires API access and adds runtime cost.
- The FactScore-style metric here is an engineering proxy built from claim support predictions, not a full reproduction of the original FActScore framework.
- Model downloads require internet access the first time they are used.
- The Llama-3 QLoRA path assumes a CUDA-capable environment.
