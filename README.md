# Faithful Clinical Summarization via Atomic Claim Verification

This repository contains a synthetic end-to-end prototype for faithful clinical summarization. It is designed to mirror the planned project pipeline while official MIMIC-III access is still pending.

The current system supports:
- summarization training with a small seq2seq baseline or an optional PEFT/QLoRA causal LM path
- claim-level verification with a DeBERTa-style verifier
- heuristic atomic claim extraction, with an optional OpenAI-backed LLM extractor
- evaluation with ROUGE, BERTScore, a FactScore-style support rate, and qualitative error summaries

## Project Status

What is implemented:
- synthetic dataset generation with binary or 3-way NLI labels
- dataset preparation into summarization and verifier splits
- `flan-t5-small` baseline training path
- optional Llama-style PEFT/QLoRA training path
- `microsoft/deberta-v3-large` verifier training path
- batched verifier inference
- config-driven experiment presets
- run metadata output for training, inference, and evaluation
- lightweight unit tests for core helpers

What is not implemented yet:
- real MIMIC-III ingestion and preprocessing
- real-data experiments and final result tables
- human evaluation
- model-based atomic fact extraction beyond the current heuristic splitter

## Pipeline Overview

1. Generate synthetic dialogue, summary, and claim-label examples.
2. Convert raw examples into:
   - a summarization dataset
   - a claim verification dataset
3. Train either a seq2seq baseline or a PEFT/QLoRA causal summarizer.
4. Train a verifier on dialogue-claim pairs with binary or NLI-style labels.
5. Generate a summary, decompose it into atomic claims, and score each claim against the source.
6. Evaluate ROUGE, BERTScore, FactScore-style support, label breakdowns, and qualitative error patterns.

## Repository Layout

```text
.
├── README.md
├── configs
│   ├── data
│   ├── evaluation
│   ├── summarizer
│   └── verifier
├── requirements.txt
├── scripts
│   ├── create_dummy_dataset.py
│   ├── evaluate_pipeline.py
│   ├── prepare_datasets.py
│   ├── run_pipeline.py
│   ├── train_summarizer.py
│   └── train_verifier.py
├── src
│   ├── config
│   ├── evaluation
│   ├── modeling
│   ├── preprocessing
│   └── utils
└── tests
```

## Config-Driven Runs

All main scripts support `--config <json-file>`. Presets are stored under `configs/`.

Synthetic data setup:

```bash
python3 scripts/create_dummy_dataset.py --config configs/data/dummy_nli.json
python3 scripts/prepare_datasets.py
```

`flan-t5-small` baseline:

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
```

Llama-3-8B QLoRA run:

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/llama3_8b_qlora.json
```

DeBERTa-v3-large verifier:

```bash
python3 scripts/train_verifier.py --config configs/verifier/deberta_v3_large.json
```

Full evaluation run:

```bash
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline.json
```

Full evaluation with LLM claim extraction:

```bash
export OPENAI_API_KEY=...
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline_llm_claims.json
```

Additional preset notes are in [configs/README.md](configs/README.md).

## Core Scripts

### `scripts/create_dummy_dataset.py`

Creates a synthetic clinical dataset in JSONL format. Each example contains:
- `dialogue`
- `summary`
- `claims`
- `metadata`

By default the verifier labels follow a 3-way NLI schema:
- `contradiction`
- `neutral`
- `entailment`

Default output:

```text
data/dummy/raw/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
```

### `scripts/prepare_datasets.py`

Transforms raw examples into task-specific splits:
- `data/dummy/processed/summarization/*.jsonl`
- `data/dummy/processed/verifier/*.jsonl`

### `scripts/train_summarizer.py`

Trains either:
- a seq2seq baseline such as `google/flan-t5-small`
- a causal LM summarizer with optional PEFT/QLoRA, suitable for Llama-style models

Outputs:
- model artifacts under the configured summarizer directory
- `generation_metrics.json`
- `run_metadata.json`

### `scripts/train_verifier.py`

Trains a claim verifier on dialogue-claim pairs.

Default backbone:
- `microsoft/deberta-v3-large`

Outputs:
- model artifacts under the configured verifier directory
- `verifier_metrics.json`
- `run_metadata.json`

### `scripts/run_pipeline.py`

Loads trained models, generates a summary for one example, decomposes it into heuristic atomic claims, scores each claim with batched verifier inference, and writes a report.

Outputs:
- `artifacts/pipeline_report.json` by default

### `scripts/evaluate_pipeline.py`

Runs the trained pipeline over the synthetic test set and reports:
- ROUGE
- BERTScore
- FactScore-style average support rate
- contradiction and neutral/unsupported rates
- verifier classification metrics
- qualitative error analysis
- an aggregate summary block

Outputs:
- `evaluation_report.json`
- `evaluation_run_metadata.json`

Claim extraction can use either:
- the default heuristic backend
- an OpenAI-backed LLM backend selected with `--claim-extractor-backend llm`

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate dummy data and processed splits

```bash
python3 scripts/create_dummy_dataset.py --config configs/data/dummy_nli.json
python3 scripts/prepare_datasets.py
```

### 3. Train a summarizer

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
```

### 4. Train a verifier

```bash
python3 scripts/train_verifier.py --config configs/verifier/deberta_v3_large.json
```

### 5. Run evaluation

```bash
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline.json
```

## Testing

Run the lightweight test suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

These tests cover:
- config loading
- dummy dataset label schema
- claim extraction behavior
- evaluation summary helpers

## Expected Outputs

After a full synthetic run, the repository will typically contain:

```text
data/
  dummy/
    raw/
      train.jsonl
      validation.jsonl
      test.jsonl
      manifest.json
    processed/
      summarization/
        train.jsonl
        validation.jsonl
        test.jsonl
      verifier/
        train.jsonl
        validation.jsonl
        test.jsonl

artifacts/
  evaluation/
  pipeline_report.json
  summarizer/
  verifier/
```

Depending on the configured paths, artifact directories may also include:
- `generation_metrics.json`
- `verifier_metrics.json`
- `run_metadata.json`
- `evaluation_run_metadata.json`

## Limitations

- The dataset is synthetic and intended for development only.
- The current atomic claim extractor is heuristic, not parser-based or model-based.
- An optional OpenAI-backed claim extractor is available, but it requires API access and adds runtime cost.
- The FactScore-style metric here is an engineering proxy built from claim support predictions, not a full reproduction of the original FActScore framework.
- Model downloads require internet access the first time they are used.
- The Llama-3 QLoRA path assumes a CUDA-capable environment.

## Transition To Real Data

The codebase is structured so the synthetic dataset can later be replaced with a real clinical dataset with minimal disruption to:
- training scripts
- verifier logic
- evaluation scripts
- config presets

The main missing step is building the real MIMIC ingestion and preprocessing layer once access is available.
