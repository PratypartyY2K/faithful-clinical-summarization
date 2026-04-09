# Faithful Clinical Summarization via Atomic Claim Verification

This repository contains a lightweight end-to-end prototype for faithful clinical summarization. The current codebase is a synthetic stand-in for the planned MIMIC-III pipeline while official data access is pending. The workflow trains:

- a summarizer that generates a note from a patient-clinician dialogue
- a verifier that scores whether each atomic claim in the generated summary is supported by the source dialogue or note

The current implementation is designed for local experimentation with a synthetic dummy dataset, not for direct use on real clinical data.

## Config-Driven Runs

All main scripts support `--config <json-file>`. Presets are provided under `configs/`.

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

## Pipeline Overview

1. Generate synthetic dialogue, summary, and claim-label examples.
2. Convert the raw examples into:
   - a summarization dataset
   - a claim verification dataset
3. Train either a seq2seq baseline or a PEFT/QLoRA causal summarizer.
4. Train a verifier on dialogue-claim pairs with binary or NLI-style labels.
5. Run inference on one dialogue, split the generated summary into atomic claims, and score each claim with the verifier.
6. Evaluate ROUGE, BERTScore, FactScore-style claim support, and qualitative error patterns on the test split.

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
├── src
│   ├── evaluation
│   ├── config
│   ├── modeling
│   └── preprocessing
└── scripts
    ├── create_dummy_dataset.py
    ├── evaluate_pipeline.py
    ├── prepare_datasets.py
    ├── run_pipeline.py
    ├── train_summarizer.py
    └── train_verifier.py
```

## Implemented Scripts

### `scripts/create_dummy_dataset.py`

Creates a synthetic clinical dataset in JSONL format. Each example contains:

- `dialogue`
- `summary`
- `claims`
- `metadata`

By default the synthetic verifier labels follow a 3-way NLI schema:

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

Example:

```bash
python3 scripts/create_dummy_dataset.py --train-size 24 --val-size 6 --test-size 6
```

### `scripts/prepare_datasets.py`

Transforms raw examples into task-specific splits:

- `data/dummy/processed/summarization/*.jsonl`
- `data/dummy/processed/verifier/*.jsonl`

Example:

```bash
python3 scripts/prepare_datasets.py
```

### `scripts/train_summarizer.py`

Trains either:

- a seq2seq baseline such as `google/flan-t5-small`, or
- a causal LM summarizer with optional PEFT/QLoRA, suitable for a Llama-style model

Default output:

- `artifacts/summarizer`

Example:

```bash
python3 scripts/train_summarizer.py --num-train-epochs 1
```

QLoRA-style example:

```bash
python3 scripts/train_summarizer.py \
  --trainer-type causal \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --use-peft \
  --use-qlora
```

### `scripts/train_verifier.py`

Trains a claim verifier on dialogue-claim pairs.

Default model:

- `microsoft/deberta-v3-large`

Default output:

- `artifacts/verifier`

Example:

```bash
python3 scripts/train_verifier.py --num-train-epochs 1
```

### `scripts/run_pipeline.py`

Loads the trained summarizer and verifier, generates a summary for the first example in the input file, decomposes the summary into heuristic atomic claims, scores each claim with batched verifier inference, and writes a JSON report.

Default output:

- `artifacts/pipeline_report.json`

Example:

```bash
python3 scripts/run_pipeline.py
```

### `scripts/evaluate_pipeline.py`

Runs the trained pipeline over the synthetic test set and reports:

- ROUGE on generated summaries
- BERTScore on generated summaries
- FactScore-style average claim support rate
- average claim support rate on generated outputs
- verifier classification metrics on held-out claim examples
- qualitative error analysis for unsupported claims

Default output:

- `artifacts/evaluation_report.json`

Example:

```bash
python3 scripts/evaluate_pipeline.py
```

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate dummy data

```bash
python3 scripts/create_dummy_dataset.py
```

### 3. Prepare training splits

```bash
python3 scripts/prepare_datasets.py
```

### 4. Train the summarizer

```bash
python3 scripts/train_summarizer.py
```

### 5. Train the verifier

```bash
python3 scripts/train_verifier.py
```

### 6. Run the full pipeline

```bash
python3 scripts/run_pipeline.py
```

### 7. Evaluate the full pipeline

```bash
python3 scripts/evaluate_pipeline.py
```

## Expected Outputs

After a full run, the repository will typically contain:

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
  evaluation_report.json
  summarizer/
  verifier/
  pipeline_report.json
```

## Notes

- The dataset in this repository is synthetic and intended for development and demonstration.
- The verifier currently operates on heuristic atomic claims produced by rule-based splitting.
- Training defaults are small so the pipeline is runnable on a local machine, but model downloads still require internet access the first time you run them.
- The `src/` package is structured so the synthetic dataset can later be swapped for a real clinical dataset with minimal changes to downstream training and evaluation scripts.
