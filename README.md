# Faithful Clinical Summarization via Atomic Claim Verification

This repository is currently a MIMIC-III summarization-first pipeline. The practical path that works today is:

1. ingest discharge summaries from `NOTEEVENTS.csv`
2. prepare summarization splits
3. train a Mistral QLoRA summarizer
4. evaluate with ROUGE and BERTScore
5. iterate on data quality and decoding

The older verifier and end-to-end faithfulness scaffolding still exists, but it is not the active workflow yet because the MIMIC ingestion path does not currently generate claim labels.

## Current Status

What works now:
- MIMIC-III discharge-summary ingestion into `data/mimiciii/raw/*.jsonl`
- processed summarization splits under `data/mimiciii/processed/summarization/*.jsonl`
- PEFT / QLoRA causal summarizer training
- summarizer-only evaluation with ROUGE and BERTScore
- debug preprocessing with shortened targets
- filtered preprocessing for cleaner narrative-only hospital-course targets

What is not active yet:
- claim-label generation during ingestion
- non-empty verifier splits
- meaningful verifier training on MIMIC examples
- full generate-and-verify pipeline evaluation on MIMIC data

The current full-data Mistral QLoRA run is a usable baseline. It produces coherent clinical summaries, but the main quality problem is still target heterogeneity: some references are concise narratives, others are problem lists, and others are long hospital-course chronologies. The current recommended next experiment is to retrain on a cleaner narrative-only subset.

## Expected Data Layout

Raw data is expected under `data/mimiciii/`.

```text
data/mimiciii/
  NOTEEVENTS.csv
  raw/
    train.jsonl
    validation.jsonl
    test.jsonl
  processed/
    summarization/
      train.jsonl
      validation.jsonl
      test.jsonl
    verifier/
      train.jsonl
      validation.jsonl
      test.jsonl
```

Each raw example contains:
- `example_id`
- `dialogue`
- `summary`
- `claims`

For the current MIMIC ingestion path, `claims` is intentionally `[]`.

Processed summarization rows contain:
- `example_id`
- `input_text`
- `target_text`
- optional `target_text_full`
- optional preprocessing metadata such as `target_sentence_limit` and `narrative_only`

## Core Scripts

`scripts/ingest_mimiciii_notes.py`
Reads `NOTEEVENTS.csv`, extracts source sections and target `brief hospital course` text, and writes raw JSONL splits.

`scripts/prepare_datasets.py`
Builds summarization and verifier splits from raw examples. It also supports:
- `--target-sentence-limit` for shortened-target debug datasets
- `--narrative-only` and related target-shape filters for cleaner narrative subsets

`scripts/train_summarizer.py`
Trains either a seq2seq baseline or a causal summarizer. The recommended path is a causal Mistral QLoRA run with checkpoint saving and auto-resume enabled.

`scripts/evaluate_summarizer.py`
Runs summarizer-only generation and overlap evaluation. It accepts:
- raw examples with `dialogue` / `summary`
- processed summarization rows with `input_text` / `target_text`

It also supports decoding controls such as:
- `--num-beams`
- `--no-repeat-ngram-size`
- `--repetition-penalty`
- `--length-penalty`

`scripts/train_verifier.py`, `scripts/run_pipeline.py`, `scripts/evaluate_pipeline.py`
These remain future-facing for the verifier-enabled pipeline. They are not currently useful on MIMIC without claim labels.

## Recommended Workflow

### 1. Ingest MIMIC-III

```bash
python3 scripts/ingest_mimiciii_notes.py \
  --input-file data/mimiciii/NOTEEVENTS.csv \
  --output-dir data/mimiciii/raw
```

### 2. Prepare the default processed dataset

```bash
python3 scripts/prepare_datasets.py \
  --input-dir data/mimiciii/raw \
  --output-dir data/mimiciii/processed
```

### 3. Train the baseline Mistral QLoRA summarizer

```bash
python3 scripts/train_summarizer.py \
  --data-dir data/mimiciii/processed/summarization \
  --config configs/summarizer/llama3_8b_qlora.json \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir artifacts/summarizer/mistral_7b_qlora \
  --save-strategy steps \
  --save-steps 200 \
  --save-total-limit 2 \
  --eval-strategy epoch \
  --logging-steps 20 \
  --auto-resume
```

### 4. Evaluate the baseline

```bash
python3 scripts/evaluate_summarizer.py \
  --input-file data/mimiciii/raw/test.jsonl \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora \
  --output-dir artifacts/evaluation/summarizer/mistral_7b_qlora_eval
```

### 5. Inspect results

```bash
cat artifacts/evaluation/summarizer/mistral_7b_qlora_eval/metrics.json
head -n 20 artifacts/evaluation/summarizer/mistral_7b_qlora_eval/predictions.jsonl
```

## Current Best Next Experiment

The current recommended follow-up is a cleaner narrative-target dataset.

Build the filtered dataset:

```bash
python3 scripts/prepare_datasets.py \
  --input-dir data/mimiciii/raw \
  --output-dir data/mimiciii/processed_narrative \
  --narrative-only \
  --min-target-words 40 \
  --max-target-words 260 \
  --min-target-sentences 2 \
  --max-structured-markers 0
```

Train on that filtered dataset:

```bash
python3 scripts/train_summarizer.py \
  --data-dir data/mimiciii/processed_narrative/summarization \
  --config configs/summarizer/llama3_8b_qlora.json \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir artifacts/summarizer/mistral_7b_qlora_narrative \
  --save-strategy steps \
  --save-steps 200 \
  --save-total-limit 2 \
  --eval-strategy epoch \
  --logging-steps 20 \
  --auto-resume
```

Evaluate it:

```bash
python3 scripts/evaluate_summarizer.py \
  --input-file data/mimiciii/raw/test.jsonl \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora_narrative \
  --output-dir artifacts/evaluation/summarizer/mistral_7b_qlora_narrative_eval \
  --limit 50
```

## Smoke Test Workflow

Use a smaller ingestion job before scaling up:

```bash
python3 scripts/ingest_mimiciii_notes.py \
  --input-file data/mimiciii/NOTEEVENTS.csv \
  --output-dir data/mimiciii/raw \
  --max-examples 1000

python3 scripts/prepare_datasets.py \
  --input-dir data/mimiciii/raw \
  --output-dir data/mimiciii/processed

python3 scripts/train_summarizer.py \
  --data-dir data/mimiciii/processed/summarization \
  --config configs/summarizer/llama3_8b_qlora.json \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir artifacts/summarizer/mistral_7b_qlora_smoke

python3 scripts/evaluate_summarizer.py \
  --input-file data/mimiciii/raw/test.jsonl \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora_smoke \
  --output-dir artifacts/evaluation/summarizer/mistral_7b_qlora_smoke_eval \
  --limit 20
```

## Debug Workflow

Create shortened targets for quick overfit checks:

```bash
python3 scripts/prepare_datasets.py \
  --input-dir data/mimiciii/raw \
  --output-dir data/mimiciii/processed_short \
  --target-sentence-limit 2
```

This preserves the original full target as `target_text_full`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/ingest_mimiciii_notes.py --input-file data/mimiciii/NOTEEVENTS.csv
python3 scripts/prepare_datasets.py
python3 scripts/train_summarizer.py \
  --data-dir data/mimiciii/processed/summarization \
  --config configs/summarizer/llama3_8b_qlora.json \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir artifacts/summarizer/mistral_7b_qlora
python3 scripts/evaluate_summarizer.py \
  --input-file data/mimiciii/raw/test.jsonl \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora
```

## Testing

Run the lightweight test suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

These tests cover:
- config loading
- MIMIC-III ingestion
- preprocessing helpers, including narrative filtering
- summarizer evaluation schema handling
- summarizer prompting and tokenizer configuration

## Limitations

- verifier training is still blocked on missing claim labels
- overlap metrics do not directly measure factual faithfulness
- the summarizer still struggles with heterogeneous hospital-course target styles
- some decoding settings can make outputs more generic or hallucination-prone
- model downloads require internet access on first use
- the recommended Mistral QLoRA path assumes CUDA

Additional preset notes are in [configs/README.md](configs/README.md).
