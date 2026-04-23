# Config Presets

These JSON presets can be passed to the main scripts with `--config`.

All bundled presets assume official data lives under `data/mimiciii/...`.

The presets in this directory are a mix of current and historical experiments:
- `configs/summarizer/llama3_8b_qlora.json` is the current recommended summarizer preset and is typically paired with `mistralai/Mistral-7B-Instruct-v0.2`
- the `flan_t5_*` summarizer presets remain for seq2seq experiment history
- verifier and full-pipeline presets remain future-facing because the current MIMIC ingestion path does not populate claim labels

Some historical presets still point at artifact paths that are not checked into the repository, such as `artifacts/summarizer/flan_t5_small`. Treat those as references for reproducing past runs, not as guaranteed local artifacts.

## Project-Relevant Use Of Presets

For the course project, the presets are most useful for organizing a small number of controlled experiments:

1. baseline summarizer training
2. narrative-only filtered-data training
3. smoke-test or debug runs
4. optional historical seq2seq comparison

The goal is not to exhaustively sweep every preset. The goal is to keep one stable baseline and change one variable at a time.

## Recommended Workflow

Prepare processed splits:

```bash
python3 scripts/ingest_mimiciii_notes.py \
  --input-file data/mimiciii/NOTEEVENTS.csv \
  --output-dir data/mimiciii/raw

python3 scripts/prepare_datasets.py \
  --input-dir data/mimiciii/raw \
  --output-dir data/mimiciii/processed
```

Train the recommended Mistral QLoRA summarizer:

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

Run summarizer-only evaluation:

```bash
python3 scripts/evaluate_summarizer.py \
  --input-file data/mimiciii/raw/test.jsonl \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora
```

## Current Best Follow-Up

The most useful next experiment is the narrative-only filtered dataset, not more decoding sweeps.

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

Train on it:

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

Run a smaller ingestion job before a longer training run:

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

## Short-Target Debug Workflow

Create shortened targets for overfit/debug checks:

```bash
python3 scripts/prepare_datasets.py \
  --input-dir data/mimiciii/raw \
  --output-dir data/mimiciii/processed_short \
  --target-sentence-limit 2
```

Then train and evaluate with the same Mistral QLoRA command, pointing at a separate output directory if needed.

## Decoding Checks

`scripts/evaluate_summarizer.py` supports quick decoding comparisons through:
- `--num-beams`
- `--no-repeat-ngram-size`
- `--repetition-penalty`
- `--length-penalty`

These are useful for small evaluation checks, but they are not currently the main recommended improvement path for this project.

## Suggested Experiment Table

The most report-worthy experiment lineup is:

- historical or simple baseline
  - optional T5 experiment or simple extractive heuristic
- full-data Mistral QLoRA baseline
- `narrative-only` Mistral QLoRA model
- decoding comparison as a small ablation or negative result

This gives you:
- baseline comparison
- ablation evidence
- a clear story about why the best model is best

## Other Presets

Some older summarizer configs and verifier/evaluation presets remain in the repository for experiment history and future work. They are not the primary documented workflow for the current project because:

- the current MIMIC ingestion pipeline does not yet generate claim labels
- verifier splits are therefore empty by design
- earlier seq2seq baselines are no longer the recommended path

If you want a checked-in artifact to inspect alongside these presets, the repository currently includes `artifacts/summarizer/mistral_7b_qlora_smoke/`, which is a smoke-test PEFT adapter directory rather than a full merged model checkpoint.
