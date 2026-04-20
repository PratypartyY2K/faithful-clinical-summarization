# Config Presets

These JSON presets can be passed to the main scripts with `--config`.

All bundled presets assume official data lives under `data/mimiciii/...`.

## Recommended Workflow

Prepare processed splits:

```bash
python3 scripts/ingest_mimiciii_notes.py
python3 scripts/prepare_datasets.py
```

Train the recommended Mistral QLoRA summarizer:

```bash
python3 scripts/train_summarizer.py \
  --config configs/summarizer/llama3_8b_qlora.json \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir artifacts/summarizer/mistral_7b_qlora
```

Run summarizer-only evaluation:

```bash
python3 scripts/evaluate_summarizer.py \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora
```

## Smoke Test Workflow

Run a smaller ingestion job before a longer training run:

```bash
python3 scripts/ingest_mimiciii_notes.py --max-examples 1000
python3 scripts/prepare_datasets.py
python3 scripts/train_summarizer.py \
  --config configs/summarizer/llama3_8b_qlora.json \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir artifacts/summarizer/mistral_7b_qlora_smoke
python3 scripts/evaluate_summarizer.py \
  --input-file data/mimiciii/raw/test.jsonl \
  --summarizer-dir artifacts/summarizer/mistral_7b_qlora_smoke \
  --limit 20
```

## Short-Target Debug Workflow

Create shortened targets for overfit/debug checks:

```bash
python3 scripts/prepare_datasets.py --target-sentence-limit 2
```

Then train and evaluate with the same Mistral QLoRA command, pointing at a separate output directory if needed.

## Other Presets

Some older summarizer configs and verifier/evaluation presets remain in the repository for experiment history and future work. They are not the primary documented workflow for the current project because:

- the current MIMIC ingestion pipeline does not yet generate claim labels
- verifier splits are therefore empty by design
- earlier seq2seq baselines are no longer the recommended path
