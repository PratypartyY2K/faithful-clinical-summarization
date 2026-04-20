# Config Presets

These JSON presets can be passed to the main scripts with `--config`.

All bundled presets now assume your official data lives under `data/mimiciii/...`.

## Prepare datasets

```bash
python3 scripts/ingest_mimiciii_notes.py
python3 scripts/prepare_datasets.py
```

## flan-t5-small baseline

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
```

## flan-t5-base baseline

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_base.json
```

## flan-t5-base long-source smoke test

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_base_long_source.json
```

## Llama-3-8B QLoRA run

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/llama3_8b_qlora.json
```

## DeBERTa-v3-large verifier

```bash
python3 scripts/train_verifier.py --config configs/verifier/deberta_v3_large.json
```

## Full evaluation run

```bash
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline.json
```

## Summarizer-only evaluation

```bash
python3 scripts/evaluate_summarizer.py --summarizer-dir artifacts/summarizer/flan_t5_small
```

## Full evaluation with LLM claim extraction

Requires `OPENAI_API_KEY`.

```bash
python3 scripts/evaluate_pipeline.py --config configs/evaluation/full_pipeline_llm_claims.json
```
