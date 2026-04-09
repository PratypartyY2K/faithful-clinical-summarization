# Config Presets

These JSON presets can be passed to the main scripts with `--config`.

## Data

```bash
python3 scripts/create_dummy_dataset.py --config configs/data/dummy_nli.json
python3 scripts/prepare_datasets.py
```

## flan-t5-small baseline

```bash
python3 scripts/train_summarizer.py --config configs/summarizer/flan_t5_small.json
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
