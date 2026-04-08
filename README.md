# Faithful Clinical Summarization via Atomic Claim Verification

This repository contains the implementation for our CSE 587 Deep Learning project at Penn State. We address the challenge of hallucinations in clinical NLP by introducing a two-stage pipeline that generates summaries and verifies them via atomic claim decomposition.

## Overview

- **Task:** Abstractive Clinical Dialogue Summarization.
- **Goal:** Minimize hallucinations (factual errors) in AI-generated medical notes.
- **Models:** Llama-3-8B (Generator) + DeBERTa-v3-large (Verifier).
- **Dataset:** MIMIC-III Clinical Database.
