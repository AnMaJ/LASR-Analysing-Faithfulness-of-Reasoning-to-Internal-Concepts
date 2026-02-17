# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating faithfulness of chain-of-thought reasoning in language models (Gemma) by analyzing internal concept representations via GemmaScope 2 Sparse Autoencoders (SAEs). It connects internal model features to human-interpretable descriptions via the Neuronpedia API, then evaluates whether stated reasoning corresponds to actual internal activations.

## Setup

```bash
# Requires Python >= 3.12
python -m venv env
source env/bin/activate
pip install -e .
# Additional undeclared deps needed at runtime:
pip install transformers tqdm datasets pandas matplotlib
```

The package is installed in editable mode. Virtual environment lives at `./env/` (gitignored).

## Project Structure

All source code is under `src/`, importable as `src.*` (e.g., `from src.configs import SAEConfig`).

### Core Pipeline

1. **configs.py** — Dataclass-based configuration: `ModelConfig`, `SAEConfig`, `DatasetConfig`, `PromptStyle` enum (controls CoT vs one-word, tags vs no-tags)
2. **gemma_model.py** — `GemmaModel` wraps HuggingFace `AutoModelForCausalLM` with generation and residual stream activation extraction via forward hooks
3. **SAE.py** — `JumpReLUSAE` (nn.Module) with JumpReLU activation, loads from HuggingFace Hub via `from_pretrained()`, encodes activations into sparse features
4. **denoiser.py** — `Denoiser` with pluggable strategies (`continuous_tfidf`, `standard_scaler`) for normalizing SAE activations
5. **aggregator.py** — `Aggregator` with pluggable strategies (`max`) for reducing token×features to a single feature vector
6. **feature.py** — `Feature` wraps a single SAE feature with index, strength, and Neuronpedia metadata; `from_activations()` factory creates from top-k
7. **neuronpedia_client.py** — REST client for Neuronpedia API (feature descriptions, activation examples, dashboards)

### Datasets (`src/dataset/`)

Abstract `BaseDataset(ABC, Dataset)` with three implementations:
- **bbq.py** — BBQ bias benchmark (3-option MC, from HuggingFace `HiTZ/bbq`)
- **ecqa.py** — ECQA commonsense QA (5-option MC, from Parquet)
- **esnli.py** — e-SNLI NLI (3-class, from CSV, supports few-shot)

Each dataset defines `_INSTRUCTIONS_` mappings, `build_prompt()`, and answer parsing logic.

### Key Patterns

- **Decorator-based method registration:** `Aggregator` and `Denoiser` use `@_aggregation_method`/`@_denoising_method` decorators that register strategies and enforce shape contracts
- **Device auto-detection:** `ModelConfig._default_device()` checks MPS → CUDA → CPU
- **Notebooks in `notebooks/`** are the primary experiment interface (demo.ipynb, gemmascope_2_first_experiment.ipynb)

## No Test Suite or Linting

There are currently no tests, linting, formatting, or CI/CD configurations.

## Git Workflow

Work on personal branches, merge to main after group discussion.
