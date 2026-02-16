---
license: apache-2.0
tags:
  - sparse-autoencoders
  - gemma-scope-2
  - mechanistic-interpretability
  - SAE-activations
  - ecqa
---

# Gemma 3 4B IT — SAE Activation Aggregations (ECQA, CoT)

## Overview

This dataset contains **aggregated SAE (Sparse Autoencoder) feature activations** 
collected across **all 34 layers** of `google/gemma-3-4b-it`, using 
**16k-width, small-L0 residual-stream SAEs** from 
[Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-4b-it).

Activations were collected on **100 samples** from the 
[ECQA dataset](https://huggingface.co/datasets/yangwang825/ecqa) (validation split) 
using **Chain-of-Thought (CoT) prompting**, with greedy decoding 
(max 300 new tokens).

**Only generated tokens are used** (prompt tokens are excluded before aggregation).

## Aggregation Methods

For each feature in the SAE, the per-token activation vector is aggregated into 
a single scalar value using three methods:

| Method | File | Description |
|--------|------|-------------|
| **Top-k Mean** | `topk_mean.safetensors` | Mean of top 10 activation values |
| **Max** | `max.safetensors` | Maximum activation across all generated tokens |
| **Top-k Sum** | `topk_sum.safetensors` | Sum of top 10 activation values |

## File Structure

```
├── metadata.json                  # Full configuration and sample metadata
├── generated_texts.json           # Prompts and generated texts for all samples
├── collection.log                 # Detailed execution log
├── layer_0/
│   ├── topk_mean.safetensors     # [100, 16384] tensor
│   ├── max.safetensors           # [100, 16384] tensor
│   └── topk_sum.safetensors      # [100, 16384] tensor
├── layer_1/
│   ├── ...
...
└── layer_33/
    ├── ...
```

Each `.safetensors` file contains a single key `"activations"` with shape 
`[100, d_sae]` where `d_sae` is the SAE dictionary size (16384 for 16k).

## Loading Example

```python
from safetensors.torch import load_file

# Load top-k mean activations for layer 15
data = load_file("layer_15/topk_mean.safetensors")
activations = data["activations"]  # shape: [100, 16384]

# activations[i, j] = aggregated activation of feature j for sample i
```

## Configuration

- **Model**: `google/gemma-3-4b-it`
- **SAE Repo**: `google/gemma-scope-2-4b-it`
- **SAE Site**: `resid_post_all` (residual stream, all layers)
- **SAE Width**: 16k
- **SAE L0**: small
- **Aggregation Top-k**: 10
- **Num Samples**: 100
- **Prompt Style**: CHAIN_OF_THOUGHT
- **Max New Tokens**: 300
