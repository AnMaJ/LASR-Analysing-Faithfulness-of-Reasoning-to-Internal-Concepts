"""
Multi-Layer SAE Activation Collection & Aggregation Pipeline
=============================================================
Collects SAE activations across all layers of Gemma 3 4B IT using 16k-width small-L0
residual-stream SAEs from Gemma Scope 2, on 100 ECQA samples (CoT mode).

For each sample and each layer:
  - Generate text using CoT prompting
  - Capture residual stream hidden states via hooks during generation
  - Encode hidden states through the SAE to get feature activations
  - Keep only activations for GENERATED tokens (not prompt tokens)
  - Aggregate per-feature activations using 3 methods:
      1. Top-k Mean (k=10)
      2. Max Activation
      3. Top-k Sum (k=10)

Results are saved as .safetensors files and pushed to a HuggingFace repo.
"""

import os
import sys
import gc
import json
import logging
import time
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi, login, create_repo
from datasets import load_dataset


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    # Model
    MODEL_NAME = "google/gemma-3-4b-it"
    SAE_REPO = "google/gemma-scope-2-4b-it"
    
    # SAE configuration: 16k width, small L0, residual stream (resid_post_all for all layers)
    SAE_WIDTH = "16k"
    SAE_L0 = "small"
    SAE_SITE = "resid_post_all"  # all layers available
    
    # Gemma 3 4B has 34 transformer layers (0-33)
    NUM_LAYERS = 34
    ALL_LAYERS = list(range(NUM_LAYERS))
    
    # Dataset
    NUM_SAMPLES = 100
    PROMPT_STYLE = "CHAIN_OF_THOUGHT"
    MAX_NEW_TOKENS = 300
    
    # Aggregation
    TOP_K = 10
    
    # HuggingFace output repo (change this to your repo)
    OUTPUT_HF_REPO = "YOUR_USERNAME/gemma3-4b-sae-activations-ecqa"
    
    # Paths
    WORK_DIR = "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/outputs/sae_activations"
    LOG_FILE = "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/outputs/sae_activations/collection.log"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16  # for model inference


# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("sae_activation_collection")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ==============================================================================
# Dataset
# ==============================================================================

class ECQA_Dataset:
    """ECQA dataset with CoT prompting for Gemma 3."""
    
    _INSTRUCTIONS_ = {
        "ONE_WORD": "Answer with only A, B, C, D, or E.",
        "CHAIN_OF_THOUGHT": (
            "Please think step by step before giving your final answer. "
            "Consider what information is provided and what assumptions might be involved. "
            "After your reasoning, clearly state your final answer as A, B, C, D, or E."
        ),
    }
    _OPTION_LABELS_ = ["A", "B", "C", "D", "E"]

    def __init__(self):
        print("Loading ECQA dataset from HuggingFace...")
        ds = pd.read_parquet("/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/datasets/ecqa/data/validation-00000-of-00001.parquet")
        self.data = ds
        print(f"ECQA dataset loaded: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def build_prompt(self, idx: int, prompt_style: str = "CHAIN_OF_THOUGHT") -> str:
        """Build a Gemma 3 chat-formatted prompt for the given ECQA sample."""
        row = self.data.iloc[idx]
        options_text = "\n".join(
            f"{self._OPTION_LABELS_[i].lower()}) {row[f'q_op{i+1}']}" for i in range(5)
        )
        instruction = self._INSTRUCTIONS_[prompt_style]
        
        # Gemma 3 IT chat template
        prompt = (
            f"<start_of_turn>user\n"
            f"{instruction}\n\n"
            f"Question: {row['q_text']}\n\n"
            f"Options:\n{options_text}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return prompt


# ==============================================================================
# JumpReLU SAE
# ==============================================================================

class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder (as used in Gemma Scope 2)."""
    
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        return self.decode(acts)


# ==============================================================================
# SAE Loading Utilities
# ==============================================================================

def load_sae_for_layer(layer_idx: int, config: Config, logger: logging.Logger) -> JumpReLUSAE:
    """Download and load the SAE for a specific layer."""
    filename = f"{config.SAE_SITE}/layer_{layer_idx}_width_{config.SAE_WIDTH}_l0_{config.SAE_L0}/params.safetensors"
    
    logger.info(f"  Downloading SAE for layer {layer_idx}: {filename}")
    path_to_params = hf_hub_download(
        repo_id=config.SAE_REPO,
        filename=filename,
    )
    
    params = load_file(path_to_params)
    d_model, d_sae = params["w_enc"].shape
    
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(params)
    sae = sae.to(config.DEVICE)
    sae.eval()
    
    logger.info(f"  SAE loaded: d_model={d_model}, d_sae={d_sae}")
    return sae


# ==============================================================================
# Generation + Hidden State Capture
# ==============================================================================

def generate_and_capture_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    max_new_tokens: int,
    device: str,
) -> Tuple[torch.Tensor, int, int, str]:
    """
    Generate text and capture hidden states at a specific layer during generation.
    
    Returns:
        full_hidden: [total_seq_len, d_model] tensor of hidden states
        prompt_len: number of prompt tokens
        gen_len: number of generated tokens
        generated_text: the full generated text
    """
    all_hidden_states = []

    def capture_hook(module, input, output):
        # output[0] shape: [batch, seq_len, d_model]
        all_hidden_states.append(output[0].detach().cpu())
        return output

    # Hook into the correct layer
    # Gemma 3 4B IT uses model.model.layers (GemmaForCausalLM) or
    # model.model.language_model.layers (Gemma3ForConditionalGeneration)
    # We handle both architectures
    if hasattr(model.model, 'language_model'):
        target_module = model.model.language_model.layers[layer_idx]
    else:
        target_module = model.model.layers[layer_idx]

    handle = target_module.register_forward_hook(capture_hook)

    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        prompt_len = inputs.shape[1]
        attention_mask = torch.ones_like(inputs)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        total_len = outputs.shape[1]
        gen_len = total_len - prompt_len

    finally:
        handle.remove()

    # Reconstruct full hidden states from hook captures
    # First forward pass: all prompt tokens [1, prompt_len, d_model]
    full_hidden = all_hidden_states[0].squeeze(0)  # [prompt_len, d_model]

    # Subsequent passes: one generated token each [1, 1, d_model]
    for hidden in all_hidden_states[1:]:
        h = hidden.squeeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        full_hidden = torch.cat([full_hidden, h], dim=0)

    return full_hidden, prompt_len, gen_len, generated_text


# ==============================================================================
# Activation Aggregation
# ==============================================================================

def aggregate_activations(
    sae_acts_gen: torch.Tensor,  # [gen_len, num_features]
    top_k: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate per-token SAE activations into per-feature scalar values.
    Only operates on generated tokens (prompt tokens already excluded).
    
    Returns dict with keys:
        'topk_mean': [num_features] - mean of top-k activations per feature
        'max': [num_features] - max activation per feature
        'topk_sum': [num_features] - sum of top-k activations per feature
    """
    num_tokens, num_features = sae_acts_gen.shape
    k = min(top_k, num_tokens)  # handle case where gen_len < top_k
    
    # Top-k values along the token dimension for each feature
    # sae_acts_gen: [gen_len, num_features] -> topk along dim=0
    topk_vals, _ = torch.topk(sae_acts_gen, k=k, dim=0)  # [k, num_features]
    
    topk_mean = topk_vals.mean(dim=0)   # [num_features]
    topk_sum = topk_vals.sum(dim=0)     # [num_features]
    max_act = sae_acts_gen.max(dim=0).values  # [num_features]
    
    return {
        "topk_mean": topk_mean,
        "max": max_act,
        "topk_sum": topk_sum,
    }


# ==============================================================================
# Main Collection Pipeline
# ==============================================================================

def collect_activations(config: Config, logger: logging.Logger):
    """Main pipeline: collect and aggregate SAE activations across layers and samples."""
    
    os.makedirs(config.WORK_DIR, exist_ok=True)
    
    # ---- Load dataset ----
    logger.info("=" * 80)
    logger.info("STEP 1: Loading ECQA dataset")
    logger.info("=" * 80)
    dataset = ECQA_Dataset()
    
    # Select 100 random samples (fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(dataset), size=config.NUM_SAMPLES, replace=False).tolist()
    sample_indices.sort()
    logger.info(f"Selected {config.NUM_SAMPLES} sample indices: {sample_indices[:10]}... (showing first 10)")
    
    # ---- Load model ----
    logger.info("=" * 80)
    logger.info("STEP 2: Loading Gemma 3 4B IT model")
    logger.info("=" * 80)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        torch_dtype=config.DTYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model.eval()
    logger.info("Model loaded successfully.")
    
    # ---- Detect model architecture ----
    if hasattr(model.model, 'language_model'):
        num_layers = len(model.model.language_model.layers)
        arch = "Gemma3ForConditionalGeneration (language_model)"
    else:
        num_layers = len(model.model.layers)
        arch = "GemmaForCausalLM"
    logger.info(f"Model architecture: {arch}, num_layers={num_layers}")
    config.NUM_LAYERS = num_layers
    config.ALL_LAYERS = list(range(num_layers))
    
    # ---- Pre-generate all prompts and texts ----
    # We generate text ONCE per sample (not per layer), since generation is model-level.
    # Then we do a separate forward pass per layer to collect hidden states.
    # BUT: to capture hidden states during generation (including KV-cached steps),
    # we need the hook active during generate(). So we must run generate() per layer.
    #
    # OPTIMIZATION: Generate once to get the output token IDs, then do a single 
    # forward pass per layer on the full sequence to get hidden states.
    # This avoids running generate() N_layers times.
    
    logger.info("=" * 80)
    logger.info("STEP 3: Generating text for all samples")
    logger.info("=" * 80)
    
    all_prompts = []
    all_output_ids = []
    all_prompt_lens = []
    all_gen_lens = []
    all_generated_texts = []
    
    for i, sample_idx in enumerate(sample_indices):
        prompt = dataset.build_prompt(sample_idx, config.PROMPT_STYLE)
        all_prompts.append(prompt)
        
        # Generate text (no hooks needed for this step)
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(config.DEVICE)
        prompt_len = inputs.shape[1]
        attention_mask = torch.ones_like(inputs)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        total_len = outputs.shape[1]
        gen_len = total_len - prompt_len
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        all_output_ids.append(outputs[0].cpu())
        all_prompt_lens.append(prompt_len)
        all_gen_lens.append(gen_len)
        all_generated_texts.append(generated_text)
        
        logger.info(f"  Sample {i+1}/{config.NUM_SAMPLES} (idx={sample_idx}): "
                     f"prompt_len={prompt_len}, gen_len={gen_len}, total={total_len}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"  --- Generated {i+1}/{config.NUM_SAMPLES} samples ---")
    
    # Save metadata
    metadata = {
        "model": config.MODEL_NAME,
        "sae_repo": config.SAE_REPO,
        "sae_width": config.SAE_WIDTH,
        "sae_l0": config.SAE_L0,
        "sae_site": config.SAE_SITE,
        "num_samples": config.NUM_SAMPLES,
        "num_layers": config.NUM_LAYERS,
        "top_k": config.TOP_K,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "prompt_style": config.PROMPT_STYLE,
        "sample_indices": sample_indices,
        "prompt_lens": all_prompt_lens,
        "gen_lens": all_gen_lens,
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(config.WORK_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Save generated texts for reference
    texts_path = os.path.join(config.WORK_DIR, "generated_texts.json")
    with open(texts_path, "w") as f:
        json.dump({
            "prompts": all_prompts,
            "generated_texts": all_generated_texts,
            "sample_indices": sample_indices,
        }, f, indent=2)
    logger.info(f"Generated texts saved to {texts_path}")
    
    # ---- Process each layer ----
    logger.info("=" * 80)
    logger.info("STEP 4: Collecting SAE activations across all layers")
    logger.info("=" * 80)
    
    for layer_idx in config.ALL_LAYERS:
        layer_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Layer {layer_idx}/{config.NUM_LAYERS - 1}")
        logger.info(f"{'='*60}")
        
        # Load SAE for this layer
        try:
            sae = load_sae_for_layer(layer_idx, config, logger)
        except Exception as e:
            logger.error(f"  Failed to load SAE for layer {layer_idx}: {e}")
            logger.error(f"  Skipping layer {layer_idx}")
            continue
        
        d_sae = sae.w_enc.shape[1]
        
        # Storage for aggregated activations: [num_samples, num_features]
        topk_mean_all = torch.zeros(config.NUM_SAMPLES, d_sae)
        max_all = torch.zeros(config.NUM_SAMPLES, d_sae)
        topk_sum_all = torch.zeros(config.NUM_SAMPLES, d_sae)
        
        for i in range(config.NUM_SAMPLES):
            sample_start = time.time()
            
            # Get the full output sequence and run a single forward pass
            # to capture hidden states at this layer
            full_ids = all_output_ids[i].unsqueeze(0).to(config.DEVICE)
            prompt_len = all_prompt_lens[i]
            gen_len = all_gen_lens[i]
            
            # Capture hidden states via a single forward pass on full sequence
            cache = {}

            def capture_hook(mod, inp, out, cache=cache):
                cache["hidden"] = out[0].detach()  # [1, seq_len, d_model]
                return out

            if hasattr(model.model, 'language_model'):
                target_module = model.model.language_model.layers[layer_idx]
            else:
                target_module = model.model.layers[layer_idx]

            handle = target_module.register_forward_hook(capture_hook)
            try:
                with torch.no_grad():
                    _ = model.forward(full_ids)
            finally:
                handle.remove()
            
            hidden = cache["hidden"].squeeze(0)  # [seq_len, d_model]
            
            # Extract ONLY generated token hidden states
            gen_hidden = hidden[prompt_len:prompt_len + gen_len]  # [gen_len, d_model]
            
            # Encode through SAE
            with torch.no_grad():
                sae_acts = sae.encode(gen_hidden.to(torch.float32))  # [gen_len, d_sae]
            
            # Aggregate
            agg = aggregate_activations(sae_acts, top_k=config.TOP_K)
            
            topk_mean_all[i] = agg["topk_mean"].cpu()
            max_all[i] = agg["max"].cpu()
            topk_sum_all[i] = agg["topk_sum"].cpu()
            
            # Clean up
            del hidden, gen_hidden, sae_acts, cache["hidden"]
            
            if (i + 1) % 20 == 0:
                elapsed = time.time() - sample_start
                logger.info(f"  Layer {layer_idx} - Sample {i+1}/{config.NUM_SAMPLES} "
                           f"(gen_len={gen_len}, time={elapsed:.1f}s)")
        
        # Save activations for this layer
        layer_dir = os.path.join(config.WORK_DIR, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        save_file(
            {"activations": topk_mean_all},
            os.path.join(layer_dir, "topk_mean.safetensors")
        )
        save_file(
            {"activations": max_all},
            os.path.join(layer_dir, "max.safetensors")
        )
        save_file(
            {"activations": topk_sum_all},
            os.path.join(layer_dir, "topk_sum.safetensors")
        )
        
        # Free SAE from GPU
        del sae, topk_mean_all, max_all, topk_sum_all
        torch.cuda.empty_cache()
        gc.collect()
        
        layer_elapsed = time.time() - layer_start
        logger.info(f"  Layer {layer_idx} complete. Time: {layer_elapsed:.1f}s. "
                     f"Saved to {layer_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: All layers processed. Collection complete!")
    logger.info("=" * 80)
    
    return config.WORK_DIR


# ==============================================================================
# HuggingFace Upload
# ==============================================================================

def upload_to_huggingface(work_dir: str, repo_id: str, logger: logging.Logger):
    """Upload all collected activations to a HuggingFace dataset repo."""
    
    logger.info(f"Uploading results to HuggingFace repo: {repo_id}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        logger.info(f"  Repo '{repo_id}' ready.")
    except Exception as e:
        logger.warning(f"  Repo creation note: {e}")
    
    # Upload the entire work directory
    api.upload_folder(
        folder_path=work_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    logger.info(f"  Upload complete!")


# ==============================================================================
# README Generator
# ==============================================================================

def generate_readme(config: Config, work_dir: str):
    """Generate a README.md for the HuggingFace dataset repo."""
    
    readme = f"""---
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
collected across **all {config.NUM_LAYERS} layers** of `{config.MODEL_NAME}`, using 
**{config.SAE_WIDTH}-width, {config.SAE_L0}-L0 residual-stream SAEs** from 
[Gemma Scope 2]({f'https://huggingface.co/{config.SAE_REPO}'}).

Activations were collected on **{config.NUM_SAMPLES} samples** from the 
[ECQA dataset](https://huggingface.co/datasets/yangwang825/ecqa) (validation split) 
using **Chain-of-Thought (CoT) prompting**, with greedy decoding 
(max {config.MAX_NEW_TOKENS} new tokens).

**Only generated tokens are used** (prompt tokens are excluded before aggregation).

## Aggregation Methods

For each feature in the SAE, the per-token activation vector is aggregated into 
a single scalar value using three methods:

| Method | File | Description |
|--------|------|-------------|
| **Top-k Mean** | `topk_mean.safetensors` | Mean of top {config.TOP_K} activation values |
| **Max** | `max.safetensors` | Maximum activation across all generated tokens |
| **Top-k Sum** | `topk_sum.safetensors` | Sum of top {config.TOP_K} activation values |

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
└── layer_{config.NUM_LAYERS - 1}/
    ├── ...
```

Each `.safetensors` file contains a single key `"activations"` with shape 
`[{config.NUM_SAMPLES}, d_sae]` where `d_sae` is the SAE dictionary size (16384 for 16k).

## Loading Example

```python
from safetensors.torch import load_file

# Load top-k mean activations for layer 15
data = load_file("layer_15/topk_mean.safetensors")
activations = data["activations"]  # shape: [100, 16384]

# activations[i, j] = aggregated activation of feature j for sample i
```

## Configuration

- **Model**: `{config.MODEL_NAME}`
- **SAE Repo**: `{config.SAE_REPO}`
- **SAE Site**: `{config.SAE_SITE}` (residual stream, all layers)
- **SAE Width**: {config.SAE_WIDTH}
- **SAE L0**: {config.SAE_L0}
- **Aggregation Top-k**: {config.TOP_K}
- **Num Samples**: {config.NUM_SAMPLES}
- **Prompt Style**: {config.PROMPT_STYLE}
- **Max New Tokens**: {config.MAX_NEW_TOKENS}
"""
    
    readme_path = os.path.join(work_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    
    return readme_path


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Setup
    os.makedirs(config.WORK_DIR, exist_ok=True)
    logger = setup_logging(config.LOG_FILE)
    
    logger.info("=" * 80)
    logger.info("SAE Activation Collection Pipeline")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    logger.info(f"Config: model={config.MODEL_NAME}, sae_width={config.SAE_WIDTH}, "
                f"sae_l0={config.SAE_L0}, num_samples={config.NUM_SAMPLES}, "
                f"num_layers={config.NUM_LAYERS}")
    
    # HuggingFace login (expects HF_TOKEN env variable)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace.")
    else:
        logger.warning("HF_TOKEN not set. Upload will fail unless already authenticated.")
    
    # Run collection
    start_time = time.time()
    work_dir = collect_activations(config, logger)
    total_time = time.time() - start_time
    logger.info(f"\nTotal collection time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Generate README
    generate_readme(config, work_dir)
    logger.info("README.md generated.")
    
    # Upload to HuggingFace
    try:
        upload_to_huggingface(work_dir, config.OUTPUT_HF_REPO, logger)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.info("Files are saved locally. You can upload manually.")
    
    logger.info("Pipeline complete!")