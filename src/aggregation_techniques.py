"""
Quick-Start Guide & Usage Examples
====================================
This file shows how to run the pipeline, customize it, and load the results.

Run the main script on a GPU machine (e.g., Colab A100, Lambda, RunPod):

    export HF_TOKEN="hf_your_token_here"
    python collect_sae_activations.py

Or import and customize:
"""

# ==============================================================================
# 1) Minimal run with fewer samples/layers for testing
# ==============================================================================

def run_test():
    """Quick test with 5 samples and 3 layers."""
    from collect_sae_activations import Config, collect_activations, setup_logging
    
    config = Config()
    config.NUM_SAMPLES = 5
    config.ALL_LAYERS = [0, 16, 33]  # early, middle, late
    config.MAX_NEW_TOKENS = 100
    config.OUTPUT_HF_REPO = "YOUR_USERNAME/test-sae-activations"
    
    logger = setup_logging(config.LOG_FILE)
    collect_activations(config, logger)


# ==============================================================================
# 2) Loading and analyzing saved activations
# ==============================================================================

def analyze_activations(data_dir: str = "./sae_activations"):
    """Example: load and analyze collected activations."""
    import json
    import torch
    import numpy as np
    from safetensors.torch import load_file
    
    # Load metadata
    with open(f"{data_dir}/metadata.json") as f:
        meta = json.load(f)
    
    num_layers = meta["num_layers"]
    num_samples = meta["num_samples"]
    
    print(f"Dataset: {num_samples} samples, {num_layers} layers")
    print(f"Model: {meta['model']}")
    print(f"SAE: {meta['sae_width']} width, {meta['sae_l0']} L0")
    
    # Load activations for a specific layer
    layer = 16
    topk_mean = load_file(f"{data_dir}/layer_{layer}/topk_mean.safetensors")["activations"]
    max_act = load_file(f"{data_dir}/layer_{layer}/max.safetensors")["activations"]
    topk_sum = load_file(f"{data_dir}/layer_{layer}/topk_sum.safetensors")["activations"]
    
    print(f"\nLayer {layer} shapes:")
    print(f"  topk_mean: {topk_mean.shape}")  # [100, 16384]
    print(f"  max:       {max_act.shape}")
    print(f"  topk_sum:  {topk_sum.shape}")
    
    # Find most active features across all samples (by mean of max activations)
    mean_max = max_act.mean(dim=0)  # average max activation per feature
    top_features = mean_max.topk(20)
    
    print(f"\nTop 20 most active features at layer {layer} (by avg max activation):")
    for val, idx in zip(top_features.values, top_features.indices):
        print(f"  Feature {idx.item():6d}: avg_max={val.item():.4f}")
    
    # Cross-layer analysis: track a feature across layers
    feature_idx = top_features.indices[0].item()
    print(f"\nTracking feature {feature_idx} across layers (topk_mean):")
    
    for l in range(0, num_layers, 4):
        try:
            acts = load_file(f"{data_dir}/layer_{l}/topk_mean.safetensors")["activations"]
            feat_vals = acts[:, feature_idx]
            print(f"  Layer {l:2d}: mean={feat_vals.mean():.4f}, "
                  f"max={feat_vals.max():.4f}, "
                  f"nonzero={(feat_vals > 0).sum().item()}/{num_samples}")
        except FileNotFoundError:
            print(f"  Layer {l:2d}: (not available)")


# ==============================================================================
# 3) Comparing aggregation methods
# ==============================================================================

def compare_aggregation_methods(data_dir: str = "./sae_activations", layer: int = 16):
    """Compare how different aggregation methods rank features."""
    import torch
    from safetensors.torch import load_file
    
    topk_mean = load_file(f"{data_dir}/layer_{layer}/topk_mean.safetensors")["activations"]
    max_act = load_file(f"{data_dir}/layer_{layer}/max.safetensors")["activations"]
    topk_sum = load_file(f"{data_dir}/layer_{layer}/topk_sum.safetensors")["activations"]
    
    # Average across samples
    avg_topk_mean = topk_mean.mean(dim=0)
    avg_max = max_act.mean(dim=0)
    avg_topk_sum = topk_sum.mean(dim=0)
    
    # Top-20 by each method
    top_by_mean = set(avg_topk_mean.topk(20).indices.tolist())
    top_by_max = set(avg_max.topk(20).indices.tolist())
    top_by_sum = set(avg_topk_sum.topk(20).indices.tolist())
    
    print(f"Layer {layer} - Top-20 feature overlap:")
    print(f"  topk_mean ∩ max:      {len(top_by_mean & top_by_max)}/20")
    print(f"  topk_mean ∩ topk_sum: {len(top_by_mean & top_by_sum)}/20")
    print(f"  max ∩ topk_sum:       {len(top_by_max & top_by_sum)}/20")
    print(f"  all three:            {len(top_by_mean & top_by_max & top_by_sum)}/20")
    
    # Spearman rank correlation between methods
    from scipy.stats import spearmanr
    
    r_mean_max, _ = spearmanr(avg_topk_mean.numpy(), avg_max.numpy())
    r_mean_sum, _ = spearmanr(avg_topk_mean.numpy(), avg_topk_sum.numpy())
    r_max_sum, _ = spearmanr(avg_max.numpy(), avg_topk_sum.numpy())
    
    print(f"\n  Spearman correlations:")
    print(f"    topk_mean vs max:      {r_mean_max:.4f}")
    print(f"    topk_mean vs topk_sum: {r_mean_sum:.4f}")
    print(f"    max vs topk_sum:       {r_max_sum:.4f}")


# ==============================================================================
# 4) Download from HuggingFace and load
# ==============================================================================

def download_and_load(repo_id: str, local_dir: str = "./downloaded_activations"):
    """Download the dataset from HuggingFace."""
    # from huggingface_hub import snapshot_download
    
    # snapshot_download(
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     local_dir=local_dir,
    # )
    # print(f"Downloaded to {local_dir}")
    local_dir = "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/outputs/sae_activations"
    analyze_activations(local_dir)


if __name__ == "__main__":
    # Uncomment whichever you want to run:
    run_test()
    # analyze_activations()
    # compare_aggregation_methods()
    # download_and_load("YOUR_USERNAME/gemma3-4b-sae-activations-ecqa")
    pass