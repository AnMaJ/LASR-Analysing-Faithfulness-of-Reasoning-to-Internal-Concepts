"""
SAE Activation Statistics Plotter
===================================
For each layer and each aggregation method, compute per-feature statistics
(mean and variance across the 100 samples), then plot histograms.

Produces:
  - Per-layer histogram pairs (mean + variance) for each aggregation method
  - Combined grid plots showing all layers side by side
  - Summary CSV with per-layer statistics
  - Uploads everything to HuggingFace

Data layout expected (from collect_sae_activations.py):
  layer_{L}/topk_mean.safetensors  ->  {"activations": [100, 16384]}
  layer_{L}/max.safetensors        ->  {"activations": [100, 16384]}
  layer_{L}/topk_sum.safetensors   ->  {"activations": [100, 16384]}
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("WARNING: safetensors not installed. Use --demo mode or install it.")

try:
    from huggingface_hub import HfApi, login, create_repo
    HAS_HF = True
except ImportError:
    HAS_HF = False


# ==============================================================================
# Configuration
# ==============================================================================

class PlotConfig:
    # ── Paths (adjust the BASE if your repo checkout lives elsewhere) ──
    BASE = "/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/src"

    # Input data directory (output of collect_sae_activations.py)
    DATA_DIR = os.path.join(BASE, "sae_activations")

    # Output directory for plots
    PLOT_DIR = os.path.join(BASE, "sae_activation_plots")

    # HuggingFace repo
    OUTPUT_HF_REPO = "MansiJerry/LASR_proj_datasets"

    # Subfolder inside the HF repo for these plots
    HF_PATH_IN_REPO = "gemma3-4b-sae-activations-ecqa/plots"

    # Aggregation methods to process
    AGG_METHODS = {
        "topk_mean": "Top-k Mean (k=10)",
        "max":       "Max Activation",
        "topk_sum":  "Top-k Sum (k=10)",
    }

    # Plot style
    HIST_BINS = 100
    FIG_DPI = 150
    GRID_COLS = 6       # columns in the all-layers grid
    COLOR_MEAN = "#2196F3"      # blue
    COLOR_VARIANCE = "#FF5722"  # deep orange

    # Log file
    LOG_FILE = os.path.join(BASE, "sae_activation_plots", "plotting.log")


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("sae_plotter")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="w")
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ==============================================================================
# Data Loading
# ==============================================================================

def load_metadata(data_dir: str) -> dict:
    """Load metadata.json from the collection run."""
    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


def load_layer_activations(data_dir: str, layer_idx: int, agg_method: str) -> np.ndarray:
    """
    Load aggregated activations for one layer and one aggregation method.
    Returns numpy array of shape [num_samples, d_sae].
    """
    path = os.path.join(data_dir, f"layer_{layer_idx}", f"{agg_method}.safetensors")
    data = load_file(path)
    return data["activations"].numpy()  # [num_samples, d_sae]


def compute_feature_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given activations of shape [num_samples, d_sae], compute per-feature
    mean and variance across the sample dimension.

    Returns:
        means:     [d_sae] - mean activation per feature across samples
        variances: [d_sae] - variance of activation per feature across samples
    """
    means = activations.mean(axis=0)        # [d_sae]
    variances = activations.var(axis=0)      # [d_sae]
    return means, variances


# ==============================================================================
# Synthetic Data for Demo / Testing
# ==============================================================================

def generate_synthetic_data(num_layers: int = 34, num_samples: int = 100,
                            d_sae: int = 16384) -> Dict:
    """
    Generate synthetic activation data that mimics real SAE activation patterns:
    - Most features are near-zero (sparse)
    - A small fraction have moderate-to-high activations
    - Statistics shift across layers (early layers sparser, mid layers more active)
    """
    rng = np.random.RandomState(42)
    data = {}

    for layer in range(num_layers):
        # Layer-dependent sparsity: mid layers are more active
        depth_ratio = layer / (num_layers - 1)
        active_frac = 0.02 + 0.08 * np.sin(np.pi * depth_ratio)  # 2-10% active
        scale = 0.5 + 2.0 * np.sin(np.pi * depth_ratio)          # activation magnitude

        for agg in ["topk_mean", "max", "topk_sum"]:
            acts = np.zeros((num_samples, d_sae), dtype=np.float32)
            for s in range(num_samples):
                n_active = int(d_sae * active_frac * (0.8 + 0.4 * rng.rand()))
                active_idx = rng.choice(d_sae, n_active, replace=False)
                vals = rng.exponential(scale, size=n_active).astype(np.float32)

                if agg == "topk_sum":
                    vals *= 10  # sums are larger
                elif agg == "max":
                    vals *= 3   # max is larger than mean

                acts[s, active_idx] = vals

            data[(layer, agg)] = acts

    return data


# ==============================================================================
# Individual Layer Plots
# ==============================================================================

def plot_layer_histograms(
    means: np.ndarray,
    variances: np.ndarray,
    layer_idx: int,
    agg_method: str,
    agg_label: str,
    config: PlotConfig,
    save_dir: str,
) -> str:
    """
    Plot side-by-side histograms of per-feature mean and variance for one layer.
    Returns path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter out exact zeros for cleaner histograms (most features are sparse)
    nonzero_means = means[means > 0]
    nonzero_vars = variances[variances > 0]

    total_features = len(means)
    nonzero_mean_count = len(nonzero_means)
    nonzero_var_count = len(nonzero_vars)
    zero_frac_mean = 1.0 - nonzero_mean_count / total_features
    zero_frac_var = 1.0 - nonzero_var_count / total_features

    # --- Mean Histogram ---
    ax = axes[0]
    if len(nonzero_means) > 0:
        ax.hist(nonzero_means, bins=config.HIST_BINS, color=config.COLOR_MEAN,
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(nonzero_means), color="black", linestyle="--",
                   linewidth=1.2, label=f"Median: {np.median(nonzero_means):.4f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "All features zero", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")

    ax.set_title(f"Per-Feature Mean Activation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Activation Value", fontsize=10)
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.text(0.97, 0.95,
            f"Non-zero: {nonzero_mean_count:,}/{total_features:,}\n"
            f"Zero frac: {zero_frac_mean:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Variance Histogram ---
    ax = axes[1]
    if len(nonzero_vars) > 0:
        ax.hist(nonzero_vars, bins=config.HIST_BINS, color=config.COLOR_VARIANCE,
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(nonzero_vars), color="black", linestyle="--",
                   linewidth=1.2, label=f"Median: {np.median(nonzero_vars):.4f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "All features zero", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")

    ax.set_title(f"Per-Feature Variance of Activation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Variance", fontsize=10)
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.text(0.97, 0.95,
            f"Non-zero: {nonzero_var_count:,}/{total_features:,}\n"
            f"Zero frac: {zero_frac_var:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(f"Layer {layer_idx} — {agg_label}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    filename = f"layer_{layer_idx}_{agg_method}_histograms.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=config.FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    return filepath


# ==============================================================================
# All-Layers Grid Plot
# ==============================================================================

def plot_all_layers_grid(
    all_means: Dict[int, np.ndarray],
    all_variances: Dict[int, np.ndarray],
    agg_method: str,
    agg_label: str,
    config: PlotConfig,
    save_dir: str,
) -> str:
    """
    Create a large grid figure with one row per pair (mean, variance)
    and columns for each layer. Two sub-grids stacked:
      Top grid:    Mean histograms for all layers
      Bottom grid: Variance histograms for all layers
    """
    layers = sorted(all_means.keys())
    num_layers = len(layers)
    ncols = config.GRID_COLS
    nrows = int(np.ceil(num_layers / ncols))

    fig_height_per_row = 2.8
    fig_width = ncols * 3.2

    # --- Mean grid ---
    fig_mean, axes_mean = plt.subplots(nrows, ncols,
                                        figsize=(fig_width, nrows * fig_height_per_row))
    if nrows == 1:
        axes_mean = axes_mean.reshape(1, -1)

    for idx, layer in enumerate(layers):
        row, col = divmod(idx, ncols)
        ax = axes_mean[row, col]
        means = all_means[layer]
        nonzero = means[means > 0]

        if len(nonzero) > 0:
            ax.hist(nonzero, bins=50, color=config.COLOR_MEAN,
                    alpha=0.85, edgecolor="none")
            median_val = np.median(nonzero)
            ax.axvline(median_val, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"L{layer}", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    # Hide empty subplots
    for idx in range(num_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes_mean[row, col].set_visible(False)

    fig_mean.suptitle(f"Per-Feature MEAN — {agg_label} — All Layers\n"
                      f"(non-zero features only, {config.HIST_BINS} bins on individual plots)",
                      fontsize=13, fontweight="bold", y=1.02)
    fig_mean.tight_layout()

    mean_path = os.path.join(save_dir, f"grid_all_layers_{agg_method}_mean.png")
    fig_mean.savefig(mean_path, dpi=config.FIG_DPI, bbox_inches="tight",
                     facecolor="white", edgecolor="none")
    plt.close(fig_mean)

    # --- Variance grid ---
    fig_var, axes_var = plt.subplots(nrows, ncols,
                                      figsize=(fig_width, nrows * fig_height_per_row))
    if nrows == 1:
        axes_var = axes_var.reshape(1, -1)

    for idx, layer in enumerate(layers):
        row, col = divmod(idx, ncols)
        ax = axes_var[row, col]
        variances = all_variances[layer]
        nonzero = variances[variances > 0]

        if len(nonzero) > 0:
            ax.hist(nonzero, bins=50, color=config.COLOR_VARIANCE,
                    alpha=0.85, edgecolor="none")
            median_val = np.median(nonzero)
            ax.axvline(median_val, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"L{layer}", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    for idx in range(num_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes_var[row, col].set_visible(False)

    fig_var.suptitle(f"Per-Feature VARIANCE — {agg_label} — All Layers\n"
                     f"(non-zero features only)",
                     fontsize=13, fontweight="bold", y=1.02)
    fig_var.tight_layout()

    var_path = os.path.join(save_dir, f"grid_all_layers_{agg_method}_variance.png")
    fig_var.savefig(var_path, dpi=config.FIG_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig_var)

    return mean_path, var_path


# ==============================================================================
# Summary Line Plots (across-layer trends)
# ==============================================================================

def plot_across_layer_trends(
    summary: Dict[str, Dict[int, dict]],
    config: PlotConfig,
    save_dir: str,
) -> str:
    """
    Plot how global mean and global variance evolve across layers,
    one subplot per aggregation method.

    summary structure: {agg_method: {layer: {"global_mean": ..., "global_var": ...,
                        "median_nonzero_mean": ..., "median_nonzero_var": ...,
                        "nonzero_frac": ...}}}
    """
    agg_methods = list(summary.keys())
    n_agg = len(agg_methods)

    fig, axes = plt.subplots(n_agg, 3, figsize=(18, 4.5 * n_agg), squeeze=False)

    for row, agg in enumerate(agg_methods):
        layers = sorted(summary[agg].keys())
        label = config.AGG_METHODS[agg]

        global_means = [summary[agg][l]["global_mean"] for l in layers]
        global_vars = [summary[agg][l]["global_var"] for l in layers]
        median_nz_means = [summary[agg][l]["median_nonzero_mean"] for l in layers]
        median_nz_vars = [summary[agg][l]["median_nonzero_var"] for l in layers]
        nonzero_fracs = [summary[agg][l]["nonzero_frac"] for l in layers]

        # Col 0: Global mean across layers
        ax = axes[row, 0]
        ax.plot(layers, global_means, "-o", color=config.COLOR_MEAN,
                markersize=4, linewidth=1.5, label="Global Mean")
        ax.plot(layers, median_nz_means, "--s", color="#1565C0",
                markersize=3, linewidth=1.0, alpha=0.7, label="Median (non-zero)")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Mean Activation", fontsize=10)
        ax.set_title(f"{label}\nPer-Feature Mean across Layers", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 1: Global variance across layers
        ax = axes[row, 1]
        ax.plot(layers, global_vars, "-o", color=config.COLOR_VARIANCE,
                markersize=4, linewidth=1.5, label="Global Variance")
        ax.plot(layers, median_nz_vars, "--s", color="#BF360C",
                markersize=3, linewidth=1.0, alpha=0.7, label="Median (non-zero)")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Variance", fontsize=10)
        ax.set_title(f"{label}\nPer-Feature Variance across Layers", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 2: Non-zero fraction across layers
        ax = axes[row, 2]
        ax.plot(layers, nonzero_fracs, "-o", color="#4CAF50",
                markersize=4, linewidth=1.5)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Non-zero Feature Fraction", fontsize=10)
        ax.set_title(f"{label}\nFeature Sparsity across Layers", fontsize=11, fontweight="bold")
        ax.set_ylim(0, min(1.0, max(nonzero_fracs) * 1.3))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Activation Statistics Across Layers — All Aggregation Methods",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(save_dir, "across_layer_trends.png")
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


# ==============================================================================
# Summary CSV
# ==============================================================================

def save_summary_csv(summary: Dict, save_dir: str) -> str:
    """Save a CSV with per-layer, per-aggregation statistics."""
    rows = []
    for agg, layer_stats in summary.items():
        for layer, stats in sorted(layer_stats.items()):
            rows.append({
                "aggregation_method": agg,
                "layer": layer,
                **stats,
            })

    # Write CSV manually (no pandas dependency required)
    if rows:
        keys = list(rows[0].keys())
        csv_path = os.path.join(save_dir, "activation_statistics_summary.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        return csv_path
    return None


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_plotting(config: PlotConfig, logger: logging.Logger, demo: bool = False):
    """
    Main function: load activations, compute statistics, generate all plots.

    Args:
        config: PlotConfig instance
        logger: Logger
        demo: If True, use synthetic data instead of loading from disk
    """
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    # --- Determine layers ---
    if demo:
        num_layers = 34
        layers = list(range(num_layers))
        logger.info("DEMO MODE: Using synthetic data (34 layers, 100 samples, 16384 features)")
        synth_data = generate_synthetic_data(num_layers=num_layers)
    else:
        meta = load_metadata(config.DATA_DIR)
        num_layers = meta["num_layers"]
        layers = list(range(num_layers))
        logger.info(f"Loaded metadata: {num_layers} layers, {meta['num_samples']} samples")
        synth_data = None

    # --- Process each aggregation method ---
    summary = {}  # {agg_method: {layer: stats_dict}}
    all_plot_paths = []

    for agg_method, agg_label in config.AGG_METHODS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing aggregation method: {agg_label} ({agg_method})")
        logger.info(f"{'='*60}")

        agg_dir = os.path.join(config.PLOT_DIR, agg_method)
        os.makedirs(agg_dir, exist_ok=True)

        all_means_dict = {}
        all_vars_dict = {}
        summary[agg_method] = {}

        for layer_idx in layers:
            # Load or generate data
            if demo:
                activations = synth_data[(layer_idx, agg_method)]
            else:
                try:
                    activations = load_layer_activations(
                        config.DATA_DIR, layer_idx, agg_method)
                except Exception as e:
                    logger.warning(f"  Skipping layer {layer_idx}: {e}")
                    continue

            # Compute per-feature statistics
            means, variances = compute_feature_statistics(activations)
            all_means_dict[layer_idx] = means
            all_vars_dict[layer_idx] = variances

            # Compute summary stats
            nonzero_mask_m = means > 0
            nonzero_mask_v = variances > 0
            total = len(means)

            stats = {
                "global_mean": float(means.mean()),
                "global_var": float(variances.mean()),
                "median_nonzero_mean": float(np.median(means[nonzero_mask_m]))
                    if nonzero_mask_m.any() else 0.0,
                "median_nonzero_var": float(np.median(variances[nonzero_mask_v]))
                    if nonzero_mask_v.any() else 0.0,
                "nonzero_frac": float(nonzero_mask_m.sum() / total),
                "max_mean": float(means.max()),
                "max_var": float(variances.max()),
                "num_nonzero_features": int(nonzero_mask_m.sum()),
                "total_features": total,
            }
            summary[agg_method][layer_idx] = stats

            # Individual layer histogram
            path = plot_layer_histograms(
                means, variances, layer_idx, agg_method, agg_label,
                config, agg_dir)
            all_plot_paths.append(path)

            if (layer_idx + 1) % 5 == 0 or layer_idx == layers[-1]:
                logger.info(f"  Layer {layer_idx}: nonzero_frac={stats['nonzero_frac']:.3f}, "
                           f"global_mean={stats['global_mean']:.6f}, "
                           f"global_var={stats['global_var']:.6f}")

        # Grid plots for all layers
        logger.info(f"  Generating all-layers grid for {agg_method}...")
        mean_grid, var_grid = plot_all_layers_grid(
            all_means_dict, all_vars_dict, agg_method, agg_label,
            config, config.PLOT_DIR)
        all_plot_paths.extend([mean_grid, var_grid])
        logger.info(f"  Grid plots saved: {mean_grid}, {var_grid}")

    # --- Cross-method trend plots ---
    logger.info("\nGenerating across-layer trend plots...")
    trend_path = plot_across_layer_trends(summary, config, config.PLOT_DIR)
    all_plot_paths.append(trend_path)
    logger.info(f"  Trend plot saved: {trend_path}")

    # --- Summary CSV ---
    csv_path = save_summary_csv(summary, config.PLOT_DIR)
    if csv_path:
        logger.info(f"  Summary CSV saved: {csv_path}")

    # --- Summary JSON (for programmatic use) ---
    json_path = os.path.join(config.PLOT_DIR, "activation_statistics_summary.json")
    with open(json_path, "w") as f:
        # Convert int keys to strings for JSON
        json_summary = {}
        for agg, layer_stats in summary.items():
            json_summary[agg] = {str(k): v for k, v in layer_stats.items()}
        json.dump(json_summary, f, indent=2)
    logger.info(f"  Summary JSON saved: {json_path}")

    logger.info(f"\nTotal plots generated: {len(all_plot_paths)}")
    logger.info(f"All outputs in: {config.PLOT_DIR}")

    return config.PLOT_DIR, all_plot_paths


# ==============================================================================
# HuggingFace Upload
# ==============================================================================

def upload_plots_to_huggingface(plot_dir: str, repo_id: str, logger: logging.Logger):
    """Upload plot directory to HuggingFace under a 'plots/' prefix."""
    if not HAS_HF:
        logger.warning("huggingface_hub not installed, skipping upload.")
        return

    logger.info(f"Uploading plots to HuggingFace repo: {repo_id}")
    api = HfApi()

    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.warning(f"  Repo note: {e}")

    api.upload_folder(
        folder_path=plot_dir,
        path_in_repo=PlotConfig.HF_PATH_IN_REPO,
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("  Upload complete!")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot SAE activation statistics")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data for demonstration")
    parser.add_argument("--data-dir", type=str, default="/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/outputs/sae_activations",
                        help="Override data directory")
    parser.add_argument("--plot-dir", type=str, default="/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/outputs/plots",
                        help="Override plot output directory")
    parser.add_argument("--upload", action="store_true",
                        help="Upload plots to HuggingFace")
    parser.add_argument("--repo", type=str, default="MansiJerry/LASR_proj_datasets",
                        help="HuggingFace repo ID")
    args = parser.parse_args()

    config = PlotConfig()
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.plot_dir:
        config.PLOT_DIR = args.plot_dir
    if args.repo:
        config.OUTPUT_HF_REPO = args.repo

    logger = setup_logging(config.LOG_FILE)
    logger.info("=" * 80)
    logger.info("SAE Activation Statistics Plotter")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Demo mode: {args.demo}")
    logger.info("=" * 80)

    start = time.time()
    plot_dir, paths = run_plotting(config, logger, demo=args.demo)
    elapsed = time.time() - start
    logger.info(f"\nPlotting complete in {elapsed:.1f}s")

    if args.upload:
        upload_plots_to_huggingface(plot_dir, config.OUTPUT_HF_REPO, logger)

    logger.info("Done!")