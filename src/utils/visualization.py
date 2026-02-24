from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import plotly.graph_objects as go

from src.feature import Feature, create_features
from src.neuronpedia_client import NeuronpediaClient, build_sae_id
from src.utils.activations_utils import top_k_features_per_token


def plot_feature_magnitudes(
    aggregated: torch.Tensor,
    label: str | None = None,
    top_k: int | None = None,
) -> None:
    """Plot a bar chart of feature magnitudes for a single aggregated vector.

    Produces one figure per call.

    Args:
        aggregated: 1-D tensor of shape ``(num_features,)``.
        label: Optional title for the plot.
        top_k: If set, only show the ``top_k`` features with highest
            magnitude to keep the plot readable.
    """
    vec = aggregated.detach().cpu().float()
    num_features = vec.shape[0]
    feature_indices = torch.arange(num_features)

    if top_k is not None:
        _, top_idx = vec.topk(top_k)
        top_idx = top_idx.sort().values
        feature_indices = top_idx
        vec = vec[top_idx]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(feature_indices.numpy(), vec.numpy(), width=1.0)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Magnitude")
    if label is not None:
        ax.set_title(label)
    plt.tight_layout()
    plt.show()


def plot_feature_activation_heatmap(
    activations: np.ndarray,
    tokens: list[str] | None = None,
    feature_indices: list[int] | None = None,
    labels: dict[int, str | None] | None = None,
    title: str = "SAE Feature Activations",
    colorscale: str | list | None = None,
) -> go.Figure:
    """Create an interactive heatmap of SAE feature activations.

    Renders as a wide horizontal plot with tokens on the x-axis (top)
    and features on the y-axis (feature #1 at top).

    Parameters
    ----------
    activations:
        2-D array of shape ``(n_features, n_tokens)`` when *tokens* is
        provided, or a 1-D array for a single feature.
    tokens:
        Token strings for the x-axis.  When ``None`` the x-axis shows
        positional indices and the input is treated as a single feature.
    feature_indices:
        Custom y-axis labels (feature IDs).
    labels:
        Optional ``{feature_index: description}`` mapping from Neuronpedia.
    title:
        Figure title.
    colorscale:
        Plotly colorscale.  Defaults to a diverging red-white-blue scale.
    """
    # Handle torch tensors.
    if hasattr(activations, "detach"):
        activations = activations.detach().cpu().numpy()

    activations = np.asarray(activations, dtype=float)

    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    n_features, n_tokens = activations.shape

    # Diverging colorscale matching plot_per_token_topk_heatmap.
    if colorscale is None:
        colorscale = [
            [0.0, "rgb(178,24,43)"],
            [0.5, "rgb(255,255,255)"],
            [1.0, "rgb(33,102,172)"],
        ]
    abs_max = max(float(np.abs(activations).max()), 1e-6)

    x_labels = tokens if tokens is not None else list(range(n_tokens))
    y_labels = (
        [str(i) for i in feature_indices]
        if feature_indices is not None
        else [str(i) for i in range(n_features)]
    )

    if labels is None:
        labels = {}

    # Compute per-token ranks: for each token, rank features by descending
    # activation.  rank 1 = highest activation for that token.
    ranks = np.zeros_like(activations, dtype=int)
    for ti in range(n_tokens):
        # argsort of argsort gives rank; negate for descending order.
        ranks[:, ti] = np.argsort(np.argsort(-activations[:, ti])) + 1

    # Build custom hover text.
    raw_indices = feature_indices if feature_indices is not None else list(range(n_features))
    hover: list[list[str]] = []
    for fi in range(n_features):
        feat_idx = raw_indices[fi]
        description = labels.get(feat_idx) or "N/A"
        row_hover: list[str] = []
        for ti in range(n_tokens):
            token_str = x_labels[ti] if tokens is not None else f"Position {ti}"
            row_hover.append(
                f"Token: {token_str}<br>"
                f"Description: {description}<br>"
                f"Activation: {activations[fi, ti]:.4f}<br>"
                f"Rank: {ranks[fi, ti]}"
            )
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=activations,
            x=x_labels,
            y=y_labels,
            zmin=-abs_max,
            zmax=abs_max,
            colorscale=colorscale,
            hoverinfo="text",
            text=hover,
            colorbar=dict(title="Activation"),
        )
    )

    # Small square cells – 20px per cell.
    cell_size = 20
    width = max(400, n_tokens * cell_size + 200)
    height = max(200, n_features * cell_size + 150)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Token" if tokens is not None else "Position",
            type="category",
            side="top",
        ),
        yaxis=dict(title="Feature Index", autorange="reversed", type="category"),
        width=width,
        height=height,
    )
    return fig


def plot_per_token_topk_heatmap(
    top_values: np.ndarray,
    top_indices: np.ndarray,
    tokens: list[str],
    labels: dict[int, str | None] | None = None,
    title: str = "Per-Token Top-K SAE Feature Activations",
) -> go.Figure:
    """Heatmap showing the top-K SAE feature activations per token.

    Renders as a wide horizontal plot with tokens on the x-axis and
    feature ranks on the y-axis (rank #1 at top).

    Parameters
    ----------
    top_values:
        2-D array of shape ``(n_tokens, k)`` -- activation values.
    top_indices:
        2-D array of shape ``(n_tokens, k)`` -- feature indices.
    tokens:
        Token strings for the x-axis.
    labels:
        Optional ``{feature_index: concept_label}`` mapping from Neuronpedia.
    title:
        Figure title.
    """
    if hasattr(top_values, "detach"):
        top_values = top_values.detach().cpu().numpy()
    if hasattr(top_indices, "detach"):
        top_indices = top_indices.detach().cpu().numpy()

    top_values = np.asarray(top_values, dtype=float)
    top_indices = np.asarray(top_indices, dtype=int)

    n_tokens, k = top_values.shape
    if labels is None:
        labels = {}

    y_labels = [f"#{r + 1}" for r in range(k)]

    # Build hover text (shape: k x n_tokens after transpose).
    hover: list[list[str]] = []
    for ri in range(k):
        row_hover: list[str] = []
        for ti in range(n_tokens):
            feat_idx = int(top_indices[ti, ri])
            feat_label = labels.get(feat_idx) or "N/A"
            row_hover.append(
                f"Token: {tokens[ti]}<br>"
                f"Activation: {top_values[ti, ri]:.4f}<br>"
                f"Feature: {feat_idx}<br>"
                f"Label: {feat_label}"
            )
        hover.append(row_hover)

    # Choose colorscale based on whether data contains negative values.
    vmin = float(top_values.min())
    vmax = float(top_values.max())
    if vmin >= 0:
        # Sequential colorscale: white (zero) -> blue (max).
        colorscale = [
            [0.0, "rgb(255,255,255)"],
            [1.0, "rgb(33,102,172)"],
        ]
        zmin = 0.0
        zmax = max(vmax, 1e-6)
    else:
        # Diverging colorscale: red (negative) -> white (zero) -> blue (positive).
        abs_max = max(float(np.abs(top_values).max()), 1e-6)
        colorscale = [
            [0.0, "rgb(178,24,43)"],
            [0.5, "rgb(255,255,255)"],
            [1.0, "rgb(33,102,172)"],
        ]
        zmin = -abs_max
        zmax = abs_max

    # Use integer positions for the x-axis so that duplicate token strings
    # do not collapse onto the same category position.
    x_positions = list(range(n_tokens))

    # Transpose: z becomes (k, n_tokens).
    fig = go.Figure(
        data=go.Heatmap(
            z=top_values.T,
            x=x_positions,
            y=y_labels,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            hoverinfo="text",
            text=hover,
            colorbar=dict(title="Activation"),
        )
    )

    # Small square cells – 20px per cell.
    cell_size = 20
    width = max(400, n_tokens * cell_size + 200)
    height = max(200, k * cell_size + 150)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Token",
            side="top",
            tickvals=x_positions,
            ticktext=tokens,
        ),
        yaxis=dict(title="Feature Rank", autorange="reversed", type="category"),
        width=width,
        height=height,
    )
    return fig


def plot_token_activation_ridgeplot(
    activations: torch.Tensor | np.ndarray,
    tokens: list[str],
    n_tokens: int = 50,
    max_tokens: int = 200,
    seed: int = 42,
    title: str | None = None,
    palette: str = "viridis",
    bw_adjust: float = 0.8,
) -> plt.Figure:
    """Ridgeplot (joy plot) of per-token activation distributions.

    Each row shows the KDE density of all activation values for one token.
    Rows are sorted by descending peak density (tokens whose distribution
    has the tallest peak appear at the top).  Row colour is mapped to the
    peak density via *palette*, so colour encodes how many features
    concentrate in the densest value bucket.

    Parameters
    ----------
    activations:
        2-D tensor/array of shape ``(n_tokens, n_features)``.
    tokens:
        Token label strings (length must match ``activations`` dim 0).
    n_tokens:
        Number of tokens to randomly sample.  Set to ``0`` to include all
        tokens (capped at *max_tokens*).
    max_tokens:
        Upper bound when *n_tokens* is ``0``.
    seed:
        Random seed for reproducible sampling.
    title:
        Optional figure title.  When ``None`` a default is generated.
    palette:
        Matplotlib colormap name used for mapping peak density to colour.
    bw_adjust:
        Bandwidth adjustment passed to :func:`seaborn.kdeplot`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.stats import gaussian_kde

    if hasattr(activations, "detach"):
        activations = activations.detach().cpu().numpy()
    activations = np.asarray(activations, dtype=float)

    n_total = activations.shape[0]

    if n_tokens == 0:
        n_sample = min(n_total, max_tokens)
    else:
        n_sample = min(n_tokens, n_total)

    rng = np.random.RandomState(seed)
    sampled_indices = list(
        rng.choice(n_total, size=n_sample, replace=False))

    peak_densities: dict[int, float] = {}
    for idx in sampled_indices:
        vals = activations[idx]
        try:
            kde = gaussian_kde(vals, bw_method=bw_adjust)
            x_grid = np.linspace(vals.min(), vals.max(), 512)
            peak_densities[idx] = float(kde(x_grid).max())
        except np.linalg.LinAlgError:
            peak_densities[idx] = 0.0

    sorted_indices = sorted(
        sampled_indices, key=lambda i: peak_densities[i], reverse=True)

    rows: list[dict] = []
    for idx in sorted_indices:
        vals = activations[idx]
        label = f"{idx}: {tokens[idx]}"
        for v in vals:
            rows.append({"token": label, "value": float(v)})

    df = pd.DataFrame(rows)

    token_order = [f"{idx}: {tokens[idx]}" for idx in sorted_indices]
    df["token"] = pd.Categorical(
        df["token"], categories=token_order, ordered=True)

    sorted_peaks = [peak_densities[i] for i in sorted_indices]
    peak_min, peak_max = min(sorted_peaks), max(sorted_peaks)
    cmap = plt.colormaps[palette]
    norm = plt.Normalize(vmin=peak_min, vmax=peak_max)
    row_colors = {
        label: cmap(norm(peak))
        for label, peak in zip(token_order, sorted_peaks)
    }

    g = sns.FacetGrid(
        df, row="token", hue="token",
        aspect=15, height=0.5,
        palette=row_colors,
    )
    g.map(sns.kdeplot, "value",
          fill=True, alpha=0.5, linewidth=1.0, bw_adjust=bw_adjust)
    g.map(sns.kdeplot, "value",
          fill=False, color="k", linewidth=0.3, bw_adjust=bw_adjust)

    g.figure.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    for ax, label in zip(g.axes.flat, token_order):
        ax.text(
            -0.02, 0.1, label, fontsize=7, ha="right",
            transform=ax.transAxes, fontweight="bold",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = g.figure.colorbar(
        sm, ax=list(g.axes.flat), location="right",
        shrink=0.5, aspect=30, pad=0.02,
    )
    cbar.set_label("Peak density (feature count concentration)", fontsize=9)

    if title is None:
        title = (f"Activation Distribution per Token "
                 f"(n={n_sample} of {n_total})")
    g.figure.suptitle(title, fontsize=12, y=1.01)

    return g.figure


def summarize_latents(
    sae_activations: torch.Tensor,
    tokens: list[str],
    sae_config,
    model_name: str,
    top_k: int = 10,
    print_first_n: int = 3,
    full_sae_activations: torch.Tensor | None = None,
    all_tokens: list[str] | None = None,
) -> dict[int, Feature]:
    """Summarize top-k SAE features per token and plot a heatmap.

    Parameters
    ----------
    sae_activations:
        SAE activation tensor of shape ``(n_tokens, n_features)`` for the
        slice to analyse (e.g. generation-only tokens).
    tokens:
        List of token strings (same length as ``sae_activations`` dim 0).
    sae_config:
        A :class:`SAEConfig` instance (used to build the Neuronpedia SAE ID).
    model_name:
        Full or short model identifier (e.g. ``"google/gemma-3-27b-it"``).
    top_k:
        Number of top features to summarize per token.
    print_first_n:
        Number of leading tokens for which to print detailed feature info.
    full_sae_activations:
        Optional full-sequence SAE activations ``(n_all_tokens, n_features)``.
        When provided together with *all_tokens*, the returned Feature objects
        will include per-token activations from the full sequence (enabling
        :meth:`Feature.inspect`, :meth:`Feature.top_tokens`, etc.).
    all_tokens:
        Full-sequence token strings.  Required when *full_sae_activations* is
        given.

    Returns
    -------
    dict[int, Feature]
        Mapping from feature index to :class:`Feature` object for every
        unique feature that appears in the top-k across all tokens.
    """
    per_token_vals, per_token_idxs = top_k_features_per_token(
        sae_activations, k=top_k)

    # Print detailed info for the first few tokens.
    if print_first_n > 0:
        print(f"\nTop {top_k} SAE features per token (first {print_first_n} tokens shown):")
        for t in range(min(print_first_n, len(tokens))):
            print(f"\nToken '{tokens[t]}':")
            for r in range(top_k):
                idx = int(per_token_idxs[t, r])
                val = float(per_token_vals[t, r])
                print(f"  Rank {r+1}  feature {idx:>5d}  activation = {val:.4f}")

    # Build Neuronpedia client and fetch Feature objects for unique features.
    np_model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    np_sae_id = build_sae_id(sae_config)
    client = NeuronpediaClient(model_id=np_model_id, sae_id=np_sae_id)

    unique_indices = sorted(
        set(per_token_idxs.cpu().numpy().ravel().tolist()))

    # Create Feature objects — one per unique feature.
    if full_sae_activations is not None and all_tokens is not None:
        # Per-token mode: Feature objects carry full-sequence activations.
        features = create_features(full_sae_activations, unique_indices, all_tokens)
        for f in features:
            f.fetch_details(client)
        feature_map = {f.feature_idx: f for f in features}
    else:
        # Aggregated mode: Feature objects carry a single strength value.
        feature_map = {}
        for idx in unique_indices:
            mask = per_token_idxs == idx
            strength = float(per_token_vals[mask].mean()) if mask.any() else 0.0
            feat = Feature(feature_idx=idx, strength=strength)
            feat.fetch_details(client)
            feature_map[idx] = feat

    print(f"\nCreated {len(feature_map)} Feature objects")
    first_key = unique_indices[0]
    print(f"Example: {feature_map[first_key]!r}")

    # Build labels dict for the heatmap.
    labels = {
        idx: feat.description
        for idx, feat in feature_map.items()
    }

    fig = plot_per_token_topk_heatmap(
        per_token_vals,
        per_token_idxs,
        tokens=tokens,
        labels=labels,
        title=f"{np_model_id} Per-Token Top-{top_k} SAE Feature Activations",
    )
    fig.show()

    return feature_map
