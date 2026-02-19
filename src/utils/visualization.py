from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

import plotly.graph_objects as go

from src.feature import Feature, create_features
from src.neuronpedia_client import NeuronpediaClient, build_sae_id
from src.utils.activations_utils import top_k_features_per_token

# ---------------------------------------------------------------------------
# Colorscale presets
# ---------------------------------------------------------------------------

# Red → white → blue.  Used when activation values span both signs.
DIVERGING_COLORSCALE: list[list] = [
    [0.0, "rgb(178,24,43)"],
    [0.5, "rgb(255,255,255)"],
    [1.0, "rgb(33,102,172)"],
]

# White → blue.  Used when all activation values are non-negative.
SEQUENTIAL_COLORSCALE: list[list] = [
    [0.0, "rgb(255,255,255)"],
    [1.0, "rgb(33,102,172)"],
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeatmapConfig:
    """Visual configuration shared across all Plotly activation heatmaps.

    Attributes
    ----------
    cell_size:
        Side-length in pixels for each heatmap cell.  Controls overall
        figure dimensions together with *min_width* / *min_height*.
    min_width / min_height:
        Lower bounds on figure size (pixels).
    colorbar_title:
        Label rendered next to the colour bar.
    colorscale:
        Explicit Plotly colorscale override.  When ``None`` (default),
        :class:`ActivationHeatmap` selects a diverging or sequential scale
        based on the data range.
    """

    cell_size: int = 20
    min_width: int = 400
    min_height: int = 200
    colorbar_title: str = "Activation"
    colorscale: str | list | None = None


# ---------------------------------------------------------------------------
# Heatmap class
# ---------------------------------------------------------------------------

class ActivationHeatmap:
    """Interactive Plotly heatmaps for SAE feature activations.

    Encapsulates shared rendering logic — colorscale selection, cell sizing,
    hover-text construction, and figure assembly — behind two public methods:

    * :meth:`plot_selected_features` – visualise a fixed set of features
      across all token positions.
    * :meth:`plot_topk_per_token` – visualise the top-K most-active features
      at every token position.
    """

    def __init__(self, config: HeatmapConfig | None = None) -> None:
        self.config = config or HeatmapConfig()

    # -- Shared helpers --------------------------------------------------------

    @staticmethod
    def _to_numpy(data, dtype=float) -> np.ndarray:
        """Convert a ``torch.Tensor`` or array-like to a NumPy array."""
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        return np.asarray(data, dtype=dtype)

    def _build_colorscale(
        self,
        values: np.ndarray,
    ) -> tuple[list, float, float]:
        """Select a colorscale and compute the symmetric z-range.

        Returns ``(colorscale, zmin, zmax)``.

        * If ``config.colorscale`` is set, it is always used with a symmetric
          range ``[-abs_max, abs_max]``.
        * Otherwise, all-non-negative data gets :data:`SEQUENTIAL_COLORSCALE`
          (``zmin=0``), while mixed-sign data gets :data:`DIVERGING_COLORSCALE`
          with a symmetric range.
        """
        if self.config.colorscale is not None:
            abs_max = max(float(np.abs(values).max()), 1e-6)
            return self.config.colorscale, -abs_max, abs_max

        vmin, vmax = float(values.min()), float(values.max())
        if vmin >= 0:
            return SEQUENTIAL_COLORSCALE, 0.0, max(vmax, 1e-6)

        abs_max = max(float(np.abs(values).max()), 1e-6)
        return DIVERGING_COLORSCALE, -abs_max, abs_max

    def _compute_dimensions(self, n_cols: int, n_rows: int) -> tuple[int, int]:
        """Return ``(width, height)`` in pixels scaled to the grid size."""
        cs = self.config.cell_size
        width = max(self.config.min_width, n_cols * cs + 200)
        height = max(self.config.min_height, n_rows * cs + 150)
        return width, height

    def _create_figure(
        self,
        *,
        z: np.ndarray,
        x_labels: Sequence,
        y_labels: Sequence[str],
        hover: list[list[str]],
        zmin: float,
        zmax: float,
        colorscale: list,
        title: str,
        x_title: str,
        y_title: str,
        x_axis_extra: dict | None = None,
    ) -> go.Figure:
        """Assemble a ``go.Figure`` containing a single ``Heatmap`` trace.

        Parameters
        ----------
        z:
            2-D array ``(n_rows, n_cols)`` of heat values.
        x_labels / y_labels:
            Tick labels for each axis.
        hover:
            Nested list of hover-text strings (same shape as *z*).
        zmin, zmax:
            Colour range limits.
        colorscale:
            Plotly colorscale definition.
        title:
            Figure title.
        x_title / y_title:
            Axis titles.
        x_axis_extra:
            Extra keys merged into the x-axis layout dict.  Use this to
            set ``type="category"`` or supply ``tickvals`` / ``ticktext``
            for integer-positioned axes.
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=list(x_labels),
                y=list(y_labels),
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                hoverinfo="text",
                text=hover,
                colorbar=dict(title=self.config.colorbar_title),
            )
        )

        width, height = self._compute_dimensions(z.shape[1], z.shape[0])

        xaxis: dict = dict(title=x_title, side="top")
        if x_axis_extra:
            xaxis.update(x_axis_extra)

        fig.update_layout(
            title=title,
            xaxis=xaxis,
            yaxis=dict(title=y_title, autorange="reversed", type="category"),
            width=width,
            height=height,
        )
        return fig

    # -- Public plotting methods -----------------------------------------------

    def plot_selected_features(
        self,
        activations: np.ndarray | torch.Tensor,
        tokens: list[str] | None = None,
        feature_indices: list[int] | None = None,
        labels: dict[int, str | None] | None = None,
        title: str = "SAE Feature Activations",
    ) -> go.Figure:
        """Heatmap of hand-picked feature activations across all tokens.

        Each row is a specific SAE feature; each column is a token position.
        Use this to compare how a curated set of features responds across the
        full input sequence.

        **Activation processing**

        1. The ``activations`` matrix (features × tokens) is mapped onto a
           symmetric diverging colorscale centred at zero: white = 0,
           blue = positive, red = negative.
        2. For each token column, features are ranked by descending activation
           (rank 1 = strongest).  The rank appears in the hover tooltip.

        Parameters
        ----------
        activations:
            2-D array ``(n_features, n_tokens)``.  A 1-D input is treated as
            a single feature.  Torch tensors are converted automatically.
        tokens:
            Token strings for the x-axis.  ``None`` → positional integers.
        feature_indices:
            Feature IDs used as y-axis labels.  Defaults to ``0 .. n-1``.
        labels:
            ``{feature_index: description}`` shown in the hover tooltip.
        title:
            Figure title.
        """
        activations = self._to_numpy(activations)
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        n_features, n_tokens = activations.shape
        labels = labels or {}

        x_labels = tokens if tokens is not None else list(range(n_tokens))
        raw_indices = (
            feature_indices if feature_indices is not None
            else list(range(n_features))
        )
        y_labels = [str(i) for i in raw_indices]

        ranks = self._rank_features_per_token(activations)
        hover = self._build_selected_features_hover(
            activations, ranks, x_labels, raw_indices, labels,
            has_tokens=tokens is not None,
        )
        colorscale, zmin, zmax = self._build_colorscale(activations)

        return self._create_figure(
            z=activations,
            x_labels=x_labels,
            y_labels=y_labels,
            hover=hover,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            title=title,
            x_title="Token" if tokens is not None else "Position",
            y_title="Feature Index",
            x_axis_extra=dict(type="category"),
        )

    def plot_topk_per_token(
        self,
        top_values: np.ndarray | torch.Tensor,
        top_indices: np.ndarray | torch.Tensor,
        tokens: list[str],
        labels: dict[int, str | None] | None = None,
        title: str = "Per-Token Top-K SAE Feature Activations",
    ) -> go.Figure:
        """Heatmap of the top-K strongest features at each token position.

        Each column is a token; each row is a rank slot (``#1`` = strongest
        activation for that token).  Because the winning features differ from
        token to token, the actual feature index is displayed in the hover
        tooltip rather than on the y-axis.

        **Activation processing**

        1. ``top_values`` ``(n_tokens, k)`` is transposed to ``(k, n_tokens)``
           so that rank increases downward.
        2. **Colorscale selection:**
           - All non-negative → sequential white→blue (``zmin=0``).
           - Mixed signs → symmetric diverging red→white→blue.
        3. Tokens are placed at integer x-positions and labelled with
           ``ticktext`` to prevent duplicate token strings from collapsing
           in categorical mode.

        Parameters
        ----------
        top_values:
            ``(n_tokens, k)`` — activation magnitudes from a top-k selection
            (e.g. :func:`~src.utils.activations_utils.top_k_features_per_token`).
        top_indices:
            ``(n_tokens, k)`` — corresponding feature indices.
        tokens:
            Token strings for the x-axis (length ``n_tokens``).
        labels:
            ``{feature_index: concept_label}`` shown in hover text.
        title:
            Figure title.
        """
        top_values = self._to_numpy(top_values)
        top_indices = self._to_numpy(top_indices, dtype=int)

        n_tokens, k = top_values.shape
        labels = labels or {}

        y_labels = [f"#{r + 1}" for r in range(k)]
        x_positions = list(range(n_tokens))

        hover = self._build_topk_hover(top_values, top_indices, tokens, labels, k)
        colorscale, zmin, zmax = self._build_colorscale(top_values)

        return self._create_figure(
            z=top_values.T,
            x_labels=x_positions,
            y_labels=y_labels,
            hover=hover,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            title=title,
            x_title="Token",
            y_title="Feature Rank",
            x_axis_extra=dict(tickvals=x_positions, ticktext=tokens),
        )

    # -- Hover-text builders ---------------------------------------------------

    @staticmethod
    def _rank_features_per_token(activations: np.ndarray) -> np.ndarray:
        """Per-token descending rank (rank 1 = highest activation)."""
        n_tokens = activations.shape[1]
        ranks = np.zeros_like(activations, dtype=int)
        for ti in range(n_tokens):
            ranks[:, ti] = np.argsort(np.argsort(-activations[:, ti])) + 1
        return ranks

    @staticmethod
    def _build_selected_features_hover(
        activations: np.ndarray,
        ranks: np.ndarray,
        x_labels: list,
        raw_indices: list[int],
        labels: dict[int, str | None],
        has_tokens: bool,
    ) -> list[list[str]]:
        """Hover text for :meth:`plot_selected_features`."""
        n_features, n_tokens = activations.shape
        hover: list[list[str]] = []
        for fi in range(n_features):
            feat_idx = raw_indices[fi]
            description = labels.get(feat_idx) or "N/A"
            row: list[str] = []
            for ti in range(n_tokens):
                token_str = x_labels[ti] if has_tokens else f"Position {ti}"
                row.append(
                    f"Token: {token_str}<br>"
                    f"Description: {description}<br>"
                    f"Activation: {activations[fi, ti]:.4f}<br>"
                    f"Rank: {ranks[fi, ti]}"
                )
            hover.append(row)
        return hover

    @staticmethod
    def _build_topk_hover(
        top_values: np.ndarray,
        top_indices: np.ndarray,
        tokens: list[str],
        labels: dict[int, str | None],
        k: int,
    ) -> list[list[str]]:
        """Hover text for :meth:`plot_topk_per_token`."""
        n_tokens = top_values.shape[0]
        hover: list[list[str]] = []
        for ri in range(k):
            row: list[str] = []
            for ti in range(n_tokens):
                feat_idx = int(top_indices[ti, ri])
                feat_label = labels.get(feat_idx) or "N/A"
                row.append(
                    f"Token: {tokens[ti]}<br>"
                    f"Activation: {top_values[ti, ri]:.4f}<br>"
                    f"Feature: {feat_idx}<br>"
                    f"Label: {feat_label}"
                )
            hover.append(row)
        return hover


# ---------------------------------------------------------------------------
# Standalone bar chart (matplotlib) — unrelated to the heatmap class
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------

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

    heatmap = ActivationHeatmap()
    fig = heatmap.plot_topk_per_token(
        per_token_vals,
        per_token_idxs,
        tokens=tokens,
        labels=labels,
        title=f"{np_model_id} Per-Token Top-{top_k} SAE Feature Activations",
    )
    fig.show()

    return feature_map
