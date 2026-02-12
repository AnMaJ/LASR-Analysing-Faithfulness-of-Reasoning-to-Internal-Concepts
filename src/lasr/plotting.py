from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_experiment_results(
    results_dir: str | Path,
    glob_pattern: str = "accuracy_report_*.csv",
) -> pd.DataFrame:
    """Load and concatenate all experiment result CSVs from *results_dir*.

    The first column (a duplicate numeric index) is dropped and the next
    column is renamed to ``class_name``.  The ``few_shot`` column is
    converted from string ``"True"``/``"False"`` to actual booleans.

    Raises ``FileNotFoundError`` when no files match *glob_pattern*.
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' in {results_dir}"
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        # Drop the first column (duplicate numeric index).
        df = df.iloc[:, 1:]
        # Rename the first remaining column to 'class_name'.
        df.rename(columns={df.columns[0]: "class_name"}, inplace=True)
        frames.append(df)

    combined = pd.concat(frames, sort=False, ignore_index=True)
    # Convert few_shot from string to bool.
    combined["few_shot"] = combined["few_shot"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    return combined


def extract_comparison_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-experiment metrics from a raw results DataFrame.

    Groups by ``(model_name, prompt_style, few_shot)`` and extracts
    ``avg_f1``, ``accuracy``, ``avg_precision``, and ``avg_recall``
    from the ``"macro avg"`` and ``"accuracy"`` summary rows.
    """
    rows: list[dict] = []
    for (model, style, fs), grp in df.groupby(
        ["model_name", "prompt_style", "few_shot"], sort=False
    ):
        macro = grp.loc[grp["class_name"] == "macro avg"]
        acc = grp.loc[grp["class_name"] == "accuracy"]

        rows.append(
            {
                "model_name": model,
                "prompt_style": style,
                "few_shot": fs,
                "avg_f1": macro["f1-score"].iloc[0] if len(macro) else np.nan,
                "accuracy": acc["f1-score"].iloc[0] if len(acc) else np.nan,
                "avg_precision": macro["precision"].iloc[0] if len(macro) else np.nan,
                "avg_recall": macro["recall"].iloc[0] if len(macro) else np.nan,
            }
        )
    return pd.DataFrame(rows)


_PROMPT_LABELS = {
    "chain_of_thought": "Chain of Thought",
    "one_word": "One-Word",
}
_SHOT_LABELS = {True: "Few-shot", False: "Zero-shot"}

_METRIC_TITLES = [
    ("avg_f1", "Avg F1-Score"),
    ("accuracy", "Accuracy"),
    ("avg_precision", "Avg Precision"),
    ("avg_recall", "Avg Recall"),
]


def _add_grouped_bar_subplot(
    fig: go.Figure,
    data: pd.DataFrame,
    group_col: str,
    label_col: str,
    metric_col: str,
    row: int,
    col: int,
    group_labels: dict,
    colors: list[str],
    show_legend: bool,
) -> None:
    """Add a grouped bar chart to a subplot cell."""
    groups = data[group_col].unique()
    for idx, group_val in enumerate(groups):
        subset = data[data[group_col] == group_val]
        fig.add_trace(
            go.Bar(
                x=subset[label_col].tolist(),
                y=subset[metric_col].tolist(),
                name=group_labels.get(group_val, str(group_val)),
                marker_color=colors[idx % len(colors)],
                legendgroup=str(group_val),
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )


def plot_experiment_comparison(
    metrics_df: pd.DataFrame,
    title: str = "Experiment Metrics Comparison",
) -> go.Figure:
    """Create a 4x2 subplot grid comparing experiment metrics.

    Left column groups by ``prompt_style`` (CoT vs One-Word).
    Right column groups by ``few_shot`` (Few-shot vs Zero-shot).
    Rows correspond to Avg F1, Accuracy, Avg Precision, Avg Recall.
    """
    subplot_titles: list[str] = []
    for _, metric_title in _METRIC_TITLES:
        subplot_titles.append(f"{metric_title} (by Prompt Style)")
        subplot_titles.append(f"{metric_title} (by Few-shot)")

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
    )

    prompt_colors = ["#636EFA", "#EF553B"]
    shot_colors = ["#00CC96", "#AB63FA"]

    df = metrics_df.copy()

    # Build x-axis labels for each column view.
    df["label_by_prompt"] = df.apply(
        lambda r: f"{r['model_name']}\n({_SHOT_LABELS[r['few_shot']]})", axis=1
    )
    df["label_by_shot"] = df.apply(
        lambda r: f"{r['model_name']}\n({_PROMPT_LABELS.get(r['prompt_style'], r['prompt_style'])})",
        axis=1,
    )

    for row_idx, (metric_col, _) in enumerate(_METRIC_TITLES, start=1):
        show_legend = row_idx == 1
        _add_grouped_bar_subplot(
            fig, df, "prompt_style", "label_by_prompt", metric_col,
            row=row_idx, col=1,
            group_labels=_PROMPT_LABELS, colors=prompt_colors,
            show_legend=show_legend,
        )
        _add_grouped_bar_subplot(
            fig, df, "few_shot", "label_by_shot", metric_col,
            row=row_idx, col=2,
            group_labels=_SHOT_LABELS, colors=shot_colors,
            show_legend=show_legend,
        )

    # Y-axis range [0, 1] for all subplots.
    for i in range(1, 9):
        fig.update_yaxes(range=[0, 1], row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

    fig.update_layout(
        title_text=title,
        barmode="group",
        height=1200,
        width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def plot_feature_activation_heatmap(
    activations: np.ndarray,
    tokens: list[str] | None = None,
    feature_indices: list[int] | None = None,
    title: str = "SAE Feature Activations",
    colorscale: str | list | None = None,
) -> go.Figure:
    """Create an interactive heatmap of SAE feature activations.

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
    title:
        Figure title.
    colorscale:
        Plotly colorscale.  Defaults to ``"RdBu_r"``.
    """
    # Handle torch tensors.
    if hasattr(activations, "detach"):
        activations = activations.detach().cpu().numpy()

    activations = np.asarray(activations, dtype=float)

    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    if colorscale is None:
        colorscale = "RdBu_r"

    n_features, n_tokens = activations.shape

    x_labels = tokens if tokens is not None else list(range(n_tokens))
    y_labels = (
        [str(i) for i in feature_indices]
        if feature_indices is not None
        else [str(i) for i in range(n_features)]
    )

    # Build custom hover text.
    hover: list[list[str]] = []
    for fi in range(n_features):
        row_hover: list[str] = []
        for ti in range(n_tokens):
            token_str = x_labels[ti] if tokens is not None else f"Position {ti}"
            row_hover.append(
                f"Token: {token_str}<br>"
                f"Feature Index: {y_labels[fi]}<br>"
                f"Activation: {activations[fi, ti]:.4f}"
            )
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=activations,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            hoverinfo="text",
            text=hover,
        )
    )

    # Dynamic sizing.
    width = max(600, n_tokens * 60 + 150)
    height = max(350, n_features * 40 + 200)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Token" if tokens is not None else "Position", type="category"),
        yaxis=dict(title="Feature Index", autorange="reversed"),
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

    Parameters
    ----------
    top_values:
        2-D array of shape ``(n_tokens, k)`` – activation values.
    top_indices:
        2-D array of shape ``(n_tokens, k)`` – feature indices.
    tokens:
        Token strings for the y-axis.
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

    x_labels = [f"#{r + 1}" for r in range(k)]

    # Build hover text: activation, feature label, feature number.
    hover: list[list[str]] = []
    for ti in range(n_tokens):
        row_hover: list[str] = []
        for ri in range(k):
            feat_idx = int(top_indices[ti, ri])
            feat_label = labels.get(feat_idx) or "N/A"
            row_hover.append(
                f"Token: {tokens[ti]}<br>"
                f"Activation: {top_values[ti, ri]:.4f}<br>"
                f"Feature: {feat_idx}<br>"
                f"Label: {feat_label}"
            )
        hover.append(row_hover)

    # Diverging colorscale: red (negative) → white (zero) → blue (positive).
    abs_max = max(float(np.abs(top_values).max()), 1e-6)
    colorscale = [
        [0.0, "rgb(178,24,43)"],
        [0.5, "rgb(255,255,255)"],
        [1.0, "rgb(33,102,172)"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=top_values,
            x=x_labels,
            y=tokens,
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
    width = max(400, k * cell_size + 200)
    height = max(400, n_tokens * cell_size + 150)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Feature Rank", type="category", side="top"),
        yaxis=dict(title="Token", autorange="reversed", type="category"),
        width=width,
        height=height,
    )
    return fig
