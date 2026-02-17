from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import torch


def plot_feature_magnitudes(
    aggregated: torch.Tensor,
    label: Optional[str] = None,
    top_k: Optional[int] = None,
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
