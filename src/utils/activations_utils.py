from __future__ import annotations

import torch


def top_k_features_per_token(
    activations: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k feature values and indices for each token.

    Parameters
    ----------
    activations:
        2-D tensor of shape ``(n_tokens, n_features)``.
    k:
        Number of top features to return per token.

    Returns
    -------
    values:
        Shape ``(n_tokens, k)`` — activation values, descending.
    indices:
        Shape ``(n_tokens, k)`` — feature indices.
    """
    return activations.topk(k, dim=1)
