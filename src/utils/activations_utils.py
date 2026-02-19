import torch


def top_k_features_per_token(
    sae_acts: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-*k* feature activations for each token independently.

    Parameters
    ----------
    sae_acts:
        Tensor of shape ``(n_tokens, n_features)``.

    Returns ``(top_values, top_indices)`` each of shape ``(n_tokens, k)``.
    """
    top_values, top_indices = sae_acts.topk(k, dim=-1)
    return top_values, top_indices
