import torch


def top_k_features(
    sae_acts: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average activations across token positions and return the top-*k* features.

    Returns ``(top_activations, top_feature_indices)``.
    """
    mean_acts = sae_acts.squeeze(0).mean(0)
    top_activations, top_feature_indices = mean_acts.topk(k)
    return top_activations, top_feature_indices


def reconstruction_metrics(
    reconstruction: torch.Tensor, original: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return MSE and fraction of variance unexplained (FVU).

    Both tensors are expected to have shape ``(batch, seq, d_model)``.
    The BOS token (position 0) is excluded from the computation.
    """
    reconstruction = reconstruction[:, 1:]
    original = original[:, 1:].float()
    mse = torch.mean((reconstruction - original) ** 2)
    fvu = mse / original.var()
    return {"mse": mse, "fvu": fvu}


def l0_sparsity(sae_acts: torch.Tensor) -> torch.Tensor:
    """Return per-token L0 count (number of active features)."""
    return (sae_acts > 1).sum(-1)
