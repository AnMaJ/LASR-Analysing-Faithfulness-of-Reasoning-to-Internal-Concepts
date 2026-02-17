from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class DenoisingMethod(Enum):
    CONTINUOUS_TFIDF = "continuous_tfidf"


@dataclass
class DenoisingConfig:
    method: DenoisingMethod = DenoisingMethod.CONTINUOUS_TFIDF


def continuous_tfidf(activations: torch.Tensor) -> torch.Tensor:
    """Apply continuous TF-IDF weighting to SAE activations.

    Args:
        activations: Tensor of shape ``(num_tokens, num_features)``.

    Returns:
        Tensor of the same shape with TF-IDF weighting applied.
    """
    num_tokens = activations.shape[0]
    tf = activations
    df = (activations > 10).sum(dim=0)  # (num_features,)
    idf = torch.log(num_tokens / (1 + df))  # (num_features,)
    return tf * idf, idf


def denoise(activations: torch.Tensor, config: DenoisingConfig) -> torch.Tensor:
    """Denoise SAE activations using the method specified in *config*.

    Args:
        activations: Tensor of shape ``(1, num_tokens, num_features)``
            (batch dimension from the SAE pipeline).
        config: A :class:`DenoisingConfig` selecting the denoising method.

    Returns:
        Tuple of (
            Tensor of the same shape as `activations`,
            Tensor of scaling factors).
    """
    squeezed = activations.squeeze(0)  # (num_tokens, num_features)

    if config.method is DenoisingMethod.CONTINUOUS_TFIDF:
        denoised_act, scaling_factors = continuous_tfidf(squeezed)
    else:
        raise ValueError(f"Unknown denoising method: {config.method}")

    return denoised_act.unsqueeze(0), scaling_factors
