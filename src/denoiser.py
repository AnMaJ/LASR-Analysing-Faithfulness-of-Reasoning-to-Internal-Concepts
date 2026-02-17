from __future__ import annotations

from functools import wraps
from typing import Any, Callable, List

import torch

F = Callable[..., Any]


def _denoising_method(func: F) -> F:
    """Decorator that registers a method as a denoising strategy and adds a shape check.

    The decorated method must support both 2-D ``(batch_size, num_features)``
    and 3-D ``(batch_size, num_tokens, num_features)`` inputs. The decorator
    validates that the output shape matches the input shape.
    """

    @wraps(func)
    def wrapper(self: Denoiser, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() not in (2, 3):
            raise RuntimeError(
                f"Denoising method '{func.__name__}' expects a 2-D or 3-D tensor, "
                f"got {activations.dim()}-D with shape {tuple(activations.shape)}"
            )
        shape = activations.shape
        result = func(self, activations)
        if result.shape != shape:
            raise RuntimeError(
                f"Denoising method '{func.__name__}' changed tensor shape "
                f"from {tuple(shape)} to {tuple(result.shape)}"
            )
        return result

    wrapper._is_denoising_method = True  # type: ignore[attr-defined]
    return wrapper


class Denoiser:
    """Container for denoising / normalization strategies on SAE activations.

    Each public method decorated with ``@_denoising_method`` is a strategy that
    can be called directly (e.g. ``denoiser.continuous_tfidf(tensor)``).

    Every strategy supports two input shapes:

    - **2-D** ``(batch_size, num_features)`` — after aggregation over tokens
      (one vector per prompt).
    - **3-D** ``(batch_size, num_tokens, num_features)`` — before aggregation
      (full token-level activations across multiple prompts).

    Every strategy is enforced to preserve the input tensor shape.

    Usage::

        denoiser = Denoiser()
        print(denoiser.get_methods())
        out_2d = denoiser.continuous_tfidf(x_2d)   # (batch, feats)
        out_3d = denoiser.continuous_tfidf(x_3d)   # (batch, tokens, feats)
    """

    def get_methods(self) -> List[str]:
        """Return the names of all available denoising strategies."""
        return [
            name
            for name in dir(self)
            if not name.startswith("_")
            and callable(getattr(self, name))
            and getattr(getattr(self, name), "_is_denoising_method", False)
        ]

    @_denoising_method
    def continuous_tfidf(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply continuous TF-IDF weighting to SAE activations.

        Operates along the **second-to-last** dimension (tokens for 3-D,
        batch entries for 2-D).

        Args:
            activations: Tensor of shape ``(batch_size, num_features)`` or
                ``(batch_size, num_tokens, num_features)``.

        Returns:
            Tensor of the same shape with TF-IDF weighting applied.
        """
        dim = -2  # tokens (3-D) or batch (2-D)
        num_docs = activations.shape[dim]
        tf = activations
        df = activations.sum(dim=dim, keepdim=True)
        idf = torch.log(num_docs / (1 + df))
        return tf * idf

    @_denoising_method
    def standard_scaler(self, activations: torch.Tensor) -> torch.Tensor:
        """Standardize features by removing the mean and scaling to unit variance.

        Computes z-score normalization along the **second-to-last** dimension
        (tokens for 3-D, batch entries for 2-D), equivalent to sklearn's
        StandardScaler.

        Args:
            activations: Tensor of shape ``(batch_size, num_features)`` or
                ``(batch_size, num_tokens, num_features)``.

        Returns:
            Tensor of the same shape with zero mean and unit variance per feature.
        """
        dim = -2  # tokens (3-D) or batch (2-D)
        mean = activations.mean(dim=dim, keepdim=True)
        std = activations.std(dim=dim, keepdim=True)
        return (activations - mean) / (std + 1e-8)
