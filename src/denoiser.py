from __future__ import annotations

from functools import wraps
from typing import Any, Callable, List

import torch

F = Callable[..., Any]


def _denoising_method(func: F) -> F:
    """Decorator that registers a method as a denoising strategy and adds a shape check.

    Expects a 2-D ``(num_tokens, num_features)`` input and validates that the
    output shape matches the input shape.
    """

    @wraps(func)
    def wrapper(self: Denoiser, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() != 2:
            raise RuntimeError(
                f"Denoising method '{func.__name__}' expects a 2-D tensor, "
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
    takes a 2-D tensor ``(num_tokens, num_features)`` and returns a tensor of
    the same shape. Can be called directly (e.g.
    ``denoiser.continuous_tfidf(tensor)``).

    Usage::

        denoiser = Denoiser()
        print(denoiser.get_methods())
        out = denoiser.continuous_tfidf(x)   # same shape as x
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

        Treats each row (token) as a document and each column (feature) as a
        term.

        Args:
            activations: Tensor of shape ``(num_tokens, num_features)``.

        Returns:
            Tensor of the same shape with TF-IDF weighting applied.
        """
        num_docs = activations.shape[0]
        tf = activations
        df = activations.sum(dim=0, keepdim=True)
        idf = torch.log(num_docs / (1 + df))
        return tf * idf

    @_denoising_method
    def standard_scaler(self, activations: torch.Tensor) -> torch.Tensor:
        """Standardize features by removing the mean and scaling to unit variance.

        Computes z-score normalization along the token dimension (``dim=0``),
        equivalent to sklearn's StandardScaler.

        Args:
            activations: Tensor of shape ``(num_tokens, num_features)``.

        Returns:
            Tensor of the same shape with zero mean and unit variance per feature.
        """
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        return (activations - mean) / (std + 1e-8)
