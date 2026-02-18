from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, List

import torch
from tqdm import tqdm

from src.neuronpedia_client import NeuronpediaClient

F = Callable[..., Any]

# Activation-density bounds for the interpretable "sweet spot" (see notes/frac_nonzero_weighting.md)
_SWEET_SPOT_MIN: float = 1e-4   # 0.01% — below this: likely noise / dead features
_SWEET_SPOT_MAX: float = 5e-2   # 5.00% — above this: unspecific / structural features
_EPS: float = 1e-8


def _denoising_method(func: F) -> F:
    """Decorator that registers a method as a denoising strategy and adds a shape check.

    Expects a 2-D input and validates that the output shape matches the input shape.
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
    takes a 2-D tensor and returns a tensor of the same shape.

    Usage::

        # Methods that do not require Neuronpedia:
        denoiser = Denoiser()
        out = denoiser.continuous_tfidf(x)

        # Methods that require Neuronpedia (global_idf):
        denoiser = Denoiser(model_id="gemma-3-27b-it", sae_id="31-gemmascope-2-res-65k")
        out = denoiser.global_idf(x)
    """

    def __init__(
        self,
        model_id: str | None = None,
        sae_id: str | None = None,
    ) -> None:
        self._client: NeuronpediaClient | None = (
            NeuronpediaClient(model_id=model_id, sae_id=sae_id)
            if model_id is not None and sae_id is not None
            else None
        )

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
        """
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        return (activations - mean) / (std + 1e-8)

    @_denoising_method
    def global_idf(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply corpus-level IDF weighting via frac_nonzero from Neuronpedia.

        For each active feature column, fetches the corpus-level activation
        density (``frac_nonzero``) from the Neuronpedia API and applies an
        inverse-frequency weight::

            weight_i = log(1 / (frac_nonzero_i + eps))

        Features whose activation density falls outside the interpretable
        sweet spot [0.01%, 5%] are zeroed out entirely.  Features for which
        Neuronpedia returns no density information are also zeroed out.

        Requires the ``Denoiser`` to be constructed with ``model_id`` and
        ``sae_id``::

            denoiser = Denoiser(model_id="gemma-3-27b-it",
                                sae_id="31-gemmascope-2-res-65k")
            out = denoiser.global_idf(activations)   # same shape as input

        Args:
            activations: 2-D tensor of shape ``(n_rows, d_sae)``.

        Returns:
            Weighted tensor of the same shape.  Only features in the sweet
            spot carry non-zero values.
        """
        if self._client is None:
            raise RuntimeError(
                "global_idf requires a NeuronpediaClient. "
                "Construct Denoiser with model_id and sae_id."
            )

        # Identify columns that are active in at least one row
        active_indices: List[int] = (
            (activations != 0).any(dim=0).nonzero(as_tuple=False).view(-1).tolist()
        )

        # Build IDF weight vector — one scalar per feature dimension
        weights = torch.zeros(activations.shape[1], dtype=torch.float32)

        for idx in tqdm(active_indices, desc="global_idf: fetching frac_nonzero"):
            info = self._client.get_feature(int(idx))
            fnz = info.frac_nonzero
            if fnz is not None and _SWEET_SPOT_MIN <= fnz <= _SWEET_SPOT_MAX:
                weights[idx] = torch.log(torch.tensor(1.0 / (fnz + _EPS)))
            time.sleep(0.1)  # respect Neuronpedia rate limit

        # Broadcast weight vector over the row dimension
        return activations * weights.to(activations.device).unsqueeze(0)
