from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, List, Optional

import torch
from tqdm import tqdm

from src.neuronpedia_client import NeuronpediaClient

F = Callable[..., Any]



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
        neuronpedia_client: Optional[NeuronpediaClient] = None
    ) -> None:
        self._client = neuronpedia_client

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
        _SWEET_SPOT_MIN: float = 1e-4   # 0.01% — below this: likely noise / dead features
        _SWEET_SPOT_MAX: float = 5e-2   # 5.00% — above this: unspecific / structural features
        _EPS: float = 1e-8
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
            time.sleep(0.05)  # respect Neuronpedia rate limit

        # Broadcast weight vector over the row dimension
        return activations * weights.to(activations.device).unsqueeze(0)

    @_denoising_method
    def pmi(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply Positive Pointwise Mutual Information (PPMI) weighting.

        Measures how much more each feature fires at each token than expected
        from its corpus-level base rate.  Unlike ``global_idf``, the score is
        computed element-wise — it depends on both the activation strength
        *and* the feature's empirical maximum, making it sensitive to the
        degree of activation rather than just its presence.

        **Formula** — for token *r* and feature *i*:

        .. code-block::

            P(i | context)_r  =  activation_{r,i} / max_act_approx_i   ∈ [0, 1]
            P(i)              =  frac_nonzero_i                         ∈ (0, 1)
            PMI_{r,i}         =  log( P(i | context)_r / P(i) )
            score_{r,i}       =  max( 0, PMI_{r,i} )   # Positive PMI

        A feature scores 0 when its activation is at or below the baseline
        expected from the corpus.  At maximum activation the score equals
        ``log(1 / frac_nonzero_i)``, which is the same as the ``global_idf``
        weight — making ``global_idf`` a special case of PPMI where every
        active feature is assumed to fire at its maximum value.

        Features outside the interpretable sweet spot [0.01%, 5%] or missing
        ``max_act_approx`` from Neuronpedia are zeroed out.

        Requires the ``Denoiser`` to be constructed with a
        :class:`~src.neuronpedia_client.NeuronpediaClient`.

        Args:
            activations: 2-D tensor of shape ``(n_tokens, d_sae)``.

        Returns:
            PPMI score tensor of the same shape.
        """
        _SWEET_SPOT_MIN: float = 1e-4
        _SWEET_SPOT_MAX: float = 5e-2
        _EPS: float = 1e-8

        if self._client is None:
            raise RuntimeError(
                "pmi requires a NeuronpediaClient. "
                "Construct Denoiser with a NeuronpediaClient instance."
            )

        device = activations.device
        d_sae = activations.shape[1]

        # Identify columns with at least one non-zero activation
        active_indices: List[int] = (
            (activations != 0).any(dim=0).nonzero(as_tuple=False).view(-1).tolist()
        )

        # Per-feature vectors: filled only for features that pass all checks
        frac_nonzero_vec = torch.ones(d_sae, dtype=torch.float32)   # default → PMI = 0
        max_act_vec = torch.ones(d_sae, dtype=torch.float32)         # default → PMI = 0
        valid_mask = torch.zeros(d_sae, dtype=torch.bool)

        for idx in tqdm(active_indices, desc="pmi: fetching Neuronpedia metadata"):
            info = self._client.get_feature(int(idx))
            fnz = info.frac_nonzero
            max_act = info.max_act_approx
            if (
                fnz is not None
                and max_act is not None
                and max_act > 0
                and _SWEET_SPOT_MIN <= fnz <= _SWEET_SPOT_MAX
            ):
                frac_nonzero_vec[idx] = fnz
                max_act_vec[idx] = max_act
                valid_mask[idx] = True
            time.sleep(0.05)

        frac_nonzero_vec = frac_nonzero_vec.to(device)
        max_act_vec = max_act_vec.to(device)
        valid_mask = valid_mask.to(device)

        # P(feature | context): normalize each activation by the feature's empirical max
        p_given_context = activations / max_act_vec.unsqueeze(0)   # (n_tokens, d_sae)

        # PMI = log( P(feature | context) / P(feature) )
        # EPS in numerator handles zero activations → log(EPS/p) very negative → clamped to 0
        pmi_scores = torch.log(
            (p_given_context + _EPS) / (frac_nonzero_vec.unsqueeze(0) + _EPS)
        )

        # Positive PMI: discard features that are no more active than their baseline
        ppmi_scores = torch.clamp(pmi_scores, min=0.0)

        # Zero out invalid features (outside sweet spot, missing metadata)
        ppmi_scores = ppmi_scores * valid_mask.unsqueeze(0)

        return ppmi_scores
