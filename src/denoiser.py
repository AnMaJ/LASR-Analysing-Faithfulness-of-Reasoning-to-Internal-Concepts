"""Denoising / normalisation strategies for SAE activations."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

import torch
from tqdm import tqdm

from src.configs import DenoisingConfig
from src.neuronpedia_client import NeuronpediaClient

F = Callable[..., Any]

_EPS: float = 1e-8


def _denoising_method(func: F) -> F:
    """Decorator that registers a method as a denoising strategy and adds a shape check.

    Expects a 2-D input and validates that the output shape matches the input shape.
    Extra keyword arguments are forwarded to the wrapped method.
    """

    @wraps(func)
    def wrapper(self: Denoiser, activations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if activations.dim() != 2:
            raise RuntimeError(
                f"Denoising method '{func.__name__}' expects a 2-D tensor, "
                f"got {activations.dim()}-D with shape {tuple(activations.shape)}"
            )
        shape = activations.shape
        result = func(self, activations, **kwargs)
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
    takes a 2-D tensor ``(n_tokens, n_features)`` and returns a tensor of the
    same shape.

    Methods can be called directly for fine-grained control, or via
    :meth:`denoise` for config-driven dispatch.

    Usage::

        denoiser = Denoiser()
        cfg = DenoisingConfig(method="continuous_tfidf", params={"threshold": 10.0})
        out = denoiser.denoise(x, cfg)

        # Or call methods directly:
        out = denoiser.continuous_tfidf(x, threshold=10.0)
    """

    # ── Introspection ─────────────────────────────────────────────────────

    def get_methods(self) -> list[str]:
        """Return the names of all available denoising strategies."""
        return [
            name
            for name in dir(self)
            if not name.startswith("_")
            and callable(getattr(self, name))
            and getattr(getattr(self, name), "_is_denoising_method", False)
        ]

    # ── Config-driven dispatch ────────────────────────────────────────────

    def denoise(
        self,
        activations: torch.Tensor,
        config: DenoisingConfig,
    ) -> torch.Tensor:
        """Denoise *activations* using the strategy specified in *config*.

        Dispatches to the appropriate method and forwards ``config.params``
        as keyword arguments.

        Args:
            activations: 2-D tensor ``(n_tokens, n_features)``.
            config: A :class:`DenoisingConfig` selecting the method and its
                parameters.

        Returns:
            Denoised tensor with the same shape as *activations*.
        """
        method_name = config.method
        method_fn = getattr(self, method_name, None)
        if method_fn is None or not getattr(method_fn, "_is_denoising_method", False):
            raise ValueError(f"Unknown denoising method: {config.method!r}")
        return method_fn(activations, **config.params)

    # ── Denoising methods ─────────────────────────────────────────────────

    @_denoising_method
    def continuous_tfidf(
        self, activations: torch.Tensor, threshold: float,
    ) -> torch.Tensor:
        """Apply continuous TF-IDF weighting to SAE activations.

        Treats each row (token) as a document and each column (feature) as a
        term.  Document frequency is the number of tokens where the feature
        activation exceeds *threshold*.

        Args:
            activations: Tensor of shape ``(n_tokens, n_features)``.
            threshold: Activation threshold for document frequency counting.

        Returns:
            Tensor of shape ``(n_tokens, n_features)`` with TF-IDF weighting.
        """
        num_docs = activations.shape[0]
        tf = activations
        df = (activations > threshold).sum(dim=0, keepdim=True)
        idf = torch.log(num_docs / (1 + df))
        return tf * idf

    @_denoising_method
    def standard_scaler(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply zero-mean, unit-variance normalization.

        A nonzero mask is applied after scaling to preserve sparsity (dead
        features stay at zero).
        """
        nonzero_mask = activations != 0
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        return ((activations - mean) / (std + _EPS)) * nonzero_mask

    @_denoising_method
    def global_idf(
        self,
        activations: torch.Tensor,
        neuronpedia_client: NeuronpediaClient,
        sweet_spot_min: float,
        sweet_spot_max: float,
    ) -> torch.Tensor:
        """Apply corpus-level IDF weighting via ``frac_nonzero`` from Neuronpedia.

        For each active feature column, fetches the corpus-level activation
        density from the Neuronpedia API and applies an inverse-frequency
        weight::

            weight_i = log(1 / (frac_nonzero_i + eps))

        Features whose density falls outside ``[sweet_spot_min, sweet_spot_max]``
        are zeroed out, as are features without Neuronpedia data.

        Args:
            activations: Tensor of shape ``(n_tokens, n_features)``.
            neuronpedia_client: Client for fetching corpus-level statistics.
            sweet_spot_min: Minimum activation density.
            sweet_spot_max: Maximum activation density.
        """
        active_indices = self._active_feature_indices(activations)

        weights = torch.zeros(activations.shape[1], dtype=torch.float32)
        for idx in tqdm(active_indices, desc="global_idf: fetching frac_nonzero"):
            info = neuronpedia_client.get_feature(int(idx))
            fnz = info.frac_nonzero
            if fnz is not None and sweet_spot_min <= fnz <= sweet_spot_max:
                weights[idx] = torch.log(torch.tensor(1.0 / (fnz + _EPS)))
            time.sleep(0.05)  # respect Neuronpedia rate limit

        return activations * weights.to(activations.device).unsqueeze(0)

    @_denoising_method
    def pmi(
        self,
        activations: torch.Tensor,
        neuronpedia_client: NeuronpediaClient,
        sweet_spot_min: float,
        sweet_spot_max: float,
    ) -> torch.Tensor:
        """Apply Positive Pointwise Mutual Information (PPMI) weighting.

        Measures how much more each feature fires at each token than expected
        from its corpus-level base rate.  Unlike :meth:`global_idf`, the score
        is element-wise — it depends on both activation strength *and* the
        feature's empirical maximum.

        **Formula** — for token *r* and feature *i*::

            P(i | context)_r  =  activation_{r,i} / max_act_approx_i   in [0, 1]
            P(i)              =  frac_nonzero_i                         in (0, 1)
            PMI_{r,i}         =  log( P(i | context)_r / P(i) )
            score_{r,i}       =  max( 0, PMI_{r,i} )   # Positive PMI

        At maximum activation the score equals ``log(1 / frac_nonzero_i)``,
        i.e. the ``global_idf`` weight — making ``global_idf`` a special case
        of PPMI where every active feature fires at its maximum.

        Args:
            activations: Tensor of shape ``(n_tokens, n_features)``.
            neuronpedia_client: Client for fetching corpus-level statistics.
            sweet_spot_min: Minimum activation density.
            sweet_spot_max: Maximum activation density.
        """
        device = activations.device
        d_sae = activations.shape[1]

        active_indices = self._active_feature_indices(activations)

        frac_nonzero_vec = torch.ones(d_sae, dtype=torch.float32)
        max_act_vec = torch.ones(d_sae, dtype=torch.float32)
        valid_mask = torch.zeros(d_sae, dtype=torch.bool)

        for idx in tqdm(active_indices, desc="pmi: fetching Neuronpedia metadata"):
            info = neuronpedia_client.get_feature(int(idx))
            fnz = info.frac_nonzero
            max_act = info.max_act_approx
            if (
                fnz is not None
                and max_act is not None
                and max_act > 0
                and sweet_spot_min <= fnz <= sweet_spot_max
            ):
                frac_nonzero_vec[idx] = fnz
                max_act_vec[idx] = max_act
                valid_mask[idx] = True
            time.sleep(0.05)

        frac_nonzero_vec = frac_nonzero_vec.to(device)
        max_act_vec = max_act_vec.to(device)
        valid_mask = valid_mask.to(device)

        p_given_context = activations / max_act_vec.unsqueeze(0)

        pmi_scores = torch.log(
            (p_given_context + _EPS) / (frac_nonzero_vec.unsqueeze(0) + _EPS)
        )

        ppmi_scores = torch.clamp(pmi_scores, min=0.0)
        ppmi_scores = ppmi_scores * valid_mask.unsqueeze(0)

        return ppmi_scores

    # ── Corpus-level helpers ──────────────────────────────────────────────

    @staticmethod
    def compute_corpus_idf(
        encodings: list[torch.Tensor],
        threshold: float,
    ) -> tuple[torch.Tensor, int]:
        """Compute IDF vector from a corpus of (possibly sparse) SAE activations.

        Streams over encodings one at a time to avoid materializing the full
        dense corpus in memory.

        Args:
            encodings: List of tensors, each ``(n_tokens_i, n_features)``.
                       May be sparse (COO) or dense.
            threshold: Activation threshold for document-frequency counting.

        Returns:
            ``(idf, total_tokens)`` where *idf* has shape ``(n_features,)``
            and *total_tokens* is the sum of all token counts.
        """
        df: torch.Tensor | None = None
        total_tokens = 0
        for enc in tqdm(encodings, desc="Computing corpus IDF"):
            dense = enc.to_dense() if enc.is_sparse else enc
            if df is None:
                df = torch.zeros(dense.shape[1], dtype=torch.float32)
            df += (dense > threshold).sum(dim=0).float().cpu()
            total_tokens += dense.shape[0]
        idf = torch.log(torch.tensor(total_tokens, dtype=torch.float32) / (1 + df))
        return idf, total_tokens

    @staticmethod
    def apply_tfidf(
        activations: torch.Tensor,
        idf: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a pre-computed IDF vector to a single sample's activations.

        Args:
            activations: 2-D tensor ``(n_tokens, n_features)``, dense.
            idf: 1-D tensor ``(n_features,)``.

        Returns:
            Tensor of same shape as *activations* with TF-IDF weighting.
        """
        return activations * idf.unsqueeze(0).to(activations.device)

    @staticmethod
    def make_special_token_mask(tokens: list[str]) -> torch.Tensor:
        """Return a boolean mask (True = special token) based on <...> pattern.

        Args:
            tokens: List of token strings.

        Returns:
            Boolean tensor of shape ``(len(tokens),)``; True where token matches
            the ``<...>`` pattern (e.g. ``<bos>``, ``<eos>``, ``<pad>``).
        """
        import re
        pattern = re.compile(r'^<[^>]+>$')
        return torch.tensor([bool(pattern.match(t)) for t in tokens], dtype=torch.bool)

    @staticmethod
    def compute_frequency_filter(
        encodings: list[torch.Tensor],
        threshold: float = 0.0,
        special_token_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute corpus-wide firing frequency for all SAE features.

        Args:
            encodings: List of (n_tokens_i, n_features) sparse or dense tensors.
            threshold: Activation threshold for counting a feature as firing (default=0).
            special_token_masks: Optional per-sample boolean masks (True = exclude token).

        Returns:
            freq: 1-D tensor of shape (n_features,) with raw firing counts per feature.
        """
        freq: torch.Tensor | None = None
        for i, enc in enumerate(tqdm(encodings, desc="Computing frequency filter")):
            dense = enc.to_dense() if enc.is_sparse else enc
            if freq is None:
                freq = torch.zeros(dense.shape[1], dtype=torch.float32)
            acts = dense.float().cpu()
            if special_token_masks is not None and i < len(special_token_masks):
                mask = special_token_masks[i]
                acts = acts[~mask]
            freq += (acts > threshold).float().sum(dim=0)
        # Return full frequency tensor for all features
        return freq  # shape: (n_features,)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _active_feature_indices(activations: torch.Tensor) -> list[int]:
        """Return column indices that have at least one non-zero activation."""
        return (
            (activations != 0).any(dim=0).nonzero(as_tuple=False).view(-1).tolist()
        )
