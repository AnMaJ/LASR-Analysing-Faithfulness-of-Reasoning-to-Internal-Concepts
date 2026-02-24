from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import torch


F = Callable[..., Any]


def _aggregation_method(func: F) -> F:
    """Decorator that registers a method as an aggregation strategy and adds a shape check.
    """

    @wraps(func)
    def wrapper(self: Aggregator, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() != 2:
            raise RuntimeError(
                f"Aggregation method '{func.__name__}' expects a 2-D tensor, "
                f"got {activations.dim()}-D with shape {tuple(activations.shape)}"
            )
        hidden_size = activations.shape[-1]
        result = func(self, activations)

        expected = (hidden_size,)
        if result.shape != expected:
            raise RuntimeError(
                f"Aggregation method '{func.__name__}' should return shape "
                f"{expected}, got {tuple(result.shape)}"
            )
        return result

    wrapper._is_aggregation_method = True  # type: ignore[attr-defined]
    return wrapper


class Aggregator:
    """Container for aggregation strategies over the token dimension.

    Each public method decorated with ``@_aggregation_method`` is a strategy
    that takes a 2-D tensor and reduces the token dimension to produce a 1-D tensor.

    Usage::

        aggregator = Aggregator()
        print(aggregator.get_methods())
        out = aggregator.max(x)
    """

    def _compute_threshold(self, activations: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute a per-feature activation threshold for consistency windowing.

        Args:
            activations: (n_tokens, d_sae) SAE activation tensor.
            mode: One of "per_feature_median", "global_median".

        Returns:
            (d_sae,) threshold vector — one value per feature.
        """
        if mode == "global_median":
            nonzero = activations[activations > 0]
            val = nonzero.median().item() if nonzero.numel() > 0 else 0.0
            return torch.full((activations.shape[1],), val, device=activations.device)

        # per_feature_median: vectorized median of each feature's nonzero activations
        masked_acts = activations.masked_fill(activations == 0, float('nan'))
        tau = torch.nanmedian(masked_acts, dim=0).values  # (d_sae,)
        tau = torch.nan_to_num(tau, nan=0.0)  # replace NaNs (all-zero features) with 0
        return tau

    def get_methods(self) -> list[str]:
        """Return the names of all available aggregation strategies."""
        return [
            name
            for name in dir(self)
            if not name.startswith("_")
            and callable(getattr(self, name))
            and getattr(getattr(self, name), "_is_aggregation_method", False)
        ]

    @_aggregation_method
    def max(self, activations: torch.Tensor) -> torch.Tensor:
        """Take the element-wise max across tokens."""
        return activations.max(dim=0).values
    
    @_aggregation_method
    def mean(self, activations: torch.Tensor) -> torch.Tensor:
        """Take the element-wise mean across tokens."""
        return activations.mean(dim=0)

    @_aggregation_method
    def consistency_max(self, activations: torch.Tensor) -> torch.Tensor:
        """Consistency-augmented max aggregation.

        Combines standard max-pooling with a *temporal consistency* score that
        rewards features which remain active across consecutive windows of tokens,
        rather than firing intensely on a single token and nowhere else.
        """
        # Alpha controls the weight given to the max activation.
        # Consistency weight is computed as (1-alpha)
        alpha: float = 0.5
        window_length: int = 3
        tau_mode: str = "per_feature_median"

        n_tokens, d_sae = activations.shape

        # ── Step 1: max pooling ──────────────────────────────────────────────
        max_act = activations.max(dim=0).values                      # (d_sae,)

        # Sequence shorter than window → no consistent windows possible
        if n_tokens < window_length:
            return max_act

        # ── Step 2: per-feature threshold ────────────────────────────────────
        tau = self._compute_threshold(activations, tau_mode)              # (d_sae,)

        # ── Step 3: sliding window consistency mask ──────────────────────────
        # above_tau[t, i] = 1 if a_i(t) >= tau_i
        above_tau = (activations >= tau.unsqueeze(0)).float()        # (n_tokens, d_sae)

        # Cumulative sum lets us count how many tokens in [t, t+l-1] are above tau
        n_windows = n_tokens - window_length + 1
        padded = torch.cat([torch.zeros(1, d_sae, device=activations.device), above_tau.cumsum(dim=0)], dim=0)
        window_above = padded[window_length:window_length + n_windows] - padded[:n_windows]
        consistent = (window_above == window_length)                 # (n_windows, d_sae)

        # ── Step 4: mean activation inside consistent windows ────────────────
        act_padded = torch.cat([torch.zeros(1, d_sae, device=activations.device), activations.cumsum(dim=0)], dim=0)
        window_means = (act_padded[window_length:window_length + n_windows] - act_padded[:n_windows]) / window_length

        # Average only over consistent windows; 0 if none exist
        n_consistent = consistent.float().sum(dim=0)                 # (d_sae,)
        consistency = torch.where(
            n_consistent > 0,
            (window_means * consistent.float()).sum(dim=0) / n_consistent,
            torch.zeros(d_sae, device=activations.device),
        )                                                            # (d_sae,)

        # ── Step 5: blend ────────────────────────────────────────────────────
        return alpha * max_act + (1.0 - alpha) * consistency
