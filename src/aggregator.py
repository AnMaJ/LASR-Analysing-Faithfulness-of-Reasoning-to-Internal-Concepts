from __future__ import annotations

from functools import wraps
from typing import Any, Callable, List

import torch

F = Callable[..., Any]


def _aggregation_method(func: F) -> F:
    """Decorator that registers a method as an aggregation strategy and adds a shape check.

    Supports both 2-D ``(num_tokens, hidden_size)`` and 3-D
    ``(batch_size, num_tokens, hidden_size)`` inputs.  Aggregation always
    reduces along ``dim=-2`` (the token dimension).
    """

    @wraps(func)
    def wrapper(self: Aggregator, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() not in (2, 3):
            raise RuntimeError(
                f"Aggregation method '{func.__name__}' expects a 2-D or 3-D tensor, "
                f"got {activations.dim()}-D with shape {tuple(activations.shape)}"
            )
        hidden_size = activations.shape[-1]
        result = func(self, activations)

        if activations.dim() == 2:
            expected = (hidden_size,)
        else:
            expected = (activations.shape[0], hidden_size)

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
    that reduces the token dimension (``dim=-2``).

    Supports two input shapes:

    - **2-D** ``(num_tokens, hidden_size)`` → ``(hidden_size,)``
    - **3-D** ``(batch_size, num_tokens, hidden_size)`` → ``(batch_size, hidden_size)``

    Usage::

        aggregator = Aggregator()
        print(aggregator.get_methods())
        out_2d = aggregator.max_pooling(x_2d)   # (hidden_size,)
        out_3d = aggregator.max_pooling(x_3d)   # (batch_size, hidden_size)
    """

    def get_methods(self) -> List[str]:
        """Return the names of all available aggregation strategies."""
        return [
            name
            for name in dir(self)
            if not name.startswith("_")
            and callable(getattr(self, name))
            and getattr(getattr(self, name), "_is_aggregation_method", False)
        ]

    @_aggregation_method
    def max_pooling(self, activations: torch.Tensor) -> torch.Tensor:
        """Take the element-wise max across tokens.

        Args:
            activations: Tensor of shape ``(num_tokens, hidden_size)`` or
                ``(batch_size, num_tokens, hidden_size)``.

        Returns:
            Tensor of shape ``(hidden_size,)`` or ``(batch_size, hidden_size)``.
        """
        return activations.max(dim=-2).values
