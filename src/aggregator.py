from __future__ import annotations

from functools import wraps
from typing import Any, Callable, List

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
        out = aggregator.max_pooling(x)
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
    def max(self, activations: torch.Tensor) -> torch.Tensor:
        """Take the element-wise max across tokens.
        """
        return activations.max(dim=0).values
    
    @_aggregation_method
    def mean(self, activations: torch.Tensor) -> torch.Tensor:
        """Take the element-wise mean across tokens.
        """
        return activations.mean(dim=0)
