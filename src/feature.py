from __future__ import annotations

from typing import List, Union

import torch

from src.neuronpedia_client import NeuronpediaClient


class Feature:
    """Container for feature information."""

    def __init__(
        self,
        feature_idx: int,
        strength: float,
        client: NeuronpediaClient,
    ):
        self.feature_idx = feature_idx
        self.strength = strength
        self.client = client
        self._get_neuronpedia_info()

    @classmethod
    def from_activations(
        cls,
        feature_indices: torch.Tensor,
        feature_strengths: torch.Tensor,
        client: NeuronpediaClient,
    ) -> Union[List[Feature], List[List[Feature]]]:
        """Create Feature objects from top-k activation results.

        Args:
            feature_indices: 1-D ``(k,)`` or 2-D ``(batch_size, k)`` tensor of
                feature indices (after aggregation + top-k).
            feature_strengths: Tensor of the same shape with corresponding
                activation strengths.
            client: An initialized :class:`NeuronpediaClient`.

        Returns:
            ``List[Feature]`` for 1-D input, or ``List[List[Feature]]``
            (one inner list per prompt) for 2-D batched input.
        """
        if feature_indices.dim() == 1:
            return [
                cls(
                    feature_idx=int(idx),
                    strength=float(strength.detach()),
                    client=client,
                )
                for idx, strength in zip(feature_indices, feature_strengths)
            ]

        return [
            [
                cls(
                    feature_idx=int(idx),
                    strength=float(strength.detach()),
                    client=client,
                )
                for idx, strength in zip(indices_row, strengths_row)
            ]
            for indices_row, strengths_row in zip(feature_indices, feature_strengths)
        ]

    def _get_neuronpedia_info(self) -> None:
        """Fetch feature metadata from Neuronpedia and populate this instance.

        Args:
            client: An initialized :class:`NeuronpediaClient`.
        """

        result = self.client.get_feature(self.feature_idx)
        self.description = result.description
        self.frac_nonzero = result.frac_nonzero
        self.max_act_approx = result.max_act_approx
        self.max_activating_examples = result.max_activating_examples
        self.url = result.url
        self.embed_url = result.embed_url
        self.error = result.error

    def set_description(self, description: str) -> None:
        """Overwrite the feature description."""
        self.description = description

    def __repr__(self) -> str:
        return f"Feature(idx={self.feature_idx}, description={self.description!r})"
