"""Feature dataclass and factory for SAE feature analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.neuronpedia_client import NeuronpediaClient


@dataclass
class Feature:
    """A single SAE feature with optional per-token activations and Neuronpedia metadata.

    Supports two modes of construction:

    **Per-token mode** (via :func:`create_features`):
        Stores a 1-D activation vector and corresponding token strings for the
        full sequence.  Enables :meth:`top_tokens`, :meth:`inspect`, and
        :meth:`max_activation`.

    **Aggregated mode** (via :meth:`from_activations`):
        Stores a single representative ``strength`` value.  Neuronpedia
        metadata is fetched eagerly during construction.
    """

    feature_idx: int

    # -- Per-token mode fields (set by create_features) --
    activations: np.ndarray | None = field(default=None, repr=False)
    tokens: list[str] | None = field(default=None, repr=False)

    # -- Aggregated mode field (set by from_activations) --
    strength: float | None = None

    # -- Neuronpedia metadata (populated by fetch_details) --
    description: str | None = None
    url: str | None = None
    embed_url: str | None = None
    frac_nonzero: float | None = None
    max_act_approx: float | None = None
    max_activating_examples: list | None = field(default=None, repr=False)
    error: str | None = None

    # ---- Construction helpers ------------------------------------------------

    @classmethod
    def from_activations(
        cls,
        feature_indices: torch.Tensor,
        feature_strengths: torch.Tensor,
        client: NeuronpediaClient,
    ) -> list[Feature] | list[list[Feature]]:
        """Create Feature objects from top-k activation results.

        Neuronpedia metadata is fetched eagerly for each feature, matching
        the behaviour expected by ``demo.ipynb``.

        Args:
            feature_indices: 1-D ``(k,)`` or 2-D ``(batch, k)`` tensor of
                feature indices.
            feature_strengths: Tensor of the same shape with corresponding
                activation strengths.
            client: An initialised :class:`NeuronpediaClient`.

        Returns:
            ``list[Feature]`` for 1-D input, or ``list[list[Feature]]`` for
            2-D batched input.
        """

        def _make(idx: int, strength: float) -> Feature:
            feat = cls(feature_idx=idx, strength=strength)
            feat.fetch_details(client)
            return feat

        if feature_indices.dim() == 1:
            return [
                _make(int(idx), float(strength.detach()))
                for idx, strength in zip(feature_indices, feature_strengths)
            ]

        return [
            [
                _make(int(idx), float(strength.detach()))
                for idx, strength in zip(row_idx, row_str)
            ]
            for row_idx, row_str in zip(feature_indices, feature_strengths)
        ]

    # ---- Neuronpedia integration ---------------------------------------------

    def fetch_details(self, client: NeuronpediaClient) -> Feature:
        """Fetch Neuronpedia metadata and populate description/url/stats fields.

        Args:
            client: An initialised :class:`NeuronpediaClient`.

        Returns:
            ``self``, for chaining.
        """
        result = client.get_feature(self.feature_idx)
        self.description = result.description
        self.url = result.url
        self.embed_url = result.embed_url
        self.frac_nonzero = result.frac_nonzero
        self.max_act_approx = result.max_act_approx
        self.max_activating_examples = result.max_activating_examples
        self.error = result.error
        return self

    # ---- Per-token analysis methods ------------------------------------------

    def max_activation(self) -> float:
        """Max absolute activation across all tokens.

        Requires per-token activations (set via :func:`create_features`).
        """
        if self.activations is None:
            raise ValueError("Per-token activations not available")
        return float(np.abs(self.activations).max())

    def top_tokens(self, k: int = 5) -> list[tuple[str, float]]:
        """Top-*k* tokens by activation strength.

        Returns a list of ``(token, activation_value)`` tuples sorted
        descending by absolute activation.
        """
        if self.activations is None or self.tokens is None:
            raise ValueError("Per-token activations not available")
        indices = np.argsort(np.abs(self.activations))[::-1][:k]
        return [(self.tokens[i], float(self.activations[i])) for i in indices]

    def inspect(
        self,
        token_range: str = "all",
        prompt_length: int | None = None,
    ) -> None:
        """Display Neuronpedia IFrame (if available) and token highlighting.

        When per-token activations are not available, an informational
        message is shown and token highlighting is skipped.  The IFrame
        and feature metadata are still displayed.

        Args:
            token_range: ``"all"`` — full sequence, ``"prompt"`` — prompt only,
                ``"generation"`` — generated tokens only.
            prompt_length: Number of prompt tokens.  Required when
                *token_range* is ``"prompt"`` or ``"generation"``.
        """
        from IPython.display import display, HTML, IFrame

        has_activations = self.activations is not None and self.tokens is not None

        if not has_activations:
            display(HTML(
                '<div style="margin-top:12px;padding:8px 12px;'
                'background:#fff3cd;border:1px solid #ffc107;'
                'border-radius:4px;font-size:13px;">'
                '<b>Note:</b> Per-token activations are not available. '
                'Token highlighting will not be displayed.'
                '</div>'
            ))

        # IFrame — only if embed_url has been populated via fetch_details()
        if self.embed_url is not None:
            display(IFrame(src=self.embed_url, width=620, height=480))

        # Token highlighting — only when per-token activations are present
        spans: list[str] = []
        if has_activations:
            if token_range in ("prompt", "generation"):
                if prompt_length is None:
                    raise ValueError(
                        f"prompt_length is required when token_range={token_range!r}"
                    )
                if token_range == "prompt":
                    acts = self.activations[:prompt_length]
                    toks = self.tokens[:prompt_length]
                else:
                    acts = self.activations[prompt_length:]
                    toks = self.tokens[prompt_length:]
            else:
                acts = self.activations
                toks = self.tokens

            abs_max = float(np.abs(acts).max()) if np.abs(acts).max() > 0 else 1.0
            normed = np.clip(acts / abs_max, 0.0, 1.0)

            for i, (tok, intensity) in enumerate(zip(toks, normed)):
                r = int(255 * (1 - intensity))
                g = int(255 - 155 * intensity)
                b = int(255 * (1 - intensity))
                text_color = "#000" if intensity < 0.6 else "#fff"
                tok_display = tok.replace("<", "&lt;").replace(">", "&gt;")
                spans.append(
                    f'<span style="background:rgb({r},{g},{b});color:{text_color};'
                    f'padding:2px 3px;margin:1px;border-radius:3px;display:inline-block;'
                    f'font-family:monospace;font-size:13px;" '
                    f'title="activation={acts[i]:.4f}">'
                    f"{tok_display}</span>"
                )

        # Feature metadata — always shown
        desc_str = self.description or "N/A"
        frac_str = (
            f"{self.frac_nonzero:.4f}" if self.frac_nonzero is not None else "N/A"
        )
        max_act_str = (
            f"{self.max_act_approx:.2f}" if self.max_act_approx is not None else "N/A"
        )
        token_highlight_html = ""
        if spans:
            token_highlight_html = (
                f'<br><span style="font-size:11px;color:#666;">'
                f"Dark green = strong activation, white = weak/no activation"
                f" (showing: {token_range})</span>"
                f'<div style="margin-top:6px;line-height:2;">{"".join(spans)}</div>'
            )
        html = (
            f'<div style="margin-top:12px;">'
            f"<b>Feature {self.feature_idx}</b> — <i>{desc_str}</i>"
            f"<br><b>frac_nonzero:</b> {frac_str} | <b>max_act_approx:</b> {max_act_str}"
            f"{token_highlight_html}"
            f"</div>"
        )
        display(HTML(html))

    # ---- Dunder methods ------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"idx={self.feature_idx}", f"description={self.description!r}"]
        if self.activations is not None:
            parts.append(f"max_act={self.max_activation():.2f}")
            parts.append(f"n_tokens={len(self.tokens)}")
        if self.strength is not None:
            parts.append(f"strength={self.strength:.2f}")
        return f"Feature({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# Factory function
# --------------------------------------------------------------------------- #


def create_features(
    sae_acts: torch.Tensor,
    feature_indices: list[int],
    tokens: list[str],
) -> list[Feature]:
    """Create Feature objects with per-token activations from SAE output.

    This is a lightweight factory that extracts per-feature activation vectors
    and builds :class:`Feature` objects.  Neuronpedia metadata can be fetched
    later via :meth:`Feature.fetch_details`.

    Args:
        sae_acts: SAE activation tensor of shape ``(n_tokens, n_features)``.
        feature_indices: Pre-filtered list of feature indices.
        tokens: Full-sequence token strings (same length as *sae_acts* dim 0).
    """
    features: list[Feature] = []
    for idx in feature_indices:
        acts_np = sae_acts[:, idx].detach().cpu().float().numpy()
        features.append(
            Feature(
                feature_idx=idx,
                activations=acts_np,
                tokens=tokens,
            )
        )
    return features
