"""Feature dataclass and factory for SAE feature analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from neuronpedia_client import NeuronpediaClient


@dataclass
class Feature:
    """A single SAE feature with activations, tokens, and optional Neuronpedia metadata."""

    # Required fields
    feature_idx: int
    activations: np.ndarray  # 1-D, length = total tokens (prompt + generated)
    tokens: list[str]  # token strings, same length as activations

    # Optional fields — populated by fetch_details()
    label: str | None = field(default=None)
    url: str | None = field(default=None)
    embed_url: str | None = field(default=None)
    frac_nonzero: float | None = field(default=None)
    max_act_approx: float | None = field(default=None)

    def fetch_details(self, model_id: str, sae_id: str) -> Feature:
        """Fetch Neuronpedia metadata and populate label/url/frac_nonzero fields.

        Parameters
        ----------
        model_id:
            Neuronpedia model identifier, e.g. ``"gemma-3-27b-it"``.
        sae_id:
            Neuronpedia SAE identifier, e.g. ``"40-gemmascope-2-res-65k"``.

        Returns
        -------
        self, for chaining.
        """
        client = NeuronpediaClient(model_id, sae_id)
        feature = client.get_feature(self.feature_idx)
        self.label = feature.description
        self.url = feature.url
        self.embed_url = feature.embed_url
        self.frac_nonzero = feature.frac_nonzero
        self.max_act_approx = feature.max_act_approx
        return self

    def max_activation(self) -> float:
        """Max absolute activation across all tokens."""
        return float(np.abs(self.activations).max())

    def top_tokens(self, k: int = 5) -> list[tuple[str, float]]:
        """Top-k tokens by activation strength.

        Returns a list of ``(token, activation_value)`` tuples sorted descending.
        """
        indices = np.argsort(np.abs(self.activations))[::-1][:k]
        return [(self.tokens[i], float(self.activations[i])) for i in indices]

    def inspect(self, token_range: str = "all", prompt_length: int | None = None) -> None:
        """Display Neuronpedia IFrame (if available) and token highlighting.

        Parameters
        ----------
        token_range:
            ``"all"`` — full sequence, ``"prompt"`` — prompt only,
            ``"generation"`` — generated tokens only.
        prompt_length:
            Number of prompt tokens. Required when *token_range* is
            ``"prompt"`` or ``"generation"``; ignored when ``"all"``.
        """
        from IPython.display import display, HTML, IFrame

        # Determine slice
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

        # IFrame — only if embed_url has been populated via fetch_details()
        if self.embed_url is not None:
            display(IFrame(src=self.embed_url, width=620, height=480))

        # Token highlighting
        abs_max = float(np.abs(acts).max()) if np.abs(acts).max() > 0 else 1.0
        normed = np.clip(acts / abs_max, 0.0, 1.0)

        spans = []
        for i, (tok, strength) in enumerate(zip(toks, normed)):
            r = int(255 * (1 - strength))
            g = int(255 - 155 * strength)  # 255 -> 100
            b = int(255 * (1 - strength))
            text_color = "#000" if strength < 0.6 else "#fff"
            tok_display = tok.replace("<", "&lt;").replace(">", "&gt;")
            spans.append(
                f'<span style="background:rgb({r},{g},{b});color:{text_color};'
                f'padding:2px 3px;margin:1px;border-radius:3px;display:inline-block;'
                f'font-family:monospace;font-size:13px;" '
                f'title="activation={acts[i]:.4f}">'
                f"{tok_display}</span>"
            )

        label_str = self.label or "N/A"
        frac_str = f"{self.frac_nonzero:.4f}" if self.frac_nonzero is not None else "N/A"
        max_act_str = f"{self.max_act_approx:.2f}" if self.max_act_approx is not None else "N/A"
        html = (
            f'<div style="margin-top:12px;">'
            f"<b>Feature {self.feature_idx}</b> — <i>{label_str}</i>"
            f"<br><b>frac_nonzero:</b> {frac_str} | <b>max_act_approx:</b> {max_act_str}"
            f'<br><span style="font-size:11px;color:#666;">'
            f"Dark green = strong activation, white = weak/no activation"
            f" (showing: {token_range})</span>"
            f'<div style="margin-top:6px;line-height:2;">{"".join(spans)}</div>'
            f"</div>"
        )
        display(HTML(html))

    def __repr__(self) -> str:
        return (
            f"Feature(idx={self.feature_idx}, label={self.label!r}, "
            f"max_act={self.max_activation():.2f}, n_tokens={len(self.tokens)})"
        )


def create_features(
    sae_acts: torch.Tensor,
    feature_indices: list[int],
    tokens: list[str],
) -> list[Feature]:
    """Create Feature objects from SAE activations.

    This is a lightweight factory that extracts per-feature activation vectors
    and builds Feature objects. Neuronpedia metadata can be fetched later via
    ``Feature.fetch_details()``.

    Parameters
    ----------
    sae_acts:
        SAE activation tensor of shape ``(1, n_tokens, n_features)``.
        Should cover the full sequence (prompt + generated) and already
        have any denoising applied.
    feature_indices:
        Pre-filtered list of feature indices to create objects for.
    tokens:
        Full sequence token strings, same length as ``sae_acts`` dim 1.
    """
    features = []
    for idx in feature_indices:
        acts_np = sae_acts[0, :, idx].detach().cpu().float().numpy()
        features.append(
            Feature(
                feature_idx=idx,
                activations=acts_np,
                tokens=tokens,
            )
        )
    return features
