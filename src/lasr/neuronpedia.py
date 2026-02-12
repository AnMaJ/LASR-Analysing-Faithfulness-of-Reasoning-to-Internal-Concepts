"""Neuronpedia API client for fetching SAE feature explanation labels."""

from __future__ import annotations

import time

import requests

from lasr.config import SAEConfig


def build_sae_id(config: SAEConfig) -> str:
    """Build a Neuronpedia SAE ID string from a SAEConfig.

    Returns e.g. ``"22-gemmascope-2-res-65k"`` for layer 22, width "65k".
    """
    return f"{config.layer}-gemmascope-2-res-{config.width}"


def get_neuronpedia_label(
    model_id: str, sae_id: str, feature_index: int
) -> str | None:
    """Fetch a single feature's explanation label from Neuronpedia.

    GET https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}

    Returns the ``description`` field from the first explanation, or ``None``
    if the request fails or no explanations are available.
    """
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        explanations = data.get("explanations")
        if explanations and len(explanations) > 0:
            return explanations[0].get("description")
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def get_neuronpedia_feature_urls(
    model_id: str, sae_id: str, feature_indices: list[int], embed: bool = False
) -> dict[int, str]:
    """Build Neuronpedia feature page URLs for a list of feature indices.

    Parameters
    ----------
    model_id:
        Neuronpedia model identifier, e.g. ``"gemma-3-27b-it"``.
    sae_id:
        Neuronpedia SAE identifier, e.g. ``"40-gemmascope-2-res-65k"``.
    feature_indices:
        Feature indices to generate URLs for.
    embed:
        If ``True``, append ``?embed=true`` for iframe embedding.

    Returns ``{feature_index: url_string}``.
    """
    base = "https://neuronpedia.org"
    suffix = "?embed=true" if embed else ""
    return {
        idx: f"{base}/{model_id}/{sae_id}/{idx}{suffix}"
        for idx in feature_indices
    }


def get_neuronpedia_labels(
    model_id: str, sae_id: str, feature_indices: list[int], delay: float = 0.1
) -> dict[int, str | None]:
    """Fetch explanation labels for multiple features.

    Calls :func:`get_neuronpedia_label` for each index with a small delay
    between requests to avoid rate-limiting.

    Returns ``{feature_index: concept_label_or_None}``.
    """
    labels: dict[int, str | None] = {}
    for i, idx in enumerate(feature_indices):
        labels[idx] = get_neuronpedia_label(model_id, sae_id, idx)
        if i < len(feature_indices) - 1:
            time.sleep(delay)
    return labels
