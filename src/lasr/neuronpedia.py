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


def get_neuronpedia_feature_data(
    model_id: str, sae_id: str, feature_index: int
) -> dict:
    """Fetch rich feature metadata from Neuronpedia.

    Returns a dict with keys: ``description``, ``frac_nonzero``,
    ``max_act_approx``, ``url``, ``embed_url``.
    """
    base = "https://neuronpedia.org"
    url = f"{base}/{model_id}/{sae_id}/{feature_index}"
    embed_url = f"{url}?embed=true"
    result: dict = {
        "description": None,
        "frac_nonzero": None,
        "max_act_approx": None,
        "url": url,
        "embed_url": embed_url,
    }
    api_url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"
    try:
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        explanations = data.get("explanations")
        if explanations and len(explanations) > 0:
            result["description"] = explanations[0].get("description")
        if "frac_nonzero" in data:
            result["frac_nonzero"] = data["frac_nonzero"]
        if "maxActApprox" in data:
            result["max_act_approx"] = data["maxActApprox"]
    except (requests.RequestException, ValueError, KeyError):
        pass
    return result


def get_neuronpedia_features_data(
    model_id: str,
    sae_id: str,
    feature_indices: list[int],
    delay: float = 0.1,
) -> dict[int, dict]:
    """Fetch rich feature metadata for multiple features with rate limiting.

    Returns ``{feature_index: data_dict}``.
    """
    results: dict[int, dict] = {}
    for i, idx in enumerate(feature_indices):
        results[idx] = get_neuronpedia_feature_data(model_id, sae_id, idx)
        if i < len(feature_indices) - 1:
            time.sleep(delay)
    return results
