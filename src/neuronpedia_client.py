"""Neuronpedia API client for fetching SAE feature explanation labels."""

from __future__ import annotations

import time

import requests
from IPython.display import IFrame, display

from configs import NeuronpediaFeature, SAEConfig


def build_sae_id(config: SAEConfig) -> str:
    """Build a Neuronpedia SAE ID string from a SAEConfig.

    Returns e.g. ``"22-gemmascope-2-res-65k"`` for layer 22, width "65k".
    """
    return f"{config.layer}-gemmascope-2-res-{config.width}"


class NeuronpediaClient:
    """Client for interacting with the Neuronpedia API."""

    BASE_URL = "https://www.neuronpedia.org/api"

    def __init__(self, model_id: str, sae_id: str):
        """
        Initialize the Neuronpedia client.

        Args:
            model_id: Model identifier (e.g., 'gemma-3-4b-it')
            sae_id: SAE identifier (e.g., '22-gemmascope-2-mlp-262k')
        """
        self.model_id = model_id
        self.sae_id = sae_id

    def get_feature(self, feature_idx: int) -> NeuronpediaFeature:
        """Fetch feature information from Neuronpedia.

        API endpoint: GET /api/feature/{modelId}/{layer}/{index}

        Args:
            feature_idx: index of the feature we want to get information about
        """
        url = f"{self.BASE_URL}/feature/{self.model_id}/{self.sae_id}/{feature_idx}"

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 404:
                return NeuronpediaFeature(
                    feature_idx=feature_idx,
                    error="Feature not found on Neuronpedia"
                )

            response.raise_for_status()
            data = response.json()

            # Extract description from explanations if available
            description = None
            if 'explanations' in data and data['explanations']:
                description = data['explanations'][0].get('description', None)

            # Extract activation density (frac_nonzero)
            frac_nonzero = data.get('frac_nonzero', None)

            # Extract max activation
            max_act_approx = data.get('maxActApprox', None)

            # Extract max activating examples
            max_examples = None
            if 'activations' in data:
                max_examples = data['activations']

            return NeuronpediaFeature(
                feature_idx=feature_idx,
                description=description,
                frac_nonzero=frac_nonzero,
                max_act_approx=max_act_approx,
                max_activating_examples=max_examples
            )

        except requests.exceptions.RequestException as e:
            return NeuronpediaFeature(
                feature_idx=feature_idx,
                error=f"API request failed: {str(e)}"
            )

    def get_label(self, feature_idx: int) -> str | None:
        """Fetch a single feature's explanation label from Neuronpedia.

        Returns the ``description`` field from the first explanation, or ``None``
        if the request fails or no explanations are available.
        """
        url = f"{self.BASE_URL}/feature/{self.model_id}/{self.sae_id}/{feature_idx}"
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

    def get_feature_data(self, feature_idx: int) -> dict:
        """Fetch rich feature metadata from Neuronpedia.

        Returns a dict with keys: ``description``, ``frac_nonzero``,
        ``max_act_approx``, ``url``, ``embed_url``.
        """
        base = "https://neuronpedia.org"
        url = f"{base}/{self.model_id}/{self.sae_id}/{feature_idx}"
        embed_url = f"{url}?embed=true"
        result: dict = {
            "description": None,
            "frac_nonzero": None,
            "max_act_approx": None,
            "url": url,
            "embed_url": embed_url,
        }
        api_url = f"{self.BASE_URL}/feature/{self.model_id}/{self.sae_id}/{feature_idx}"
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

    def get_feature_urls(
        self, feature_indices: list[int], embed: bool = False
    ) -> dict[int, str]:
        """Build Neuronpedia feature page URLs for a list of feature indices.

        Parameters
        ----------
        feature_indices:
            Feature indices to generate URLs for.
        embed:
            If ``True``, append ``?embed=true`` for iframe embedding.

        Returns ``{feature_index: url_string}``.
        """
        base = "https://neuronpedia.org"
        suffix = "?embed=true" if embed else ""
        return {
            idx: f"{base}/{self.model_id}/{self.sae_id}/{idx}{suffix}"
            for idx in feature_indices
        }

    def get_labels(
        self, feature_indices: list[int], delay: float = 0.1
    ) -> dict[int, str | None]:
        """Fetch explanation labels for multiple features.

        Calls :meth:`get_label` for each index with a small delay
        between requests to avoid rate-limiting.

        Returns ``{feature_index: concept_label_or_None}``.
        """
        labels: dict[int, str | None] = {}
        for i, idx in enumerate(feature_indices):
            labels[idx] = self.get_label(idx)
            if i < len(feature_indices) - 1:
                time.sleep(delay)
        return labels

    def get_features_data(
        self, feature_indices: list[int], delay: float = 0.1
    ) -> dict[int, dict]:
        """Fetch rich feature metadata for multiple features with rate limiting.

        Returns ``{feature_index: data_dict}``.
        """
        results: dict[int, dict] = {}
        for i, idx in enumerate(feature_indices):
            results[idx] = self.get_feature_data(idx)
            if i < len(feature_indices) - 1:
                time.sleep(delay)
        return results

    def get_dashboard_url(self, feature_idx: int) -> str:
        """Get the Neuronpedia dashboard URL for a feature."""
        return f"https://neuronpedia.org/{self.model_id}/{self.sae_id}/{feature_idx}"

    def get_dashboard_embed(self, feature_idx: int) -> str:
        """Get embeddable dashboard URL for a feature."""
        base = self.get_dashboard_url(feature_idx)
        return f"{base}?embed=true&embedexplanation=true&embedplots=true&embedtest=true"

    def display_feature_dashboard(self, feature_idx: int, height: int = 400):
        """Display an embedded Neuronpedia dashboard for a feature."""
        url = self.get_dashboard_embed(feature_idx)
        display(IFrame(url, width=1000, height=height))
