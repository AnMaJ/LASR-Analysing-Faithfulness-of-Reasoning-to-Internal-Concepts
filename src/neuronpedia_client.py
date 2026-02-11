from src.data_classes import NeuronpediaFeature
import requests
from IPython.display import IFrame, display

# Class used to interact with Neuonpedia API (https://www.neuronpedia.org/)
# Main functionalities are:
# 1) Initialization with model and SAE identifiers
# 2) Fetching feature information (description, activation density, max activation, examples)
# 3) Returning dashboard URLs for features (to manually check the dashboard if needed)

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