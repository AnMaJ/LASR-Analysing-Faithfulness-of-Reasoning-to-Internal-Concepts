"""LASR — Analysing Faithfulness of Reasoning to Internal Concept Use."""

from src.configs import (
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    NeuronpediaFeature,
    PromptStyle,
    SAEConfig,
)
from src.feature import Feature, create_features
from src.denoiser import Denoiser, DenoisingConfig, DenoisingMethod, denoise
from src.aggregator import Aggregator
from src.neuronpedia_client import NeuronpediaClient, build_sae_id

# Heavy imports (require torch, transformers, huggingface_hub) are not
# eagerly loaded here.  Import them directly when needed:
#   from src.SAE import JumpReLUSAE
#   from src.gemma_model import GemmaModel

__all__ = [
    "DatasetConfig",
    "InferenceConfig",
    "ModelConfig",
    "NeuronpediaFeature",
    "PromptStyle",
    "SAEConfig",
    "Feature",
    "create_features",
    "Denoiser",
    "DenoisingConfig",
    "DenoisingMethod",
    "denoise",
    "Aggregator",
    "NeuronpediaClient",
    "build_sae_id",
]
