from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch


class PromptStyle(Enum):
    ONE_WORD = "one_word"
    CHAIN_OF_THOUGHT = "chain_of_thought"


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class ModelConfig:
    model_name: str = "google/gemma-3-4b-it"
    device: str = field(default_factory=_default_device)


@dataclass
class InferenceConfig:
    batch_size: int = 8
    max_new_tokens: int = 256
    downsample_rate: int = 10


@dataclass
class SAEConfig:
    layer: int = 22
    width: str = "262k"
    l0: str = "medium"
    repo_id: str = "google/gemma-scope-2-4b-pt"


@dataclass
class NeuronpediaFeature:
    """Container for feature information from Neuronpedia."""
    feature_idx: int
    description: Optional[str] = None
    frac_nonzero: Optional[float] = None  # Activation density
    max_act_approx: Optional[float] = None  # Max activation value
    max_activating_examples: Optional[List[Dict]] = None
    url: Optional[str] = None
    embed_url: Optional[str] = None
    error: Optional[str] = None
