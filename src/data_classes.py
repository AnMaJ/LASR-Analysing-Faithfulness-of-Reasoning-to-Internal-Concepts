from typing import List, Dict, Optional
from dataclasses import dataclass, field
import torch


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

@dataclass
class NeuronpediaFeature:
    """Container for feature information from Neuronpedia."""
    feature_idx: int
    description: Optional[str] = None
    frac_nonzero: Optional[float] = None  # Activation density
    max_act_approx: Optional[float] = None  # Max activation value
    max_activating_examples: Optional[List[Dict]] = None
    error: Optional[str] = None

@dataclass
class ModelConfig:
    model_name: str = "google/gemma-3-4b-it"
    device: str = field(default_factory=_default_device)
