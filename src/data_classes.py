from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class NeuronpediaFeature:
    """Container for feature information from Neuronpedia."""
    feature_idx: int
    description: Optional[str] = None
    frac_nonzero: Optional[float] = None  # Activation density
    max_act_approx: Optional[float] = None  # Max activation value
    max_activating_examples: Optional[List[Dict]] = None
    error: Optional[str] = None
