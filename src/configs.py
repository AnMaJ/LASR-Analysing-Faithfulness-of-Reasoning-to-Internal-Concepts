from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Dataset Configuration

class PromptStyle(Enum):
    ONE_WORD_NO_TAGS = "one_word_no_tags"
    ONE_WORD_TAGS = "one_word_tags"
    CHAIN_OF_THOUGHT_NO_TAGS = "chain_of_thought_no_tags"
    CHAIN_OF_THOUGHT_TAGS = "chain_of_thought_tags"

@dataclass
class DatasetConfig:
    """
    Configuration for dataset class.

    Args:
        path: path on HF where the dataset is stored.
        prompt_style: how to build the prompt. The key is used to retrieve template from
        _INSTRUCTIONS_ data in the dataset object.
        use_chat_template: whether to apply, after the instruction template, the chat template specific 
        for the selected model. 
        hf_data_config: Optional. Contains extra arguments for loading the dataset from HF (such as 
        split, categories, etc.). It it specific for each dataset.
        few_shot: whether to append an example to the prompt.
    """
    path: str
    prompt_style: PromptStyle
    use_chat_template: bool
    hf_data_config: dict | None = None
    few_shot: bool = False

    def __post_init__(self):
        if isinstance(self.prompt_style, str):
            try:
                self.prompt_style = PromptStyle(self.prompt_style)
            except ValueError:
                valid = [e.value for e in PromptStyle]
                print(f"Invalid prompt_style '{self.prompt_style}'. Must be one of {valid}")
                raise
        if not isinstance(self.prompt_style, PromptStyle):
            valid = [e.value for e in PromptStyle]
            print(f"Invalid prompt_style '{self.prompt_style}'. Must be one of {valid}")
            raise ValueError(f"prompt_style must be a PromptStyle enum, got {type(self.prompt_style)}")


@dataclass
class ModelConfig:
    model_name: str
    device: str = field(default_factory=_default_device)


@dataclass
class InferenceConfig:
    batch_size: int = 8
    max_new_tokens: int = 256
    downsample_rate: int = 10


@dataclass
class SAEConfig:
    repo_id: str
    sae_type: str # resid_post, mlp_out, attn_out
    layer: int
    width: str # 16k, 65k, 262k, 1m
    l0: str

    @property
    def sae_path(self) -> str:
        """Construct the SAE path for HuggingFace download."""
        return f"{self.sae_type}/layer_{self.layer}_width_{self.width}_l0_{self.l0}/params.safetensors"


@dataclass
class TranscoderConfig:
    repo_id: str = "google/gemma-scope-2-27b-it"
    layer: int = 31
    width: str = "262k"  # 262k, etc.
    l0: str = "medium"   # medium, etc.
    affine: bool = False  # use the affine variant

    @property
    def transcoder_path(self) -> str:
        """Construct the transcoder path for HuggingFace download."""
        suffix = "_affine" if self.affine else ""
        return f"transcoder/layer_{self.layer}_width_{self.width}_l0_{self.l0}{suffix}/params.safetensors"


@dataclass
class DenoisingConfig:
    """Configuration for a denoising method.

    Args:
        method: Name of the denoising strategy (must match a method on
            :class:`~src.denoiser.Denoiser` decorated with
            ``@_denoising_method``).
        params: Keyword arguments forwarded to the chosen method.
    """
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuronpediaFeature:
    """Container for feature information from Neuronpedia."""
    feature_idx: int
    description: str | None = None
    frac_nonzero: float | None = None
    max_act_approx: float | None = None
    max_activating_examples: list[dict] | None = None
    url: str | None = None
    embed_url: str | None = None
    error: str | None = None
