from dataclasses import dataclass, field
from enum import Enum

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
