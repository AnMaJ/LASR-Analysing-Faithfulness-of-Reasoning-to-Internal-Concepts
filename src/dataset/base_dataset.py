from abc import ABC
from typing import Optional, Dict

from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset

from src.configs import DatasetConfig, PromptStyle

class BaseDataset(ABC, Dataset):
    """Abstract base class for all evaluation datasets."""

    def __init__(self, config: DatasetConfig):
        self.path = config.path
        self.prompt_style = config.prompt_style
        self.use_chat_template = config.use_chat_template
        self.few_shot = config.few_shot

        if not hasattr(self, '_INSTRUCTIONS_'):
            raise ValueError(
                "Dataset class doesn't have any instruction specified"
            )

        if self.prompt_style not in self._INSTRUCTIONS_:
            raise ValueError(
                f"Prompt style '{self.prompt_style}' not found in "
                f"{self.__class__.__name__}._INSTRUCTIONS_. "
                f"Available: {list(self._INSTRUCTIONS_.keys())}"
            )

        hf_kwargs = config.hf_data_config or {}

        # Load data directly from HF.
        self.data = self.load_dataset(config.path, **hf_kwargs)

    def load_dataset(self, path: str, hf_kwargs: Dict):
        """
        Function to load the data.
        
        Override it in your class for a custom load dataset method
        (e.g. read local CSV)
        """
        data = hf_load_dataset(path, **hf_kwargs)
        print("Data successfully loaded.")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, indx):
        return self.build_prompt(indx)

    def build_prompt(self, indx: int):
        """Build a model-ready prompt for sample *indx*.

        Subclasses that support prompt formatting must override this method.

        Args:
            indx: Integer index of the sample.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement build_prompt()"
        )
    
    def parse_model_answer(self, response: str) -> Optional[str]:
        """Extract the predicted answer letter from a BBQ model response.

        Applies a cascade of regex patterns to locate the chosen option
        (A, B, or C).

        Args:
            response: The full model-generated response string.

        Returns:
            Final answer
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement parse_model_answer()"
        )