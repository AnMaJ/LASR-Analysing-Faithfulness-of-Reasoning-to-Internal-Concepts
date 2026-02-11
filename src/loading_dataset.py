import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from datasets import load_dataset


    
class ECQA_Dataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = self.__load_dataset__(path)
    
    def __load_dataset__(self, path: str) -> Dataset:
        print(f"Loading ECQA dataset from {path}...")
        self.data = pd.read_parquet(path)
        print(f"ECQA dataset loaded successfully with {len(self.data)} samples.")
        return self.data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, indx):
        item = self.data.iloc[indx]
        return item

        
        
class ESNLI_Dataset(Dataset):
    
    _INSTRUCTIONS_ = {
        "ONE_WORD": (
            "Determine if statement B is an entailment, contradiction or neutral "
            "with respect to statement A. Answer with a single word: entailment, "
            "contradiction, or neutral.\n"
        ),
        "CHAIN_OF_THOUGHT": (
            "Determine if statement B is an entailment, contradiction or neutral. "
            "Reason step by step and finally provide a one-word answer.\n"
        ),
    }
    def __init__(self, path: str):
        self.path = path
        self.data = self.__load_dataset__(path)
        
        
    def __load_dataset__(self, path: str) -> Dataset:
        print(f"Loading e-SNLI dataset from {path}...")
        self.data = pd.read_csv(path)
        print(f"e-SNLI dataset loaded successfully with {len(self.data)} samples.")
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, indx):
        item = self.data.iloc[indx]
        return item
    
    def build_few_shot_examples(self) -> str:
        """Generate the few-shot instruction-tuning example string from the dataset."""
        unique_samples = self.data.drop_duplicates(subset=["gold_label"]).copy()
        unique_samples["formatted_input"] = unique_samples.apply(
            lambda x: (
                f"A: {x['Sentence1']}\n"
                f"B: {x['Sentence2']}\n"
                f"{x['Explanation_1']}\n"
                f"Label: {x['gold_label']}"
            ),
            axis=1,
        )
        return "\n".join(
            f"Example {i+1}:\n{text}"
            for i, text in enumerate(unique_samples["formatted_input"])
        )
        
    def build_prompts(self, few_shot_examples: str, prompt_style: str,) -> pd.Series:
        """Build the full prompt for each row in the dataset."""
        instruction = self._INSTRUCTIONS_[prompt_style]
        return (
            "<start_of_turn>user "
            + instruction
            + few_shot_examples
            + "\nA: " + self.data["Sentence1"]
            + "\nB: " + self.data["Sentence2"]
            + "\n<end_of_turn>model"
        )
    
    
    
class BBQ_Dataset(Dataset):
    def __init__(self, config: dict):
        self.bbq_category = config.bbq_category
        self.bbq_condition = config.bbq_condition
        self.split = config.bbq_split
        self.data = self.__load_dataset__(self.__bbq_config__(), config.bbq_path)
        
    def __bbq_config__(self) -> str:
        """Construct BBQ dataset config name."""
        return f"{self.bbq_category}_{self.bbq_condition}"
        
    def __load_dataset__(self, name: str, path: str) -> Dataset:
        config_name = self.__bbq_config__()
        print(f"Loading BBQ dataset for config: {config_name}...")
        complete_dataset = load_dataset("HiTZ/bbq", config_name)
        self.data = complete_dataset[self.split]
        print(f"BBQ dataset {self.split} split loaded successfully with {len(self.data)} samples.")
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, indx):
        item = self.data[indx]
        return item