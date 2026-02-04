import torch
from torch.utils.data import Dataset
import json
import pandas as pd

class ReasoningDataset(Dataset):
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.data = self.__load_dataset__(name, path)
    
    def __load_dataset__(self, name: str, path: str) -> Dataset:
        if name == "ECQA":
            # reading the jsonl file at the provided path
            self.data = pd.read_parquet(path)
            
            return self.data
        
        elif name == "e-SNLI":
            return ValueError("e-SNLI dataset loading not implemented yet.")
        elif name == "BBQ":
            return ValueError("BBQ dataset loading not implemented yet.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, indx):
        item = self.data.iloc[indx]
        return item