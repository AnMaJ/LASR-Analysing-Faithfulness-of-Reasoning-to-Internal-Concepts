import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import json
import pandas as pd
import re
from datasets import load_dataset


class BaseDataset(ABC, Dataset):

    def __init__(self, path: str = None, config: dict = None):
        self.path = path
        self.data = self.load_dataset(path, config)

    @abstractmethod
    def load_dataset(self, path: str = None, config: dict = None):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        return self.data.iloc[indx] if isinstance(self.data, pd.DataFrame) else self.data[indx]


class ECQA_Dataset(BaseDataset):
    _INSTRUCTIONS_= {
        "ONE_WORD": f"""
          Answer with only A, B, C, D, or E.
          """
          ,
        "CHAIN_OF_THOUGHT": f"""
              Please think step by step before giving your final answer. Consider what information is provided and what assumptions might be involved. After your reasoning, clearly state your final answer as A, B, C, D, or E.
              """,
    }
    _OPTION_LABELS_ = ["A", "B", "C", "D", "E"]
    
    def load_dataset(self, path: str):
        print(f"Loading ECQA dataset from {path}...")
        data = pd.read_parquet(path)
        print(f"ECQA dataset loaded successfully with {len(data)} samples.")
        return data
    
    def extract_answer_letter_cot(text: str) -> str | None:
        text_upper = text.upper().strip()

        final_patterns = [
            r'FINAL\s+ANSWER\s*[:\-\s]*\(?([A-E])\)?',
            r'THE\s+ANSWER\s+IS\s*[:\-\s]*\(?([A-E])\)?',
            r'(?:MY\s+)?ANSWER\s*[:\-\s]+\(?([A-E])\)?',
            r'CORRECT\s+(?:OPTION|ANSWER)\s*[:\-\s]*\(?([A-E])\)?',
            r'I\s+(?:WOULD\s+)?CHOOSE\s*[:\-\s]*\(?([A-E])\)?',
        ]
        for pattern in final_patterns:
            matches = list(re.finditer(pattern, text_upper))
            if matches:
                return matches[-1].group(1)  # take the LAST match (closest to end)


        tail = text_upper[-50:]
        tail_match = re.search(r'\b([A-E])\b(?:\s*[\.\):]?\s*$)', tail)
        if tail_match:
            return tail_match.group(1)


        bold_match = re.search(r'\*\*([A-E])\*\*', text_upper)
        if bold_match:
            return bold_match.group(1)

        all_matches = re.findall(r'\b([A-E])\b', text_upper)
        if all_matches:
            return all_matches[-1]

        return None
    
    def extract_answer_letter_no_cot(self, text: str) -> str | None:
        text = text.strip()
        if text.upper() in self._OPTION_LABELS_:
            return text.upper()
        match = re.search(r'\b([A-E])\b', text.upper())
        return match.group(1) if match else None
    
    def get_correct_letter(self, row):
        
        answer = row["q_ans"].strip().lower()
        for i in range(5):
            if row[f"q_op{i+1}"].strip().lower() == answer:
                return self._OPTION_LABELS_[i]
        return None
    
    def build_prompt(self, indx: int, prompt_style: str) -> str:
        """Build a chain-of-thought prompt for the given ECQA row."""
        row = self.__getitem__(indx)
        options_text = "\n".join(
            f"{self._OPTION_LABELS_[i].lower()}) {row[f'q_op{i+1}']}" for i in range(5)
        )
        final_prompt = self._INSTRUCTIONS_[prompt_style] + "\n\n" + f"Question: {row['q_text']}\n\nOptions:\n{options_text}"
        
        return final_prompt


class ESNLI_Dataset(BaseDataset):

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

    def load_dataset(self, path: str):
        print(f"Loading e-SNLI dataset from {path}...")
        data = pd.read_csv(path)
        print(f"e-SNLI dataset loaded successfully with {len(data)} samples.")
        return data

    def build_few_shot_examples(self) -> str:
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

    def build_prompts(self, few_shot_examples: str, prompt_style: str) -> pd.Series:
        instruction = self._INSTRUCTIONS_[prompt_style]
        return (
            "<start_of_turn>user "
            + instruction
            + few_shot_examples
            + "\nA: " + self.data["Sentence1"]
            + "\nB: " + self.data["Sentence2"]
            + "\n<end_of_turn>model"
        )


class BBQ_Dataset(BaseDataset):
    
    _INSTRUCTIONS_= {
        "ONE_WORD": f"""
          Answer with only A, B, or C.
          """
          ,
        "CHAIN_OF_THOUGHT": f"""
              Please think step by step before giving your final answer. Consider what information is provided and what assumptions might be involved. After your reasoning, clearly state your final answer as A, B, or C.
              """,
    }
    
    def __init__(self, config: dict):
        self.bbq_category = config.bbq_category
        self.bbq_condition = config.bbq_condition
        self.split = config.bbq_split
        # Call the parent with path + extra kwargs
        super().__init__(path=config.bbq_path, config=config)

    def load_dataset(self):
        config_name = f"{self.bbq_category}_{self.bbq_condition}"
        print(f"Loading BBQ dataset for config: {config_name}...")
        complete_dataset = load_dataset("HiTZ/bbq", config_name)
        data = complete_dataset[self.split]
        print(f"BBQ dataset {self.split} split loaded successfully with {len(data)} samples.")
        return data
    
    def format_bbq_question(example: dict) -> str:
        """Format a BBQ example into a question with answer choices."""
        context = example.get('context', '')
        question = example.get('question', '')

        # Get answer choices
        ans0 = example.get('ans0', '')
        ans1 = example.get('ans1', '')
        ans2 = example.get('ans2', '')

        formatted = f"""
            Context: {context}

            Question: {question}

            Answer choices:
            A) {ans0}
            B) {ans1}
            C) {ans2}
            
            """

        return formatted
    
    def parse_model_answer(response: str) -> Optional[str]:
        """Parse the model's final answer (A, B, or C) from the response.
    
        Looks for patterns like "Final Answer: A", "answer is A", "A)", etc.
        """
        
        response_lower = response.lower()
        
        # Pattern 1: "Final Answer: X" or "final answer is X"
        match = re.search(r'final answer[:\s]+([abc])\b', response_lower)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: "the answer is X" or "answer: X"
        match = re.search(r'answer[:\s]+(?:is\s+)?([abc])\b', response_lower)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: Look for standalone "A)", "B)", "C)" at the end
        match = re.search(r'\b([abc])\)?[\s.]*$', response_lower.strip())
        if match:
            return match.group(1).upper()
        
        # Pattern 4: "I choose X" or "I select X"
        match = re.search(r'i (?:choose|select|pick)\s+([abc])\b', response_lower)
        if match:
            return match.group(1).upper()
        
        # Pattern 5: For direct answers, check if response starts with A, B, or C
        match = re.match(r'^([abc])\b', response_lower.strip())
        if match:
            return match.group(1).upper()
        
        return None