from __future__ import annotations

import re

import pandas as pd

from src.dataset.base_dataset import BaseDataset, PromptStyle

class ECQA_Dataset(BaseDataset):
    """Explainable Common-sense Question Answering (ECQA) dataset.

    Each sample is a five-option multiple-choice question.  Supports
    ``"ONE_WORD"`` and ``"CHAIN_OF_THOUGHT"`` prompt styles and provides
    helpers for extracting the predicted answer letter from model output.

    Attributes:
        _INSTRUCTIONS_: Mapping from prompt-style key to instruction text.
        _OPTION_LABELS_: The canonical option letters ``["A", "B", "C", "D", "E"]``.
    """

    _INSTRUCTIONS_ = {
        PromptStyle.ONE_WORD_TAGS: "Answer with only A, B, C, D, or E.\n",
        PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS: (
            "Please think step by step before giving your final answer. "
            "Consider what information is provided and what assumptions might be involved. "
            "After your reasoning, clearly state your final answer as A, B, C, D, or E.\n"
        ),
    }
    _OPTION_LABELS_ = ["A", "B", "C", "D", "E"]

    def load_dataset(self, path: str, config: dict):
        """Load the ECQA dataset from a Parquet file.

        Args:
            path: Path to a ``.parquet`` file containing the ECQA data.
            config: Unused; kept for interface compatibility.

        Returns:
            pandas.DataFrame: The loaded ECQA data.
        """
        print(f"Loading ECQA dataset from {path}...")
        data = pd.read_parquet(path)
        print(f"ECQA dataset loaded successfully with {len(data)} samples.")
        return data

    def build_prompt(self, indx: int) -> str:
        """Build a multiple-choice prompt for the given sample index.

        Args:
            indx: Integer index of the sample in the dataset.

        Returns:
            str: A prompt containing the instruction, question, and the five
            labelled answer options.
        """
        row = self.data.iloc[indx]
        options_text = "\n".join(
            f"{self._OPTION_LABELS_[i].lower()}) {row[f'q_op{i+1}']}" for i in range(5)
        )
        return (
            f"{self._INSTRUCTIONS_[self.prompt_style]}\n\n"
            f"Question: {row['q_text']}\n\nOptions:\n{options_text}"
        )

    def get_correct_letter(self, row) -> str | None:
        """Return the option letter (A–E) that matches the gold answer.

        Args:
            row: A ``pandas.Series`` (or dict-like) with keys ``q_ans`` and
                ``q_op1`` … ``q_op5``.

        Returns:
            The matching option letter, or ``None`` if no option matches.
        """
        answer = row["q_ans"].strip().lower()
        for i in range(5):
            if row[f"q_op{i+1}"].strip().lower() == answer:
                return self._OPTION_LABELS_[i]
        return None

    @staticmethod
    def extract_answer_letter_cot(text: str) -> str | None:
        """Extract the predicted answer letter from a chain-of-thought response.

        Applies a cascade of regex patterns—from the most explicit
        (e.g. "Final answer: B") down to the last bare A–E token—to
        robustly locate the model's chosen option.

        Args:
            text: The full model-generated response string.

        Returns:
            A single uppercase letter ``"A"``–``"E"``, or ``None`` if no
            answer could be identified.
        """
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
                return matches[-1].group(1)

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

    @staticmethod
    def extract_answer_letter_no_cot(text: str) -> str | None:
        """Extract the predicted answer letter from a short (non-CoT) response.

        Expects the response to be a single letter or a very short string
        that contains exactly one A–E token.

        Args:
            text: The model-generated response string.

        Returns:
            A single uppercase letter ``"A"``–``"E"``, or ``None`` if no
            answer could be identified.
        """
        text = text.strip()
        if text.upper() in ECQA_Dataset._OPTION_LABELS_:
            return text.upper()
        match = re.search(r'\b([A-E])\b', text.upper())
        return match.group(1) if match else None