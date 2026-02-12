"""
Dataset abstractions for evaluation benchmarks.

Provides a common interface (BaseDataset) and concrete implementations
for ECQA, e-SNLI and BBQ, with optional chat-template prompt formatting.
"""

from abc import ABC, abstractmethod
from typing import Optional
import re
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for all evaluation datasets.

    Combines :class:`torch.utils.data.Dataset` with a simple prompt-formatting
    layer so that the same object can feed both raw-row access and
    ready-to-send chat prompts.

    Attributes:
        path: Filesystem or URL path used to load the data.
        with_prompt_formatting: When ``True``, :meth:`__getitem__` returns a
            formatted prompt string instead of the raw row.
        prompt_style: Key into the subclass ``_INSTRUCTIONS_`` dict that
            selects the prompting strategy (e.g. ``"ONE_WORD"``,
            ``"CHAIN_OF_THOUGHT"``).
        data: The loaded dataset, typically a :class:`pandas.DataFrame` or a
            HuggingFace :class:`datasets.Dataset`.
    """

    def __init__(
        self,
        path: str = None,
        config: dict = None,
        with_prompt_formatting: bool = False,
        prompt_style: str = None,
    ):
        """Initialise the dataset and optionally enable prompt formatting.

        Args:
            path: Filesystem or URL path passed through to :meth:`load_dataset`.
            config: Arbitrary configuration dict forwarded to :meth:`load_dataset`.
            with_prompt_formatting: If ``True``, :meth:`__getitem__` delegates
                to :meth:`build_prompt` so that each item is a ready-to-send
                prompt string.
            prompt_style: Prompting strategy key (must exist in the subclass
                ``_INSTRUCTIONS_`` dict when *with_prompt_formatting* is ``True``).

        Raises:
            ValueError: If *with_prompt_formatting* is ``True`` but
                *prompt_style* is not a valid key in ``_INSTRUCTIONS_``.
        """
        self.path = path
        self.with_prompt_formatting = with_prompt_formatting
        self.prompt_style = prompt_style

        if with_prompt_formatting:
            if not hasattr(self, '_INSTRUCTIONS_') or prompt_style not in self._INSTRUCTIONS_:
                valid = list(self._INSTRUCTIONS_.keys()) if hasattr(self, '_INSTRUCTIONS_') else []
                raise ValueError(
                    f"prompt_style must be one of {valid} "
                    f"when with_prompt_formatting=True, got '{prompt_style}'"
                )

        self.data = self.load_dataset(path, config)

    @abstractmethod
    def load_dataset(self, path: str = None, config: dict = None):
        """Load the underlying data from *path* and/or *config*.

        Args:
            path: Filesystem or URL path to the data source.
            config: Optional configuration dict with loader-specific options.

        Returns:
            The loaded dataset (typically a ``DataFrame`` or HuggingFace
            ``Dataset``).
        """
        ...

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Total sample count.
        """
        return len(self.data)

    def __getitem__(self, indx):
        """Retrieve a single sample by index.

        If *with_prompt_formatting* is enabled, returns the formatted prompt
        string produced by :meth:`build_prompt`. Otherwise returns the raw
        data row.

        Args:
            indx: Integer index of the sample.

        Returns:
            A formatted prompt string **or** the raw row (``pandas.Series``
            / dict) depending on *with_prompt_formatting*.
        """
        if self.with_prompt_formatting:
            return self.build_prompt(indx)
        return self.data.iloc[indx] if isinstance(self.data, pd.DataFrame) else self.data[indx]

    def build_prompt(self, indx: int) -> str:
        """Build a model-ready prompt for sample *indx*.

        Subclasses that support prompt formatting must override this method.

        Args:
            indx: Integer index of the sample.

        Returns:
            str: The formatted prompt string.

        Raises:
            NotImplementedError: If the subclass has not overridden this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement build_prompt()"
        )


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
        "ONE_WORD": "Answer with only A, B, C, D, or E.\n",
        "CHAIN_OF_THOUGHT": (
            "Please think step by step before giving your final answer. "
            "Consider what information is provided and what assumptions might be involved. "
            "After your reasoning, clearly state your final answer as A, B, C, D, or E.\n"
        ),
    }
    _OPTION_LABELS_ = ["A", "B", "C", "D", "E"]

    def load_dataset(self, path: str = None, config: dict = None):
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

    def get_correct_letter(self, row) -> Optional[str]:
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
    def extract_answer_letter_cot(text: str) -> Optional[str]:
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
    def extract_answer_letter_no_cot(text: str) -> Optional[str]:
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


class ESNLI_Dataset(BaseDataset):
    """Explainable Stanford Natural Language Inference (e-SNLI) dataset.

    Each sample is a sentence pair labelled as *entailment*, *contradiction*,
    or *neutral*.  Prompt formatting includes automatically generated
    few-shot examples (one per label) drawn from the dataset itself.

    Attributes:
        _INSTRUCTIONS_: Mapping from prompt-style key to instruction text.
    """

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

    def load_dataset(self, path: str = None, config: dict = None):
        """Load the e-SNLI dataset from a CSV file.

        Args:
            path: Path to a ``.csv`` file containing the e-SNLI data.
            config: Unused; kept for interface compatibility.

        Returns:
            pandas.DataFrame: The loaded e-SNLI data.
        """
        print(f"Loading e-SNLI dataset from {path}...")
        data = pd.read_csv(path)
        print(f"e-SNLI dataset loaded successfully with {len(data)} samples.")
        return data

    def build_prompt(self, indx: int) -> str:
        """Build a Gemma-chat-formatted NLI prompt with few-shot examples.

        Args:
            indx: Integer index of the sample in the dataset.

        Returns:
            str: A prompt containing the instruction, few-shot demonstrations,
            and the target sentence pair wrapped in Gemma chat turn markers.
        """
        row = self.data.iloc[indx]
        few_shot = self._build_few_shot_examples()
        instruction = self._INSTRUCTIONS_[self.prompt_style]
        return (
            f"<start_of_turn>user {instruction}"
            f"{few_shot}\n"
            f"A: {row['Sentence1']}\n"
            f"B: {row['Sentence2']}\n"
            f"<end_of_turn>model"
        )

    def _build_few_shot_examples(self) -> str:
        """Construct few-shot demonstrations (one per unique gold label).

        Returns:
            str: Concatenated, numbered examples each containing a sentence
            pair, an explanation, and the gold label.
        """
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


class BBQ_Dataset(BaseDataset):
    """Bias Benchmark for QA (BBQ) dataset.

    Each sample is a three-option multiple-choice question designed to
    probe social biases. The dataset is loaded from HuggingFace Hub
    (``HiTZ/bbq``) and filtered by category, condition, and split.

    Attributes:
        _INSTRUCTIONS_: Mapping from prompt-style key to instruction text.
        bbq_category: Bias category string (e.g. ``"age"``, ``"gender"``).
        bbq_condition: Condition string (e.g. ``"ambiguous"``, ``"disambiguous"``).
        split: Dataset split (e.g. ``"test"``).
    """

    _INSTRUCTIONS_ = {
        "ONE_WORD": "Answer with only A, B, or C.\n",
        "CHAIN_OF_THOUGHT": (
            "Please think step by step before giving your final answer. "
            "Consider what information is provided and what assumptions might be involved. "
            "After your reasoning, clearly state your final answer as A, B, or C.\n"
        ),
    }

    def __init__(
        self,
        config: dict,
        with_prompt_formatting: bool = False,
        prompt_style: str = None,
    ):
        """Initialise BBQ with category/condition/split from *config*.

        Args:
            config: Object (namespace or dict) with attributes ``bbq_category``,
                ``bbq_condition``, ``bbq_split``, and ``bbq_path``.
            with_prompt_formatting: If ``True``, :meth:`__getitem__` returns
                formatted prompt strings.
            prompt_style: Prompting strategy key (``"ONE_WORD"`` or
                ``"CHAIN_OF_THOUGHT"``).
        """
        self.bbq_category = config.bbq_category
        self.bbq_condition = config.bbq_condition
        self.split = config.bbq_split
        super().__init__(
            path=config.bbq_path,
            config=config,
            with_prompt_formatting=with_prompt_formatting,
            prompt_style=prompt_style,
        )

    def load_dataset(self, path: str = None, config: dict = None):
        """Load a BBQ split from HuggingFace Hub.

        Downloads the ``HiTZ/bbq`` dataset for the configuration
        ``{bbq_category}_{bbq_condition}`` and selects the stored split.

        Args:
            path: Unused; the HuggingFace Hub identifier is hard-coded.
            config: Unused; category/condition/split are read from instance
                attributes set in :meth:`__init__`.

        Returns:
            datasets.Dataset: The requested split of the BBQ configuration.
        """
        config_name = f"{self.bbq_category}_{self.bbq_condition}"
        print(f"Loading BBQ dataset for config: {config_name}...")
        complete_dataset = hf_load_dataset("HiTZ/bbq", config_name)
        data = complete_dataset[self.split]
        print(f"BBQ dataset {self.split} split loaded successfully with {len(data)} samples.")
        return data

    def build_prompt(self, indx: int) -> str:
        """Build a three-option multiple-choice prompt for the given sample.

        Args:
            indx: Integer index of the sample in the dataset.

        Returns:
            str: A prompt containing the instruction, context passage,
            question, and the three labelled answer options.
        """
        example = self.data[indx]
        instruction = self._INSTRUCTIONS_[self.prompt_style]
        return (
            f"{instruction}\n"
            f"Context: {example.get('context', '')}\n\n"
            f"Question: {example.get('question', '')}\n\n"
            f"Answer choices:\n"
            f"A) {example.get('ans0', '')}\n"
            f"B) {example.get('ans1', '')}\n"
            f"C) {example.get('ans2', '')}\n"
        )

    @staticmethod
    def parse_model_answer(response: str) -> Optional[str]:
        """Extract the predicted answer letter from a BBQ model response.

        Applies a cascade of regex patterns to locate the chosen option
        (A, B, or C).

        Args:
            response: The full model-generated response string.

        Returns:
            A single uppercase letter ``"A"``, ``"B"``, or ``"C"``, or
            ``None`` if no answer could be identified.
        """
        response_lower = response.lower()
        patterns = [
            r'final answer[:\s]+([abc])\b',
            r'answer[:\s]+(?:is\s+)?([abc])\b',
            r'\b([abc])\)?[\s.]*$',
            r'i (?:choose|select|pick)\s+([abc])\b',
            r'^([abc])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, response_lower.strip())
            if match:
                return match.group(1).upper()
        return None