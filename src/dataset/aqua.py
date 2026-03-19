"""AQuA-RAT dataset class.

Loaded from ``aqua_rat`` on HuggingFace, which is the same underlying dataset
as divelab/aqua but with a cleaner schema (options as a separate list field).

Reference: https://huggingface.co/datasets/divelab/aqua
"""

import re
from typing import Optional

from src.dataset.base_dataset import BaseDataset, PromptStyle
from src.dataset.logiqa import _extract_mc_letter  # shared extraction logic

_OPTION_LABELS = ["A", "B", "C", "D", "E"]


class AQuA_Dataset(BaseDataset):
    """AQuA-RAT algebraic word-problem dataset (5-option MCQ).

    Loaded from ``aqua_rat`` on HuggingFace (equivalent to ``divelab/aqua``).
    Each example has a ``question`` (str), ``options`` (list of strings in
    ``"A)…"`` format), ``rationale`` (str), and ``correct`` (letter str).

    Attributes:
        _INSTRUCTIONS_: Prompt instruction text keyed by PromptStyle.
        _OPTION_LABELS_: ``["A", "B", "C", "D", "E"]``.
    """

    _INSTRUCTIONS_ = {
        PromptStyle.ONE_WORD_NO_TAGS: (
            "Solve the math problem and reply with only A, B, C, D, or E."
        ),
        PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS: (
            "Solve the math problem step by step, then clearly state your "
            "final answer as A, B, C, D, or E."
        ),
    }
    _OPTION_LABELS_ = _OPTION_LABELS

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, indx: int):
        """Build a multiple-choice prompt for the example at *indx*.

        Options are reformatted from ``"A)value"`` to ``"A) value"`` for
        readability.

        Args:
            indx: Integer index.

        Returns:
            List of chat dicts when ``use_chat_template`` is True, else str.
        """
        ex = self.data[indx]

        def _fmt_option(opt: str) -> str:
            # "A)value" -> "A) value", leave anything else unchanged
            m = re.match(r"^([A-E])\)(.*)", opt)
            return f"{m.group(1)}) {m.group(2).strip()}" if m else opt

        opts_text = "\n".join(_fmt_option(opt) for opt in ex["options"])
        instruction = self._INSTRUCTIONS_[self.prompt_style]
        content = f"Question: {ex['question']}\n\nOptions:\n{opts_text}"

        if self.use_chat_template:
            return [{"role": "user", "content": f"{instruction}\n\n{content}"}]
        return f"{instruction}\n\n{content}"

    # ------------------------------------------------------------------
    # Answer helpers
    # ------------------------------------------------------------------

    def get_correct_letter(self, indx: int) -> str:
        """Return the gold answer letter (A–E) for example *indx*."""
        return self.data[indx]["correct"].strip().upper()

    def parse_model_answer(self, response: str) -> Optional[str]:
        is_cot = self.prompt_style == PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS
        return _extract_mc_letter(response, valid="ABCDE", is_cot=is_cot)
