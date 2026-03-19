"""GSM8K dataset class.

Loaded from ``openai/gsm8k`` on HuggingFace (config name ``"main"``).
Each example has a ``question`` (str) and an ``answer`` (str) that contains
the chain-of-thought working followed by the final answer in the canonical
``#### <number>`` format.

Reference: https://huggingface.co/datasets/openai/gsm8k
"""

import re
from typing import Dict, List, Optional

from src.dataset.base_dataset import BaseDataset, PromptStyle


def _extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a model or gold response.

    Tries the canonical GSM8K ``#### <number>`` marker first, then falls
    back to the last standalone number in the text.

    Args:
        text: Response string to parse.

    Returns:
        Numeric string with commas removed, or ``None`` if nothing found.
    """
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def _normalize_number(ans: str) -> str:
    """Normalise a numeric string for exact-match comparison.

    Strips commas and converts to a canonical float/int representation so
    that ``"3.0"`` and ``"3"`` compare equal.

    Args:
        ans: Raw numeric string.

    Returns:
        Canonical string representation.
    """
    ans = ans.replace(",", "").strip()
    try:
        return str(float(ans)) if "." in ans else str(int(ans))
    except ValueError:
        return ans


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class GSM8K_Dataset(BaseDataset):
    """GSM8K grade-school math dataset (free-form numeric answer).

    Loaded from ``openai/gsm8k`` (config ``"main"``) on HuggingFace.
    Each example has a ``question`` (str) and an ``answer`` (str) where
    the gold final answer is written after a ``####`` marker.

    Unlike the MCQ datasets, answers are numeric strings rather than
    option letters.  Use ``get_correct_answer`` to retrieve the gold
    value and ``parse_model_answer`` to extract a predicted value from a
    model response.

    Attributes:
        _INSTRUCTIONS_: Prompt instruction text keyed by PromptStyle.
    """

    _INSTRUCTIONS_ = {
        PromptStyle.ONE_WORD_NO_TAGS: (
            "Solve the math problem. Reply with only the final numeric answer, "
            "no working."
        ),
        PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS: (
            "Solve the math problem step by step. "
            "At the end of your response, write the final numeric answer on its "
            "own line in the format: #### <number>"
        ),
    }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataset(self, path: str, hf_kwargs: Dict) -> List[Dict]:
        """Load a GSM8K split from HuggingFace.

        Args:
            path:      HuggingFace dataset ID (e.g. ``"openai/gsm8k"``).
            hf_kwargs: Optional dict; ``split`` key selects the split
                       (default: ``"test"``).  The ``"main"`` config name
                       is added automatically.

        Returns:
            List of example dicts with keys ``question`` and ``answer``.
        """
        from datasets import load_dataset as hf_load_dataset

        split = hf_kwargs.get("split", "test")
        print(f"Loading GSM8K '{split}' split from HuggingFace …")
        ds = hf_load_dataset(path, "main", split=split)
        data = list(ds)
        print(f"GSM8K {split}: {len(data)} examples loaded.")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, indx):
        return self.build_prompt(indx)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, indx: int):
        """Build a prompt for the example at *indx*.

        Args:
            indx: Integer index.

        Returns:
            List of chat dicts when ``use_chat_template`` is True, else str.
        """
        ex = self.data[indx]
        instruction = self._INSTRUCTIONS_[self.prompt_style]
        content = f"Question: {ex['question']}"

        if self.use_chat_template:
            return [{"role": "user", "content": f"{instruction}\n\n{content}"}]
        return f"{instruction}\n\n{content}"

    # ------------------------------------------------------------------
    # Answer helpers
    # ------------------------------------------------------------------

    def get_correct_answer(self, indx: int) -> Optional[str]:
        """Return the gold numeric answer string for example *indx*.

        Extracts the value after the ``####`` marker in the gold ``answer``
        field.

        Args:
            indx: Integer index.

        Returns:
            Numeric string (commas removed), or ``None`` if unparseable.
        """
        return _extract_numeric_answer(self.data[indx]["answer"])

    def parse_model_answer(self, response: str) -> Optional[str]:
        """Extract the predicted numeric answer from a model response.

        For chain-of-thought responses the ``#### <number>`` marker is
        preferred; for direct responses the last number in the text is used.

        Args:
            response: Full model-generated response string.

        Returns:
            Numeric string (commas removed), or ``None`` if nothing found.
        """
        return _extract_numeric_answer(response)

    def is_correct(self, indx: int, response: str) -> bool:
        """Check whether a model response matches the gold answer.

        Comparison is done after numeric normalisation so that ``"3.0"``
        and ``"3"`` are treated as equal.

        Args:
            indx:     Integer index of the example.
            response: Full model-generated response string.

        Returns:
            ``True`` if predicted and gold answers normalise to the same value.
        """
        predicted = self.parse_model_answer(response)
        gold = self.get_correct_answer(indx)
        if predicted is None or gold is None:
            return False
        return _normalize_number(predicted) == _normalize_number(gold)
