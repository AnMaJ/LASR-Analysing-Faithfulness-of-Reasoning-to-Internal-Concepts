"""LogiQA dataset class.

The dataset is fetched directly from the canonical GitHub repository
(https://github.com/lgw863/LogiQA-dataset) using the same parsing logic as
the ``lucasmccabe/logiqa`` HuggingFace loading script, which is no longer
loadable with recent versions of the ``datasets`` library.

Equivalent usage intent:
    from datasets import load_dataset
    ds = load_dataset("lucasmccabe/logiqa")
"""

import re
from typing import Dict, List, Optional

import requests

from src.dataset.base_dataset import BaseDataset, PromptStyle

_SPLIT_URLS: Dict[str, str] = {
    "train":      "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
    "validation": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
    "test":       "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
}

_OPTION_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Parsing helpers (mirrors the lucasmccabe/logiqa loading script exactly)
# ---------------------------------------------------------------------------

def _process_sentence(text: str) -> str:
    """Clean a raw line from the LogiQA text file."""
    text = text.replace("\n", "")
    sents = text.split(".")
    result = ""
    for sent in sents:
        if not sent:
            continue
        if not result:
            result = sent
        elif sent[0].isnumeric():
            result += "." + sent
        else:
            result += ". " + sent
    result = result.replace("  ", " ").replace("\\'", "'").rstrip()
    if re.match(r"^[A-Z][\w\s]+[?.!]$", result) is None:
        result += "."
    return result.replace("?.", "?").replace("!.", "!").replace("..", ".")


def _strip_option_prefix(opt: str) -> str:
    """Remove leading 'A) ' style prefix from an option string."""
    if opt and opt[0] in "ABCD":
        return opt[3:]
    return opt


def _load_split(split: str) -> List[Dict]:
    """Fetch and parse one LogiQA split from GitHub.

    Args:
        split: One of ``"train"``, ``"validation"``, or ``"test"``.

    Returns:
        List of example dicts with keys ``context``, ``query``, ``options``
        (list of 4 strings, prefix-stripped), and ``correct_option`` (int 0–3).
    """
    url = _SPLIT_URLS[split]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    lines = [_process_sentence(line) for line in resp.text.splitlines()]
    examples: List[Dict] = []

    for key in range(len(lines) // 8):
        row = 8 * key
        correct_raw = lines[row + 1].replace(".", "").strip().lower()
        correct_idx = "abcd".index(correct_raw) if correct_raw in "abcd" else -1
        examples.append({
            "context": lines[row + 2],
            "query":   lines[row + 3],
            "options": [_strip_option_prefix(lines[row + 4 + i]) for i in range(4)],
            "correct_option": correct_idx,
        })
    return examples


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_mc_letter(text: str, valid: str, is_cot: bool) -> Optional[str]:
    """Extract a multiple-choice answer letter from a model response.

    Args:
        text:    Full model response string.
        valid:   String of valid option letters in order, e.g. ``"ABCD"``.
        is_cot:  Whether the response is a chain-of-thought (longer) response.

    Returns:
        Single uppercase letter from *valid*, or ``None`` if none found.
    """
    upper = text.upper().strip()
    pat = f"[{valid}]"  # e.g. "[ABCD]"

    if is_cot:
        explicit_patterns = [
            rf"FINAL\s+ANSWER\s*[:\-\s]*\(?({pat})\)?",
            rf"THE\s+ANSWER\s+IS\s*[:\-\s]*\(?({pat})\)?",
            rf"(?:MY\s+)?ANSWER\s*[:\-\s]+\(?({pat})\)?",
            rf"CORRECT\s+(?:OPTION|ANSWER)\s*[:\-\s]*\(?({pat})\)?",
            rf"I\s+(?:WOULD\s+)?CHOOSE\s*[:\-\s]*\(?({pat})\)?",
            rf"OPTION\s+({pat})\s+IS\s+(?:CORRECT|RIGHT)",
        ]
        for pattern in explicit_patterns:
            matches = list(re.finditer(pattern, upper))
            if matches:
                return matches[-1].group(1)

        # Check the last 100 characters for a trailing letter
        tail = upper[-100:]
        m = re.search(rf"\b({pat})\b(?:\s*[.\):]?\s*$)", tail)
        if m:
            return m.group(1)

        # Bold letter **X**
        m = re.search(rf"\*\*({pat})\*\*", upper)
        if m:
            return m.group(1)

        # Last standalone letter in full response
        all_m = re.findall(rf"\b({pat})\b", upper)
        return all_m[-1] if all_m else None

    else:
        # Direct (no-CoT) response: expect a very short answer
        stripped = text.strip().upper()
        if stripped in list(valid):
            return stripped
        m = re.match(rf"^({pat})\b", stripped)
        if m:
            return m.group(1)
        m = re.search(rf"\b({pat})\b", stripped)
        return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LogiQA_Dataset(BaseDataset):
    """LogiQA logical reasoning dataset (English split, 4-option MCQ).

    Data is fetched from https://github.com/lgw863/LogiQA-dataset and parsed
    with the same logic as the ``lucasmccabe/logiqa`` HuggingFace loading
    script.  Each example has a reading passage (``context``), a question
    (``query``), four answer options, and a ``correct_option`` index (0–3).

    Attributes:
        _INSTRUCTIONS_: Prompt instruction text keyed by PromptStyle.
        _OPTION_LABELS_: ``["A", "B", "C", "D"]``.
    """

    _INSTRUCTIONS_ = {
        PromptStyle.ONE_WORD_NO_TAGS: (
            "Read the passage and answer the multiple-choice question. "
            "Reply with only A, B, C, or D."
        ),
        PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS: (
            "Read the passage and answer the multiple-choice question. "
            "Think step by step, then clearly state your final answer as A, B, C, or D."
        ),
    }
    _OPTION_LABELS_ = _OPTION_LABELS

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataset(self, path: str, hf_kwargs: Dict) -> List[Dict]:
        """Fetch and parse a LogiQA split from GitHub.

        Args:
            path:      Ignored (data is fetched from GitHub, not a local path).
            hf_kwargs: Optional dict; ``split`` key selects the split
                       (default: ``"test"``).

        Returns:
            List of example dicts.
        """
        split = hf_kwargs.get("split", "test")
        print(f"Fetching LogiQA '{split}' split from GitHub …")
        data = _load_split(split)
        print(f"LogiQA {split}: {len(data)} examples loaded.")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, indx):
        return self.build_prompt(indx)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, indx: int):
        """Build a multiple-choice prompt for the example at *indx*.

        Args:
            indx: Integer index.

        Returns:
            List of chat dicts when ``use_chat_template`` is True, else str.
        """
        ex = self.data[indx]
        opts_text = "\n".join(
            f"{_OPTION_LABELS[i]}) {ex['options'][i]}" for i in range(4)
        )
        instruction = self._INSTRUCTIONS_[self.prompt_style]
        content = (
            f"Context: {ex['context']}\n\n"
            f"Question: {ex['query']}\n\n"
            f"Options:\n{opts_text}"
        )
        if self.use_chat_template:
            return [{"role": "user", "content": f"{instruction}\n\n{content}"}]
        return f"{instruction}\n\n{content}"

    # ------------------------------------------------------------------
    # Answer helpers
    # ------------------------------------------------------------------

    def get_correct_letter(self, indx: int) -> str:
        """Return the gold answer letter (A–D) for example *indx*."""
        return _OPTION_LABELS[self.data[indx]["correct_option"]]

    def parse_model_answer(self, response: str) -> Optional[str]:
        is_cot = self.prompt_style == PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS
        return _extract_mc_letter(response, valid="ABCD", is_cot=is_cot)
