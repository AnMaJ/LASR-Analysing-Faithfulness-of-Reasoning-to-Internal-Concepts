from __future__ import annotations

import re
from typing import NamedTuple

from datasets import Dataset as HFDataset

from src.dataset.base_dataset import BaseDataset, PromptStyle


class ParsedAnswer(NamedTuple):
    """Parsed model output for an NLI task."""
    label: str | None      # "entailment", "neutral", "contradiction", or None
    reasoning: str | None  # Extracted reasoning text, or None


_LABEL_MAP_ = {0: "entailment", 1: "neutral", 2: "contradiction"}

_VALID_LABELS_ = set(_LABEL_MAP_.values())


class ESNLI_Dataset(BaseDataset):
    """Explainable Stanford Natural Language Inference (e-SNLI) dataset.

    Each sample is a sentence pair labelled as *entailment*, *contradiction*,
    or *neutral*.  Prompt formatting includes automatically generated
    few-shot examples (one per label) drawn from the dataset itself.

    Attributes:
        _INSTRUCTIONS_: Mapping from prompt-style key to instruction text.
    """

    _INSTRUCTIONS_ = {
        PromptStyle.ONE_WORD_NO_TAGS: (
            "Determine if statement B is an entailment, contradiction or neutral "
            "with respect to statement A. Answer with a single word: entailment, "
            "contradiction, or neutral.\n"
        ),
        PromptStyle.ONE_WORD_TAGS: (
            "Classify the relationship between the following Premise and Hypothesis.\n"
            "Premise: {premise}\n"
            "Hypothesis: {hypothesis}\n\n"
            "Instructions:\n"
            "- Step 1: Analyze the relationship step-by-step.\n"
            "- Step 2: Output your analysis inside <reasoning> tags.\n"
            "- Step 3: Output the final classification (entailment, neutral, or contradiction) inside <label> tags.\n\n"
            "Format:\n"
            "<label>[label]</label>\n"
        ),
        PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS: (
            "Determine if statement B is an entailment, contradiction or neutral. "
            "Reason step by step and finally provide a one-word answer.\n"
        ),
        PromptStyle.CHAIN_OF_THOUGHT_TAGS: (
            "Task: Determine the logical relationship between a Premise and a Hypothesis.\n"
            "Options: entailment, contradiction, neutral.\n\n"
            "Rules:\n"
            "1. You MUST provide your reasoning inside <reasoning> tags.\n"
            "2. You MUST provide the final label inside <label> tags.\n"
            "3. The reasoning must come BEFORE the label.\n\n"
            "Premise: {premise}\n"
            "Hypothesis: {hypothesis}\n"
        ),
    }

    def load_dataset(self, path: str, hf_kwargs: dict):
        """Load e-SNLI from HuggingFace Hub and add a string ``gold_label`` column.

        The ``esnli/esnli`` repository uses a legacy loading script that is no
        longer supported by ``datasets >= 4``.  We work around this by loading
        from the auto-converted Parquet branch (``refs/convert/parquet``) when
        no explicit ``revision`` is provided.

        Args:
            path: HuggingFace dataset identifier (e.g. ``"esnli/esnli"``).
            hf_kwargs: Extra keyword arguments forwarded to
                ``datasets.load_dataset`` (e.g. ``{"split": "test"}``).

        Returns:
            A HuggingFace ``Dataset`` with an additional ``gold_label`` column.
        """
        hf_kwargs.setdefault("revision", "refs/convert/parquet")
        data = super().load_dataset(path, hf_kwargs)
        data = data.map(
            lambda row: {"gold_label": _LABEL_MAP_[row["label"]]},
            desc="Adding gold_label",
        )
        return data

    def build_prompt(self, indx: int):
        """Build a model-ready NLI prompt for sample *indx*.

        Args:
            indx: Integer index of the sample in the dataset.

        Returns:
            If ``use_chat_template`` is ``True``, a list of message dicts;
            otherwise a string with Gemma chat turn markers.
        """
        example = self.data[indx]
        instruction = self._INSTRUCTIONS_[self.prompt_style]

        examples_block = ""
        if self.few_shot:
            if not hasattr(self, "_cached_few_shot"):
                self._cached_few_shot = self._build_few_shot_examples()
            examples_block = self._cached_few_shot + "\n"

        has_placeholders = "{premise}" in instruction
        kwargs = {"premise": example["premise"], "hypothesis": example["hypothesis"]}
        if "{examples_block}" in instruction:
            kwargs["examples_block"] = examples_block

        if has_placeholders:
            user_content = instruction.format(**kwargs)
        else:
            user_content = (
                f"{instruction}\n"
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}\n"
            )

        if self.use_chat_template:
            return [{"role": "user", "content": user_content}]

        return f"<start_of_turn>user {user_content}\n<end_of_turn>model "

    def build_prompts(self) -> HFDataset:
        """Build prompts for the entire dataset using HF ``.map()``.

        Returns:
            A new HuggingFace Dataset with columns:
            - ``"prompt"``: The formatted prompt string for each sample.
            - ``"gold_label"``: The string label (entailment/neutral/contradiction).
        """
        examples_block = ""
        if self.few_shot:
            if not hasattr(self, "_cached_few_shot"):
                self._cached_few_shot = self._build_few_shot_examples()
            examples_block = self._cached_few_shot + "\n"

        instruction = self._INSTRUCTIONS_[self.prompt_style]
        has_placeholders = "{premise}" in instruction

        def _format_row(row):
            kwargs = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
            if "{examples_block}" in instruction:
                kwargs["examples_block"] = examples_block

            if has_placeholders:
                user_content = instruction.format(**kwargs)
            else:
                user_content = (
                    f"{instruction}\n"
                    f"Premise: {row['premise']}\n"
                    f"Hypothesis: {row['hypothesis']}\n"
                )

            if self.use_chat_template:
                prompt = [{"role": "user", "content": user_content}]
            else:
                prompt = f"<start_of_turn>user {user_content}\n<end_of_turn>model "

            return {"prompt": prompt, "gold_label": row["gold_label"]}

        return self.data.map(_format_row, desc="Building prompts")

    def _build_few_shot_examples(self) -> str:
        """Construct few-shot demonstrations (one per unique gold label).

        Iterates through the dataset to collect the first example for each
        unique label class, then formats them according to the prompt style.

        Returns:
            Concatenated, numbered examples each containing a sentence
            pair, an explanation, and the gold label.
        """
        seen = {}
        for row in self.data:
            lbl = row["gold_label"]
            if lbl not in seen:
                seen[lbl] = row
            if len(seen) == len(_LABEL_MAP_):
                break

        examples = []
        for i, (_, row) in enumerate(seen.items()):
            if self.prompt_style == PromptStyle.CHAIN_OF_THOUGHT_TAGS:
                text = (
                    f"Premise: {row['premise']}\n"
                    f"Hypothesis: {row['hypothesis']}\n"
                    f"<reasoning>{row['explanation_1']}</reasoning>\n"
                    f"<label>{row['gold_label']}</label>"
                )
            elif self.prompt_style == PromptStyle.ONE_WORD_TAGS:
                text = (
                    f"Premise: {row['premise']}\n"
                    f"Hypothesis: {row['hypothesis']}\n"
                    f"<label>{row['gold_label']}</label>"
                )
            else:
                text = (
                    f"Premise: {row['premise']}\n"
                    f"Hypothesis: {row['hypothesis']}\n"
                    f"Final answer is {row['gold_label']}"
                )
            examples.append(f"Example {i + 1}:\n{text}")

        return "\n".join(examples)

    def parse_model_answer(self, response: str) -> ParsedAnswer:
        """Extract the predicted label and optional reasoning from a model response.

        Only tag-based prompt styles are supported. Returns ``"FormatFailure"``
        for any tag that is absent, empty, or contains no valid label word.

        Args:
            response: The full model-generated response string.

        Returns:
            A ``ParsedAnswer`` named tuple with ``label`` and ``reasoning`` fields.

        Raises:
            NotImplementedError: If called with a no-tags prompt style.
        """
        tags_style = self.prompt_style in (
            PromptStyle.CHAIN_OF_THOUGHT_TAGS,
            PromptStyle.ONE_WORD_TAGS,
        )

        if not tags_style:
            raise NotImplementedError(
                f"parse_model_answer is not implemented for prompt style {self.prompt_style!r}. "
                "Only tag-based styles are supported."
            )

        # --- label ---
        lm = re.search(
            r"<label>\s*(entailment|neutral|contradiction)\s*</label>",
            response, re.IGNORECASE,
        )
        label = lm.group(1).lower() if lm else "FormatFailure"

        # --- reasoning ---
        rm = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        reasoning = rm.group(1).strip() if rm else "FormatFailure"
        if reasoning == "":
            reasoning = "FormatFailure"

        return ParsedAnswer(label=label, reasoning=reasoning)
