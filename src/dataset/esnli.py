from typing import Dict

import pandas as pd

from src.dataset.base_dataset import BaseDataset, PromptStyle

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
        ), # Useless?
        PromptStyle.ONE_WORD_TAGS: (
            """Classify the relationship between the following Premise and Hypothesis.
            Premise: {premise}
            Hypothesis: {hypothesis}
            
            Instructions:
            - Step 1: Analyze the relationship step-by-step.
            - Step 2: Output your analysis inside <reasoning> tags.
            - Step 3: Output the final classification (entailment, neutral, or contradiction) inside <label> tags.
            
            Format:
            <reasoning>[Your analysis here]</reasoning>
            <label>[label]</label>
            """
        ),
        PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS: (
            "Determine if statement B is an entailment, contradiction or neutral. "
            "Reason step by step and finally provide a one-word answer.\n"
        ), # Useless?
        PromptStyle.CHAIN_OF_THOUGHT_TAGS: (
            """Task: Determine the logical relationship between a Premise and a Hypothesis. 
            Options: entailment, contradiction, neutral.
            
            Rules:
            1. You MUST provide your reasoning inside <reasoning> tags.
            2. You MUST provide the final label inside <label> tags.
            3. The reasoning must come BEFORE the label.

            {examples_block}
            Premise: {premise}
            Hypothesis: {hypothesis}
            """
        )
    }

    # Change it or remove to load the data directly from HF.
    def load_dataset(self, path: str, config: Dict):
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
        example = self.data[indx]
        instruction = self._INSTRUCTIONS_[self.prompt_style]
        examples_block = ""
        if self.few_shot:
            examples_block = self._build_few_shot_examples() + "\n"

        def _format_row(row):
            kwargs = {"premise": row["Sentence1"], "hypothesis": row["Sentence2"]}
            if "{examples_block}" in instruction:
                kwargs["examples_block"] = examples_block
            return instruction.format(**kwargs)

        return (
            "<start_of_turn>user "
            + self.data.apply(_format_row, axis=1)
            + "\n<end_of_turn>model"
        )

    def _build_few_shot_examples(self) -> str:
        """Construct few-shot demonstrations (one per unique gold label).

        Returns:
            str: Concatenated, numbered examples each containing a sentence
            pair, an explanation, and the gold label.
        """
        unique_samples = self.data.drop_duplicates(subset=["gold_label"]).copy()

        if self.prompt_style == PromptStyle.CHAIN_OF_THOUGHT_TAGS:
            unique_samples["formatted_input"] = unique_samples.apply(
                lambda x: (
                    f"Premise: {x['Sentence1']}\n"
                    f"Hypothesis: {x['Sentence2']}\n"
                    f"<reasoning>{x['Explanation_1']}</reasoning>\n"
                    f"<label>{x['gold_label']}</label>"
                ),
                axis=1,
            )
        elif self.prompt_style == PromptStyle.ONE_WORD_TAGS:
            unique_samples["formatted_input"] = unique_samples.apply(
                lambda x: (
                    f"Premise: {x['Sentence1']}\n"
                    f"Hypothesis: {x['Sentence2']}\n"
                    f"<label>{x['gold_label']}</label>"
                ),
                axis=1,
            )
        else:
            unique_samples["formatted_input"] = unique_samples.apply(
                lambda x: (
                    f"Premise: {x['Sentence1']}\n"
                    f"Hypothesis: {x['Sentence2']}\n"
                    f"Final answer is {x['gold_label']}" # Not sure about this, but probably not needed (we always use tags)
                ),
                axis=1,
            )

        return "\n".join(
            f"Example {i+1}:\n{text}"
            for i, text in enumerate(unique_samples["formatted_input"])
        )