import pandas as pd

from lasr.config import PromptStyle


def load_esnli(url: str) -> pd.DataFrame:
    """Download and return the e-SNLI dataset as a DataFrame."""
    print("Downloading...")
    df = pd.read_csv(url)
    print(f"Done! {len(df)} rows loaded.")
    return df


def build_few_shot_examples(df: pd.DataFrame, prompt_style: PromptStyle) -> str:
    """Generate the few-shot example string, formatted for *prompt_style*.

    * CHAIN_OF_THOUGHT: includes the explanation before the label.
    * ONE_WORD: only sentence pair and label (no explanation).
    """
    unique_samples = df.drop_duplicates(subset=["gold_label"]).copy()

    if prompt_style == PromptStyle.CHAIN_OF_THOUGHT:
        unique_samples["formatted_input"] = unique_samples.apply(
            lambda x: (
                f"Premise: {x['Sentence1']}\n"
                f"Hypothesis: {x['Sentence2']}\n"
                f"<reasoning>{x['Explanation_1']}</reasoning>\n"
                f"<label>{x['gold_label']}</label>"
            ),
            axis=1,
        )
    else:
        unique_samples["formatted_input"] = unique_samples.apply(
            lambda x: (
                f"Premise: {x['Sentence1']}\n"
                f"Hypothesis: {x['Sentence2']}\n"
                f"<label>{x['gold_label']}</label>"
            ),
            axis=1,
        )

    return "\n".join(
        f"Example {i+1}:\n{text}"
        for i, text in enumerate(unique_samples["formatted_input"])
    )


_INSTRUCTIONS = {
    PromptStyle.ONE_WORD: (
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
    PromptStyle.CHAIN_OF_THOUGHT: (
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
    ),
}


def build_prompts(
    df: pd.DataFrame,
    prompt_style: PromptStyle,
    few_shot: bool = True,
    few_shot_examples: str | None = None,
) -> pd.Series:
    """Build the full prompt for each row in *df*.

    Parameters
    ----------
    df : DataFrame with ``Sentence1`` and ``Sentence2`` columns.
    prompt_style : Which instruction/example style to use.
    few_shot : If *True* (default), prepend few-shot examples.
    few_shot_examples : Pre-built example string (from
        ``build_few_shot_examples``). Required when *few_shot* is True.
    """
    instruction = _INSTRUCTIONS[prompt_style]
    examples_block = ""
    if few_shot:
        if few_shot_examples is None:
            raise ValueError(
                "few_shot_examples must be provided when few_shot=True"
            )
        examples_block = few_shot_examples + "\n"
        prompt = instruction.format(premise=df["Sentence1"], hypothesis=df["Sentence2"], examples_block=examples_block)
    else:
        prompt = instruction.format(premise=df["Sentence1"], hypothesis=df["Sentence2"])

    return (
        "<start_of_turn>user "
        + prompt
        + "\n<end_of_turn>model"
    )
