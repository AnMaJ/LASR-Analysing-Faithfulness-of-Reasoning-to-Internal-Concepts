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
                f"A: {x['Sentence1']}\n"
                f"B: {x['Sentence2']}\n"
                f"{x['Explanation_1']}\n"
                f"Label: {x['gold_label']}"
            ),
            axis=1,
        )
    else:
        unique_samples["formatted_input"] = unique_samples.apply(
            lambda x: (
                f"A: {x['Sentence1']}\n"
                f"B: {x['Sentence2']}\n"
                f"Label: {x['gold_label']}"
            ),
            axis=1,
        )

    return "\n".join(
        f"Example {i+1}:\n{text}"
        for i, text in enumerate(unique_samples["formatted_input"])
    )


_INSTRUCTIONS = {
    PromptStyle.ONE_WORD: (
        "Determine if statement B is an entailment, contradiction or neutral "
        "with respect to statement A. Answer with a single word: entailment, "
        "contradiction, or neutral.\n"
    ),
    PromptStyle.CHAIN_OF_THOUGHT: (
        "Determine if statement B is an entailment, contradiction or neutral. "
        "Reason step by step and finally provide a one-word answer as 'Label: (entailment, contradiction, neutral).'\n"
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
    return (
        "<start_of_turn>user "
        + instruction
        + examples_block
        + "A: " + df["Sentence1"]
        + "\nB: " + df["Sentence2"]
        + "\n<end_of_turn>model"
    )
