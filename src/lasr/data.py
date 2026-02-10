import pandas as pd

from lasr.config import PromptStyle


def load_esnli(url: str) -> pd.DataFrame:
    """Download and return the e-SNLI dataset as a DataFrame."""
    print("Downloading...")
    df = pd.read_csv(url)
    print(f"Done! {len(df)} rows loaded.")
    return df


def build_few_shot_examples(df: pd.DataFrame) -> str:
    """Generate the few-shot instruction-tuning example string from the dataset."""
    unique_samples = df.drop_duplicates(subset=["gold_label"]).copy()
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


_INSTRUCTIONS = {
    PromptStyle.ONE_WORD: (
        "Determine if statement B is an entailment, contradiction or neutral "
        "with respect to statement A. Answer with a single word: entailment, "
        "contradiction, or neutral.\n"
    ),
    PromptStyle.CHAIN_OF_THOUGHT: (
        "Determine if statement B is an entailment, contradiction or neutral. "
        "Reason step by step and finally provide a one-word answer.\n"
    ),
}


def build_prompts(
    df: pd.DataFrame,
    few_shot_examples: str,
    prompt_style: PromptStyle,
) -> pd.Series:
    """Build the full prompt for each row in *df*."""
    instruction = _INSTRUCTIONS[prompt_style]
    return (
        "<start_of_turn>user "
        + instruction
        + few_shot_examples
        + "\nA: " + df["Sentence1"]
        + "\nB: " + df["Sentence2"]
        + "\n<end_of_turn>model"
    )
