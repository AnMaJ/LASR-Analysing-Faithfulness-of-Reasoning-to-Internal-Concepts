from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from configs import PromptStyle

VALID_LABELS = {"entailment", "contradiction", "neutral"}


def _clean_token(token: str) -> str:
    """Lowercase and strip trailing punctuation / markdown formatting."""
    return token.strip(".*,;!?\"'`").strip("*").lower()


def extract_prediction(text: str, prompt_style: PromptStyle) -> str | None:
    """Extract the NLI label from model output text.

    For CHAIN_OF_THOUGHT the label is expected after a ``Label:`` prefix.
    For ONE_WORD the ``Label:`` prefix is tried first; if absent the last
    word of the output is used as the prediction.

    Returns the normalised label string, or *None* when parsing fails.
    """
    if not isinstance(text, str):
        return None

    # Try the "Label:" prefix first (works for both styles).
    label = re.search(r".*<label>(.*)</label>", text, re.IGNORECASE | re.DOTALL)
    if label:
        prediction = _clean_token(label.group(1))
        if prediction in VALID_LABELS:
            return prediction

    return None


def evaluate(y_true, y_pred) -> str:
    """Return a formatted classification report string."""
    labels = sorted(VALID_LABELS)
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def _build_filename(
    model_name: str,
    prompt_style: PromptStyle,
    few_shot: bool,
) -> str:
    """Build the metrics filename from experiment parameters."""
    short_model = model_name.split("/")[-1]
    style = prompt_style.value  # "one_word" or "chain_of_thought"
    shot = "few-shot" if few_shot else "one-shot"
    return f"model-metrics_{short_model}_{style}_{shot}"


def run_evaluation(
    df: pd.DataFrame,
    decoded_outputs: list[str],
    sampled_mask: pd.Series,
    prompt_style: PromptStyle,
    model_name: str,
    few_shot: bool,
    results_dir: str | Path = "experiment_results",
) -> tuple[pd.DataFrame, str]:
    """Store outputs, extract predictions, compute metrics.

    The classification report is saved to
    ``<results_dir>/model-metrics_<model>_<style>_<shot>.txt``.

    Returns the updated DataFrame and the classification report string.
    """
    df.loc[sampled_mask, "decoded_output"] = decoded_outputs
    df["prediction"] = df["decoded_output"].apply(
        lambda text: extract_prediction(text, prompt_style)
    )

    eval_mask = sampled_mask & df["prediction"].notna()
    y_true = df.loc[eval_mask, "gold_label"].str.strip().str.lower()
    y_pred = df.loc[eval_mask, "prediction"]

    n_total = sampled_mask.sum()
    n_evaluated = eval_mask.sum()
    n_failed = n_total - n_evaluated
    header = (
        f"Evaluated {n_evaluated} / {n_total} rows "
        f"({n_failed} failed to parse)\n"
    )
    report = header + evaluate(y_true, y_pred)

    # Save report to disk.
    filename = _build_filename(model_name, prompt_style, few_shot)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename}.txt"
    out_path.write_text(report)
    print(f"Report saved to {out_path}")

    return df, report
