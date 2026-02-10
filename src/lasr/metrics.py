from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import classification_report

VALID_LABELS = {"entailment", "contradiction", "neutral"}


def extract_prediction(text: str) -> str | None:
    """Extract the NLI label from model output text."""
    if not isinstance(text, str):
        return None
    matches = re.findall(r"Label[:\s]\s*(\S+)", text, re.IGNORECASE)
    if not matches:
        return None
    prediction = matches[-1].strip(".*,;!").lower()
    return prediction if prediction in VALID_LABELS else None


def evaluate(y_true, y_pred) -> str:
    """Return a formatted classification report string."""
    labels = sorted(VALID_LABELS)
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def run_evaluation(
    df: pd.DataFrame,
    decoded_outputs: list[str],
    sampled_mask: pd.Series,
) -> tuple[pd.DataFrame, str]:
    """Store outputs, extract predictions, compute metrics.

    Returns the updated DataFrame and the classification report string.
    """
    df.loc[sampled_mask, "decoded_output"] = decoded_outputs
    df["prediction"] = df["decoded_output"].apply(extract_prediction)

    eval_mask = sampled_mask & df["prediction"].notna()
    y_true = df.loc[eval_mask, "gold_label"].str.strip().str.lower()
    y_pred = df.loc[eval_mask, "prediction"]

    n_total = sampled_mask.sum()
    n_evaluated = eval_mask.sum()
    header = (
        f"Evaluated {n_evaluated} / {n_total} rows "
        f"({n_total - n_evaluated} had unparseable predictions)\n"
    )
    report = header + evaluate(y_true, y_pred)
    return df, report
