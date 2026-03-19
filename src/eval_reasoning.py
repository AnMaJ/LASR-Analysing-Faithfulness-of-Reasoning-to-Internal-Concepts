"""Evaluate Gemma-3-1B-IT and Gemma-3-27B-IT on LogiQA and AQuA-RAT.

Runs accuracy evaluation with and without chain-of-thought prompting, and
prints 1–2 qualitative CoT examples per dataset/model combination.

Usage:
    python -m src.eval_reasoning [options]

Options:
    --models   1b 27b        Models to evaluate (default: both)
    --datasets logiqa aqua   Datasets to evaluate (default: both)
    --num_samples N          Limit to N examples per dataset (default: all)
    --batch_size B           Generation batch size (default: 4)
    --output_path PATH       Save full results as JSON
    --cot_examples N         Number of CoT examples to print (default: 2)
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Optional

import torch

from src.configs import DatasetConfig, ModelConfig, PromptStyle
from src.dataset.logiqa import LogiQA_Dataset
from src.dataset.aqua import AQuA_Dataset
from src.gemma_model import GemmaModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = {
    "1b":  "google/gemma-3-1b-it",
    "27b": "google/gemma-3-27b-it",
}

DATASETS = {
    "logiqa": {
        "cls":        LogiQA_Dataset,
        "hf_path":    "lucasmccabe/logiqa",  # data fetched from GitHub source
        "hf_kwargs":  {"split": "test"},
        "valid_opts": "ABCD",
        "label":      "LogiQA",
    },
    "aqua": {
        "cls":        AQuA_Dataset,
        "hf_path":    "aqua_rat",
        "hf_kwargs":  {"split": "test"},
        "valid_opts": "ABCDE",
        "label":      "AQuA-RAT",
    },
}

COT_MODES = [
    ("no_cot", PromptStyle.ONE_WORD_NO_TAGS,         "No CoT",  64),
    ("cot",    PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS, "CoT",    2048),
]

# Gemma chat marker used to isolate the model's generated response
_MODEL_TURN = "<start_of_turn>model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response(full_text: str) -> str:
    """Strip the prompt from the decoded output, keeping only the model reply."""
    return full_text.split(_MODEL_TURN)[-1].strip()


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _print_cot_examples(examples: list, n: int, dataset_label: str, model_label: str) -> None:
    print(f"\n{'='*70}")
    print(f"  CoT Examples  |  {dataset_label}  |  {model_label}")
    print(f"{'='*70}")
    for i, ex in enumerate(examples[:n], 1):
        print(f"\n[Example {i}/{min(n, len(examples))}]")
        print("  Question:")
        # For LogiQA show context too if present
        if ex.get("context"):
            print(_wrap(f"(Context) {ex['context']}", indent="    "))
        print(_wrap(ex["question"], indent="    "))
        if ex.get("options"):
            print("  Options:")
            for opt in ex["options"]:
                print(f"    {opt}")
        print("  Model Response:")
        # Wrap each line of the (potentially long) CoT response
        for line in ex["response"].splitlines():
            print(_wrap(line, indent="    ") if line.strip() else "")
        print(
            f"  Predicted: {ex['predicted'] or '?'}  |  "
            f"Ground Truth: {ex['ground_truth']}  |  "
            f"Correct: {'Yes' if ex['correct'] else 'No'}"
        )
        print("-" * 70)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model: GemmaModel,
    dataset_key: str,
    prompt_style: PromptStyle,
    max_new_tokens: int,
    batch_size: int,
    num_samples: Optional[int],
) -> dict:
    """Run one evaluation pass (one dataset × one prompt mode × one model).

    Returns a dict with keys: accuracy, correct, total, results (list of
    per-example dicts), and cot_examples (list for qualitative printing).
    """
    cfg_info = DATASETS[dataset_key]
    ds_cls   = cfg_info["cls"]
    is_cot   = prompt_style == PromptStyle.CHAIN_OF_THOUGHT_NO_TAGS

    config = DatasetConfig(
        path=cfg_info["hf_path"],
        prompt_style=prompt_style,
        use_chat_template=True,
        hf_data_config=cfg_info["hf_kwargs"],
    )
    dataset = ds_cls(config)

    total = len(dataset)
    if num_samples is not None:
        total = min(num_samples, total)

    prompts      = [dataset[i]               for i in range(total)]
    ground_truth = [dataset.get_correct_letter(i) for i in range(total)]

    texts, _, _ = model.generate_batch(
        prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
    )

    results       = []
    cot_examples  = []
    correct_count = 0

    for i, (text, gt) in enumerate(zip(texts, ground_truth)):
        response  = _extract_response(text)
        predicted = dataset.parse_model_answer(response)
        is_correct = predicted is not None and predicted.upper() == gt.upper()
        correct_count += int(is_correct)

        # Build a lightweight record for JSON output
        ex = dataset.data[i]
        record = {
            "index":        i,
            "ground_truth": gt,
            "predicted":    predicted,
            "correct":      is_correct,
            "response":     response,
        }
        # Attach human-readable fields for qualitative examples
        if dataset_key == "logiqa":
            record["question"] = ex.get("query", "")
            record["context"]  = ex.get("context", "")
            record["options"]  = [
                f"{chr(65+j)}) {ex['options'][j]}" for j in range(4)
            ]
        else:  # aqua
            record["question"] = ex.get("question", "")
            record["context"]  = ""
            record["options"]  = ex.get("options", [])

        results.append(record)
        if is_cot and len(cot_examples) < 2:
            cot_examples.append(record)

    accuracy = correct_count / total if total else 0.0
    return {
        "accuracy":     accuracy,
        "correct":      correct_count,
        "total":        total,
        "results":      results,
        "cot_examples": cot_examples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma models on LogiQA and AQuA-RAT (with/without CoT)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=["1b", "27b"], default=["1b"],
        help="Which model(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["logiqa", "aqua"], default=["logiqa", "aqua"],
        help="Which dataset(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Limit evaluation to N examples per dataset (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Generation batch size (default: 4; reduce for 27b if OOM)",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save full results as JSON",
    )
    parser.add_argument(
        "--cot_examples", type=int, default=2,
        help="Number of CoT examples to print per combination (default: 2)",
    )
    args = parser.parse_args()

    all_results: dict = {}
    summary_rows: list[dict] = []

    for model_key in args.models:
        model_name  = MODELS[model_key]
        model_label = f"Gemma-3-{model_key.upper()}-IT"
        print(f"\n{'#'*70}")
        print(f"  Loading model: {model_name}")
        print(f"{'#'*70}\n")

        model = GemmaModel(ModelConfig(model_name=model_name))
        all_results[model_key] = {}

        for dataset_key in args.datasets:
            ds_label = DATASETS[dataset_key]["label"]
            all_results[model_key][dataset_key] = {}

            for mode_key, prompt_style, mode_label, max_new_tokens in COT_MODES:
                print(
                    f"\n--- {ds_label} | {model_label} | {mode_label} "
                    f"(max_new_tokens={max_new_tokens}) ---"
                )

                run = evaluate_dataset(
                    model=model,
                    dataset_key=dataset_key,
                    prompt_style=prompt_style,
                    max_new_tokens=max_new_tokens,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                )

                acc = run["accuracy"]
                print(
                    f"Accuracy: {run['correct']}/{run['total']} = "
                    f"{acc:.4f} ({acc*100:.2f}%)"
                )

                if mode_key == "cot" and run["cot_examples"]:
                    _print_cot_examples(
                        run["cot_examples"],
                        n=args.cot_examples,
                        dataset_label=ds_label,
                        model_label=model_label,
                    )

                all_results[model_key][dataset_key][mode_key] = {
                    k: v for k, v in run.items() if k != "cot_examples"
                }
                summary_rows.append({
                    "model":    model_label,
                    "dataset":  ds_label,
                    "mode":     mode_label,
                    "accuracy": acc,
                    "correct":  run["correct"],
                    "total":    run["total"],
                })

        # Free model memory before loading the next one
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  Results Summary")
    print(f"{'='*70}")
    col_w = [22, 10, 8, 10, 20]
    header = (
        f"{'Dataset':<{col_w[0]}} {'Model':<{col_w[1]}} {'Mode':<{col_w[2]}} "
        f"{'Accuracy':>{col_w[3]}} {'Correct/Total':<{col_w[4]}}"
    )
    print(header)
    print("-" * sum(col_w))
    for row in summary_rows:
        print(
            f"{row['dataset']:<{col_w[0]}} {row['model']:<{col_w[1]}} "
            f"{row['mode']:<{col_w[2]}} {row['accuracy']*100:>{col_w[3]-1}.2f}% "
            f"{row['correct']}/{row['total']}"
        )
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"summary": summary_rows, "details": all_results}, f, indent=2)
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
