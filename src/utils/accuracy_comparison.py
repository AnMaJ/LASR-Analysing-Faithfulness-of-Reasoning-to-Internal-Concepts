"""Compare accuracy of ONE_WORD_TAGS vs CHAIN_OF_THOUGHT_TAGS prompting on BBQ.

Loads 100 disambig samples from each of the 9 BBQ categories (same split as
generate_activations.py), generates responses with both prompt styles using
Gemma3-27B, computes accuracy for each, then prints up to 100 examples that
were answered incorrectly with ONE_WORD_TAGS but correctly with
CHAIN_OF_THOUGHT_TAGS.

Usage:
    python -m src.utils.accuracy_comparison [--batch_size 8] [--max_new_tokens 512]
"""

import argparse
import textwrap

from src.configs import DatasetConfig, ModelConfig, PromptStyle
from src.dataset.bbq import BBQ_Dataset
from src.gemma_model import GemmaModel


BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]

SAMPLES_PER_CATEGORY = 100
LABEL_MAP = {0: "A", 1: "B", 2: "C"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ONE_WORD_TAGS vs CoT accuracy on BBQ")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument(
        "--wrong_examples", type=int, default=100,
        help="Number of one-word-wrong / CoT-correct examples to print",
    )
    return parser.parse_args()


def load_category(category: str, prompt_style: PromptStyle):
    config = DatasetConfig(
        path="HiTZ/bbq",
        prompt_style=prompt_style,
        use_chat_template=True,
        hf_data_config={"name": f"{category}_disambig", "split": "test"},
    )
    return BBQ_Dataset(config)


def load_all(prompt_style: PromptStyle):
    """Return (prompts, ground_truths, categories, raw_data) for all categories."""
    prompts, ground_truths, categories, raw_data = [], [], [], []
    for category in BBQ_CATEGORIES:
        dataset = load_category(category, prompt_style)
        n = min(SAMPLES_PER_CATEGORY, len(dataset))
        for i in range(n):
            prompts.append(dataset[i])
            ground_truths.append(LABEL_MAP[dataset.data[i]["label"]])
            categories.append(category)
            raw_data.append(dataset.data[i])
    return prompts, ground_truths, categories, raw_data


def compute_accuracy(predictions, ground_truths):
    correct = sum(p == g for p, g in zip(predictions, ground_truths) if p is not None)
    parsed = sum(p is not None for p in predictions)
    total = len(ground_truths)
    return correct / total, correct, parsed, total


def main():
    args = parse_args()

    model_cfg = ModelConfig(model_name=args.model_name)
    model = GemmaModel(model_cfg)

    # We parse answers with a ONE_WORD_TAGS dataset instance (parse_model_answer
    # is style-agnostic for BBQ — same regex cascade works for both styles).
    _parser_dataset = load_category("Age", PromptStyle.ONE_WORD_TAGS)

    print("=" * 70)
    print("Loading ONE_WORD_TAGS dataset …")
    ow_prompts, ground_truths, categories, raw_data = load_all(PromptStyle.ONE_WORD_TAGS)

    print("Loading CHAIN_OF_THOUGHT_TAGS dataset …")
    cot_prompts, _, _, _ = load_all(PromptStyle.CHAIN_OF_THOUGHT_TAGS)

    print(f"\nTotal samples: {len(ow_prompts)}")
    print("=" * 70)

    # ── Generate ────────────────────────────────────────────────────────────────
    print("\nGenerating ONE_WORD_TAGS responses …")
    ow_texts, _, _ = model.generate_batch(
        ow_prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    print("\nGenerating CHAIN_OF_THOUGHT_TAGS responses …")
    cot_texts, _, _ = model.generate_batch(
        cot_prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    # ── Parse answers ───────────────────────────────────────────────────────────
    ow_preds  = [_parser_dataset.parse_model_answer(t) for t in ow_texts]
    cot_preds = [_parser_dataset.parse_model_answer(t) for t in cot_texts]

    # ── Accuracy ────────────────────────────────────────────────────────────────
    ow_acc,  ow_correct,  ow_parsed,  total = compute_accuracy(ow_preds,  ground_truths)
    cot_acc, cot_correct, cot_parsed, _     = compute_accuracy(cot_preds, ground_truths)

    print("\n" + "=" * 70)
    print("ACCURACY RESULTS")
    print("=" * 70)
    print(f"{'Style':<30} {'Accuracy':>10}  {'Correct':>8}  {'Parsed':>8}  {'Total':>7}")
    print("-" * 70)
    print(f"{'ONE_WORD_TAGS':<30} {ow_acc:>10.2%}  {ow_correct:>8}  {ow_parsed:>8}  {total:>7}")
    print(f"{'CHAIN_OF_THOUGHT_TAGS':<30} {cot_acc:>10.2%}  {cot_correct:>8}  {cot_parsed:>8}  {total:>7}")

    delta = cot_acc - ow_acc
    print(f"\nCoT improvement over one-word: {delta:+.2%}")

    # ── Per-category breakdown ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-CATEGORY ACCURACY")
    print("=" * 70)
    print(f"{'Category':<28} {'ONE_WORD':>10}  {'CoT':>10}  {'Delta':>8}")
    print("-" * 70)
    for cat in BBQ_CATEGORIES:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        ow_cat  = sum(ow_preds[i]  == ground_truths[i] for i in idxs if ow_preds[i]  is not None)
        cot_cat = sum(cot_preds[i] == ground_truths[i] for i in idxs if cot_preds[i] is not None)
        n = len(idxs)
        print(f"{cat:<28} {ow_cat/n:>10.2%}  {cot_cat/n:>10.2%}  {(cot_cat-ow_cat)/n:>+8.2%}")

    # ── Examples: wrong OW, correct CoT ─────────────────────────────────────────
    interesting = [
        i for i in range(total)
        if ow_preds[i] != ground_truths[i] and cot_preds[i] == ground_truths[i]
    ]

    print("\n" + "=" * 70)
    print(f"EXAMPLES: wrong with ONE_WORD_TAGS, correct with CHAIN_OF_THOUGHT_TAGS")
    print(f"Found {len(interesting)} such examples. Showing up to {args.wrong_examples}.")
    print("=" * 70)

    for rank, i in enumerate(interesting[: args.wrong_examples], start=1):
        row = raw_data[i]
        context      = row.get("context", "")
        question     = row.get("question", "")
        ans0, ans1, ans2 = row.get("ans0", ""), row.get("ans1", ""), row.get("ans2", "")
        gt = ground_truths[i]

        print(f"\n[{rank}] Global index: {i}  |  Category: {categories[i]}")
        print(f"    Ground truth : {gt}  |  OW pred: {ow_preds[i]}  |  CoT pred: {cot_preds[i]}")
        print(f"    Context      : {textwrap.fill(context, width=80, subsequent_indent=' ' * 19)}")
        print(f"    Question     : {textwrap.fill(question, width=80, subsequent_indent=' ' * 19)}")
        print(f"    A) {ans0}")
        print(f"    B) {ans1}")
        print(f"    C) {ans2}")
        print(f"    --- ONE_WORD response ---")
        print(textwrap.fill(ow_texts[i].strip(), width=80, initial_indent="    ", subsequent_indent="    "))
        print(f"    --- CoT response ---")
        print(textwrap.fill(cot_texts[i].strip(), width=80, initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()