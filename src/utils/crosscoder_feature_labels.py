"""Auto-label Gemma-scope-2 crosscoder features using e-SNLI or BBQ samples.

Runs three phases:
  1. Activation generation — forward each sample through Gemma-3-27b-it,
     encode residuals at layers [16,31,40,53] through the crosscoder, and
     record the max-pooled feature activation across tokens and layers.
  2. Feature analysis — group samples by which features they fire, building
     top-20 activating and 20 non-activating example lists per feature.
  3. LLM labeling — call Claude (Anthropic API) on each feature to produce a
     ≤5-word interpretable label and SEMANTIC/SYNTACTIC classification.

Usage:
    python -m src.utils.crosscoder_feature_labels --dataset esnli --num_samples 100
    python -m src.utils.crosscoder_feature_labels --dataset bbq   --num_samples 50 --skip_llm
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm
from dotenv import load_dotenv

from src.SAE import JumpReLUMultiLayerSAE
from src.configs import ModelConfig, DatasetConfig, PromptStyle, CrosscoderConfig
from src.gemma_model import GemmaModel
from src.dataset.esnli import ESNLI_Dataset
from src.dataset.bbq import BBQ_Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYERS     = [16, 31, 40, 53]
WIDTH      = "262k"
L0         = "medium"
REPO_ID    = "google/gemma-scope-2-27b-it"
D_MODEL    = 3584
MODEL_NAME = "google/gemma-3-27b-it"

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crosscoder feature extraction and auto-labeling"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["esnli", "bbq"],
        help="Dataset to use for feature extraction",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        help="HuggingFace split (default: validation; BBQ only has 'test')",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Max number of samples to use (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/crosscoder_features",
        help="Root directory for output files",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens to generate per sample",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Force rebuild of feature analysis even if it exists",
    )
    parser.add_argument(
        "--skip_llm", action="store_true",
        help="Skip LLM labeling phase",
    )
    parser.add_argument(
        "--llm_model", type=str, default="claude-haiku-4-5-20251001",
        help="Anthropic model for feature labeling",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Flush GPU cache and checkpoint to disk every N samples (default: 32)",
    )
    parser.add_argument(
        "--min_prompt_activations", type=int, default=10,
        help="Features that fire on fewer than this many prompts are labeled N/A (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

def load_samples(args: argparse.Namespace) -> list[dict]:
    """Load dataset samples and build prompts.

    Returns a list of dicts with keys: text_a, text_b, label, prompt.
    """
    samples = []

    if args.dataset == "esnli":
        config = DatasetConfig(
            path="esnli/esnli",
            prompt_style=PromptStyle.CHAIN_OF_THOUGHT_TAGS,
            use_chat_template=True,
            hf_data_config={"split": args.split},
            few_shot=False,
        )
        dataset = ESNLI_Dataset(config)
        prompted = dataset.build_prompts()

        for row in prompted:
            samples.append({
                "text_a": row["premise"],
                "text_b": row["hypothesis"],
                "label":  row["gold_label"],
                "prompt": row["prompt"],
            })

    elif args.dataset == "bbq":
        for category in BBQ_CATEGORIES:
            config = DatasetConfig(
                path="HiTZ/bbq",
                prompt_style=PromptStyle.CHAIN_OF_THOUGHT_TAGS,
                use_chat_template=False,
                hf_data_config={"name": f"{category}_disambig", "split": "test"},
            )
            dataset = BBQ_Dataset(config)
            for i in range(len(dataset)):
                prompt = dataset.build_prompt(i)
                samples.append({
                    "text_a": dataset.data[i].get("context", ""),
                    "text_b": dataset.data[i].get("question", ""),
                    "label":  dataset.data[i].get("label", ""),
                    "prompt": prompt,
                })

    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    print(f"Loaded {len(samples)} samples from {args.dataset}/{args.split}")
    return samples


def generate_and_save_activations(
    args: argparse.Namespace,
    model_wrapper: GemmaModel,
    crosscoder: JumpReLUMultiLayerSAE,
    samples: list[dict],
    output_path: Path,
    batch_size: int = 32,
) -> None:
    """Run forward passes, encode through crosscoder, save max-pooled acts."""
    sample_records = []

    for i, sample in enumerate(tqdm(samples, desc="Generating activations")):
        # Generate — returns (text, output_ids, prompt_len)
        # output_ids shape: (1, total_tokens)
        _text, output_ids, prompt_len = model_wrapper.generate(
            sample["prompt"], max_new_tokens=args.max_new_tokens
        )

        # Register hooks to capture residuals at each crosscoder layer
        cache: dict[int, torch.Tensor] = {}
        handles = []

        def _hook(module, inputs, outputs, layer_key: int):
            acts = outputs[0] if isinstance(outputs, tuple) else outputs
            cache[layer_key] = acts.detach().squeeze(0).cpu()  # move to CPU immediately

        for layer in LAYERS:
            h = model_wrapper.model.model.language_model.layers[layer].register_forward_hook(
                lambda mod, inp, out, lk=layer: _hook(mod, inp, out, lk)
            )
            handles.append(h)

        try:
            with torch.no_grad():
                model_wrapper.model(input_ids=output_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        # Reset any internal HF cache state (HybridCache/StaticCache in HF >= 4.40)
        for attr in ("_cache", "past_key_values"):
            if hasattr(model_wrapper.model, attr):
                setattr(model_wrapper.model, attr, None)

        del output_ids  # free before encode

        # Build cc_acts on CPU (already bfloat16 from hook); shape: (n_tokens, num_layers, d_model)
        cc_acts = torch.stack([cache[layer] for layer in LAYERS], dim=1)
        del cache

        # Chunked encode — avoids (n_tokens, 4, 262144) intermediate on GPU
        device = next(crosscoder.parameters()).device
        chunk_size = 64
        max_acts: torch.Tensor | None = None
        for start in range(0, cc_acts.shape[0], chunk_size):
            chunk = cc_acts[start : start + chunk_size].to(device)  # (<=64, 4, d_model) bfloat16
            tc_chunk = crosscoder.encode(chunk)                      # (<=64, 4, 262144)
            chunk_max = tc_chunk.max(dim=0).values.max(dim=0).values  # (262144,)
            max_acts = chunk_max if max_acts is None else torch.maximum(max_acts, chunk_max)
            del chunk, tc_chunk, chunk_max

        del cc_acts

        sample_records.append({
            "text_a":   sample["text_a"],
            "text_b":   sample["text_b"],
            "label":    sample["label"],
            "max_acts": max_acts.cpu().to_sparse(),
        })
        del max_acts
        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % batch_size == 0 or (i + 1) == len(samples):
            # Incremental checkpoint (overwrites after each batch for crash recovery)
            torch.save(
                {
                    "samples": sample_records,
                    "metadata": {
                        "dataset":   args.dataset,
                        "split":     args.split,
                        "model":     MODEL_NAME,
                        "layers":    LAYERS,
                        "cc_width":  WIDTH,
                        "cc_l0":     L0,
                        "n_samples": len(samples),
                    },
                },
                output_path,
            )

    print(f"Saved activations for {len(sample_records)} samples to {output_path}")


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def build_feature_analysis(
    acts_path: Path,
    output_path: Path,
    n_top: int = 20,
    n_nonfiring: int = 20,
    min_prompt_activations: int = 10,
) -> None:
    """Build per-feature top-activating and non-activating example lists."""
    data = torch.load(acts_path, weights_only=False)
    samples = data["samples"]

    # Map feature index → list of (activation_value, sample_index)
    feature_to_firing: dict[int, list[tuple[float, int]]] = {}

    for i, record in enumerate(samples):
        max_acts = record["max_acts"].to_dense()
        firing_indices = max_acts.nonzero(as_tuple=True)[0].tolist()
        for f in firing_indices:
            feature_to_firing.setdefault(f, []).append((max_acts[f].item(), i))

    all_indices = set(range(len(samples)))
    result: dict[int, dict] = {}

    for f, firing_list in feature_to_firing.items():
        # Top-n by activation value
        firing_list.sort(key=lambda x: -x[0])

        if len(firing_list) < min_prompt_activations:
            result[f] = {"top_activating": [], "non_activating": [], "n_firing_prompts": len(firing_list)}
            continue

        top_entries = firing_list[:n_top]

        top_activating = []
        for act_val, idx in top_entries:
            rec = samples[idx]
            top_activating.append({
                "text_a":     rec["text_a"],
                "text_b":     rec["text_b"],
                "label":      rec["label"],
                "activation": act_val,
            })

        # Non-firing: sample from indices that never fired this feature
        firing_set = {idx for _, idx in firing_list}
        non_firing_pool = list(all_indices - firing_set)
        chosen = random.sample(non_firing_pool, min(n_nonfiring, len(non_firing_pool)))

        non_activating = []
        for idx in chosen:
            rec = samples[idx]
            non_activating.append({
                "text_a":     rec["text_a"],
                "text_b":     rec["text_b"],
                "label":      rec["label"],
                "sample_idx": idx,
            })

        result[f] = {
            "top_activating":  top_activating,
            "non_activating":  non_activating,
        }

    torch.save(result, output_path)
    n_below = sum(1 for v in result.values() if not v["top_activating"])
    print(f"Feature analysis: {len(result)} features → {output_path}  ({n_below} below min_prompt_activations threshold)")


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def build_llm_prompt(
    feature_idx: int,
    feature_data: dict,
    dataset_name: str,
) -> str:
    """Build a Claude prompt for labeling a single crosscoder feature."""
    if dataset_name == "esnli":
        field_a, field_b = "Premise", "Hypothesis"
    else:
        field_a, field_b = "Context", "Question"

    top_examples = feature_data["top_activating"]
    non_examples = feature_data["non_activating"]

    lines = [
        f"You are a researcher analyzing a Gemma-scope-2 crosscoder feature "
        f"from layers {LAYERS} (feature index {feature_idx}).",
        "",
        "ACTIVATING EXAMPLES (inputs that strongly activate this feature):",
    ]
    for n, ex in enumerate(top_examples, 1):
        lines.append(
            f'{n}. {field_a}: "{ex["text_a"]}" | {field_b}: "{ex["text_b"]}" | Label: {ex["label"]}'
        )

    lines += ["", "NON-ACTIVATING EXAMPLES (inputs that do NOT activate this feature):"]
    for n, ex in enumerate(non_examples, 1):
        lines.append(
            f'{n}. {field_a}: "{ex["text_a"]}" | {field_b}: "{ex["text_b"]}" | Label: {ex["label"]}'
        )

    lines += [
        "",
        "Based on the contrast between activating and non-activating examples, provide:",
        "1. A concise label (5 words or fewer) describing what this feature detects.",
        "2. Whether the feature is SEMANTIC (captures meaning/content) or SYNTACTIC "
        "(captures grammatical structure/patterns).",
        "",
        "Output format (use exactly these two lines):",
        "LABEL: <5 words or fewer>",
        "TYPE: <SEMANTIC or SYNTACTIC>",
    ]

    return "\n".join(lines)


def label_features_with_llm(
    features_path: Path,
    dataset_name: str,
    output_path: Path,
    api_key: str,
    llm_model: str,
) -> None:
    """Call Claude to label each feature and write results to a JSONL file."""
    import anthropic

    features: dict[int, dict] = torch.load(features_path, weights_only=False)

    # Resume: skip features already in output file
    processed: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed.add(json.loads(line)["feature_idx"])
                except (json.JSONDecodeError, KeyError):
                    pass

    remaining = [fi for fi in features if fi not in processed]
    print(f"Labeling {len(remaining)} features (already done: {len(processed)})")

    client = anthropic.Anthropic(api_key=api_key)

    with open(output_path, "a") as f:
        for feature_idx in tqdm(remaining, desc="LLM labeling"):
            if not features[feature_idx]["top_activating"]:
                record = {"feature_idx": feature_idx, "label": "N/A", "type": "N/A", "raw_response": "below_threshold"}
                f.write(json.dumps(record) + "\n")
                f.flush()
                continue

            prompt = build_llm_prompt(feature_idx, features[feature_idx], dataset_name)

            for attempt in range(3):
                try:
                    response = client.messages.create(
                        model=llm_model,
                        max_tokens=64,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = response.content[0].text

                    label = None
                    label_type = None
                    for line in raw.splitlines():
                        if line.startswith("LABEL:"):
                            label = line[len("LABEL:"):].strip()
                        elif line.startswith("TYPE:"):
                            label_type = line[len("TYPE:"):].strip()

                    record = {
                        "feature_idx":  feature_idx,
                        "label":        label,
                        "type":         label_type,
                        "raw_response": raw,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    break

                except anthropic.RateLimitError:
                    if attempt == 2:
                        raise
                    wait = min(60, 5 * (2 ** attempt))
                    print(f"Rate limit hit, retrying in {wait}s…")
                    time.sleep(wait)

            time.sleep(0.1)

    print(f"Labels written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.dataset}-{args.split}-cc-{WIDTH}-l0{L0}"
    acts_path     = output_dir / f"{prefix}-acts.pt"
    features_path = output_dir / f"{prefix}-features.pt"
    labels_path   = output_dir / f"{prefix}-labels.jsonl"

    # ------------------------------------------------------------------
    # Phase 1 — Activation generation
    # ------------------------------------------------------------------
    if not acts_path.exists() or args.regenerate:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = GemmaModel(ModelConfig(MODEL_NAME, torch_dtype=torch.bfloat16))

        cc_config = CrosscoderConfig(repo_id=REPO_ID, layers=LAYERS, width=WIDTH, l0=L0)
        crosscoder = JumpReLUMultiLayerSAE.from_pretrained(
            cc_config, device=device, hf_token=hf_token
        )

        samples = load_samples(args)
        generate_and_save_activations(args, model, crosscoder, samples, acts_path, batch_size=args.batch_size)

        del model, crosscoder
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"Activations found at {acts_path}, skipping generation")

    # ------------------------------------------------------------------
    # Phase 2 — Feature analysis
    # ------------------------------------------------------------------
    if not features_path.exists() or args.regenerate:
        print("Building feature analysis…")
        build_feature_analysis(acts_path, features_path, min_prompt_activations=args.min_prompt_activations)
    else:
        print(f"Feature analysis found at {features_path}, skipping (use --regenerate to redo)")

    # ------------------------------------------------------------------
    # Phase 3 — LLM labeling
    # ------------------------------------------------------------------
    if not args.skip_llm:
        label_features_with_llm(
            features_path, args.dataset, labels_path, anthropic_api_key, args.llm_model
        )


if __name__ == "__main__":
    main()
