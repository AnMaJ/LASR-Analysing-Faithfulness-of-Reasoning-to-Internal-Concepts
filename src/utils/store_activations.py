"""Generate SAE-encoded activations for BBQ or e-SNLI and save to disk.

BBQ mode:  loads 100 disambig samples from each of the 9 BBQ categories.
e-SNLI mode: streams N samples from HuggingFace, builds few-shot CoT prompts.

Usage:
    python -m src.utils.store_activations --dataset bbq  [options]
    python -m src.utils.store_activations --dataset esnli [options]
"""

import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset, Dataset as HFDataset

from src.configs import ModelConfig, PromptStyle, DatasetConfig, SAEConfig
from src.gemma_model import GemmaModel
from src.dataset.bbq import BBQ_Dataset
from src.dataset.esnli import ESNLI_Dataset, _LABEL_MAP_
from src.SAE import JumpReLUSAE

# ---------------------------------------------------------------------------
# BBQ constants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# e-SNLI streaming helper
# ---------------------------------------------------------------------------

class _StreamedESNLI(ESNLI_Dataset):
    """Thin wrapper that accepts pre-loaded data to skip re-downloading."""
    def __init__(self, config, preloaded_data):
        self.path = config.path
        self.prompt_style = config.prompt_style
        self.use_chat_template = config.use_chat_template
        self.few_shot = config.few_shot
        self.data = preloaded_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save SAE-encoded activations")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "esnli"],
                        help="Which dataset pipeline to run")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens per generation")
    parser.add_argument("--output_dir", type=str, default="activations", help="Directory to save activations")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--repo_id", type=str, default="google/gemma-scope-2-27b-it")
    parser.add_argument("--sae_layer", type=int, default=31)
    parser.add_argument("--sae_width", type=str, default="65k")
    parser.add_argument("--sae_l0", type=str, default="medium")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=list(TORCH_DTYPE_MAP.keys()),
                        help="Dtype for model loading (default: bfloat16)")

    # e-SNLI specific
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of e-SNLI samples to load (esnli only)")
    parser.add_argument("--split", type=str, default="validation",
                        help="HF split for e-SNLI (esnli only)")
    parser.add_argument("--few_shot", action="store_true",
                        help="Enable few-shot prompts (esnli only)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_bbq():
    """Load 100 disambig samples from each BBQ category and return concatenated prompts + metadata."""
    all_prompts = []
    all_categories = []
    all_ground_truths = []

    for category in BBQ_CATEGORIES:
        print(f"Loading category: {category}")
        config = DatasetConfig(
            path="HiTZ/bbq",
            prompt_style=PromptStyle.CHAIN_OF_THOUGHT_TAGS,
            use_chat_template=True,
            hf_data_config={"name": f"{category}_disambig", "split": "test"},
        )
        dataset = BBQ_Dataset(config)

        n = min(SAMPLES_PER_CATEGORY, len(dataset))
        for i in range(n):
            all_prompts.append(dataset[i])
            all_categories.append(category)
            all_ground_truths.append(dataset.data[i])

    print(f"Total samples: {len(all_prompts)} ({len(BBQ_CATEGORIES)} categories x up to {SAMPLES_PER_CATEGORY})")
    return all_prompts, all_categories, all_ground_truths


def load_esnli(args):
    """Stream e-SNLI, materialize *num_samples* rows, build few-shot CoT prompts."""
    print(f"Streaming e-SNLI ({args.split}), taking {args.num_samples} samples...")
    stream = hf_load_dataset(
        "esnli/esnli",
        split=args.split,
        streaming=True,
        revision="refs/convert/parquet",
    )

    rows = []
    for i, row in enumerate(stream):
        if i >= args.num_samples:
            break
        row["gold_label"] = _LABEL_MAP_[row["label"]]
        rows.append(row)

    data = HFDataset.from_list(rows)
    print(f"Materialized {len(data)} samples from streaming")

    dataset_config = DatasetConfig(
        path="esnli/esnli",
        prompt_style=PromptStyle.CHAIN_OF_THOUGHT_TAGS,
        use_chat_template=False,
        few_shot=args.few_shot,
    )
    esnli_dataset = _StreamedESNLI(dataset_config, data)
    prompted_data = esnli_dataset.build_prompts()

    all_prompts = prompted_data["prompt"]
    all_ground_truths = prompted_data["gold_label"]
    print(f"Built {len(all_prompts)} prompts (few_shot={args.few_shot})")
    return all_prompts, all_ground_truths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Device ---
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Configs ---
    model_config = ModelConfig(
        model_name=args.model_name,
        device=device,
        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
    )

    sae_config = SAEConfig(
        repo_id=args.repo_id,
        sae_type="resid_post",
        layer=args.sae_layer,
        width=args.sae_width,
        l0=args.sae_l0,
    )

    # --- Load dataset ---
    all_categories = None
    if args.dataset == "bbq":
        all_prompts, all_categories, all_ground_truths = load_bbq()
    else:
        all_prompts, all_ground_truths = load_esnli(args)

    # --- Load model ---
    model = GemmaModel(model_config)

    # --- Generate text ---
    generations, generation_ids, prompt_lens = model.generate_batch(
        all_prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    # --- Extract residual activations (generation tokens only) ---
    residuals = []
    for generation_id, prompt_len in tqdm(zip(generation_ids, prompt_lens), total=len(generation_ids), desc="Extracting residuals"):
        residual_acts = model.gather_residual_activations(sae_config.layer, generation_id)
        residuals.append(residual_acts[prompt_len:].cpu())

    # --- Move all CUDA tensors to CPU and free Gemma model to reclaim VRAM ---
    generation_ids = [ids.cpu() for ids in generation_ids]

    sequences = []
    prompt_char_lens = []
    for gen_ids, prompt_len in zip(generation_ids, prompt_lens):
        full_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)
        prompt_text = model.tokenizer.decode(gen_ids[:prompt_len], skip_special_tokens=True)
        sequences.append(full_text)
        prompt_char_lens.append(len(prompt_text))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Encode with SAE and compute reconstruction stats ---
    sae = JumpReLUSAE.from_pretrained(sae_config, device=device)

    sae_encodings = []
    recon_stats = []
    for residual in tqdm(residuals, desc="SAE encoding"):
        gen_residual = residual.to(device).float()
        encoded = sae.encode(gen_residual)
        sae_encodings.append(encoded.cpu().to_sparse())
        stats = sae.get_reconstruction_stats(gen_residual)
        recon_stats.append(stats)
        del gen_residual, encoded
        torch.cuda.empty_cache()

    del residuals
    gc.collect()

    del sae
    gc.collect()
    torch.cuda.empty_cache()

    # --- Build dataset_info ---
    if args.dataset == "bbq":
        dataset_info = {
            "categories": all_categories,
            "ground_truths": all_ground_truths,
        }
    else:
        dataset_info = {
            "ground_truths": all_ground_truths,
            "num_samples": args.num_samples,
            "split": args.split,
            "few_shot": args.few_shot,
        }

    # --- Save everything ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{args.dataset}-l{args.sae_layer}-{args.sae_width}"

    torch.save({
        "sae_encodings": sae_encodings,
        "recon_stats": recon_stats,
        "sequence": sequences,
        "prompt_lens": prompt_char_lens,
        "dataset_info": dataset_info,
        "sae_config": {
            "repo_id": sae_config.repo_id,
            "sae_type": sae_config.sae_type,
            "layer": sae_config.layer,
            "width": sae_config.width,
            "l0": sae_config.l0,
        },
    }, output_dir / f"{filename}.pt")

    print(f"Saved {len(sae_encodings)} samples to {output_dir / f'{filename}.pt'}")
    print(f"  Mean FVU: {sum(s['fvu'] for s in recon_stats) / len(recon_stats):.4f}")
    print(f"  Mean L0:  {sum(s['l0'] for s in recon_stats) / len(recon_stats):.1f}")


if __name__ == "__main__":
    main()
