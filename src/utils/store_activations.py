"""Reproduce the demo notebook pipeline up to activation generation and save results.

Loads 100 disambig samples from each of the 9 BBQ categories, concatenates them
into a single 900-sample dataset, generates text + residual/SAE activations, and
saves everything to an activations folder.

Usage:
    python scripts/generate_activations.py [--batch_size 8] [--max_new_tokens 1024] [--output_dir activations]
"""

import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from src.configs import ModelConfig, PromptStyle, DatasetConfig, SAEConfig
from src.gemma_model import GemmaModel
from src.dataset.bbq import BBQ_Dataset
from src.SAE import JumpReLUSAE

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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save model/SAE activations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens per generation")
    parser.add_argument("--output_dir", type=str, default="activations", help="Directory to save activations")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--repo_id", type=str, default="google/gemma-scope-2-27b-it")
    parser.add_argument("--sae_layer", type=int, default=31)
    parser.add_argument("--sae_width", type=str, default="262k")
    parser.add_argument("--sae_l0", type=str, default="medium")
    return parser.parse_args()


def load_all_categories():
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


def main():
    args = parse_args()

    # --- Device ---
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Configs ---
    model_config = ModelConfig(model_name=args.model_name, device=device)

    sae_config = SAEConfig(
        repo_id=args.repo_id,
        sae_type="resid_post",
        layer=args.sae_layer,
        width=args.sae_width,
        l0=args.sae_l0,
    )

    # --- Load all categories ---
    all_prompts, all_categories, all_ground_truths = load_all_categories()

    # --- Load model ---
    model = GemmaModel(model_config)

    # --- Generate text ---
    generations, generation_ids, prompt_lens = model.generate_batch(
        all_prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    residuals = []
    for generation_id, prompt_len in tqdm(zip(generation_ids, prompt_lens), total=len(generation_ids), desc="Extracting residuals"):
        residual_acts = model.gather_residual_activations(sae_config.layer, generation_id)
        residuals.append(residual_acts[prompt_len:].cpu())
        del residual_acts

    # Decode full sequences (prompt + generation) and compute character-level prompt boundaries
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

    sae_activations = []
    recon_stats = []  # per-sample FVU and L0
    for residual in tqdm(residuals, desc="SAE encoding"):
        gen_residual = residual.to(device).float()
        encoded = sae.encode(gen_residual)
        sae_activations.append(encoded.detach().cpu().to_sparse())
        stats = sae.get_reconstruction_stats(gen_residual, encoded)
        recon_stats.append(stats)
        del gen_residual, encoded
        torch.cuda.empty_cache()

    # Free residuals — reconstruction stats are already computed
    del residuals
    gc.collect()

    del sae
    gc.collect()
    torch.cuda.empty_cache()

    # --- Save everything ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"bbq-l{args.sae_layer}-{args.sae_width}"

    torch.save({
        "sae_activations": sae_activations,
        "recon_stats": recon_stats,
        "sequence": sequences,
        "prompt_lens": prompt_char_lens,
        "generations_isd": generation_ids,
        "dataset_info": {"categories": all_categories,
        "ground_truths": all_ground_truths},
        "sae_config": {
            "repo_id": sae_config.repo_id,
            "sae_type": sae_config.sae_type,
            "layer": sae_config.layer,
            "width": sae_config.width,
            "l0": sae_config.l0,
        },
    }, output_dir / f"{filename}.pt")

    print(f"Saved {len(sae_activations)} samples to {output_dir / 'activations.pt'}")
    print(f"  Mean FVU: {sum(s['fvu'] for s in recon_stats) / len(recon_stats):.4f}")
    print(f"  Mean L0:  {sum(s['l0'] for s in recon_stats) / len(recon_stats):.1f}")


if __name__ == "__main__":
    main()
