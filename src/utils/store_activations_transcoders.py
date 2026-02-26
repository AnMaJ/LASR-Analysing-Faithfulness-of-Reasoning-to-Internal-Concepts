"""Reproduce the demo notebook pipeline up to activation generation and save results.

Loads 100 disambig samples from each of the 9 BBQ categories, concatenates them
into a single 900-sample dataset, generates text + transcoder activations, and
saves everything to an activations folder.

The transcoder (google/gemma-scope-2-27b-it, layer 31, width 262k, l0 medium)
hooks into:
    in:  model.layers.<layer>.pre_feedforward_layernorm.output
    out: model.layers.<layer>.post_feedforward_layernorm.output

Usage:
    python -m src.utils.store_activations_transcoders [--batch_size 8] [--max_new_tokens 1024] [--output_dir activations]
"""

import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from src.configs import ModelConfig, PromptStyle, DatasetConfig, TranscoderConfig
from src.gemma_model import GemmaModel
from src.dataset.bbq import BBQ_Dataset
from src.transcoder import JumpReLUTranscoder

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
    parser = argparse.ArgumentParser(description="Generate and save model/transcoder activations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens per generation")
    parser.add_argument("--output_dir", type=str, default="activations", help="Directory to save activations")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--repo_id", type=str, default="google/gemma-scope-2-27b-it")
    parser.add_argument("--transcoder_layer", type=int, default=31)
    parser.add_argument("--transcoder_width", type=str, default="262k")
    parser.add_argument("--transcoder_l0", type=str, default="medium")
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

    transcoder_config = TranscoderConfig(
        repo_id=args.repo_id,
        layer=args.transcoder_layer,
        width=args.transcoder_width,
        l0=args.transcoder_l0,
    )

    # --- Load all categories ---
    all_prompts, all_categories, all_ground_truths = load_all_categories()

    # --- Load model ---
    model = GemmaModel(model_config)

    # --- Generate text ---
    generations, generation_ids, prompt_lens = model.generate_batch(
        all_prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    # --- Gather feedforward activations (transcoder input + reconstruction target) ---
    pre_ffn_acts_list = []
    post_ffn_acts_list = []
    for generation_id, prompt_len in tqdm(
        zip(generation_ids, prompt_lens), total=len(generation_ids), desc="Extracting feedforward activations"
    ):
        pre_ffn, post_ffn = model.gather_feedforward_activations(
            transcoder_config.layer, generation_id
        )
        pre_ffn_acts_list.append(pre_ffn[prompt_len:].cpu())
        post_ffn_acts_list.append(post_ffn[prompt_len:].cpu())
        del pre_ffn, post_ffn

    # Decode full sequences and compute character-level prompt boundaries
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

    # --- Load transcoder and encode ---
    transcoder = JumpReLUTranscoder.from_pretrained(transcoder_config, device=device)

    transcoder_activations = []
    recon_stats = []

    # Accumulators for global FVU (pooled across all tokens, not averaged per-sample)
    total_sq_error = 0.0   # sum of (recon - target)^2 over all tokens × d_model
    total_n_elem   = 0     # total number of scalar elements
    total_target_sum  = 0.0
    total_target_sum2 = 0.0

    for pre_ffn, post_ffn in tqdm(
        zip(pre_ffn_acts_list, post_ffn_acts_list), total=len(pre_ffn_acts_list), desc="Transcoder encoding"
    ):
        pre_ffn_dev  = pre_ffn.to(device).float()
        post_ffn_dev = post_ffn.to(device).float()

        encoded = transcoder.encode(pre_ffn_dev)
        transcoder_activations.append(encoded.detach().cpu().to_sparse())

        # Per-sample stats (L0, per-sample FVU for analysis)
        stats = transcoder.get_reconstruction_stats(pre_ffn_dev, post_ffn_dev, encoded)
        recon_stats.append(stats)

        # Accumulate for global FVU
        recon = transcoder.decode(encoded)
        n_elem = post_ffn_dev.numel()
        total_sq_error    += ((recon - post_ffn_dev) ** 2).sum().item()
        total_n_elem      += n_elem
        total_target_sum  += post_ffn_dev.sum().item()
        total_target_sum2 += (post_ffn_dev ** 2).sum().item()

        del pre_ffn_dev, post_ffn_dev, encoded, recon
        torch.cuda.empty_cache()

    del pre_ffn_acts_list, post_ffn_acts_list
    gc.collect()

    # Global FVU: MSE / Var(target), where Var is computed over all tokens pooled
    global_mean     = total_target_sum / total_n_elem
    global_variance = total_target_sum2 / total_n_elem - global_mean ** 2
    global_fvu      = (total_sq_error / total_n_elem) / global_variance

    del transcoder
    gc.collect()
    torch.cuda.empty_cache()

    # --- Save everything ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"bbq-transcoder-l{args.transcoder_layer}-{args.transcoder_width}"

    torch.save({
        "transcoder_activations": transcoder_activations,
        "recon_stats": recon_stats,
        "sequence": sequences,
        "prompt_lens": prompt_char_lens,
        "generations_ids": generation_ids,
        "dataset_info": {
            "categories": all_categories,
            "ground_truths": all_ground_truths,
        },
        "transcoder_config": {
            "repo_id": transcoder_config.repo_id,
            "layer": transcoder_config.layer,
            "width": transcoder_config.width,
            "l0": transcoder_config.l0,
        },
    }, output_dir / f"{filename}.pt")

    print(f"Saved {len(transcoder_activations)} samples to {output_dir / f'{filename}.pt'}")
    print(f"  Global FVU (pooled): {global_fvu:.4f}")
    print(f"  Mean FVU (avg/sample): {sum(s['fvu'] for s in recon_stats) / len(recon_stats):.4f}")
    print(f"  Mean L0:  {sum(s['l0'] for s in recon_stats) / len(recon_stats):.1f}")


if __name__ == "__main__":
    main()
