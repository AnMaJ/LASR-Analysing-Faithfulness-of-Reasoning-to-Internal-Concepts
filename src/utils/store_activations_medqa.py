"""Generate transcoder activations for the MedQA dataset and save results.

Loads questions from data/raw/medqa_formatted.parquet, applies a chat template,
generates text + transcoder activations, and saves everything to an activations
folder.

The transcoder (google/gemma-scope-2-27b-it, layer 31, width 262k, l0 medium)
hooks into:
    in:  model.layers.<layer>.pre_feedforward_layernorm.output
    out: model.layers.<layer>.post_feedforward_layernorm.output

Usage:
    python -m src.utils.store_activations_medqa [--batch_size 8] [--max_new_tokens 1024] [--output_dir activations]
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.configs import ModelConfig, TranscoderConfig
from src.gemma_model import GemmaModel
from src.transcoder import JumpReLUTranscoder


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save transcoder activations for MedQA")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens per generation")
    parser.add_argument("--output_dir", type=str, default="activations", help="Directory to save activations")
    parser.add_argument("--parquet_path", type=str, default="data/raw/medqa_formatted.parquet",
                        help="Path to MedQA parquet file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to process (default: all)")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--repo_id", type=str, default="google/gemma-scope-2-27b-it")
    parser.add_argument("--transcoder_layer", type=int, default=31)
    parser.add_argument("--transcoder_width", type=str, default="262k")
    parser.add_argument("--transcoder_l0", type=str, default="medium")
    parser.add_argument("--affine", action="store_true", help="Use the affine transcoder variant")
    return parser.parse_args()


def load_medqa(parquet_path: str, model: GemmaModel, max_samples: int | None = None):
    """Load MedQA questions from parquet, wrap in chat template, return prompts + metadata."""
    df = pd.read_parquet(parquet_path)
    if max_samples is not None:
        df = df.head(max_samples)

    all_prompts = []
    all_question_indices = []

    for _, row in df.iterrows():
        # Wrap the raw question text in the model's chat template
        messages = [{"role": "user", "content": row["question"]}]
        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)
        all_question_indices.append(int(row["question_idx"]))

    print(f"Total samples: {len(all_prompts)}")
    return all_prompts, all_question_indices


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
        affine=args.affine,
    )

    # --- Load model ---
    model = GemmaModel(model_config)

    # --- Load dataset ---
    all_prompts, all_question_indices = load_medqa(args.parquet_path, model, args.max_samples)

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
    total_sq_error = 0.0
    total_n_elem = 0
    total_target_sum = 0.0
    total_target_sum2 = 0.0

    for pre_ffn, post_ffn in tqdm(
        zip(pre_ffn_acts_list, post_ffn_acts_list), total=len(pre_ffn_acts_list), desc="Transcoder encoding"
    ):
        pre_ffn_dev = pre_ffn.to(device).float()
        post_ffn_dev = post_ffn.to(device).float()

        encoded = transcoder.encode(pre_ffn_dev)
        transcoder_activations.append(encoded.detach().cpu().to_sparse())

        # Per-sample stats (L0, per-sample FVU for analysis)
        stats = transcoder.get_reconstruction_stats(pre_ffn_dev, post_ffn_dev, encoded)
        recon_stats.append(stats)

        # Accumulate for global FVU
        recon = transcoder.decode(encoded, input_acts=pre_ffn_dev)
        n_elem = post_ffn_dev.numel()
        total_sq_error += ((recon - post_ffn_dev) ** 2).sum().item()
        total_n_elem += n_elem
        total_target_sum += post_ffn_dev.sum().item()
        total_target_sum2 += (post_ffn_dev ** 2).sum().item()

        del pre_ffn_dev, post_ffn_dev, encoded, recon
        torch.cuda.empty_cache()

    del pre_ffn_acts_list, post_ffn_acts_list
    gc.collect()

    # Global FVU: MSE / Var(target), where Var is computed over all tokens pooled
    global_mean = total_target_sum / total_n_elem
    global_variance = total_target_sum2 / total_n_elem - global_mean ** 2
    global_fvu = (total_sq_error / total_n_elem) / global_variance

    del transcoder
    gc.collect()
    torch.cuda.empty_cache()

    # --- Save everything ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    affine_tag = "-affine" if args.affine else ""
    filename = f"medqa-transcoder-l{args.transcoder_layer}-{args.transcoder_width}{affine_tag}"

    torch.save({
        "transcoder_activations": transcoder_activations,
        "recon_stats": recon_stats,
        "sequence": sequences,
        "prompt_lens": prompt_char_lens,
        "generations_ids": generation_ids,
        "dataset_info": {
            "question_indices": all_question_indices,
        },
        "transcoder_config": {
            "repo_id": transcoder_config.repo_id,
            "layer": transcoder_config.layer,
            "width": transcoder_config.width,
            "l0": transcoder_config.l0,
            "affine": transcoder_config.affine,
        },
    }, output_dir / f"{filename}.pt")

    print(f"Saved {len(transcoder_activations)} samples to {output_dir / f'{filename}.pt'}")
    print(f"  Global FVU (pooled): {global_fvu:.4f}")
    print(f"  Mean FVU (avg/sample): {sum(s['fvu'] for s in recon_stats) / len(recon_stats):.4f}")
    print(f"  Mean L0:  {sum(s['l0'] for s in recon_stats) / len(recon_stats):.1f}")


if __name__ == "__main__":
    main()
