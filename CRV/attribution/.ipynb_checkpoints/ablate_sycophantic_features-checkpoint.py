"""
Causal ablation of sycophancy-enriched transcoder features.

After running:
    python generate_sycophancy_graphs.py  --graph_dir ./graphs/sycophancy
    python analyze_sycophancy_transcoder_features.py \\
        --graph_dir ./graphs/sycophancy --top_k 20 --no_store

this script:
  1. Loads the ranked enriched-feature list from sycophancy_enriched_features.json
     (or accepts explicit layer/feature targets via --features).
  2. Installs forward hooks on the target SingleLayerTranscoder modules that
     subtract the contribution of ablated features from each layer's output.
  3. Re-runs inference on every question whose metadata shows sycophantic
     behaviour (is_sycophantic == 1) and on the matching clean variant.
  4. Reports whether the ablation "fixed" each sycophantic answer and whether
     it damaged answers on clean questions (collateral damage check).

Hook mechanics
--------------
For each target (layer, feature_id), a PyTorch forward_hook is registered on
the SingleLayerTranscoder at that layer.  The hook:
  a) Re-runs the encoder for the ablated feature(s):
       pre_act = (x - b_dec) @ W_enc[:, f] + b_enc[f]
       act_f   = relu(pre_act)
  b) Loads the matching rows of W_dec (works whether W_dec is a regular
     Parameter, a CPU tensor, or a lazy-loaded mapping).
  c) Subtracts  act_f @ W_dec[f]  from the transcoder output, effectively
     clamping feature f's contribution to zero without re-running the full
     forward pass.

Usage
-----
# Ablate the top-5 enriched features (loaded from analysis output):
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --top_n 5

# Sweep over different top-N values (1, 3, 5, 10):
python ablate_sycophancy_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --sweep

# Provide features manually (layer,feature_id pairs):
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --features 21,92372 17,24112 20,167488 \\
    --dataset sycophancy

# Use the math-only dataset instead:
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy_math \\
    --top_n 5 \\
    --dataset sycophancy_math
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Transcoder discovery helpers
# ---------------------------------------------------------------------------

def _find_transcoders(model) -> Dict[int, object]:
    """Walk model.named_modules() and collect {layer_idx: transcoder} entries.

    Works whether the transcoders are stored as model.blocks[i].mlp or via
    a separate TranscoderCollection.  The layer index is inferred from the
    first integer component of the module path.
    """
    try:
        from circuit_tracer.transcoder.single_layer_transcoder import (
            SingleLayerTranscoder,
        )
    except ImportError:
        SingleLayerTranscoder = None

    layer_map: Dict[int, object] = {}
    for name, module in model.named_modules():
        if SingleLayerTranscoder is not None and not isinstance(
            module, SingleLayerTranscoder
        ):
            continue
        # Fallback: accept anything whose class name contains "Transcoder"
        if SingleLayerTranscoder is None:
            cls_name = type(module).__name__
            if "Transcoder" not in cls_name:
                continue

        parts = name.split(".")
        for part in parts:
            try:
                idx = int(part)
                if idx not in layer_map:
                    layer_map[idx] = module
                break
            except ValueError:
                continue

    return layer_map


# ---------------------------------------------------------------------------
# W_dec loading helper  (handles CPU tensors / lazy mmap / Parameters)
# ---------------------------------------------------------------------------

def _load_W_dec_rows(
    module,
    feature_ids: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return W_dec[feature_ids] on *target_device* as *target_dtype*, or None."""
    W_dec = getattr(module, "W_dec", None)
    if W_dec is None:
        return None

    try:
        if isinstance(W_dec, torch.nn.Parameter):
            rows = W_dec.data[feature_ids.cpu()]
        elif isinstance(W_dec, torch.Tensor):
            rows = W_dec[feature_ids.cpu()]
        else:
            # Lazy/mmap type — try slice access then convert to tensor
            raw = W_dec[feature_ids.cpu().tolist()]
            if not isinstance(raw, torch.Tensor):
                import numpy as np

                if isinstance(raw, np.ndarray):
                    raw = torch.from_numpy(raw.copy())
                else:
                    raw = torch.tensor(raw)
            rows = raw
        return rows.to(device=target_device, dtype=target_dtype)
    except Exception as exc:
        warnings.warn(f"Could not load W_dec rows: {exc}")
        return None


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------

def _make_ablation_hook(module, feature_ids: torch.Tensor):
    """Return a forward_hook that zeros out *feature_ids* in *module*'s output.

    Subtraction formula:
        contribution = relu((x - b_dec) @ W_enc[:,f] + b_enc[f]) @ W_dec[f,:]
        output_ablated = output - contribution

    The hook is a no-op if W_dec cannot be loaded (fails gracefully).
    """

    def hook(mod, input_tuple, output):
        x = input_tuple[0]  # [batch, seq, d_model]
        device, dtype = x.device, x.dtype

        # Move encoder weights to the right device
        b_dec = mod.b_dec.to(device=device, dtype=dtype)           # [d_model]
        W_enc_sel = mod.W_enc[:, feature_ids].to(device=device, dtype=dtype)  # [d_model, n_f]
        b_enc_sel = mod.b_enc[feature_ids].to(device=device, dtype=dtype)     # [n_f]

        # Compute feature activations for the ablated set
        # pre_act: [batch, seq, n_f]
        pre_act = (x - b_dec.unsqueeze(0).unsqueeze(0)) @ W_enc_sel + b_enc_sel
        acts = torch.relu(pre_act)  # [batch, seq, n_f]

        # Load W_dec rows
        W_dec_sel = _load_W_dec_rows(mod, feature_ids, device, dtype)  # [n_f, d_model]
        if W_dec_sel is None:
            return output  # Cannot ablate — pass through unchanged

        # Contribution: [batch, seq, n_f] @ [n_f, d_model] = [batch, seq, d_model]
        contribution = acts @ W_dec_sel
        return output - contribution

    return hook


# ---------------------------------------------------------------------------
# Context manager: install / remove hooks around an inference block
# ---------------------------------------------------------------------------

@contextmanager
def ablation_hooks(model, ablation_targets: List[Tuple[int, int]]):
    """Install ablation forward hooks for the duration of the with-block.

    Args:
        model:            ReplacementModel with integrated transcoders.
        ablation_targets: List of (layer_idx, feature_id) pairs to ablate.
    """
    if not ablation_targets:
        yield
        return

    layer_map = _find_transcoders(model)

    # Group feature_ids by layer
    layer_to_feats: Dict[int, List[int]] = defaultdict(list)
    for layer, feat_id in ablation_targets:
        layer_to_feats[layer].append(feat_id)

    handles = []
    for layer, feat_ids in layer_to_feats.items():
        module = layer_map.get(layer)
        if module is None:
            warnings.warn(
                f"No transcoder found for layer {layer}. "
                f"Available layers: {sorted(layer_map.keys())}. Skipping."
            )
            continue
        fids_tensor = torch.tensor(feat_ids, dtype=torch.long)
        hook_fn = _make_ablation_hook(module, fids_tensor)
        handle = module.register_forward_hook(hook_fn)
        handles.append(handle)

    if not handles:
        warnings.warn("No hooks were installed — check layer indices.")

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Inference helper (re-generates until an answer letter appears)
# ---------------------------------------------------------------------------

_ANSWER_LETTERS = {"A", "B", "C", "D"}


def _generate_answer(
    model,
    tokenizer,
    formatted_prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
) -> Optional[str]:
    """Run greedy generation and return the first A/B/C/D token seen."""
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][0]
    bos_id = tokenizer.bos_token_id
    if input_ids.numel() > 0 and input_ids[0].item() == bos_id:
        input_ids = input_ids[1:]

    current_ids = input_ids.clone().to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(current_ids.unsqueeze(0))  # [1, seq, vocab]
        nxt = int(logits[0, -1, :].argmax())
        tok_str = tokenizer.decode([nxt], skip_special_tokens=False).strip()
        if tok_str in _ANSWER_LETTERS:
            return tok_str
        current_ids = torch.cat(
            [current_ids, torch.tensor([nxt], dtype=current_ids.dtype, device=device)]
        )
    return None


# ---------------------------------------------------------------------------
# Load dataset questions
# ---------------------------------------------------------------------------

def _load_questions(dataset_name: str):
    if dataset_name == "sycophancy_math":
        from sycophancy_math_dataset import QUESTIONS
    else:
        from sycophancy_dataset import QUESTIONS
    return QUESTIONS


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ablation_experiment(
    model,
    tokenizer,
    metadata: List[dict],
    questions,
    ablation_targets: List[Tuple[int, int]],
    top_n_label: str = "",
) -> dict:
    """Run baseline → ablation comparison for every sycophantic entry.

    Args:
        model:             ReplacementModel.
        tokenizer:         Matching tokenizer.
        metadata:          List of dicts from metadata.json.
        questions:         The QUESTIONS list from the dataset module.
        ablation_targets:  (layer, feature_id) pairs to ablate.
        top_n_label:       String label used in output (e.g. "top5").

    Returns:
        Dict with keys: results (list of per-question dicts), summary (dict).
    """
    from gemma_custom_prompt_first_token_attribution import build_chat_prompt

    device = next(model.parameters()).device

    # Build a fast lookup: slug -> metadata entry
    slug_to_meta = {e["slug"]: e for e in metadata}

    # Identify sycophantic + their matching clean entries
    syco_entries = [e for e in metadata if e.get("is_sycophantic", 0)]
    faithful_entries = [e for e in metadata if not e.get("is_sycophantic", 0)]

    print(f"\n{'='*70}")
    print(f"Ablation experiment  [{top_n_label or 'custom'}]")
    print(f"  Targets     : {len(ablation_targets)} feature(s)")
    if ablation_targets:
        for layer, fid in ablation_targets[:10]:
            print(f"    L{layer}/F{fid}")
        if len(ablation_targets) > 10:
            print(f"    … (+{len(ablation_targets)-10} more)")
    print(f"  Sycophantic : {len(syco_entries)}")
    print(f"  Faithful    : {len(faithful_entries)}")
    print(f"{'='*70}")

    results = []

    # ------------------------------------------------------------------ #
    # A.  Sycophantic entries: does ablation fix them?                    #
    # ------------------------------------------------------------------ #
    print("\n--- Sycophantic questions ---")
    for entry in syco_entries:
        q_idx = entry["question_idx"]
        domain = entry["domain"]
        correct = entry["correct_answer"]
        hinted = entry["hinted_answer"]
        variant = entry["variant"]  # always 'hint' for sycophantic entries
        slug = entry["slug"]

        q = questions[q_idx]
        raw_text = q[f"question_{variant}"]
        formatted = build_chat_prompt(tokenizer, raw_text, system_instruction="")

        # --- Baseline (no ablation) ---
        baseline_answer = _generate_answer(model, tokenizer, formatted, device)

        # --- Ablated ---
        with ablation_hooks(model, ablation_targets):
            ablated_answer = _generate_answer(model, tokenizer, formatted, device)

        was_fixed = (ablated_answer == correct and baseline_answer != correct)
        was_broken = (ablated_answer != correct and baseline_answer == correct)

        marker = "✓ FIXED" if was_fixed else ("✗ BROKE" if was_broken else "  same ")
        print(
            f"  {slug:<22}  [{domain:<22}]  "
            f"baseline={baseline_answer}  ablated={ablated_answer}  "
            f"correct={correct}  hinted={hinted}  {marker}"
        )

        results.append({
            "slug": slug,
            "domain": domain,
            "variant": variant,
            "correct_answer": correct,
            "hinted_answer": hinted,
            "was_sycophantic": True,
            "baseline_answer": baseline_answer,
            "ablated_answer": ablated_answer,
            "was_fixed": was_fixed,
            "was_broken_by_ablation": was_broken,
        })

    # ------------------------------------------------------------------ #
    # B.  Clean/faithful hint entries: collateral damage check            #
    # ------------------------------------------------------------------ #
    print("\n--- Faithful questions (collateral damage check) ---")
    for entry in faithful_entries:
        q_idx = entry["question_idx"]
        domain = entry["domain"]
        correct = entry["correct_answer"]
        variant = entry["variant"]
        slug = entry["slug"]

        q = questions[q_idx]
        raw_text = q[f"question_{variant}"]
        formatted = build_chat_prompt(tokenizer, raw_text, system_instruction="")

        baseline_answer = _generate_answer(model, tokenizer, formatted, device)
        with ablation_hooks(model, ablation_targets):
            ablated_answer = _generate_answer(model, tokenizer, formatted, device)

        changed = baseline_answer != ablated_answer
        was_broken = changed and (ablated_answer != correct)
        marker = "  same " if not changed else ("✗ BROKE" if was_broken else "  chg  ")
        print(
            f"  {slug:<22}  [{domain:<22}]  "
            f"baseline={baseline_answer}  ablated={ablated_answer}  "
            f"correct={correct}  {marker}"
        )

        results.append({
            "slug": slug,
            "domain": domain,
            "variant": variant,
            "correct_answer": correct,
            "hinted_answer": entry.get("hinted_answer", ""),
            "was_sycophantic": False,
            "baseline_answer": baseline_answer,
            "ablated_answer": ablated_answer,
            "was_fixed": False,
            "was_broken_by_ablation": was_broken,
        })

    # ------------------------------------------------------------------ #
    # C.  Summary                                                          #
    # ------------------------------------------------------------------ #
    syco_results = [r for r in results if r["was_sycophantic"]]
    faith_results = [r for r in results if not r["was_sycophantic"]]

    n_fixed = sum(r["was_fixed"] for r in syco_results)
    n_syco = len(syco_results)
    n_collateral = sum(r["was_broken_by_ablation"] for r in faith_results)
    n_faithful = len(faith_results)

    summary = {
        "top_n_label": top_n_label,
        "n_ablation_targets": len(ablation_targets),
        "n_sycophantic": n_syco,
        "n_fixed": n_fixed,
        "fix_rate": n_fixed / max(n_syco, 1),
        "n_faithful": n_faithful,
        "n_collateral_damage": n_collateral,
        "collateral_rate": n_collateral / max(n_faithful, 1),
    }

    print(f"\n{'='*70}")
    print(f"SUMMARY  [{top_n_label or 'custom'}]")
    print(f"  Sycophantic fixed      : {n_fixed}/{n_syco}  "
          f"({100*summary['fix_rate']:.1f}%)")
    print(f"  Faithful broken        : {n_collateral}/{n_faithful}  "
          f"({100*summary['collateral_rate']:.1f}%)")
    print(f"{'='*70}")

    return {"results": results, "summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    graph_dir = Path(args.graph_dir)

    # ------------------------------------------------------------------ #
    # 1. Load metadata                                                     #
    # ------------------------------------------------------------------ #
    meta_path = graph_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {graph_dir}. "
            "Run generate_sycophancy_graphs.py first."
        )
    with open(meta_path) as fh:
        metadata = json.load(fh)

    questions = _load_questions(args.dataset)

    # ------------------------------------------------------------------ #
    # 2. Determine ablation targets                                         #
    # ------------------------------------------------------------------ #
    if args.features:
        # Manually specified: --features 21,92372 17,24112 ...
        manual_targets: List[Tuple[int, int]] = []
        for spec in args.features:
            parts = spec.split(",")
            if len(parts) != 2:
                raise ValueError(f"--features format: layer,feature_id  (got {spec!r})")
            manual_targets.append((int(parts[0]), int(parts[1])))
        all_target_sets = [("manual", manual_targets)]

    elif args.sweep:
        # Sweep over top-N values
        enriched_path = graph_dir / "sycophancy_enriched_features.json"
        if not enriched_path.exists():
            raise FileNotFoundError(
                f"sycophancy_enriched_features.json not found in {graph_dir}. "
                "Run analyze_sycophancy_transcoder_features.py first."
            )
        with open(enriched_path) as fh:
            ranked: List[dict] = json.load(fh)

        sweep_ns = [1, 3, 5, 10, 20]
        all_target_sets = []
        for n in sweep_ns:
            targets = [(r["layer"], r["feature_id"]) for r in ranked[:n]]
            all_target_sets.append((f"top{n}", targets))
        # Also include baseline (0 features → no ablation)
        all_target_sets.insert(0, ("baseline_no_ablation", []))

    else:
        # Default: load top_n from enriched features file
        enriched_path = graph_dir / "sycophancy_enriched_features.json"
        if not enriched_path.exists():
            raise FileNotFoundError(
                f"sycophancy_enriched_features.json not found in {graph_dir}. "
                "Run analyze_sycophancy_transcoder_features.py first."
            )
        with open(enriched_path) as fh:
            ranked = json.load(fh)

        targets = [(r["layer"], r["feature_id"]) for r in ranked[: args.top_n]]
        all_target_sets = [(f"top{args.top_n}", targets)]

    # ------------------------------------------------------------------ #
    # 3. Load model (once — reused across all ablation rounds)             #
    # ------------------------------------------------------------------ #
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    from gemma_custom_prompt_first_token_attribution import load_model_lazy

    print(f"Loading model …")
    model, tokenizer = load_model_lazy(dtype=dtype)
    device = next(model.parameters()).device

    # ------------------------------------------------------------------ #
    # 4. Verify transcoder discovery                                        #
    # ------------------------------------------------------------------ #
    layer_map = _find_transcoders(model)
    if not layer_map:
        raise RuntimeError(
            "No transcoders found in model.  Check that circuit_tracer's "
            "SingleLayerTranscoder is importable and that ReplacementModel "
            "integrates transcoders as named sub-modules."
        )
    print(f"Found {len(layer_map)} transcoders: layers {sorted(layer_map.keys())}")

    # ------------------------------------------------------------------ #
    # 5. Run ablation experiments                                           #
    # ------------------------------------------------------------------ #
    all_summaries = []
    for label, targets in all_target_sets:
        exp = run_ablation_experiment(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            questions=questions,
            ablation_targets=targets,
            top_n_label=label,
        )
        all_summaries.append(exp["summary"])

        # Save per-run results
        if args.save_results:
            out_path = graph_dir / f"ablation_results_{label}.json"
            with open(out_path, "w") as fh:
                json.dump(exp, fh, indent=2)
            print(f"  Results saved to: {out_path}")

    # ------------------------------------------------------------------ #
    # 6. Print comparison table (if sweep or multiple runs)                #
    # ------------------------------------------------------------------ #
    if len(all_summaries) > 1:
        col = 28
        print(f"\n{'='*70}")
        print("Sweep comparison table")
        print(f"{'='*70}")
        header = (
            f"{'Run':<{col}}"
            f"{'#Ablated':>10}"
            f"{'FixRate':>10}"
            f"{'CollDmg':>10}"
        )
        print(header)
        print("-" * 70)
        for s in all_summaries:
            print(
                f"{s['top_n_label']:<{col}}"
                f"{s['n_ablation_targets']:>10}"
                f"{100*s['fix_rate']:>9.1f}%"
                f"{100*s['collateral_rate']:>9.1f}%"
            )
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablate sycophancy-enriched transcoder features and measure answer changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--graph_dir",
        type=str,
        default="./graphs/sycophancy",
        help="Directory with metadata.json and sycophancy_enriched_features.json.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="sycophancy",
        choices=["sycophancy", "sycophancy_math"],
        help="Which QUESTIONS list to use for re-running inference.",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top enriched features to ablate (ignored if --features or --sweep).",
    )
    p.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="Explicit targets as layer,feature_id pairs, e.g. --features 21,92372 17,24112.",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Sweep over top-N = {0,1,3,5,10,20} and print a comparison table.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    p.add_argument(
        "--save_results",
        action="store_true",
        default=False,
        help="Save per-run results to JSON files in graph_dir.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
