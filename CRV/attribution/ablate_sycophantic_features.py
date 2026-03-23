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
# Negative steering / ablation (scale=0): fix sycophantic answers
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --top_n 5 --scale 0.0

# Positive steering / amplification (scale=5): induce sycophancy on clean questions
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --top_n 5 --scale 5.0

# Full scale sweep [1.0, 0.5, 0.25, 0.0, 2.0, 5.0, 10.0] for top-5 features:
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --top_n 5 --scale_sweep --save_results

# Sweep feature counts AND scales:
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --sweep --scale_sweep

# Provide features manually:
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --features 21,92372 17,24112 20,167488 --scale 0.0

# Math dataset:
python ablate_sycophantic_features.py \\
    --graph_dir ./graphs/sycophancy_math \\
    --top_n 5 --scale_sweep --dataset sycophancy_math
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

def _make_steering_hook(module, feature_ids: torch.Tensor, scale: float):
    """Return a forward_hook that steers *feature_ids* in *module*'s output.

    Generalised steering formula:
        contribution = relu((x - b_dec) @ W_enc[:,f] + b_enc[f]) @ W_dec[f,:]
        output_steered = output + (scale - 1) * contribution

    Special cases:
        scale = 0.0  → full ablation   (zero the feature)
        scale = 1.0  → identity        (no change)
        scale > 1.0  → amplification   (positive steering)
        scale < 0.0  → sign inversion  (flip the feature's contribution)

    The hook is a no-op if W_dec cannot be loaded (fails gracefully).
    """

    def hook(mod, input_tuple, output):
        x = input_tuple[0]  # [batch, seq, d_model]
        device, dtype = x.device, x.dtype

        b_dec    = mod.b_dec.to(device=device, dtype=dtype)                        # [d_model]
        W_enc_sel = mod.W_enc[:, feature_ids].to(device=device, dtype=dtype)       # [d_model, n_f]
        b_enc_sel = mod.b_enc[feature_ids].to(device=device, dtype=dtype)          # [n_f]

        # Feature activations for the steered set: [batch, seq, n_f]
        pre_act = (x - b_dec.unsqueeze(0).unsqueeze(0)) @ W_enc_sel + b_enc_sel
        acts = torch.relu(pre_act)

        W_dec_sel = _load_W_dec_rows(mod, feature_ids, device, dtype)  # [n_f, d_model]
        if W_dec_sel is None:
            return output  # W_dec unavailable — pass through unchanged

        # contribution: [batch, seq, d_model]
        contribution = acts @ W_dec_sel
        # (scale - 1) == -1 for full ablation, +1 for doubling, etc.
        return output + (scale - 1.0) * contribution

    return hook


# ---------------------------------------------------------------------------
# Context manager: install / remove hooks around an inference block
# ---------------------------------------------------------------------------

@contextmanager
def steering_hooks(model, targets: List[Tuple[int, int]], scale: float):
    """Install steering forward hooks for the duration of the with-block.

    Args:
        model:   ReplacementModel with integrated transcoders.
        targets: List of (layer_idx, feature_id) pairs to steer.
        scale:   Steering multiplier (0 = ablate, >1 = amplify, <0 = invert).
    """
    if not targets or scale == 1.0:
        yield
        return

    layer_map = _find_transcoders(model)

    layer_to_feats: Dict[int, List[int]] = defaultdict(list)
    for layer, feat_id in targets:
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
        hook_fn = _make_steering_hook(module, fids_tensor, scale)
        handle = module.register_forward_hook(hook_fn)
        handles.append(handle)

    if not handles:
        warnings.warn("No hooks were installed — check layer indices.")

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# Keep the old name as an alias for backwards compatibility
@contextmanager
def ablation_hooks(model, ablation_targets: List[Tuple[int, int]]):
    with steering_hooks(model, ablation_targets, scale=0.0):
        yield


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

def run_steering_experiment(
    model,
    tokenizer,
    metadata: List[dict],
    questions,
    targets: List[Tuple[int, int]],
    scale: float,
    label: str = "",
) -> dict:
    """Run a single steering pass (one scale value) over all questions.

    scale < 1  → negative / ablation steering:
                 test sycophantic hint questions  (does answer flip to correct?)
                 + faithful questions             (collateral damage?)
    scale > 1  → positive / amplification steering:
                 test clean questions             (does answer flip to hinted wrong?)
                 + sycophantic hint questions     (does it make things worse?)
    scale = 1  → baseline (no steering), run all questions once.

    Args:
        model:     ReplacementModel.
        tokenizer: Matching tokenizer.
        metadata:  List of dicts from metadata.json.
        questions: QUESTIONS list from dataset module.
        targets:   (layer, feature_id) pairs to steer.
        scale:     Steering multiplier (0 = ablate, 1 = baseline, >1 = amplify).
        label:     Human-readable label for the run.

    Returns:
        Dict with keys: results, summary.
    """
    from gemma_custom_prompt_first_token_attribution import build_chat_prompt

    device = next(model.parameters()).device
    is_ablation = scale < 1.0
    is_amplify  = scale > 1.0

    syco_entries  = [e for e in metadata if e.get("is_sycophantic", 0)]
    clean_entries = [e for e in metadata if e.get("variant") == "clean"]
    faith_entries = [e for e in metadata if not e.get("is_sycophantic", 0)]

    direction = "ablation" if is_ablation else ("amplify" if is_amplify else "baseline")
    print(f"\n{'='*70}")
    print(f"Steering experiment  [{label}]  scale={scale:.2f}  ({direction})")
    print(f"  Targets       : {len(targets)} feature(s)")
    for ly, fi in targets[:8]:
        print(f"    L{ly}/F{fi}")
    if len(targets) > 8:
        print(f"    … (+{len(targets)-8} more)")
    print(f"  Sycophantic   : {len(syco_entries)}")
    print(f"  Clean         : {len(clean_entries)}")
    print(f"{'='*70}")

    results = []

    # ------------------------------------------------------------------ #
    # A. Negative steering / ablation: fix sycophantic hint questions     #
    # ------------------------------------------------------------------ #
    if is_ablation or scale == 1.0:
        print(f"\n--- Sycophantic (hint) questions  [scale={scale:.2f}] ---")
        for entry in syco_entries:
            q_idx   = entry["question_idx"]
            domain  = entry["domain"]
            correct = entry["correct_answer"]
            hinted  = entry["hinted_answer"]
            variant = entry["variant"]
            slug    = entry["slug"]

            raw  = questions[q_idx][f"question_{variant}"]
            fmt  = build_chat_prompt(tokenizer, raw, system_instruction="")

            baseline = _generate_answer(model, tokenizer, fmt, device)
            with steering_hooks(model, targets, scale):
                steered = _generate_answer(model, tokenizer, fmt, device)

            fixed    = steered == correct and baseline != correct
            worsened = steered != correct and baseline == correct
            marker   = "✓ FIXED" if fixed else ("✗ WORSE" if worsened else "  same ")
            print(
                f"  {slug:<22}  [{domain:<20}]  "
                f"base={baseline}  steered={steered}  "
                f"correct={correct}  hinted={hinted}  {marker}"
            )
            results.append({
                "slug": slug, "domain": domain, "variant": variant,
                "scale": scale, "label": label,
                "correct_answer": correct, "hinted_answer": hinted,
                "question_type": "sycophantic_hint",
                "baseline_answer": baseline, "steered_answer": steered,
                "was_fixed": fixed, "was_worsened": worsened,
                "induced_sycophancy": False,
            })

        print(f"\n--- Faithful questions (collateral damage)  [scale={scale:.2f}] ---")
        for entry in faith_entries:
            q_idx   = entry["question_idx"]
            domain  = entry["domain"]
            correct = entry["correct_answer"]
            variant = entry["variant"]
            slug    = entry["slug"]

            raw  = questions[q_idx][f"question_{variant}"]
            fmt  = build_chat_prompt(tokenizer, raw, system_instruction="")

            baseline = _generate_answer(model, tokenizer, fmt, device)
            with steering_hooks(model, targets, scale):
                steered = _generate_answer(model, tokenizer, fmt, device)

            changed = baseline != steered
            broken  = changed and steered != correct
            marker  = "  same " if not changed else ("✗ BROKE" if broken else "  chg  ")
            print(
                f"  {slug:<22}  [{domain:<20}]  "
                f"base={baseline}  steered={steered}  correct={correct}  {marker}"
            )
            results.append({
                "slug": slug, "domain": domain, "variant": variant,
                "scale": scale, "label": label,
                "correct_answer": correct,
                "hinted_answer": entry.get("hinted_answer", ""),
                "question_type": "faithful",
                "baseline_answer": baseline, "steered_answer": steered,
                "was_fixed": False, "was_worsened": broken,
                "induced_sycophancy": False,
            })

    # ------------------------------------------------------------------ #
    # B. Positive steering / amplification: induce sycophancy on clean   #
    # ------------------------------------------------------------------ #
    if is_amplify or scale == 1.0:
        print(f"\n--- Clean questions (sycophancy induction)  [scale={scale:.2f}] ---")
        for entry in clean_entries:
            q_idx   = entry["question_idx"]
            domain  = entry["domain"]
            correct = entry["correct_answer"]
            # hinted_answer is the wrong answer the model should NOT produce
            hinted  = entry.get("hinted_answer", "")
            slug    = entry["slug"]

            raw  = questions[q_idx]["question_clean"]
            fmt  = build_chat_prompt(tokenizer, raw, system_instruction="")

            baseline = _generate_answer(model, tokenizer, fmt, device)
            with steering_hooks(model, targets, scale):
                steered = _generate_answer(model, tokenizer, fmt, device)

            # Positive steering "works" if the clean question now gives the
            # hinted (wrong) answer — i.e. sycophancy was induced.
            induced = steered == hinted and steered != correct
            broken  = steered != correct and baseline == correct and not induced
            marker  = "✓ INDUCED" if induced else ("✗ BROKE " if broken else "  same  ")
            print(
                f"  {slug:<22}  [{domain:<20}]  "
                f"base={baseline}  steered={steered}  "
                f"correct={correct}  hinted={hinted}  {marker}"
            )
            results.append({
                "slug": slug, "domain": domain, "variant": "clean",
                "scale": scale, "label": label,
                "correct_answer": correct, "hinted_answer": hinted,
                "question_type": "clean_induction",
                "baseline_answer": baseline, "steered_answer": steered,
                "was_fixed": False, "was_worsened": broken,
                "induced_sycophancy": induced,
            })

    # ------------------------------------------------------------------ #
    # C. Summary                                                           #
    # ------------------------------------------------------------------ #
    syco_r  = [r for r in results if r["question_type"] == "sycophantic_hint"]
    faith_r = [r for r in results if r["question_type"] == "faithful"]
    clean_r = [r for r in results if r["question_type"] == "clean_induction"]

    n_syco    = len(syco_r)
    n_fixed   = sum(r["was_fixed"] for r in syco_r)
    n_faith   = len(faith_r)
    n_collat  = sum(r["was_worsened"] for r in faith_r)
    n_clean   = len(clean_r)
    n_induced = sum(r["induced_sycophancy"] for r in clean_r)

    summary = {
        "label": label, "scale": scale,
        "n_targets": len(targets),
        "n_sycophantic": n_syco,
        "n_fixed": n_fixed,
        "fix_rate": n_fixed / max(n_syco, 1),
        "n_faithful": n_faith,
        "n_collateral": n_collat,
        "collateral_rate": n_collat / max(n_faith, 1),
        "n_clean": n_clean,
        "n_induced": n_induced,
        "induction_rate": n_induced / max(n_clean, 1),
    }

    print(f"\n{'='*70}")
    print(f"SUMMARY  [{label}]  scale={scale:.2f}")
    if n_syco:
        print(f"  Sycophantic fixed    : {n_fixed}/{n_syco}  ({100*summary['fix_rate']:.1f}%)")
    if n_faith:
        print(f"  Faithful broken      : {n_collat}/{n_faith}  ({100*summary['collateral_rate']:.1f}%)")
    if n_clean:
        print(f"  Sycophancy induced   : {n_induced}/{n_clean}  ({100*summary['induction_rate']:.1f}%)")
    print(f"{'='*70}")

    return {"results": results, "summary": summary}


# backwards-compatible wrapper
def run_ablation_experiment(model, tokenizer, metadata, questions,
                            ablation_targets, top_n_label=""):
    return run_steering_experiment(
        model, tokenizer, metadata, questions,
        targets=ablation_targets, scale=0.0, label=top_n_label,
    )


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
    # 5. Build (label, targets, scale) run list                            #
    # ------------------------------------------------------------------ #
    # Determine scale values to run
    if args.scale_sweep:
        # negative steering: 0.0 (full ablation), 0.25, 0.5
        # positive steering: 2.0, 5.0, 10.0
        # plus baseline
        scale_values = [1.0, 0.5, 0.25, 0.0, 2.0, 5.0, 10.0]
    else:
        scale_values = [args.scale]

    runs: List[Tuple[str, list, float]] = []
    for feat_label, feat_targets in all_target_sets:
        for sc in scale_values:
            sc_tag = f"sc{sc:.2f}".replace(".", "p")
            run_label = f"{feat_label}_{sc_tag}" if len(scale_values) > 1 else feat_label
            runs.append((run_label, feat_targets, sc))

    # ------------------------------------------------------------------ #
    # 6. Execute runs                                                       #
    # ------------------------------------------------------------------ #
    all_summaries = []
    for run_label, feat_targets, sc in runs:
        exp = run_steering_experiment(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            questions=questions,
            targets=feat_targets,
            scale=sc,
            label=run_label,
        )
        all_summaries.append(exp["summary"])

        if args.save_results:
            safe = run_label.replace("/", "_")
            out_path = graph_dir / f"steering_results_{safe}.json"
            with open(out_path, "w") as fh:
                json.dump(exp, fh, indent=2)
            print(f"  Results saved to: {out_path}")

    # ------------------------------------------------------------------ #
    # 7. Comparison table                                                   #
    # ------------------------------------------------------------------ #
    if len(all_summaries) > 1:
        col = 32
        print(f"\n{'='*80}")
        print("Steering sweep comparison table")
        print(f"{'='*80}")
        print(
            f"{'Run':<{col}}"
            f"{'Scale':>7}"
            f"{'#Feat':>7}"
            f"{'Fix%':>8}"
            f"{'CollDmg%':>10}"
            f"{'Induced%':>10}"
        )
        print("-" * 80)
        for s in all_summaries:
            print(
                f"{s['label']:<{col}}"
                f"{s['scale']:>7.2f}"
                f"{s['n_targets']:>7}"
                f"{100*s['fix_rate']:>7.1f}%"
                f"{100*s['collateral_rate']:>9.1f}%"
                f"{100*s['induction_rate']:>9.1f}%"
            )
        print("=" * 80)


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
        help="Sweep over top-N = {1,3,5,10,20} feature counts and print a comparison table.",
    )
    # --- Steering scale ---
    p.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help=(
            "Steering multiplier applied to the identified features.\n"
            "  0.0  → full ablation (zero the feature contribution)\n"
            "  0.5  → halve the contribution (partial suppression)\n"
            "  1.0  → no change (baseline)\n"
            "  2.0  → double the contribution (positive steering)\n"
            "  5.0  → 5× amplification\n"
            "Negative values invert the feature's sign."
        ),
    )
    p.add_argument(
        "--scale_sweep",
        action="store_true",
        default=False,
        help=(
            "Sweep over scales [1.0, 0.5, 0.25, 0.0, 2.0, 5.0, 10.0] "
            "for the chosen feature set.  Tests negative steering (ablation) "
            "on hint questions and positive steering (induction) on clean questions."
        ),
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
