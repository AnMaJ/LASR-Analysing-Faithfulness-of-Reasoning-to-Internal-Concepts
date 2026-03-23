"""
Identify transcoder features most associated with sycophantic behaviour.

After running:
    python generate_sycophancy_graphs.py --graph_dir ./graphs/sycophancy
    python train_sycophancy_crv.py   --graph_dir ./graphs/sycophancy --save_model

this script:
  1. Loads every .pt attribution graph from graph_dir.
  2. For each graph, extracts the top-k (layer, feature_id) nodes by direct
     logit influence (same ranking used by print_top_feature_examples).
  3. Tallies how often each (layer, feature_id) appears in sycophantic graphs
     vs. faithful graphs.
  4. Ranks features by enrichment score =
         freq_sycophantic / (freq_faithful + smoothing)
  5. Looks up human-readable descriptions from the mwhanna feature store
     (optional, requires network access).
  6. Prints the top-N enriched features and their activation statistics,
     ready for ablation experiments.

For a single sample prompt the --sample flag lets you run attribution on the
fly and print its top features directly (wraps generate+attribute internally
so that the answer-letter position is attributed correctly).

Usage
-----
# Analyse saved graphs
python analyze_sycophancy_transcoder_features.py \\
    --graph_dir ./graphs/sycophancy \\
    --top_k 20 \\
    --top_per_graph 50

# Analyse a single sycophancy prompt (attribute on the fly)
python analyze_sycophancy_transcoder_features.py \\
    --sample \\
    --graph_dir ./graphs/sycophancy \\
    --question_idx 0 \\
    --top_k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mwhanna_feature_store import MwhannaFeatureStore


# ---------------------------------------------------------------------------
# Feature extraction from a saved .pt graph
# ---------------------------------------------------------------------------

def extract_top_transcoder_features(
    graph_dict: dict,
    top_per_graph: int = 50,
) -> List[Tuple[int, int, float]]:
    """Return the top transcoder features sorted by direct-logit influence.

    Args:
        graph_dict:      dict loaded from a .pt attribution graph file.
        top_per_graph:   how many (layer, feature_id) entries to return.

    Returns:
        List of (layer, feature_id, influence_score) tuples, sorted descending.
    """
    active = graph_dict.get("active_features")    # [n_active, 3]
    selected = graph_dict.get("selected_features") # [n_selected]
    adj = graph_dict.get("adjacency_matrix")       # [N, N]
    logit_targets = graph_dict.get("logit_targets", [])

    if active is None or selected is None or adj is None:
        return []

    n_logits = len(logit_targets)
    n_selected = int(selected.shape[0]) if selected.numel() > 0 else 0
    N = adj.shape[0]

    if n_logits == 0 or n_selected == 0 or N < n_logits + n_selected:
        return []

    # Rows of selected features in active_features
    sel_idx = selected.long().cpu()
    sel_idx = sel_idx.clamp(0, int(active.shape[0]) - 1)
    feature_rows = active[sel_idx]  # [n_selected, 3] — (layer, pos, feature_id)

    # Direct logit influence = sum of abs edge weights from logit rows
    logit_rows = adj[-n_logits:, :n_selected].abs()  # [n_logits, n_selected]
    influence = logit_rows.sum(dim=0)                 # [n_selected]

    top_k = min(top_per_graph, n_selected)
    top_idx = influence.topk(top_k).indices.tolist()

    results: List[Tuple[int, int, float]] = []
    for idx in top_idx:
        row = feature_rows[idx]
        layer = int(row[0].item())
        feature_id = int(row[2].item())
        score = float(influence[idx].item())
        results.append((layer, feature_id, score))

    return results


# ---------------------------------------------------------------------------
# Cross-graph frequency analysis
# ---------------------------------------------------------------------------

def load_all_features(
    graph_dir: Path,
    top_per_graph: int = 50,
) -> Tuple[
    Dict[Tuple[int, int], List[float]],  # sycophantic: (l,f) -> list of scores
    Dict[Tuple[int, int], List[float]],  # faithful: (l,f) -> list of scores
    int,  # n_syco
    int,  # n_faithful
]:
    """Load graphs and collect per-class feature frequency maps.

    Args:
        graph_dir:      Directory containing metadata.json and .pt files.
        top_per_graph:  Top features to extract per graph.

    Returns:
        syco_map, faith_map, n_syco, n_faithful
    """
    meta_path = graph_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {graph_dir}. "
            "Run generate_sycophancy_graphs.py first."
        )

    with open(meta_path) as fh:
        metadata = json.load(fh)

    syco_map: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    faith_map: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    n_syco = 0
    n_faithful = 0

    for entry in metadata:
        graph_path = Path(entry.get("graph_path", ""))
        if not graph_path.exists():
            graph_path = graph_dir / entry.get("graph_file", "")
        if not graph_path.exists():
            warnings.warn(f"Graph file missing for slug={entry.get('slug')}. Skipping.")
            continue

        try:
            gd = torch.load(str(graph_path), weights_only=False)
        except Exception as exc:
            warnings.warn(f"Load error for {graph_path}: {exc}. Skipping.")
            continue

        feats = extract_top_transcoder_features(gd, top_per_graph=top_per_graph)
        is_syco = bool(entry.get("is_sycophantic", 0))

        if is_syco:
            n_syco += 1
            target = syco_map
        else:
            n_faithful += 1
            target = faith_map

        for layer, feat_id, score in feats:
            target[(layer, feat_id)].append(score)

    return syco_map, faith_map, n_syco, n_faithful


def rank_by_enrichment(
    syco_map: Dict[Tuple[int, int], List[float]],
    faith_map: Dict[Tuple[int, int], List[float]],
    n_syco: int,
    n_faithful: int,
    smoothing: float = 0.5,
    min_syco_freq: int = 1,
) -> List[dict]:
    """Rank (layer, feature_id) pairs by enrichment in sycophantic graphs.

    Enrichment = (freq_syco / n_syco) / ((freq_faith / n_faithful) + smoothing/n_faithful)

    Args:
        syco_map:       Mapping from (layer, feat_id) to list of influence scores
                        from sycophantic graphs.
        faith_map:      Same for faithful graphs.
        n_syco:         Total number of sycophantic graphs.
        n_faithful:     Total number of faithful graphs.
        smoothing:      Laplace-style smoothing added to the faithful frequency.
        min_syco_freq:  Minimum number of sycophantic graphs a feature must
                        appear in to be included.

    Returns:
        List of dicts sorted descending by enrichment, each with keys:
        layer, feature_id, enrichment, freq_syco, freq_faithful,
        mean_score_syco, mean_score_faithful.
    """
    all_keys = set(syco_map.keys()) | set(faith_map.keys())
    rows = []
    for key in all_keys:
        layer, feat_id = key
        s_scores = syco_map.get(key, [])
        f_scores = faith_map.get(key, [])

        freq_s = len(s_scores)
        freq_f = len(f_scores)

        if freq_s < min_syco_freq:
            continue

        rate_s = freq_s / max(n_syco, 1)
        rate_f = freq_f / max(n_faithful, 1)
        enrichment = rate_s / (rate_f + smoothing / max(n_faithful, 1))

        rows.append({
            "layer": layer,
            "feature_id": feat_id,
            "enrichment": enrichment,
            "freq_syco": freq_s,
            "freq_faithful": freq_f,
            "rate_syco": rate_s,
            "rate_faithful": rate_f,
            "mean_score_syco": float(np.mean(s_scores)) if s_scores else 0.0,
            "mean_score_faithful": float(np.mean(f_scores)) if f_scores else 0.0,
        })

    rows.sort(key=lambda r: r["enrichment"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Single-sample attribution (on-the-fly)
# ---------------------------------------------------------------------------

def run_single_sample(
    question_idx: int,
    variant: str,
    graph_dir: Path,
    top_k: int,
    dtype: torch.dtype,
    use_store: bool,
) -> None:
    """Attribute a single sycophancy question and print its top transcoder features.

    Args:
        question_idx: Index into sycophancy_dataset.QUESTIONS.
        variant:      'clean' or 'hint'.
        graph_dir:    Where to save the .pt graph file.
        top_k:        Number of top features to display.
        dtype:        Floating-point dtype for the model.
        use_store:    Whether to look up feature descriptions.
    """
    from sycophancy_dataset import QUESTIONS
    from gemma_custom_prompt_first_token_attribution import (
        load_model_lazy,
        build_chat_prompt,
        attribute_first_token,
    )
    from generate_sycophancy_graphs import find_answer_prefix

    q = QUESTIONS[question_idx]
    raw_text = q[f"question_{variant}"]
    correct = q["correct_answer"]
    hinted = q["hinted_answer"]
    domain = q["domain"]

    model, tokenizer = load_model_lazy(dtype=dtype)
    device = next(model.parameters()).device

    formatted = build_chat_prompt(tokenizer, raw_text, system_instruction="")

    print(f"\nFinding answer prefix for q{question_idx} [{domain}] variant={variant} …")
    prefix, answer_letter = find_answer_prefix(
        model, tokenizer, formatted, device=device, max_new_tokens=80
    )

    if prefix is None:
        print("No answer letter found within max_new_tokens. Aborting.")
        return

    is_syco = (variant == "hint" and answer_letter == hinted and answer_letter != correct)
    print(f"  Model answer : {answer_letter!r}")
    print(f"  Correct      : {correct!r}   Hinted: {hinted!r}")
    print(f"  Sycophantic  : {is_syco}")

    slug = f"q{question_idx:02d}_{variant}_sample"
    vis_dir = graph_dir / "vis"

    graph, graph_path = attribute_first_token(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prefix,
        graph_dir=graph_dir,
        graph_file_dir=vis_dir,
        slug=slug,
        verbose=True,
    )

    gd = torch.load(str(graph_path), weights_only=False)
    top_feats = extract_top_transcoder_features(gd, top_per_graph=top_k)

    store: Optional[MwhannaFeatureStore] = MwhannaFeatureStore() if use_store else None

    print(f"\n{'='*70}")
    print(f"Top {top_k} transcoder features by direct-logit influence")
    print(f"  Question {question_idx} [{domain}]  variant={variant}  answer={answer_letter}")
    print(f"{'='*70}")
    for rank, (layer, feat_id, score) in enumerate(top_feats, 1):
        print(f"\n[{rank:>2}]  L{layer}/F{feat_id}   influence={score:.4f}")
        print(f"       Ablation hook target: layer={layer}, feature_id={feat_id}")
        if store is not None:
            info = store.fetch(layer=layer, feature_id=feat_id)
            if info is not None:
                print("       " + store.format_for_print(info).replace("\n", "\n       "))
            else:
                print("       (not found in feature store)")
    print(f"\nGraph saved to: {graph_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    graph_dir = Path(args.graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # Mode A: attribute a single sample on the fly
    # ------------------------------------------------------------------
    if args.sample:
        run_single_sample(
            question_idx=args.question_idx,
            variant=args.variant,
            graph_dir=graph_dir,
            top_k=args.top_k,
            dtype=dtype,
            use_store=not args.no_store,
        )
        return

    # ------------------------------------------------------------------
    # Mode B: analyse saved graphs from the full dataset run
    # ------------------------------------------------------------------
    print(f"Loading saved graphs from: {graph_dir}")
    syco_map, faith_map, n_syco, n_faithful = load_all_features(
        graph_dir, top_per_graph=args.top_per_graph
    )

    print(f"  Sycophantic graphs : {n_syco}")
    print(f"  Faithful graphs    : {n_faithful}")
    print(f"  Unique syco features seen: {len(syco_map)}")

    if n_syco == 0:
        print("\nNo sycophantic graphs found — cannot compute enrichment. Exiting.")
        return

    ranked = rank_by_enrichment(
        syco_map, faith_map, n_syco, n_faithful,
        smoothing=args.smoothing,
        min_syco_freq=args.min_syco_freq,
    )

    top = ranked[: args.top_k]

    store: Optional[MwhannaFeatureStore] = (
        MwhannaFeatureStore() if not args.no_store else None
    )

    print(f"\n{'='*70}")
    print(f"Top {len(top)} transcoder features enriched in sycophantic graphs")
    print(f"  enrichment = rate_syco / (rate_faithful + laplace_smoothing)")
    print(f"{'='*70}")
    print(
        f"{'Rank':<5} {'Layer':<7} {'FeatID':<10} {'Enrich':>8} "
        f"{'FreqS':>7} {'FreqF':>7} {'MeanScoreS':>12}"
    )
    print("-" * 70)

    for rank, row in enumerate(top, 1):
        print(
            f"{rank:<5} {row['layer']:<7} {row['feature_id']:<10} "
            f"{row['enrichment']:>8.2f} {row['freq_syco']:>7} "
            f"{row['freq_faithful']:>7} {row['mean_score_syco']:>12.4f}"
        )
        if store is not None:
            info = store.fetch(layer=row["layer"], feature_id=row["feature_id"])
            if info is not None:
                desc = store.format_for_print(info)
                for line in desc.splitlines():
                    print(f"       {line}")

    # Save ranked list as JSON for ablation scripts
    out_path = graph_dir / "sycophancy_enriched_features.json"
    with open(out_path, "w") as fh:
        json.dump(ranked, fh, indent=2)
    print(f"\nFull ranked list saved to: {out_path}")
    print("\nAblation recipe — clamp these features to zero with a forward hook:")
    for row in top[:5]:
        print(f"  layer={row['layer']:>2}, feature_id={row['feature_id']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Identify transcoder features enriched in sycophantic attribution graphs. "
            "Use --sample to run attribution on a single question on the fly."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--graph_dir",
        type=str,
        default="./graphs/sycophancy",
        help="Directory containing metadata.json and .pt graph files.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top enriched features to display.",
    )
    p.add_argument(
        "--top_per_graph",
        type=int,
        default=50,
        help="Top features extracted per graph for the frequency tally.",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Laplace smoothing in the enrichment denominator.",
    )
    p.add_argument(
        "--min_syco_freq",
        type=int,
        default=1,
        help="Minimum sycophantic-graph count for a feature to be ranked.",
    )
    p.add_argument(
        "--no_store",
        action="store_true",
        default=False,
        help="Skip mwhanna feature-store lookups (no network needed).",
    )

    # --- single-sample mode ---
    p.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Attribute a single question on the fly instead of analysing saved graphs.",
    )
    p.add_argument(
        "--question_idx",
        type=int,
        default=0,
        help="Index into sycophancy_dataset.QUESTIONS for --sample mode.",
    )
    p.add_argument(
        "--variant",
        type=str,
        default="hint",
        choices=["clean", "hint"],
        help="Question variant to run in --sample mode.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for --sample mode.",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
