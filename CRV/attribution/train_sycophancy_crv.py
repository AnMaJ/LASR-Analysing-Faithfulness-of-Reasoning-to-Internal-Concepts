"""
Train a CRV-style classifier on sycophancy attribution graphs.

Loads saved .pt graph files, extracts structural features, and trains
a Gradient Boosting Classifier following the CRV paper methodology.

Usage:
    python train_sycophancy_crv.py --graph_dir ./graphs/sycophancy
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Allow running the script directly from its own directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crv_feature_extractor import extract_features, get_feature_names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features_and_labels(
    graph_dir: Path,
    n_model_layers: int = 26,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[dict]]:
    """Load graph files, extract features, and build the design matrix.

    Args:
        graph_dir:       Directory containing metadata.json and .pt graph files.
        n_model_layers:  Number of transformer layers in the base model.

    Returns:
        X              : float32 array of shape [n_samples, n_features]
        y              : int32 array of shape [n_samples]  (0=faithful, 1=sycophantic)
        feature_names  : list of feature name strings (length n_features)
        valid_metadata : list of metadata dicts for successfully loaded entries
    """
    metadata_path = graph_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {graph_dir}. "
            "Run generate_sycophancy_graphs.py first."
        )

    with open(metadata_path) as fh:
        metadata = json.load(fh)

    feature_names = get_feature_names(n_model_layers)
    X_rows: List[np.ndarray] = []
    y_vals: List[int] = []
    valid_meta: List[dict] = []

    for entry in metadata:
        graph_path = Path(entry.get("graph_path", ""))
        if not graph_path.exists():
            # Try relative path inside graph_dir
            graph_path = graph_dir / entry.get("graph_file", "")
        if not graph_path.exists():
            warnings.warn(
                f"Graph file not found for slug={entry.get('slug')}. Skipping."
            )
            continue

        try:
            graph_dict = torch.load(str(graph_path), weights_only=False)
        except Exception as exc:
            warnings.warn(
                f"Failed to load {graph_path}: {exc}. Skipping."
            )
            continue

        try:
            feats = extract_features(graph_dict, n_model_layers=n_model_layers)
        except Exception as exc:
            warnings.warn(
                f"Feature extraction failed for {graph_path}: {exc}. Skipping."
            )
            continue

        X_rows.append(feats)
        y_vals.append(int(entry.get("is_sycophantic", 0)))
        valid_meta.append(entry)

    if len(X_rows) == 0:
        raise RuntimeError(
            f"No valid graph files could be loaded from {graph_dir}."
        )

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_vals, dtype=np.int32)
    return X, y, feature_names, valid_meta


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_fpr_at_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Return the FPR at which TPR first meets or exceeds *tpr_target*.

    Args:
        y_true:     Binary ground-truth labels.
        y_score:    Predicted positive-class probabilities.
        tpr_target: TPR threshold (default 0.95 → FPR@95).

    Returns:
        FPR value (float), or 1.0 if the target TPR is never reached.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # fpr and tpr are sorted in ascending order of threshold (TPR ascending).
    for fp, tp in zip(fpr, tpr):
        if tp >= tpr_target:
            return float(fp)
    return 1.0


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate *clf* via stratified k-fold cross-validation.

    Each fold independently fits a StandardScaler on the training partition.
    Folds where the test set contains only one class are skipped to avoid
    ill-defined metrics.

    Args:
        X:        Feature matrix [n_samples, n_features].
        y:        Binary labels [n_samples].
        clf:      Scikit-learn classifier with predict_proba support.
        n_splits: Number of folds.

    Returns:
        Dict with keys: auroc_mean, auroc_std, aupr_mean, aupr_std,
                        fpr95_mean, fpr95_std.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aurocs: List[float] = []
    auprs: List[float] = []
    fpr95s: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Skip folds with only one class in either split
        if len(np.unique(y_te)) < 2:
            warnings.warn(
                f"Fold {fold_idx}: test set has only one class — skipping."
            )
            continue
        if len(np.unique(y_tr)) < 2:
            warnings.warn(
                f"Fold {fold_idx}: train set has only one class — skipping."
            )
            continue

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Clone classifier to avoid state leakage between folds
        import copy
        fold_clf = copy.deepcopy(clf)
        fold_clf.fit(X_tr_s, y_tr)

        y_score = fold_clf.predict_proba(X_te_s)[:, 1]

        aurocs.append(roc_auc_score(y_te, y_score))
        auprs.append(average_precision_score(y_te, y_score))
        fpr95s.append(compute_fpr_at_tpr(y_te, y_score, tpr_target=0.95))

    if len(aurocs) == 0:
        warnings.warn("No valid folds were computed.")
        return {
            "auroc_mean": float("nan"), "auroc_std": float("nan"),
            "aupr_mean": float("nan"),  "aupr_std": float("nan"),
            "fpr95_mean": float("nan"), "fpr95_std": float("nan"),
        }

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std":  float(np.std(aurocs)),
        "aupr_mean":  float(np.mean(auprs)),
        "aupr_std":   float(np.std(auprs)),
        "fpr95_mean": float(np.mean(fpr95s)),
        "fpr95_std":  float(np.std(fpr95s)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results_dict: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted results table in the style of CRV Table 1.

    Args:
        results_dict: Mapping from method name to metrics dict (output of
                      :func:`evaluate_with_cv`).
    """
    col_w = 26
    header = (
        f"{'Method':<{col_w}}"
        f"{'AUROC':>14}"
        f"{'AUPR':>14}"
        f"{'FPR@95':>14}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for method_name, res in results_dict.items():
        auroc = f"{res['auroc_mean']:.3f} ± {res['auroc_std']:.3f}"
        aupr  = f"{res['aupr_mean']:.3f} ± {res['aupr_std']:.3f}"
        fpr95 = f"{res['fpr95_mean']:.3f} ± {res['fpr95_std']:.3f}"
        print(
            f"{method_name:<{col_w}}"
            f"{auroc:>14}"
            f"{aupr:>14}"
            f"{fpr95:>14}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    graph_dir = Path(args.graph_dir)

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    print(f"Loading graphs from: {graph_dir}")
    X, y, feature_names, valid_metadata = load_features_and_labels(
        graph_dir, n_model_layers=args.n_model_layers
    )
    n_samples, n_features = X.shape
    n_sycophantic = int(y.sum())
    n_faithful = n_samples - n_sycophantic

    print(f"\nDataset summary")
    print(f"  Total samples        : {n_samples}")
    print(f"  Sycophantic (label=1): {n_sycophantic} ({100*n_sycophantic/n_samples:.1f}%)")
    print(f"  Faithful    (label=0): {n_faithful} ({100*n_faithful/n_samples:.1f}%)")
    print(f"  Feature vector size  : {n_features}")

    # Per-domain breakdown
    domain_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "sycophantic": 0})
    for entry, label in zip(valid_metadata, y):
        d = entry.get("domain", "unknown")
        domain_counts[d]["total"] += 1
        domain_counts[d]["sycophantic"] += int(label)

    print(f"\nPer-domain breakdown")
    print(f"  {'Domain':<25} {'Total':>7} {'Syco':>7} {'Pct':>7}")
    print(f"  {'-'*48}")
    for domain, counts in sorted(domain_counts.items()):
        t = counts["total"]
        s = counts["sycophantic"]
        pct = 100 * s / t if t > 0 else 0.0
        print(f"  {domain:<25} {t:>7} {s:>7} {pct:>6.1f}%")

    if n_sycophantic == 0 or n_faithful == 0:
        print(
            "\nWARNING: Dataset has only one class. "
            "Cannot train a meaningful classifier. Exiting."
        )
        return

    # ------------------------------------------------------------------ #
    # 2. Cross-validation with three classifiers                          #
    # ------------------------------------------------------------------ #
    classifiers = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }

    print(f"\nRunning {args.n_folds}-fold stratified cross-validation …")
    results: Dict[str, Dict[str, float]] = {}
    for name, clf in classifiers.items():
        print(f"  Evaluating {name} …")
        results[name] = evaluate_with_cv(X, y, clf, n_splits=args.n_folds)

    print_results_table(results)

    # ------------------------------------------------------------------ #
    # 3. Final GBC trained on all data + feature importances              #
    # ------------------------------------------------------------------ #
    print("\nTraining final GradientBoostingClassifier on full dataset …")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    final_clf.fit(X_scaled, y)

    importances = final_clf.feature_importances_
    top_k = min(15, n_features)
    top_idx = np.argsort(importances)[::-1][:top_k]

    print(f"\nTop {top_k} feature importances (GradientBoosting, full data)")
    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12}")
    print(f"  {'-'*55}")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank:<6} {feature_names[idx]:<35} {importances[idx]:>12.5f}")

    # ------------------------------------------------------------------ #
    # 4. Per-domain answer breakdown                                       #
    # ------------------------------------------------------------------ #
    print(f"\nPer-domain model-answer breakdown (clean vs hint)")
    domain_answers: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"clean_answers": [], "hint_answers": [], "clean_correct": 0,
                 "hint_correct": 0, "hint_sycophantic": 0, "hint_total": 0}
    )
    for entry in valid_metadata:
        d = entry.get("domain", "unknown")
        v = entry.get("variant", "")
        ans = entry.get("model_answer", "?")
        correct = entry.get("correct_answer", "?")
        is_syco = entry.get("is_sycophantic", 0)
        if v == "clean":
            domain_answers[d]["clean_answers"].append(ans)
            if ans == correct:
                domain_answers[d]["clean_correct"] += 1
        elif v == "hint":
            domain_answers[d]["hint_answers"].append(ans)
            domain_answers[d]["hint_total"] += 1
            if ans == correct:
                domain_answers[d]["hint_correct"] += 1
            if is_syco:
                domain_answers[d]["hint_sycophantic"] += 1

    header2 = (
        f"  {'Domain':<25} "
        f"{'CleanAcc':>10} "
        f"{'HintAcc':>10} "
        f"{'SycoPct':>10}"
    )
    print(header2)
    print(f"  {'-'*60}")
    for domain, da in sorted(domain_answers.items()):
        n_clean = len(da["clean_answers"])
        n_hint = da["hint_total"]
        clean_acc = (
            100 * da["clean_correct"] / n_clean if n_clean > 0 else float("nan")
        )
        hint_acc = (
            100 * da["hint_correct"] / n_hint if n_hint > 0 else float("nan")
        )
        syco_pct = (
            100 * da["hint_sycophantic"] / n_hint if n_hint > 0 else float("nan")
        )
        print(
            f"  {domain:<25} "
            f"{clean_acc:>9.1f}% "
            f"{hint_acc:>9.1f}% "
            f"{syco_pct:>9.1f}%"
        )

    # ------------------------------------------------------------------ #
    # 5. Optionally save model                                             #
    # ------------------------------------------------------------------ #
    if args.save_model:
        save_path = graph_dir / "sycophancy_crv_model.pkl"
        with open(save_path, "wb") as fh:
            pickle.dump({"classifier": final_clf, "scaler": final_scaler,
                         "feature_names": feature_names}, fh)
        print(f"\nModel saved to: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a CRV-style sycophancy classifier on attribution graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--graph_dir",
        type=str,
        default="./graphs/sycophancy",
        help="Directory containing metadata.json and .pt graph files.",
    )
    p.add_argument(
        "--n_model_layers",
        type=int,
        default=26,
        help="Number of transformer layers in the base model (Gemma 3 1B = 26).",
    )
    p.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    p.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save the final GBC + scaler to a .pkl file in graph_dir.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
