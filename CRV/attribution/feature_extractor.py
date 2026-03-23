"""
CRV-style structural feature extractor for attribution graphs.

Extracts a fixed-length feature vector from a saved attribution graph
dictionary (loaded from a .pt file). Feature groups follow the CRV paper
Appendix C.1 methodology, adapted for the Gemma 3 1B IT transcoder graphs
produced by gemma_custom_prompt_first_token_attribution.py.

Expected graph dict keys
------------------------
adjacency_matrix  : torch.Tensor[N, N]        pruned attribution graph
active_features   : torch.Tensor[n_active, 3] all active (layer, pos, feature_id)
selected_features : torch.Tensor[n_selected]  indices into active_features
activation_values : torch.Tensor[n_active]    activation magnitudes
logit_targets     : list[LogitTarget]          each has .token_str
logit_probabilities: torch.Tensor[n_logits]   predicted probabilities
cfg               : UnifiedConfig              has .n_layers
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

# Threshold for counting non-zero edges in the adjacency matrix.
_EDGE_THRESHOLD = 1e-4


def extract_features(graph_dict: dict, n_model_layers: int = 26) -> np.ndarray:
    """Extract a fixed-length CRV-style feature vector from an attribution graph.

    The returned array has length  6 + (8 + n_model_layers) + 10 = 24 + n_model_layers.
    For the default n_model_layers=26 (Gemma 3 1B IT) this is 50 features.

    Feature groups
    --------------
    Group 1 — Global Stats (6 values)
        n_active_features, n_selected_features, pruned_ratio,
        top_logit_prob, logit_entropy, n_logit_targets

    Group 2 — Node Influence & Activation Stats (8 + n_model_layers values)
        activation_mean, activation_max, activation_std           (selected nodes)
        feature_influence_mean, _max, _std, _total                (direct logit rows)
        residual_influence_ratio
        layer_histogram[0..n_model_layers-1]                      (per-layer counts)

    Group 3 — Topological Features (10 values)
        n_edges, graph_density,
        edge_weight_mean, edge_weight_std, edge_weight_max
        ff_subgraph_density,
        mean_out_degree, max_out_degree,
        mean_in_degree,
        fl_density

    Args:
        graph_dict:      Mapping loaded from a .pt attribution graph file.
        n_model_layers:  Number of transformer layers in the base model.

    Returns:
        1-D numpy float32 array of length 24 + n_model_layers.
    """
    feats: List[float] = []

    # ------------------------------------------------------------------
    # Unpack tensors with safe fallbacks
    # ------------------------------------------------------------------
    adj: torch.Tensor = graph_dict.get("adjacency_matrix", torch.zeros(0, 0))
    active_features: torch.Tensor = graph_dict.get("active_features", torch.zeros(0, 3))
    selected_idx: torch.Tensor = graph_dict.get("selected_features", torch.zeros(0, dtype=torch.long))
    activation_values: torch.Tensor = graph_dict.get("activation_values", torch.zeros(0))
    logit_targets = graph_dict.get("logit_targets", [])
    logit_probs: torch.Tensor = graph_dict.get("logit_probabilities", torch.zeros(0))

    # Convert everything to float32 numpy for numerical ops
    adj_np = adj.float().cpu().numpy() if adj.numel() > 0 else np.zeros((0, 0), dtype=np.float32)
    act_np = activation_values.float().cpu().numpy()

    n_active = int(active_features.shape[0]) if active_features.ndim >= 1 else 0
    n_selected = int(selected_idx.shape[0]) if selected_idx.numel() > 0 else 0
    n_logits = len(logit_targets)
    N = adj_np.shape[0]

    # Selected activation values
    if n_selected > 0 and n_active > 0 and act_np.shape[0] == n_active:
        sel_idx_np = selected_idx.long().cpu().numpy()
        # Clamp indices to valid range in case of graph format quirks
        sel_idx_np = np.clip(sel_idx_np, 0, n_active - 1)
        sel_activations = act_np[sel_idx_np]
    else:
        sel_activations = np.zeros(max(n_selected, 1), dtype=np.float32)

    # Logit probabilities
    lp_np = logit_probs.float().cpu().numpy() if logit_probs.numel() > 0 else np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Group 1 — Global Stats (6 features)
    # ------------------------------------------------------------------
    pruned_ratio = float(n_selected) / float(n_active) if n_active > 0 else 0.0
    top_logit_prob = float(lp_np[0]) if lp_np.size > 0 else 0.0

    # Entropy over logit distribution
    if lp_np.size > 0:
        p = lp_np + 1e-10
        p = p / p.sum()
        logit_entropy = float(-np.sum(p * np.log(p)))
    else:
        logit_entropy = 0.0

    feats += [
        float(n_active),
        float(n_selected),
        pruned_ratio,
        top_logit_prob,
        logit_entropy,
        float(n_logits),
    ]

    # ------------------------------------------------------------------
    # Group 2 — Node Influence & Activation Stats (8 + n_model_layers)
    # ------------------------------------------------------------------

    # Activation stats over selected features
    if sel_activations.size > 0:
        act_mean = float(np.mean(sel_activations))
        act_max = float(np.max(sel_activations))
        act_std = float(np.std(sel_activations))
    else:
        act_mean = act_max = act_std = 0.0

    # Per-feature direct logit influence: adj[-n_logits:, :n_selected]
    if n_logits > 0 and n_selected > 0 and N > 0 and N >= n_logits and N >= n_selected:
        logit_rows = adj_np[-n_logits:, :n_selected]           # [n_logits, n_selected]
        feature_influence = np.abs(logit_rows).sum(axis=0)     # [n_selected]
    else:
        feature_influence = np.zeros(max(n_selected, 1), dtype=np.float32)

    fi_mean = float(np.mean(feature_influence))
    fi_max = float(np.max(feature_influence)) if feature_influence.size > 0 else 0.0
    fi_std = float(np.std(feature_influence)) if feature_influence.size > 0 else 0.0
    fi_total = float(np.sum(feature_influence))

    # Residual influence ratio: fraction of total logit-row weight from
    # non-feature (error / embedding) nodes.
    if n_logits > 0 and N > 0 and N >= n_logits:
        logit_all_cols = adj_np[-n_logits:, :]                  # [n_logits, N]
        total_weight = np.abs(logit_all_cols).sum()
        if total_weight > 0:
            feature_weight = fi_total
            non_feature_weight = total_weight - feature_weight
            residual_ratio = float(non_feature_weight / total_weight)
        else:
            residual_ratio = 0.0
    else:
        residual_ratio = 0.0

    feats += [act_mean, act_max, act_std,
              fi_mean, fi_max, fi_std, fi_total,
              residual_ratio]

    # Layer histogram: count of selected features per layer
    layer_hist = np.zeros(n_model_layers, dtype=np.float32)
    if n_selected > 0 and n_active > 0 and active_features.ndim == 2 and active_features.shape[1] >= 1:
        sel_idx_clamped = np.clip(
            selected_idx.long().cpu().numpy(), 0, n_active - 1
        )
        sel_layers = active_features[sel_idx_clamped, 0].long().cpu().numpy()
        for layer_id in sel_layers:
            if 0 <= layer_id < n_model_layers:
                layer_hist[int(layer_id)] += 1

    feats += layer_hist.tolist()

    # ------------------------------------------------------------------
    # Group 3 — Topological Features (10 features)
    # ------------------------------------------------------------------
    if N > 0:
        abs_adj = np.abs(adj_np)
        mask = abs_adj > _EDGE_THRESHOLD
        n_edges = int(mask.sum())
        graph_density = float(n_edges) / float(N * (N - 1)) if N > 1 else 0.0

        nonzero_weights = abs_adj[mask]
        if nonzero_weights.size > 0:
            ew_mean = float(nonzero_weights.mean())
            ew_std = float(nonzero_weights.std())
            ew_max = float(nonzero_weights.max())
        else:
            ew_mean = ew_std = ew_max = 0.0

        # Feature-to-feature submatrix density
        if n_selected > 0 and N >= n_selected:
            ff_sub = abs_adj[:n_selected, :n_selected]
            ff_cells = n_selected * (n_selected - 1)
            ff_density = float((ff_sub > _EDGE_THRESHOLD).sum()) / ff_cells if ff_cells > 0 else 0.0
        else:
            ff_density = 0.0

        # Out-degree of selected feature nodes (rows 0..n_selected-1 of adj)
        if n_selected > 0 and N >= n_selected:
            out_weights = abs_adj[:n_selected, :].sum(axis=1)   # [n_selected]
            mean_out = float(out_weights.mean())
            max_out = float(out_weights.max())
        else:
            mean_out = max_out = 0.0

        # In-degree of selected feature nodes (cols 0..n_selected-1 of adj)
        if n_selected > 0 and N >= n_selected:
            in_weights = abs_adj[:, :n_selected].sum(axis=0)    # [n_selected]
            mean_in = float(in_weights.mean())
        else:
            mean_in = 0.0

        # Feature-to-logit density
        if n_logits > 0 and n_selected > 0 and N >= n_logits and N >= n_selected:
            fl_sub = abs_adj[-n_logits:, :n_selected]
            fl_cells = n_logits * n_selected
            fl_density = float((fl_sub > _EDGE_THRESHOLD).sum()) / fl_cells if fl_cells > 0 else 0.0
        else:
            fl_density = 0.0
    else:
        n_edges = 0
        graph_density = ew_mean = ew_std = ew_max = 0.0
        ff_density = 0.0
        mean_out = max_out = mean_in = 0.0
        fl_density = 0.0

    feats += [
        float(n_edges),
        graph_density,
        ew_mean,
        ew_std,
        ew_max,
        ff_density,
        mean_out,
        max_out,
        mean_in,
        fl_density,
    ]

    return np.array(feats, dtype=np.float32)


def get_feature_names(n_model_layers: int = 26) -> List[str]:
    """Return the ordered list of feature names produced by :func:`extract_features`.

    The length of the returned list equals 24 + n_model_layers.

    Args:
        n_model_layers: Number of transformer layers in the base model.

    Returns:
        List of feature name strings, in the same order as extract_features output.
    """
    names: List[str] = []

    # Group 1
    names += [
        "n_active_features",
        "n_selected_features",
        "pruned_ratio",
        "top_logit_prob",
        "logit_entropy",
        "n_logit_targets",
    ]

    # Group 2
    names += [
        "activation_mean",
        "activation_max",
        "activation_std",
        "feature_influence_mean",
        "feature_influence_max",
        "feature_influence_std",
        "feature_influence_total",
        "residual_influence_ratio",
    ]
    names += [f"layer_hist_{i}" for i in range(n_model_layers)]

    # Group 3
    names += [
        "n_edges",
        "graph_density",
        "edge_weight_mean",
        "edge_weight_std",
        "edge_weight_max",
        "ff_subgraph_density",
        "mean_out_degree",
        "max_out_degree",
        "mean_in_degree",
        "fl_density",
    ]

    return names
