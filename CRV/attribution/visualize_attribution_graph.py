"""
Visualize attribution graph JSON files produced by gemma_gsm8k_circuit_tracing.py.

Each JSON file corresponds to one generated token.  The script renders:
  - Nodes coloured by type (embedding / transcoder feature / CLT feature /
    MLP reconstruction error / logit)
  - Node size scaled by |influence|
  - Edges coloured and sized by weight (blue = positive, red = negative)
  - A two-panel figure: left = full graph (spring layout),
    right = top-k most influential nodes only

Usage:
    # Visualize a single file
    python visualize_attribution_graph.py path/to/gemma_gsm8k_ex0000_tok0005.json

    # Visualize all JSON files in a directory
    python visualize_attribution_graph.py path/to/graph_files/gemma_gsm8k/

    # Show only the top-20 nodes by influence
    python visualize_attribution_graph.py graph.json --top_k 20

    # Save figures instead of showing them
    python visualize_attribution_graph.py graph.json --save_dir ./figures
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Colour / style config
# ---------------------------------------------------------------------------

NODE_COLOURS = {
    "embedding":               "#4C72B0",   # blue
    "feature":                 "#55A868",   # green
    "cross layer transcoder":  "#8172B2",   # purple
    "mlp reconstruction error":"#C44E52",   # red
    "logit":                   "#CCB974",   # gold
    "unknown":                 "#aaaaaa",
}

LAYER_ORDER = {
    "E": -1,   # embeddings come first
}


def _layer_sort_key(layer: str) -> int:
    """Convert a layer label to an integer for vertical ordering."""
    if layer == "E":
        return -1
    try:
        return int(layer)
    except ValueError:
        return 999


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def build_nx_graph(data: dict) -> tuple[nx.DiGraph, dict, dict]:
    """
    Returns
    -------
    G          : directed networkx graph
    node_meta  : {node_id: {layer, feature_type, ctx_idx, influence, activation, clerp}}
    prompt_tokens : list[str]
    """
    G = nx.DiGraph()
    node_meta: dict[str, dict] = {}

    for node in data.get("nodes", []):
        nid = node["node_id"]
        G.add_node(nid)
        node_meta[nid] = {
            "layer":        node.get("layer", "?"),
            "feature_type": node.get("feature_type", "unknown"),
            "ctx_idx":      node.get("ctx_idx", 0),
            "influence":    node.get("influence"),
            "activation":   node.get("activation"),
            "clerp":        node.get("clerp", ""),
            "feature":      node.get("feature", -1),
            "is_target":    node.get("is_target_logit", False),
        }

    for link in data.get("links", []):
        src = link["source"]
        tgt = link["target"]
        w   = link.get("weight", 0.0)
        G.add_edge(src, tgt, weight=w)

    prompt_tokens = data.get("metadata", {}).get("prompt_tokens", [])
    return G, node_meta, prompt_tokens


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _hierarchical_pos(
    G: nx.DiGraph,
    node_meta: dict,
    x_spread: float = 3.0,
    y_spread: float = 2.5,
) -> dict[str, tuple[float, float]]:
    """
    Arrange nodes in a grid: rows = layers (E at top, logit at bottom),
    columns = context position within each layer.
    """
    # Group nodes by layer
    layers: dict[int, list[str]] = {}
    for nid, m in node_meta.items():
        if nid not in G:
            continue
        lk = _layer_sort_key(m["layer"])
        layers.setdefault(lk, []).append(nid)

    sorted_layer_keys = sorted(layers.keys())
    n_layers = len(sorted_layer_keys)
    layer_y = {lk: (n_layers - i - 1) * y_spread
               for i, lk in enumerate(sorted_layer_keys)}

    pos: dict[str, tuple[float, float]] = {}
    for lk, nids in layers.items():
        # Sort by context index within layer
        nids_sorted = sorted(nids, key=lambda n: node_meta[n]["ctx_idx"])
        n = len(nids_sorted)
        for j, nid in enumerate(nids_sorted):
            x = (j - (n - 1) / 2.0) * x_spread / max(n, 1)
            pos[nid] = (x, layer_y[lk])
    return pos


# ---------------------------------------------------------------------------
# Single-graph plotting
# ---------------------------------------------------------------------------

def _node_colours(nodes: list[str], node_meta: dict) -> list[str]:
    return [NODE_COLOURS.get(node_meta[n]["feature_type"], NODE_COLOURS["unknown"])
            for n in nodes]


def _node_sizes(nodes: list[str], node_meta: dict, base: float = 300) -> list[float]:
    sizes = []
    for n in nodes:
        inf = node_meta[n].get("influence")
        if inf is None:
            sizes.append(base * 0.5)
        else:
            sizes.append(base * (0.5 + min(abs(inf), 3.0)))
    return sizes


def _edge_colours_widths(
    edges: list[tuple[str, str]],
    G: nx.DiGraph,
) -> tuple[list[str], list[float]]:
    colours, widths = [], []
    for u, v in edges:
        w = G[u][v].get("weight", 0.0)
        colours.append("#2166ac" if w >= 0 else "#d6604d")
        widths.append(max(0.3, min(abs(w) * 1.5, 5.0)))
    return colours, widths


def _node_labels(nodes: list[str], node_meta: dict, prompt_tokens: list[str]) -> dict[str, str]:
    labels = {}
    for n in nodes:
        m = node_meta[n]
        tok = prompt_tokens[m["ctx_idx"]] if m["ctx_idx"] < len(prompt_tokens) else ""
        if m["feature_type"] == "logit":
            labels[n] = f"OUT\n{m['clerp'][:20]}"
        elif m["feature_type"] == "embedding":
            labels[n] = f"E:{tok!r}"
        elif m["feature_type"] == "mlp reconstruction error":
            labels[n] = f"Err\nL{m['layer']}@{m['ctx_idx']}"
        else:
            labels[n] = f"L{m['layer']}\nF{m['feature']}\n@{m['ctx_idx']}"
    return labels


def plot_graph(
    data: dict,
    title: str = "",
    top_k: int | None = None,
    ax: plt.Axes | None = None,
    layout: str = "hierarchical",
) -> plt.Figure:
    G, node_meta, prompt_tokens = build_nx_graph(data)

    # Optionally restrict to top-k nodes by |influence|
    if top_k is not None:
        ranked = sorted(
            [n for n in G.nodes if node_meta[n]["influence"] is not None],
            key=lambda n: abs(node_meta[n]["influence"]),
            reverse=True,
        )
        # Always keep the target logit node
        keep = set(ranked[:top_k])
        for n in G.nodes:
            if node_meta[n].get("is_target"):
                keep.add(n)
        G = G.subgraph(keep).copy()

    nodes = list(G.nodes)
    if not nodes:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No nodes to display", ha="center", va="center")
        return fig

    # Layout
    if layout == "hierarchical":
        pos = _hierarchical_pos(G, node_meta)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(14, len(nodes) * 0.15), 9))
    else:
        fig = ax.get_figure()

    edges      = list(G.edges)
    ecols, ew  = _edge_colours_widths(edges, G)
    ncols      = _node_colours(nodes, node_meta)
    nsizes     = _node_sizes(nodes, node_meta)
    labels     = _node_labels(nodes, node_meta, prompt_tokens)

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges,
                           edge_color=ecols, width=ew,
                           arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.1",
                           alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                           node_color=ncols, node_size=nsizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=5, font_color="white")

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=ft)
        for ft, c in NODE_COLOURS.items()
        if ft != "unknown"
    ]
    legend_handles += [
        mpatches.Patch(color="#2166ac", label="positive edge"),
        mpatches.Patch(color="#d6604d", label="negative edge"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=7, framealpha=0.8)

    meta   = data.get("metadata", {})
    target = next((n["clerp"] for n in data.get("nodes", [])
                   if n.get("is_target_logit")), "?")
    ax.set_title(
        f"{title}\nTarget: {target}   |   "
        f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}",
        fontsize=9,
    )
    ax.axis("off")

    if standalone:
        plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary heatmap: influence by (layer, ctx_idx)
# ---------------------------------------------------------------------------

def plot_influence_heatmap(data: dict, title: str = "") -> plt.Figure:
    node_meta = {}
    for node in data.get("nodes", []):
        node_meta[node["node_id"]] = node

    # Collect (layer_int, ctx_idx, influence) triples
    entries = []
    for n in node_meta.values():
        if n.get("influence") is None:
            continue
        lk = _layer_sort_key(n.get("layer", "?"))
        entries.append((lk, n["ctx_idx"], n["influence"]))

    if not entries:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No influence data", ha="center", va="center")
        return fig

    layers  = sorted(set(e[0] for e in entries))
    ctxs    = sorted(set(e[1] for e in entries))
    l2i     = {l: i for i, l in enumerate(layers)}
    c2i     = {c: i for i, c in enumerate(ctxs)}
    mat     = np.zeros((len(layers), len(ctxs)))

    for lk, cx, inf in entries:
        mat[l2i[lk], c2i[cx]] += inf  # sum if multiple features per cell

    fig, ax = plt.subplots(figsize=(max(10, len(ctxs) * 0.25),
                                    max(5,  len(layers) * 0.4)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    plt.colorbar(im, ax=ax, label="summed influence")

    prompt_tokens = data.get("metadata", {}).get("prompt_tokens", [])
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels(
        [f"{ci}\n{prompt_tokens[ci]!r}" if ci < len(prompt_tokens) else str(ci)
         for ci in ctxs],
        fontsize=5, rotation=90,
    )
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(
        ["E" if lk == -1 else f"L{lk}" for lk in layers],
        fontsize=7,
    )
    ax.set_xlabel("Context position (token index)")
    ax.set_ylabel("Layer")
    ax.set_title(f"Influence heatmap — {title}", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to a single JSON file or a directory of JSON files")
    p.add_argument("--top_k", type=int, default=None,
                   help="Show only the top-K nodes by |influence| (default: all)")
    p.add_argument("--layout", choices=["hierarchical", "spring"], default="hierarchical",
                   help="Graph layout algorithm (default: hierarchical)")
    p.add_argument("--heatmap", action="store_true",
                   help="Also render an influence heatmap per graph")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Save figures to this directory instead of displaying them")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def process_file(path: Path, args) -> None:
    print(f"  Visualising {path.name} …")
    data  = load_graph(path)
    slug  = data.get("metadata", {}).get("slug", path.stem)
    title = slug

    fig = plot_graph(data, title=title, top_k=args.top_k, layout=args.layout)

    if args.save_dir:
        out = Path(args.save_dir) / f"{path.stem}_graph.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {out}")
    else:
        plt.show()
        plt.close(fig)

    if args.heatmap:
        fig2 = plot_influence_heatmap(data, title=title)
        if args.save_dir:
            out2 = Path(args.save_dir) / f"{path.stem}_heatmap.png"
            fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig2)
            print(f"    Saved → {out2}")
        else:
            plt.show()
            plt.close(fig2)


def main():
    args = parse_args()
    inp  = Path(args.input)

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if inp.is_dir():
        files = sorted(inp.glob("*.json"))
        if not files:
            print(f"No JSON files found in {inp}")
            return
        print(f"Found {len(files)} JSON files in {inp}")
        for f in files:
            process_file(f, args)
    elif inp.is_file():
        process_file(inp, args)
    else:
        print(f"Error: {inp} is not a file or directory")


if __name__ == "__main__":
    main()
