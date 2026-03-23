"""
Attribution graph for the first generated token of a custom prompt.

Uses Gemma 3 1B IT with Gemma Scope 2 transcoders loaded via the lazy-decoder
strategy (mwhanna HF mirror with circuit-tracer cache). Only a single attribution
pass is run — for the first token the model would generate given the prompt.

Usage
-----
# First run: downloads + converts transcoders into ~/.cache/circuit_tracer
python gemma_custom_prompt_first_token_attribution.py

# Subsequent runs: reuses the local cache
python gemma_custom_prompt_first_token_attribution.py

Flags
-----
--custom_prompt   Raw user-turn text (wrapped in Gemma 3 IT chat template).
--cache_dir       Override the default circuit-tracer cache directory.
--graph_dir       Where to write the .pt attribution graph file.
--graph_file_dir  Where to write the visualiser JSON files.
--port            Local port for the graph visualiser server.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from circuit_tracer.utils.caching import (
    is_cached,
    save_transcoders_to_cache,
    load_transcoders_from_cache,
)
import circuit_tracer.graph as _ct_graph
from mwhanna_feature_store import MwhannaFeatureStore


def _patch_compute_influence() -> None:
    """Monkey-patch circuit_tracer.graph.compute_influence to not raise on
    non-convergence.

    The default implementation raises RuntimeError after 1000 iterations of
    power iteration.  This happens reliably when there is only one salient
    logit (the model is extremely confident), causing the influence matrix to
    be rank-1 and the iteration to oscillate rather than converge.  The patch
    returns the last iterate instead of raising, which is accurate enough for
    visualisation purposes.
    """
    import inspect

    orig_fn = _ct_graph.compute_influence
    src = inspect.getsource(orig_fn)

    # Only patch if the function still raises (guard against future versions
    # that may already handle this gracefully).
    if "raise RuntimeError" not in src:
        return

    def _patched_compute_influence(adjacency_matrix, logit_weights):
        try:
            return orig_fn(adjacency_matrix, logit_weights)
        except RuntimeError as exc:
            if "converge" not in str(exc):
                raise
            # Fall back: return uniform influence (all nodes treated equally).
            print(
                f"Warning: influence computation did not converge ({exc}). "
                "Using uniform node influence for pruning — all nodes will be kept."
            )
            n = adjacency_matrix.shape[0]
            return torch.ones(n, dtype=adjacency_matrix.dtype,
                              device=adjacency_matrix.device)

    _ct_graph.compute_influence = _patched_compute_influence


_patch_compute_influence()


def _patch_compute_skip() -> None:
    """Monkey-patch SingleLayerTranscoder.compute_skip to fix a device mismatch.

    When transcoders are loaded with lazy_decoder=True, W_skip stays on CPU even
    when the rest of the model is on GPU.  This patch moves W_skip to the input
    device on the first call (in-place, so subsequent calls are free).
    """
    try:
        from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
    except ImportError:
        return

    orig_fn = SingleLayerTranscoder.compute_skip

    def _patched_compute_skip(self, inputs):
        if (hasattr(self, "W_skip")
                and isinstance(self.W_skip, torch.Tensor)
                and self.W_skip.device != inputs.device):
            self.W_skip.data = self.W_skip.data.to(inputs.device)
        return orig_fn(self, inputs)

    SingleLayerTranscoder.compute_skip = _patched_compute_skip


_patch_compute_skip()

# ---------------------------------------------------------------------------
# Constants (shared with gemma_gsm8k_circuit_tracing.py)
# ---------------------------------------------------------------------------

GEMMA_MODEL_ID = "google/gemma-3-1b-it"

# mwhanna mirror ships a config.yaml required by the circuit-tracer caching API.
# Format: "owner/repo/subfolder"
MWHANNA_HF_REF = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_262k_l0_small_affine"

# Default custom prompt — kept deliberately ambiguous so the model spreads
# probability across several tokens, which makes the attribution graph richer.
DEFAULT_CUSTOM_PROMPT = "The capital of a country which had the first fight for democracy is "

DEFAULT_SYSTEM_INSTRUCTION = ""

# Scan passed to create_graph_files so it can resolve feature labels.
NEURONPEDIA_CT_ID = "google/gemma-scope-2-1b-it//transcoder_all/width_262k_l0_small_affine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prompt_to_slug(prompt: str, max_words: int = 6) -> str:
    """Turn the first few words of *prompt* into a URL-safe slug.

    A 6-char hash suffix is appended so that two prompts whose first words
    collide still get distinct slugs (and therefore distinct graph directories).
    """
    words = re.sub(r"[^a-zA-Z0-9 ]", "", prompt).split()
    base = "-".join(w.lower() for w in words[:max_words])
    suffix = hashlib.md5(prompt.encode()).hexdigest()[:6]
    return f"{base}-{suffix}" if base else suffix


# Module-level feature store — initialised lazily on first use.
_feature_store: MwhannaFeatureStore | None = None


def _get_store() -> MwhannaFeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = MwhannaFeatureStore()
    return _feature_store


def _get_feature_list(graph):
    """Return the list of (layer, position, feature_idx) tuples from the graph.

    Tries the attribute names used by different circuit-tracer versions.
    """
    for attr in ("selected_features", "active_features", "real_features",
                 "features", "feature_nodes"):
        val = getattr(graph, attr, None)
        if val is not None and hasattr(val, "__len__") and len(val) > 0:
            try:
                first = val[0]
                if hasattr(first, "__len__") and len(first) == 3:
                    return list(val)
            except (IndexError, TypeError):
                pass
    attrs = [a for a in dir(graph) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot find feature list on Graph object. "
        f"Available attributes: {attrs}"
    )


def print_top_feature_examples(graph, top_n: int = 5) -> None:
    """Print mwhanna feature store examples for the *top_n* most influential features.

    Graph layout (confirmed from .pt dict inspection):
      active_features  : Tensor[n_active, 3]  — all active (layer, pos, feature_id)
      selected_features: Tensor[n_selected]   — indices into active_features;
                         these map 1-to-1 to the first n_selected columns of
                         the adjacency matrix.
    """
    active = getattr(graph, "active_features", None)   # [n_active, 3]
    selected = getattr(graph, "selected_features", None)  # [n_selected]

    if active is not None and selected is not None:
        # Resolve selected indices → (layer, pos, feature_id) rows
        feature_rows = active[selected]   # [n_selected, 3]
        n_features = len(feature_rows)
    else:
        # Fallback for older graph formats
        feature_rows_list = _get_feature_list(graph)
        feature_rows = feature_rows_list
        n_features = len(feature_rows)

    n_logits = len(graph.logit_targets)

    # Last n_logits rows of the adjacency matrix are logit nodes;
    # first n_features columns are the selected feature nodes.
    logit_rows = graph.adjacency_matrix[-n_logits:, :n_features]
    feature_importance = logit_rows.abs().sum(dim=0)  # [n_features]

    top_k = min(top_n, n_features)
    top_indices = feature_importance.topk(top_k).indices.tolist()

    store = _get_store()

    print(f"\n{'='*60}")
    print(f"Top {top_k} features by direct logit influence")
    print(f"{'='*60}")

    for rank, feat_idx in enumerate(top_indices, start=1):
        entry = feature_rows[feat_idx]
        if isinstance(entry, torch.Tensor):
            layer, position, feature_id = int(entry[0]), int(entry[1]), int(entry[2])
        else:
            layer, position, feature_id = entry

        importance = feature_importance[feat_idx].item()
        print(f"\n[{rank}] L{layer}/F{feature_id}  pos={position}  "
              f"direct-logit-influence={importance:.3f}")

        info = store.fetch(layer=layer, feature_id=feature_id)
        if info is None:
            print("     (feature not found in mwhanna store)")
            continue

        print(store.format_for_print(info))


# ---------------------------------------------------------------------------
# Model loading — lazy-decoder path only
# ---------------------------------------------------------------------------

def load_model_lazy(
    model_id: str = GEMMA_MODEL_ID,
    transcoder_hf_ref: str = MWHANNA_HF_REF,
    cache_dir: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ReplacementModel, AutoTokenizer]:
    """Load Gemma 3 1B IT with lazy W_dec transcoders from the mwhanna mirror.

    On the first call, ``save_transcoders_to_cache`` downloads and converts the
    transcoders from ``transcoder_hf_ref`` into circuit-tracer format and stores
    them in ``cache_dir`` (default: ``~/.cache/circuit_tracer``).  On subsequent
    calls the conversion is skipped and the cache is loaded directly.

    W_enc is kept in GPU memory (accessed every forward pass); W_dec is read from
    disk on demand (``lazy_decoder=True``), reducing peak VRAM by ~1 GB per layer.

    Args:
        model_id: HuggingFace ID of the base Gemma 3 1B IT model.
        transcoder_hf_ref: HF reference for the mwhanna transcoder mirror.
        cache_dir: Local directory for the circuit-tracer cache.
        dtype: Floating-point dtype for both the model and transcoders.

    Returns:
        Tuple of (ReplacementModel, AutoTokenizer).
    """
    print(f"Lazy-load mode: transcoder ref = {transcoder_hf_ref}")

    if not is_cached(transcoder_hf_ref, cache_dir=cache_dir):
        print(
            "Cache not found — downloading and converting transcoders …\n"
            "  (one-time operation; subsequent runs reuse the cache)"
        )
        save_transcoders_to_cache(
            transcoder_hf_ref,
            cache_dir=cache_dir,
            sequential=True,        # one layer at a time — keeps peak RAM low
            device=torch.device("cpu"),
            dtype=dtype,
            delete_hf_cache=True,   # free HF download cache after each layer
        )
        print("Cache populated.")
    else:
        print("Cache found — skipping download.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transcoders, _ = load_transcoders_from_cache(
        transcoder_hf_ref,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
        lazy_encoder=False,     # W_enc needed every forward pass → keep in VRAM
        lazy_decoder=True,      # W_dec read from disk per feature → saves VRAM
    )

    # load_transcoders_from_cache with lazy_decoder=True does not move W_skip
    # (and other nn.Parameters like b_enc, b_dec) to the target device.
    # W_dec is lazy (stored as transcoder_path, not a Parameter) so .to() is safe.
    for tc in transcoders.transcoders:
        tc.to(device)

    print(f"Loading base model: {model_id}")
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=model_id,
        transcoders=transcoders,
        dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model and tokenizer ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_chat_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    system_instruction: str,
) -> str:
    """Wrap *question* in the Gemma 3 IT chat template.

    Args:
        tokenizer: Gemma 3 tokenizer with a built-in chat template.
        question: Raw user-turn content.
        system_instruction: Instruction prepended to the question.

    Returns:
        Fully formatted prompt string, including the model-turn opener.
    """
    content = f"{system_instruction}\n\n{question}" if system_instruction else question
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# First-token attribution
# ---------------------------------------------------------------------------

def attribute_first_token(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    graph_dir: Path,
    graph_file_dir: Path,
    slug: str = "custom-prompt-first-token",
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 64,
    max_feature_nodes: int = 4096,
    offload: str | None = "cpu",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    neuronpedia_id: str | None = NEURONPEDIA_CT_ID,
    verbose: bool = True,
) -> Path:
    """Run attribution for the first token the model generates from *prompt_text*.

    ``attribute()`` is called with the raw prompt (no generated tokens appended),
    so its logit targets correspond to the distribution over the *first* generated
    token — exactly what we want.

    The BOS token is stripped from the prompt before passing it to circuit-tracer
    because TransformerLens / circuit-tracer prepend BOS automatically.

    Args:
        model: ReplacementModel with lazy-loaded Gemma Scope 2 transcoders.
        tokenizer: Gemma 3 tokenizer.
        prompt_text: Fully formatted chat prompt string.
        graph_dir: Directory to save the .pt attribution graph file.
        graph_file_dir: Directory to save visualiser JSON files.
        slug: Identifier used by the visualiser frontend.
        max_n_logits: Maximum number of output logits to attribute from.
        desired_logit_prob: Probability mass covered by attributed logits.
        batch_size: Internal batch size for the attribution pass.
        max_feature_nodes: Maximum transcoder feature nodes in the graph.
        offload: Where to offload intermediates ('disk', 'cpu', or None).
        node_threshold: Cumulative-influence threshold for pruning nodes.
        edge_threshold: Cumulative-influence threshold for pruning edges.
        verbose: Show tqdm progress bar inside circuit-tracer.

    Returns:
        Path to the saved .pt attribution graph file.
    """
    # Strip leading BOS so circuit-tracer doesn't see a double BOS.
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    bos_id = tokenizer.bos_token_id
    if len(input_ids) > 0 and input_ids[0].item() == bos_id:
        input_ids = input_ids[1:]
    prompt_no_bos = tokenizer.decode(input_ids, skip_special_tokens=False)

    print("\nRunning attribution for the next generated token …")
    print(f"  Prompt (first 200 chars): {prompt_no_bos[:200]!r}")

    graph = attribute(
        prompt=prompt_no_bos,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
    )

    logit_tokens = [lt.token_str for lt in graph.logit_targets]
    print(f"  Attributed token(s): {logit_tokens}  →  top predicted: {logit_tokens[0]!r}")
    print(f"  graph.scan = {graph.scan!r}")

    # Override scan so create_graph_files fetches the right Neuronpedia labels.
    # The mwhanna transcoders are a repackaging of the original Gemma Scope 2
    # 1B IT transcoders; the feature indices are identical, so the Neuronpedia
    # scan for the originals applies here.
    if neuronpedia_id is not None:
        graph.scan = neuronpedia_id

    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"{slug}.pt"
    graph.to_pt(graph_path)
    print(f"Graph saved to: {graph_path}")

    graph_file_dir.mkdir(parents=True, exist_ok=True)
    create_graph_files(
        graph_or_path=graph_path,
        slug=slug,
        output_path=graph_file_dir,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    print(f"Visualiser files written to: {graph_file_dir}")

    return graph, graph_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attribution graph for the first generated token of a custom prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Prompt
    p.add_argument("--custom_prompt", type=str, default=DEFAULT_CUSTOM_PROMPT,
                   help="Raw user-turn content (wrapped in Gemma 3 IT chat template).")
    p.add_argument("--system_instruction", type=str, default=DEFAULT_SYSTEM_INSTRUCTION,
                   help="System instruction prepended to the prompt.")

    # Model / caching
    p.add_argument("--model_id", type=str, default=GEMMA_MODEL_ID)
    p.add_argument("--transcoder_hf_ref", type=str, default=MWHANNA_HF_REF,
                   help="HF reference for the mwhanna transcoder mirror.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Local circuit-tracer cache directory. "
                        "Defaults to ~/.cache/circuit_tracer.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # Attribution
    p.add_argument("--num_tokens", type=int, default=1,
                   help="Number of generated tokens to attribute (one graph per token).")
    p.add_argument("--max_n_logits", type=int, default=10)
    p.add_argument("--desired_logit_prob", type=float, default=0.95)
    p.add_argument("--max_feature_nodes", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--offload", type=str, default="cpu",
                   choices=["disk", "cpu", "none"],
                   help="Where to offload attribution intermediates.")
    p.add_argument("--node_threshold", type=float, default=0.8)
    p.add_argument("--edge_threshold", type=float, default=0.98)
    p.add_argument("--no_verbose", dest="verbose", action="store_false", default=True)

    # Output / visualisation
    p.add_argument("--graph_dir", type=str, default="./graphs/custom_prompt",
                   help="Directory for the .pt graph file.")
    p.add_argument("--graph_file_dir", type=str, default="./graph_files/custom_prompt",
                   help="Directory for visualiser JSON files.")
    p.add_argument("--slug", type=str, default=None,
                   help="Visualiser slug (graph name). Auto-derived from the prompt if omitted.")
    p.add_argument("--port", type=int, default=8046,
                   help="Port for the local visualiser server.")
    p.add_argument("--no_serve", action="store_true", default=False,
                   help="Skip launching the visualiser server after attribution.")

    # Feature examples
    p.add_argument("--top_features", type=int, default=5,
                   help="Number of top features to look up in the mwhanna feature store after attribution.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    offload = None if args.offload == "none" else args.offload
    slug = args.slug if args.slug else prompt_to_slug(args.custom_prompt)

    # ---- 1. Load model with lazy-decoder transcoders -----------------------
    model, tokenizer = load_model_lazy(
        model_id=args.model_id,
        transcoder_hf_ref=args.transcoder_hf_ref,
        cache_dir=args.cache_dir,
        dtype=dtype,
    )

    # ---- 2. Build chat prompt ----------------------------------------------
    prompt_text = build_chat_prompt(tokenizer, args.custom_prompt, args.system_instruction)
    print("\n" + "=" * 60)
    print("Prompt (formatted):")
    print(prompt_text)
    print("=" * 60)

    print(f"Slug: {slug}")

    # ---- 3. Attribution loop — one graph per generated token ---------------
    current_prompt = prompt_text
    for tok_idx in range(args.num_tokens):
        token_slug = f"{slug}-tok{tok_idx}" if args.num_tokens > 1 else slug

        if tok_idx > 0:
            # Append the previously generated token to the prompt so that
            # attribution targets the *next* token in the sequence.
            # TransformerLensReplacementModel doesn't have .generate(); instead
            # run a forward pass and take the argmax of the last-position logits.
            input_ids = tokenizer(current_prompt, return_tensors="pt")["input_ids"][0]
            bos_id = tokenizer.bos_token_id
            if len(input_ids) > 0 and input_ids[0].item() == bos_id:
                input_ids = input_ids[1:]  # strip BOS — model adds it internally
            device = next(model.parameters()).device
            with torch.no_grad():
                logits = model(input_ids.unsqueeze(0).to(device))  # [1, seq, vocab]
            next_token_id = int(logits[0, -1, :].argmax())
            next_token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)
            current_prompt = current_prompt + next_token_str
            print(f"\n[Token {tok_idx}] Appended {next_token_str!r} → prompt now ends with "
                  f"…{current_prompt[-60:]!r}")

        print(f"\n{'='*60}")
        print(f"Attributing token position {tok_idx}  (slug: {token_slug})")
        print(f"{'='*60}")

        graph, graph_path = attribute_first_token(
            model=model,
            tokenizer=tokenizer,
            prompt_text=current_prompt,
            graph_dir=Path(args.graph_dir),
            graph_file_dir=Path(args.graph_file_dir),
            slug=token_slug,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            max_feature_nodes=args.max_feature_nodes,
            offload=offload,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            verbose=args.verbose,
        )

        print(f"\nAttribution complete. Graph: {graph_path}")

        # ---- 4. Top-feature examples from mwhanna store --------------------
        if args.top_features > 0:
            print_top_feature_examples(graph, top_n=args.top_features)

    # ---- 5. Optional: launch visualiser ------------------------------------
    if not args.no_serve:
        from circuit_tracer.frontend.local_server import serve

        server = serve(data_dir=str(args.graph_file_dir), port=args.port)
        print(
            f"\nVisualiser running at http://localhost:{args.port}/index.html\n"
            "Press Ctrl-C to stop."
        )
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()
            print("Server stopped.")


if __name__ == "__main__":
    main()
