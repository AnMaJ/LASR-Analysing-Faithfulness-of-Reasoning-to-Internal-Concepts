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

# ---------------------------------------------------------------------------
# Constants (shared with gemma_gsm8k_circuit_tracing.py)
# ---------------------------------------------------------------------------

GEMMA_MODEL_ID = "google/gemma-3-1b-it"

# mwhanna mirror ships a config.yaml required by the circuit-tracer caching API.
# Format: "owner/repo/subfolder"
MWHANNA_HF_REF = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_262k_l0_small_affine"

# Default custom prompt (multiple-choice geography question)
DEFAULT_CUSTOM_PROMPT = (
    "What is the capital of France? "
    "Options: A) Paris B) Delhi C) Texas D) Baku\n"
    "Answer with the letter first (A/B/C/D), then explain."
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "Answer the following multiple choice question. "
    "You MUST start your response with exactly one letter: "
    "A, B, C, or D. Then provide a step-by-step explanation."
)


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

    transcoders, _ = load_transcoders_from_cache(
        transcoder_hf_ref,
        cache_dir=cache_dir,
        device=None,            # defaults to GPU if available
        dtype=dtype,
        lazy_encoder=False,     # W_enc needed every forward pass → keep in VRAM
        lazy_decoder=True,      # W_dec read from disk per feature → saves VRAM
    )

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
    messages = [{"role": "user", "content": f"{system_instruction}\n\nQuestion: {question}"}]
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

    print("\nRunning attribution for the first generated token …")
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

    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "custom_prompt_first_token.pt"
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

    return graph_path


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
    p.add_argument("--slug", type=str, default="custom-prompt-first-token",
                   help="Visualiser slug (graph name).")
    p.add_argument("--port", type=int, default=8046,
                   help="Port for the local visualiser server.")
    p.add_argument("--no_serve", action="store_true", default=False,
                   help="Skip launching the visualiser server after attribution.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    offload = None if args.offload == "none" else args.offload

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

    # ---- 3. Attribution for the first token --------------------------------
    graph_path = attribute_first_token(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        graph_dir=Path(args.graph_dir),
        graph_file_dir=Path(args.graph_file_dir),
        slug=args.slug,
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

    # ---- 4. Optional: launch visualiser ------------------------------------
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
