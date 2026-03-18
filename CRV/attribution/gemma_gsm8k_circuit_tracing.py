"""
Attribution graph generation for Gemma 3 1B IT on GSM8K.

Generates one attribution graph per generated token for each GSM8K example,
using Gemma Scope 2 transcoders via the circuit-tracer library.

Usage:
    python gemma_gsm8k_circuit_tracing.py \
        --num_examples 10 \
        --max_new_tokens 256 \
        --graph_dir ./graphs/gemma_gsm8k \
        --graph_file_dir ./graph_files/gemma_gsm8k \
        --output_json ./gemma_gsm8k_results.json

Notes:
    - Requires the circuit-tracer library: https://github.com/decoderesearch/circuit-tracer
    - Transcoders loaded from google/gemma-scope-2-1b-it (transcoder_all subfolder)
    - Each generated token yields one .pt attribution graph file
    - Per-token attribution is expensive; use --max_new_tokens and --max_tokens_to_attribute
      to limit the number of graphs generated per example
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

# Add project root so we can import src.dataset.gsm8k
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files

# ---------------------------------------------------------------------------
# Model / transcoder IDs
# ---------------------------------------------------------------------------

GEMMA_MODEL_ID = "google/gemma-3-1b-it"
GEMMA_SCOPE_REPO = "google/gemma-scope-2-1b-it"
GEMMA_SCOPE_SUBFOLDER = "transcoder_all"

# Hook names used by TransformerLens for Gemma 3 MLP replacement.
# feature_input_hook  = residual stream AFTER attention, BEFORE MLP
# feature_output_hook = MLP output that is added back to the residual stream
FEATURE_INPUT_HOOK = "hook_resid_mid"
FEATURE_OUTPUT_HOOK = "hook_mlp_out"

# Which transcoder variant to load.  Each layer in gemma-scope-2 ships several
# variants distinguished by width (number of features) and l0 (sparsity target).
# E.g. "width_262k_l0_big", "width_16k_l0_big", "width_262k_l0_small", …
TRANSCODER_VARIANT = "width_262k_l0_big"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _discover_transcoder_files(
    repo_id: str,
    subfolder: str,
    variant: str = TRANSCODER_VARIANT,
) -> List[str]:
    """Return one safetensors HF URI per layer, filtered to *variant*.

    The gemma-scope-2 repo contains multiple transcoder variants per layer,
    e.g. ``layer_0_width_262k_l0_big/params.safetensors`` and
    ``layer_0_width_16k_l0_small/params.safetensors``.  This function keeps
    only the files whose path contains *variant*, then sorts them by layer
    index so they are passed to ``load_transcoder_set`` in layer order.

    Args:
        repo_id:   HuggingFace repo (e.g. ``"google/gemma-scope-2-1b-it"``).
        subfolder: Path within the repo (e.g. ``"transcoder_all"``).
        variant:   Substring that must appear in the file path, used to select
                   one specific variant per layer (e.g. ``"width_262k_l0_big"``).

    Returns:
        Sorted list of ``hf://<repo_id>/<path>`` URI strings, one per layer.

    Raises:
        FileNotFoundError: if no files matching *variant* are found.
    """
    from huggingface_hub import list_repo_files
    import re

    prefix = f"{subfolder}/"
    all_files = list(list_repo_files(repo_id))

    # Keep only the weight file (params.safetensors) for the chosen variant.
    # - "examples.safetensors" stores pre-computed max-activating examples for
    #   human interpretability (Neuronpedia-style) — not needed for attribution.
    # - The variant is matched as a complete path segment so "width_262k_l0_big"
    #   does NOT accidentally match "width_262k_l0_big_affine".
    layer_files = [
        f for f in all_files
        if f.startswith(prefix)
        and f.endswith("params.safetensors")
        and re.search(re.escape(variant) + r"/", f)
    ]

    if not layer_files:
        raise FileNotFoundError(
            f"No params.safetensors files matching variant '{variant}' found under "
            f"{repo_id}/{subfolder}.\n"
            f"Available files (first 20): {[f for f in all_files if f.startswith(prefix)][:20]}"
        )

    # Sort by the layer index extracted from the filename so that layer 0 comes
    # first, layer 1 second, etc.  Path format: "<subfolder>/layer_<N>_<...>.safetensors"
    def _layer_idx(path: str) -> int:
        m = re.search(r"layer_(\d+)", path)
        return int(m.group(1)) if m else 0

    layer_files = sorted(layer_files, key=_layer_idx)
    uris = [f"hf://{repo_id}/{f}" for f in layer_files]
    print(f"Selected {len(uris)} '{variant}' transcoder files from {repo_id}/{subfolder}:")
    for u in uris:
        print(f"  {u}")
    return uris


def load_model_and_tokenizer(
    model_id: str = GEMMA_MODEL_ID,
    transcoder_repo: str = GEMMA_SCOPE_REPO,
    transcoder_subfolder: str = GEMMA_SCOPE_SUBFOLDER,
    feature_input_hook: str = FEATURE_INPUT_HOOK,
    feature_output_hook: str = FEATURE_OUTPUT_HOOK,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ReplacementModel, AutoTokenizer]:
    """Load Gemma 3 1B IT with Gemma Scope 2 transcoders.

    The ``google/gemma-scope-2-1b-it`` repo does not ship a ``config.yaml``
    in the format circuit-tracer expects, so we bypass
    ``ReplacementModel.from_pretrained()`` (which calls
    ``load_transcoder_from_hub`` → looks for ``config.yaml``) and instead:

    1. Enumerate the actual ``.safetensors`` files in the repo.
    2. Build the config dict in memory (matching what ``config.yaml`` would
       contain, including ``repo_id`` so the ``"gemma-scope-2"`` branch in
       ``load_transcoders`` is triggered).
    3. Call ``load_transcoders()`` directly.
    4. Create the model via ``ReplacementModel.from_pretrained_and_transcoders()``.

    Args:
        model_id: HuggingFace ID for the base Gemma 3 1B IT model.
        transcoder_repo: HuggingFace repo for Gemma Scope 2 transcoders.
        transcoder_subfolder: Subfolder within the repo.
        feature_input_hook: TransformerLens hook name for the MLP input
            (residual stream after attention, before MLP).
        feature_output_hook: TransformerLens hook name for the MLP output.
        dtype: Floating-point dtype (bfloat16 recommended).

    Returns:
        Tuple of (ReplacementModel, AutoTokenizer).
    """
    from circuit_tracer.utils.hf_utils import load_transcoders

    print(f"Loading model: {model_id}")
    print(f"Loading transcoders: {transcoder_repo}/{transcoder_subfolder}")

    # Step 1: discover the per-layer safetensors files
    hf_uris = _discover_transcoder_files(transcoder_repo, transcoder_subfolder)

    # Step 2: build the config dict that load_transcoders() expects.
    # Including repo_id with "gemma-scope-2" ensures load_transcoders() picks
    # load_gemma_scope_2_transcoder (which handles the w_enc/w_dec key names).
    scan_id = f"{transcoder_repo}//{transcoder_subfolder}"
    config = {
        "model_kind": "transcoder_set",
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "repo_id": transcoder_repo,   # triggers "gemma-scope-2" special loader
        "revision": None,
        "subfolder": transcoder_subfolder,
        "scan": scan_id,
        "transcoders": hf_uris,       # list of hf:// URIs, one per layer
    }

    # Step 3: load transcoders (lazy_decoder=False required for gemma-scope-2 format)
    transcoders = load_transcoders(
        config,
        device=None,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    # Step 4: build the ReplacementModel (skips config.yaml lookup entirely)
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=model_id,
        transcoders=transcoders,
        dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer: AutoTokenizer, question: str, system_instruction: str) -> str:
    """Format the GSM8K question as a Gemma 3 IT chat prompt.

    Applies the tokenizer's built-in chat template so the prompt matches
    exactly what the model was fine-tuned on.

    Args:
        tokenizer: Gemma 3 tokenizer with chat template.
        question: Raw GSM8K question string.
        system_instruction: Instruction prepended to the question.

    Returns:
        Formatted prompt string (does NOT include the model-turn opener so
        that the model generates from scratch).
    """
    messages = [{"role": "user", "content": f"{system_instruction}\n\nQuestion: {question}"}]
    # add_generation_prompt=True appends the model-turn opener
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def generate_response(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> Tuple[torch.Tensor, str]:
    """Run greedy decoding and return the generated token IDs and decoded text.

    ReplacementModel wraps a TransformerLens HookedTransformer, whose
    generate() signature differs from HuggingFace:
      - First argument is the token tensor directly (not input_ids=)
      - Returns the full sequence (prompt + generated tokens) as a 2-D tensor
      - Uses stop_at_eos / do_sample rather than pad_token_id

    Args:
        model: ReplacementModel backed by a HookedTransformer.
        tokenizer: Gemma 3 tokenizer.
        prompt_text: Fully formatted prompt string.
        max_new_tokens: Maximum tokens to generate.
        device: Torch device string.

    Returns:
        Tuple of (generated_ids tensor of shape [n_gen_tokens],
                  decoded generated text string).
    """
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        # TransformerLens generate: first positional arg is the token tensor.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_at_eos=True,
            verbose=False,
        )

    generated_ids = output_ids[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, generated_text


# ---------------------------------------------------------------------------
# Per-token attribution
# ---------------------------------------------------------------------------

def attribute_token(
    model: ReplacementModel,
    prefix_text: str,
    graph_name: str,
    slug: str,
    graph_dir: Path,
    graph_file_dir: Path,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 64,
    max_feature_nodes: int = 4096,
    offload: Optional[str] = "disk",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    verbose: bool = False,
) -> Path:
    """Compute and save the attribution graph for the next predicted token.

    The circuit-tracer `attribute()` function traces the computation from
    transcoder features through the network to the final logits, building a
    directed graph where nodes are transcoder features and edges are weighted
    by their causal influence.

    Args:
        model: ReplacementModel with transcoders.
        prefix_text: All text seen by the model before the token to attribute.
        graph_name: Filename stem (no extension) for saving the .pt graph.
        slug: Identifier string used by create_graph_files for the visualiser.
        graph_dir: Directory to write the .pt graph tensor file.
        graph_file_dir: Directory to write visualisation JSON files.
        max_n_logits: Maximum number of output logits to include in attribution.
        desired_logit_prob: Probability mass covered by attributed logits.
        batch_size: Internal batch size for the attribution pass.
        max_feature_nodes: Maximum transcoder feature nodes in the graph.
        offload: Where to offload intermediate tensors ('disk', 'cpu', None).
        node_threshold: Cumulative influence threshold for pruning nodes.
        edge_threshold: Cumulative influence threshold for pruning edges.
        verbose: Show tqdm progress bars inside circuit-tracer.

    Returns:
        Path to the saved .pt graph file.
    """
    graph = attribute(
        prompt=prefix_text,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
    )

    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"{graph_name}.pt"
    graph.to_pt(graph_path)

    graph_file_dir.mkdir(parents=True, exist_ok=True)
    create_graph_files(
        graph_or_path=graph_path,
        slug=slug,
        output_path=graph_file_dir,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    return graph_path


# ---------------------------------------------------------------------------
# Main per-example pipeline
# ---------------------------------------------------------------------------

def process_example(
    example_idx: int,
    question: str,
    gold_answer: str,
    prompt_text: str,
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    graph_name_prefix: str,
    slug_prefix: str,
    graph_dir: Path,
    graph_file_dir: Path,
    max_new_tokens: int = 256,
    max_tokens_to_attribute: Optional[int] = None,
    device: str = "cuda",
    # attribution kwargs passed through
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 64,
    max_feature_nodes: int = 4096,
    offload: Optional[str] = "disk",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    verbose: bool = False,
) -> Dict:
    """Generate attribution graphs for every token in the model's response.

    For a response of length T, this function runs T attribution passes,
    each with the prompt truncated just before token t.  The resulting
    graphs capture which features drove each individual generation step.

    Args:
        example_idx: Position of this example in the dataset (used for naming).
        question: Raw question text (for logging and metadata).
        gold_answer: Ground-truth answer string.
        prompt_text: Fully formatted chat prompt.
        model: ReplacementModel with Gemma Scope 2 transcoders.
        tokenizer: Gemma 3 tokenizer.
        graph_name_prefix: Prefix for graph file names.
        slug_prefix: Prefix for visualiser slugs.
        graph_dir: Root directory for .pt files.
        graph_file_dir: Root directory for visualiser JSON files.
        max_new_tokens: Cap on generation length.
        max_tokens_to_attribute: If set, only attribute the first N generated
            tokens (useful to limit compute while keeping the full response).
        device: Torch device string.
        max_n_logits: Passed to attribute().
        desired_logit_prob: Passed to attribute().
        batch_size: Passed to attribute().
        max_feature_nodes: Passed to attribute().
        offload: Passed to attribute().
        node_threshold: Passed to create_graph_files().
        edge_threshold: Passed to create_graph_files().
        verbose: Passed to attribute().

    Returns:
        Dict with keys: example_idx, question, gold_answer, generated_text,
        n_tokens, token_results (list of per-token dicts).
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Example {example_idx}: {question[:80]}...")

    # ---- 1. Generate full response ----------------------------------------
    generated_ids, generated_text = generate_response(
        model, tokenizer, prompt_text, max_new_tokens=max_new_tokens, device=device
    )
    n_gen = len(generated_ids)
    print(f"Generated {n_gen} tokens.")
    print(f"Response preview: {generated_text[:200]}...")

    # ---- 2. Determine which tokens to attribute ---------------------------
    n_to_attribute = n_gen
    if max_tokens_to_attribute is not None:
        n_to_attribute = min(n_gen, max_tokens_to_attribute)

    # Encode the prompt once to get its token IDs (keep on CPU for decode)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].cpu()
    generated_ids_cpu = generated_ids.cpu()

    # ---- 3. Attribution loop -----------------------------------------------
    token_results = []
    for tok_pos in range(n_to_attribute):
        # Build the prefix: original prompt + all previously generated tokens
        prefix_ids = torch.cat([prompt_ids, generated_ids_cpu[:tok_pos]])
        # Strip leading BOS: circuit-tracer/TransformerLens prepends it automatically
        bos_id = tokenizer.bos_token_id
        ids_no_bos = prefix_ids[1:] if (len(prefix_ids) > 0 and prefix_ids[0].item() == bos_id) else prefix_ids
        prefix_text = tokenizer.decode(ids_no_bos, skip_special_tokens=False)

        tok_text = tokenizer.decode([generated_ids_cpu[tok_pos].item()], skip_special_tokens=True)
        print(f"  [{tok_pos+1}/{n_to_attribute}] attributing token '{tok_text}'")

        graph_name = f"{graph_name_prefix}_ex{example_idx:04d}_tok{tok_pos:04d}"
        slug = f"{slug_prefix}_ex{example_idx:04d}_tok{tok_pos:04d}"

        try:
            graph_path = attribute_token(
                model=model,
                prefix_text=prefix_text,
                graph_name=graph_name,
                slug=slug,
                graph_dir=graph_dir,
                graph_file_dir=graph_file_dir,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                verbose=verbose,
            )
            token_results.append({
                "token_pos": tok_pos,
                "token_text": tok_text,
                "graph_path": str(graph_path),
                "status": "success",
            })
        except Exception as exc:
            print(f"    ERROR at token {tok_pos}: {exc}")
            token_results.append({
                "token_pos": tok_pos,
                "token_text": tok_text,
                "graph_path": None,
                "status": f"error: {exc}",
            })

    return {
        "example_idx": example_idx,
        "question": question,
        "gold_answer": gold_answer,
        "generated_text": generated_text,
        "n_tokens": n_gen,
        "n_attributed": n_to_attribute,
        "token_results": token_results,
    }


# ---------------------------------------------------------------------------
# Dataset loading (standalone, no BaseDataset dependency)
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "test") -> List[Dict]:
    """Load the GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ('train' or 'test').

    Returns:
        List of dicts with keys 'question' and 'answer'.
    """
    from datasets import load_dataset as hf_load
    print(f"Loading GSM8K '{split}' split …")
    ds = hf_load("openai/gsm8k", "main", split=split)
    data = list(ds)
    print(f"Loaded {len(data)} examples.")
    return data


def extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer after the #### marker."""
    import re
    m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", answer_text)
    return nums[-1].replace(",", "").strip() if nums else ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-token attribution graphs for Gemma 3 1B IT on GSM8K."
    )

    # Dataset
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="GSM8K split to use (default: test).")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First example index (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Last example index (exclusive). None = all.")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to process (overrides end_idx).")

    # Model
    parser.add_argument("--model_id", type=str, default=GEMMA_MODEL_ID,
                        help="HuggingFace model ID for the base Gemma 3 1B IT model.")
    parser.add_argument("--transcoder_repo", type=str, default=GEMMA_SCOPE_REPO,
                        help=f"HuggingFace repo for Gemma Scope 2 transcoders "
                             f"(default: {GEMMA_SCOPE_REPO}).")
    parser.add_argument("--transcoder_subfolder", type=str,
                        default=GEMMA_SCOPE_SUBFOLDER,
                        help=f"Subfolder in the transcoder repo "
                             f"(default: {GEMMA_SCOPE_SUBFOLDER}).")
    parser.add_argument("--feature_input_hook", type=str,
                        default=FEATURE_INPUT_HOOK,
                        help="TransformerLens hook name for the MLP input "
                             f"(default: {FEATURE_INPUT_HOOK}).")
    parser.add_argument("--feature_output_hook", type=str,
                        default=FEATURE_OUTPUT_HOOK,
                        help="TransformerLens hook name for the MLP output "
                             f"(default: {FEATURE_OUTPUT_HOOK}).")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (default: bfloat16).")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate per example.")
    parser.add_argument("--max_tokens_to_attribute", type=int, default=None,
                        help="Only attribute the first N generated tokens. "
                             "None = attribute all generated tokens.")

    # Prompt
    parser.add_argument("--cot", action="store_true", default=True,
                        help="Use chain-of-thought prompt style (default: True).")
    parser.add_argument("--no_cot", dest="cot", action="store_false",
                        help="Use direct-answer prompt style.")

    # Attribution parameters
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--desired_logit_prob", type=float, default=0.95)
    parser.add_argument("--max_feature_nodes", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--offload", type=str, default="disk",
                        choices=["disk", "cpu", "none"],
                        help="Where to offload attribution intermediates.")
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Show circuit-tracer progress bars.")

    # Output
    parser.add_argument("--graph_dir", type=str, default="./graphs/gemma_gsm8k",
                        help="Directory to write .pt attribution graph files.")
    parser.add_argument("--graph_file_dir", type=str,
                        default="./graph_files/gemma_gsm8k",
                        help="Directory to write visualisation JSON files.")
    parser.add_argument("--output_json", type=str,
                        default="./gemma_gsm8k_results.json",
                        help="Path for the results JSON summary.")
    parser.add_argument("--graph_name_prefix", type=str, default="gemma_gsm8k",
                        help="Prefix for graph file names.")
    parser.add_argument("--slug_prefix", type=str, default="gemma_gsm8k",
                        help="Prefix for visualiser slugs.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    offload = None if args.offload == "none" else args.offload

    # ---- Load model --------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        transcoder_repo=args.transcoder_repo,
        transcoder_subfolder=args.transcoder_subfolder,
        feature_input_hook=args.feature_input_hook,
        feature_output_hook=args.feature_output_hook,
        dtype=dtype,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load dataset ------------------------------------------------------
    data = load_gsm8k(split=args.split)

    start = args.start_idx
    end = args.end_idx
    if args.num_examples is not None:
        end = start + args.num_examples
    if end is None or end > len(data):
        end = len(data)

    examples = data[start:end]
    print(f"Processing examples {start}-{end-1} ({len(examples)} total).")

    # ---- Prompt instruction ------------------------------------------------
    if args.cot:
        system_instruction = (
            "Solve the math problem step by step. "
            "At the end of your response, write the final numeric answer on its "
            "own line in the format: #### <number>"
        )
    else:
        system_instruction = (
            "Solve the math problem. Reply with only the final numeric answer, "
            "no working."
        )

    # ---- Output paths ------------------------------------------------------
    graph_dir = Path(args.graph_dir)
    graph_file_dir = Path(args.graph_file_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # ---- Main loop ---------------------------------------------------------
    all_results = []
    for i, ex in enumerate(examples):
        global_idx = start + i
        question = ex["question"]
        gold_answer = extract_gold_answer(ex["answer"])

        prompt_text = build_chat_prompt(tokenizer, question, system_instruction)

        result = process_example(
            example_idx=global_idx,
            question=question,
            gold_answer=gold_answer,
            prompt_text=prompt_text,
            model=model,
            tokenizer=tokenizer,
            graph_name_prefix=args.graph_name_prefix,
            slug_prefix=args.slug_prefix,
            graph_dir=graph_dir,
            graph_file_dir=graph_file_dir,
            max_new_tokens=args.max_new_tokens,
            max_tokens_to_attribute=args.max_tokens_to_attribute,
            device=device,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            max_feature_nodes=args.max_feature_nodes,
            offload=offload,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            verbose=args.verbose,
        )
        all_results.append(result)

        # Save incrementally so progress is not lost on failure
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved results so far → {output_json}")

    print(f"\nDone. Processed {len(all_results)} examples.")
    print(f"Results written to: {output_json}")


if __name__ == "__main__":
    main()
