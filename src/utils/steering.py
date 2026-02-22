import torch


def generate_with_steering_and_capture(
    model, sae, inputs, target_layer, feature_idx, coeff=None, max_new_tokens=500
):
    """
    Generate text while capturing SAE activations at `target_layer`.

    Args:
        coeff: if None, the residual stream is left untouched (pure capture).
               if a scalar, adds  coeff * sae.w_dec[feature_idx]  to the
               residual stream at every forward pass through that layer.

    Returns:
        (generated_text, token_ids, sae_activations)
        sae_activations has shape [n_tokens, d_sae] and does not require grad.
    """
    captured_sae_acts = []
    hook_call_count = [0]

    target_module = _get_target_layer(model, target_layer)

    # Pre-compute the steering direction once (detached, on the right device/dtype)
    steering_vec = None
    if coeff is not None:
        steering_vec = sae.w_dec[feature_idx].detach().clone()  # [d_model]

    def hook_fn(mod, input, output):
        hook_call_count[0] += 1

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        original_dtype = hidden_states.dtype
        h = hidden_states.to(dtype=sae.w_dec.dtype)

        # Encode through SAE → sparse feature activations
        with torch.no_grad():
            sae_acts = sae.encode(h)

        # Capture pre-modification activations
        captured_sae_acts.append(sae_acts.detach().clone().cpu())

        if steering_vec is not None:
            # Add coeff * decoder direction to every token position
            sv = steering_vec.to(device=h.device, dtype=h.dtype)
            h_steered = h + coeff * sv  # broadcast over batch and sequence dims
            hidden_states[:] = h_steered.to(dtype=original_dtype)

        return output

    handle = target_module.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            )
        output_str = model.tokenizer.decode(out[0])
    finally:
        handle.remove()

    assert hook_call_count[0] > 0, "Hook never fired!"

    # [n_tokens, d_sae] — guaranteed detached, on CPU
    sae_acts_full = torch.cat(captured_sae_acts, dim=1).squeeze(0).detach().cpu()

    text = output_str.split("<start_of_turn>model")[-1].strip()
    return text, out[0], sae_acts_full


# Backward-compatible alias
generate_with_ablation_and_capture = generate_with_steering_and_capture


def _get_target_layer(model, layer_idx):
    """
    Try common module paths to find the decoder layer.
    Falls back to searching named_modules if hardcoded paths fail.
    """
    # Common paths for various model wrappers (most specific first)
    candidates = [
        lambda: model.model.model.language_model.layers[layer_idx],  # PaliGemma / multimodal wrapper
        lambda: model.model.model.layers[layer_idx],                 # double-wrapped
        lambda: model.model.layers[layer_idx],                       # standard HF (GemmaForCausalLM)
        lambda: model.language_model.model.layers[layer_idx],        # some multimodal
    ]

    for getter in candidates:
        try:
            layer = getter()
            # Sanity check: it should have an attention sub-module
            if hasattr(layer, "self_attn") or hasattr(layer, "attention"):
                return layer
        except (AttributeError, IndexError):
            continue

    # Fallback: search all named modules
    for name, mod in model.named_modules():
        if (
            f"layers.{layer_idx}" in name
            and name.endswith(f".{layer_idx}")
            and (hasattr(mod, "self_attn") or hasattr(mod, "attention"))
        ):
            print(f"Auto-discovered layer path: {name}")
            return mod

    raise ValueError(
        f"Could not find layer {layer_idx}. Available layer-like modules:\n"
        + "\n".join(
            f"  {name}" for name, _ in model.named_modules()
            if "layers" in name and str(layer_idx) in name
        )
    )


def find_all_layer_paths(model):
    """Utility: print all modules that look like transformer layers."""
    print("Modules containing 'layers':")
    for name, mod in model.named_modules():
        if "layers." in name and name.endswith(name.split(".")[-1]):
            # Only print leaf-ish layer modules
            if hasattr(mod, "self_attn") or hasattr(mod, "attention"):
                print(f"  {name}  [{mod.__class__.__name__}]")
