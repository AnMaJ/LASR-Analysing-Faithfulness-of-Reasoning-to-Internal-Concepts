from functools import partial

import torch

from lasr.sae import JumpReLUSAE


def _gather_acts_hook(
    mod, inputs, outputs, cache: dict, key: str, use_input: bool
):
    if use_input:
        acts = inputs[0].squeeze(0)
    else:
        acts = outputs[0] if isinstance(outputs, tuple) else outputs
    # Ensure 3-D (batch, seq, d_model) even if the layer squeezed the batch dim
    if acts.ndim == 2:
        acts = acts.unsqueeze(0)
    cache[key] = acts.detach()
    return outputs


def gather_residual_activations(
    model, target_layer: int, inputs: torch.Tensor
) -> torch.Tensor:
    """Run a forward pass and capture the residual stream output at *target_layer*."""
    cache: dict[str, torch.Tensor] = {}

    handle = model.model.language_model.layers[target_layer].register_forward_hook(
        partial(_gather_acts_hook, cache=cache, key="resid_post", use_input=False)
    )

    try:
        model.forward(inputs)
    finally:
        handle.remove()

    return cache["resid_post"]


def encode_activations(
    sae: JumpReLUSAE, activations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode *activations* through the SAE and return ``(sae_acts, reconstruction)``."""
    activations = activations.to(torch.float32)
    sae_acts = sae.encode(activations)
    reconstruction = sae.decode(sae_acts)
    return sae_acts, reconstruction
