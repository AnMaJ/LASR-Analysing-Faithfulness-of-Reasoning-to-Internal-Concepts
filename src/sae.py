from huggingface_hub import hf_hub_download
import torch
from functools import partial
import torch.nn as nn
from safetensors.torch import load_file
from loading_dataset import ReasoningDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JumpReLUSAE(nn.Module):
    def __init__(self, d_in, d_sae, affine_skip_connection=False):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts):
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.w_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon
    

def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    """Generic hook function whic stores activations (either input or output of a particular PyTorch module)."""
    acts = inputs[0].squeeze(0) if use_input else outputs[0]  # inputs usually have a batch dim
    cache[key] = acts
    return outputs


def gather_residual_activations(model, target_layer, inputs):

  cache = {}

  handle = model.model.language_model.layers[target_layer].register_forward_hook(
        partial(gather_acts_hook, cache=cache, key="resid_post", use_input=False)
  )

  try:
    _ = model.forward(inputs)
  finally:
    handle.remove()

  return cache["resid_post"]

def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    """Generic hook function whic stores activations (either input or output of a particular PyTorch module)."""
    acts = inputs[0].squeeze(0) if use_input else outputs[0]  # inputs usually have a batch dim
    cache[key] = acts
    return outputs


def gather_residual_activations(model, target_layer, inputs):
    cache = {}

    handle = model.model.language_model.layers[target_layer].register_forward_hook(
            partial(gather_acts_hook, cache=cache, key="resid_post", use_input=False)
    )

    try:
        _ = model.forward(inputs)
    finally:
        handle.remove()

    return cache["resid_post"]

def initialize_sae_from_pretrained_weights(sae_store: str, layer: int, width: str, l0: str):
    # downloading the saes tored on hf
    path_to_params = hf_hub_download(
    repo_id=sae_store,
    filename=f"resid_post/layer_{layer}_width_{width}_l0_{l0}/params.safetensors",
    )
    params = load_file(path_to_params)
    
    d_model, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(params)
    sae.to(device)
    return sae


def fwd_pass_with_sae_intervention(model, sae, target_layer, inputs):
      # Forward pass to get logits & hidden activations
  model_output_clean = model.forward(inputs, output_hidden_states=True)
  logits_clean = model_output_clean.logits[0]  # (seq, d_vocab)
  input_acts = model_output_clean.hidden_states[target_layer + 1][0]  # (seq, d_model)

  # Get the SAE reconstruction
  recon = sae.forward(input_acts.to(torch.float32))

  def intervene_on_target_act_hook(mod, inputs, outputs):
    outputs[0, 1:] = recon[1:]
    return outputs

  handle = model.model.language_model.layers[target_layer].register_forward_hook(intervene_on_target_act_hook)
  try:
    model_output = model.forward(inputs)
  finally:
    handle.remove()

  # Get logits from this corrupted forward pass
  logits = model_output.logits[0]

  return logits_clean, logits


def cross_entropy_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
  """Measures avg cross entropy loss."""
  logprobs = logits[:-1].log_softmax(dim=-1)
  tokens = tokens[1:]
  correct_logprobs = logprobs[torch.arange(len(tokens)), tokens]
  return -correct_logprobs


def get_top_k_features_from_sae(args, sample_input, model, tokenizer, k):
    
    inputs = tokenizer(sample_input, return_tensors="pt", add_special_tokens=True).to(device)
    target_activations = gather_residual_activations(model, args.target_layer, inputs)
    
    sae = initialize_sae_from_pretrained_weights(args.sae_store, args.target_layer, args.sae_width, args.l0_sparsity)
    
    sae_activations = sae.encode(target_activations)
    reconstruction = sae.decode(sae_activations)
    
    reconstruction_mse = torch.mean((reconstruction[:, 1:] - target_activations[:, 1:].float()) ** 2)
    target_variance = target_activations[:, 1:].float().var()

    fvu = reconstruction_mse / target_variance
    print(f"Fraction of variance unexplained: {fvu:.2%}")
    
    l0_per_token = (sae_activations > 1).sum(-1)
    print(l0_per_token.tolist())

    print(f"Average L0: {l0_per_token[1:].float().mean():.2f}")
    
    
    logits_clean, logits_sae = fwd_pass_with_sae_intervention(model, sae, args.target_layer, inputs)
    loss_clean = cross_entropy_loss(logits_clean, inputs[0])
    loss_sae = cross_entropy_loss(logits_sae, inputs[0])

    print(f"Loss (clean): {loss_clean.mean():.4f}")
    print(f"Loss (corrupted): {loss_sae.mean():.4f}")
    print(f"Delta loss: {loss_sae.mean() - loss_clean.mean():.4f}")
    
    top_activations, top_features = sae_activations.max(-1)
    top_acts, top_latents = sae_activations.squeeze().mean(0).topk(k)

    for act, idx in zip(top_acts, top_latents):
        print(f"{act:>6.1f} | {idx}")
    
    return top_activations, top_latents, sae_activations


        