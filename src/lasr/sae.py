import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from configs import SAEConfig


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, affine_skip_connection: bool = False):
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

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon


def load_sae(config: SAEConfig) -> JumpReLUSAE:
    """Download SAE weights from HuggingFace and return a ready-to-use model."""
    path_to_params = hf_hub_download(
        repo_id=config.repo_id,
        filename=f"resid_post/layer_{config.layer}_width_{config.width}_l0_{config.l0}/params.safetensors",
    )
    params = load_file(path_to_params)
    d_model, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(params)
    sae.cuda()
    return sae
