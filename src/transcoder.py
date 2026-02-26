import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.configs import TranscoderConfig


class JumpReLUTranscoder(nn.Module):
    """JumpReLU Transcoder for Gemma Scope 2.

    A transcoder maps the pre-feedforward-layernorm activations (input to MLP)
    to post-feedforward-layernorm activations (output of MLP), using a sparse
    intermediate representation.  It replaces the MLP block with an
    interpretable, sparse computation.

    Hook points (layer 31 example):
        in:  model.layers.31.pre_feedforward_layernorm.output
        out: model.layers.31.post_feedforward_layernorm.output
    """

    def __init__(self, d_in: int, d_sae: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.d_out = d_out

        # Encoder
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

        # Decoder
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode pre-FFN activations to sparse transcoder features."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to post-FFN activation space."""
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @classmethod
    def from_pretrained(cls, config: TranscoderConfig, device: str = "cpu") -> "JumpReLUTranscoder":
        """Download weights from Hugging Face and return an initialized transcoder.

        Args:
            config: TranscoderConfig with repo_id, layer, width, and l0.
            device: Device to load the transcoder onto.
        """
        print(f"Loading transcoder {config.transcoder_path} from {config.repo_id}")
        path_to_params = hf_hub_download(
            repo_id=config.repo_id,
            filename=config.transcoder_path,
        )
        params = load_file(path_to_params)
        d_in, d_sae = params["w_enc"].shape
        d_out = params["w_dec"].shape[1]

        transcoder = cls(d_in, d_sae, d_out)
        transcoder.load_state_dict(params)
        transcoder = transcoder.to(device=device, dtype=torch.float32)

        return transcoder

    def get_reconstruction_stats(
        self,
        pre_ffn_acts: torch.Tensor,
        post_ffn_acts: torch.Tensor,
        encoded: torch.Tensor = None,
    ) -> dict:
        """Compute reconstruction quality for a single prompt.

        Args:
            pre_ffn_acts: Tensor of shape ``(n_tokens, d_in)`` — transcoder input.
            post_ffn_acts: Tensor of shape ``(n_tokens, d_out)`` — reconstruction target.
            encoded: Pre-computed encoded activations (optional).

        Returns:
            Dict with ``fvu`` (fraction of variance unexplained) and
            ``l0`` (average number of active features per token).
        """
        if encoded is None:
            encoded = self.encode(pre_ffn_acts)
        recon = self.decode(encoded)
        reconstruction_mse = torch.mean((recon - post_ffn_acts.float()) ** 2)
        target_variance = post_ffn_acts.float().var()
        fvu = reconstruction_mse / target_variance
        return {
            "fvu": fvu.item(),
            "l0": (encoded > 0).float().sum(dim=-1).mean().item(),
        }
