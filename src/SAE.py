import torch
import torch.nn as nn
import einops
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.configs import SAEConfig, CrosscoderConfig

class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder for Gemma Scope 2.

    This architecture uses a JumpReLU activation function which applies
    a threshold before the ReLU, promoting sparsity in the feature activations.
    """

    def __init__(self, d_in: int, d_sae: int, affine_skip_connection: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Encoder weights
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

        # Decoder weights
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Optional affine skip connection
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse feature activations."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon
    
    @classmethod
    def from_pretrained(cls, config: SAEConfig, device: str = "cpu") -> "JumpReLUSAE":
        """Download weights from Hugging Face and return an initialized SAE.

        Args:
            config: SAEConfig with repo_id, sae_type, layer, width, and l0.
            device: Device to load the SAE onto.
        """
        print(f"Load SAE {config.sae_path} from {config.repo_id}")
        path_to_params = hf_hub_download(
            repo_id=config.repo_id,
            filename=config.sae_path,
        )
        params = load_file(path_to_params)
        d_model, d_sae = params["w_enc"].shape

        sae = cls(d_model, d_sae)
        sae.load_state_dict(params)
        sae = sae.to(device=device, dtype=torch.float32)

        return sae


    def get_reconstruction_stats(self, activations: torch.Tensor):
        """Compute reconstruction quality for a single prompt.

        Args:
            activations: Tensor of shape ``(n_tokens, d_model)``.

        Returns:
            Dict with ``fvu`` (fraction of variance unexplained) and
            ``l0`` (average number of active features per token).
        """
        recon = self.forward(activations)
        reconstruction_mse = torch.mean((recon[1:] - activations[1:].float()) ** 2)
        target_variance = activations[1:].float().var()

        fvu = reconstruction_mse / target_variance

        return {
            "fvu": fvu.item(),
            "l0": (self.encode(activations) > 0).float().sum(dim=-1).mean().item(),
        }


class JumpReLUMultiLayerSAE(nn.Module):
    """Weakly-causal multi-layer crosscoder using JumpReLU activation.

    Used with Gemma-scope-2 crosscoders that span multiple residual stream layers.
    The encoder maps (num_layers, d_in) → (num_layers, d_sae) and the all-to-all
    decoder maps (num_layers, d_sae) → (num_layers, d_in).

    Args:
        d_in: Residual stream dimension (inferred from checkpoint weights).
        d_sae: Number of crosscoder features (262144 for 262k).
        num_layers: Number of hooked layers (4 for layers [16, 31, 40, 53]).
    """

    def __init__(self, d_in: int, d_sae: int, num_layers: int):
        super().__init__()
        self.w_enc   = nn.Parameter(torch.zeros(num_layers, d_in, d_sae))
        self.w_dec   = nn.Parameter(torch.zeros(num_layers, d_sae, num_layers, d_in))
        self.threshold = nn.Parameter(torch.zeros(num_layers, d_sae))
        self.b_enc   = nn.Parameter(torch.zeros(num_layers, d_sae))
        self.b_dec   = nn.Parameter(torch.zeros(num_layers, d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode residual activations to sparse crosscoder features.

        Args:
            x: Tensor of shape (..., num_layers, d_in).

        Returns:
            Sparse feature activations of shape (..., num_layers, d_sae).
        """
        pre = einops.einsum(
            x, self.w_enc,
            "... layer d_in, layer d_in d_sae -> ... layer d_sae",
        ) + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode crosscoder features back to residual stream space.

        Args:
            acts: Feature activations of shape (..., li, d_sae).

        Returns:
            Reconstructed activations of shape (..., lo, d_in).
        """
        return (
            einops.einsum(
                acts, self.w_dec,
                "... li d_sae, li d_sae lo d -> ... lo d",
            )
            + self.b_dec
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode."""
        return self.decode(self.encode(x))

    @classmethod
    def from_pretrained(
        cls,
        config: "CrosscoderConfig",
        device: str = "cpu",
        hf_token: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "JumpReLUMultiLayerSAE":
        """Download per-layer weight files from HuggingFace and assemble crosscoder.

        Args:
            config: CrosscoderConfig specifying repo, layers, width, l0.
            device: Target device.
            hf_token: Optional HuggingFace token for private repos.
            dtype: Weight dtype. Defaults to bfloat16 to halve crosscoder memory usage.

        Returns:
            Initialized JumpReLUMultiLayerSAE in eval mode on *device*.
        """
        num_layers = len(config.layers)
        params_list = []
        for i in range(num_layers):
            path = hf_hub_download(
                repo_id=config.repo_id,
                filename=config.params_path(i),
                token=hf_token,
            )
            params_list.append(load_file(path))

        params_stacked = {
            k: torch.stack([p[k] for p in params_list])
            for k in params_list[0].keys()
        }

        d_in  = params_stacked["w_enc"].shape[1]   # infer from weights
        d_sae = params_stacked["w_enc"].shape[-1]
        cc = cls(d_in, d_sae, num_layers)
        cc.load_state_dict(params_stacked)
        cc = cc.to(device=device, dtype=dtype)
        cc.eval()
        return cc