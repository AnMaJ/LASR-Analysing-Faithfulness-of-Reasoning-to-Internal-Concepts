import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.configs import SAEConfig

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


    def get_reconstruction_stats(self, activations: torch.Tensor, encoded: torch.Tensor = None):
        """Compute reconstruction quality for a single prompt.

        Args:
            activations: Tensor of shape ``(n_tokens, d_model)``.
            encoded: Pre-computed encoded activations (optional). If provided,
                avoids a redundant encode call.

        Returns:
            Dict with ``fvu`` (fraction of variance unexplained) and
            ``l0`` (average number of active features per token).
        """
        if encoded is None:
            encoded = self.encode(activations)
        recon = self.decode(encoded)
        if self.affine_skip_connection is not None:
            recon = recon + activations @ self.affine_skip_connection
        reconstruction_mse = torch.mean((recon[1:] - activations[1:].float()) ** 2)
        target_variance = activations[1:].float().var()

        fvu = reconstruction_mse / target_variance

        return {
            "fvu": fvu.item(),
            "l0": (encoded > 0).float().sum(dim=-1).mean().item(),
        }