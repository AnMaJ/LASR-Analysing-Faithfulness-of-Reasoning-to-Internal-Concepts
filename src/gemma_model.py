from __future__ import annotations

from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs import ModelConfig


class GemmaModel:
    """Wrapper around a Gemma causal-LM for easy loading and generation."""

    def __init__(self, config: ModelConfig):
        """Load the model and tokenizer specified by *config*.

        Args:
            config: A ``ModelConfig`` instance.
        """
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device,
        )
        self.model.eval()

    def generate(self, prompt: str | list[dict], max_new_tokens: int = 256) -> tuple[str, torch.Tensor, int]:
        """Generate a response for the given *prompt*.

        Args:
            prompt: The input text to send to the model (it can be with the chat template)
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            A tuple of (decoded_text, full_output_ids, prompt_length_in_tokens).
        """
        # Handle chat template
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        assert isinstance(output_ids, torch.Tensor)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True), output_ids, prompt_len

    def generate_batch(
        self,
        prompts: list[str | list[dict]],
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ) -> tuple[list[str], list[torch.Tensor], list[int]]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts (strings or chat templates).
            max_new_tokens: Maximum number of tokens to generate per prompt.
            batch_size: Number of prompts to process in parallel.

        Returns:
            A tuple of (texts, output_ids, prompt_lengths) where each element
            is a list with one entry per prompt.
        """
        # Apply chat template where needed
        processed: list[str] = []
        for p in prompts:
            if isinstance(p, list):
                p = self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True,
                )
            processed.append(p)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        all_texts: list[str] = []
        all_ids: list[torch.Tensor] = []
        all_prompt_lens: list[int] = []

        for i in tqdm(range(0, len(processed), batch_size), desc="Generating"):
            batch = processed[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=True,
            ).to(self.model.device)

            # Per-prompt lengths (excluding padding)
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                )

            assert isinstance(output_ids, torch.Tensor)
            padded_input_len = inputs["input_ids"].shape[1]

            pad_id = self.tokenizer.pad_token_id
            padding_len = padded_input_len - torch.tensor(prompt_lens, dtype=torch.long)

            for j in range(output_ids.shape[0]):
                # Strip left-padding, keep prompt + generation
                ids = output_ids[j, int(padding_len[j]):]

                # Strip trailing pad tokens produced when this sequence is shorter
                # than the longest one in the batch (pad_id == eos_id, so find the
                # first EOS in the generated portion and truncate after it).
                prompt_len = int(prompt_lens[j])
                eos_positions = (ids[prompt_len:] == pad_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    ids = ids[: prompt_len + int(eos_positions[0]) + 1]

                all_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
                all_ids.append(ids)
                all_prompt_lens.append(prompt_len)

        return all_texts, all_ids, all_prompt_lens

    @staticmethod
    def _gather_acts_hook(
        _mod, _inputs, outputs, cache: dict, key: str,
    ):
        acts = outputs[0] if isinstance(outputs, tuple) else outputs
        # Remove batch dim: (1, seq, d_model) -> (seq, d_model)
        cache[key] = acts.detach().squeeze(0)
        return outputs

    def gather_residual_activations(
        self, target_layer: int, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Run a forward pass on a single prompt and capture the residual stream at *target_layer*.

        Args:
            target_layer: Index of the transformer layer to hook.
            inputs: 1-D token IDs tensor of shape ``(n_tokens,)``.

        Returns:
            Tensor of shape ``(n_tokens, d_model)``.
        """
        cache: dict[str, torch.Tensor] = {}

        handle = self.model.model.language_model.layers[target_layer].register_forward_hook(
            partial(self._gather_acts_hook, cache=cache, key="resid_post")
        )

        try:
            with torch.no_grad():
                self.model.forward(input_ids=inputs.unsqueeze(0), use_cache=False)
        finally:
            handle.remove()

        return cache["resid_post"]

    def generate_with_ablation(
        self,
        prompt: str | list[dict],
        sae,
        feature_idx: int,
        target_layer: int,
        max_new_tokens: int = 500,
        response_split_token: str = "<start_of_turn>model",
    ) -> dict[str, str]:
            """Generate ablated and normal responses for a given prompt.

            Removes the contribution of a specific SAE feature direction at
            *target_layer* during generation by projecting it out of the
            residual stream, and returns both outputs for comparison.

            Args:
                prompt: Input text or chat-template list.
                sae: A JumpReLUSAE (or compatible) instance with ``w_dec`` attribute.
                feature_idx: Index of the SAE feature to ablate.
                target_layer: Transformer layer index at which to apply the hook.
                max_new_tokens: Maximum number of tokens to generate.
                response_split_token: Token string used to split off the model's
                    response from the full decoded output.

            Returns:
                A dict with keys:
                    - ``"ablated"``: the model response with the feature ablated.
                    - ``"normal"``: the normal (unmodified) model response.
            """
            # Handle chat template
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).to(self.model.device)

            def _run(ablate: bool) -> str:
                def ablation_hook(mod, hook_inputs, outputs):
                    output = outputs[0] if isinstance(outputs, tuple) else outputs
                    dtype = output.dtype
                    direction = sae.w_dec[feature_idx].to(dtype=dtype, device=output.device)
                    # Normalise to a unit vector for a clean projection
                    direction = direction / (direction.norm() + 1e-8)

                    # Project out the feature direction: v - (v · d) d
                    dots = torch.einsum("...d,d->...", output, direction)
                    output = output - dots.unsqueeze(-1) * direction

                    if isinstance(outputs, tuple):
                        return (output,) + outputs[1:]
                    return output

                if ablate:
                    handle = self.model.model.language_model.layers[target_layer].register_forward_hook(
                        ablation_hook
                    )
                try:
                    with torch.no_grad():
                        out_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    decoded = self.tokenizer.decode(out_ids[0])
                finally:
                    if ablate:
                        handle.remove()

                return decoded.split(response_split_token)[-1].strip()

            return {
                "normal": _run(ablate=False),
                "ablated": _run(ablate=True),
            }