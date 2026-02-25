from __future__ import annotations

from functools import partial
from typing import Dict, List, Union

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
        load_kwargs = {"device_map": config.device}
        if config.torch_dtype is not None:
            load_kwargs["torch_dtype"] = config.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
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
        # TODO in case it is needed, make it usable for batch of prompt
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
                # Strip only the left-padding, keep prompt + generation
                ids = output_ids[j, int(padding_len[j]):]
                all_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
                all_ids.append(ids)
                all_prompt_lens.append(int(prompt_lens[j]))

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
            self.model.forward(input_ids=inputs.unsqueeze(0))
        finally:
            handle.remove()

        return cache["resid_post"]
    
    def generate_steered(
        self,
        prompt: Union[str, List[Dict]],
        sae,
        feature_idx: int,
        coeff: float,
        target_layer: int,
        max_new_tokens: int = 500,
        response_split_token: str = "<start_of_turn>model",
    ) -> dict:
        """Generate steered and unsteered responses for a given prompt.

        Applies activation steering along a given SAE feature direction at
        *target_layer* during generation, and returns both the steered and
        unsteered outputs for comparison.

        Args:
            prompt: Input text or chat-template list.
            sae: A JumpReLUSAE (or compatible) instance with ``w_dec`` attribute.
            feature_idx: Index of the SAE feature to steer along.
            coeff: Steering coefficient. Positive amplifies the feature,
                   negative suppresses it. Use 0.0 to get unsteered output only.
            target_layer: Transformer layer index at which to apply the hook.
            max_new_tokens: Maximum number of tokens to generate.
            response_split_token: Token string used to split off the model's
                response from the full decoded output.

        Returns:
            A dict with keys ``"steered"``, ``"unsteered"`` (response strings)
            and ``"steered_ids"``, ``"unsteered_ids"`` (token-ID tensors).
        """
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.model.device)

        def _run(steering_coeff: float):
            def steering_hook(mod, hook_inputs, outputs):
                output = outputs[0] if isinstance(outputs, tuple) else outputs

                if output.dim() == 3:
                    output = output.squeeze(0)

                dtype = output.dtype
                steering_vec = sae.w_dec[feature_idx].to(dtype=dtype, device=output.device)

                if output.shape[0] == 1:  # cached decode step
                    avg_norm = torch.norm(output, dim=-1, keepdim=True)
                    output = output + steering_coeff * avg_norm * steering_vec
                else:  # prefill
                    avg_norm = torch.norm(output[-1:], dim=-1, keepdim=True)
                    output = output.clone()
                    output[-1:] = output[-1:] + steering_coeff * avg_norm * steering_vec

                if isinstance(outputs, tuple):
                    return (output,) + outputs[1:]
                return output

            handle = self.model.model.language_model.layers[target_layer].register_forward_hook(
                steering_hook
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
                handle.remove()

            text = decoded.split(response_split_token)[-1].strip()
            return text, out_ids[0]

        unsteered_text, unsteered_ids = _run(steering_coeff=0.0)
        steered_text, steered_ids = _run(steering_coeff=coeff)

        return {
            "unsteered": unsteered_text,
            "steered": steered_text,
            "unsteered_ids": unsteered_ids,
            "steered_ids": steered_ids,
        }


