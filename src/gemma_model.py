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

    def gather_feedforward_activations(
        self, target_layer: int, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and capture pre- and post-feedforward layernorm activations.

        These are the input and reconstruction-target for a transcoder at *target_layer*.

        Args:
            target_layer: Index of the transformer layer to hook.
            inputs: 1-D token IDs tensor of shape ``(n_tokens,)``.

        Returns:
            Tuple of ``(pre_ffn_acts, post_ffn_acts)``, each of shape
            ``(n_tokens, d_model)``.
        """
        cache: dict[str, torch.Tensor] = {}

        handle_pre = self.model.model.language_model.layers[target_layer].pre_feedforward_layernorm.register_forward_hook(
            partial(self._gather_acts_hook, cache=cache, key="pre_ffn")
        )
        handle_post = self.model.model.language_model.layers[target_layer].post_feedforward_layernorm.register_forward_hook(
            partial(self._gather_acts_hook, cache=cache, key="post_ffn")
        )

        try:
            with torch.no_grad():
                self.model.forward(input_ids=inputs.unsqueeze(0), use_cache=False)
        finally:
            handle_pre.remove()
            handle_post.remove()

        return cache["pre_ffn"], cache["post_ffn"]

    def generate_steered(
        self,
        prompt: Union[str, List[Dict]],
        sae,
        feature_idx: Union[int, List[int]],
        coeff: Union[float, List[float]],
        target_layer: int,
        max_new_tokens: int = 500,
        response_split_token: str = "<start_of_turn>model",
    ) -> dict:
        """Generate steered and unsteered responses for a given prompt.

        Applies activation steering along one or more SAE feature directions at
        *target_layer* during generation, and returns both the steered and
        unsteered outputs for comparison.

        Args:
            prompt: Input text or chat-template list.
            sae: A JumpReLUSAE (or compatible) instance with ``w_dec`` attribute.
            feature_idx: Index (or list of indices) of the SAE feature(s) to steer along.
            coeff: Steering coefficient (or list, one per feature). Positive amplifies
                   the feature, negative suppresses it.
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

        # Normalize to lists
        if isinstance(feature_idx, int):
            feature_idxs = [feature_idx]
            coeffs_list = [coeff] if isinstance(coeff, (int, float)) else list(coeff)
        else:
            feature_idxs = list(feature_idx)
            coeffs_list = [coeff] * len(feature_idxs) if isinstance(coeff, (int, float)) else list(coeff)

        def _run(apply_steering: bool):
            def steering_hook(mod, hook_inputs, outputs):
                if not apply_steering:
                    return outputs
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                # output shape: (batch, seq, d_model) — preserve 3D, do NOT squeeze

                dtype = output.dtype

                # Combined steering vector: sum of coeff_i * w_dec[feature_i]
                combined_vec = torch.zeros(output.shape[-1], dtype=dtype, device=output.device)
                for fi, c in zip(feature_idxs, coeffs_list):
                    combined_vec = combined_vec + c * sae.w_dec[fi].to(dtype=dtype, device=output.device)

                if output.shape[1] == 1:  # cached decode step (seq=1)
                    avg_norm = torch.norm(output, dim=-1, keepdim=True)
                    output = output + avg_norm * combined_vec
                else:  # prefill
                    avg_norm = torch.norm(output[:, -1:], dim=-1, keepdim=True)
                    output = output.clone()
                    output[:, -1:] = output[:, -1:] + avg_norm * combined_vec

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

        unsteered_text, unsteered_ids = _run(apply_steering=False)
        steered_text, steered_ids = _run(apply_steering=True)

        return {
            "unsteered": unsteered_text,
            "steered": steered_text,
            "unsteered_ids": unsteered_ids,
            "steered_ids": steered_ids,
        }

    def generate_steered_transcoder(
        self,
        prompt: Union[str, List[Dict]],
        transcoder,
        feature_idx: Union[int, List[int]],
        coeff: Union[float, List[float]],
        target_layer: int,
        max_new_tokens: int = 500,
        response_split_token: str = "<start_of_turn>model",
    ) -> dict:
        """Generate steered and unsteered responses using transcoder feature intervention.

        Unlike SAE steering (which adds a vector to the residual stream), transcoder
        steering works by substituting the entire MLP block with a modified transcoder
        forward pass:

            1. Hook ``pre_feedforward_layernorm`` to capture the MLP input ``x``.
            2. Encode: ``f = transcoder.encode(x)``  →  sparse feature activations.
            3. Steer:  ``f[..., feature_idx] += coeff``  (amplify / suppress features).
            4. Decode: ``y' = transcoder.decode(f)``  →  modified MLP output.
            5. Hook ``post_feedforward_layernorm`` to replace its output with ``y'``,
               so the residual addition uses the transcoder output instead of the real MLP.

        The **unsteered** run also routes through the transcoder (no feature
        modification) so that the only difference between the two runs is the
        feature intervention, not transcoder reconstruction error.

        Args:
            prompt: Input text or chat-template list.
            transcoder: A ``JumpReLUTranscoder`` instance (must already be on the
                correct device).
            feature_idx: Index (or list of indices) of the transcoder feature(s)
                to steer.
            coeff: Amount to add to each feature activation.  Positive values
                amplify the feature; negative values suppress it.  Can be a
                scalar (applied to all features) or a list aligned with
                ``feature_idx``.
            target_layer: Transformer layer index at which to intervene.
            max_new_tokens: Maximum number of new tokens to generate.
            response_split_token: Token used to split off the model response
                from the full decoded string.

        Returns:
            Dict with keys ``"steered"``, ``"unsteered"`` (response strings)
            and ``"steered_ids"``, ``"unsteered_ids"`` (token-ID tensors).
        """
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.model.device)

        # Normalise scalar / single-index inputs to lists
        if isinstance(feature_idx, int):
            feature_idxs = [feature_idx]
            coeffs_list = [coeff] if isinstance(coeff, (int, float)) else list(coeff)
        else:
            feature_idxs = list(feature_idx)
            coeffs_list = [coeff] * len(feature_idxs) if isinstance(coeff, (int, float)) else list(coeff)

        def _run(apply_steering: bool):
            # Cache for the pre-FFN layernorm output shared between the two hooks
            _cache: dict[str, torch.Tensor] = {}

            def pre_ffn_hook(_mod, _inp, outputs):
                # Capture the normalised input that will be fed to the MLP.
                # LayerNorm returns a plain tensor (not a tuple).
                acts = outputs[0] if isinstance(outputs, tuple) else outputs
                _cache["pre_ffn"] = acts  # shape: (batch, seq, d_model)
                return outputs

            def post_ffn_hook(_mod, _inp, outputs):
                # Replace the real MLP output with the transcoder reconstruction
                # (optionally with modified feature activations).
                pre_ffn = _cache["pre_ffn"].to(dtype=torch.float32)

                # Encode to sparse transcoder feature space
                encoded = transcoder.encode(pre_ffn)  # (batch, seq, d_sae)

                # Apply per-feature additive interventions when steering
                if apply_steering:
                    encoded = encoded.clone()
                    for fi, c in zip(feature_idxs, coeffs_list):
                        encoded[..., fi] = encoded[..., fi] + c

                # Decode back to model activation space
                transcoder_out = transcoder.decode(encoded)  # (batch, seq, d_model)

                # Match the original output dtype (bfloat16 / float16 on GPU)
                orig = outputs[0] if isinstance(outputs, tuple) else outputs
                transcoder_out = transcoder_out.to(dtype=orig.dtype)

                if isinstance(outputs, tuple):
                    return (transcoder_out,) + outputs[1:]
                return transcoder_out

            layer = self.model.model.language_model.layers[target_layer]
            handle_pre = layer.pre_feedforward_layernorm.register_forward_hook(pre_ffn_hook)
            handle_post = layer.post_feedforward_layernorm.register_forward_hook(post_ffn_hook)

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
                handle_pre.remove()
                handle_post.remove()

            text = decoded.split(response_split_token)[-1].strip()
            return text, out_ids[0]

        unsteered_text, unsteered_ids = _run(apply_steering=False)
        steered_text, steered_ids = _run(apply_steering=True)

        return {
            "unsteered": unsteered_text,
            "steered": steered_text,
            "unsteered_ids": unsteered_ids,
            "steered_ids": steered_ids,
        }

    def generate_ablated_transcode(
        self,
        prompt: Union[str, List[Dict]],
        transcoder,
        feature_idx: Union[int, List[int]],
        target_layer: int,
        max_new_tokens: int = 500,
        response_split_token: str = "<start_of_turn>model",
    ) -> dict:
        """Ablate transcoder feature(s) for the entire generation.

        Both runs route through the transcoder (not the original MLP) so the
        only difference is the targeted feature ablation.  The specified
        features are zeroed out for every forward pass during generation.

        Args:
            prompt: Input text or chat-template list.
            transcoder: A ``JumpReLUTranscoder`` instance.
            feature_idx: Index (or list of indices) of the feature(s) to ablate.
            target_layer: Transformer layer index at which to intervene.
            max_new_tokens: Maximum number of new tokens to generate.
            response_split_token: Token used to split off the model response.

        Returns:
            Dict with keys ``"unablated"``, ``"ablated"`` (response strings)
            and ``"unablated_ids"``, ``"ablated_ids"`` (token-ID tensors).
        """
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.model.device)

        # Normalise to list
        feature_idxs = [feature_idx] if isinstance(feature_idx, int) else list(feature_idx)

        def _run(apply_ablation: bool):
            _cache: dict[str, torch.Tensor] = {}

            # -- Transcoder hooks ----------------------------------------------
            def pre_ffn_hook(_mod, _inp, outputs):
                acts = outputs[0] if isinstance(outputs, tuple) else outputs
                _cache["pre_ffn"] = acts
                return outputs

            def post_ffn_hook(_mod, _inp, outputs):
                pre_ffn = _cache["pre_ffn"].to(dtype=torch.float32)
                encoded = transcoder.encode(pre_ffn)

                # Zero-out target features for the entire generation
                if apply_ablation:
                    encoded = encoded.clone()
                    for fi in feature_idxs:
                        encoded[..., fi] = 0.0

                transcoder_out = transcoder.decode(encoded)

                orig = outputs[0] if isinstance(outputs, tuple) else outputs
                transcoder_out = transcoder_out.to(dtype=orig.dtype)

                if isinstance(outputs, tuple):
                    return (transcoder_out,) + outputs[1:]
                return transcoder_out

            layer = self.model.model.language_model.layers[target_layer]
            handle_pre = layer.pre_feedforward_layernorm.register_forward_hook(
                pre_ffn_hook
            )
            handle_post = layer.post_feedforward_layernorm.register_forward_hook(
                post_ffn_hook
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
                handle_pre.remove()
                handle_post.remove()

            text = decoded.split(response_split_token)[-1].strip()
            return text, out_ids[0]

        unablated_text, unablated_ids = _run(apply_ablation=False)
        ablated_text, ablated_ids = _run(apply_ablation=True)

        return {
            "unablated": unablated_text,
            "ablated": ablated_text,
            "unablated_ids": unablated_ids,
            "ablated_ids": ablated_ids,
        }