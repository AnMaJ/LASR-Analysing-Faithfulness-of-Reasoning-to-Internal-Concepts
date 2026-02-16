from typing import Dict, List, Union, Tuple
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs import ModelConfig


class GemmaModel:
    """Wrapper around a Gemma causal-LM for easy loading and generation."""

    def __init__(self, config: ModelConfig | None = None):
        """Load the model and tokenizer specified by *config*.

        Args:
            config: A ``ModelConfig`` instance. Uses the defaults
                (``google/gemma-3-4b-it`` on the best available device)
                when ``None``.
        """
        if config is None:
            config = ModelConfig()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device,
            dtype=torch.bfloat16,
        )
        self.model.eval()

    def generate(self, prompt: Union[str, List[Dict]], max_new_tokens: int = 256) -> Tuple[str, torch.Tensor]:
        """Generate a response for the given *prompt*.

        Args:
            prompt: The input text to send to the model (it can be with the chat template)
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The model's generated text (excluding the original prompt).
        """
        # Handle chat template
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
        # Strip the prompt tokens from the output
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]

        return prompt, self.tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids

    def generate_batch(
        self,
        prompts: list[str],
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Run batched generation over *prompts* and return decoded outputs.

        Args:
            prompts: List of input texts to send to the model.
            batch_size: Number of prompts to process at once.
            max_new_tokens: Maximum number of tokens to generate per prompt.

        Returns:
            A list of generated texts (excluding the original prompts).
        """
        decoded_outputs: list[str] = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded_outputs.extend(
                self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            )

            del inputs, output_tokens
            torch.cuda.empty_cache()

        print(f"Generated {len(decoded_outputs)} outputs")
        return decoded_outputs

    @staticmethod
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
        self, target_layer: int, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Run a forward pass and capture the residual stream output at *target_layer*."""
        cache: dict[str, torch.Tensor] = {}

        handle = self.model.model.language_model.layers[target_layer].register_forward_hook(
            partial(self._gather_acts_hook, cache=cache, key="resid_post", use_input=False)
        )

        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        try:
            self.model.forward(input_ids=inputs)
        finally:
            handle.remove()

        return cache["resid_post"]
