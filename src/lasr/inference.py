from __future__ import annotations

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import InferenceConfig, ModelConfig


def load_model(config: ModelConfig):
    """Load and return (model, tokenizer) for the given ModelConfig."""
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, tokenizer


def generate_predictions(
    prompts: list[str],
    model,
    tokenizer,
    config: InferenceConfig,
    device: str = "cuda",
) -> list[str]:
    """Run batched generation over *prompts* and return decoded outputs."""
    decoded_outputs: list[str] = []
    for i in tqdm(range(0, len(prompts), config.batch_size), desc="Generating"):
        batch = prompts[i : i + config.batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        output_tokens = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_outputs.extend(
            tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        )

        del inputs, output_tokens
        torch.cuda.empty_cache()

    print(f"Generated {len(decoded_outputs)} outputs")
    return decoded_outputs
