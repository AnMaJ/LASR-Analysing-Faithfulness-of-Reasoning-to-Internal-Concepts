from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch
from safetensors.torch import load_file
import torch.nn as nn
from functools import partial
import numpy as np
from IPython.display import display, HTML
import textwrap


def load_model_and_tokenizer(model_name: str, use_4bit: bool = False):
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    )
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


load_model_and_tokenizer("google/gemma-3-4b-it")
