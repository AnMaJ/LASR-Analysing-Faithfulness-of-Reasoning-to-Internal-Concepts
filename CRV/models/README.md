# Models and Transcoders

The trained per-layer transcoders for this project are hosted on [HuggingFace](https://huggingface.co/facebook/crv-8b-instruct-transcoders).

## Prerequisites

To run the model with transcoders, you need the [Circuit Tracer](https://github.com/safety-research/circuit-tracer) library.
It can be installed from [this](https://github.com/zsquaredz/circuit-tracer) project page.


**Important**: You must install the version linked above. The upstream library does not currently support the Top-K Transcoders used in our work.

## Usage

After installing the library, you can download and load the Llama 3.1 8B Instruct model with the attached transcoders using the `ReplacementModel` class.

```python
import torch
from circuit_tracer import ReplacementModel

# Automatically downloads base model and transcoders from HF
model = ReplacementModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    "facebook/crv-8b-instruct-transcoders",
    dtype=torch.bfloat16
)

print("Model loaded successfully with Transcoders.")
```

## Next Steps
Once the model is loaded, you can perform attribution or intervention.

- **Demos**: See the [official circuit-tracer demo](https://github.com/safety-research/circuit-tracer/blob/main/demos/llama_demo.ipynb) for general usage patterns.
- **CRV Implementation**: We provide the specific scripts used to generate our attribution graphs in the `attribution/` folder of this repository.
