# Attribution Graph Construction
This directory contains scripts to construct attribution graphs using the [Circuit Tracer](https://github.com/safety-research/circuit-tracer) library.

## Prerequisites
1. **Custom Library Fork:** You must install the version of circuit-tracer that supports Top-K Transcoders. You can use this [fork](https://github.com/zsquaredz/circuit-tracer) of the library.
```bash
git clone https://github.com/zsquaredz/circuit-tracer.git
cd circuit-tracer
pip install .
```
2. **Transcoders:** Ensure you have downloaded the transcoders (see `../models/README.md)`.

*Note: You can also use other transcoders that is supported by the Circuit Tracer library.*

## 1. Interactive Usage

The following example (adapted from [this demo](https://github.com/safety-research/circuit-tracer/blob/main/demos/attribute_demo.ipynb)) demonstrates how to load the model, attribute a single prompt, and prune the graph for analysis.

### Step A: Run Attribution

```python
import torch
from pathlib import Path
from circuit_tracer import ReplacementModel, attribute


# 1. Load Model & Transcoders
model = ReplacementModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    "facebook/crv-8b-instruct-transcoders",
    dtype=torch.bfloat16
)

# 2. Configuration
prompt = "The capital of state containing Dallas is"
max_n_logits = 10   # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 8192  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size=256  # Batch size when attributing
offload='disk' if IN_COLAB else 'cpu' # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
verbose = True  # Whether to display a tqdm progress bar and timing report

# 3. Run attribution and generate the graph
graph = attribute(
    prompt=prompt,
    model=model,
    max_n_logits=max_n_logits,
    desired_logit_prob=desired_logit_prob,
    batch_size=batch_size,
    max_feature_nodes=max_feature_nodes,
    offload=offload,
    verbose=verbose
)

# 4. Save the raw graph
graph_dir = Path('graphs')
graph_dir.mkdir(exist_ok=True)
graph.to_pt(graph_dir / 'example_graph.pt')
```

### Step B: Prune the Graph and Visualization
Once the graph is computed, you can prune it for visualization.
```python
from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve
from IPython.display import IFrame

# 1. Prune and export for visualization
create_graph_files(
    graph_or_path=graph_dir / 'example_graph.pt',
    slug="dallas-austin",
    output_path='./graph_files',
    node_threshold=0.8,  # Keep nodes contributing to top 80% influence
    edge_threshold=0.98  # Keep edges contributing to top 98% influence
)

# 2. Serve visualization
port = 8046
server = serve(data_dir='./graph_files/', port=port)
print(f"Open your graph here: http://localhost:{port}/index.html")

# (Optional) Display in Notebook
display(IFrame(src=f'http://localhost:{port}/index.html', width='100%', height='800px'))
```

## 2. Batch Usage
To reproduce the experiments in the paper, we provide the `circuit_tracing.py` script to run attribution over the full datasets in batches.

```bash
python circuit_tracing.py \
  --expressions_json /path/to/your/json/files \
  --graph_name_prefix graph_prefix \
  --slug_prefix arith \
  --start_idx 0 \
  --end_idx 100 \
  --max_n_logits 10 \
  --desired_logit_prob 0.95 \
  --max_feature_nodes 4096 \
  --batch_size 16 \
  --offload none \
  --graph_dir ./graph \
  --graph_file_dir ./graph_files \
  --before_after before
```
### Argument Description
- `--expressions_json`: Path to the JSONL file containing the CoT steps (generated in the `processing/` stage).
- `--graph_name_prefix`: This is the prefix you can add to the graph file which can be used to identify the graph.
- `--start_idx` / `--end_idx`: Controls the range of examples to process. Attribution is computationally expensive; we recommend running this in parallel batches (e.g., via Slurm arrays).
- `--graph_dir`: Directory where raw `.pt` graph objects will be saved.
- `--graph_file_dir`: Directory where pruned visualization files will be saved.
- `--before_after`: Controls the token position used for attribution (see **Appendix B.2** of the paper).
  - `after`: Attributes from the last token of the current step.
  - `before`: Attributes from the last token of the previous step.
  - `both`: Saves graphs for both positions.

The remaining arguments (`max_n_logits`, `desired_logit_prob`, etc.) follow the standard circuit-tracer API.
