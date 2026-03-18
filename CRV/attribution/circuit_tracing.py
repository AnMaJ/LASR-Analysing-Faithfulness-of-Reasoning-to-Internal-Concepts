# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This script is heavily based on this demo: https://github.com/safety-research/circuit-tracer/blob/main/demos/attribute_demo.ipynb

import sys
import os
from collections import namedtuple
from functools import partial
import torch
from circuit_tracer import ReplacementModel
from pathlib import Path
from typing import Union, List
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
import json
import argparse

# Below is the model we have used for the paper, you can also load any other transcoders/LLMs you like
model = ReplacementModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", "facebook/crv-8b-instruct-transcoders", dtype=torch.bfloat16)

def generate_graph(prompt: str, graph_name: str, slug_name: str, graph_dir: Union[str, Path], graph_file_dir: Union[str, Path], node_threshold: float = 0.8, edge_threshold: float = 0.98):
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
    graph_name = f'{graph_name}.pt'
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(exist_ok=True)
    graph_path = graph_dir / graph_name

    graph.to_pt(graph_path)

    slug = slug_name
    create_graph_files(
        graph_or_path=graph_path,
        slug=slug,
        output_path=graph_file_dir,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold
    )


def process_expressions_and_generate_graphs(expressions_json_path, graph_name_prefix, slug_prefix, start_idx=0, end_idx=None, node_threshold=0.8, edge_threshold=0.98, before_after='both'):
    with open(expressions_json_path, 'r') as f:
        expressions = json.load(f)
    if end_idx is None or end_idx > len(expressions):
        end_idx = len(expressions)
    for expr in expressions[start_idx:end_idx]:
        for step in expr['step_expressions']:
            # Generate 'before' graph
            if before_after in ['before', 'both']:
                before = step.get('formatted_assistant_content_before', '')
                prompt = before
                graph_name = f"{graph_name_prefix}_expr{expr['expression_id']}_step{step['step_number']}_before"
                slug = f"{slug_prefix}_expr{expr['expression_id']}_step{step['step_number']}_before"

                try:
                    generate_graph(prompt, graph_name, slug, graph_dir=graph_dir, graph_file_dir=graph_file_dir, node_threshold=node_threshold, edge_threshold=edge_threshold)
                    print(f"Generated 'before' graph for expression {expr['expression_id']} step {step['step_number']}")
                except Exception as e:
                    print(f"Error generating 'before' graph for expression {expr['expression_id']} step {step['step_number']}: {e}")
                    pass

            # Generate 'after' graph
            if before_after in ['after', 'both']:
                after = step.get('formatted_assistant_content_after', '')
                prompt = after
                graph_name = f"{graph_name_prefix}_expr{expr['expression_id']}_step{step['step_number']}_after"
                slug = f"{slug_prefix}_expr{expr['expression_id']}_step{step['step_number']}_after"
                try:
                    generate_graph(prompt, graph_name, slug, graph_dir=graph_dir, graph_file_dir=graph_file_dir, node_threshold=node_threshold, edge_threshold=edge_threshold)
                    print(f"Generated 'after' graph for expression {expr['expression_id']} step {step['step_number']}")
                except Exception as e:
                    print(f"Error generating 'after' graph for expression {expr['expression_id']} step {step['step_number']}: {e}")
                    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process expressions and generate circuit graphs.')
    parser.add_argument('--expressions_json', type=str, required=True, help='Path to expressions JSON file')
    parser.add_argument('--graph_name_prefix', type=str, required=True, help='Prefix for graph file names')
    parser.add_argument('--slug_prefix', type=str, required=True, help='Prefix for slug names')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for batching')
    parser.add_argument('--end_idx', type=int, default=None, help='End index for batching')
    parser.add_argument('--before_after', type=str, default='both', choices=['before', 'after', 'both'],
                       help='Which graphs to generate: before, after, or both (default: both)')
    parser.add_argument('--max_n_logits', type=int, default=10)
    parser.add_argument('--desired_logit_prob', type=float, default=0.95)
    parser.add_argument('--max_feature_nodes', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--offload', type=str, default='disk', choices=['disk', 'cpu', 'none'])
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--graph_dir', type=str, default='./graphs/')
    parser.add_argument('--graph_file_dir', type=str, default='./graph_files/')
    parser.add_argument('--node_threshold', type=float, default=0.8, help='Node threshold for create_graph_files')
    parser.add_argument('--edge_threshold', type=float, default=0.98, help='Edge threshold for create_graph_files')
    return parser.parse_args()

args = None

# Parse arguments if run as main
if __name__ == "__main__":
    args = parse_args()
    # Set config variables from args
    max_n_logits = args.max_n_logits
    desired_logit_prob = args.desired_logit_prob
    max_feature_nodes = args.max_feature_nodes
    batch_size = args.batch_size
    offload = args.offload if args.offload != 'none' else None
    verbose = args.verbose
    graph_dir = args.graph_dir
    graph_file_dir = args.graph_file_dir
    node_threshold = args.node_threshold
    edge_threshold = args.edge_threshold
    process_expressions_and_generate_graphs(
        args.expressions_json,
        args.graph_name_prefix,
        args.slug_prefix,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        before_after=args.before_after
    )
