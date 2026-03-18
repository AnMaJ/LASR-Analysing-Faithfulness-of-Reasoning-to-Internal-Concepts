# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Module for feature extraction

Load circuit graphs and step labels, prepare data for visualization.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Import circuit_tracer components for advanced feature extraction
from circuit_tracer.graph import Graph, prune_graph

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before any multiprocessing operations
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


class GraphDataset(Dataset):
    """
    Dataset class for loading circuit graphs in parallel.
    """

    def __init__(self, graph_paths: List[Dict], feature_type: str = 'adjacency_matrix', use_cpu: bool = True):
        """
        Initialize the dataset.

        Args:
            graph_paths: List of dictionaries containing graph metadata
            feature_type: Type of feature to extract
            use_cpu: Whether to load tensors to CPU (True) or GPU (False)
        """
        self.graph_paths = graph_paths
        self.feature_type = feature_type
        self.use_cpu = use_cpu

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        """
        Load a single graph and extract features.

        Args:
            idx: Index of the graph to load

        Returns:
            Dictionary containing features and metadata
        """
        item = self.graph_paths[idx]
        graph_path = item['graph_path']

        try:
            # Load graph to CPU or GPU based on flag
            if self.use_cpu:
                graph = torch.load(graph_path, weights_only=False, map_location='cpu')
            else:
                graph = torch.load(graph_path, weights_only=False)


            # Extract features
            features = self._extract_single_feature(graph, graph_path, self.feature_type)

            # Explicitly delete graph to free memory
            del graph

            if features is not None:
                return {
                    'expr_id': item['expr_id'],
                    'step_number': item['step_number'],
                    'before_after': item['before_after'],
                    'features': features,
                    'step_labels': item['step_labels'],
                    'original_expression': item['original_expression'],
                    'success': True
                }
            else:
                return {
                    'expr_id': item['expr_id'],
                    'step_number': item['step_number'],
                    'before_after': item['before_after'],
                    'success': False
                }
        except Exception as e:
            print(f"Error loading {graph_path}: {e}")
            return {
                'expr_id': item['expr_id'],
                'step_number': item['step_number'],
                'before_after': item['before_after'],
                'success': False
            }

    def _extract_single_feature(self, graph: dict, graph_path: str, feature_type: str) -> Optional[np.ndarray]:
        """
        Extract features from a single graph.

        Args:
            graph: Graph dictionary
            feature_type: Type of feature to extract

        Returns:
            Feature array or None if not found
        """
        if feature_type == 'adjacency_matrix':
            adj_matrix = graph.get('adjacency_matrix', torch.tensor([]))
            if adj_matrix.numel() > 0:
                # Extract to numpy and delete tensor immediately
                features = adj_matrix.flatten().cpu().numpy()
                del adj_matrix  # Explicit cleanup
                return features

        elif feature_type == 'active_features':
            active_features = graph.get('active_features', torch.tensor([]))
            if active_features.numel() > 0:
                features = active_features.flatten().cpu().numpy()
                del active_features  # Explicit cleanup
                return features

        elif feature_type == 'activation_values':
            activation_values = graph.get('activation_values', torch.tensor([]))
            if activation_values.numel() > 0:
                features = activation_values.flatten().cpu().numpy()
                del activation_values  # Explicit cleanup
                return features

        elif feature_type == 'selected_features':
            selected_features = graph.get('selected_features', torch.tensor([]))
            if selected_features.numel() > 0:
                features = selected_features.flatten().cpu().numpy()
                del selected_features
                return features

        elif feature_type == 'advanced_graph_features':
            graph_obj = Graph.from_pt(graph_path)
            # Use the advanced feature extraction function
            return self._extract_advanced_features(graph_obj)

        return None


    def _extract_advanced_features(self, graph: 'Graph', node_threshold: float = 0.8) -> np.ndarray:
        """
        Extracts a flat, fixed-size feature vector from a circuit-tracer Graph object.

        Args:
            graph: Circuit-tracer Graph object
            node_threshold: Threshold for pruning

        Returns:
            Feature vector as numpy array
        """
        # Ensure graph tensors are on the CPU for processing
        graph.to("cpu")

        # Prune the graph to focus on influential components
        node_mask, _, cumulative_scores = prune_graph(graph, node_threshold=node_threshold)

        features = []
        n_layers = graph.cfg.n_layers
        n_pos = graph.n_pos

        # --- Level 1: High-Level Stats ---
        pruned_node_indices = node_mask.nonzero().squeeze().tolist()
        if isinstance(pruned_node_indices, int):
            pruned_node_indices = [pruned_node_indices]

        n_features_total = len(graph.selected_features)
        n_error_nodes_total = n_layers * n_pos

        pruned_feature_nodes = [i for i in pruned_node_indices if i < n_features_total]
        pruned_error_nodes = [i for i in pruned_node_indices if n_features_total <= i < n_features_total + n_error_nodes_total]

        features.append(graph.activation_values.shape[0]) # Total active features
        features.append(len(pruned_feature_nodes))       # Pruned feature node count
        features.append(len(pruned_error_nodes))         # Pruned error node count

        # Logit stats
        if graph.logit_probabilities.numel() > 0:
            probs = graph.logit_probabilities
            features.append(probs[0].item()) # Top logit probability
            features.append(-torch.sum(probs * torch.log(probs + 1e-8)).item()) # Logit entropy (add small epsilon)
        else:
            features.extend([0.0, 0.0])

        # --- Level 2: Aggregated Node Stats ---
        node_influences = cumulative_scores[node_mask]
        features.append(node_influences.mean().item()) # Mean node influence

        if len(pruned_error_nodes) > 0:
            error_influences = cumulative_scores[pruned_error_nodes]
            features.append(error_influences.sum().item()) # Total error influence
            features.append(error_influences.mean().item())
        else:
            features.extend([0.0, 0.0])

        # Activation stats for pruned feature nodes
        if len(pruned_feature_nodes) > 0:
            selected_indices_for_pruned_nodes = [graph.selected_features[i] for i in pruned_feature_nodes]
            pruned_activations = graph.activation_values[selected_indices_for_pruned_nodes]
            if pruned_activations.numel() > 0:
                features.append(pruned_activations.mean().item())
                features.append(pruned_activations.max().item())
                features.append(pruned_activations.std().item())
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Layer-wise feature counts (histogram)
        layer_counts = [0] * n_layers
        for node_idx in pruned_feature_nodes:
            if node_idx < len(graph.selected_features):
                feature_idx = graph.selected_features[node_idx]
                if feature_idx < len(graph.active_features):
                    layer, _, _ = graph.active_features[feature_idx].tolist()
                    if layer < n_layers:
                        layer_counts[layer] += 1
        features.extend(layer_counts)

        # --- Level 3: NEW Topological and Edge-Based Features ---
        # Call the new helper function
        topo_features_dict = self._extract_topological_and_edge_features(pruned_node_indices, graph)

        # Append the new features in a fixed, reliable order
        feature_keys_ordered = [
            'sum_edge_weights', 'mean_edge_weights', 'std_edge_weights', 'n_edges_pruned',
            'graph_density', 'n_connected_components', 'mean_degree_centrality',
            'max_degree_centrality', 'mean_betweenness_centrality', 'max_betweenness_centrality',
            'avg_shortest_path_length', 'input_to_logit_shortest_path'
        ]
        for key in feature_keys_ordered:
            features.append(topo_features_dict[key])

        return np.array(features)

    def _extract_topological_and_edge_features(
        self,
        pruned_indices: List[int],
        graph: Graph
    ) -> Dict[str, float]:
        """
        Calculates edge-based and topological features for a pruned subgraph.

        Args:
            pruned_indices: A list of integer indices for the nodes in the pruned graph.
            graph: The full Graph object, used to access the adjacency matrix and metadata.

        Returns:
            A dictionary of calculated features.
        """
        features = {}
        num_nodes_pruned = len(pruned_indices)

        # Define default values for all features to ensure a fixed vector size.
        # This is crucial if the pruned graph is too small for meaningful stats.
        default_features = {
            'sum_edge_weights': 0.0, 'mean_edge_weights': 0.0, 'std_edge_weights': 0.0,
            'n_edges_pruned': 0, 'graph_density': 0.0, 'n_connected_components': float(num_nodes_pruned),
            'mean_degree_centrality': 0.0, 'max_degree_centrality': 0.0,
            'mean_betweenness_centrality': 0.0, 'max_betweenness_centrality': 0.0,
            'avg_shortest_path_length': -1.0, # Use -1 to indicate not computed
            'input_to_logit_shortest_path': -1.0 # Use -1 to indicate no path found
        }

        if num_nodes_pruned < 2:
            return default_features

        # --- 1. Create the pruned subgraph from the full adjacency matrix ---
        full_adj_matrix = graph.adjacency_matrix
        pruned_adj_matrix = full_adj_matrix[np.ix_(pruned_indices, pruned_indices)]

        # Create a NetworkX Graph object for topological analysis.
        # We create a directed graph as influence is directional.
        G = nx.from_numpy_array(pruned_adj_matrix.numpy(), create_using=nx.DiGraph)

        # --- 2. Calculate Edge-Based Features ---
        # `get_edge_attributes` is a clean way to get all weights.
        edge_weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))

        if edge_weights.size > 0:
            features['sum_edge_weights'] = np.sum(edge_weights)
            features['mean_edge_weights'] = np.mean(edge_weights)
            features['std_edge_weights'] = np.std(edge_weights)
        else:
            features['sum_edge_weights'] = 0.0
            features['mean_edge_weights'] = 0.0
            features['std_edge_weights'] = 0.0

        # --- 3. Calculate Topological and Structural Features ---
        # Number of paths (interpreted as number of active edges)
        features['n_edges_pruned'] = G.number_of_edges()
        features['graph_density'] = nx.density(G)

        # For directed graphs, we consider weakly connected components.
        features['n_connected_components'] = nx.number_weakly_connected_components(G)

        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight') # Weighted is more meaningful

        features['mean_degree_centrality'] = np.mean(list(degree_centrality.values()))
        features['max_degree_centrality'] = np.max(list(degree_centrality.values())) if degree_centrality else 0.0
        features['mean_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
        features['max_betweenness_centrality'] = np.max(list(betweenness_centrality.values())) if betweenness_centrality else 0.0

        # --- 4. Calculate Shortest Path Features ---
        # Average shortest path length for the largest connected component
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        if len(largest_cc) > 1:
            subgraph = G.subgraph(largest_cc)
            try:
                features['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph, weight='weight')
            except nx.NetworkXError:
                features['avg_shortest_path_length'] = -1.0 # Disconnected component
        else:
            features['avg_shortest_path_length'] = -1.0

        # --- 5. Calculate specific "Input Token -> Logit" shortest path ---
        # First, identify the global indices for each node type
        n_features_total = len(graph.selected_features)
        n_error_nodes_total = graph.cfg.n_layers * graph.n_pos
        start_token_nodes = n_features_total + n_error_nodes_total
        end_token_nodes = start_token_nodes + graph.n_pos
        start_logit_nodes = end_token_nodes

        # Create a mapping from global node index to the local index in the pruned graph `G`
        global_to_local_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(pruned_indices)}

        # Find which input tokens and logit nodes are present in our pruned graph
        pruned_input_nodes_local = [
            global_to_local_idx_map[i] for i in pruned_indices
            if start_token_nodes <= i < end_token_nodes
        ]
        pruned_logit_nodes_local = [
            global_to_local_idx_map[i] for i in pruned_indices
            if i >= start_logit_nodes
        ]

        min_path_len = float('inf')
        if pruned_input_nodes_local and pruned_logit_nodes_local:
            for source_node in pruned_input_nodes_local:
                for target_node in pruned_logit_nodes_local:
                    if nx.has_path(G, source=source_node, target=target_node):
                        path_len = nx.shortest_path_length(G, source=source_node, target=target_node, weight='weight')
                        if path_len < min_path_len:
                            min_path_len = path_len

        features['input_to_logit_shortest_path'] = min_path_len if min_path_len != float('inf') else -1.0

        # Ensure all keys are present before returning
        for key in default_features:
            if key not in features:
                features[key] = default_features[key]

        return features


class CircuitAnalyzer:
    """
    Simple class to load circuit graphs and step labels for visualization.
    """

    def __init__(self, graph_dir: str = "/graphs/"):
        self.graph_dir = Path(graph_dir)
        self.data = []  # List to store all loaded data

    def load_all_data(self, expressions_json_path: str, graph_name_prefix: str,
                      feature_type: str = 'adjacency_matrix',
                      before_after: str = 'both',
                      num_workers: int = 4,
                      batch_size: int = 32,
                      use_cpu: bool = True):
        """
        Load all available graphs and their corresponding step labels in parallel.
        Extract features immediately to avoid memory issues.

        Args:
            expressions_json_path: Path to expressions JSON file
            graph_name_prefix: Prefix for graph file names
            feature_type: Type of feature to extract ('adjacency_matrix', 'active_features', etc.)
            before_after: Which graphs to load ('before', 'after', or 'both')
            num_workers: Number of parallel workers for loading
            batch_size: Batch size for DataLoader
            use_cpu: Whether to load tensors to CPU (True) or GPU (False)
        """
        # Load expressions
        with open(expressions_json_path, 'r') as f:
            expressions = json.load(f)

        print(f"Loading data from {len(expressions)} expressions...")
        print(f"Extracting {feature_type} features from {before_after} graphs...")
        print(f"Using {num_workers} workers with batch size {batch_size}")
        print(f"Loading tensors to {'CPU' if use_cpu else 'GPU'}...")

        # Clear GPU cache at start if using GPU
        if not use_cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Determine which graph types to load
        if before_after == 'both':
            graph_types = ['before', 'after']
        else:
            graph_types = [before_after]

        # Prepare list of all graph paths to load
        graph_paths = []
        print("Scanning for available graph files...")

        for expr in tqdm(expressions, desc="Scanning expressions"):
            expr_id = expr['expression_id']

            for step in expr['step_expressions']:
                step_number = step['step_number']

                # Check which graph types exist
                for graph_type in graph_types:
                    graph_path = self._get_graph_path(graph_name_prefix, expr_id, step_number, graph_type)

                    if graph_path.exists():
                        graph_paths.append({
                            'graph_path': graph_path,
                            'expr_id': expr_id,
                            'step_number': step_number,
                            'before_after': graph_type,
                            'step_labels': step,
                            'original_expression': expr['original_expression']
                        })

        print(f"Found {len(graph_paths)} graph files to load")

        # Create dataset and dataloader
        dataset = GraphDataset(graph_paths, feature_type, use_cpu)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

        # Load data in parallel with progress bar
        self.data = []
        total_batches = len(dataloader)

        with tqdm(total=len(graph_paths), desc="Loading graphs") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # Add successful items to data
                batch_success_count = 0
                for item in batch:
                    if item['success']:
                        self.data.append({
                            'expr_id': item['expr_id'],
                            'step_number': item['step_number'],
                            'before_after': item['before_after'],
                            'features': item['features'],
                            'step_labels': item['step_labels'],
                            'original_expression': item['original_expression']
                        })
                        batch_success_count += 1

                # Update progress bar
                pbar.update(len(batch))
                pbar.set_postfix({
                    'batch': f"{batch_idx + 1}/{total_batches}",
                    'loaded': len(self.data),
                    'batch_success': batch_success_count
                })

        print(f"Successfully loaded {len(self.data)} graph files")

        # Print loading summary by expression
        self._print_loading_summary()

        # Automatically save the processed data
        # self.save_data()

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of items from dataset

        Returns:
            List of items (no batching needed for this use case)
        """
        return batch

    def _print_loading_summary(self):
        """Print summary of loaded graphs by expression."""
        if not self.data:
            return

        # Count graphs per expression
        expr_counts = {}
        for item in self.data:
            expr_id = item['expr_id']
            if expr_id not in expr_counts:
                expr_counts[expr_id] = 0
            expr_counts[expr_id] += 1

        # Print summary for each expression
        for expr_id in sorted(expr_counts.keys()):
            print(f"Expression {expr_id}: loaded {expr_counts[expr_id]} graphs")


    def _get_graph_path(self, graph_name_prefix: str, expr_id: int, step_number: int, before_after: str) -> Path:
        """Construct path to graph file."""
        graph_name = f"{graph_name_prefix}_expr{expr_id}_step{step_number}_{before_after}.pt"
        return self.graph_dir / graph_name

    def extract_features(self) -> np.ndarray:
        """
        Get the pre-extracted features from all loaded data.

        Returns:
            Feature matrix (n_samples, n_features)
        """
        if not self.data:
            print("No data loaded")
            return np.array([])

        features = [item['features'] for item in self.data]

        # Pad features to same length if needed
        max_len = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                padded_features.append(padded)
            else:
                padded_features.append(f[:max_len])  # Truncate if too long

        return np.array(padded_features)

    def visualize_with_pca(self, save_path: Optional[str] = None, n_components: int = 2):
        """
        Visualize features using PCA.

        Args:
            n_components: Number of PCA components
        """

        features = self.extract_features()
        print(f"Extracted features shape: {features.shape}")
        if features.size == 0:
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(X_scaled)

        # Create visualization
        plt.figure(figsize=(10, 8))

        # Color by step labels
        colors = []
        for item in self.data[:len(reduced_features)]:
            step_label = item['step_labels'].get('step_label', False)
            colors.append('red' if step_label else 'blue')

        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=colors, alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.title(f'PCA of features (Red=True, Blue=False)')
        # plt.show()
        plt.savefig(save_path if save_path else 'pca_visualization.png')

        return reduced_features, pca

    def visualize_with_tsne(self, save_path: Optional[str] = None, perplexity: int = 30):
        """
        Visualize features using t-SNE.

        Args:
            perplexity: t-SNE perplexity parameter
        """

        features = self.extract_features()
        print(f"Extracted features shape: {features.shape}")
        if features.size == 0:
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_features = tsne.fit_transform(X_scaled)

        # Create visualization
        plt.figure(figsize=(10, 8))

        # Color by step labels
        colors = []
        for item in self.data[:len(reduced_features)]:
            step_label = item['step_labels'].get('step_label', False)
            colors.append('red' if step_label else 'blue')

        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=colors, alpha=0.6)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f't-SNE of features (Red=True, Blue=False)')
        # plt.show()
        plt.savefig(save_path if save_path else 'tsne_visualization.png')

        return reduced_features

    def save_data(self, save_path: Optional[str] = None, format: str = 'pickle'):
        """
        Save the processed data to disk.

        Args:
            save_path: Path to save the data. If None, uses a default path.
            format: Format to save ('pickle', 'numpy'). Default is 'pickle'.
        """
        if not self.data:
            print("No data to save")
            return

        if save_path is None:
            save_path = f"processed_circuit_data_{len(self.data)}_samples"

        # Ensure save_path has correct extension
        if format == 'pickle':
            if not save_path.endswith('.pkl'):
                save_path += '.pkl'

            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Saved {len(self.data)} samples to {save_path} (pickle format)")

        elif format == 'numpy':
            if not save_path.endswith('.npz'):
                save_path += '.npz'

            # Extract features and metadata separately
            features = self.extract_features()
            metadata = []
            for item in self.data:
                metadata.append({
                    'expr_id': item['expr_id'],
                    'step_number': item['step_number'],
                    'before_after': item['before_after'],
                    'step_labels': item['step_labels'],
                    'original_expression': item['original_expression']
                })

            np.savez(save_path, features=features, metadata=metadata)
            print(f"Saved {len(self.data)} samples to {save_path} (numpy format)")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'numpy'.")

    def load_data(self, load_path: str, format: str = 'pickle'):
        """
        Load previously saved data from disk.

        Args:
            load_path: Path to load the data from.
            format: Format to load ('pickle', 'numpy'). Default is 'pickle'.
        """
        if format == 'pickle':
            import pickle
            with open(load_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} samples from {load_path} (pickle format)")

        elif format == 'numpy':
            data = np.load(load_path, allow_pickle=True)
            features = data['features']
            metadata = data['metadata']

            # Reconstruct self.data
            self.data = []
            for i, meta in enumerate(metadata):
                self.data.append({
                    'expr_id': meta['expr_id'],
                    'step_number': meta['step_number'],
                    'before_after': meta['before_after'],
                    'features': features[i],
                    'step_labels': meta['step_labels'],
                    'original_expression': meta['original_expression']
                })
            print(f"Loaded {len(self.data)} samples from {load_path} (numpy format)")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'numpy'.")

    def get_summary(self):
        """Print summary of loaded data."""
        print(f"Total loaded samples: {len(self.data)}")

        if self.data:
            # Count by before/after
            before_count = sum(1 for item in self.data if item['before_after'] == 'before')
            after_count = sum(1 for item in self.data if item['before_after'] == 'after')
            print(f"Before graphs: {before_count}, After graphs: {after_count}")

            # Count by step labels
            true_labels = sum(1 for item in self.data if item['step_labels'].get('step_label', False))
            false_labels = len(self.data) - true_labels
            print(f"True labels: {true_labels}, False labels: {false_labels}")

            # Expression range
            expr_ids = [item['expr_id'] for item in self.data]
            print(f"Expression IDs: {min(expr_ids)} to {max(expr_ids)}")

            # Feature info
            sample_features = self.data[0]['features']
            print(f"Feature dimensions: {sample_features.shape}")
            print(f"Feature type: {type(sample_features)}")
            print(f"Feature range: {sample_features.min():.4f} to {sample_features.max():.4f}")

            # Memory usage estimate
            total_features = sum(item['features'].nbytes for item in self.data)
            print(f"Total feature memory: {total_features / (1024**2):.2f} MB")


def main():
    """Simple example usage."""
    parser = argparse.ArgumentParser(description='Load circuit graphs and step labels')
    parser.add_argument('--expressions_json', type=str,
                       help='Path to expressions JSON file (required if not loading from file)')
    parser.add_argument('--graph_name_prefix', type=str,
                       help='Prefix for graph file names (required if not loading from file)')
    parser.add_argument('--graph_dir', type=str,
                       default='./graphs/',
                       help='Directory containing graph checkpoints')
    parser.add_argument('--feature_type', type=str, default='advanced_graph_features',
                       help='Feature type to extract: adjacency_matrix, active_features, activation_values, selected_features, advanced_graph_features')
    parser.add_argument('--before_after', type=str, default='both',
                       choices=['before', 'after', 'both'],
                       help='Which graph types to load')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne', 'none'],
                       help='Visualization method')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for loading')
    parser.add_argument('--use_cpu', action='store_true', default=True,
                       help='Load tensors to CPU (default: True)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Load tensors to GPU (overrides --use_cpu)')

    # Data loading/saving arguments
    parser.add_argument('--load_data', type=str,
                       help='Path to load previously saved data from')
    parser.add_argument('--load_format', type=str, default='pickle', choices=['pickle', 'numpy'],
                       help='Format of data to load (default: pickle)')
    parser.add_argument('--save_data', type=str,
                       help='Path to save processed data to')
    parser.add_argument('--save_format', type=str, default='pickle', choices=['pickle', 'numpy'],
                       help='Format to save data in (default: pickle)')

    # Visualization arguments
    parser.add_argument('--save_plot', type=str,
                       help='Path to save visualization plot')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CircuitAnalyzer(graph_dir=args.graph_dir)

    # Load data either from file or by processing graphs
    if args.load_data:
        print(f"Loading data from {args.load_data}...")
        analyzer.load_data(args.load_data, format=args.load_format)
    else:
        # Validate required arguments for graph processing
        if not args.expressions_json or not args.graph_name_prefix:
            parser.error("--expressions_json and --graph_name_prefix are required when not loading from file")

        # Handle CPU/GPU flag logic
        use_cpu = args.use_cpu and not args.use_gpu

        # Load data from graphs
        analyzer.load_all_data(args.expressions_json, args.graph_name_prefix,
                              args.feature_type, args.before_after,
                              args.num_workers, args.batch_size, use_cpu)

    # Print summary
    analyzer.get_summary()

    # Save data if requested
    if args.save_data:
        print(f"Saving data to {args.save_data}...")
        analyzer.save_data(args.save_data, format=args.save_format)

    # Visualize
    if args.method == 'pca':
        analyzer.visualize_with_pca(save_path=args.save_plot)
    elif args.method == 'tsne':
        analyzer.visualize_with_tsne(save_path=args.save_plot)
    elif args.method == 'none':
        print("No visualization method selected")
        pass


if __name__ == "__main__":
    main()
