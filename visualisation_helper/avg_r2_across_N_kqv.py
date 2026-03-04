#!/usr/bin/env python3
# recompute_r2_from_top_n_kqv.py
# ===============================
#
# Given three GML files (for k, q, v where edge weights are initial R^2 values) and corresponding
# activation data, this script re-evaluates connectivity for all three simultaneously. For each target head,
# it identifies the top N incoming edges (highest R^2), then trains new linear regression models using 'cuml' 
# on the GPU. The script computes and logs statistics of the R^2 values obtained from these newly trained models
# for key, query, and value components.
#
# Usage:
# ------
# python visualisation_helper/avg_r2_barplot_across_N.py \
#   --k_gml <k_gml_file> --q_gml <q_gml_file> --v_gml <v_gml_file> \
#   --k_pickle <k_activation_pickle> --q_pickle <q_activation_pickle> --v_pickle <v_activation_pickle> \
#   --top_n <N> [--num_samples <num>]
#
# Example:
# python visualisation_helper/avg_r2_barplot_across_N.py \
#   --k_gml gmls/thought_graph_key.gml --q_gml gmls/thought_graph_query.gml --v_gml gmls/thought_graph_value.gml \
#   --k_pickle processed_data/k_activations.pkl --q_pickle processed_data/q_activations.pkl --v_pickle processed_data/v_activations.pkl \
#   --max_n 5
#

import argparse
import pickle
import os
import logging
import sys
import re

import networkx as nx
import numpy as np
import cupy as cp
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.metrics import r2_score as cu_r2_score
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to sys.path for relative imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Recompute R^2 values for target heads using activations from Top-N influencing source heads identified from KQV GML graphs. Computes and logs comparative statistics of these new R^2 values across key, query, and value components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # GML files
    parser.add_argument("--k_gml", required=True, help="Path to the key GML file. Edge weights ('weight') are assumed to be initial R^2 values.")
    parser.add_argument("--q_gml", required=True, help="Path to the query GML file. Edge weights ('weight') are assumed to be initial R^2 values.")
    parser.add_argument("--v_gml", required=True, help="Path to the value GML file. Edge weights ('weight') are assumed to be initial R^2 values.")
    
    # Pickle files
    parser.add_argument("--k_pickle", required=True, help="Path to the key activation pickle file.")
    parser.add_argument("--q_pickle", required=True, help="Path to the query activation pickle file.")
    parser.add_argument("--v_pickle", required=True, help="Path to the value activation pickle file.")
    
    # Matrix keys (now with defaults for each component)
    parser.add_argument("--k_matrix_key", default="k_matrix", help="Key for the key activation matrix in the pickle file.")
    parser.add_argument("--q_matrix_key", default="q_matrix", help="Key for the query activation matrix in the pickle file.")
    parser.add_argument("--v_matrix_key", default="v_matrix", help="Key for the value activation matrix in the pickle file.")
    
    # Modified to support multiple N values
    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--top_n", type=int, help="Single Num_Ref value - number of top incoming edges to select for constructing features.")
    n_group.add_argument("--max_n", type=int, help="Maximum Num_Ref value - compute for Num_Ref=1,2,...,max_n.")
    
    parser.add_argument(
        "--num_samples_subsample", type=int, default=None,
        help="Number of samples to randomly select from activations for training new models (default: all)."
    )
    parser.add_argument(
        "--search_best_n_depth", type=int, default=None,
        help="If specified, for each Num_Ref (from --top_n or --max_n), enables a local search for the optimal number of source heads. The search considers 1 up to min(Num_Ref, search_best_n_depth) source heads, selected from Num_Ref's top candidates, to maximize R^2. E.g., a value of 10 means searching within the top 10 (or fewer if Num_Ref is smaller) of Num_Ref's candidates."
    )
    return parser.parse_args()

def load_activation_data(file_path, matrix_key):
    """Loads activation data from a pickle file, similar to trace_thought.py."""
    logging.info(f"Loading activation data from {file_path} with key '{matrix_key}'...")
    if not os.path.exists(file_path):
        logging.error(f"Activation data file not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            logging.info(f"Loaded data is a dictionary. Keys: {list(data.keys())}")
            if matrix_key not in data:
                logging.error(f"Matrix key '{matrix_key}' not found in the data dictionary.")
                return None
            activations = data[matrix_key]
        elif isinstance(data, list) and data:
            logging.info("Loaded data is a list. Assuming the activation matrix is the first element.")
            activations = data[0]
            if not isinstance(activations, np.ndarray):
                 logging.error(f"First element of the list is not a NumPy array, but type {type(activations)}. Cannot proceed.")
                 return None
        elif isinstance(data, np.ndarray):
             logging.info("Loaded data is directly a NumPy array.")
             activations = data
        else:
            logging.error(f"Loaded data is of unexpected type: {type(data)}. Expected dict, list, or numpy.ndarray.")
            return None

        logging.info(f"Identified activation matrix. Shape: {activations.shape}")
        if activations.ndim != 4:
            # Expected shape: (num_samples, num_layers, num_heads, activation_dim)
            logging.error(f"Unexpected activation matrix dimensions: {activations.ndim}. Expected 4.")
            return None
        return activations
    except Exception as e:
        logging.error(f"Error loading or processing activation data: {e}")
        return None

def get_node_activations_gpu(activations_tensor_gpu, node_tuple, num_total_samples, activation_dim):
    """
    Extracts activations for a specific node (layer, head) from the GPU tensor.
    Args:
        activations_tensor_gpu (cp.ndarray): Full activation tensor (samples, layers, heads, dim) on GPU.
        node_tuple (tuple): (layer_idx, head_idx).
        num_total_samples (int): Total number of samples in the tensor.
        activation_dim (int): Activation dimension.
    Returns:
        cp.ndarray: Activations for the node (num_samples, activation_dim), or None if indices are invalid.
    """
    layer_idx, head_idx = node_tuple
    _, num_layers, num_heads, _ = activations_tensor_gpu.shape

    if not (0 <= layer_idx < num_layers and 0 <= head_idx < num_heads):
        logging.warning(f"Node {node_tuple} is out of bounds for activation tensor dimensions ({num_layers} layers, {num_heads} heads). Skipping.")
        return None
    
    # Reshape to (num_samples, activation_dim) for direct use in regression
    return activations_tensor_gpu[:, layer_idx, head_idx, :].reshape(num_total_samples, activation_dim)

def parse_node_label(node_id, node_data):
    """
    Parse a node label into a tuple (layer, head).
    
    Args:
        node_id: The node identifier in the graph
        node_data: The node attributes dictionary
    
    Returns:
        tuple: (layer_idx, head_idx) or None if parsing fails
    """
    if isinstance(node_id, tuple) and len(node_id) == 2:
        # Node ID is already a tuple
        return node_id
    
    # Try to parse from label
    label = node_data.get('label')
    if label:
        match = re.match(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', label)
        if match:
            layer_idx = int(match.group(1))
            head_idx = int(match.group(2))
            return (layer_idx, head_idx)
    
    logging.warning(f"Node {node_id} is not in expected (layer, head) format. Skipping.")
    return None

def recompute_r2_for_n(G, node_id, node_data, activations_gpu, num_samples, activation_dim, n_value, component_name):
    """
    Recomputes R^2 for a specific target node using the top N influencing source heads.
    """
    target_node_tuple = parse_node_label(node_id, node_data)
    if target_node_tuple is None:
        return float('nan')
    
    # 1. Get target activations (Y)
    Y_target_gpu = get_node_activations_gpu(activations_gpu, target_node_tuple, num_samples, activation_dim)
    if Y_target_gpu is None:
        logging.warning(f"Could not get activations for target node {target_node_tuple} ({component_name}). Skipping.")
        return float('nan')
    
    # 2. Identify Top N incoming edges and their source nodes
    incoming_edges = sorted(
        [edge for edge in G.in_edges(node_id, data=True) if 'weight' in edge[2]],
        key=lambda x: x[2]['weight'],
        reverse=True
    )
    
    top_n_source_nodes_with_data = incoming_edges[:n_value]
    
    if not top_n_source_nodes_with_data:
        logging.warning(f"Target node {target_node_tuple} ({component_name}) has no valid incoming edges with 'weight' or fewer than 1 for top {n_value}. Assigning NaN R^2.")
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        return float('nan')
    
    # 3. Construct feature matrix (X) from source node activations
    X_source_activations_list_gpu = []
    valid_sources_count = 0
    for source_node_data in top_n_source_nodes_with_data:
        source_node_id = source_node_data[0]
        source_node_attributes = G.nodes[source_node_id]
        
        # Parse source node label into tuple
        source_node_tuple = parse_node_label(source_node_id, source_node_attributes)
        if source_node_tuple is None:
            continue
        
        source_acts_gpu = get_node_activations_gpu(activations_gpu, source_node_tuple, num_samples, activation_dim)
        if source_acts_gpu is not None:
            X_source_activations_list_gpu.append(source_acts_gpu)
            valid_sources_count += 1
        else:
            logging.warning(f"  Could not get activations for source node {source_node_tuple} (target {target_node_tuple}, {component_name}). Skipping this source.")
    
    if not X_source_activations_list_gpu:
        logging.warning(f"No valid source activations found for target {target_node_tuple} ({component_name}) from its top influencers. Assigning NaN R^2.")
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        return float('nan')
    
    X_sources_gpu = cp.concatenate(X_source_activations_list_gpu, axis=1)
    del X_source_activations_list_gpu
    
    num_features = X_sources_gpu.shape[1]
    logging.debug(f"  Target {target_node_tuple} ({component_name}): Training with {valid_sources_count} source heads, {num_features} total features (activation_dim={activation_dim}).")
    
    if num_samples <= num_features:
        logging.warning(f"  Target {target_node_tuple} ({component_name}): Number of samples ({num_samples}) is not greater than number of features ({num_features}). Regression may be ill-conditioned. Assigning NaN R^2.")
        del X_sources_gpu
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        return float('nan')
    
    # 4. Train Linear Regression model using cuML
    try:
        regression_model = cuLinearRegression(fit_intercept=True, copy_X=False, algorithm='svd')
        regression_model.fit(X_sources_gpu, Y_target_gpu)
        
        # 5. Predict and Calculate R^2 score
        Y_pred_gpu = regression_model.predict(X_sources_gpu)
        
        current_r2_score = float(cu_r2_score(Y_target_gpu, Y_pred_gpu))
        r2_score = max(-1.0, current_r2_score)
        logging.debug(f"  Target {target_node_tuple} ({component_name}): Recomputed R^2 = {current_r2_score:.4f} (N={n_value}, using {valid_sources_count} sources).")
        
        return r2_score
    
    except Exception as e:
        logging.error(f"  Error during regression for target {target_node_tuple} ({component_name}): {e}. Assigning NaN R^2.")
        return float('nan')
    finally:
        # Clean up GPU memory for this iteration
        del X_sources_gpu
        del Y_target_gpu
        if 'Y_pred_gpu' in locals(): del Y_pred_gpu
        if 'regression_model' in locals(): del regression_model
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

def _train_model_and_get_r2(X_sources_gpu, Y_target_gpu, num_samples, target_node_tuple_for_logging, context_str=""):
    """
    Helper function to train a linear regression model and return the R^2 score.
    Does not delete X_sources_gpu or Y_target_gpu.
    """
    num_features = X_sources_gpu.shape[1]
    if num_samples <= num_features:
        logging.warning(f"  Target {target_node_tuple_for_logging} ({context_str}): Number of samples ({num_samples}) is not greater than number of features ({num_features}). Regression may be ill-conditioned. Assigning NaN R^2.")
        return float('nan')
    
    try:
        regression_model = cuLinearRegression(fit_intercept=True, copy_X=False, algorithm='svd')
        regression_model.fit(X_sources_gpu, Y_target_gpu)
        Y_pred_gpu = regression_model.predict(X_sources_gpu)
        current_r2_score = float(cu_r2_score(Y_target_gpu, Y_pred_gpu))
        r2_score = max(-1.0, current_r2_score) # Ensure R2 is not below -1
        del regression_model, Y_pred_gpu # Clean up model and prediction
        # cp.cuda.Stream.null.synchronize() # Syncing here might be too frequent
        # cp.get_default_memory_pool().free_all_blocks()
        return r2_score
    except Exception as e:
        logging.error(f"  Error during regression for target {target_node_tuple_for_logging} ({context_str}): {e}. Assigning NaN R^2.")
        return float('nan')

def recompute_r2_with_local_n_search(G, node_id, node_data, activations_gpu, num_samples, activation_dim, current_n_outer, search_depth, component_name):
    """
    Recomputes R^2 for a target node by performing a local search for the best number of source heads.
    The search is up to min(current_n_outer, search_depth) sources.
    """
    target_node_tuple = parse_node_label(node_id, node_data)
    if target_node_tuple is None:
        return float('nan')

    Y_target_gpu = get_node_activations_gpu(activations_gpu, target_node_tuple, num_samples, activation_dim)
    if Y_target_gpu is None:
        logging.warning(f"Could not get activations for target node {target_node_tuple} ({component_name}, local search). Skipping.")
        return float('nan')

    # Identify top `current_n_outer` incoming edges as the pool of candidates
    incoming_edges = sorted(
        [edge for edge in G.in_edges(node_id, data=True) if 'weight' in edge[2]],
        key=lambda x: x[2]['weight'],
        reverse=True
    )
    top_n_outer_sources_info = incoming_edges[:current_n_outer]

    if not top_n_outer_sources_info:
        logging.warning(f"Target node {target_node_tuple} ({component_name}, local search, Num_Ref={current_n_outer}) has no candidate sources. Assigning NaN R^2.")
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        return float('nan')

    best_r2_in_search = float('-inf')
    
    # Determine the actual number of source heads to try in the inner loop
    # These sources are selected from the top_n_outer_sources_info
    num_sources_for_inner_search_max = min(len(top_n_outer_sources_info), search_depth)
    
    # Pre-fetch activations for these sources to avoid redundant GPU calls in the inner loop
    source_activations_for_search_list_gpu = []
    valid_source_nodes_for_search = [] # For logging purposes

    for i in range(num_sources_for_inner_search_max):
        source_node_data_local = top_n_outer_sources_info[i]
        source_node_id_local = source_node_data_local[0]
        source_node_attributes_local = G.nodes[source_node_id_local]
        source_node_tuple_local = parse_node_label(source_node_id_local, source_node_attributes_local)
        
        if source_node_tuple_local is None:
            logging.warning(f"  (Local search for {target_node_tuple}, {component_name}) Invalid source node {source_node_id_local} encountered. Skipping this source for search.")
            continue
        
        source_acts_gpu = get_node_activations_gpu(activations_gpu, source_node_tuple_local, num_samples, activation_dim)
        if source_acts_gpu is not None:
            source_activations_for_search_list_gpu.append(source_acts_gpu)
            valid_source_nodes_for_search.append(source_node_tuple_local)
        else:
            logging.warning(f"  (Local search for {target_node_tuple}, {component_name}) Could not get activations for source {source_node_tuple_local}. Skipping this source for search.")

    if not source_activations_for_search_list_gpu:
        logging.warning(f"Target {target_node_tuple} ({component_name}, local search, Num_Ref={current_n_outer}): No valid source activations found from candidates within search depth {search_depth}. Assigning NaN R^2.")
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        return float('nan')

    # Inner loop: try using 1, 2, ..., up to len(source_activations_for_search_list_gpu) sources
    for k_sources_to_use in range(1, len(source_activations_for_search_list_gpu) + 1):
        current_X_sources_list_gpu = source_activations_for_search_list_gpu[:k_sources_to_use]
        X_k_sources_gpu = cp.concatenate(current_X_sources_list_gpu, axis=1)
        
        num_actual_features = X_k_sources_gpu.shape[1]
        logging.debug(f"  Target {target_node_tuple} ({component_name}, local search, Num_Ref={current_n_outer}, search_depth={search_depth}): trying with {k_sources_to_use} source heads ({num_actual_features} features).")

        current_r2 = _train_model_and_get_r2(X_k_sources_gpu, Y_target_gpu, num_samples, target_node_tuple, f"{component_name} local search k={k_sources_to_use}/{len(source_activations_for_search_list_gpu)}")
        
        if not np.isnan(current_r2):
            if current_r2 > best_r2_in_search: # Check to handle initial -inf and subsequent NaNs
                 best_r2_in_search = current_r2
        
        del X_k_sources_gpu # Clean up concatenated array for this inner iteration

    # Clean up GPU memory used in this function
    del source_activations_for_search_list_gpu # List of cp.arrays
    del Y_target_gpu
    cp.cuda.Stream.null.synchronize()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    if best_r2_in_search == float('-inf'): # No successful regression in the search
        logging.info(f"  Target {target_node_tuple} ({component_name}, local search, Num_Ref={current_n_outer}): No valid R^2 found. Assigning NaN.")
        return float('nan')
    
    logging.debug(f"  Target {target_node_tuple} ({component_name}, local search, Num_Ref={current_n_outer}): Best R^2 = {best_r2_in_search:.4f} found by searching up to {search_depth} sources (using {len(valid_source_nodes_for_search)} valid sources for search).")
    return best_r2_in_search

def process_component(component_name, gml_file, pickle_file, matrix_key, n_values, args):
    """
    Process a single component (K, Q, or V) and return R^2 scores for all N values.
    """
    logging.info(f"--- Processing {component_name.upper()} component ---")
    
    # Load GML Graph
    logging.info(f"Loading {component_name} GML graph from {gml_file}...")
    if not os.path.exists(gml_file):
        logging.error(f"{component_name} GML file not found: {gml_file}")
        return None
    try:
        G = nx.read_gml(gml_file, label='id')
        logging.info(f"{component_name} GML graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    except Exception as e:
        logging.error(f"Error reading {component_name} GML file {gml_file}: {e}")
        return None

    # Load Activations
    activations_np = load_activation_data(pickle_file, matrix_key)
    if activations_np is None:
        return None

    # Subsample if requested
    if args.num_samples_subsample is not None and 0 < args.num_samples_subsample < activations_np.shape[0]:
        logging.info(f"Randomly sampling {args.num_samples_subsample} from {activations_np.shape[0]} available samples for {component_name} regression tasks.")
        sample_indices = np.random.choice(activations_np.shape[0], args.num_samples_subsample, replace=False)
        activations_np = activations_np[sample_indices, :, :, :]
    elif args.num_samples_subsample is not None:
        logging.warning(f"--num_samples_subsample ({args.num_samples_subsample}) is invalid or not less than total samples for {component_name}. Using all {activations_np.shape[0]} samples.")
    
    num_samples, num_layers, num_heads, activation_dim = activations_np.shape
    logging.info(f"{component_name} activation data for regression: Samples={num_samples}, Layers={num_layers}, Heads={num_heads}, Dim={activation_dim}")

    if num_samples == 0 or activation_dim == 0:
        logging.error(f"No samples or zero activation dimension in the {component_name} data. Cannot proceed.")
        return None

    # Move activations to GPU
    logging.info(f"Moving {component_name} activation data to GPU (CuPy array)...")
    try:
        activations_gpu = cp.asarray(activations_np)
        del activations_np  # Free CPU memory
        cp.cuda.Stream.null.synchronize()
        logging.info(f"{component_name} activation data successfully moved to GPU.")
    except Exception as e:
        logging.error(f"Failed to move {component_name} activation data to GPU: {e}. Try with smaller --num_samples_subsample or ensure GPU memory is available.")
        return None
    
    # Dictionary to store R^2 scores for each N value
    component_r2_scores = {n: [] for n in n_values}
    
    # Main Processing Loop
    logging.info(f"Processing {component_name} target nodes to recompute R^2 for Num_Ref values: {n_values}...")
    processed_node_count = 0
    
    for node_id, node_data in G.nodes(data=True):
        processed_node_count += 1
        target_node_tuple = parse_node_label(node_id, node_data)
        if target_node_tuple is None:
            continue
        
        if processed_node_count <= 5 or processed_node_count % 50 == 0:
            logging.info(f"Processing {component_name} target node {target_node_tuple} ({processed_node_count}/{G.number_of_nodes()})...")
        
        # Process each N value for this target node
        for n_outer_iteration_val in n_values:
            if args.search_best_n_depth is not None:
                r2_score = recompute_r2_with_local_n_search(
                    G, node_id, node_data, activations_gpu,
                    num_samples, activation_dim,
                    n_outer_iteration_val,
                    args.search_best_n_depth,
                    component_name
                )
            else:
                r2_score = recompute_r2_for_n(
                    G, node_id, node_data, activations_gpu,
                    num_samples, activation_dim,
                    n_outer_iteration_val,
                    component_name
                )
            
            component_r2_scores[n_outer_iteration_val].append(r2_score)
            
            # Log only for the first few nodes to avoid excessive output
            if processed_node_count <= 5 or processed_node_count % 50 == 0:
                log_prefix = "SearchOptimized " if args.search_best_n_depth is not None else ""
                logging.info(f"  {component_name} Target {target_node_tuple}, Num_Ref={n_outer_iteration_val}: {log_prefix}R^2 = {r2_score:.4f}")

    # Clean up GPU memory
    del activations_gpu
    cp.cuda.Stream.null.synchronize()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    
    return component_r2_scores

def main():
    args = parse_arguments()

    # Convert to a list of N values
    if args.top_n is not None:
        if args.top_n <= 0:
            logging.error("--top_n must be a positive integer.")
            return 1
        n_values = [args.top_n]
        multiple_n = False
    else:  # Using --max_n
        if args.max_n <= 0:
            logging.error("--max_n must be a positive integer.")
            return 1
        n_values = list(range(1, args.max_n + 1))
        multiple_n = True

    # Define components and their corresponding files
    components_config = {
        'key': {
            'gml_file': args.k_gml,
            'pickle_file': args.k_pickle,
            'matrix_key': args.k_matrix_key
        },
        'query': {
            'gml_file': args.q_gml,
            'pickle_file': args.q_pickle,
            'matrix_key': args.q_matrix_key
        },
        'value': {
            'gml_file': args.v_gml,
            'pickle_file': args.v_pickle,
            'matrix_key': args.v_matrix_key
        }
    }

    # Dictionary to store all R^2 scores for all components
    all_components_r2_scores = {}
    
    # Process each component
    for component_name, config in components_config.items():
        component_scores = process_component(
            component_name,
            config['gml_file'],
            config['pickle_file'],
            config['matrix_key'],
            n_values,
            args
        )
        
        if component_scores is None:
            logging.error(f"Failed to process {component_name} component. Exiting.")
            return 1
            
        all_components_r2_scores[component_name] = component_scores

    # --- Generate Comparative Statistics ---
    logging.info("=== COMPARATIVE STATISTICS ACROSS K, Q, V COMPONENTS ===")
    
    # Check for R² values greater than 1 across all components
    for component_name, component_scores in all_components_r2_scores.items():
        logging.info(f"--- {component_name.upper()} Component Analysis ---")
        for n, scores in component_scores.items():
            values_over_one = [score for score in scores if score > 1.0]
            if values_over_one:
                logging.info(f"  {component_name} Num_Ref={n}: Found {len(values_over_one)} R² values greater than 1.0")
                logging.info(f"  {component_name} Num_Ref={n}: Max R² value: {max(values_over_one):.4f}")
                
    # Prepare filtered statistics data for all components
    all_components_filtered_scores = {}
    for component_name, component_scores in all_components_r2_scores.items():
        filtered_scores = {}
        for n, scores in component_scores.items():
            filtered_scores[n] = [s for s in scores if not np.isnan(s)]
            logging.info(f"{component_name} Num_Ref={n}: {len(filtered_scores[n])} valid R^2 scores (out of {len(scores)} targets)")
        all_components_filtered_scores[component_name] = filtered_scores

    # --- Print Detailed Statistics for each component and N ---
    logging.info("--- DETAILED R^2 STATISTICS PER COMPONENT AND NUM_REF ---")
    for component_name in ['key', 'query', 'value']:
        logging.info(f"\n{component_name.upper()} COMPONENT:")
        filtered_scores = all_components_filtered_scores[component_name]
        for n in n_values:
            scores_for_n = filtered_scores[n]
            if scores_for_n:
                scores_np = np.array(scores_for_n)
                logging.info(f"  Num_Ref={n} (based on {len(scores_np)} valid R^2 values):")
                logging.info(f"    Mean:       {np.mean(scores_np):.4f}")
                logging.info(f"    Median:     {np.median(scores_np):.4f}")
                logging.info(f"    Min:        {np.min(scores_np):.4f}")
                logging.info(f"    Max:        {np.max(scores_np):.4f}")
                logging.info(f"    Std Dev:    {np.std(scores_np):.4f}")
                logging.info(f"    25th Pctl:  {np.percentile(scores_np, 25):.4f}")
                logging.info(f"    75th Pctl:  {np.percentile(scores_np, 75):.4f}")
            else:
                logging.info(f"  Num_Ref={n}: No valid R^2 scores to compute statistics.")

    # --- Cross-Component Comparison ---
    logging.info("\n--- CROSS-COMPONENT COMPARISON (MEAN R^2 VALUES) ---")
    logging.info(f"{'Num_Ref':<8} {'Key':<10} {'Query':<10} {'Value':<10}")
    logging.info("-" * 40)
    for n in n_values:
        key_mean = np.mean(all_components_filtered_scores['key'][n]) if all_components_filtered_scores['key'][n] else float('nan')
        query_mean = np.mean(all_components_filtered_scores['query'][n]) if all_components_filtered_scores['query'][n] else float('nan')
        value_mean = np.mean(all_components_filtered_scores['value'][n]) if all_components_filtered_scores['value'][n] else float('nan')
        
        logging.info(f"{n:<8} {key_mean:<10.4f} {query_mean:<10.4f} {value_mean:<10.4f}")

    # --- Summary Statistics ---
    logging.info("\n--- SUMMARY: BEST PERFORMING COMPONENT PER NUM_REF ---")
    for n in n_values:
        component_means = {}
        for component_name in ['key', 'query', 'value']:
            scores = all_components_filtered_scores[component_name][n]
            if scores:
                component_means[component_name] = np.mean(scores)
            else:
                component_means[component_name] = float('-inf')
        
        if component_means:
            best_component = max(component_means, key=component_means.get)
            best_score = component_means[best_component]
            if best_score != float('-inf'):
                logging.info(f"  Num_Ref={n}: Best component is {best_component.upper()} with mean R^2 = {best_score:.4f}")
            else:
                logging.info(f"  Num_Ref={n}: No valid scores for any component")

    # --- Extract mean values for plotting ---
    logging.info("--- EXTRACTING MEAN VALUES FOR PLOTTING ---")
    
    # Extract mean R² values for each component across N
    key_means = []
    query_means = []
    value_means = []
    
    for n in n_values:
        key_scores = all_components_filtered_scores['key'][n]
        query_scores = all_components_filtered_scores['query'][n]
        value_scores = all_components_filtered_scores['value'][n]
        
        key_mean = np.mean(key_scores) if key_scores else 0.0
        query_mean = np.mean(query_scores) if query_scores else 0.0
        value_mean = np.mean(value_scores) if value_scores else 0.0
        
        key_means.append(key_mean)
        query_means.append(query_mean)
        value_means.append(value_mean)
        
        logging.info(f"N={n}: Key={key_mean:.4f}, Query={query_mean:.4f}, Value={value_mean:.4f}")
    
    # --- Create Bar Plots ---
    logging.info("--- CREATING BAR PLOTS ---")
    create_barplot(n_values, key_means, query_means, value_means)
    
    logging.info("=== SCRIPT COMPLETED SUCCESSFULLY ===")
    return 0

def create_barplot(n_values, key_means, query_means, value_means):
    """
    Create bar plots exactly like avg_r2_accross_N.py
    """
    # Define the PLOT_STYLE based on the style used in avg_r2_accross_N.py
    PLOT_STYLE = {
        'background_color': 'white',
        'colors_list': [
            '#FF7070',  # Darker Pastel Red (for Key)
            '#FFA050',  # Darker Pastel Orange (for Query)
            '#A993C0',  # Darker Pastel Purple (for Value)
            '#8FB1D9',  # Darker Pastel Blue
            '#7BC86C',  # Darker Pastel Green
            '#B08078',  # Darker Pastel Brown
            '#F098B8',  # Darker Pastel Pink
            '#B0B0B0',  # Darker Pastel Gray
            '#C8C86C',  # Darker Pastel Yellow-Green
            '#7FC0D0'   # Darker Pastel Blue-Teal
        ],
        'hatches_list': ['\\\\', '-', '//', '', 'x', ''],
        'line_color_default': '#1f77b4',
        'marker_color_default': '#1f77b4',
        'opacity_default': 1.0,
        'opacity_fill_kde': 0.4,
        'grid_linestyle': '--',
        'grid_linewidth': 0.5,
        'grid_alpha': 0.7,
        'font_family': 'sans-serif',
        'title_fontsize': 34,
        'label_fontsize': 34,
        'tick_fontsize': 28,
        'legend_fontsize': 26,
        'fontweight': 'bold',
        'plot_edgecolor': 'black',
        'spine_top_visible': True,
        'spine_right_visible': True,
    }

    # Apply plot style using rcParams
    plt.rcParams['font.family'] = PLOT_STYLE.get('font_family', 'sans-serif')
    plt.rcParams['axes.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['figure.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['axes.spines.top'] = PLOT_STYLE.get('spine_top_visible', True)
    plt.rcParams['axes.spines.right'] = PLOT_STYLE.get('spine_right_visible', True)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = PLOT_STYLE.get('grid_linestyle', '--')
    plt.rcParams['grid.linewidth'] = PLOT_STYLE.get('grid_linewidth', 0.5)
    plt.rcParams['grid.alpha'] = PLOT_STYLE.get('grid_alpha', 0.7)

    bar_width = 0.25
    x = np.arange(len(n_values))

    # Create the plot
    fig = plt.figure(figsize=(12, 8))  # Increased figure size

    # Use colors from PLOT_STYLE['colors_list']
    plt.bar(x - bar_width, key_means, width=bar_width, label='Key', 
           color=PLOT_STYLE['colors_list'][0], alpha=PLOT_STYLE['opacity_default'])
    plt.bar(x, query_means, width=bar_width, label='Query', 
           color=PLOT_STYLE['colors_list'][1], alpha=PLOT_STYLE['opacity_default'])
    plt.bar(x + bar_width, value_means, width=bar_width, label='Value', 
           color=PLOT_STYLE['colors_list'][2], alpha=PLOT_STYLE['opacity_default'])

    # Add labels and formatting with the new style
    plt.xlabel('Number of Reference Heads (N)', fontsize=PLOT_STYLE['label_fontsize'], fontweight=PLOT_STYLE['fontweight'])
    plt.ylabel('Mean $R^2$', fontsize=PLOT_STYLE['label_fontsize'], fontweight=PLOT_STYLE['fontweight'])
    # Title removed as requested
    plt.xticks(x, n_values, fontsize=PLOT_STYLE['tick_fontsize'])
    plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'])
    
    # Determine y-axis limits based on data
    all_means = key_means + query_means + value_means
    y_min = max(0.0, min(all_means) - 0.05)
    y_max = max(all_means) + 0.05
    plt.ylim(y_min, y_max)
    
    plt.legend(fontsize=PLOT_STYLE['legend_fontsize'])
    plt.grid(True, linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])

    plt.tight_layout()

    # Define output paths for saving
    output_dir = "plots"  # User confirmed this directory always exists
    output_base_filename = "avg_r2_across_N_kqv"
    output_path_pdf = os.path.join(output_dir, f"{output_base_filename}.pdf")
    output_path_png = os.path.join(output_dir, f"{output_base_filename}.png")

    # Save the plots
    logging.info(f"Saving PDF plot to: {output_path_pdf}")
    fig.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())

    logging.info(f"Saving PNG plot to: {output_path_png}")
    fig.savefig(output_path_png, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)

    logging.info("Bar plots created and saved successfully!")
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    sys.exit(main()) 