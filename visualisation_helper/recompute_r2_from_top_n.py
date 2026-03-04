#!/usr/bin/env python3
# recompute_r2_from_top_n.py
# ============================
#
# Given a GML file (where edge weights are initial R^2 values) and corresponding
# activation data, this script re-evaluates connectivity. For each target head,
# it identifies the top N incoming edges (highest R^2), then trains a new
# linear regression model using 'cuml' on the GPU. The features for this model are
# the activations of these top N source heads, and the target is the activation
# of the target head. The script then plots the CDF and KDE of the R^2 values
# obtained from these newly trained models.
#
# Usage:
# ------
# python visualisation_helper/recompute_r2_from_top_n.py <gml_file> <activation_pickle_file> \
#   --matrix_key <key_in_pickle> --top_n <N> \
#   [-o <output_base_path>] [-t <plot_title_suffix>] [--num_samples <num>]
#
# Example:
# python visualisation_helper/recompute_r2_from_top_n.py gmls/thought_graph.gml \
#   processed_data/activations.pkl --matrix_key k_matrix --top_n 5 \
#   -o plots/recomputed_top5
#
# For multiple N values (1 through max_n):
# python visualisation_helper/recompute_r2_from_top_n.py gmls/thought_graph.gml \
#   processed_data/activations.pkl --matrix_key k_matrix --max_n 5 \
#   -o plots/recomputed_multiple_n
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
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to sys.path for relative imports if PLOT_STYLE is in a common module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Define the new PLOT_STYLE based on theme_ref_code.py
# This PLOT_STYLE will be used throughout the script.
PLOT_STYLE = {
    'background_color': 'white',
    'colors_list': [ # Updated with more distinct colors
        '#1f77b4',  # Muted Blue
        '#ff7f0e',  # Safety Orange
        '#2ca02c',  # Cooked Asparagus Green
        '#d62728',  # Brick Red
        '#9467bd',  # Muted Purple
        '#8c564b',  # Chestnut Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Medium Gray
        '#bcbd22',  # Curry Yellow-Green
        '#17becf'   # Blue-Teal
    ],
    'hatches_list': ['\\\\\\\\', '-', '//', '', 'x', ''], # Corrected: '\\\\\\\\' in string literal becomes '\\\\' in Python code
    'line_color_default': '#1f77b4', # Updated to match the first color in the new list
    'marker_color_default': '#1f77b4', # Updated to match the first color in the new list
    'opacity_default': 0.8,
    'opacity_fill_kde': 0.4,
    'grid_linestyle': '--',
    'grid_linewidth': 0.5,
    'grid_alpha': 0.7,
    'font_family': 'sans-serif',
    'title_fontsize': 32,
    'label_fontsize': 32,
    'tick_fontsize': 28,
    'legend_fontsize': 26,
    'fontweight': 'normal',
    'plot_edgecolor': 'black',
    'spine_top_visible': False,
    'spine_right_visible': False,
}

try:
    # Attempt to import the original pastel_theme.PLOT_STYLE.
    # This is now mainly for logging as the PLOT_STYLE defined above takes precedence.
    from visualisation_helper.pastel_theme import PLOT_STYLE as PASTEL_IMPORTED_STYLE
    logging.info("Successfully imported PLOT_STYLE from visualisation_helper.pastel_theme. "
                 "The new theme configuration (based on theme_ref_code.py) is active and takes precedence.")
except ImportError:
    logging.warning("Could not import PLOT_STYLE from visualisation_helper.pastel_theme. "
                    "Using the new theme configuration based on theme_ref_code.py.")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Recompute R^2 values for target heads using activations from Top-N influencing source heads identified from a GML graph. Generates CDF and KDE plots of these new R^2 values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("gml_file", help="Path to the input GML file. Edge weights ('weight') are assumed to be initial R^2 values.")
    parser.add_argument("activation_file", help="Path to the pickle file containing activation data (e.g., 4D tensor: samples, layers, heads, dim).")
    parser.add_argument("--matrix_key", default="k_matrix", help="Key for the activation matrix in the pickle file, if the pickle contains a dictionary.")
    
    # Modified to support multiple N values
    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--top_n", type=int, help="Single Num_Ref value - number of top incoming edges to select for constructing features.")
    n_group.add_argument("--max_n", type=int, help="Maximum Num_Ref value - compute for Num_Ref=1,2,...,max_n and plot all CDFs together.")
    
    parser.add_argument(
        "-o", "--output_base", default="plots/recomputed_top_n_r2",
        help="Base path for output plots (e.g., 'my_plots/run1'). '_cdf.pdf', '_kde.png', etc., will be appended."
    )
    parser.add_argument(
        "-t", "--title_suffix", default=None,
        help="Suffix to append to plot titles (e.g., 'My Model Run')."
    )
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
    logging.info(f"Loading activation data from {file_path}...")
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

def recompute_r2_for_n(G, node_id, node_data, activations_gpu, num_samples, activation_dim, n_value):
    """
    Recomputes R^2 for a specific target node using the top N influencing source heads.
    """
    target_node_tuple = parse_node_label(node_id, node_data)
    if target_node_tuple is None:
        return float('nan')
    
    # 1. Get target activations (Y)
    Y_target_gpu = get_node_activations_gpu(activations_gpu, target_node_tuple, num_samples, activation_dim)
    if Y_target_gpu is None:
        logging.warning(f"Could not get activations for target node {target_node_tuple}. Skipping.")
        return float('nan')
    
    # 2. Identify Top N incoming edges and their source nodes
    incoming_edges = sorted(
        [edge for edge in G.in_edges(node_id, data=True) if 'weight' in edge[2]],
        key=lambda x: x[2]['weight'],
        reverse=True
    )
    
    top_n_source_nodes_with_data = incoming_edges[:n_value]
    
    if not top_n_source_nodes_with_data:
        logging.warning(f"Target node {target_node_tuple} has no valid incoming edges with 'weight' or fewer than 1 for top {n_value}. Assigning NaN R^2.")
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
            logging.warning(f"  Could not get activations for source node {source_node_tuple} (target {target_node_tuple}). Skipping this source.")
    
    if not X_source_activations_list_gpu:
        logging.warning(f"No valid source activations found for target {target_node_tuple} from its top influencers. Assigning NaN R^2.")
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        return float('nan')
    
    X_sources_gpu = cp.concatenate(X_source_activations_list_gpu, axis=1)
    del X_source_activations_list_gpu
    
    num_features = X_sources_gpu.shape[1]
    logging.debug(f"  Target {target_node_tuple}: Training with {valid_sources_count} source heads, {num_features} total features (activation_dim={activation_dim}).")
    
    if num_samples <= num_features:
        logging.warning(f"  Target {target_node_tuple}: Number of samples ({num_samples}) is not greater than number of features ({num_features}). Regression may be ill-conditioned. Assigning NaN R^2.")
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
        logging.debug(f"  Target {target_node_tuple}: Recomputed R^2 = {current_r2_score:.4f} (N={n_value}, using {valid_sources_count} sources).")
        
        return r2_score
    
    except Exception as e:
        logging.error(f"  Error during regression for target {target_node_tuple}: {e}. Assigning NaN R^2.")
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

def recompute_r2_with_local_n_search(G, node_id, node_data, activations_gpu, num_samples, activation_dim, current_n_outer, search_depth):
    """
    Recomputes R^2 for a target node by performing a local search for the best number of source heads.
    The search is up to min(current_n_outer, search_depth) sources.
    """
    target_node_tuple = parse_node_label(node_id, node_data)
    if target_node_tuple is None:
        return float('nan')

    Y_target_gpu = get_node_activations_gpu(activations_gpu, target_node_tuple, num_samples, activation_dim)
    if Y_target_gpu is None:
        logging.warning(f"Could not get activations for target node {target_node_tuple} (local search). Skipping.")
        return float('nan')

    # Identify top `current_n_outer` incoming edges as the pool of candidates
    incoming_edges = sorted(
        [edge for edge in G.in_edges(node_id, data=True) if 'weight' in edge[2]],
        key=lambda x: x[2]['weight'],
        reverse=True
    )
    top_n_outer_sources_info = incoming_edges[:current_n_outer]

    if not top_n_outer_sources_info:
        logging.warning(f"Target node {target_node_tuple} (local search, Num_Ref={current_n_outer}) has no candidate sources. Assigning NaN R^2.")
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
            logging.warning(f"  (Local search for {target_node_tuple}) Invalid source node {source_node_id_local} encountered. Skipping this source for search.")
            continue
        
        source_acts_gpu = get_node_activations_gpu(activations_gpu, source_node_tuple_local, num_samples, activation_dim)
        if source_acts_gpu is not None:
            source_activations_for_search_list_gpu.append(source_acts_gpu)
            valid_source_nodes_for_search.append(source_node_tuple_local)
        else:
            logging.warning(f"  (Local search for {target_node_tuple}) Could not get activations for source {source_node_tuple_local}. Skipping this source for search.")

    if not source_activations_for_search_list_gpu:
        logging.warning(f"Target {target_node_tuple} (local search, Num_Ref={current_n_outer}): No valid source activations found from candidates within search depth {search_depth}. Assigning NaN R^2.")
        del Y_target_gpu
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        return float('nan')

    # Inner loop: try using 1, 2, ..., up to len(source_activations_for_search_list_gpu) sources
    for k_sources_to_use in range(1, len(source_activations_for_search_list_gpu) + 1):
        current_X_sources_list_gpu = source_activations_for_search_list_gpu[:k_sources_to_use]
        X_k_sources_gpu = cp.concatenate(current_X_sources_list_gpu, axis=1)
        
        num_actual_features = X_k_sources_gpu.shape[1]
        logging.debug(f"  Target {target_node_tuple} (local search, Num_Ref={current_n_outer}, search_depth={search_depth}): trying with {k_sources_to_use} source heads ({num_actual_features} features).")

        current_r2 = _train_model_and_get_r2(X_k_sources_gpu, Y_target_gpu, num_samples, target_node_tuple, f"local search k={k_sources_to_use}/{len(source_activations_for_search_list_gpu)}")
        
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
        logging.info(f"  Target {target_node_tuple} (local search, Num_Ref={current_n_outer}): No valid R^2 found. Assigning NaN.")
        return float('nan')
    
    logging.debug(f"  Target {target_node_tuple} (local search, Num_Ref={current_n_outer}): Best R^2 = {best_r2_in_search:.4f} found by searching up to {search_depth} sources (using {len(valid_source_nodes_for_search)} valid sources for search).")
    return best_r2_in_search

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

    # --- Load GML Graph ---
    logging.info(f"Loading GML graph from {args.gml_file}...")
    if not os.path.exists(args.gml_file):
        logging.error(f"GML file not found: {args.gml_file}")
        return 1
    try:
        G = nx.read_gml(args.gml_file, label='id')
        logging.info(f"GML graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    except Exception as e:
        logging.error(f"Error reading GML file {args.gml_file}: {e}")
        return 1

    # --- Load Activations ---
    activations_np = load_activation_data(args.activation_file, args.matrix_key)
    if activations_np is None:
        return 1

    # Subsample if requested
    if args.num_samples_subsample is not None and 0 < args.num_samples_subsample < activations_np.shape[0]:
        logging.info(f"Randomly sampling {args.num_samples_subsample} from {activations_np.shape[0]} available samples for regression tasks.")
        sample_indices = np.random.choice(activations_np.shape[0], args.num_samples_subsample, replace=False)
        activations_np = activations_np[sample_indices, :, :, :]
    elif args.num_samples_subsample is not None:
        logging.warning(f"--num_samples_subsample ({args.num_samples_subsample}) is invalid or not less than total samples. Using all {activations_np.shape[0]} samples.")
    
    num_samples, num_layers, num_heads, activation_dim = activations_np.shape
    logging.info(f"Activation data for regression: Samples={num_samples}, Layers={num_layers}, Heads={num_heads}, Dim={activation_dim}")

    if num_samples == 0 or activation_dim == 0:
        logging.error("No samples or zero activation dimension in the data. Cannot proceed.")
        return 1

    # Move activations to GPU
    logging.info("Moving activation data to GPU (CuPy array)...")
    try:
        activations_gpu = cp.asarray(activations_np)
        del activations_np  # Free CPU memory
        cp.cuda.Stream.null.synchronize()
        logging.info("Activation data successfully moved to GPU.")
    except Exception as e:
        logging.error(f"Failed to move activation data to GPU: {e}. Try with smaller --num_samples_subsample or ensure GPU memory is available.")
        return 1
    
    # Dictionary to store R^2 scores for each N value
    all_r2_scores = {n: [] for n in n_values}
    
    # --- Main Processing Loop ---
    logging.info(f"Processing target nodes to recompute R^2 for Num_Ref values: {n_values}...")
    processed_node_count = 0
    
    for node_id, node_data in G.nodes(data=True):
        processed_node_count += 1
        target_node_tuple = parse_node_label(node_id, node_data)
        if target_node_tuple is None:
            continue
        
        logging.info(f"Processing target node {target_node_tuple} ({processed_node_count}/{G.number_of_nodes()})...")
        
        # Process each N value for this target node
        for n_outer_iteration_val in n_values: # n_values derived from --top_n or --max_n
            if args.search_best_n_depth is not None:
                r2_score = recompute_r2_with_local_n_search(
                    G, node_id, node_data, activations_gpu,
                    num_samples, activation_dim,
                    n_outer_iteration_val,     # This is current_N_iteration (max sources to consider from initial pool)
                    args.search_best_n_depth # This is the depth of the local search
                )
            else:
                r2_score = recompute_r2_for_n( # Original function
                    G, node_id, node_data, activations_gpu,
                    num_samples, activation_dim,
                    n_outer_iteration_val # The specific N to use directly
                )
            
            all_r2_scores[n_outer_iteration_val].append(r2_score)
            
            # Log only for the first few nodes to avoid excessive output
            if processed_node_count <= 5 or processed_node_count % 50 == 0:
                log_prefix = "SearchOptimized " if args.search_best_n_depth is not None else ""
                logging.info(f"  Target {target_node_tuple}, Num_Ref={n_outer_iteration_val}: {log_prefix}R^2 = {r2_score:.4f}")

    # --- Plotting ---
    # Check for R² values greater than 1
    for n, scores in all_r2_scores.items():
        values_over_one = [score for score in scores if score > 1.0]
        if values_over_one:
            logging.info(f"Found {len(values_over_one)} R² values greater than 1.0 for Num_Ref={n}")
            logging.info(f"Max R² value: {max(values_over_one):.4f}")
            
    # Prepare plotting data
    plottable_r2_scores = {}
    for n, scores in all_r2_scores.items():
        plottable_r2_scores[n] = [s for s in scores if not np.isnan(s)]
        logging.info(f"Num_Ref={n}: {len(plottable_r2_scores[n])} valid R^2 scores (out of {len(scores)} targets)")

    # --- Calculate and Print Statistics for each N (using filtered scores) ---
    logging.info("--- R^2 Statistics per Num_Ref (filtered) ---")
    for n, scores_for_n in plottable_r2_scores.items():
        if scores_for_n:
            scores_np = np.array(scores_for_n)
            logging.info(f"Statistics for Num_Ref={n} (based on {len(scores_np)} valid R^2 values):")
            logging.info(f"  Mean:       {np.mean(scores_np):.4f}")
            logging.info(f"  Median:     {np.median(scores_np):.4f}")
            logging.info(f"  Min:        {np.min(scores_np):.4f}")
            logging.info(f"  Max:        {np.max(scores_np):.4f}")
            logging.info(f"  Std Dev:    {np.std(scores_np):.4f}")
            logging.info(f"  25th Pctl:  {np.percentile(scores_np, 25):.4f}")
            logging.info(f"  75th Pctl:  {np.percentile(scores_np, 75):.4f}")
        else:
            logging.info(f"Statistics for Num_Ref={n}: No valid R^2 scores to compute statistics.")
    logging.info("------------------------------------")

    # --- Calculate and Print Overall Statistics ---
    logging.info("--- Overall R^2 Statistics (all Num_Ref values combined) ---")
    all_valid_scores_flat = [score for sublist in plottable_r2_scores.values() for score in sublist]
    if all_valid_scores_flat:
        all_scores_np = np.array(all_valid_scores_flat)
        logging.info(f"Overall statistics based on {len(all_scores_np)} valid R^2 values across all N:")
        logging.info(f"  Mean:       {np.mean(all_scores_np):.4f}")
        logging.info(f"  Median:     {np.median(all_scores_np):.4f}")
        logging.info(f"  Min:        {np.min(all_scores_np):.4f}")
        logging.info(f"  Max:        {np.max(all_scores_np):.4f}")
        logging.info(f"  Std Dev:    {np.std(all_scores_np):.4f}")
        logging.info(f"  25th Pctl:  {np.percentile(all_scores_np, 25):.4f}")
        logging.info(f"  75th Pctl:  {np.percentile(all_scores_np, 75):.4f}")
    else:
        logging.info("No valid R^2 scores across all N to compute overall statistics.")
    logging.info("----------------------------------------------------")

    # Skip plotting if no valid scores
    if all(len(s_list) == 0 for s_list in plottable_r2_scores.values()):
        logging.warning("No valid R^2 scores were recomputed for any N value. Skipping plot generation.")
        del activations_gpu
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        return 0
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_base)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating output directory {output_dir}: {e}")
    
    # Set up plotting style
    # plt.style.use('seaborn-v0_8-pastel') # Removed to use PLOT_STYLE fully
    plt.rcParams['font.family'] = PLOT_STYLE.get('font_family', 'sans-serif')
    plt.rcParams['axes.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['figure.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['axes.spines.top'] = True  # Show all spines
    plt.rcParams['axes.spines.right'] = True  # Show all spines
    plt.rcParams['axes.grid'] = True  # Enable grid
    plt.rcParams['grid.linestyle'] = PLOT_STYLE.get('grid_linestyle', '--')
    plt.rcParams['grid.linewidth'] = PLOT_STYLE.get('grid_linewidth', 0.5)
    plt.rcParams['grid.alpha'] = PLOT_STYLE.get('grid_alpha', 0.7)

    # --- CDF Plot for all N values on the same plot ---
    if multiple_n:
        base_title = f" of R² using multiple reference inputs"
    else:
        base_title = f" of R² using top Num_Ref={n_values[0]}"
        
    if args.title_suffix:
        base_title += f" - {args.title_suffix}"
    
    # CDF Plot for all N values on the same graph
    fig_cdf_combined, ax_cdf_combined = plt.subplots(figsize=(12, 8))
    fig_cdf_combined.set_facecolor(PLOT_STYLE.get('background_color', 'white'))
    ax_cdf_combined.set_facecolor(PLOT_STYLE.get('background_color', 'white'))
    
    # Color map for different N values
    # colors = plt.cm.viridis(np.linspace(0, 1, len(n_values))) # Replaced with PLOT_STYLE colors
    
    for i, n in enumerate(n_values):
        if not plottable_r2_scores[n]:
            logging.warning(f"No valid R^2 scores for Num_Ref={n}. Skipping in combined plot.")
            continue
            
        r2_values_sorted = np.sort(plottable_r2_scores[n])
        cdf_y = np.arange(1, len(r2_values_sorted) + 1) / len(r2_values_sorted)
        
        ax_cdf_combined.plot(
            r2_values_sorted, cdf_y, 
            # marker='.', # Removed marker for combined CDF
            linestyle='-', 
            linewidth=3.5,
            markersize=6, # This won't have an effect as marker is removed, but kept for consistency if markers are re-added
            alpha=PLOT_STYLE.get('opacity_default', 0.8),
            color=PLOT_STYLE['colors_list'][i % len(PLOT_STYLE['colors_list'])],
            label=f"Ref Heads={n}"
        )
    
    ax_cdf_combined.set_xlabel("R²", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
    ax_cdf_combined.set_ylabel("Cumulative Probability", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
    ax_cdf_combined.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 28))
    ax_cdf_combined.set_ylim(0, 1.05)
    
    # Find global min/max R^2 values across all N values
    all_r2_values = [val for sublist in plottable_r2_scores.values() for val in sublist]
    if all_r2_values:
        global_min_r2 = -0.05
        global_max_r2 = 1.0  # Cap at 1.0 instead of using max(all_r2_values)
        ax_cdf_combined.set_xlim(left=global_min_r2, right=global_max_r2)
    
    # ax_cdf_combined.grid(True, linestyle=PLOT_STYLE.get('grid_linestyle', '--'), 
    #                       linewidth=PLOT_STYLE.get('grid_linewidth', 0.5), alpha=0.7) # Grid managed by rcParams
    ax_cdf_combined.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 22), loc='best')
    
    # Save combined CDF plot
    output_path_cdf_combined_pdf = args.output_base + "_combined_cdf.pdf"
    output_path_cdf_combined_png = args.output_base + "_combined_cdf.png"
    
    try:
        fig_cdf_combined.savefig(output_path_cdf_combined_pdf, format='pdf', 
                                 bbox_inches='tight', facecolor=fig_cdf_combined.get_facecolor())
        logging.info(f"Combined CDF PDF plot saved to: {output_path_cdf_combined_pdf}")
        
        fig_cdf_combined.savefig(output_path_cdf_combined_png, format='png', 
                                 bbox_inches='tight', facecolor=fig_cdf_combined.get_facecolor(), dpi=300)
        logging.info(f"Combined CDF PNG plot saved to: {output_path_cdf_combined_png}")
    except Exception as e:
        logging.error(f"Error saving combined CDF plot: {e}")
    plt.close(fig_cdf_combined)
    
    # --- Individual CDF and KDE plots for each N value ---
    if not multiple_n:
        # If only one N value, generate the original individual plots
        n = n_values[0]
        r2_values_sorted_np = np.array(plottable_r2_scores[n])
        
        # CDF Plot
        cdf_plot_title = f"CDF of R² using top Num_Ref={n}"
        if args.title_suffix:
            cdf_plot_title += f" - {args.title_suffix}"
            
        fig_cdf, ax_cdf = plt.subplots(figsize=(10, 7))
        # fig_cdf.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Managed by rcParams
        # ax_cdf.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Managed by rcParams
        
        cdf_y = np.arange(1, len(r2_values_sorted_np) + 1) / len(r2_values_sorted_np)
        ax_cdf.plot(r2_values_sorted_np, cdf_y, marker='.', linestyle='-',
                    # color=PLOT_STYLE.get('line_color', 'blue'), # Using default line color
                    markerfacecolor=PLOT_STYLE.get('marker_color_default', 'blue'),
                    markeredgecolor=PLOT_STYLE.get('plot_edgecolor', 'black'),
                    alpha=PLOT_STYLE.get('opacity_default', 0.8),
                    linewidth=3.5,
                    markersize=8, label=f"Ref Heads={len(r2_values_sorted_np)}")
        
        ax_cdf.set_xlabel("R²", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
        ax_cdf.set_ylabel("Cumulative Probability", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
        ax_cdf.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 28))
        ax_cdf.set_ylim(0.0, 1.05)
        ax_cdf.set_xlim(left=-0.05, right=1.0)  # Cap at 1.0
        # ax_cdf.grid(True, linestyle=PLOT_STYLE.get('grid_linestyle', '--'), linewidth=PLOT_STYLE.get('grid_linewidth', 0.5), alpha=0.7) # Managed by rcParams
        ax_cdf.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 22))
        
        # KDE Plot
        kde_plot_title = f"KDE of R² using top Num_Ref={n}"
        if args.title_suffix:
            kde_plot_title += f" - {args.title_suffix}"
            
        fig_kde, ax_kde = plt.subplots(figsize=(10, 7))
        # fig_kde.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Managed by rcParams
        # ax_kde.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Managed by rcParams
        
        sns.kdeplot(data=plottable_r2_scores[n], ax=ax_kde,
                    color=PLOT_STYLE.get('line_color_default', 'blue'), 
                    fill=True, 
                    alpha=PLOT_STYLE.get('opacity_fill_kde', 0.4), 
                    linewidth=1.5,
                    edgecolor=None, # Set edgecolor to None to remove border
                    label=f"Ref Heads={len(plottable_r2_scores[n])}")
        
        ax_kde.set_xlabel("R²", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
        ax_kde.set_ylabel("Density", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
        ax_kde.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 28))
        ax_kde.set_xlim(left=min(0, r2_values_sorted_np.min() - 0.05 if r2_values_sorted_np.size > 0 else 0), 
                        right=1.0)  # Cap at 1.0
        # ax_kde.grid(True, linestyle=PLOT_STYLE.get('grid_linestyle', '--'), linewidth=PLOT_STYLE.get('grid_linewidth', 0.5), alpha=0.7) # Managed by rcParams
        if plottable_r2_scores[n]:
            ax_kde.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 22))
        
        # Save individual plots
        output_path_cdf_pdf = args.output_base + "_cdf.pdf"
        output_path_cdf_png = args.output_base + "_cdf.png"
        output_path_kde_pdf = args.output_base + "_kde.pdf"
        output_path_kde_png = args.output_base + "_kde.png"
        
        try:
            fig_cdf.savefig(output_path_cdf_pdf, format='pdf', bbox_inches='tight', facecolor=fig_cdf.get_facecolor())
            logging.info(f"CDF PDF plot saved to: {output_path_cdf_pdf}")
            fig_cdf.savefig(output_path_cdf_png, format='png', bbox_inches='tight', facecolor=fig_cdf.get_facecolor(), dpi=300)
            logging.info(f"CDF PNG plot saved to: {output_path_cdf_png}")
        except Exception as e:
            logging.error(f"Error saving CDF plots: {e}")
        plt.close(fig_cdf)
        
        if plottable_r2_scores[n]:
            try:
                fig_kde.savefig(output_path_kde_pdf, format='pdf', bbox_inches='tight', facecolor=fig_kde.get_facecolor())
                logging.info(f"KDE PDF plot saved to: {output_path_kde_pdf}")
                fig_kde.savefig(output_path_kde_png, format='png', bbox_inches='tight', facecolor=fig_kde.get_facecolor(), dpi=300)
                logging.info(f"KDE PNG plot saved to: {output_path_kde_png}")
            except Exception as e:
                logging.error(f"Error saving KDE plots: {e}")
        plt.close(fig_kde)
    
    # --- Combined KDE plot for multiple N values ---
    if multiple_n:
        fig_kde_combined, ax_kde_combined = plt.subplots(figsize=(12, 8))
        # fig_kde_combined.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Managed by rcParams
        # ax_kde_combined.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Managed by rcParams
        
        for i, n in enumerate(n_values):
            if not plottable_r2_scores[n]:
                continue
                
            sns.kdeplot(
                data=plottable_r2_scores[n], 
                ax=ax_kde_combined,
                color=PLOT_STYLE['colors_list'][i % len(PLOT_STYLE['colors_list'])],
                fill=True, 
                alpha=PLOT_STYLE.get('opacity_fill_kde', 0.4),
                linewidth=2,
                edgecolor=None, # Set edgecolor to None to remove border
                label=f"Ref Heads={n}"
            )
        
        ax_kde_combined.set_xlabel("R²", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
        ax_kde_combined.set_ylabel("Density", fontsize=PLOT_STYLE.get('label_fontsize', 28), fontweight='bold')
        ax_kde_combined.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 28))
        
        if all_r2_values:
            ax_kde_combined.set_xlim(left=global_min_r2, right=1.0)  # Cap at 1.0
            
        # ax_kde_combined.grid(True, linestyle=PLOT_STYLE.get('grid_linestyle', '--'), 
        #                      linewidth=PLOT_STYLE.get('grid_linewidth', 0.5), alpha=0.7) # Managed by rcParams
        ax_kde_combined.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 22), loc='best')
        
        # Save combined KDE plot
        output_path_kde_combined_pdf = args.output_base + "_combined_kde.pdf"
        output_path_kde_combined_png = args.output_base + "_combined_kde.png"
        
        try:
            fig_kde_combined.savefig(output_path_kde_combined_pdf, format='pdf', 
                                     bbox_inches='tight', facecolor=fig_kde_combined.get_facecolor())
            logging.info(f"Combined KDE PDF plot saved to: {output_path_kde_combined_pdf}")
            
            fig_kde_combined.savefig(output_path_kde_combined_png, format='png', 
                                     bbox_inches='tight', facecolor=fig_kde_combined.get_facecolor(), dpi=300)
            logging.info(f"Combined KDE PNG plot saved to: {output_path_kde_combined_png}")
        except Exception as e:
            logging.error(f"Error saving combined KDE plot: {e}")
        plt.close(fig_kde_combined)
    
    # --- Final Cleanup ---
    del activations_gpu
    cp.cuda.Stream.null.synchronize()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    logging.info(f"GPU memory after all operations: Used {mempool.used_bytes()/(1024**2):.2f}MB / Total {mempool.total_bytes()/(1024**2):.2f}MB")
    logging.info("Script finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 