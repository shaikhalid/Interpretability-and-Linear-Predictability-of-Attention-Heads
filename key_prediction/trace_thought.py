"""
Trace Thought Graph Construction

This script loads a 4D activation tensor from a pickle file of shape
(num_samples, num_layers, num_heads, activation_dim). It optionally
sub-samples the activations, moves data to GPU via CuPy if available,
and for every source-target head pair trains a linear regression model
to compute a connection strength metric (R^2 or cosine similarity).
A directed graph is built with nodes as (layer, head) tuples and edges
weighted by the computed connection values. The resulting graph is
saved in GML format, head degree rankings are logged, and an optional
visualization can be generated.
"""

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# from sklearn.metrics.pairwise import cosine_similarity # Replaced by custom/cupy calculation
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import argparse

# Add GPU imports
import cupy as cp
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.metrics import r2_score as cu_r2_score
# from cuml.metrics.pairwise import cosine_similarity as cu_cosine_similarity # Use custom calculation for now

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper for new batched function ---
def _get_lib_and_funcs(use_gpu):
    if use_gpu:
        return cp, cu_r2_score, cp.linalg.lstsq
    else:
        # For CPU R2 score, we need a wrapper to match cuML's default multioutput behavior
        def sklearn_r2_wrapper(y_true, y_pred):
            if y_true.ndim > 1 and y_true.shape[1] > 1: # Multi-output case
                return r2_score(y_true, y_pred, multioutput='uniform_average')
            return r2_score(y_true, y_pred) # Single output or 1D target
        return np, sklearn_r2_wrapper, np.linalg.lstsq

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description='Trace thought graph construction parameters')
    parser.add_argument('--data_file_path', type=str, default='processed_data/v_activations_4d_mmlu_college_bio.pkl',
                        help='Path to the data file')
    parser.add_argument('--matrix_key', type=str, default='k_matrix',
                        help='Key for the activation matrix in the pickle file')
    parser.add_argument('--num_layers', type=int, default=32,
                        help='Total number of layers')
    parser.add_argument('--connection_metric', type=str, choices=['r2', 'cosine'], default='r2',
                        help="Metric for connection: 'r2' or 'cosine'")
    parser.add_argument('--connection_threshold', type=float, default=0.50,
                        help='Threshold for the chosen metric (adjust based on metric)')
    parser.add_argument('--allow_intralayer_connections', action='store_true',
                        help='Allow connections between heads in the same layer', default=True)
    parser.add_argument('--output_graph_file', type=str, default='gmls/thought_graph.gml',
                        help='File to save the graph')
    parser.add_argument('--output_plot_file', type=str, default='gmls/thought_graph.png',
                        help='File to save the graph visualization')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to randomly select for analysis (default: all)')
    return parser.parse_args()

# Use args in place of constants
args = parse_args()
DATA_FILE_PATH = args.data_file_path
MATRIX_KEY = args.matrix_key
NUM_LAYERS = args.num_layers
CONNECTION_METRIC = args.connection_metric
CONNECTION_THRESHOLD = args.connection_threshold
ALLOW_INTRALAYER_CONNECTIONS = args.allow_intralayer_connections
OUTPUT_GRAPH_FILE = args.output_graph_file
OUTPUT_PLOT_FILE = args.output_plot_file

# --- Functions ---

def load_data(file_path, matrix_key):
    """Loads the activation data from a pickle file."""
    logging.info(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Determine the structure of the loaded data
        if isinstance(data, dict):
            logging.info(f"Loaded data is a dictionary. Keys: {list(data.keys())}")
            if matrix_key not in data:
                logging.error(f"Matrix key '{matrix_key}' not found in the data dictionary.")
                return None
            activations = data[matrix_key]
        elif isinstance(data, list):
            logging.info("Loaded data is a list. Assuming the activation matrix is the first element.")
            if not data:
                logging.error("Loaded list is empty.")
                return None
            activations = data[0] # Assume the matrix is the first element
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
        # Expected shape: (num_samples, num_layers, num_heads, activation_dim)
        if activations.ndim != 4:
            logging.error(f"Unexpected activation matrix dimensions: {activations.ndim}. Expected 4.")
            return None
        return activations
    except Exception as e:
        # Catch more specific errors if possible, e.g., PickleError
        logging.error(f"Error loading or processing data: {e}")
        return None

def calculate_metrics_for_layer_pair(
    source_layer_activations, # (num_samples, num_source_heads, activation_dim)
    target_layer_activations, # (num_samples, num_target_heads, activation_dim)
    metric_type,
    use_gpu=False
):
    """
    Calculates connection metrics between all head pairs of a source layer and a target layer
    using batched operations.

    Args:
        source_layer_activations (np.ndarray or cp.ndarray): (samples, src_heads, dim)
        target_layer_activations (np.ndarray or cp.ndarray): (samples, trg_heads, dim)
        metric_type (str): 'r2' or 'cosine'.
        use_gpu (bool): Whether to use GPU (cupy/cuml) or CPU (numpy/sklearn).

    Returns:
        list: A list of tuples (s_head_idx, t_head_idx, metric_value).
    """
    lib, r2_func_selected, lstsq_func = _get_lib_and_funcs(use_gpu)
    device = "GPU" if use_gpu else "CPU"
    num_samples, num_source_heads, activation_dim_src = source_layer_activations.shape
    _, num_target_heads, activation_dim_trg = target_layer_activations.shape

    if activation_dim_src != activation_dim_trg:
        logging.error(f"Source and target activation dimensions must match. Got {activation_dim_src} and {activation_dim_trg}.")
        return [] # Or raise error
    
    activation_dim = activation_dim_src

    # --- Prepare Y_all_targets (once) ---
    # target_layer_activations is expected to be a C-contiguous slice 
    # of shape (num_samples, num_heads, activation_dim) from the main transposed tensor.
    # It is also already on the correct device (GPU/CPU).
    Y_all_targets_dev = target_layer_activations
    logging.debug(f"Y_all_targets_dev assigned (assumed C-contiguous from transposed main activations on device {device}).")
    
    Y_all_targets_flat = Y_all_targets_dev.reshape(num_samples, -1)

    # --- Pre-allocate GPU arrays if use_gpu ---
    X_s_aug_gpu = None
    Y_pred_flat_gpu = None
    # Y_pred_batched_gpu will be a view of Y_pred_flat_gpu, no separate allocation needed here.

    if use_gpu:
        dtype = Y_all_targets_dev.dtype # Use consistent dtype
        X_s_aug_gpu = lib.empty((num_samples, activation_dim + 1), dtype=dtype)
        Y_pred_flat_gpu = lib.empty((num_samples, num_target_heads * activation_dim), dtype=dtype)

    results = []
    for s_head_idx in range(num_source_heads):
        current_rank = None
        current_svals = None
        current_Y_pred_batched = None
        # CPU specific iteration variables that need cleanup
        X_s_cpu_iter, intercept_cpu_iter, Y_pred_flat_cpu_iter = None, None, None

        # Cosine specific, created if metric_type == 'cosine'
        dot_products, norm_pred, norm_target, denominator, safe_mask, sims, average_similarities_per_target_head = [None]*7

        try:
            if use_gpu:
                # X_s_slice is a view of the source_layer_activations (already on GPU as per main() logic)
                X_s_slice = source_layer_activations[:, s_head_idx, :] 
                lib.copyto(X_s_aug_gpu[:, :-1], X_s_slice) # Copy slice into pre-allocated array
                X_s_aug_gpu[:, -1] = 1 # Set intercept column
                current_X_to_regr = X_s_aug_gpu
            else: # CPU path (allocates X_s_aug_cpu_iter per iteration)
                X_s_cpu_iter = lib.ascontiguousarray(source_layer_activations[:, s_head_idx, :])
                intercept_cpu_iter = lib.ones((num_samples, 1), dtype=X_s_cpu_iter.dtype)
                X_s_aug_cpu_iter = lib.hstack((X_s_cpu_iter, intercept_cpu_iter))
                current_X_to_regr = X_s_aug_cpu_iter

            # --- Perform Linear Regression ---
            if num_samples < current_X_to_regr.shape[1] and device == "CPU": # Check for CPU only for now
                logging.warning(f"Source head {s_head_idx} on CPU: Number of samples ({num_samples}) is less than "
                                f"number of features ({current_X_to_regr.shape[1]}). lstsq may be ill-conditioned.")
            # elif num_samples < current_X_to_regr.shape[1] and device == "GPU":
                # cuPy lstsq might handle this or give different errors. Let it try.

            # Batched Linear Regression: Solve X_s_aug @ beta_flat = Y_all_targets_flat
            # beta_flat will have shape (activation_dim + 1, num_target_heads * activation_dim)
            current_beta_flat, current_residuals, current_rank, current_svals = lstsq_func(current_X_to_regr, Y_all_targets_flat, rcond=None) # Use default rcond

            # --- Calculate Predictions ---
            if use_gpu:
                lib.matmul(current_X_to_regr, current_beta_flat, out=Y_pred_flat_gpu) # Use out parameter
                current_Y_pred_batched = Y_pred_flat_gpu.reshape(num_samples, num_target_heads, activation_dim)
            else: # CPU path (allocates Y_pred_flat_cpu_iter per iteration)
                Y_pred_flat_cpu_iter = lib.matmul(current_X_to_regr, current_beta_flat)
                current_Y_pred_batched = Y_pred_flat_cpu_iter.reshape(num_samples, num_target_heads, activation_dim)

            # --- Calculate Specific Metric using current_Y_pred_batched and Y_all_targets_dev ---
            if metric_type == 'r2':
                for t_head_idx in range(num_target_heads):
                    Y_t_current_head = Y_all_targets_dev[:, t_head_idx, :] # (num_samples, activation_dim)
                    Y_p_current_head = current_Y_pred_batched[:, t_head_idx, :] # (num_samples, activation_dim)
                    try:
                        score = float(r2_func_selected(Y_t_current_head, Y_p_current_head))
                        results.append((s_head_idx, t_head_idx, max(-1.0, score)))
                    except Exception as e_r2:
                        logging.error(f"Error calculating R^2 for S{s_head_idx}-T{t_head_idx} on {device}: {e_r2}")
                        results.append((s_head_idx, t_head_idx, -1.0))

            elif metric_type == 'cosine':
                # Calculate Average Cosine Similarity between Prediction (current_Y_pred_batched) and Actual Target (Y_all_targets_dev)
                # current_Y_pred_batched: (num_samples, num_target_heads, activation_dim)
                # Y_all_targets_dev:  (num_samples, num_target_heads, activation_dim)
                
                # Element-wise product and sum over activation_dim (axis=2)
                dot_products = lib.sum(current_Y_pred_batched * Y_all_targets_dev, axis=2) # (num_samples, num_target_heads)

                norm_pred = lib.linalg.norm(current_Y_pred_batched, axis=2)   # (num_samples, num_target_heads)
                norm_target = lib.linalg.norm(Y_all_targets_dev, axis=2) # (num_samples, num_target_heads)
                
                epsilon = 1e-9 # To avoid division by zero
                denominator = norm_pred * norm_target
                
                # Initialize sims array correctly based on library
                sims = lib.zeros_like(dot_products, dtype=lib.float64 if use_gpu else np.float64) # Use float64 for precision

                safe_mask = denominator > epsilon
                
                # Perform division safely
                # Note: cupy's advanced indexing for assignment differs from numpy's direct `out` and `where` in divide
                if use_gpu:
                    # For cupy, divide on slices and assign back
                    sims[safe_mask] = dot_products[safe_mask] / denominator[safe_mask]
                else: # numpy
                    # For numpy, can use out and where arguments for safe division
                    lib.divide(dot_products, denominator, out=sims, where=safe_mask)

                sims = lib.clip(sims, -1.0, 1.0) # Clip potential floating point inaccuracies
                
                # Average similarity over samples (axis=0) for each target head
                average_similarities_per_target_head = lib.mean(sims, axis=0) # (num_target_heads,)

                for t_head_idx in range(num_target_heads):
                    results.append((s_head_idx, t_head_idx, float(average_similarities_per_target_head[t_head_idx])))
            
            else:
                logging.error(f"Invalid metric_type for layer pair: {metric_type}")
                for t_head_idx in range(num_target_heads):
                    results.append((s_head_idx, t_head_idx, -1.0))
                # No need to break, loop for s_head_idx continues

        except (np.linalg.LinAlgError if not use_gpu else cp.linalg.LinAlgError) as e_lstsq:
             logging.warning(f"lstsq failed for source_head {s_head_idx} on {device}: {e_lstsq}. Assigning -1.0 to its connections.")
             for t_head_idx in range(num_target_heads):
                 results.append((s_head_idx, t_head_idx, -1.0)) # Default score for all targets from this source
        except Exception as e_outer: # Catch any other unexpected errors for this source head
            logging.error(f"Unexpected error processing source_head {s_head_idx} on {device}: {e_outer}")
            for t_head_idx in range(num_target_heads):
                results.append((s_head_idx, t_head_idx, -1.0))

        finally:
            
            if use_gpu:
                # Iteration-specific GPU arrays to delete by name
                # current_X_to_regr points to X_s_aug_gpu (pre-allocated), no del here.
                # current_Y_pred_batched is a view of Y_pred_flat_gpu (pre-allocated), no del here.
                # X_s_slice is a view.
                del current_beta_flat, current_residuals, current_rank, current_svals, Y_t_current_head, Y_p_current_head
                if metric_type == 'cosine':
                    # These are created new within the cosine block if it runs
                    if dot_products is not None: del dot_products
                    if norm_pred is not None: del norm_pred
                    if norm_target is not None: del norm_target
                    if denominator is not None: del denominator
                    if safe_mask is not None: del safe_mask
                    if sims is not None: del sims
                    if average_similarities_per_target_head is not None: del average_similarities_per_target_head
                
                cp.cuda.Stream.null.synchronize()
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                cp.cuda.Device().synchronize() # User added this line
                
                used_mb = mempool.used_bytes() / (1024**2); total_mb = mempool.total_bytes() / (1024**2)
                logging.debug(f"GPU memory (iter end s_idx {s_head_idx}): Used: {used_mb:.2f}MB, Total: {total_mb:.2f}MB")
            else: # CPU path cleanup (rely mostly on Python GC for iter-specific temps)
                # Explicitly delete larger arrays created in CPU iteration if desired
                del current_beta_flat, current_residuals, current_rank, current_svals
                del X_s_cpu_iter, intercept_cpu_iter, current_X_to_regr # current_X_to_regr was X_s_aug_cpu_iter
                del Y_pred_flat_cpu_iter
                if metric_type == 'cosine':
                    if dot_products is not None: del dot_products
                    if norm_pred is not None: del norm_pred
                    if norm_target is not None: del norm_target
                    if denominator is not None: del denominator
                    if safe_mask is not None: del safe_mask
                    if sims is not None: del sims
                    if average_similarities_per_target_head is not None: del average_similarities_per_target_head
    
    # --- Cleanup function-scoped pre-allocated/once-allocated arrays ---
    if use_gpu:
        logging.debug("Cleaning up pre-allocated and function-scoped GPU arrays after loop...")
        del X_s_aug_gpu, Y_pred_flat_gpu 
        del Y_all_targets_dev, Y_all_targets_flat 
        # current_Y_pred_batched was a view, X_s_slice was a view
        cp.cuda.Stream.null.synchronize()
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        cp.cuda.Device().synchronize() # Sync again after final cleanup
        used_mb = mempool.used_bytes() / (1024**2); total_mb = mempool.total_bytes() / (1024**2)
        print(f"GPU memory (func end): Used: {used_mb:.2f}MB, Total: {total_mb:.2f}MB")
    else: # CPU path
        logging.debug("Cleaning up function-scoped CPU arrays after loop...")
        del Y_all_targets_dev, Y_all_targets_flat 
        # Other CPU arrays were iteration specific and handled by del or GC

    return results

def visualize_graph(graph, output_path):
    """Visualizes the thought graph using matplotlib and networkx."""
    if not graph.nodes:
        logging.warning("Graph is empty, skipping visualization.")
        return

    logging.info(f"Visualizing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges...")
    plt.figure(figsize=(20, 20))

    # Position nodes in layers
    pos = {}
    layer_counts = {}
    heads_in_layer = {}
    
    # First pass to count heads in each layer and get node info
    for node in graph.nodes():
        layer, head = node
        if layer not in layer_counts:
            layer_counts[layer] = 0
            heads_in_layer[layer] = []
        heads_in_layer[layer].append(head)
        layer_counts[layer] += 1
    
    # Second pass to position nodes with consistent y-positioning based on head index
    for node in graph.nodes():
        layer, head = node
        # Position based on head index to maintain consistent vertical positioning
        pos[node] = (layer, -head)  # Negative to put lower head indices at the top

    # Improve visualization
    options = {
        "font_size": 8,
        "node_size": 500,
        "node_color": "skyblue",
        "edge_color": "gray",
        "linewidths": 1,
        "width": 1,
        "with_labels": True,
        "arrows": True,
        "arrowstyle": "-|>",
        "arrowsize": 10,
    }

    nx.draw(graph, pos, **options)
    
    # Get the unique head indices for y-ticks
    all_heads = sorted(list(set([head for _, head in graph.nodes()])))
    y_positions = [-h for h in all_heads]  # Convert to the negative values we used for positioning
    
    plt.title("Thought Trace Graph (Layer -> Head Connections)", fontsize=16, pad=20)
    plt.xlabel("Layer Index", fontsize=14, labelpad=15)
    plt.ylabel("Head Index", fontsize=14, labelpad=15)
    plt.xticks(range(NUM_LAYERS), fontsize=12)  # Ensure all layer indices are shown
    plt.yticks(y_positions, all_heads, fontsize=12)  # Show actual head indices
    plt.grid(axis='both', linestyle='--', alpha=0.6)

    # Add a legend for edges
    plt.figtext(0.01, 0.01, f"Edge = Strong connection ({CONNECTION_METRIC} > {CONNECTION_THRESHOLD})", 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    try:
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save graph visualization: {e}")
    plt.close()  # Close the plot figure

# --- Main Execution ---

def main():
    # Add argument parsing for GPU flag (optional, can be hardcoded for now)
    use_gpu = True # Hardcode to True for now, or use argparse
    if use_gpu:
        try:
            # Check if GPU is available and initialize cuML (optional check)
            cp.cuda.runtime.getDeviceCount()
            logging.info("GPU detected. Running in GPU mode.")
        except cp.cuda.runtime.CUDARuntimeError:
            logging.error("No CUDA-enabled GPU found or cuPy/CUDA setup issue. Falling back to CPU.")
            use_gpu = False
        except ImportError:
             logging.error("cuML/cuPy not installed. Falling back to CPU.")
             use_gpu = False


    activations_np = load_data(DATA_FILE_PATH, MATRIX_KEY)
    if activations_np is None:
        return

    total_samples = activations_np.shape[0]
    use_samples = total_samples
    sample_indices = None

    if args.num_samples is not None and args.num_samples > 0 and args.num_samples < total_samples:
        logging.info(f"Randomly sampling {args.num_samples} out of {total_samples} available samples.")
        sample_indices = np.random.choice(total_samples, args.num_samples, replace=False)
        activations_np = activations_np[sample_indices, :, :, :]
    elif args.num_samples is not None:
        logging.warning(f"--num_samples ({args.num_samples}) is invalid or >= total samples ({total_samples}). Using all samples.")

    # activations_intermediate_ref will point to the current representation of the data (CPU or GPU)
    activations_intermediate_ref = activations_np 

    if use_gpu:
        logging.info("Converting activation data to GPU (CuPy array)...")
        try:
            # Try to convert to GPU
            activations_on_gpu = cp.asarray(activations_np) 
            # If successful, update activations_intermediate_ref to point to GPU array
            activations_intermediate_ref = activations_on_gpu
            cp.cuda.Device().synchronize()
            logging.info("Data conversion to GPU complete.")
        except Exception as e:
            logging.error(f"Failed to convert data to GPU: {e}. Will proceed with CPU.")
            use_gpu = False # Explicitly fall back
            activations_intermediate_ref = activations_np # Ensure ref is CPU if GPU fails

    # Determine the library (cp or np) and device string based on final use_gpu status
    current_lib = cp if use_gpu else np
    device_str = "GPU" if use_gpu else "CPU"

    logging.info(f"Transposing activations on {device_str} from (N, L, H, d) to (L, N, H, d) and ensuring C-contiguity...")
    # Original shape was (num_samples, num_layers, num_heads, activation_dim)
    # Transpose to (num_layers, num_samples, num_heads, activation_dim)
    activations = current_lib.transpose(activations_intermediate_ref, (1, 0, 2, 3)).copy()
    
    if use_gpu:
        cp.cuda.Device().synchronize() # Ensure transpose and copy are done on GPU
        # If GPU path was successful and activations_np was the original source for activations_on_gpu,
        # and activations_on_gpu was then used for the transpose, we can delete activations_np.
        if 'activations_np' in locals() and activations_np is not None:
            del activations_np 
        if 'activations_on_gpu' in locals():
            del activations_on_gpu
            logging.debug("Deleted original CPU activations_np after successful GPU transpose and copy.")
    elif 'activations_np' in locals() and activations_np is not activations_intermediate_ref and activations_np is not activations :
        # If CPU path, and activations_np was copied to activations_intermediate_ref (which is unlikely here)
        # or to activations, ensure it's cleaned up if no longer the main reference.
        # This case is less likely with current direct assignment flow for CPU.
        # The primary concern is deleting activations_np after GPU successfully takes over.
        pass


    logging.info(f"Activations transposed and copied on {device_str}. New shape: {activations.shape}")

    # Clean up intermediate reference if it's different from the final `activations` variable.
    if 'activations_intermediate_ref' in locals() and activations_intermediate_ref is not activations:
        del activations_intermediate_ref


    # Shape unpacking updated for new (L, N, H, d) structure
    num_layers_data, num_samples_actual, num_heads, activation_dim = activations.shape 
    logging.info(f"Data Details (post-transpose): Layers={num_layers_data}, Samples={num_samples_actual}, Heads={num_heads}, Dim={activation_dim}")

    if num_layers_data != NUM_LAYERS:
        logging.warning(f"Configured NUM_LAYERS ({NUM_LAYERS}) differs from data ({num_layers_data}). Using data's layer count.")
        effective_num_layers = num_layers_data
    else:
        effective_num_layers = NUM_LAYERS


    thought_graph = nx.DiGraph()
    metric_name = "R^2" if CONNECTION_METRIC == 'r2' else "Cosine Similarity"
    logging.info(f"Using {metric_name} with threshold {CONNECTION_THRESHOLD} to determine connections.")
    if not ALLOW_INTRALAYER_CONNECTIONS:
        logging.info("Intra-layer connections (connections within the same layer) are disabled.")
    else:
        logging.info("Intra-layer connections are enabled.")


    intralayer_connections = 0
    interlayer_connections = 0

    # Add all nodes to the graph first to ensure they exist even if they have no connections
    for layer_idx in range(effective_num_layers):
        for head_idx in range(num_heads):
            thought_graph.add_node((layer_idx, head_idx))

    # Iterate through all possible target layers
    for target_layer_idx in range(effective_num_layers):
        logging.info(f"Processing connections TO Layer {target_layer_idx}...")
        # Iterate through all possible source layers (from layer 0 up to and including target_layer)
        # This ensures that connections are directed from earlier/same layers to target_layer_idx
        for source_layer_idx in range(target_layer_idx + 1):
            current_processing_msg = f"  Calculating metrics from Layer {source_layer_idx} TO Layer {target_layer_idx}"
            if source_layer_idx == target_layer_idx:
                current_processing_msg += " (intra-layer)"
            logging.info(current_processing_msg)

            # Extract full layer activations based on whether using GPU or CPU
            # The `activations` tensor is now (L, N, H, d)
            # Slices will be (N, H, d) and should be C-contiguous due to earlier .copy()
            current_source_layer_activations = activations[source_layer_idx, :, :, :]
            current_target_layer_activations = activations[target_layer_idx, :, :, :]
            
            # Basic checks before calling the potentially expensive calculation
            if num_samples_actual == 0:
                logging.warning(f"Skipping L{source_layer_idx}->L{target_layer_idx}: zero samples.")
                continue
            if activation_dim == 0:
                logging.warning(f"Skipping L{source_layer_idx}->L{target_layer_idx}: zero activation dimension.")
                continue

            # Call the new batched metric calculation function
            layer_pair_metrics = calculate_metrics_for_layer_pair(
                current_source_layer_activations,
                current_target_layer_activations,
                CONNECTION_METRIC,
                use_gpu=use_gpu
            )

            # Add results to the graph
            for s_head_idx_in_layer, t_head_idx_in_layer, metric_value in layer_pair_metrics:
                source_node = (source_layer_idx, s_head_idx_in_layer)
                target_node = (target_layer_idx, t_head_idx_in_layer)

                # Skip self-connection (a head to itself)
                # This check is important if source_layer_idx == target_layer_idx
                if source_node == target_node:
                    continue

                # Skip intralayer connections if this pair is within the same layer AND it's disabled
                # This applies only if source_layer_idx == target_layer_idx
                if not ALLOW_INTRALAYER_CONNECTIONS and source_layer_idx == target_layer_idx:
                    continue
                
                # Add edge with calculated metric
                # The nodes should already exist due to pre-population.
                thought_graph.add_edge(source_node, target_node, weight=metric_value, metric=CONNECTION_METRIC)
                
                # Increment counters based on the connection type after filters
                if source_layer_idx == target_layer_idx:
                    intralayer_connections += 1
                else:
                    interlayer_connections += 1
    
    # --- Output ---
    logging.info("Finished processing all layers.")
    logging.info(f"Final graph has {thought_graph.number_of_nodes()} nodes and {thought_graph.number_of_edges()} edges.")
    logging.info(f"Total Intralayer Connections: {intralayer_connections}")
    logging.info(f"Total Interlayer Connections: {interlayer_connections}")

    # --- Head Ranking Calculation ---
    logging.info("Calculating head rankings based on total degree (in + out)...")
    head_degrees = {}
    head_in_degrees = {} # Added for in-degree ranking
    head_out_degrees = {} # Added for out-degree ranking

    for layer_idx in range(effective_num_layers):
        for head_idx in range(num_heads):
            node = (layer_idx, head_idx)
            if thought_graph.has_node(node): # Only consider nodes present in the graph
                 in_degree = thought_graph.in_degree(node)
                 out_degree = thought_graph.out_degree(node)
                 total_degree = in_degree + out_degree

                 # Aggregate total degree
                 if head_idx not in head_degrees:
                     head_degrees[head_idx] = 0
                 head_degrees[head_idx] += total_degree

                 # Aggregate in-degree
                 if head_idx not in head_in_degrees:
                     head_in_degrees[head_idx] = 0
                 head_in_degrees[head_idx] += in_degree

                 # Aggregate out-degree
                 if head_idx not in head_out_degrees:
                     head_out_degrees[head_idx] = 0
                 head_out_degrees[head_idx] += out_degree


    # Sort heads by degrees in descending order
    ranked_heads_total = sorted(head_degrees.items(), key=lambda item: item[1], reverse=True)
    ranked_heads_in = sorted(head_in_degrees.items(), key=lambda item: item[1], reverse=True)
    ranked_heads_out = sorted(head_out_degrees.items(), key=lambda item: item[1], reverse=True)

    logging.info("--- Head Ranking (Total Degree: In + Out) ---")
    if not ranked_heads_total:
        logging.info("No head degrees calculated (graph might be empty or nodes not processed).")
    else:
        for head_idx, total_degree in ranked_heads_total:
             logging.info(f"  Head {head_idx}: {total_degree}")
    logging.info("---------------------------------------------\n")

    logging.info("--- Head Ranking (In-Degree) ---")
    if not ranked_heads_in:
        logging.info("No head in-degrees calculated.")
    else:
        for head_idx, in_degree_val in ranked_heads_in:
            logging.info(f"  Head {head_idx}: {in_degree_val}")
    logging.info("-------------------------------\n")

    logging.info("--- Head Ranking (Out-Degree) ---")
    if not ranked_heads_out:
         logging.info("No head out-degrees calculated.")
    else:
         for head_idx, out_degree_val in ranked_heads_out:
             logging.info(f"  Head {head_idx}: {out_degree_val}")
    logging.info("--------------------------------\n")


    # --- End Head Ranking Calculation ---

    # Save the graph to a file (e.g., GML format)
    try:
        nx.write_gml(thought_graph, OUTPUT_GRAPH_FILE)
        logging.info(f"Thought graph saved to {OUTPUT_GRAPH_FILE}")
    except Exception as e:
        logging.error(f"Failed to save graph to GML: {e}")

    # Visualize the graph (disabled for fully connected graph)
    # visualize_graph(thought_graph, OUTPUT_PLOT_FILE)


if __name__ == "__main__":
    main()