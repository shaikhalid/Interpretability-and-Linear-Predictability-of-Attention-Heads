import pickle
import numpy as np
import pandas as pd
import time # Added time import
import os # Added os import for directory creation
import matplotlib.pyplot as plt # Added matplotlib import for plotting
from typing import List, Dict, Union, Tuple
from itertools import combinations # Needed for CKA pairs
# Import pastel theme
from visualisation_helper.pastel_theme import PASTEL_COLORS, PLOT_STYLE

def load_all_kv_caches(pickle_path: str) -> Dict[Union[int, str], Dict[int, np.ndarray]]:
    """
    Loads K-cache matrices from a pickle file containing records,
    aggregates them per layer for each unique batch_id found in the data.

    Args:
        pickle_path: Path to the pickle file.

    Returns:
        A dictionary where keys are batch_ids (Union[int, str]) and values are
        dictionaries mapping layer_id (int) to the aggregated K-cache matrix (np.ndarray)
        for that specific batch.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        ValueError: If the pickle file is missing required columns ('layer_id', 'k_matrix', 'batch_id'),
                    if no batch IDs are found, or if no valid layers can be aggregated for any batch.
        Exception: For other potential loading or processing errors.
    """
    loaded_data = []
    try:
        # Load all objects sequentially from the pickle file
        with open(pickle_path, 'rb') as f:
            while True:
                try:
                    data_chunk = pickle.load(f)
                    # Handle case where the file might contain lists of records
                    if isinstance(data_chunk, list):
                        loaded_data.extend(data_chunk)
                    else:
                        # Assume it's a single record if not a list
                        loaded_data.append(data_chunk)
                except EOFError:
                    # End of file reached
                    break
        print(f"Successfully loaded raw data from {pickle_path}. Found {len(loaded_data)} records.")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(loaded_data)
        print("Converted raw data to DataFrame.")

        # --- Validation for required columns ---
        required_cols = ['layer_id', 'k_matrix', 'batch_id']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Pickle file records are missing required columns: {missing}. Available columns: {df.columns.tolist()}")

        unique_batch_ids = df['batch_id'].unique()
        if not unique_batch_ids.any():
             raise ValueError("No batch IDs found in the 'batch_id' column.")

        print(f"Found {len(unique_batch_ids)} unique batch IDs. Processing each batch...")

        # --- Process and aggregate K-cache matrices per batch, then per layer ---
        all_batch_caches: Dict[Union[int, str], Dict[int, np.ndarray]] = {}
        # Ensure layer_id is integer for grouping within batches
        df['layer_id'] = df['layer_id'].astype(int)

        # Group by batch_id first
        grouped_by_batch = df.groupby('batch_id')

        for batch_id, batch_df in grouped_by_batch:
            print(f"Processing batch_id: {batch_id}...")
            aggregated_k_cache_for_batch: Dict[int, np.ndarray] = {}
            # Group this batch's data by layer_id
            grouped_by_layer = batch_df.groupby('layer_id')

            print(f"  Found {len(grouped_by_layer)} unique layers for batch '{batch_id}'. Aggregating K-cache matrices...")

            for layer_id, group in grouped_by_layer:
                layer_matrices = []
                # Iterate through k_matrix entries for the current layer within the current batch
                for k_matrix in group['k_matrix']:
                    if isinstance(k_matrix, np.ndarray):
                        # Expected shapes might be (1, num_tokens, embed_dim) or (1, 1, embed_dim)
                        # We want to reshape to (num_tokens, embed_dim) before concatenating
                        if k_matrix.ndim == 3 and k_matrix.shape[0] == 1:
                            # Squeeze the first dimension if it's singular
                            reshaped_matrix = k_matrix.squeeze(axis=0)
                            # Ensure it's now 2D before adding
                            if reshaped_matrix.ndim == 2:
                                 # Make sure it's not empty after squeeze
                                 if reshaped_matrix.shape[0] > 0 and reshaped_matrix.shape[1] > 0:
                                    layer_matrices.append(reshaped_matrix)
                                 else:
                                    print(f"  Warning (Batch {batch_id}, Layer {layer_id}): Matrix became empty after squeeze. Original: {k_matrix.shape}. Skipping.")
                            else:
                                 print(f"  Warning (Batch {batch_id}, Layer {layer_id}): Unexpected shape after squeeze: {reshaped_matrix.shape}. Original: {k_matrix.shape}. Skipping this matrix.")
                        elif k_matrix.ndim == 2:
                             # Accept if it's already 2D and not empty
                             if k_matrix.shape[0] > 0 and k_matrix.shape[1] > 0:
                                layer_matrices.append(k_matrix)
                             else:
                                print(f"  Warning (Batch {batch_id}, Layer {layer_id}): Skipping empty 2D matrix. Shape: {k_matrix.shape}.")
                        else:
                            print(f"  Warning (Batch {batch_id}, Layer {layer_id}): Unexpected matrix dimensions ({k_matrix.ndim}). Shape: {k_matrix.shape}. Skipping this matrix.")
                    else:
                        print(f"  Warning (Batch {batch_id}, Layer {layer_id}): Non-numpy array found in k_matrix column. Type: {type(k_matrix)}. Skipping.")

                if not layer_matrices:
                    print(f"  Warning (Batch {batch_id}, Layer {layer_id}): No valid K-matrices found or processed. Skipping this layer for this batch.")
                    continue

                # Concatenate all valid matrices for the current layer vertically (axis=0)
                try:
                    # Check if all matrices have the same number of columns (embedding dim)
                    embed_dim = layer_matrices[0].shape[1]
                    if not all(mat.shape[1] == embed_dim for mat in layer_matrices):
                        print(f"  Error (Batch {batch_id}, Layer {layer_id}): Inconsistent embedding dimension. Skipping concatenation.")
                        continue

                    concatenated_layer_matrix = np.concatenate(layer_matrices, axis=0)
                    aggregated_k_cache_for_batch[layer_id] = concatenated_layer_matrix
                    # print(f"    Layer {layer_id}: Aggregated shape = {concatenated_layer_matrix.shape}") # Optional debug print
                except ValueError as e:
                    print(f"  Error concatenating matrices for Batch {batch_id}, Layer {layer_id}: {e}. Skipping this layer for this batch.")

            if not aggregated_k_cache_for_batch:
                print(f"Warning: No valid layers could be aggregated for batch_id '{batch_id}'. Skipping this batch.")
            else:
                all_batch_caches[batch_id] = aggregated_k_cache_for_batch
                print(f"  Successfully aggregated K-cache for {len(aggregated_k_cache_for_batch)} layers in batch {batch_id}.")


        if not all_batch_caches:
            raise ValueError("No valid batches could be processed from the pickle file.")

        print(f"\nSuccessfully processed {len(all_batch_caches)} batches.")
        return all_batch_caches

    except FileNotFoundError:
        print(f"Error: Pickle file not found at {pickle_path}")
        raise
    except ValueError as e: # Catch specific errors like missing columns or aggregation failure
        print(f"Data processing error: {e}")
        raise
    except Exception as e:
        print(f"Error loading or processing pickle file: {e}")
        raise


def get_required_rank_ratio(matrix: np.ndarray, variance_threshold: float = 0.95) -> Tuple[float, int, float, np.ndarray]:
    """
    Calculates the rank ratio needed to preserve a certain variance threshold,
    the required rank itself, the estimated memory savings, and the singular values.
    Also returns the time taken for the SVD computation.

    Args:
        matrix: The input matrix.
        variance_threshold: The target variance preservation threshold (0.0 to 1.0).

    Returns:
        A tuple containing:
        - rank_ratio (float): required_rank / total_possible_rank.
        - required_rank (int): The rank needed to meet the variance threshold.
        - memory_saving_ratio (float): Estimated memory saving (0.0 to 1.0).
        - s (np.ndarray): The singular values of the matrix.
        - svd_duration (float): Time taken for np.linalg.svd in seconds.
        Returns (0.0, 0, 0.0, np.array([]), 0.0) if calculation is not possible.
    """
    if matrix.ndim != 2 or matrix.size == 0 or not (0 < variance_threshold <= 1.0):
        print("Warning: Invalid matrix shape, size, or variance threshold. Returning default values.")
        return 0.0, 0, 0.0, np.array([]), 0.0 # Added svd_duration default

    rows, cols = matrix.shape
    total_possible_rank = min(rows, cols) # Rank is limited by the smaller dimension
    svd_duration = 0.0 # Initialize duration

    # Perform SVD - we only need the singular values 's'
    try:
        # Use compute_uv=False for efficiency if only singular values are needed
        start_time = time.time()
        s = np.linalg.svd(matrix, compute_uv=False)
        svd_duration = time.time() - start_time # Calculate duration

        # Singular values are sorted in descending order
        # Variance is proportional to the square of singular values
        squared_singular_values = s**2
        total_variance = np.sum(squared_singular_values)

        if total_variance <= 1e-10: # Use a small epsilon for zero check
             print("Warning: Total variance is close to zero. Returning default values.")
             return 0.0, 0, 0.0, s, svd_duration # Return calculated s and duration

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(squared_singular_values) / total_variance

        # Find the smallest rank 'r' that meets the threshold
        required_rank_indices = np.where(cumulative_variance >= variance_threshold)[0]
        if len(required_rank_indices) == 0:
            # This can happen if threshold is 1.0 and there's numerical noise
            required_rank = total_possible_rank # Need all ranks
        else:
            required_rank = required_rank_indices[0] + 1


        # Calculate rank ratio based on the number of columns (features)
        # This aligns with how it might be used in feature reduction contexts.
        feature_dim = cols # Use cols as the denominator for the ratio
        if feature_dim == 0:
             print("Warning: Matrix has zero columns. Returning rank ratio 0.0.")
             rank_ratio = 0.0
        else:
             # Rank ratio represents compression relative to original feature dimension
             rank_ratio = required_rank / feature_dim


        # --- Memory Savings Calculation ---
        original_size = rows * cols
        # Approximate size using truncated SVD components (U_r * S_r * V_r^T)
        # Storage needed: (rows * r) + r + (r * cols)
        reduced_size = (rows * required_rank) + required_rank + (required_rank * cols)

        if original_size > 0:
            memory_saving_ratio = 1.0 - (reduced_size / original_size)
            # Clamp saving ratio between 0 and 1, as reduction might not always save space
            # if required_rank is very high relative to original dimensions.
            memory_saving_ratio = max(0.0, min(1.0, memory_saving_ratio))
        else:
            memory_saving_ratio = 0.0 # No savings if original size is zero


        return rank_ratio, required_rank, memory_saving_ratio, s, svd_duration # Return duration

    except np.linalg.LinAlgError as e:
        print(f"SVD computation failed: {e}")
        return 0.0, 0, 0.0, np.array([]), 0.0 # Return default duration on error
    except Exception as e:
        print(f"An unexpected error occurred in get_required_rank_ratio: {e}")
        return 0.0, 0, 0.0, np.array([]), 0.0 # Return default duration on error

def generate_stride_groups(num_layers: int, group_size: int) -> List[List[int]]:
    """
    Generates groups of layers based on a fixed stride.
    (Function remains the same as before)
    """
    groups = []
    for i in range(0, num_layers, group_size):
        group = list(range(i, min(i + group_size, num_layers)))
        if group: # Avoid adding empty groups if num_layers is not multiple of group_size
             groups.append(group)
    return groups

def expand_range_groups(range_groups: List[List[int]]) -> List[List[int]]:
    """
    Expands groups defined by [start, end] ranges into lists of explicit layer indices.
    (Function remains the same as before)
    """
    expanded_groups = []
    for group_range in range_groups:
        if len(group_range) == 2 and group_range[0] <= group_range[1]:
            start, end = group_range
            expanded_groups.append(list(range(start, end + 1)))
        else:
            print(f"Warning: Skipping invalid group range format: {group_range}. Expected [start, end].")
    return expanded_groups


# --- UPDATED: Evaluation Function ---
def evaluate_grouping_strategy(
    kv_cache: Dict[int, np.ndarray], # Expect Dict for easier layer access
    groups: List[List[int]], # Expects expanded groups (list of indices)
    variance_threshold: float = 0.95
    # Removed max_tokens_for_eval
) -> Tuple[float, List[float], float, List[float], float, List[float], float]: # UPDATED Return Type -> Added Avg Mem Saving, List of Mem Savings, List of SVD times, Max SVD time, Total SVD Time
    """
    Evaluates a given grouping strategy by calculating the average rank ratio
    and the average intra-group pairwise CKA similarity, using data from a single sample.

    Args:
        kv_cache: The loaded KV cache data for a single sample
                  (Dictionary mapping layer_id -> matrix).
        groups: A list of lists representing the layer groups (indices listed explicitly).
        variance_threshold: The variance preservation target for rank calculation.
        # Removed max_tokens_for_eval

    Returns:
        A tuple containing:
        - Average Rank Ratio: Lower is better (more compressible).
        - List of rank ratios for each group evaluated.
        - Average Memory Saving Ratio: Higher is better (more compression).
        - List of memory saving ratios for each group evaluated.
        - List of SVD computation times (seconds) for each group evaluated.
        - Maximum SVD time among all groups for this sample (potential bottleneck).
        - Total SVD time across all groups for this sample.
        Returns (1.0, [], 0.0, [], [], 0.0, 0.0) if errors occur or no valid groups are processed.
    """
    all_rank_ratios = []
    all_memory_savings = [] # NEW: Store memory savings per group
    all_svd_times = []      # NEW: Store SVD times per group
    # Removed rng = np.random.default_rng()

    for group_idx, group in enumerate(groups):
        if not group: # Need at least 1 layer for rank
            print(f"Skipping empty or single-layer group: {group}")
            continue

        # --- 0. Check Layer Existence and Prepare Caches --- (Simplified)
        group_caches_list = []
        valid_group = True
        for layer_idx in group:
            if layer_idx not in kv_cache:
                print(f"Warning: Layer index {layer_idx} not found in cache for group {group}. Skipping this group entirely.")
                valid_group = False
                break
            group_caches_list.append(kv_cache[layer_idx])

        # Skip group if any layer was missing
        if not valid_group:
            continue

        # --- 1. Rank Ratio Calculation --- (Uses group_caches_list)
        if group_caches_list: # Should always be true if valid_group is true
            try:
                # Horizontally concatenate the caches from the single sample
                concatenated_cache = np.concatenate(group_caches_list, axis=1)
                # UPDATED: Get rank ratio, memory saving, and SVD duration
                rank_ratio, _, memory_saving, _, svd_duration = get_required_rank_ratio(concatenated_cache, variance_threshold)
                all_rank_ratios.append(rank_ratio)
                all_memory_savings.append(memory_saving) # Store saving
                all_svd_times.append(svd_duration)       # Store SVD time
            except ValueError as e:
                 print(f"Error concatenating caches for rank ratio/saving/time in group {group}: {e}. Skipping stats for group.")
            except Exception as e:
                print(f"Unexpected error during rank ratio calculation for group {group}: {e}")


    # Calculate overall averages and totals
    average_rank_ratio = np.mean(all_rank_ratios) if all_rank_ratios else 1.0 # Default to worst case
    average_memory_saving = np.mean(all_memory_savings) if all_memory_savings else 0.0 # Default to no savings
    max_svd_time = max(all_svd_times) if all_svd_times else 0.0 # Max SVD time (bottleneck)
    total_svd_time = sum(all_svd_times) if all_svd_times else 0.0 # Total SVD time

    # RETURN avg memory saving, list of savings, list of times, max time, total time
    return average_rank_ratio, all_rank_ratios, average_memory_saving, all_memory_savings, all_svd_times, max_svd_time, total_svd_time

# --- Configuration ---
PICKLE_FILE_PATH = 'kv_pickles/k_cache_data_mmlu_pro_math_l8b.pkl' # <--- IMPORTANT: REPLACE THIS
VARIANCE_TARGET = 0.95 # Target variance preservation (e.g., 95%)
RANDOM_SEED = 41 # Seed for reproducible random sample selection
# MAX_TOKENS_FOR_EVAL = 10000 # NEW: Subsample to this many tokens for eval (e.g., 10k). Set to None to disable. # Removed
NUM_SAMPLES_TO_EVAL = 100 # <--- NEW: Number of random samples to evaluate

# --- Define Your Custom Grouping Strategy ---
# Format: List of [start_layer_index, end_layer_index] (inclusive)
# Example: Replace this with the actual groups from your method
custom_groups_ranges = [[0, 2], [3, 8], [9, 14], [15, 20], [21, 25], [26, 31]]

# --- Main Execution ---
if __name__ == "__main__":
    results = {} # Store results for summary
    all_samples_results = {
        'Stride-2 (xKV)': {'rank_ratios': [], 'memory_savings': [], 'max_svd_times': [], 'total_svd_times': []}, # Added timing metrics
        'Stride-4 (xKV)': {'rank_ratios': [], 'memory_savings': [], 'max_svd_times': [], 'total_svd_times': []}, # Added timing metrics
        'Stride - 2 to 6 (Ours)': {'rank_ratios': [], 'per_group_ranks': [],
                   'memory_savings': [], 'per_group_savings': [],
                   'max_svd_times': [], 'total_svd_times': [], 'per_group_svd_times': []} # Added timing metrics
    }
    try:
        # 1. Load All Data
        print(f"Loading all K-cache data from: {PICKLE_FILE_PATH}")
        all_k_caches = load_all_kv_caches(PICKLE_FILE_PATH)

        if not isinstance(all_k_caches, dict) or not all_k_caches:
             raise ValueError("Failed to load K-cache or loaded data is not a dictionary.")

        all_batch_ids = list(all_k_caches.keys())
        num_total_samples = len(all_batch_ids)
        print(f"Total samples available: {num_total_samples}")

        if num_total_samples == 0:
            raise ValueError("No samples found in the loaded data.")

        # Determine number of layers (use the first sample's keys, assuming consistency)
        first_batch_id = all_batch_ids[0]
        num_layers = max(all_k_caches[first_batch_id].keys()) + 1 if all_k_caches[first_batch_id] else 0

        if num_layers == 0:
            print("Error: No layers found in the first sample's cache.")
            exit()

        print(f"\nDetermined number of layers: {num_layers}")
        print(f"Target Variance Preservation: {VARIANCE_TARGET * 100:.1f}%\n")

        # --- Sample Selection ---
        # Ensure we don't try to select more samples than available
        num_samples_to_select = min(NUM_SAMPLES_TO_EVAL, num_total_samples)
        if num_samples_to_select < NUM_SAMPLES_TO_EVAL:
            print(f"Warning: Requested {NUM_SAMPLES_TO_EVAL} samples, but only {num_total_samples} are available. Evaluating on {num_samples_to_select} samples.")

        # Set seed for reproducibility of sample selection
        np.random.seed(RANDOM_SEED)
        selected_batch_ids = np.random.choice(all_batch_ids, size=num_samples_to_select, replace=False)

        print(f"Evaluating grouping strategies across {num_samples_to_select} randomly selected samples (Seed: {RANDOM_SEED})...")

        # --- Evaluate for each selected sample ---
        for i, sample_id in enumerate(selected_batch_ids):
            print(f"\n[{i+1}/{num_samples_to_select}] Processing Sample ID: {sample_id}")
            k_cache_single_sample = all_k_caches[sample_id]

            # 2a. Evaluate Stride-based Grouping (for this sample)
            for group_size in [2, 4]:
                stride_groups = generate_stride_groups(num_layers, group_size)
                # UPDATED: Get memory savings and timing metrics
                avg_rank_ratio, _, avg_mem_saving, _, _, max_svd_time, total_svd_time = evaluate_grouping_strategy(
                    k_cache_single_sample, stride_groups, VARIANCE_TARGET
                )
                strategy_key = f'Stride-{group_size} (xKV)' # UPDATED strategy key
                all_samples_results[strategy_key]['rank_ratios'].append(avg_rank_ratio)
                all_samples_results[strategy_key]['memory_savings'].append(avg_mem_saving)
                all_samples_results[strategy_key]['max_svd_times'].append(max_svd_time)      # Store max time for sample
                all_samples_results[strategy_key]['total_svd_times'].append(total_svd_time)  # Store total time for sample
                print(f"  {strategy_key:<12} Rank Ratio: {avg_rank_ratio:.6f}, Mem Saving: {avg_mem_saving:.1%}, SVD Time: {max_svd_time:.4f}s (Max), {total_svd_time:.4f}s (Total)") # Print times

            # 2b. Evaluate Custom Grouping (for this sample)
            custom_groups_expanded = expand_range_groups(custom_groups_ranges)
            if custom_groups_expanded:
                 # UPDATED: Get memory savings and timing metrics
                avg_rank_ratio_custom, custom_group_ranks, avg_mem_saving_custom, custom_group_savings, custom_group_svd_times, max_svd_time_custom, total_svd_time_custom = evaluate_grouping_strategy(
                    k_cache_single_sample, custom_groups_expanded, VARIANCE_TARGET
                )
                # UPDATED key for custom results
                custom_key = 'Stride - 2 to 6 (Ours)'
                all_samples_results[custom_key]['rank_ratios'].append(avg_rank_ratio_custom)
                all_samples_results[custom_key]['per_group_ranks'].append(custom_group_ranks)
                all_samples_results[custom_key]['memory_savings'].append(avg_mem_saving_custom)
                all_samples_results[custom_key]['per_group_savings'].append(custom_group_savings)
                all_samples_results[custom_key]['max_svd_times'].append(max_svd_time_custom)         # Store max time for sample
                all_samples_results[custom_key]['total_svd_times'].append(total_svd_time_custom)     # Store total time for sample
                all_samples_results[custom_key]['per_group_svd_times'].append(custom_group_svd_times) # Store list of times per group for this sample
                print(f"  {custom_key:<12} Rank Ratio: {avg_rank_ratio_custom:.6f}, Mem Saving: {avg_mem_saving_custom:.1%}, SVD Time: {max_svd_time_custom:.4f}s (Max), {total_svd_time_custom:.4f}s (Total)") # Print times
            else:
                print("  Skipping custom grouping for this sample (invalid or empty groups).")
                # Append NaN or some indicator if needed, or just skip

        # --- Aggregate and Summarize Results ---
        print("\n--- Overall Summary (Averaged Across Samples) ---")
        # Adjusted width to accommodate longer names
        print(f"{'Strategy':<25} | {'Avg Rank Ratio':<20} | {'Avg Memory Saving':<20} | {'Avg Max SVD Time':<20} | {'Avg Total SVD Time':<20}")
        print("-" * 120) # Adjusted width

        plot_data = {'strategies': [], 'rank_ratios': [], 'memory_savings': [], 'max_svd_times': [], 'total_svd_times': []} # Data for plotting

        for name, metrics_dict in all_samples_results.items():
            avg_rank = np.mean(metrics_dict['rank_ratios']) if metrics_dict['rank_ratios'] else float('nan')
            avg_saving = np.mean(metrics_dict['memory_savings']) if metrics_dict.get('memory_savings') else float('nan')
            # Calculate average of max and total SVD times across samples
            avg_max_svd = np.mean(metrics_dict['max_svd_times']) if metrics_dict.get('max_svd_times') else float('nan')
            avg_total_svd = np.mean(metrics_dict['total_svd_times']) if metrics_dict.get('total_svd_times') else float('nan')

            # Store data for plotting
            plot_data['strategies'].append(name)
            plot_data['rank_ratios'].append(avg_rank)
            plot_data['memory_savings'].append(avg_saving * 100) # Store as percentage
            plot_data['max_svd_times'].append(avg_max_svd)
            plot_data['total_svd_times'].append(avg_total_svd)


            rank_ratio_str = f"{avg_rank:.6f}"
            saving_str = f"{avg_saving:.1%}"
            max_svd_str = f"{avg_max_svd:.4f}s"
            total_svd_str = f"{avg_total_svd:.4f}s"
            # Adjusted print formatting width
            print(f"{name:<25} | {rank_ratio_str:<20} | {saving_str:<20} | {max_svd_str:<20} | {total_svd_str:<20}")

        # Optional: Print detailed per-group custom results averaged across samples
        custom_key = 'Stride - 2 to 6 (Ours)' # Use the updated key
        # Check if per_group_svd_times exists and is not empty using the updated key
        if all_samples_results[custom_key]['per_group_ranks'] and all_samples_results[custom_key]['per_group_savings'] and all_samples_results[custom_key].get('per_group_svd_times'):
            print(f"\n--- {custom_key}: Per-Group Stats (Averaged Across Samples) ---") # Use updated key in title
            # Assuming the number of groups is consistent across samples
            num_custom_groups = len(custom_groups_expanded)
            avg_per_group_ranks = [[] for _ in range(num_custom_groups)]
            avg_per_group_savings = [[] for _ in range(num_custom_groups)]
            avg_per_group_svd_times = [[] for _ in range(num_custom_groups)] # NEW: Store avg SVD times per group

            # Collect ranks for each group position across all samples using updated key
            for sample_ranks in all_samples_results[custom_key]['per_group_ranks']:
                if len(sample_ranks) == num_custom_groups: # Basic check for consistency
                    for group_idx, rank in enumerate(sample_ranks):
                        avg_per_group_ranks[group_idx].append(rank)
                else:
                     print("Warning: Inconsistent number of per-group rank ratios found across samples. Skipping averaging for some groups.")

            # Collect savings for each group position across all samples using updated key
            for sample_savings in all_samples_results[custom_key]['per_group_savings']:
                if len(sample_savings) == num_custom_groups:
                    for group_idx, saving in enumerate(sample_savings):
                        avg_per_group_savings[group_idx].append(saving)
                else:
                    print("Warning: Inconsistent number of per-group memory savings found across samples. Skipping averaging for some groups.")

            # Collect SVD times for each group position across all samples using updated key
            for sample_svd_times in all_samples_results[custom_key]['per_group_svd_times']:
                if len(sample_svd_times) == num_custom_groups:
                    for group_idx, svd_time in enumerate(sample_svd_times):
                         avg_per_group_svd_times[group_idx].append(svd_time)
                else:
                    print("Warning: Inconsistent number of per-group SVD times found across samples. Skipping averaging for some groups.")

            # Calculate and print averages
            print(f"      {'Group':<20} | {'Avg Rank Ratio':<20} | {'Avg Memory Saving':<20} | {'Avg SVD Time':<20}")
            print("      " + "-" * 88) # Adjusted width
            for i, group in enumerate(custom_groups_expanded):
                group_str = f"Group {str(group)}"
                avg_rank_for_group = np.mean(avg_per_group_ranks[i]) if avg_per_group_ranks[i] else float('nan')
                avg_saving_for_group = np.mean(avg_per_group_savings[i]) if avg_per_group_savings[i] else float('nan')
                avg_svd_time_for_group = np.mean(avg_per_group_svd_times[i]) if avg_per_group_svd_times[i] else float('nan') # Calc avg svd_time

                rank_str = f"{avg_rank_for_group:.6f}"
                saving_str = f"{avg_saving_for_group:.1%}"
                svd_time_str = f"{avg_svd_time_for_group:.4f}s" # Format time
                print(f"      {group_str:<20} | {rank_str:<20} | {saving_str:<20} | {svd_time_str:<20}") # Print time

        # --- Plotting ---
        print("\nGenerating plots...")
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True) # Create plots directory if it doesn't exist

        strategies = plot_data['strategies']
        x_pos = np.arange(len(strategies))
        bar_colors = [PASTEL_COLORS['blue'], PASTEL_COLORS['orange'], PASTEL_COLORS['green']] # Define bar colors
        background_color = PLOT_STYLE.get('background_color', PASTEL_COLORS['pale_yellow']) # Get background color

        # Plot 1: Average Rank Ratio
        plt.figure(figsize=(8, 6), facecolor=background_color) # Set figure background
        ax = plt.gca() # Get current axes
        ax.set_facecolor(background_color) # Set axes background
        plt.bar(x_pos, plot_data['rank_ratios'], color=bar_colors)
        plt.xticks(x_pos, strategies, rotation=15, ha='right', fontsize=PLOT_STYLE['tick_fontsize'])
        plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'])
        plt.ylabel('Average Rank Ratio', fontsize=PLOT_STYLE['label_fontsize'])
        plt.title('Average Rank Ratio by Grouping Strategy', fontsize=PLOT_STYLE['title_fontsize'])
        plt.grid(axis='y', linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=0.7)
        ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
        ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'avg_rank_ratio.pdf'))
        plt.close() # Close the figure to free memory

        # Plot 2: Average Memory Saving
        plt.figure(figsize=(8, 6), facecolor=background_color) # Set figure background
        ax = plt.gca() # Get current axes
        ax.set_facecolor(background_color) # Set axes background
        plt.bar(x_pos, plot_data['memory_savings'], color=bar_colors)
        plt.xticks(x_pos, strategies, rotation=15, ha='right', fontsize=PLOT_STYLE['tick_fontsize'])
        plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'])
        plt.ylabel('Average Memory Saving (%)', fontsize=PLOT_STYLE['label_fontsize'])
        plt.title('Average Memory Saving by Grouping Strategy', fontsize=PLOT_STYLE['title_fontsize'])
        plt.ylim(0, max(plot_data['memory_savings']) * 1.1 if plot_data['memory_savings'] else 100) # Adjust y-limit
        plt.grid(axis='y', linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=0.7)
        ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
        ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'avg_memory_saving.pdf'))
        plt.close()

        # Plot 3: Average Max SVD Time
        plt.figure(figsize=(8, 6), facecolor=background_color) # Set figure background
        ax = plt.gca() # Get current axes
        ax.set_facecolor(background_color) # Set axes background
        plt.bar(x_pos, plot_data['max_svd_times'], color=bar_colors)
        plt.xticks(x_pos, strategies, rotation=15, ha='right', fontsize=PLOT_STYLE['tick_fontsize'])
        plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'])
        plt.ylabel('Average Max SVD Time (s)', fontsize=PLOT_STYLE['label_fontsize'])
        plt.title('Average Maximum SVD Computation Time by Grouping Strategy', fontsize=PLOT_STYLE['title_fontsize'])
        plt.grid(axis='y', linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=0.7)
        ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
        ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'avg_max_svd_time.pdf'))
        plt.close()

        # Plot 4: Average Total SVD Time
        plt.figure(figsize=(8, 6), facecolor=background_color) # Set figure background
        ax = plt.gca() # Get current axes
        ax.set_facecolor(background_color) # Set axes background
        plt.bar(x_pos, plot_data['total_svd_times'], color=bar_colors)
        plt.xticks(x_pos, strategies, rotation=15, ha='right', fontsize=PLOT_STYLE['tick_fontsize'])
        plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'])
        plt.ylabel('Average Total SVD Time (s)', fontsize=PLOT_STYLE['label_fontsize'])
        plt.title('Average Total SVD Computation Time by Grouping Strategy', fontsize=PLOT_STYLE['title_fontsize'])
        plt.grid(axis='y', linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=0.7)
        ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
        ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'avg_total_svd_time.pdf'))
        plt.close()

        print(f"Plots saved to '{plot_dir}' directory.")

    except (FileNotFoundError, TypeError, ValueError, Exception) as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        print(traceback.format_exc()) # Print full traceback for debugging