import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import Ridge
import argparse
import importlib.util
import os
import logging # Added logging
import json # Add json for saving/loading head selections
from cuml.preprocessing import Normalizer as cuMLNormalizer

# Import from preprocess_activations
from .preprocess_activations import load_data, process_data

# Import the model functions
from .models import (
    train_and_evaluate_model,
    train_and_evaluate_polynomial_cpu,
    train_and_evaluate_lasso,
    train_and_evaluate_ridge,
    train_and_evaluate_model_gpu,
    train_and_evaluate_linear_cosine_gpu,
    train_and_evaluate_polynomial_gpu,
    train_and_evaluate_torch_linear_gpu
)

# --- Configuration (Loaded from file via command-line arg) ---

# --- Data Loading and Initial Processing ---
# Functions load_data, process_prefill_data, process_decode_data, process_data removed
# They will be imported from preprocess_activations.py

# --- Regression Model Training and Evaluation ---
def flatten_matrix(matrix_entry):
    """Convert a matrix to a flattened numpy array."""
    try:
        arr = np.array(matrix_entry).flatten()
        return arr
    except Exception as e:
        return None

def prepare_merged_data(target_layer_id, target_head_id, expanded_df, ref_heads, matrix_type):
    """
    Optimized version: Prepare merged dataframe containing target head and cross-layer reference heads data.

    Args:
        target_layer_id: The layer identifier for the target head.
        target_head_id: The target head identifier.
        expanded_df: The full dataset.
        ref_heads: List of [layer_id, head_id] pairs for reference heads.
        matrix_type: Type of matrix to use (k_matrix or v_matrix).

    Returns:
        merged_df: DataFrame with aligned data, or None if data preparation fails.
    """
    # Collect all head combinations we need (target + references)
    all_heads_needed = [(target_layer_id, target_head_id)] + ref_heads
    
    # Create boolean masks for each (layer, head) combination and combine them efficiently
    combined_mask = None
    for layer_id, head_id in all_heads_needed:
        mask = (expanded_df['layer_id'] == layer_id) & (expanded_df['head_id'] == head_id)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask |= mask
    
    if combined_mask is None:
        print(f"No heads specified for target L{target_layer_id}H{target_head_id}")
        return None
        
    # Filter data for all needed heads at once - single pass through the data
    filtered_df = expanded_df[combined_mask].copy()
    
    if filtered_df.empty:
        print(f"Warning: No data found for any heads for target L{target_layer_id}H{target_head_id}")
        return None
    
    # Get target head data directly from filtered data
    target_mask = (filtered_df['layer_id'] == target_layer_id) & (filtered_df['head_id'] == target_head_id)
    target_df = filtered_df[target_mask][['sample_number', 'token_position', matrix_type]].copy()
    
    if target_df.empty:
        print(f"Warning: Missing target head L{target_layer_id}H{target_head_id}, skipping.")
        return None
    
    # Rename target column
    target_df = target_df.rename(columns={matrix_type: f'{matrix_type}_target'})
    
    # Start with target dataframe
    merged_df = target_df
    
    # Process reference heads one by one, but use the pre-filtered data
    missing_refs = []
    for ref_layer_id, ref_head_id in ref_heads:
        ref_mask = (filtered_df['layer_id'] == ref_layer_id) & (filtered_df['head_id'] == ref_head_id)
        ref_df = filtered_df[ref_mask][['sample_number', 'token_position', matrix_type]].copy()
        
        if ref_df.empty:
            missing_refs.append(f"L{ref_layer_id}H{ref_head_id}")
            continue
        
        # Prepare reference data for merging
        ref_col_name = f'{matrix_type}_ref_{ref_layer_id}_{ref_head_id}'
        ref_df = ref_df.rename(columns={matrix_type: ref_col_name})
        
        # Merge with existing data - this is still much faster than the original
        # because we're working with pre-filtered, smaller dataframes
        merged_df = pd.merge(
            merged_df,
            ref_df,
            on=['sample_number', 'token_position'],
            how='inner'
        )
        
        # Check if merge resulted in empty DataFrame
        if merged_df.empty:
            print(f"Error: Merging reference head L{ref_layer_id}H{ref_head_id} resulted in an empty DataFrame for target L{target_layer_id}H{target_head_id}.")
            return None
    
    if missing_refs:
        print(f"Warning: Missing reference heads {missing_refs} for target L{target_layer_id}H{target_head_id}. Continuing with available references.")
    
    if merged_df.empty:
        print(f"Error: No matching entries found for target L{target_layer_id}H{target_head_id} after merging.")
        return None
    else:
        print(f"Successfully merged {len(merged_df)} aligned entries for target L{target_layer_id}H{target_head_id}.")
        return merged_df

def process_data_for_regression(merged_df, ref_heads, matrix_type):
    """
    Process merged data for regression by flattening matrices. Handles cross-layer ref heads.

    Args:
        merged_df: DataFrame with merged reference and target data.
        ref_heads: List of [layer_id, head_id] pairs for reference heads.
        matrix_type: Type of matrix to use (k_matrix or v_matrix).

    Returns:
        merged_df_filtered: DataFrame with flattened matrices, or None if processing fails.
    """
    # Flatten target head matrix
    target_flat_col = f'{matrix_type}_target_flat'
    merged_df[target_flat_col] = merged_df[f'{matrix_type}_target'].apply(flatten_matrix)

    # Flatten reference heads matrices
    ref_flat_cols = []
    for ref_layer_id, ref_head_id in ref_heads:
        ref_col_name = f'{matrix_type}_ref_{ref_layer_id}_{ref_head_id}'
        ref_flat_col_name = f'{ref_col_name}_flat'
        # Check if the original ref column exists (it might have been skipped in merge if missing)
        if ref_col_name in merged_df.columns:
            merged_df[ref_flat_col_name] = merged_df[ref_col_name].apply(flatten_matrix)
            ref_flat_cols.append(ref_flat_col_name)
        else:
            print(f"Warning: Reference column {ref_col_name} not found in merged_df during flattening. Skipping.")

    # Filter out rows where flattening failed for target or any *available* reference
    columns_to_check = [target_flat_col] + ref_flat_cols
    merged_df_filtered = merged_df.dropna(subset=columns_to_check).copy()

    if merged_df_filtered.empty:
        print("Error: Could not prepare any valid data for regression after flattening matrices.")
        return None
    else:
        print(f"Prepared {len(merged_df_filtered)} valid rows for splitting.")
        return merged_df_filtered

def prepare_train_test_data(merged_df_filtered, config, ref_heads):
    """
    Prepare train and test datasets using cross-layer reference heads.
    Optionally filters out prefill tokens and initial decode tokens based on config.

    Args:
        merged_df_filtered: DataFrame with filtered and flattened data.
        config: Dictionary containing configuration parameters (test_size, random_state, include_prefill, exclude_N_decode).
        ref_heads: List of [layer_id, head_id] pairs used for features (X).

    Returns:
        X_train, X_test, y_train, y_test, train_sample_info, test_sample_info:
        Processed training and test data with sample info.
    """
    matrix_type = config.get('matrix_type', 'k_matrix')
    include_prefill = config.get('include_prefill', True) # Default to including prefill
    exclude_N_decode = config.get('exclude_N_decode', 0) # Default to 0

    # --- Filter based on include_prefill ---
    if not include_prefill:
        if 'prefill' in merged_df_filtered.columns:
            print("Filtering out prefill tokens...")
            original_count = len(merged_df_filtered)
            merged_df_filtered = merged_df_filtered[merged_df_filtered['prefill'] == False].copy()
            print(f"Removed {original_count - len(merged_df_filtered)} prefill samples. Remaining: {len(merged_df_filtered)}")
            if merged_df_filtered.empty:
                print("Error: No data remaining after filtering out prefill tokens.")
                return None, None, None, None, None, None
        else:
            print("Warning: 'include_prefill' is False, but 'prefill' column not found in data. Cannot filter.")

    # --- NEW: Filter based on exclude_N_decode ---
    if exclude_N_decode > 0:
        if 'prefill' in merged_df_filtered.columns and 'num_decode_tokens' in merged_df_filtered.columns:
            print(f"Filtering out the first {exclude_N_decode} decode tokens for each sample...")
            original_count = len(merged_df_filtered)
            # Keep rows that are prefill OR are decode tokens beyond the first N
            # Note: num_decode_tokens is 1-based index for decode tokens
            keep_condition = (merged_df_filtered['prefill'] == True) | (merged_df_filtered['num_decode_tokens'] > exclude_N_decode)
            merged_df_filtered = merged_df_filtered[keep_condition].copy()
            print(f"Removed {original_count - len(merged_df_filtered)} initial decode samples. Remaining: {len(merged_df_filtered)}")
            if merged_df_filtered.empty:
                print(f"Error: No data remaining after filtering out the first {exclude_N_decode} decode tokens.")
                return None, None, None, None, None, None
        else:
            print(f"Warning: 'exclude_N_decode' is {exclude_N_decode}, but 'prefill' or 'num_decode_tokens' column not found. Cannot filter initial decode tokens.")
    # --- End NEW ---

    try:
        train_df, test_df = train_test_split(
            merged_df_filtered,
            test_size=config['test_size'],
            random_state=config['random_state']
        )
    except ValueError:
        print("Warning: Not enough data to perform a train-test split. Using all data for training and evaluation.")
        train_df = merged_df_filtered.copy()
        test_df = merged_df_filtered.copy()

    print(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples.")

    # Prepare X and y arrays from the split DataFrames
    X_train_parts = []
    X_test_parts = []

    # Use the provided ref_heads list to build X
    for ref_layer_id, ref_head_id in ref_heads:
        ref_flat_col = f'{matrix_type}_ref_{ref_layer_id}_{ref_head_id}_flat'
        print(f"Debug: Processing reference head L{ref_layer_id}H{ref_head_id}, column: {ref_flat_col}")
        # Ensure the flattened column exists in both train_df and test_df
        if ref_flat_col in train_df.columns and ref_flat_col in test_df.columns:
            # Debug: Check flattened feature vector lengths
            train_lens = train_df[ref_flat_col].apply(lambda arr: arr.shape[0] if hasattr(arr, 'shape') else 0)
            test_lens = test_df[ref_flat_col].apply(lambda arr: arr.shape[0] if hasattr(arr, 'shape') else 0)
            print(f"Debug: {ref_flat_col} train flattened lengths min={train_lens.min()}, max={train_lens.max()}, count={len(train_lens)}")
            print(f"Debug: {ref_flat_col} test  flattened lengths min={test_lens.min()}, max={test_lens.max()}, count={len(test_lens)}")
            X_train_parts.append(np.vstack(train_df[ref_flat_col].values))
            X_test_parts.append(np.vstack(test_df[ref_flat_col].values))
        else:
            # This means the reference head was missing or failed flattening earlier.
            # It should have been handled, but log a warning just in case.
            print(f"Warning: Column {ref_flat_col} not found in train/test split. Skipping ref head L{ref_layer_id}H{ref_head_id}.")

    # Check if any parts were successfully added
    if not X_train_parts or not X_test_parts:
         print("Error: No valid reference head features could be prepared for X_train or X_test.")
         return None, None, None, None, None, None

    # Concatenate all reference heads' features horizontally
    X_train = np.hstack(X_train_parts)
    X_test = np.hstack(X_test_parts)

    # Target column name is simpler now
    target_flat_col = f'{matrix_type}_target_flat'
    y_train = np.vstack(train_df[target_flat_col].values)
    y_test = np.vstack(test_df[target_flat_col].values)

    # Preserve sample information
    train_sample_info = train_df[['sample_number', 'token_position']].copy()
    test_sample_info = test_df[['sample_number', 'token_position']].copy()

    # Add prefill indicator if it exists in the data
    if 'prefill' in train_df.columns:
        train_sample_info['prefill'] = train_df['prefill']
        test_sample_info['prefill'] = test_df['prefill']
    # Add num_decode_tokens if it exists, might be useful for debugging
    if 'num_decode_tokens' in train_df.columns:
        train_sample_info['num_decode_tokens'] = train_df['num_decode_tokens']
        test_sample_info['num_decode_tokens'] = test_df['num_decode_tokens']


    print(f"Train shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test shapes:  X_test {X_test.shape}, y_test {y_test.shape}")

    return X_train, X_test, y_train, y_test, train_sample_info, test_sample_info

def print_model_performance(layer_id, target_head_id, r2_test, mse_test, cosine_similarities_test, ref_heads, model_type, matrix_type='k_matrix'):
    """Print the performance metrics of a model."""
    print("\n--- Test Set Performance ---")
    print(f"Model Type: {model_type}")
    print(f"Matrix Type: {matrix_type}")
    print(f"Layer {layer_id}, Ref Heads {ref_heads} -> Target Head {target_head_id}")
    print(f"  R-squared: {r2_test:.4f}")
    print(f"  Mean Squared Error: {mse_test:.4f}")
    
    # Always print cosine similarity metrics
    if cosine_similarities_test is not None and cosine_similarities_test.size > 0:
        # Check for NaNs which can occur if norms are zero
        if np.isnan(cosine_similarities_test).all():
             print(f"  Cosine Similarity (Mean): NaN")
             print(f"  Cosine Similarity (Median): NaN")
             print(f"  Cosine Similarity (Min): NaN")
             print(f"  Cosine Similarity (Max): NaN")
        else:
             # Use nanmean, nanmedian, etc. to handle potential NaNs if not all are NaN
             print(f"  Cosine Similarity (Mean): {np.nanmean(cosine_similarities_test):.4f}")
             print(f"  Cosine Similarity (Median): {np.nanmedian(cosine_similarities_test):.4f}")
             # Filter out NaNs for min/max before calculation
             valid_similarities = cosine_similarities_test[~np.isnan(cosine_similarities_test)]
             if valid_similarities.size > 0:
                 print(f"  Cosine Similarity (Min): {np.min(valid_similarities):.4f}")
                 print(f"  Cosine Similarity (Max): {np.max(valid_similarities):.4f}")
             else: # Should not happen if not all were NaN, but handle defensively
                 print(f"  Cosine Similarity (Min): NaN")
                 print(f"  Cosine Similarity (Max): NaN")
    else:
        print(f"  Cosine Similarity (Mean): N/A (Evaluation Error)")
        print(f"  Cosine Similarity (Median): N/A")
        print(f"  Cosine Similarity (Min): N/A")
        print(f"  Cosine Similarity (Max): N/A")

def check_reference_heads(layer_id, expanded_df, ref_heads):
    """
    Check if all reference heads are available for a given layer.
    
    Args:
        layer_id: The layer identifier
        expanded_df: The full dataset
        
    Returns:
        bool: True if all reference heads are available, False otherwise
    """
    for head_id in ref_heads:
        head_data = expanded_df[(expanded_df['layer_id'] == layer_id) & 
                               (expanded_df['head_id'] == head_id)]
        if head_data.empty:
            return False
    return True

def search_optimal_alpha(X_train, y_train, config):
    """
    Perform grid search to find the optimal alpha parameter for Ridge regression.
    
    Args:
        X_train: Training features
        y_train: Training targets
        config: Configuration dictionary containing alpha search parameters
        
    Returns:
        best_alpha: The optimal alpha value found by grid search
    """
    print("\n--- Alpha Parameter Search for Ridge Regression ---")
    
    # Get alpha search parameters from config
    alpha_values = config.get('alpha_values', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    cv_folds = config.get('cv_folds', 5)
    
    print(f"Performing grid search with alpha values: {alpha_values}")
    print(f"Using {cv_folds}-fold cross-validation")
    
    # Create parameter grid
    param_grid = {'alpha': alpha_values}
    
    # Create Ridge model
    ridge = Ridge(fit_intercept=True)
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    
    # Perform grid search
    try:
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and score
        best_alpha = grid_search.best_params_['alpha']
        best_score = grid_search.best_score_
        
        print(f"Best alpha: {best_alpha}")
        print(f"Best score (negative MSE): {best_score:.4f}")
        
        # Print results for all alphas
        print("\nResults for all alpha values:")
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for alpha, mean, std in zip(alpha_values, means, stds):
            print(f"Alpha: {alpha:.6f}, Mean: {mean:.4f}, Std: {std:.4f}")
        
        return best_alpha
    except Exception as e:
        print(f"Error during alpha parameter search: {e}")
        print(f"Using default alpha value: {config.get('alpha', 1.0)}")
        return config.get('alpha', 1.0)

# Helper function to get the appropriate model function based on config
def _get_model_function(config):
    """Get the appropriate model training function based on the config."""
    model_type = config.get('model_type', 'linear') # Default to 'linear'
    use_gpu = model_type.endswith('_gpu')

    if model_type == 'linear':
        return train_and_evaluate_model
    elif model_type == 'linear_gpu':
        return train_and_evaluate_model_gpu
    elif model_type == 'linear_cosine_gpu':
        return train_and_evaluate_linear_cosine_gpu
    elif model_type == 'linear_torch_gpu': # Add case for torch gpu linear
        return train_and_evaluate_torch_linear_gpu
    elif model_type == 'polynomial_cpu':
        return train_and_evaluate_polynomial_cpu
    elif model_type == 'polynomial_gpu':
        # Make sure the polynomial GPU function exists and is imported
        try:
            from .models import train_and_evaluate_polynomial_gpu
            return train_and_evaluate_polynomial_gpu
        except ImportError:
            raise ValueError(f"Model type '{model_type}' selected, but train_and_evaluate_polynomial_gpu function not found in models.py")
    elif model_type == 'lasso':
        return lambda X_train, X_test, y_train, y_test, test_sample_info, cfg: train_and_evaluate_lasso(X_train, X_test, y_train, y_test, test_sample_info, cfg, use_gpu=False)
    elif model_type == 'lasso_gpu':
        return lambda X_train, X_test, y_train, y_test, test_sample_info, cfg: train_and_evaluate_lasso(X_train, X_test, y_train, y_test, test_sample_info, cfg, use_gpu=True)
    elif model_type == 'ridge':
        return lambda X_train, X_test, y_train, y_test, test_sample_info, cfg: train_and_evaluate_ridge(X_train, X_test, y_train, y_test, test_sample_info, cfg, use_gpu=False)
    elif model_type == 'ridge_gpu':
        return lambda X_train, X_test, y_train, y_test, test_sample_info, cfg: train_and_evaluate_ridge(X_train, X_test, y_train, y_test, test_sample_info, cfg, use_gpu=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_single_ref_head(layer_id, target_head_id, single_ref_head_id, expanded_df, config):
    """
    Evaluate the performance using a single reference head to predict the target head.

    Args:
        layer_id: Layer identifier.
        target_head_id: Target head identifier.
        single_ref_head_id: The single reference head to use.
        expanded_df: The full dataset.
        config: Configuration dictionary.

    Returns:
        Performance metric value (e.g., cos_sim_mean or mse) or None if evaluation fails.
    """
    metric_key = config.get('ref_head_metric', 'cos_sim_mean') # e.g., 'cos_sim_mean' or 'mse'
    matrix_type = config.get('matrix_type', 'k_matrix')
    model_type = config.get('model_type', 'linear_gpu') # Assume default if not specified

    print(f"  Evaluating ref_head {single_ref_head_id} for target_head {target_head_id}...")

    # 1. Prepare data using only the single reference head
    temp_ref_heads = [single_ref_head_id]
    merged_df = prepare_merged_data(layer_id, target_head_id, expanded_df, temp_ref_heads, matrix_type)
    if merged_df is None:
        print(f"    Failed to merge data for single ref head {single_ref_head_id}.")
        return None

    merged_df_filtered = process_data_for_regression(merged_df, temp_ref_heads, matrix_type)
    if merged_df_filtered is None:
        print(f"    Failed to process data for single ref head {single_ref_head_id}.")
        return None

    # Create a temporary config with the single reference head for data preparation
    temp_config = config.copy()
    temp_config['ref_heads'] = temp_ref_heads
    X_train, X_test, y_train, y_test, train_sample_info, test_sample_info = prepare_train_test_data(merged_df_filtered, temp_config, temp_ref_heads)

    # Check if data preparation failed
    if X_train is None:
        print(f"    Failed to prepare train/test data for single ref head {single_ref_head_id}.")
        return None

    # 2. Train and evaluate model
    model_func = _get_model_function(config) # Use helper to get model func
    if model_func is None:
        print(f"    Could not get model function for {model_type}.")
        return None

    # Update kwargs for the model function
    model_kwargs = {'config': config, 'test_sample_info': test_sample_info}

    try:
        # Update the call to expect the model package as the 5th return value
        r2, mse, cos_sim, _, _model_package = model_func(X_train, X_test, y_train, y_test, **model_kwargs)

        # 3. Return the desired metric
        if metric_key == 'cos_sim_mean':
            metric_value = np.nanmean(cos_sim) if cos_sim is not None and cos_sim.size > 0 else -np.inf # Handle NaN/empty
        elif metric_key == 'mse':
            metric_value = mse
        elif metric_key == 'r2':
             metric_value = r2
        else:
            print(f"    Warning: Unknown ref_head_metric '{metric_key}'. Defaulting to cos_sim_mean.")
            metric_value = np.nanmean(cos_sim) if cos_sim is not None and cos_sim.size > 0 else -np.inf

        # Check for NaN explicitly after calculation
        if np.isnan(metric_value):
            print(f"    Metric '{metric_key}' resulted in NaN for ref_head {single_ref_head_id}. Treating as worst score.")
            metric_value = -np.inf if metric_key != 'mse' else np.inf

        # Ensure the metric value is a standard Python float before returning
        # This handles potential CuPy scalars/arrays returned by GPU models
        try:
            # Check if it has a .get() method (like cupy arrays)
            if hasattr(metric_value, 'get'):
                # Extract scalar value if it's an array
                if hasattr(metric_value, 'item'):
                    final_metric_value = float(metric_value.item())
                else:
                    final_metric_value = float(metric_value.get())
            else:
                final_metric_value = float(metric_value)
        except Exception as convert_e:
            print(f"    Warning: Could not convert metric_value ({type(metric_value)}) to float: {convert_e}. Returning None.")
            return None

        print(f"    Ref head {single_ref_head_id} -> Target head {target_head_id}: {metric_key} = {final_metric_value:.4f}")
        return final_metric_value

    except Exception as e:
        print(f"    Error during model training/evaluation for single ref head {single_ref_head_id}: {e}")
        # Fit its own normalizer temporarily if needed for evaluation
        # NOTE: This part requires careful thought. If the main model uses normalization,
        # this single-head evaluation should probably also normalize its input
        # using a temporary normalizer fitted only on its single-head X_train.
        # For now, we assume the model function handles normalization internally if needed,
        # or that the metric calculation is robust enough without explicit normalization here.
        # If issues arise, add temporary normalization fitting/application here.
        return None

def load_head_selections_from_file(config):
    """
    Loads head selections from the JSON file specified in the config.
    Parses keys like 'layer_target' into (layer_id, target_head_id) tuples.
    """
    filepath = args.head_selection_file # Default for safety
    if not os.path.exists(filepath):
        print(f"Warning: Head selection file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            selections = json.load(f)

        # Parse keys like "layer_target" into (layer_id, target_head_id) tuples
        parsed_selections = {}
        for key, value in selections.items():
            try:
                layer_id_str, target_head_id_str = key.split('_')
                layer_id = int(layer_id_str)
                target_head_id = int(target_head_id_str)
                # Validate the value structure (optional but good practice)
                if isinstance(value, dict) and 'ref_heads' in value and 'target_heads' in value:\
                    # Ensure target_heads in the file matches the key
                    if value['target_heads'] == [[layer_id, target_head_id]]:
                        parsed_selections[(layer_id, target_head_id)] = value
                    else:
                        print(f"Warning: Mismatch between key '{key}' and target_heads {value.get('target_heads')} in {filepath}. Skipping entry.")
                else:
                    print(f"Warning: Invalid value structure for key '{key}' in {filepath}. Skipping entry.")
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse key '{key}' in {filepath}: {e}. Skipping entry.")

        if not parsed_selections:
             print(f"Warning: No valid head selections could be parsed from {filepath}.")
             return None

        print(f"Successfully loaded and parsed {len(parsed_selections)} head selections from {filepath}")
        return parsed_selections
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error loading or parsing head selection file {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading head selection file {filepath}: {e}")
        return None

# Renamed from find_best_ref_heads to find_best_layer_ref_heads
def find_best_layer_ref_heads(layer_id, num_ref_heads_to_select, all_head_ids_in_layer, expanded_df, config):
    """
    NOTE: This function is incompatible with loading specific cross-layer references
    from a file like ref_target_heads.json. It should not be called if
    config['load_head_selection'] is True and the loaded format is per-target.

    Find the best set of reference heads for the entire layer by evaluating
    each head's average performance as a reference predicting all other heads.

    Args:
        layer_id: Layer identifier.
        num_ref_heads_to_select: How many reference heads to select for the layer.
        all_head_ids_in_layer: List of all available head IDs in the current layer.
        expanded_df: The full dataset.
        config: Configuration dictionary.

    Returns:
        List of the best reference head IDs for the layer, or None if search fails.
    """
    # Add a check at the beginning
    metric_key = config.get('ref_head_metric', 'cos_sim_mean')
    higher_is_better = metric_key != 'mse'

    print(f"\nFinding best {num_ref_heads_to_select} layer-wide reference heads for layer {layer_id} using metric '{metric_key}'...")

    if len(all_head_ids_in_layer) <= num_ref_heads_to_select:
         print(f"  Warning: Number of heads to select ({num_ref_heads_to_select}) is >= total heads ({len(all_head_ids_in_layer)}). Selecting all heads as reference.")
         # This case might not be desired, adjust if needed (e.g., return None or fewer heads)
         return all_head_ids_in_layer

    head_avg_performance = []

    # Evaluate each head's potential as a reference
    for potential_ref_head_id in all_head_ids_in_layer:
        print(f"Evaluating head {potential_ref_head_id} as a potential reference head:")
        performance_metrics = []
        # Test against all other heads as targets
        possible_target_ids = [h for h in all_head_ids_in_layer if h != potential_ref_head_id]

        if not possible_target_ids:
             print(f"   Skipping head {potential_ref_head_id} as reference: No other heads to target.")
             continue # Should only happen if only 1 head in layer

        for target_head_id in possible_target_ids:
            metric_value = evaluate_single_ref_head(
                layer_id, target_head_id, potential_ref_head_id, # Use potential_ref as the single ref
                expanded_df, config
            )
            if metric_value is not None:
                performance_metrics.append(metric_value)
            else:
                 # Assign worst score if evaluation fails for this pair
                 worst_score = -np.inf if higher_is_better else np.inf
                 performance_metrics.append(worst_score)

        # Calculate average performance for this head as a reference
        if performance_metrics:
            avg_metric = np.nanmean(performance_metrics)
            # Handle case where all evaluations might have failed or resulted in NaN
            if np.isnan(avg_metric):
                 print(f"  Warning: Average metric for head {potential_ref_head_id} as reference resulted in NaN. Assigning worst score.")
                 avg_metric = -np.inf if higher_is_better else np.inf
            print(f"  Average performance for head {potential_ref_head_id} as reference: {avg_metric:.4f}")
            head_avg_performance.append({'head_id': potential_ref_head_id, 'avg_metric': avg_metric})
        else:
            print(f"  Could not evaluate head {potential_ref_head_id} against any targets.")
            # Assign worst score if no evaluations succeeded
            worst_score = -np.inf if higher_is_better else np.inf
            head_avg_performance.append({'head_id': potential_ref_head_id, 'avg_metric': worst_score})

    if not head_avg_performance:
        print("  Failed to evaluate average performance for any head as a reference.")
        return None

    # Sort heads by their average performance as a reference
    head_avg_performance.sort(key=lambda x: x['avg_metric'], reverse=higher_is_better)

    print("\n  Layer-wide Reference Head Performance Ranking (based on avg metric when predicting others):")
    for item in head_avg_performance:
         print(f"    Head {item['head_id']}: {item['avg_metric']:.4f}")

    # Select top N heads as the best layer-wide reference heads
    best_layer_ref_heads = [item['head_id'] for item in head_avg_performance[:num_ref_heads_to_select]]

    print(f"\n  Selected best {num_ref_heads_to_select} layer-wide reference heads: {best_layer_ref_heads}")
    return best_layer_ref_heads

def find_best_ref_heads(layer_id, target_head_id, num_ref_heads_to_select, all_head_ids_in_layer, expanded_df, config):
    """
    DEPRECATED: Use find_best_layer_ref_heads for the new logic.
    Also incompatible with loading specific cross-layer references.
    """
    raise DeprecationWarning("find_best_ref_heads is deprecated and incompatible with loading specific cross-layer refs.")
    # return find_best_layer_ref_heads(layer_id, num_ref_heads_to_select, all_head_ids_in_layer, expanded_df, config)

def analyze_layers(expanded_df, config):
    """
    Process all specified target heads using their defined cross-layer references.

    Args:
        expanded_df: The full dataset.
        config: Dictionary containing configuration parameters.

    Returns:
        results_df: DataFrame with results for all processed models.
        low_sim_df: DataFrame with all low similarity samples.
        low_sim_summary: Summary analysis of low similarity samples.
        all_head_selections: Dict mapping (target_layer, target_head) -> head selection used.
        all_models: Nested Dict mapping (target_layer, target_head) -> model_package.
    """
    all_results = []
    all_low_sim_samples = {} # Maps (target_layer, target_head) -> low_sim_df
    all_head_selections = {} # Maps (target_layer, target_head) -> loaded selection
    all_models = {}          # Maps (target_layer, target_head) -> model_package
    model_type = config.get('model_type', 'linear')
    matrix_type = config.get('matrix_type', 'k_matrix')
    
    # Different thresholds for different matrix types
    low_sim_threshold = config.get('low_similarity_threshold', 0.85)
    # For v_matrix, use high MSE as the threshold for problematic predictions
    high_mse_threshold = config.get('high_mse_threshold', float('inf'))  # Default to infinity if not specified

    # --- Load Head Selections ---
    # This mode REQUIRES loading specific head selections.
    if not config.get('load_head_selection', False):
        raise ValueError("Configuration error: 'load_head_selection' must be True to use the cross-layer reference workflow.")
    if config.get('dynamic_ref_heads', False):
        print("Warning: 'dynamic_ref_heads' is True but loading specific heads. Dynamic search will be skipped.")
        # Disable dynamic search explicitly if it was set
        config['dynamic_ref_heads'] = False

    target_head_definitions_all = load_head_selections_from_file(config)

    if not target_head_definitions_all:
        raise ValueError(f"Failed to load or parse head definitions from {args.head_selection_file}. Cannot proceed.")

    # --- Filter Targets based on Layer Range ---
    layer_start = config.get('layer_start', 0) # Default start to 0
    layer_end = config.get('layer_end', float('inf')) # Default end to infinity if not specified
    print(f"Filtering loaded target heads for layers between {layer_start} and {layer_end} (inclusive).")

    target_head_definitions = {
        (l_id, h_id): data
        for (l_id, h_id), data in target_head_definitions_all.items()
        if layer_start <= l_id <= layer_end
    }

    if not target_head_definitions:
        print(f"Warning: No target heads found within the specified layer range [{layer_start}, {layer_end}] after filtering.")
        # Return empty results consistent with no processing happening
        return pd.DataFrame(all_results), None, None, all_head_selections, all_models

    print(f"Processing {len(target_head_definitions)} target heads within the specified layer range (out of {len(target_head_definitions_all)} total loaded)...")

    # --- Iterate through each defined Target Head ---
    for (target_layer_id, target_head_id), selection_data in target_head_definitions.items():
        print(f"\n{'-'*50}")
        print(f"Processing Target L{target_layer_id}H{target_head_id}")
        print(f"{'-'*50}")

        current_ref_heads = selection_data['ref_heads'] # This is now [[L, H], [L, H], ...]
        # The target head is implicitly defined by the key, but we can check consistency
        # expected_target = [[target_layer_id, target_head_id]]
        # if selection_data['target_heads'] != expected_target:
        #     print(f"Warning: Mismatch target in data {selection_data['target_heads']} for key ({target_layer_id},{target_head_id}). Using key.")

        print(f"Using Specific Ref Heads: {current_ref_heads}")

        # --- Start of logic adapted from process_layer ---

        # 1. Prepare and align data using the specific cross-layer reference heads
        merged_df = prepare_merged_data(target_layer_id, target_head_id, expanded_df, current_ref_heads, matrix_type)
        if merged_df is None:
            print(f"Skipping Target L{target_layer_id}H{target_head_id} due to data merging failure.")
            continue

        # 2. Process data for regression
        merged_df_filtered = process_data_for_regression(merged_df, current_ref_heads, matrix_type)
        if merged_df_filtered is None:
            print(f"Skipping Target L{target_layer_id}H{target_head_id} due to data processing failure.")
            continue

        # 3. Prepare train and test data - Pass the specific ref_heads
        X_train, X_test, y_train, y_test, train_sample_info, test_sample_info = prepare_train_test_data(
            merged_df_filtered, config, current_ref_heads
        )
        if X_train is None:
            print(f"Skipping Target L{target_layer_id}H{target_head_id} due to train/test split failure.")
            continue

        # 4. Train and evaluate the final model
        print(f"Training final model ({model_type}) for target L{target_layer_id}H{target_head_id}")
        model_func = _get_model_function(config) # Use helper
        if model_func is None:
            print(f"Cannot proceed for target L{target_layer_id}H{target_head_id} due to model function error. Skipping.")
            continue

        model_kwargs = {'config': config, 'test_sample_info': test_sample_info}

        r2_test, mse_test, cosine_similarities_test, low_sim_info = -1, float('inf'), np.array([]), None # Defaults
        model_package = None
        try:
            r2_test, mse_test, cosine_similarities_test, low_sim_info, model_package = model_func(
                X_train, X_test, y_train, y_test, **model_kwargs
            )
        except Exception as e:
             print(f"Error during model training/evaluation for L{target_layer_id}H{target_head_id}: {e}")
             # Keep default bad values

        # 5. Print performance metrics
        print_model_performance(target_layer_id, target_head_id, r2_test, mse_test, cosine_similarities_test, current_ref_heads, model_type, matrix_type)

        # 6. Store results
        cos_sim_mean = np.nanmean(cosine_similarities_test) if cosine_similarities_test is not None and cosine_similarities_test.size > 0 else np.nan
        cos_sim_median = np.nanmedian(cosine_similarities_test) if cosine_similarities_test is not None and cosine_similarities_test.size > 0 else np.nan

        model_results = {
            'model_type': model_type,
            'matrix_type': matrix_type,
            'target_layer_id': target_layer_id,
            'target_head_id': target_head_id,
            'ref_heads': current_ref_heads, # Store the specific ref heads used
            'r2': r2_test,
            'mse': mse_test,
            'cos_sim_mean': cos_sim_mean,
            'cos_sim_median': cos_sim_median,
            'low_sim_count': 0 if low_sim_info is None else len(low_sim_info)
        }
        # Add alpha value if using ridge, etc. (Keep existing logic)
        if model_type in ['ridge', 'ridge_gpu']:
            model_results['alpha'] = config.get('alpha', 1.0)
        if model_type == 'linear_cosine_gpu':
            model_results['learning_rate'] = config.get('learning_rate')
            model_results['epochs'] = config.get('epochs')
            model_results['regularization'] = config.get('regularization')
            model_results['batch_size'] = config.get('batch_size')

        all_results.append(model_results)

        # Store head selection used for this target
        target_key = (target_layer_id, target_head_id)
        all_head_selections[target_key] = selection_data # Store the loaded data

        # 7. Store model weights if enabled
        if config.get('save_model_weights', False):
            if model_package is not None:
                all_models[target_key] = model_package
                print(f"Stored model package for L{target_layer_id}H{target_head_id} in memory.")
            else:
                all_models[target_key] = None
                print(f"Warning: Model saving enabled, but no model package for L{target_layer_id}H{target_head_id}.")

        # 8. Store low similarity samples for both k_matrix and v_matrix
        if low_sim_info is not None and not low_sim_info.empty:
            all_low_sim_samples[target_key] = low_sim_info
            if matrix_type == 'k_matrix':
                print(f"\n--- Low Similarity Samples Analysis (Similarity < {low_sim_threshold}) for L{target_layer_id}H{target_head_id} --- ")
            else:  # v_matrix
                print(f"\n--- High Error Samples Analysis for L{target_layer_id}H{target_head_id} --- ")
            print(f"Number of problematic samples: {len(low_sim_info)}")
        else:
            if matrix_type == 'k_matrix':
                print(f"\nNo samples with similarity < {low_sim_threshold} found for L{target_layer_id}H{target_head_id}.")
            else:  # v_matrix
                print(f"\nNo samples with high MSE found for L{target_layer_id}H{target_head_id}.")

        # --- End of logic adapted from process_layer ---

    # --- Post-processing (largely unchanged) ---

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)

    # Analyze low similarity samples (function needs adjustment for new dict keys)
    low_sim_df, low_sim_summary = analyze_low_similarity_samples(all_low_sim_samples, config)

    # Return selections and models with the new key format
    return results_df, low_sim_df, low_sim_summary, all_head_selections, all_models

def analyze_low_similarity_samples(all_low_sim_samples, config):
    """
    Analyze low similarity samples across all processed target heads.
    Handles dictionary keys as (target_layer_id, target_head_id).

    Args:
        all_low_sim_samples: Dict mapping (target_layer, target_head) -> low_sim_df
        config: Dictionary containing configuration parameters

    Returns:
        low_sim_df, low_sim_summary: DataFrames with analysis results
    """
    if not all_low_sim_samples:
        print("No low similarity samples to analyze.")
        return None, None

    low_sim_threshold = config['low_similarity_threshold']

    # Flatten the nested dictionary of low similarity samples
    all_samples = []
    for (layer_id, target_head_id), samples_df in all_low_sim_samples.items():
        if samples_df is not None and not samples_df.empty:
            samples_df = samples_df.copy()
            samples_df['target_layer_id'] = layer_id      # Use new column name
            samples_df['target_head_id'] = target_head_id
            all_samples.append(samples_df)
        else:
            print(f"Warning: Empty or None low_sim_df found for ({layer_id}, {target_head_id}). Skipping.")

    if not all_samples:
        print("No valid low similarity sample DataFrames found to analyze.")
        return None, None

    # Combine all samples into a single DataFrame
    low_sim_df = pd.concat(all_samples, ignore_index=True)

    print("\n\n")
    print("="*80)
    print(f"LOW SIMILARITY SAMPLES ANALYSIS (Similarity < {low_sim_threshold})")
    print("="*80)

    # 1. Overall statistics
    print(f"Total number of low similarity predictions: {len(low_sim_df)}")
    if 'cosine_similarity' in low_sim_df.columns and not low_sim_df['cosine_similarity'].isnull().all():
        print(f"Average cosine similarity: {low_sim_df['cosine_similarity'].mean():.4f}")
    else:
        print("Cosine similarity data missing or all NaN.")

    # 2. Distribution across target layers and heads
    # Use new column names for grouping
    layer_head_counts = low_sim_df.groupby(['target_layer_id', 'target_head_id']).size().unstack(fill_value=0)
    print("\nDistribution of low similarity samples across target layers and heads:")
    print(layer_head_counts)

    # 3. Sample number analysis
    sample_counts = low_sim_df['sample_number'].value_counts()
    print("\nTop 10 sample numbers with the most low similarity predictions:")
    print(sample_counts.head(10))

    # 4. Prefill analysis if available
    if 'is_prefill' in low_sim_df.columns:
        # Group by target_layer_id and is_prefill
        prefill_counts = low_sim_df.groupby(['target_layer_id', 'is_prefill']).size().unstack(fill_value=0)
        prefill_cols = prefill_counts.columns.tolist()
        prefill_col_names = ['Non-prefill', 'Prefill'] if len(prefill_cols) == 2 else (['Prefill'] if 1 in prefill_cols else ['Non-prefill'] if 0 in prefill_cols else [])
        prefill_counts.columns = prefill_col_names
        print("\nPrefill vs Non-prefill distribution by target layer:")
        print(prefill_counts)

    # 5. Token position analysis
    position_analysis = low_sim_df.groupby('token_position').agg({
        'cosine_similarity': ['count', 'mean', 'min']
    })
    position_analysis.columns = ['count', 'avg_similarity', 'min_similarity']
    position_analysis = position_analysis.sort_values('count', ascending=False)
    print("\nToken position analysis (top 10 by count):")
    print(position_analysis.head(10))

    # 6. Create summary DataFrame for returning
    low_sim_summary = low_sim_df.groupby(['target_layer_id', 'target_head_id']).agg({
        'sample_number': ['nunique', lambda x: ', '.join(map(str, x.unique()[:5]))],
        'cosine_similarity': ['count', 'mean', 'min', 'max']
    })

    low_sim_summary.columns = [
        'unique_samples', 'top5_samples', 'count', 'avg_sim', 'min_sim', 'max_sim'
    ]

    low_sim_summary = low_sim_summary.reset_index()

    return low_sim_df, low_sim_summary

def summarize_results(results_df):
    """
    Generate summary tables and visualizations of the results.
    
    Args:
        results_df: DataFrame with results for all processed models
    """
    if results_df.empty:
        print("No results to summarize.")
        return
    
    print("\n\n")
    print("="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    print(results_df)
    
    # Get matrix_type if it exists
    matrix_type = results_df['matrix_type'].iloc[0] if 'matrix_type' in results_df.columns else 'k_matrix'
    
    # Calculate per-layer average metrics, grouped by model type
    metrics_to_agg = ['r2', 'mse', 'cos_sim_mean', 'cos_sim_median']
    
    agg_dict = {metric: 'mean' for metric in metrics_to_agg}
    
    per_layer_summary = results_df.groupby(['model_type', 'target_layer_id']).agg(agg_dict).reset_index()
    
    # Rename columns for better readability
    column_map = {
        'r2': 'avg_r2',
        'mse': 'avg_mse',
        'cos_sim_mean': 'avg_cos_sim',
        'cos_sim_median': 'avg_cos_sim_median'
    }
    
    per_layer_summary.columns = [column_map.get(col, col) for col in per_layer_summary.columns]
    per_layer_summary = per_layer_summary.sort_values(['model_type', 'target_layer_id'])
    
    print("\n\n")
    print("="*80)
    print("SUMMARY BY LAYER (AVERAGE ACROSS ALL HEADS)")
    print("="*80)
    print(per_layer_summary)
    
    # Print top models by performance metrics for each model type
    # Use cosine similarity as the primary metric for both k_matrix and v_matrix
    print("\n--- Top Performing (Layer, Target Head) combinations ---")
    sort_metric = 'cos_sim_mean'  # Use cosine similarity for both k_matrix and v_matrix
    ascending = False  # Higher cosine similarity is better

    # Ensure the sort metric column exists before sorting
    if (sort_metric in results_df.columns):
        top_models_df = results_df.sort_values(sort_metric, ascending=ascending)
        print(f"\nTop 10 combinations by {sort_metric}:")
        # Make sure ref_heads column is included
        print(top_models_df[['target_layer_id', 'target_head_id', 'ref_heads', sort_metric, 'r2', 'mse']].head(10))
    else:
        print(f"Warning: Sort metric '{sort_metric}' not found in results columns.")
        # Make sure ref_heads column is included
        print(results_df[['target_layer_id', 'target_head_id', 'ref_heads', 'r2', 'mse']].head(10)) # Show some info anyway

    return per_layer_summary # Return the per-layer average summary for potential plotting

def plot_layer_cosine_similarities(per_layer_summary):
    """
    Plot the average cosine similarity for each layer.
    
    Args:
        per_layer_summary: DataFrame with per-layer summary metrics
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='target_layer_id', y='avg_cos_sim', data=per_layer_summary)
    plt.title('Average Cosine Similarity by Layer')
    plt.xlabel('Layer ID')
    plt.ylabel('Average Cosine Similarity')
    plt.tight_layout()
    plt.show()

def plot_layer_mse(per_layer_summary):
    """
    Plot the average MSE for each layer.
    
    Args:
        per_layer_summary: DataFrame with per-layer summary metrics
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='target_layer_id', y='avg_mse', data=per_layer_summary)
    plt.title('Average Mean Squared Error by Layer')
    plt.xlabel('Layer ID')
    plt.ylabel('Average MSE (lower is better)')
    plt.tight_layout()
    plt.show()

def plot_low_similarity_analysis(low_sim_df, config):
    """
    Plot visualizations for low similarity samples analysis.
    
    Args:
        low_sim_df: DataFrame with low similarity samples data
        config: Dictionary containing configuration parameters
    """
    low_sim_threshold = config['low_similarity_threshold']
    
    # 1. Histogram of cosine similarities
    plt.figure(figsize=(12, 6))
    sns.histplot(low_sim_df['cosine_similarity'], bins=20, kde=True)
    plt.title(f'Distribution of Cosine Similarities < {low_sim_threshold}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # 2. Heatmap of low similarity counts by layer and target head
    low_sim_counts = low_sim_df.pivot_table(
        index='target_layer_id', 
        columns='target_head_id', 
        values='cosine_similarity',
        aggfunc='count',
        fill_value=0
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(low_sim_counts, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Low Similarity Sample Counts by Layer and Target Head')
    plt.xlabel('Target Head ID')
    plt.ylabel('Layer ID')
    plt.tight_layout()
    plt.show()
    
    # 3. If prefill data is available, plot prefill vs non-prefill
    if 'is_prefill' in low_sim_df.columns:
        plt.figure(figsize=(12, 6))
        prefill_counts = low_sim_df.groupby(['target_layer_id', 'is_prefill']).size().unstack(fill_value=0)
        prefill_cols = prefill_counts.columns.tolist()
        prefill_col_names = ['Non-prefill', 'Prefill'] if len(prefill_cols) == 2 else ['Prefill']
        prefill_counts.columns = prefill_col_names
        prefill_counts.plot(kind='bar', stacked=True)
        plt.title('Prefill vs Non-prefill Low Similarity Samples by Layer')
        plt.xlabel('Layer ID')
        plt.ylabel('Count')
        plt.legend(title='Sample Type')
        plt.tight_layout()
        plt.show()
    
    # 4. Top samples with low similarity
    plt.figure(figsize=(12, 6))
    sample_counts = low_sim_df['sample_number'].value_counts().sort_values(ascending=False).head(10)
    sns.barplot(x=sample_counts.index, y=sample_counts.values)
    plt.title('Top 10 Sample Numbers with Low Similarity Predictions')
    plt.xlabel('Sample Number')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_analysis(expanded_df, config):
    """
    Main function to execute the entire analysis pipeline.
    
    Args:
        expanded_df: The preprocessed data DataFrame
        config: Dictionary containing configuration parameters
    """
    # Analyze all layers
    results_df, low_sim_df, low_sim_summary, all_head_selections, all_models = analyze_layers(expanded_df, config)

    if results_df.empty:
        print("Analysis did not produce any results. Exiting.")
        return None, None, None, None, None

    # Get matrix type
    matrix_type = config.get('matrix_type', 'k_matrix')

    # Generate summary and visualizations
    per_layer_summary = summarize_results(results_df)

    # Plot metrics based on matrix type
    if per_layer_summary is not None:
        # Always plot cosine similarity
        plot_layer_cosine_similarities(per_layer_summary)
        # Always plot MSE as it's relevant for both k_matrix and v_matrix
        plot_layer_mse(per_layer_summary)

    # Process low similarity samples for both k_matrix and v_matrix
    print("\n\n")
    print("="*80)
    if matrix_type == 'k_matrix':
        print("LOW SIMILARITY SAMPLES SUMMARY")
    else:  # v_matrix
        print("HIGH ERROR SAMPLES SUMMARY")
    print("="*80)
    
    if low_sim_summary is not None:
        print(low_sim_summary)
    else:
        if matrix_type == 'k_matrix':
            print("No low similarity summary available.")
        else:  # v_matrix
            print("No high error summary available.")

    # Plot additional visualizations for low similarity samples
    if low_sim_df is not None and not low_sim_df.empty:
        plot_low_similarity_analysis(low_sim_df, config)
    else:
        if matrix_type == 'k_matrix':
            print("No low similarity samples found for plotting.")
        else:  # v_matrix
            print("No high error samples found for plotting.")

    # NEW: Save all collected models to a single file if enabled
    if config.get('save_model_weights', False) and all_models:
        if args.matrix_type == 'k_matrix':
            save_filepath = f'model_pickles/all_model_weights_{args.alias}_keys.pkl'
        elif args.matrix_type == 'v_matrix':
            save_filepath = f'model_pickles/all_model_weights_{args.alias}_values.pkl'
        else:
            raise ValueError(f"Invalid matrix type: {args.matrix_type}")
        try:
            with open(save_filepath, 'wb') as f:
                pickle.dump(all_models, f)
            print(f"Saved all model packages (model + normalizers) to {save_filepath}")
        except Exception as e:
            print(f"Error saving all model packages to {save_filepath}: {e}")
    elif config.get('save_model_weights', False) and not all_models: # Adjusted condition
        print("Warning: Model saving enabled, but no model packages were collected.")

    return results_df, per_layer_summary, low_sim_df, low_sim_summary, all_head_selections, all_models

def load_config(config_path):
    """Load the CONFIG dictionary from a Python file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, 'CONFIG') or not isinstance(config_module.CONFIG, dict):
        raise ValueError(f"Config file {config_path} must contain a dictionary named CONFIG.")

    return config_module.CONFIG

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate key prediction models.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Python configuration file (e.g., key_prediction/config_linear.py)")
    parser.add_argument("--file", type=str, required=False,
                        help="Path to the pkl data file. If provided, this overrides the file_path in the config.")
    parser.add_argument("--alias", type=str, required=True,
                        help="Alias for the dataset. If provided, this overrides the alias in the config.")
    parser.add_argument("--matrix_type", type=str, required=True,
                        help="Matrix type to use. If provided, this overrides the matrix_type in the config.")
    parser.add_argument("--head_selection_file", type=str, required=True,
                        help="Path to the head selection file. If provided, this overrides the head_selection_file in the config.")
    # Add argument to explicitly choose CPU/GPU if not encoded in config's model_type
    # parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'gpu'],
    #                     help="Device to run the model on ('cpu' or 'gpu')")
    
    # Setup basic logging for trainer messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args = parser.parse_args()

    # Load configuration from the specified file
    try:
        CONFIG = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        
        # Override file_path with command line argument if provided
        if args.file:
            CONFIG['file_path'] = args.file
            print(f"Overriding file_path with command line argument: {args.file}")
            
        print(f"Using data file: {CONFIG.get('file_path', 'N/A')}")
        print(f"Using model type: {CONFIG.get('model_type', 'N/A')}")
        print(f"Include Prefill Tokens: {CONFIG.get('include_prefill', True)}") # <-- Add print statement
        print(f"Dynamic Reference Head Selection Enabled: {CONFIG.get('dynamic_ref_heads', False)}")
        if CONFIG.get('dynamic_ref_heads', False):
             print(f"Reference Head Selection Metric: {CONFIG.get('ref_head_metric', 'cos_sim_mean')}")
             # Check if static heads are present and warn if dynamic is True
             if 'ref_heads' in CONFIG or 'target_heads' in CONFIG:
                  print("Warning: Static 'ref_heads' or 'target_heads' found in config, but dynamic_ref_heads is True. Static values will be ignored.")
        else:
             # Ensure static heads are present if dynamic is False
             if 'ref_heads' not in CONFIG or 'target_heads' not in CONFIG:
                  print("Error: Dynamic reference head selection is disabled, but 'ref_heads' or 'target_heads' are missing from the config.")
                  exit(1)
        
        # Set matrix type based on config or guess from file name
        if 'matrix_type' not in CONFIG:
            raise ValueError(f"Matrix type not specified in config. Please provide a valid matrix type.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    # Load and process data using imported functions
    try:
        df_original, matrix_type = load_data(args.file, args.matrix_type)
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {args.file}") # Use logging
        exit(1)
    except ValueError as e:
       
        logging.error(f"Error loading data: {e}") # Use logging
        print("Matrix type: ", args.matrix_type)   
        print("File: ", args.file)
        exit(1)
    except Exception as e: # Catch other potential load errors
        logging.error(f"An unexpected error occurred during data loading: {e}")
        exit(1)

    try:
        # process_data now returns expanded_df and activation_dim, we only need expanded_df here
        expanded_df, _ = process_data(df_original, matrix_type)
    except ValueError as e:
         logging.error(f"Error processing data: {e}") # Use logging
         exit(1)
    except Exception as e: # Catch other potential processing errors
        logging.error(f"An unexpected error occurred during data processing: {e}")
        exit(1)

    # Check if data processing was successful
    if expanded_df is None or expanded_df.empty:
         logging.error("Data processing failed or resulted in an empty DataFrame. Exiting.") # Use logging
    else:
        logging.info("Data preprocessing complete. Starting analysis...") # Use logging
        # Run the main analysis pipeline
        results_df, per_layer_summary, low_sim_df, low_sim_summary, final_head_selections, final_models = run_analysis(expanded_df, CONFIG)

        # Optional: Save results to files
        if CONFIG.get('save_csv_results', True):
            if results_df is not None:
                model_name = CONFIG.get('model_type', 'unknown')
                matrix_name = CONFIG.get('matrix_type', 'unknown')
                # Append suffix if heads were loaded
                suffix = "_loaded_heads" if CONFIG.get('load_head_selection', False) else ""

                # Create artifacts/training_logs directory if it doesn't exist
                log_dir = 'artifacts/training_logs'
                os.makedirs(log_dir, exist_ok=True)
                
                log_path = f'{log_dir}/{model_name}_{matrix_name}_results{suffix}.csv'
                results_df.to_csv(log_path, index=False)
                print(f"Saved model results to {log_path}")
                
            if per_layer_summary is not None:
                log_path = f'{log_dir}/{model_name}_{matrix_name}_layer_summary{suffix}.csv'
                per_layer_summary.to_csv(log_path, index=False)
                print(f"Saved layer summary to {log_path}")
                
            if low_sim_df is not None:
                log_path = f'{log_dir}/{model_name}_{matrix_name}_low_similarity_samples{suffix}.csv'
                low_sim_df.to_csv(log_path, index=False)
                print(f"Saved low similarity samples to {log_path}")
                
            if low_sim_summary is not None:
                log_path = f'{log_dir}/{model_name}_{matrix_name}_low_similarity_summary{suffix}.csv'
                low_sim_summary.to_csv(log_path, index=False)
                print(f"Saved low similarity summary to {log_path}")

        # NEW: Save head selections if enabled and generated (Do not save if loaded)
        if CONFIG.get('save_head_selection', False) and final_head_selections and not CONFIG.get('load_head_selection', False):
            save_filepath = CONFIG.get('head_selection_file', 'selected_heads.json')
            try:
                with open(save_filepath, 'w') as f:
                    json.dump(final_head_selections, f, indent=4)
                print(f"Saved selected heads to {save_filepath}")
            except Exception as e:
                print(f"Error saving head selections to {save_filepath}: {e}")

        print("Analysis complete.")

# Example usage in a Jupyter notebook:
# This section remains commented out as it's not part of the direct script execution
# config = { ... } # Define config if running interactively
# df_original, matrix_type = load_data(config['file_path'], config['matrix_type']) # Update call signature
# expanded_df, _ = process_data(df_original, matrix_type) # Update call signature and handle return
# results_df, per_layer_summary, low_sim_df, low_sim_summary, final_head_selections, final_models = run_analysis(expanded_df, config)