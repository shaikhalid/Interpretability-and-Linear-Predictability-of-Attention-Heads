import sys
import os
import random
import re

# Add the parent directory of visualisation_helper to the Python path
# This allows the import 'from visualisation_helper.pastel_theme import PLOT_STYLE' to work
# even if the script is run directly from the visualisation_helper directory.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import pandas as pd
import numpy as np
import time
import argparse
import importlib.util
import logging
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial.distance import euclidean as euclidean_distance
from itertools import combinations
from tqdm import tqdm # Import tqdm for progress bar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import the pastel theme
from visualisation_helper.pastel_theme import PLOT_STYLE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading (Copied from preprocess_activations.py) ---
def load_config(config_path):
    """Load the CONFIG dictionary from a Python file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, 'CONFIG') or not isinstance(config_module.CONFIG, dict):
        raise ValueError(f"Config file {config_path} must contain a dictionary named CONFIG.")

    config = config_module.CONFIG
    # Ensure matrix_type is set, guessing if necessary
    if 'matrix_type' not in config:
        file_path = config.get('file_path', '')
        # Simple guess based on common naming, adjust if needed
        if 'v_cache' in file_path.lower() or 'value' in file_path.lower():
            config['matrix_type'] = 'v_matrix'
            logging.info(f"Automatically setting matrix_type to 'v_matrix' based on file name: {file_path}")
        elif 'k_cache' in file_path.lower() or 'key' in file_path.lower():
             config['matrix_type'] = 'k_matrix'
             logging.info(f"Automatically setting matrix_type to 'k_matrix' based on file name: {file_path}")
        elif 'q_proj' in file_path.lower() or 'query' in file_path.lower():
            # Add a guess for Q if needed, assuming a column name like 'q_matrix'
            # This part depends on how Q activations are stored.
            # config['matrix_type'] = 'q_matrix' # Example
            # logging.info(f"Automatically setting matrix_type to 'q_matrix' based on file name: {file_path}")
            # For now, default to K if unsure and Q isn't clearly indicated
             config['matrix_type'] = 'k_matrix'
             logging.warning(f"Could not determine matrix type (K/V/Q) from filename. Defaulting to 'k_matrix'. Please specify 'matrix_type' in config if incorrect.")
        else:
            config['matrix_type'] = 'k_matrix' # Default fallback
            logging.warning(f"Defaulting matrix_type to 'k_matrix'. Please specify 'matrix_type' in config if incorrect.")


    return config

# --- Data Loading and Processing (Copied from preprocess_activations.py) ---
def load_data(file_path, matrix_type='k_matrix'):
    """Load and perform initial processing on the data from pickle file."""
    loaded_data = []
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logging.info(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data_chunk = pickle.load(f)
                    if isinstance(data_chunk, list):
                        loaded_data.extend(data_chunk)
                    else:
                        # Handle potential single dict items if necessary
                        if isinstance(data_chunk, dict):
                             loaded_data.append(data_chunk)
                        else:
                             logging.warning(f"Loaded unexpected data type: {type(data_chunk)}")
                             loaded_data.append(data_chunk) # Try adding anyway
                except EOFError:
                    break
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

    logging.info(f"Loaded {len(loaded_data)} records.")
    logging.info(f"Using matrix type: {matrix_type}")

    try:
        df = pd.DataFrame(loaded_data)
        logging.info("Successfully converted loaded data to DataFrame.")
    except ValueError as e:
        logging.error(f"Failed to convert loaded data to DataFrame: {e}")
        logging.error("Ensure the pickle file contains data suitable for DataFrame creation (e.g., list of dicts or single dicts per chunk).")
        # Attempt to diagnose - check first few items
        if loaded_data:
            logging.error(f"First item type: {type(loaded_data[0])}")
            if isinstance(loaded_data[0], dict):
                 logging.error(f"First item keys: {loaded_data[0].keys()}")
            if len(loaded_data) > 1:
                 logging.error(f"Second item type: {type(loaded_data[1])}")


        raise

    if 'batch_id' in df.columns:
        df = df.rename(columns={'batch_id': 'sample_number'})
    if 'request_id' in df.columns:
        df = df.drop(columns=['request_id'])

    if matrix_type not in df.columns:
        # Check if other common names exist (e.g., q_matrix, k_matrix, v_matrix)
        potential_cols = [col for col in ['q_matrix', 'k_matrix', 'v_matrix'] if col in df.columns]
        if potential_cols:
             logging.warning(f"Config specified matrix '{matrix_type}', but it wasn't found. Found potential alternatives: {potential_cols}. Using '{potential_cols[0]}'.")
             matrix_type = potential_cols[0] # Use the first one found
        else:
            raise ValueError(f"Error: Matrix column '{matrix_type}' not found. Available columns: {df.columns.tolist()}")

    return df, matrix_type


def process_prefill_data(prefill_df, matrix_type):
    """Process prefill data by expanding matrices into single tokens."""
    expanded_prefill_rows = []
    if not prefill_df.empty:
        logging.info(f"Processing {len(prefill_df)} prefill rows...")
        for _, row in prefill_df.iterrows():
            matrix = row[matrix_type]
            # Use num_tokens_in_matrix if available, otherwise infer from shape
            if 'num_tokens_in_matrix' in row and pd.notna(row['num_tokens_in_matrix']):
                 num_prefill_tokens = int(row['num_tokens_in_matrix'])
            elif hasattr(matrix, 'shape') and len(matrix.shape) > 1:
                 num_prefill_tokens = matrix.shape[1]
            else:
                 # Check if this row represents a single decode token (num_decode_tokens > 0)
                 # even if it ended up in prefill_df due to matrix shape check
                 if 'num_decode_tokens' in row and pd.notna(row['num_decode_tokens']) and row['num_decode_tokens'] > 0:
                     logging.debug(f"Row initially marked as prefill seems to be a decode step based on 'num_decode_tokens'. Skipping prefill expansion.")
                     continue # Skip expanding this, it should be handled by decode logic
                 else:
                     logging.warning(f"Could not determine number of prefill tokens for sample {row.get('sample_number', 'N/A')}, layer {row.get('layer_id', 'N/A')}, head {row.get('head_id', 'N/A')}. Skipping row.")
                     continue # Skip this row if token count is unclear


            # Ensure matrix has at least 3 dimensions (channels, tokens, embed_dim)
            # If shape is (tokens, embed_dim), add a channel dimension
            if hasattr(matrix, 'shape') and len(matrix.shape) == 2:
                 matrix = np.expand_dims(matrix, axis=0)
                 # logging.debug(f"Added channel dimension. New shape: {matrix.shape}")
            elif not (hasattr(matrix, 'shape') and len(matrix.shape) >= 3):
                 logging.warning(f"Unexpected matrix shape {getattr(matrix, 'shape', 'N/A')} for prefill sample {row.get('sample_number', 'N/A')}, layer {row.get('layer_id', 'N/A')}, head {row.get('head_id', 'N/A')}. Skipping row.")
                 continue

            # Check if inferred num_prefill_tokens matches matrix shape
            # Adjust num_prefill_tokens based on actual matrix dimension if mismatch
            if matrix.shape[1] != num_prefill_tokens:
                 logging.warning(f"Mismatch between detected 'num_tokens_in_matrix' ({num_prefill_tokens}) and actual matrix shape ({matrix.shape[1]}) for sample {row.get('sample_number', 'N/A')}. Using actual matrix shape.")
                 num_prefill_tokens = matrix.shape[1]


            base_info_cols = list(row.index.drop([matrix_type], errors='ignore'))
            # Also remove shape/token count related columns explicitly if they exist
            base_info_cols = [c for c in base_info_cols if c not in ['num_tokens_in_matrix', 'matrix_shape']]
            base_info = row[base_info_cols]

            for token_idx in range(num_prefill_tokens):
                try:
                    # Slice to get (channels, 1, embed_dim)
                    token_matrix = matrix[:, token_idx:token_idx+1, :]
                    row_data = base_info.to_dict()
                    row_data[matrix_type] = token_matrix
                    row_data['token_position'] = token_idx
                    row_data['prefill'] = True
                    expanded_prefill_rows.append(row_data)
                except IndexError:
                    logging.warning(f"IndexError during prefill expansion for sample {row.get('sample_number', 'N/A')}, token {token_idx}. Matrix shape: {matrix.shape}. Skipping token.")
                    continue
                except Exception as e:
                    logging.warning(f"Error expanding prefill token {token_idx} for sample {row.get('sample_number', 'N/A')}: {e}. Skipping token.")
                    continue

        if expanded_prefill_rows:
            expanded_prefill_df = pd.DataFrame(expanded_prefill_rows)
        else: # Handle case where all prefill rows were skipped or df was empty
            # Try to get columns from original prefill_df if possible
            if not prefill_df.empty:
                 base_cols = list(prefill_df.columns.drop([matrix_type], errors='ignore'))
                 base_cols = [c for c in base_cols if c not in ['num_tokens_in_matrix', 'matrix_shape']]
            else: # Define essential columns if prefill_df was empty
                 base_cols = ['sample_number', 'layer_id', 'head_id'] # Add other expected base columns if known
            final_cols = base_cols + [matrix_type, 'token_position', 'prefill']
            # Ensure no duplicate columns before creating DataFrame
            final_cols = list(dict.fromkeys(final_cols))
            expanded_prefill_df = pd.DataFrame(columns=final_cols)


        # Create sample_n_map: map sample_number to its total number of prefill tokens
        # This is crucial for calculating decode token positions correctly.
        # We should base this on the original prefill prompts (where num_decode_tokens == 0 or not present)
        # We need the original df for this if prefill_df was modified
        # Let's recalculate N based on the expanded prefill tokens per sample_number

        if not expanded_prefill_df.empty:
             # Calculate N = max token_position + 1 for prefill tokens per sample
             # This map should contain the length of the prefill sequence for each sample
             sample_n_map = expanded_prefill_df.groupby('sample_number')['token_position'].max() + 1
             sample_n_map = sample_n_map.fillna(0).astype(int)
        else:
             sample_n_map = pd.Series(dtype=int)


    else:
        # Define necessary columns even if empty
        # Try to get columns from original prefill_df if possible
        if not prefill_df.empty:
            base_cols = list(prefill_df.columns.drop([matrix_type], errors='ignore'))
            base_cols = [c for c in base_cols if c not in ['num_tokens_in_matrix', 'matrix_shape']]
        else: # Define essential columns if prefill_df was empty
            base_cols = ['sample_number', 'layer_id', 'head_id'] # Add other expected base columns if known
        final_cols = base_cols + [matrix_type, 'token_position', 'prefill']
        # Ensure no duplicate columns before creating DataFrame
        final_cols = list(dict.fromkeys(final_cols))
        expanded_prefill_df = pd.DataFrame(columns=final_cols)
        sample_n_map = pd.Series(dtype=int)

    return expanded_prefill_df, sample_n_map


def process_decode_data(decode_df, sample_n_map, matrix_type):
    """Process decode data by calculating token positions."""
    if not decode_df.empty:
        logging.info(f"Processing {len(decode_df)} decode rows...")
        if 'sample_number' not in decode_df.columns:
             logging.error("Missing 'sample_number' column in decode data. Cannot calculate token positions.")
             # Return empty or raise error depending on desired behavior
             decode_df['token_position'] = -1 # Indicate invalid position
             decode_df['prefill'] = False
             return decode_df # Return with invalid positions


        # Ensure 'num_decode_tokens' exists
        if 'num_decode_tokens' not in decode_df.columns:
             logging.warning("Missing 'num_decode_tokens' column in decode data. Attempting to infer token position assuming sequential decode steps.")
             # Attempt to infer position by ranking within sample/layer/head group
             # This is less reliable than using num_decode_tokens
             # It assumes the decode_df is sorted correctly or groupable
             decode_df = decode_df.sort_values(by=['sample_number', 'layer_id', 'head_id']) # Ensure some order
             # Calculate inferred decode step within each group
             # This assumes each row is one step, might be wrong if data isn't structured that way
             decode_df['inferred_decode_step'] = decode_df.groupby(['sample_number', 'layer_id', 'head_id']).cumcount()
             decode_df['N'] = decode_df['sample_number'].map(sample_n_map).fillna(0).astype(int)
             decode_df['token_position'] = decode_df['inferred_decode_step'] + decode_df['N']
             logging.warning("Calculated token_position based on inferred sequential order. Accuracy depends on data structure.")

        else:
             # Calculate N (length of prefill) for each sample
             decode_df['N'] = decode_df['sample_number'].map(sample_n_map).fillna(0).astype(int)
             # Decode token position = (num_decode_tokens - 1) + N
             # Ensure num_decode_tokens is numeric, handle potential NaNs
             decode_df['num_decode_tokens'] = pd.to_numeric(decode_df['num_decode_tokens'], errors='coerce')
             decode_df.dropna(subset=['num_decode_tokens'], inplace=True) # Drop rows where conversion failed
             decode_df['num_decode_tokens'] = decode_df['num_decode_tokens'].astype(int)

             decode_df['token_position'] = (decode_df['num_decode_tokens'] - 1) + decode_df['N']


        decode_df['prefill'] = False

        # Ensure the matrix column contains single token activations
        # Expected shape might be (channels, 1, embed_dim) or similar
        # Add check if needed based on data format
        def check_decode_matrix(matrix):
             if hasattr(matrix, 'shape') and len(matrix.shape) >= 2:
                 if matrix.shape[1] != 1:
                     logging.warning(f"Decode matrix has unexpected token dimension: {matrix.shape}. Expected 1 token. Taking first token slice.")
                     # Attempt to fix by taking the first token: matrix[:, 0:1, :]
                     return matrix[:, 0:1, :]
             # Removed check for 1D matrix shape - assume pre-processing handles dimensionality
             # elif len(matrix.shape) == 1: # If shape is (embed_dim)
             #    logging.warning(f"Decode matrix has shape {matrix.shape}. Adding channel and token dimensions.")
             #    return np.expand_dims(np.expand_dims(matrix, axis=0), axis=0) # Add channel=1, token=1

             return matrix # Return original or potentially modified matrix


        # Apply the check/fix
        decode_df[matrix_type] = decode_df[matrix_type].apply(check_decode_matrix)


        # Drop temporary columns used for calculation
        drop_cols = ['N', 'num_tokens_in_matrix', 'matrix_shape', 'inferred_decode_step']
        decode_df = decode_df.drop(columns=[col for col in drop_cols if col in decode_df.columns], errors='ignore')

    return decode_df


def process_data(df, matrix_type):
    """Main function to process the DataFrame and expand matrices."""
    start_time = time.time()
    logging.info("Starting data processing to expand matrices...")
    df_processed = df.copy()

    required_cols = [matrix_type, 'sample_number']
    # Check for essential columns needed for processing
    if 'layer_id' not in df_processed.columns or 'head_id' not in df_processed.columns:
         raise ValueError("Input data must contain 'layer_id' and 'head_id' columns.")

    # Determine matrix shape and number of tokens robustly
    df_processed['matrix_shape'] = df_processed[matrix_type].apply(lambda x: getattr(x, 'shape', None))

    # Infer num_tokens_in_matrix from shape[1] if possible, default to 1 otherwise
    def get_token_count(x):
        if hasattr(x, 'shape') and len(x.shape) > 1:
            return x.shape[1]
        # Consider if shape is just (embed_dim,) -> means 1 token
        elif hasattr(x, 'shape') and len(x.shape) == 1:
             return 1
        # Default for non-array or unexpected shapes
        return 1

    df_processed['num_tokens_in_matrix'] = df_processed[matrix_type].apply(get_token_count)


    # Identify prefill rows vs decode rows more carefully
    # A row is prefill if it has multiple tokens in its matrix AND num_decode_tokens is 0 or NaN/missing
    # A row is decode if it has 1 token in its matrix OR num_decode_tokens > 0
    is_prefill = (df_processed['num_tokens_in_matrix'] > 1)
    if 'num_decode_tokens' in df_processed.columns:
         # Make sure 'num_decode_tokens' is numeric before comparison
         df_processed['num_decode_tokens_numeric'] = pd.to_numeric(df_processed['num_decode_tokens'], errors='coerce')
         # Prefill is where num_tokens > 1 AND (num_decode is 0 or NaN)
         is_prefill = is_prefill & ( (df_processed['num_decode_tokens_numeric'] == 0) | df_processed['num_decode_tokens_numeric'].isna() )
         # Decode is where num_tokens == 1 OR num_decode > 0
         is_decode = (df_processed['num_tokens_in_matrix'] == 1) | ( (df_processed['num_decode_tokens_numeric'] > 0) & df_processed['num_decode_tokens_numeric'].notna() )
    else:
         # If num_decode_tokens is missing, rely solely on matrix shape
         is_decode = (df_processed['num_tokens_in_matrix'] == 1)
         logging.warning("'num_decode_tokens' column not found. Distinguishing prefill/decode based solely on matrix token count.")


    prefill_df = df_processed[is_prefill].copy()
    # Combine explicit decode rows and single-token rows that weren't marked prefill
    # Use original index to avoid duplicates if logic overlaps, then select from df_processed
    decode_indices = df_processed.index[is_decode]
    # Ensure indices used for prefill aren't included in decode
    decode_indices = decode_indices.difference(prefill_df.index)
    decode_df = df_processed.loc[decode_indices].copy()

    # Drop the temporary numeric column if created
    if 'num_decode_tokens_numeric' in df_processed.columns:
         df_processed = df_processed.drop(columns=['num_decode_tokens_numeric'])
         prefill_df = prefill_df.drop(columns=['num_decode_tokens_numeric'], errors='ignore')
         decode_df = decode_df.drop(columns=['num_decode_tokens_numeric'], errors='ignore')


    logging.info(f"Identified {len(prefill_df)} prefill rows and {len(decode_df)} decode rows.")


    expanded_prefill_df, sample_n_map = process_prefill_data(prefill_df, matrix_type)
    decode_df_processed = process_decode_data(decode_df, sample_n_map, matrix_type)

    # Combine results
    # Use columns from the processed decode df as the base + any extra from prefill
    common_columns = list(decode_df_processed.columns)
    for col in expanded_prefill_df.columns:
        if col not in common_columns:
            common_columns.append(col)

    # Ensure both DFs have the common columns before concat, fill missing with NA/default
    for df_to_check in [expanded_prefill_df, decode_df_processed]:
        for col in common_columns:
            if col not in df_to_check.columns:
                # Determine appropriate fill value based on expected dtype if possible
                fill_value = pd.NA
                if col in ['prefill']: fill_value = False # Default prefill to False if missing
                # Add more type-specific defaults if needed
                df_to_check[col] = fill_value

    # Concatenate using only common columns to ensure alignment
    expanded_df = pd.concat([expanded_prefill_df[common_columns], decode_df_processed[common_columns]], ignore_index=True)

    # Convert types for sorting and ensure essential columns are numeric
    for col in ['sample_number', 'layer_id', 'head_id', 'token_position']:
         if col in expanded_df.columns:
             expanded_df[col] = pd.to_numeric(expanded_df[col], errors='coerce') # Coerce errors to NaN


    # Drop rows where essential identifiers are NaN after conversion/concat
    essential_cols_check = ['sample_number', 'layer_id', 'head_id', 'token_position', matrix_type]
    expanded_df.dropna(subset=[col for col in essential_cols_check if col in expanded_df.columns], inplace=True)

     # Cast identifiers to int after dropping NaNs
    for col in ['sample_number', 'layer_id', 'head_id', 'token_position']:
         if col in expanded_df.columns:
             # Check if column is already int to avoid errors
             if not pd.api.types.is_integer_dtype(expanded_df[col]):
                 expanded_df[col] = expanded_df[col].astype(int)


    expanded_df = expanded_df.sort_values(by=['sample_number', 'layer_id', 'head_id', 'token_position'], ignore_index=True)


    logging.info(f"Data processing time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Original DF shape: {df.shape}")
    logging.info(f"Expanded DF shape (token level): {expanded_df.shape}")
    # logging.info(f"Columns in expanded DF: {expanded_df.columns.tolist()}")


    return expanded_df


# --- Distance Calculation ---
def calculate_distance(vec1, vec2, metric='cosine'):
    """Calculates distance between two vectors."""
    # Ensure vectors are 1D numpy arrays
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    if vec1.shape != vec2.shape:
        logging.debug(f"Shape mismatch for distance calculation: {vec1.shape} - {vec2.shape}. Returning NaN.") # Changed to debug
        return np.nan

    # Check for NaN/inf values
    if not np.isfinite(vec1).all() or not np.isfinite(vec2).all():
        logging.debug("NaN or inf found in vectors. Returning NaN.")
        return np.nan


    if metric == 'cosine':
        # Check for zero vectors before cosine calculation
        if np.all(vec1 == 0) or np.all(vec2 == 0):
             logging.debug("Zero vector encountered in cosine distance calculation. Returning NaN.")
             return np.nan # Cosine distance is undefined for zero vectors

        try:
             # Ensure inputs are float64 for precision in distance calculation
             dist = cosine_distance(vec1.astype(np.float64), vec2.astype(np.float64))
             # Clamp dist between 0 and 2 (cosine similarity is -1 to 1)
             # Check for potential numerical instability resulting in NaN
             if np.isnan(dist):
                  logging.debug("Cosine distance resulted in NaN. Vectors might be identical or near-identical with precision issues.")
                  # If vectors are extremely close, distance is near 0
                  if np.allclose(vec1, vec2): return 0.0
                  else: return np.nan # Return NaN if it's not due to closeness

             return np.clip(dist, 0.0, 2.0)
        except Exception as e:
             logging.warning(f"Error during cosine distance calculation: {e}. Returning NaN.")
             return np.nan
    # elif metric == 'euclidean': # Removing euclidean for now based on request focusing on cosine std dev
    #     try:
    #         return euclidean_distance(vec1.astype(np.float64), vec2.astype(np.float64))
    #     except Exception as e:
    #         logging.warning(f"Error during euclidean distance calculation: {e}. Returning NaN.")
    #         return np.nan
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


# --- New function to calculate std dev for a pair ---
def calculate_pair_distance_stddev(pair, df_filtered_by_layerhead, matrix_type, metric='cosine'):
    """
    Calculates the standard deviation of distances and R2 score between two layer-head pairs.

    Args:
        pair (tuple): A tuple containing two layer-head tuples, e.g., ((l1, h1), (l2, h2)).
        df_filtered_by_layerhead (dict): A dictionary mapping layer-head tuples to their filtered DataFrames.
        matrix_type (str): The name of the column containing activation matrices.
        metric (str): The distance metric to use (currently only 'cosine').

    Returns:
        tuple: (std_dev, r2_score_val, merged_df_with_distances)
               std_dev and r2_score_val are np.nan if calculation fails or insufficient data.
               merged_df_with_distances contains matched tokens and calculated distances.
    """
    lh1, lh2 = pair
    l1, h1 = lh1
    l2, h2 = lh2
    std_dev = np.nan # Initialize std_dev
    r2_score_val = np.nan # Initialize r2_score

    # Check if data exists for both layer-heads in the pre-filtered dict
    if lh1 not in df_filtered_by_layerhead or lh2 not in df_filtered_by_layerhead:
        logging.warning(f"Data not pre-filtered for pair {lh1}-{lh2}. Skipping.")
        return std_dev, r2_score_val, None

    df1 = df_filtered_by_layerhead[lh1]
    df2 = df_filtered_by_layerhead[lh2]

    if df1.empty or df2.empty:
        logging.debug(f"Empty data for one or both parts of pair {lh1}-{lh2}. Skipping std dev & R2 calculation.")
        return std_dev, r2_score_val, None

    # Merge dataframes on sample and token position
    merged_df = pd.merge(
        df1[['sample_number', 'token_position', matrix_type]],
        df2[['sample_number', 'token_position', matrix_type]],
        on=['sample_number', 'token_position'],
        suffixes=('_1', '_2')
    )

    if merged_df.empty:
        logging.debug(f"No matching sample/token positions found for pair {lh1}-{lh2}.")
        return std_dev, r2_score_val, None

    # Extract vectors (squeeze if needed)
    def get_vector(matrix):
        if isinstance(matrix, np.ndarray):
            # Squeeze potential channel dim if it's 1, then flatten
            squeezed = matrix.squeeze()
            # Ensure it's flattened to 1D
            return squeezed.flatten()
        else:
            logging.warning(f"Encountered non-ndarray type in matrix column: {type(matrix)}. Returning None.")
            return None

    merged_df['vec1'] = merged_df[matrix_type + '_1'].apply(get_vector)
    merged_df['vec2'] = merged_df[matrix_type + '_2'].apply(get_vector)

    # Drop rows where vector extraction failed
    merged_df.dropna(subset=['vec1', 'vec2'], inplace=True)

    if merged_df.empty:
        logging.debug(f"No valid vector pairs found after extraction for pair {lh1}-{lh2}.")
        return std_dev, r2_score_val, None

    # Calculate distances
    merged_df['distance'] = merged_df.apply(
        lambda row: calculate_distance(row['vec1'], row['vec2'], metric=metric),
        axis=1
    )

    # Drop rows where distance calculation resulted in NaN
    merged_df.dropna(subset=['distance'], inplace=True)

    # Check if enough data points remain for std dev calculation (at least 2)
    if len(merged_df) < 2:
        logging.debug(f"Insufficient valid distance points (<2) for std dev/R2 calculation for pair {lh1}-{lh2}.")
        return std_dev, r2_score_val, merged_df # Return NaN std dev/R2 but keep df if needed

    # --- Calculate Standard Deviation ---
    std_dev = merged_df['distance'].std()
    if pd.isna(std_dev):
        std_dev = 0.0 # Std dev is 0 if all values are the same

    # --- Calculate R2 Score using Linear Regression ---
    try:
        # Prepare data for regression (ensure shapes are consistent)
        X_lr_raw = merged_df['vec1'].tolist()
        y_lr_raw = merged_df['vec2'].tolist()

        # Filter out None and ensure they are numpy arrays before stacking
        X_lr_list = [x for x in X_lr_raw if isinstance(x, np.ndarray)]
        y_lr_list = [y for y in y_lr_raw if isinstance(y, np.ndarray)]

        if len(X_lr_list) != len(y_lr_list):
             logging.warning(f"Mismatch in valid vector counts for pair {lh1}-{lh2} after filtering. X: {len(X_lr_list)}, Y: {len(y_lr_list)}. Skipping R2.")
             return std_dev, r2_score_val, merged_df

        if not X_lr_list: # Check if list is empty
             logging.warning(f"No valid vectors found for linear regression for pair {lh1}-{lh2}. Skipping R2.")
             return std_dev, r2_score_val, merged_df

        # Check dimensions before stacking
        dim_x = X_lr_list[0].shape[0]
        dim_y = y_lr_list[0].shape[0]
        if not all(x.shape[0] == dim_x for x in X_lr_list) or not all(y.shape[0] == dim_y for y in y_lr_list):
             logging.warning(f"Inconsistent vector dimensions within pair {lh1}-{lh2}. Skipping R2.")
             # Optionally, filter further based on consistent dimensions
             return std_dev, r2_score_val, merged_df

        X_lr = np.stack(X_lr_list)
        y_lr = np.stack(y_lr_list)

        if X_lr.shape[0] >= 2: # Need at least 2 samples for regression
            model = LinearRegression()
            model.fit(X_lr, y_lr)
            y_pred = model.predict(X_lr)
            r2_score_val = r2_score(y_lr, y_pred)
            # Handle case where R2 might be calculated per output feature if y_lr is 2D
            # For a single R2 score representing overall fit, we might need multioutput='variance_weighted' or 'uniform_average'
            if isinstance(r2_score_val, np.ndarray): # If r2_score returns array (per target feature)
                 r2_score_val = r2_score(y_lr, y_pred, multioutput='variance_weighted') # Example: weighted average R2

        else:
            logging.debug(f"Less than 2 valid samples for Linear Regression for pair {lh1}-{lh2}. Skipping R2.")

    except Exception as e:
        logging.warning(f"Error during Linear Regression or R2 calculation for pair {lh1}-{lh2}: {e}. Skipping R2.")
        # Keep r2_score_val as np.nan

    return std_dev, r2_score_val, merged_df


# --- Modified Plotting Function ---
def plot_comparison_distances(pairs_data, output_path, matrix_type_name, metric):
    """Plots the mean distances for multiple layer-head pairs on the same axes.
    Note: R2 scores are calculated but not shown in this plot version.
    """
    # --- Apply Font and Style Settings from Theme ---
    plt.rcParams['font.family'] = PLOT_STYLE['font_family']
    # ---

    fig = plt.figure(figsize=(14, 7)) # Slightly wider for legend
    ax = plt.gca()

    # Set background colors
    fig.set_facecolor(PLOT_STYLE['background_color'])
    ax.set_facecolor(PLOT_STYLE['background_color'])

    # Define distinct styles (can add more if needed)
    # Use theme colors first, then cycle if more pairs are needed
    colors = [PLOT_STYLE['line_color'], '#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6'] # Pastel + some distinct ones
    # *** CHANGE: Use fixed linestyle and marker ***
    fixed_linestyle = '-' # Example: solid line
    fixed_marker = '.'   # Example: point marker
    style_idx = 0 # Index for cycling through colors

    if not pairs_data:
        logging.warning("No valid pair data provided for plotting.")
        plt.close()
        return

    # *** NEW: Sort pairs_data by standard deviation (index 3) in descending order ***
    try:
        pairs_data.sort(key=lambda item: item[3], reverse=True)
        logging.info("Plotting lines ordered by descending standard deviation.")
    except IndexError:
        logging.error("Could not sort plot data by standard deviation. Ensure std_dev is included in pairs_data.")
        # Continue without sorting if error occurs
    except Exception as e:
        logging.error(f"Error sorting plot data: {e}")
        # Continue without sorting

    # --- Plot each pair's mean distance ---
    # pairs_data contains tuples of (label_prefix, pair_tuple, distance_df)
    # R2 is no longer passed here
    for label, pair_tuple, distance_df in pairs_data:
        if distance_df is None or distance_df.empty or 'distance' not in distance_df.columns or distance_df['distance'].isnull().all():
            logging.warning(f"Skipping plot for '{label}' due to invalid or empty distance data.")
            continue

        # Calculate mean distance per token position for this pair
        mean_distances = distance_df.groupby('token_position')['distance'].mean()

        if mean_distances.empty:
             logging.warning(f"Skipping plot for '{label}' as no mean distances could be calculated.")
             continue


        lh1_str = f"L{pair_tuple[0][0]}H{pair_tuple[0][1]}"
        lh2_str = f"L{pair_tuple[1][0]}H{pair_tuple[1][1]}"
        # R2 score is no longer included in the label
        # r2_str = f"R²={r2:.3f}" if pd.notna(r2) else "R²=N/A"
        # plot_label = f"{label}: {lh1_str}-{lh2_str} ({r2_str})"
        plot_label = f"{label}: {lh1_str}-{lh2_str}" # Simple label

        # *** CHANGE: Use fixed style, cycle color ***
        current_color = colors[style_idx % len(colors)]
        # current_linestyle = linestyles[style_idx % len(linestyles)] # REMOVED
        # current_marker = markers[style_idx % len(markers)] # REMOVED


        plt.plot(mean_distances.index, mean_distances.values,
                 marker=fixed_marker, linestyle=fixed_linestyle, color=current_color, # USE FIXED STYLES
                 label=plot_label,
                 linewidth=PLOT_STYLE.get('line_width', 1.5), # Use theme linewidth or default
                 markerfacecolor=current_color, # Use line color for markers too
                 markeredgecolor=current_color,
                 markersize=5) # Smaller markers might be better for multiple lines

        style_idx += 1 # Increment only for color cycling

    # Check if any lines were actually plotted
    if not ax.lines:
         logging.error("No data was plotted. Check input data and filtering steps.")
         plt.close()
         return


    # Apply text styles from theme
    plt.xlabel("Token Position", fontsize=PLOT_STYLE['label_fontsize'])
    plt.ylabel(f"Mean {metric.capitalize()} Distance", fontsize=PLOT_STYLE['label_fontsize'])
    plt.title(f"Comparison of Mean {metric.capitalize()} Distance between {matrix_type_name} Activations",
              fontsize=PLOT_STYLE['title_fontsize'])

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
    ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])

    # Add grid
    plt.grid(True, which='both', linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'])

    # Add legend
    # Place legend outside the plot area if possible
    plt.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 'small'), bbox_to_anchor=(1.04, 1), loc="upper left")


    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for external legend


    # Determine output format, force PDF
    output_format = 'pdf'
    # Ensure output path has .pdf extension
    if not output_path.lower().endswith('.pdf'):
        output_path = os.path.splitext(output_path)[0] + '.pdf'

    logging.info(f"Saving comparison plot as PDF vector graphic to: {output_path}")
    try:
        plt.savefig(output_path, format=output_format, bbox_inches='tight', facecolor=fig.get_facecolor())
        logging.info(f"Plot saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {output_path}: {e}")


    plt.close() # Close the figure to free memory


# --- New CDF Plotting Function ---
def plot_comparison_cdf(pairs_data, output_path, matrix_type_name, metric):
    """Plots the CDF of distances for multiple layer-head pairs on the same axes.
    Note: R2 scores are calculated but not shown in this plot version.
    """
    # --- Apply Font and Style Settings from Theme ---
    plt.rcParams['font.family'] = PLOT_STYLE['font_family']
    # ---

    fig = plt.figure(figsize=(14, 7)) # Slightly wider for legend
    ax = plt.gca()

    # Set background colors
    fig.set_facecolor(PLOT_STYLE['background_color'])
    ax.set_facecolor(PLOT_STYLE['background_color'])

    # Define the exact same distinct styles as the mean plot function
    colors = [PLOT_STYLE['line_color'], '#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    fixed_linestyle = '-' # Use the same fixed style
    fixed_marker = '.'   # Use the same fixed marker (though markers might overlap on CDF)
    style_idx = 0 # Index for cycling through colors

    if not pairs_data:
        logging.warning("No valid pair data provided for CDF plotting.")
        plt.close()
        return

    # --- Plot each pair's distance CDF ---
    # pairs_data contains tuples of (label_prefix, pair_tuple, distance_df)
    # R2 is no longer passed here
    for label, pair_tuple, distance_df in pairs_data:
        if distance_df is None or distance_df.empty or 'distance' not in distance_df.columns or distance_df['distance'].isnull().all():
            logging.warning(f"Skipping CDF plot for '{label}' due to invalid or empty distance data.")
            continue

        distances = distance_df['distance'].dropna().sort_values()
        if distances.empty:
            logging.warning(f"Skipping CDF plot for '{label}' as no valid distances were found.")
            continue

        n = len(distances)
        cdf_y = np.arange(1, n + 1) / n

        lh1_str = f"L{pair_tuple[0][0]}H{pair_tuple[0][1]}"
        lh2_str = f"L{pair_tuple[1][0]}H{pair_tuple[1][1]}"
        # R2 score is no longer included in the label
        # r2_str = f"R²={r2:.3f}" if pd.notna(r2) else "R²=N/A"
        # plot_label = f"{label}: {lh1_str}-{lh2_str} ({r2_str})"
        plot_label = f"{label}: {lh1_str}-{lh2_str}" # Simple label

        # Use the same color cycling and fixed styles
        current_color = colors[style_idx % len(colors)]

        plt.plot(distances, cdf_y,
                 marker=None, # Typically, markers are not used for CDF plots
                 linestyle=fixed_linestyle, color=current_color,
                 label=plot_label,
                 linewidth=PLOT_STYLE.get('line_width', 1.5),
                 markersize=3) # Keep marker size small if used, but usually None is better

        style_idx += 1 # Increment only for color cycling

    # Check if any lines were actually plotted
    if not ax.lines:
         logging.error("No CDF data was plotted. Check input data and filtering steps.")
         plt.close()
         return

    # Apply text styles from theme
    plt.xlabel(f"{metric.capitalize()} Distance", fontsize=PLOT_STYLE['label_fontsize'])
    plt.ylabel("Cumulative Probability", fontsize=PLOT_STYLE['label_fontsize'])
    plt.title(f"CDF of {metric.capitalize()} Distances between {matrix_type_name} Activations",
              fontsize=PLOT_STYLE['title_fontsize'])

    # Set y-axis limits for CDF
    plt.ylim(0, 1.05)

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
    ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])

    # Add grid
    plt.grid(True, which='both', linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'])

    # Add legend
    plt.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 'small'), bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for external legend

    # Determine output format (PDF) and filename
    output_format = 'pdf'
    base, ext = os.path.splitext(output_path)
    if not ext.lower() == '.pdf': # Ensure base name doesn't include original extension if not pdf
        ext = '.pdf'
    cdf_output_path = f"{base}_cdf{ext}"

    logging.info(f"Saving comparison CDF plot as PDF vector graphic to: {cdf_output_path}")
    try:
        plt.savefig(cdf_output_path, format=output_format, bbox_inches='tight', facecolor=fig.get_facecolor())
        logging.info(f"CDF plot saved successfully to {cdf_output_path}")
    except Exception as e:
        logging.error(f"Failed to save CDF plot to {cdf_output_path}: {e}")

    plt.close() # Close the figure to free memory


# --- Helper function to parse layer-head string ---
def parse_layer_head_string(lh_str):
    """Parses a string like 'L12H3' into a tuple (12, 3)."""
    match = re.match(r"L(\d+)H(\d+)", lh_str, re.IGNORECASE)
    if match:
        layer = int(match.group(1))
        head = int(match.group(2))
        return (layer, head)
    else:
        raise ValueError(f"Invalid layer-head format: '{lh_str}'. Expected format like 'L<num>H<num>'.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find layer-head pairs with min/max/2nd-min standard deviation of cosine distance, or analyze specific heads, and plot their mean distances and CDFs.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Python configuration file (e.g., key_prediction/config_linear.py) containing 'file_path' and potentially 'matrix_type'.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output comparison plot PDF (e.g., distance_comparison.pdf). Suffixes like '_cdf' will be added.")
    parser.add_argument("--num_pairs_to_sample", type=int, default=None, # Make default None
                        help="Number (N) of random adjacent layer-head pairs to sample for analysis (used if --heads is not provided). Required if --heads is not used.")
    parser.add_argument("--heads", type=str, nargs=4, default=None,
                        help="Specify exactly four layer-head pairs (e.g., L0H6 L1H3 L5H2 L6H1) to analyze directly, bypassing sampling and std dev selection. Format: L<layer>H<head>")
    parser.add_argument("--exclude_prefill", action='store_true',
                        help="Exclude prefill tokens from the analysis.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (used only for sampling).")

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.heads is None and args.num_pairs_to_sample is None:
        parser.error("Either --heads (with 4 values) or --num_pairs_to_sample must be specified.")
    if args.heads is not None and args.num_pairs_to_sample is not None:
        logging.warning("Both --heads and --num_pairs_to_sample specified. --heads will be used, --num_pairs_to_sample will be ignored.")
        args.num_pairs_to_sample = None # Ensure sampling is not triggered

    # --- Setup ---
    start_run_time = time.time()
    metric = 'cosine' # Force cosine distance for std dev calculation and plotting
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        # 1. Load Config
        logging.info(f"Loading configuration from: {args.config}")
        CONFIG = load_config(args.config)
        input_file = CONFIG['file_path']
        matrix_type = CONFIG['matrix_type'] # Use the type determined by load_config
        matrix_type_name = matrix_type.replace('_matrix','').upper() # For plot title (K, V, Q)

        logging.info(f"Input data file: {input_file}")
        logging.info(f"Matrix type: {matrix_type}")
        logging.info(f"Distance metric for analysis and plot: {metric}")
        logging.info(f"Number of pairs to sample: {args.num_pairs_to_sample}")
        logging.info(f"Exclude prefill tokens: {args.exclude_prefill}")
        logging.info(f"Random seed: {args.seed}")


        # 2. Load Data into DataFrame
        df_original, matrix_type = load_data(input_file, matrix_type) # Update matrix_type if load_data changed it

        # 3. Process Data (Expand Tokens)
        expanded_df = process_data(df_original, matrix_type)
        if expanded_df.empty:
            logging.error("Expanded DataFrame is empty after processing. Cannot proceed.")
            exit(1)

        # --- Optionally filter out prefill tokens ---
        if args.exclude_prefill:
            initial_rows = len(expanded_df)
            expanded_df = expanded_df[~expanded_df['prefill']].copy()
            logging.info(f"Excluding prefill tokens. Filtered {initial_rows - len(expanded_df)} rows.")
            if expanded_df.empty:
                logging.error("DataFrame is empty after removing prefill tokens. Cannot proceed.")
                exit(1)
        else:
            logging.info("Including prefill tokens in the analysis.")

        # Ensure matrix column contains numpy arrays after processing
        # Optional: Add stricter check here if needed
        # if not expanded_df[matrix_type].apply(lambda x: isinstance(x, np.ndarray)).all():
        #     logging.warning(f"Not all entries in '{matrix_type}' column are NumPy arrays after processing. Proceeding, but check data integrity.")

        # --- Identify Layer/Head Pairs ---
        if 'layer_id' not in expanded_df.columns or 'head_id' not in expanded_df.columns:
             logging.error("Processed data missing 'layer_id' or 'head_id'. Cannot identify pairs.")
             exit(1)

        # Ensure layer/head IDs are integers
        expanded_df['layer_id'] = expanded_df['layer_id'].astype(int)
        expanded_df['head_id'] = expanded_df['head_id'].astype(int)

        available_lh = set(expanded_df[['layer_id', 'head_id']].drop_duplicates().apply(tuple, axis=1))
        if not available_lh:
            logging.error("No layer-head pairs found in the processed data.")
            exit(1)
        logging.info(f"Found {len(available_lh)} unique layer-head pairs in the data.")

        plot_data = [] # Initialize plot_data list

        # --- Mode Selection: Specific Heads or Sampling/Selection --- #
        if args.heads:
            # --- Specific Heads Mode ---
            logging.info(f"Analyzing specific heads provided: {args.heads}")
            try:
                HeadA, HeadB, HeadC, HeadD = [parse_layer_head_string(h) for h in args.heads]
            except ValueError as e:
                logging.error(f"Error parsing provided heads: {e}")
                exit(1)

            specified_heads_list = [HeadA, HeadB, HeadC, HeadD]
            logging.info(f"Parsed specific heads: {specified_heads_list}")

            # Check if specified heads exist in the data
            missing_heads = [head for head in specified_heads_list if head not in available_lh]
            if missing_heads:
                logging.error(f"The following specified heads were not found in the data: {missing_heads}. Available heads: {sorted(list(available_lh))}")
                exit(1)

            # Define the three pairs to analyze based on the four heads
            pairs_to_analyze = [(HeadA, HeadB), (HeadB, HeadC), (HeadC, HeadD)]
            logging.info(f"Analyzing pairs derived from specified heads: {pairs_to_analyze}")

            # --- Pre-filter DataFrame for the specified heads ---
            logging.info("Pre-filtering data for specified layer-heads...")
            df_filtered_by_lh = {}
            involved_lh = set(specified_heads_list)
            for lh in tqdm(involved_lh, desc="Filtering Data"):
                 l, h = lh
                 # Use the already computed expanded_df
                 df_filtered_by_lh[lh] = expanded_df[(expanded_df['layer_id'] == l) & (expanded_df['head_id'] == h)].copy()

            # --- Calculate Distances and R2 for the 3 specific pairs ---
            logging.info(f"Calculating {metric} distance & R2 score for the 3 specified pairs...")
            results_specific = []
            for pair in tqdm(pairs_to_analyze, desc="Analyzing Specific Pairs"):
                std_dev, r2, distance_df = calculate_pair_distance_stddev(pair, df_filtered_by_lh, matrix_type, metric)
                if distance_df is not None: # Store result even if std_dev/r2 is NaN, if df exists
                    results_specific.append({'pair': pair, 'std_dev': std_dev, 'r2': r2, 'distance_df': distance_df})
                else:
                    logging.warning(f"Could not process pair {pair}. Skipping.")

            if len(results_specific) != 3:
                logging.warning(f"Expected results for 3 pairs, but only got {len(results_specific)}. Plotting available results.")

            # --- Prepare Data for Plotting --- 
            def format_lh(lh):
                return f"L{lh[0]}H{lh[1]}"

            # Create labels for the specific pairs
            label1 = f"Pair 1 ({format_lh(HeadA)}-{format_lh(HeadB)})"
            label2 = f"Pair 2 ({format_lh(HeadB)}-{format_lh(HeadC)})"
            label3 = f"Pair 3 ({format_lh(HeadC)}-{format_lh(HeadD)})"
            labels = [label1, label2, label3]

            for i, result in enumerate(results_specific):
                 # Remove R2 score (result['r2']) when appending to plot_data
                 plot_data.append((labels[i], result['pair'], result['distance_df']))

        else:
            # --- Sampling and Selection Mode --- 
            logging.info("Proceeding with sampling and std dev based selection...")
            if args.num_pairs_to_sample <= 0:
                logging.error("--num_pairs_to_sample must be positive.")
                exit(1)

            # Generate all possible unique pairs of layer-heads
            all_possible_pairs_comb = list(combinations(available_lh, 2))

            # Filter for only adjacent layer pairs (where layers differ by exactly 1)
            adjacent_layer_pairs = []
            for pair in all_possible_pairs_comb:
                (l1, h1), (l2, h2) = pair
                if abs(l1 - l2) == 1:  # Check if layers are adjacent
                    adjacent_layer_pairs.append(pair)

            all_possible_pairs = adjacent_layer_pairs  # Use only adjacent pairs

            num_total_pairs = len(all_possible_pairs)
            logging.info(f"Total possible adjacent-layer pairs: {num_total_pairs}")

            if num_total_pairs == 0:
                logging.error("No adjacent-layer pairs found. Please check your data or remove the adjacency constraint.")
                exit(1)

            # --- Sample Pairs ---
            N = args.num_pairs_to_sample
            if N >= num_total_pairs:
                logging.info(f"Sampling N={N} >= total adjacent pairs ({num_total_pairs}). Analyzing all adjacent pairs.")
                sampled_pairs = all_possible_pairs
                N = num_total_pairs # Adjust N if analyzing all
            else:
                logging.info(f"Randomly sampling {N} adjacent pairs out of {num_total_pairs} using seed {args.seed}.")
                random.seed(args.seed)
                np.random.seed(args.seed)
                random.shuffle(all_possible_pairs) # Shuffle using the seed
                sampled_pairs = all_possible_pairs[:N]

            # --- Pre-filter DataFrame by Layer-Head for efficiency --- 
            logging.info("Pre-filtering data for involved layer-heads...")
            involved_lh = set(lh for pair in sampled_pairs for lh in pair)
            df_filtered_by_lh = {}
            for lh in tqdm(involved_lh, desc="Filtering Data"):
                 l, h = lh
                 df_filtered_by_lh[lh] = expanded_df[(expanded_df['layer_id'] == l) & (expanded_df['head_id'] == h)].copy()

            # --- Calculate Standard Deviation for Sampled Pairs ---
            logging.info(f"Calculating {metric} distance standard deviation and R2 score for {len(sampled_pairs)} pairs...")
            pair_results = [] # Store dicts of {'pair': pair, 'std_dev': std_dev, 'r2': r2, 'distance_df': distance_df}

            for pair in tqdm(sampled_pairs, desc="Analyzing Pairs"):
                std_dev, r2, distance_df = calculate_pair_distance_stddev(pair, df_filtered_by_lh, matrix_type, metric)
                if pd.notna(std_dev):
                     pair_results.append({'pair': pair, 'std_dev': std_dev, 'r2': r2, 'distance_df': distance_df})
                else:
                     logging.debug(f"Skipping pair {pair} due to NaN standard deviation.")

            logging.info(f"Successfully calculated standard deviation for {len(pair_results)} pairs.")

            if not pair_results:
                logging.error("No valid standard deviations could be calculated for any sampled pair. Cannot proceed.")
                exit(1)

            # --- New Selection Logic (A-B min, B-C max, C-D min) --- 
            logging.info("Selecting pairs based on A-B (min), B-C (max), C-D (min) standard deviation criteria...")

            def order_pair(p):
                lh1, lh2 = p
                return tuple(sorted([lh1, lh2], key=lambda x: (x[0], x[1])))

            for r in pair_results:
                r['ordered_pair'] = order_pair(r['pair'])

            if not pair_results:
                 logging.error("No valid pairs remaining after std dev calculation.")
                 exit(1)

            pair_results.sort(key=lambda x: x['std_dev'])
            min_std_data = pair_results[0]
            A, B = min_std_data['ordered_pair']
            # Log R2 here
            logging.info(f"Found A-B pair with Min Std Dev: {A}-{B} (Std Dev: {min_std_data['std_dev']:.4f}) R2: {min_std_data.get('r2', np.nan):.4f}")

            pairs_involving_B = [r for r in pair_results if B in r['ordered_pair']]
            valid_bc_candidates = [r for r in pairs_involving_B if A not in r['ordered_pair']]

            if not valid_bc_candidates:
                logging.error(f"No other sampled pairs involving head B={B} found (excluding pair with A={A}). Cannot find B-C pair.")
                exit(1)

            max_std_data_b = max(valid_bc_candidates, key=lambda x: x['std_dev'])
            pair_bc_ordered = max_std_data_b['ordered_pair']
            C = pair_bc_ordered[0] if pair_bc_ordered[1] == B else pair_bc_ordered[1]
            if C == B:
                 logging.error("Error determining C. Found C=B.")
                 exit(1)
            # Log R2 here
            logging.info(f"Found B-C pair with Max Std Dev involving B: {B}-{C} (Std Dev: {max_std_data_b['std_dev']:.4f}) R2: {max_std_data_b.get('r2', np.nan):.4f}")

            pairs_involving_C = [r for r in pair_results if C in r['ordered_pair']]
            valid_cd_candidates = [r for r in pairs_involving_C if B not in r['ordered_pair']]

            if not valid_cd_candidates:
                logging.error(f"No other sampled pairs involving head C={C} found (excluding pair with B={B}). Cannot find C-D pair.")
                exit(1)

            min_std_data_c = min(valid_cd_candidates, key=lambda x: x['std_dev'])
            pair_cd_ordered = min_std_data_c['ordered_pair']
            D = pair_cd_ordered[0] if pair_cd_ordered[1] == C else pair_cd_ordered[1]
            if D == C:
                logging.error("Error determining D. Found D=C.")
                exit(1)
            # Log R2 here
            logging.info(f"Found C-D pair with Min Std Dev involving C: {C}-{D} (Std Dev: {min_std_data_c['std_dev']:.4f}) R2: {min_std_data_c.get('r2', np.nan):.4f}")

            # --- Prepare Data for Plotting --- 
            def format_lh(lh):
                 return f"L{lh[0]}H{lh[1]}"

            lhA_str, lhB_str, lhC_str, lhD_str = format_lh(A), format_lh(B), format_lh(C), format_lh(D)

            # Remove R2 score from plot_data tuples
            plot_data.append((f"Min Std Dev ({lhA_str}-{lhB_str})", min_std_data['pair'], min_std_data['distance_df']))
            plot_data.append((f"Max Std Dev ({lhB_str}-{lhC_str})", max_std_data_b['pair'], max_std_data_b['distance_df']))
            plot_data.append((f"Min Std Dev ({lhC_str}-{lhD_str})", min_std_data_c['pair'], min_std_data_c['distance_df']))

            unique_heads = {A, B, C, D}
            logging.info(f"Plotting relationships between {len(unique_heads)} unique heads selected via std dev: {sorted(list(unique_heads), key=lambda x: (x[0], x[1]))}")
            if len(unique_heads) < 4:
                 logging.warning(f"Fewer than 4 unique heads were selected via std dev analysis: {sorted(list(unique_heads), key=lambda x: (x[0], x[1]))}. This might happen if A=C or B=D.")

        # --- END Mode Selection ---

        # --- Plot Comparison --- (Uses plot_data populated by either mode)
        if not plot_data:
             logging.error("No data available for plotting. Exiting.")
             exit(1)

        logging.info("Generating comparison plot...")
        plot_comparison_distances(plot_data, args.output, matrix_type_name, metric)

        # --- Plot CDF Comparison --- 
        logging.info("Generating comparison CDF plot...")
        plot_comparison_cdf(plot_data, args.output, matrix_type_name, metric)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        exit(1)
    except ValueError as e:
        logging.error(f"Error: {e}")
        exit(1)
    except KeyError as e:
         logging.error(f"Configuration or data structure error: Missing key {e}")
         exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())
        exit(1)

    logging.info(f"Script finished. Total execution time: {time.time() - start_run_time:.2f} seconds")

# --- Removed old plot_distances function ---
# def plot_distances(...):
#    ...
