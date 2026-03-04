import pickle
import pandas as pd
import numpy as np
import time
import argparse
import importlib.util
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
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
        if 'v_cache' in file_path.lower():
            config['matrix_type'] = 'v_matrix'
            logging.info(f"Automatically setting matrix_type to 'v_matrix' based on file name: {file_path}")
        else:
            config['matrix_type'] = 'k_matrix'
            logging.info(f"Defaulting to matrix_type 'k_matrix'")

    return config


# --- Data Loading and Processing (Adapted from trainer.py) ---
def load_data(file_path, matrix_type='k_matrix'):
    """Load and perform initial processing on the data from pickle file."""
    loaded_data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return loaded_data

    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    # Load each object (which is a list of records dumped previously)
                    data_chunk = pickle.load(f)
                    if isinstance(data_chunk, list):
                         loaded_data.extend(data_chunk)
                    else:
                         # Handle potential older format or unexpected data
                         print(f"Warning: loaded unexpected data type: {type(data_chunk)}")
                         loaded_data.append(data_chunk)
                except EOFError:
                    # End of file reached
                    break
    except Exception as e:
        print(f"Error loading KV cache data from {file_path}: {e}")


    logging.info(f"Type of loaded data: {type(loaded_data)}")
    logging.info(f"Using matrix type: {matrix_type}")

    # Convert to DataFrame
    try:
        df = pd.DataFrame(loaded_data)
        logging.info("Successfully converted loaded data to DataFrame.")
    except ValueError as e:
         logging.error(f"Failed to convert loaded data to DataFrame: {e}")
         logging.error("Ensure the pickle file contains data suitable for DataFrame creation (e.g., list of dicts).")
         raise

    # Basic renaming and checks
    if 'batch_id' in df.columns:
        df = df.rename(columns={'batch_id': 'sample_number'})
    if 'request_id' in df.columns:
        df = df.drop(columns=['request_id'])

    if matrix_type not in df.columns:
        raise ValueError(f"Error: Matrix column '{matrix_type}' not found. Available columns: {df.columns.tolist()}")

    # --- Ensure matrix precision is at least float32 ---
    # Check the specified matrix_type column and also the other potential one
    potential_matrix_cols = [col for col in ['k_matrix', 'v_matrix'] if col in df.columns]
    if not potential_matrix_cols:
         logging.warning("Could not find 'k_matrix' or 'v_matrix' columns after loading.")
    # else:
    #     for matrix_col in potential_matrix_cols:
    #         logging.info(f"Checking precision for column: {matrix_col}")
    #         # Iterate and check dtype, raise error if insufficient precision found
    #         precision_ok = True
    #         for i, matrix in enumerate(df[matrix_col]):
    #             try:
    #                 # Check if it's a numpy array and if its dtype is lower precision than float32
    #                 if isinstance(matrix, np.ndarray) and np.issubdtype(matrix.dtype, np.floating) and matrix.dtype < np.float32:
    #                     logging.warning(f"Warning: Matrix at index {i} in column '{matrix_col}' has dtype {matrix.dtype}, which is less than float32. Continuing processing, but this may lead to precision issues.")
    #                     # precision_ok = False # No longer stopping execution
    #                     # break # Continue checking other matrices in the column
    #             except Exception as e:
    #                 logging.warning(f"Could not check dtype for matrix at index {i} in column '{matrix_col}': {e}")
    #                 # Optionally, you might want to treat this as an error depending on requirements
    #                 # precision_ok = False
    #                 # break

    #         # if not precision_ok: # Removed the error raising block
    #         #     raise ValueError(f"Insufficient precision detected in column '{matrix_col}'. All matrices must have dtype float32 or higher.")
    #         # else:
    #         #     logging.info(f"Precision check passed for column '{matrix_col}'. All matrices are float32 or higher.") # Keep this or adjust wording if needed
    #         logging.info(f"Precision check completed for column '{matrix_col}'.")

    return df, matrix_type

def process_prefill_data(prefill_df, matrix_type):
    """Process prefill data by expanding matrices into single tokens."""
    expanded_prefill_rows = []
    if not prefill_df.empty:
        for _, row in prefill_df.iterrows():
            matrix = row[matrix_type]
            num_prefill_tokens = row['num_tokens_in_matrix']
            base_info = row.drop([matrix_type, 'num_tokens_in_matrix', 'matrix_shape'])
            for token_idx in range(num_prefill_tokens):
                # Ensure slicing maintains dimensions for consistency, aiming for (1, 1, embed_dim)
                # The original shape might be (channels, tokens, embed_dim)
                # We assume channels=1 here based on typical transformer activations
                token_matrix = matrix[:, token_idx:token_idx+1, :]
                row_data = base_info.to_dict()
                row_data[matrix_type] = token_matrix
                row_data['token_position'] = token_idx
                row_data['prefill'] = True
                expanded_prefill_rows.append(row_data)
        expanded_prefill_df = pd.DataFrame(expanded_prefill_rows)
        sample_n_map = prefill_df.loc[prefill_df['num_decode_tokens'] == 0].groupby('sample_number')['num_tokens_in_matrix'].first()
        sample_n_map = sample_n_map.fillna(0).astype(int)
    else:
        # Define necessary columns even if empty
        base_cols = list(prefill_df.columns.drop([matrix_type, 'num_tokens_in_matrix', 'matrix_shape']))
        final_cols = base_cols + [matrix_type, 'token_position', 'prefill']
        expanded_prefill_df = pd.DataFrame(columns=final_cols)
        sample_n_map = pd.Series(dtype=int)

    return expanded_prefill_df, sample_n_map

def process_decode_data(decode_df, sample_n_map, matrix_type):
    """Process decode data by calculating token positions."""
    if not decode_df.empty:
        decode_df['N'] = decode_df['sample_number'].map(sample_n_map).fillna(0).astype(int)
        decode_df['token_position'] = (decode_df['num_decode_tokens'] - 1) + decode_df['N']
        decode_df['prefill'] = False
        decode_df = decode_df.drop(columns=['N', 'num_tokens_in_matrix', 'matrix_shape'], errors='ignore')
    return decode_df

def process_data(df, matrix_type):
    """Main function to process the DataFrame and expand matrices (adapted from trainer.py)."""
    start_time = time.time()
    logging.info("Starting data processing to expand matrices...")
    df_processed = df.copy()

    # Check for necessary columns
    required_cols = [matrix_type, 'num_decode_tokens', 'sample_number']
    if not all(col in df_processed.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df_processed.columns]
         raise ValueError(f"Missing required columns for processing: {missing}")

    # Calculate num_tokens_in_matrix, handle potential errors if matrix isn't shaped as expected
    try:
        df_processed['matrix_shape'] = df_processed[matrix_type].apply(lambda x: x.shape if hasattr(x, 'shape') else None)
        df_processed['num_tokens_in_matrix'] = df_processed[matrix_type].apply(lambda x: x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else 0)
    except Exception as e:
        logging.error(f"Error determining matrix shapes or tokens: {e}")
        logging.error("Ensure the '{matrix_type}' column contains NumPy arrays with expected dimensions.")
        raise

    is_prefill = df_processed['num_tokens_in_matrix'] > 1
    prefill_df = df_processed[is_prefill].copy()
    decode_df = df_processed[~is_prefill].copy()
    logging.info(f"Identified {len(prefill_df)} prefill rows and {len(decode_df)} decode rows.")

    expanded_prefill_df, sample_n_map = process_prefill_data(prefill_df, matrix_type)
    decode_df = process_decode_data(decode_df, sample_n_map, matrix_type)

    # Combine results, ensuring columns align
    # Identify common essential columns expected after processing
    common_columns = ['sample_number', 'layer_id', 'head_id', matrix_type, 'prefill', 'token_position']
    # Add 'num_decode_tokens' if it exists in decode_df, useful for context
    if 'num_decode_tokens' in decode_df.columns:
        if 'num_decode_tokens' not in common_columns: common_columns.insert(1, 'num_decode_tokens')
    elif 'num_decode_tokens' in prefill_df.columns:
         if 'num_decode_tokens' not in common_columns: common_columns.insert(1, 'num_decode_tokens')

    # Ensure both DFs have the common columns before concat
    for col in common_columns:
        if col not in expanded_prefill_df.columns:
             expanded_prefill_df[col] = pd.NA # Or appropriate default
        if col not in decode_df.columns:
             decode_df[col] = pd.NA

    # Select only common columns in the correct order
    expanded_prefill_df = expanded_prefill_df[common_columns]
    decode_df = decode_df[common_columns]

    expanded_df = pd.concat([expanded_prefill_df, decode_df], ignore_index=True)
    expanded_df = expanded_df.sort_values(by=['sample_number', 'layer_id', 'head_id', 'token_position'], ignore_index=True)

    logging.info(f"Data processing time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Original DF shape: {df.shape}")
    logging.info(f"Expanded DF shape (token level): {expanded_df.shape}")
    logging.info(f"Columns in expanded DF: {expanded_df.columns.tolist()}")

    # Add a check for activation dim consistency if possible
    try:
        first_matrix = expanded_df[matrix_type].iloc[0]
        activation_dim = first_matrix.shape[-1]
        logging.info(f"Inferred activation dimension: {activation_dim}")
    except (IndexError, AttributeError, TypeError):
        logging.warning("Could not infer activation dimension from the first matrix.")
        activation_dim = -1 # Indicate unknown

    return expanded_df, activation_dim

# --- NEW: Instruction Token Filtering ---
def filter_instruction_tokens(expanded_df, matrix_type, keep_instruction_samples, min_samples_for_detection=3):
    """Identifies and filters initial identical prefill tokens for each layer/head.

    Args:
        expanded_df (pd.DataFrame): DataFrame with token-level data (output of process_data).
        matrix_type (str): Name of the column containing activation matrices.
        keep_instruction_samples (int): Number of initial samples per layer/head for which
                                         to keep the instruction tokens.
        min_samples_for_detection (int): Minimum number of samples required to detect instructions.

    Returns:
        pd.DataFrame: DataFrame with instruction tokens potentially filtered.
    """
    start_time = time.time()
    logging.info(f"Starting instruction token filtering (keeping first {keep_instruction_samples} samples)...")

    if keep_instruction_samples < 0:
        logging.warning("`keep_instruction_samples` is negative, keeping instructions for all samples.")
        return expanded_df

    prefill_mask = expanded_df['prefill']
    if not prefill_mask.any():
        logging.info("No prefill tokens found, skipping instruction filtering.")
        return expanded_df

    prefill_part = expanded_df[prefill_mask].copy()
    decode_part = expanded_df[~prefill_mask].copy()

    # Ensure correct data types for comparison and indexing
    prefill_part['layer_id'] = prefill_part['layer_id'].astype(int)
    prefill_part['head_id'] = prefill_part['head_id'].astype(int)
    prefill_part['token_position'] = prefill_part['token_position'].astype(int)
    prefill_part['sample_number'] = prefill_part['sample_number'].astype(int)

    filtered_prefill_rows = []
    total_filtered_tokens = 0

    # Group by layer and head
    grouped = prefill_part.groupby(['layer_id', 'head_id'])

    for (layer_id, head_id), group_df in grouped:
        unique_samples = sorted(group_df['sample_number'].unique())
        num_samples = len(unique_samples)

        # If fewer samples than needed for detection, or if keeping all samples anyway
        if num_samples < min_samples_for_detection or keep_instruction_samples >= num_samples:
            filtered_prefill_rows.extend(group_df.to_dict('records'))
            continue # Keep all data for this group

        # Pivot to easily compare tokens across samples
        try:
            # Ensure index is unique before pivoting
            group_df = group_df.drop_duplicates(subset=['sample_number', 'token_position'])
            pivot_df = group_df.pivot(index='token_position', columns='sample_number', values=matrix_type)
        except ValueError as e:
             logging.warning(f"Could not pivot data for L{layer_id}H{head_id} due to duplicates or missing values: {e}. Skipping instruction detection for this group.")
             filtered_prefill_rows.extend(group_df.to_dict('records'))
             continue
        except KeyError:
            logging.warning(f"KeyError during pivot for L{layer_id}H{head_id}. Skipping instruction detection.")
            filtered_prefill_rows.extend(group_df.to_dict('records'))
            continue


        max_token_pos = pivot_df.index.max()
        num_instruction_tokens = 0

        # Iterate through token positions to find the common prefix length
        for token_pos in range(max_token_pos + 1):
            if token_pos not in pivot_df.index:
                break # Token position missing, end of common sequence

            # Get activation vectors for this token position across available samples
            token_vectors = pivot_df.loc[token_pos].dropna().tolist()

            if len(token_vectors) < min_samples_for_detection:
                # Not enough samples have this token position to make a reliable decision
                break

            # Use the first vector as reference
            ref_vector = token_vectors[0]
            if not isinstance(ref_vector, np.ndarray): # Ensure it's a numpy array
                 logging.warning(f"Non-numpy array encountered at L{layer_id}H{head_id}, token {token_pos}, sample {pivot_df.columns[pivot_df.loc[token_pos].dropna().index[0]]}. Type: {type(ref_vector)}. Skipping further checks for this token.")
                 break

            # Check if at least min_samples_for_detection vectors are close to the reference
            similar_count = 1 # Start with the reference itself
            for vec in token_vectors[1:]:
                 if isinstance(vec, np.ndarray) and vec.shape == ref_vector.shape:
                     try:
                         if np.allclose(ref_vector, vec):
                             similar_count += 1
                     except Exception as e:
                          logging.warning(f"Error during np.allclose for L{layer_id}H{head_id}, token {token_pos}: {e}")
                          # Treat as not similar
                 # else: # Log type/shape mismatch if needed
                 #    logging.warning(f"Type or shape mismatch at L{layer_id}H{head}, token {token_pos}. Ref: {ref_vector.shape}, Current: {getattr(vec, 'shape', type(vec))}")


            if similar_count >= min_samples_for_detection:
                num_instruction_tokens += 1
            else:
                # The sequence of identical tokens ended
                break

        if num_instruction_tokens > 0:
            logging.info(f"Detected {num_instruction_tokens} instruction tokens for L{layer_id}H{head_id}.")

        # Filter the group based on detected instructions
        samples_to_keep_instructions = set(unique_samples[:keep_instruction_samples])
        for _, row_series in group_df.iterrows():
            row = row_series.to_dict()
            sample_num = row['sample_number']
            token_pos = row['token_position']

            if sample_num in samples_to_keep_instructions:
                filtered_prefill_rows.append(row)
            elif token_pos >= num_instruction_tokens:
                # Adjust token position for samples where instructions are removed
                row['token_position'] = token_pos - num_instruction_tokens
                filtered_prefill_rows.append(row)
                total_filtered_tokens += 1 # Count how many tokens were effectively removed *per sample* beyond the keep_instruction_samples threshold

    if not filtered_prefill_rows:
         logging.warning("Prefill filtering resulted in an empty prefill part.")
         filtered_prefill_df = pd.DataFrame(columns=prefill_part.columns)
    else:
        filtered_prefill_df = pd.DataFrame(filtered_prefill_rows)

    logging.info(f"Instruction filtering removed/shifted {total_filtered_tokens} token entries across samples.")
    logging.info(f"Instruction filtering time: {time.time() - start_time:.2f} seconds")

    # Combine filtered prefill with original decode
    final_df = pd.concat([filtered_prefill_df, decode_part], ignore_index=True)
    final_df = final_df.sort_values(by=['sample_number', 'layer_id', 'head_id', 'token_position'], ignore_index=True)

    return final_df

# --- Aggregation into 4D NumPy Array ---
def aggregate_to_4d(expanded_df, matrix_type, activation_dim_inferred):
    """Aggregate the token-level data into a 4D NumPy array.

    Args:
        expanded_df: DataFrame with one row per sample, layer, head, token.
        matrix_type: The name of the column containing activation matrices.
        activation_dim_inferred: The embedding dimension size inferred earlier.

    Returns:
        numpy.ndarray: A 4D array (num_sample_tokens, num_layers, num_heads, activation_dim).
    """
    start_time = time.time()
    logging.info("Starting aggregation into 4D NumPy array...")

    if expanded_df.empty:
        logging.error("Expanded DataFrame is empty. Cannot aggregate.")
        return None

    # Drop rows where essential identifiers might be missing
    expanded_df = expanded_df.dropna(subset=['sample_number', 'token_position', 'layer_id', 'head_id', matrix_type])
    if expanded_df.empty:
        logging.error("DataFrame is empty after dropping rows with missing identifiers. Cannot aggregate.")
        return None

    # Determine dimensions
    num_layers = int(expanded_df['layer_id'].max() + 1)
    num_heads = int(expanded_df['head_id'].max() + 1)

    # Try to get activation_dim from the first valid matrix in the specified column
    activation_dim = -1
    for matrix_entry in expanded_df[matrix_type]:
        if hasattr(matrix_entry, 'shape') and len(matrix_entry.shape) >= 3:
            activation_dim = matrix_entry.shape[-1]
            break

    if activation_dim == -1:
        if activation_dim_inferred > 0:
            activation_dim = activation_dim_inferred
            logging.warning(f"Could not confirm activation dimension during aggregation. Using inferred value: {activation_dim}")
        else:
             raise ValueError("Could not determine activation dimension from matrices in the DataFrame.")
    else:
        logging.info(f"Confirmed activation dimension: {activation_dim}")

    # Create unique identifiers for the first dimension (sample-token pairs)
    sample_token_pairs = expanded_df[['sample_number', 'token_position']].drop_duplicates().sort_values(by=['sample_number', 'token_position'])
    num_sample_tokens = len(sample_token_pairs)
    sample_token_to_idx = { (row.sample_number, row.token_position): i for i, row in enumerate(sample_token_pairs.itertuples(index=False)) }

    logging.info(f"Dimensions for 4D array: Samples/Tokens={num_sample_tokens}, Layers={num_layers}, Heads={num_heads}, ActivationDim={activation_dim}")

    # Initialize the 4D array
    activations_4d = np.zeros((num_sample_tokens, num_layers, num_heads, activation_dim), dtype=np.float32)
    logging.info(f"Initialized 4D array with shape {activations_4d.shape} and type {activations_4d.dtype}")

    # Populate the array
    processed_count = 0
    skipped_count = 0
    for _, row in expanded_df.iterrows():
        sample = row['sample_number']
        token_pos = row['token_position']
        layer = int(row['layer_id'])
        head = int(row['head_id'])
        matrix = row[matrix_type]

        # Get the index for the first dimension
        idx = sample_token_to_idx.get((sample, token_pos))

        if idx is None:
            # This shouldn't happen if sample_token_to_idx was built correctly
            skipped_count += 1
            continue

        # Extract the activation vector
        try:
            # The matrix should be ~ (1, 1, activation_dim) after expansion
            activation_vector = matrix.squeeze()
            if activation_vector.shape != (activation_dim,):
                 # Handle cases where squeeze might result in unexpected shapes
                 # e.g. if original matrix wasn't (1, 1, dim)
                 if activation_vector.size == activation_dim:
                     activation_vector = activation_vector.reshape(activation_dim)
                 else:
                      logging.warning(f"Skipping entry due to unexpected activation vector shape: {activation_vector.shape} for L{layer}H{head}, sample {sample}, token {token_pos}. Expected ({activation_dim},). Matrix shape was {matrix.shape}")
                      skipped_count += 1
                      continue
        except Exception as e:
            logging.warning(f"Skipping entry due to error extracting activation vector: {e} for L{layer}H{head}, sample {sample}, token {token_pos}. Matrix shape was {getattr(matrix, 'shape', 'N/A')}")
            skipped_count += 1
            continue

        # Assign to the 4D array
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            activations_4d[idx, layer, head, :] = activation_vector
            processed_count += 1
        else:
             logging.warning(f"Skipping entry due to out-of-bounds layer/head index: L{layer}, H{head}")
             skipped_count += 1

    logging.info(f"Aggregation complete. Processed {processed_count} entries, skipped {skipped_count} entries.")
    logging.info(f"Aggregation time: {time.time() - start_time:.2f} seconds")

    if processed_count == 0:
        logging.error("No valid entries were processed during aggregation. Returning None.")
        return None

    return activations_4d

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess activation data from DataFrame format to a 4D NumPy array.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input pickle file containing activation data.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output 4D NumPy array pickle file (e.g., activations_4d.pkl).")
    parser.add_argument("--matrix_type", type=str, default="k_matrix", choices=["k_matrix", "v_matrix", "q_matrix"],
                        help="Type of matrix to process (k_matrix or v_matrix).")
    parser.add_argument("--keep_instruction_samples", type=int, default=None,
                        help="Number of initial samples per layer/head for which to keep instruction tokens.")
    parser.add_argument("--exclude_prefill", action='store_true',
                        help="Exclude all prefill tokens from the output array.")

    args = parser.parse_args()

    try:
        # 1. Use direct file input instead of loading from config
        input_file = args.input
        matrix_type = args.matrix_type
        keep_instruction_samples = args.keep_instruction_samples
        logging.info(f"Input data file: {input_file}")
        logging.info(f"Matrix type: {matrix_type}")
        if keep_instruction_samples is not None:
            logging.info(f"Keeping instruction tokens for first {keep_instruction_samples} samples per layer/head.")
        else:
            logging.info("No instruction token filtering will be applied.")

        # 2. Load Data into DataFrame
        df_original, _ = load_data(input_file, matrix_type)

        # 3. Process Data (Expand Tokens)
        expanded_df, activation_dim = process_data(df_original, matrix_type)

        # --- NEW Step 3.5: Filter Instruction Tokens ---
        if keep_instruction_samples is not None: # Allow skipping if not defined or explicitly None
             expanded_df = filter_instruction_tokens(expanded_df, matrix_type, keep_instruction_samples)
             logging.info(f"Shape after instruction filtering: {expanded_df.shape}")
        else:
             logging.info("Skipping instruction token filtering as `keep_instruction_samples` is not set.")

        # --- NEW Step 3.6: Filter Prefill Tokens (if requested) ---
        if args.exclude_prefill:
            logging.info("Excluding all prefill tokens as requested by the --exclude_prefill flag.")
            initial_rows = len(expanded_df)
            expanded_df = expanded_df[expanded_df['prefill'] == False].copy()
            logging.info(f"Removed {initial_rows - len(expanded_df)} prefill token rows. Remaining rows: {len(expanded_df)}")
            if expanded_df.empty:
                logging.error("DataFrame is empty after removing prefill tokens. Cannot proceed.")
                exit(1)
        else:
            logging.info("Including prefill tokens in the output.")

        # 4. Aggregate into 4D Array
        activations_4d = aggregate_to_4d(expanded_df, matrix_type, activation_dim)

        # 5. Save Output
        if activations_4d is not None:
            logging.info(f"Saving 4D NumPy array to {args.output}...")
            with open(args.output, 'wb') as f_out:
                pickle.dump(activations_4d, f_out)
            logging.info("Successfully saved the 4D array.")
        else:
            logging.error("Aggregation resulted in None. Output file was not saved.")
            exit(1)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        exit(1)
    except ValueError as e:
        logging.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        # Consider adding traceback logging here for debugging
        # import traceback
        # logging.error(traceback.format_exc())
        exit(1)

    logging.info("Preprocessing finished.") 