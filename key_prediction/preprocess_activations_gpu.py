import pickle
import pandas as pd
import cupy as cp  # Use CuPy instead of NumPy
import time
import argparse
import importlib.util
import os
import logging
import numpy as np

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

    # Default precision if not specified
    if 'precision' not in config:
        config['precision'] = 'float64' # Default to float64 for GPU
        logging.info(f"Defaulting GPU precision to 'float64'")

    return config

# --- Data Loading and Processing (Adapted for GPU) ---
def load_data_cpu(file_path):
    """Load data from pickle file using Pandas (runs on CPU)."""
    logging.info(f"Loading data from {file_path} using Pandas (CPU)...")
    with open(file_path, 'rb') as file:
        # Load with pickle, assuming it contains data pandas can handle
        loaded_data = pickle.load(file)

    logging.info(f"Type of loaded data: {type(loaded_data)}")

    # Convert to DataFrame on CPU first
    try:
        df = pd.DataFrame(loaded_data)
        logging.info("Successfully converted loaded data to Pandas DataFrame on CPU.")
    except ValueError as e:
         logging.error(f"Failed to convert loaded data to DataFrame: {e}")
         logging.error("Ensure the pickle file contains data suitable for DataFrame creation (e.g., list of dicts).")
         raise

    # Basic renaming and checks (still on CPU)
    if 'batch_id' in df.columns:
        df = df.rename(columns={'batch_id': 'sample_number'})
    if 'request_id' in df.columns:
        df = df.drop(columns=['request_id'])

    # --- Ensure matrix precision is at least float32 ---
    # Check the specified matrix_type column and also the other potential one
    potential_matrix_cols = [col for col in ['k_matrix', 'v_matrix'] if col in df.columns]
    if not potential_matrix_cols:
         logging.warning("Could not find 'k_matrix' or 'v_matrix' columns after loading.")
    else:
        for matrix_col in potential_matrix_cols:
            logging.info(f"Checking precision for column: {matrix_col}")
            # Iterate and check dtype, raise error if insufficient precision found
            precision_ok = True
            for i, matrix in enumerate(df[matrix_col]):
                try:
                    # Check if it's a numpy array and if its dtype is lower precision than float32
                    if isinstance(matrix, np.ndarray) and np.issubdtype(matrix.dtype, np.floating) and matrix.dtype < np.float32:
                        logging.error(f"Error: Matrix at index {i} in column '{matrix_col}' has dtype {matrix.dtype}, which is less than float32.")
                        precision_ok = False
                        break # Stop checking this column once an issue is found
                except Exception as e:
                    logging.warning(f"Could not check dtype for matrix at index {i} in column '{matrix_col}': {e}")
                    # Optionally, you might want to treat this as an error depending on requirements
                    # precision_ok = False
                    # break

            if not precision_ok:
                raise ValueError(f"Insufficient precision detected in column '{matrix_col}'. All matrices must have dtype float32 or higher.")
            else:
                logging.info(f"Precision check passed for column '{matrix_col}'. All matrices are float32 or higher.")

    return df

def process_prefill_data_cpu(prefill_df, matrix_type):
    """Process prefill data on CPU by expanding matrices into single tokens."""
    expanded_prefill_rows = []
    if not prefill_df.empty:
        for _, row in prefill_df.iterrows():
            matrix = row[matrix_type] # Keep as NumPy array for now
            num_prefill_tokens = row['num_tokens_in_matrix']
            base_info = row.drop([matrix_type, 'num_tokens_in_matrix', 'matrix_shape'])
            for token_idx in range(num_prefill_tokens):
                token_matrix = matrix[:, token_idx:token_idx+1, :] # Slice NumPy array
                row_data = base_info.to_dict()
                row_data[matrix_type] = token_matrix # Store NumPy slice
                row_data['token_position'] = token_idx
                row_data['prefill'] = True
                expanded_prefill_rows.append(row_data)
        expanded_prefill_df = pd.DataFrame(expanded_prefill_rows)
        # Calculate N map on CPU
        sample_n_map = prefill_df.loc[prefill_df['num_decode_tokens'] == 0].groupby('sample_number')['num_tokens_in_matrix'].first()
        sample_n_map = sample_n_map.fillna(0).astype(int)
    else:
        # Define necessary columns even if empty
        base_cols = list(prefill_df.columns.drop([matrix_type, 'num_tokens_in_matrix', 'matrix_shape'], errors='ignore'))
        final_cols = base_cols + [matrix_type, 'token_position', 'prefill']
        expanded_prefill_df = pd.DataFrame(columns=final_cols)
        sample_n_map = pd.Series(dtype=int)

    return expanded_prefill_df, sample_n_map

def process_decode_data_cpu(decode_df, sample_n_map, matrix_type):
    """Process decode data on CPU by calculating token positions."""
    if not decode_df.empty:
        decode_df['N'] = decode_df['sample_number'].map(sample_n_map).fillna(0).astype(int)
        decode_df['token_position'] = (decode_df['num_decode_tokens'] - 1) + decode_df['N']
        decode_df['prefill'] = False
        # Drop columns on CPU DF
        decode_df = decode_df.drop(columns=['N', 'num_tokens_in_matrix', 'matrix_shape'], errors='ignore')
         # Keep matrix column as NumPy arrays for now
    return decode_df

def process_data_cpu(df, matrix_type):
    """Main function to process the DataFrame on CPU and expand matrices."""
    start_time = time.time()
    logging.info("Starting data processing on CPU to expand matrices...")
    df_processed = df.copy() # Operate on CPU copy

    # Check for necessary columns
    required_cols = [matrix_type, 'num_decode_tokens', 'sample_number']
    if not all(col in df_processed.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df_processed.columns]
         raise ValueError(f"Missing required columns for processing: {missing}")

    # Calculate num_tokens_in_matrix on CPU
    try:
        # Assume matrices are NumPy arrays at this stage
        df_processed['matrix_shape'] = df_processed[matrix_type].apply(lambda x: x.shape if hasattr(x, 'shape') else None)
        df_processed['num_tokens_in_matrix'] = df_processed[matrix_type].apply(lambda x: x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else 0)
    except Exception as e:
        logging.error(f"Error determining matrix shapes or tokens: {e}")
        logging.error("Ensure the '{matrix_type}' column contains NumPy arrays with expected dimensions.")
        raise

    is_prefill = df_processed['num_tokens_in_matrix'] > 1
    prefill_df = df_processed[is_prefill].copy()
    decode_df = df_processed[~is_prefill].copy()
    logging.info(f"Identified {len(prefill_df)} prefill rows and {len(decode_df)} decode rows on CPU.")

    expanded_prefill_df, sample_n_map = process_prefill_data_cpu(prefill_df, matrix_type)
    decode_df = process_decode_data_cpu(decode_df, sample_n_map, matrix_type)

    # Combine results on CPU
    common_columns = ['sample_number', 'layer_id', 'head_id', matrix_type, 'prefill', 'token_position']
    if 'num_decode_tokens' in decode_df.columns:
        if 'num_decode_tokens' not in common_columns: common_columns.insert(1, 'num_decode_tokens')
    elif 'num_decode_tokens' in prefill_df.columns:
         if 'num_decode_tokens' not in common_columns: common_columns.insert(1, 'num_decode_tokens')

    for col in common_columns:
        if col not in expanded_prefill_df.columns:
             expanded_prefill_df[col] = pd.NA # Or appropriate default
        if col not in decode_df.columns:
             decode_df[col] = pd.NA

    expanded_prefill_df = expanded_prefill_df[common_columns]
    decode_df = decode_df[common_columns]

    expanded_df_cpu = pd.concat([expanded_prefill_df, decode_df], ignore_index=True)
    expanded_df_cpu = expanded_df_cpu.sort_values(by=['sample_number', 'layer_id', 'head_id', 'token_position'], ignore_index=True)

    logging.info(f"CPU Data processing time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Original DF shape: {df.shape}")
    logging.info(f"Expanded DF shape (token level, CPU): {expanded_df_cpu.shape}")
    logging.info(f"Columns in expanded CPU DF: {expanded_df_cpu.columns.tolist()}")

    # Infer activation dim from CPU data before potentially moving to GPU
    activation_dim = -1
    try:
        # Find first valid matrix (still NumPy)
        valid_matrices = expanded_df_cpu[matrix_type].dropna()
        if not valid_matrices.empty:
            first_matrix = valid_matrices.iloc[0]
            if hasattr(first_matrix, 'shape') and len(first_matrix.shape) >= 3:
                 activation_dim = first_matrix.shape[-1]
                 logging.info(f"Inferred activation dimension from CPU data: {activation_dim}")
            else:
                 logging.warning(f"First matrix has unexpected shape: {getattr(first_matrix, 'shape', 'N/A')}")
        else:
            logging.warning("No valid matrices found in the CPU DataFrame to infer dimension.")

    except Exception as e:
        logging.warning(f"Could not infer activation dimension from CPU data: {e}")

    if activation_dim <= 0:
         logging.error("Failed to determine activation dimension. Cannot proceed.")
         raise ValueError("Activation dimension could not be determined.")


    return expanded_df_cpu, activation_dim


# --- Aggregation into 4D CuPy Array ---
def aggregate_to_4d_gpu(expanded_df_cpu, matrix_type, activation_dim, precision='float64'):
    """Aggregate the token-level data from CPU DataFrame into a 4D CuPy array.

    Args:
        expanded_df_cpu: Pandas DataFrame with one row per sample, layer, head, token.
                         Matrix column contains NumPy arrays.
        matrix_type: The name of the column containing activation matrices.
        activation_dim: The embedding dimension size (already inferred).
        precision: Data type for the GPU array ('float32' or 'float64').

    Returns:
        cupy.ndarray: A 4D array (num_sample_tokens, num_layers, num_heads, activation_dim) on GPU.
    """
    start_time = time.time()
    logging.info("Starting aggregation into 4D CuPy array...")

    if expanded_df_cpu.empty:
        logging.error("Expanded CPU DataFrame is empty. Cannot aggregate.")
        return None

    # Drop rows with missing identifiers (still on CPU DF)
    expanded_df_cpu = expanded_df_cpu.dropna(subset=['sample_number', 'token_position', 'layer_id', 'head_id', matrix_type])
    if expanded_df_cpu.empty:
        logging.error("CPU DataFrame is empty after dropping rows with missing identifiers. Cannot aggregate.")
        return None

    # Determine dimensions from CPU data
    num_layers = int(expanded_df_cpu['layer_id'].max() + 1)
    num_heads = int(expanded_df_cpu['head_id'].max() + 1)

    # Create unique identifiers for the first dimension (sample-token pairs) from CPU data
    sample_token_pairs = expanded_df_cpu[['sample_number', 'token_position']].drop_duplicates().sort_values(by=['sample_number', 'token_position'])
    num_sample_tokens = len(sample_token_pairs)
    # Build map on CPU for faster lookup
    sample_token_to_idx = { (row.sample_number, row.token_position): i for i, row in enumerate(sample_token_pairs.itertuples(index=False)) }

    logging.info(f"Dimensions for 4D CuPy array: Samples/Tokens={num_sample_tokens}, Layers={num_layers}, Heads={num_heads}, ActivationDim={activation_dim}")

    # Choose CuPy dtype
    dtype = cp.float64 if precision == 'float64' else cp.float32
    logging.info(f"Initializing 4D CuPy array with shape {(num_sample_tokens, num_layers, num_heads, activation_dim)} and type {dtype}")

    # Initialize the 4D array ON THE GPU
    activations_4d_gpu = cp.zeros((num_sample_tokens, num_layers, num_heads, activation_dim), dtype=dtype)

    # Populate the GPU array by iterating through CPU DataFrame
    # This involves transferring each vector individually, which might be slow.
    # A potentially faster (but more complex) approach would involve batch transfers.
    processed_count = 0
    skipped_count = 0
    transfer_errors = 0
    shape_errors = 0

    logging.info("Populating 4D CuPy array (transferring vectors individually)...")
    # Using itertuples for potentially faster iteration than iterrows
    for row in expanded_df_cpu.itertuples(index=False):
        sample = row.sample_number
        token_pos = row.token_position
        layer = int(row.layer_id)
        head = int(row.head_id)
        matrix_np = getattr(row, matrix_type) # Matrix is still NumPy here

        # Get the index for the first dimension (using CPU map)
        idx = sample_token_to_idx.get((sample, token_pos))

        if idx is None:
            skipped_count += 1
            continue

        # Extract, reshape, and transfer the activation vector
        try:
            # Ensure matrix_np is a valid NumPy array before processing
            if not isinstance(matrix_np, np.ndarray):
                 # logging.warning(f"Skipping entry: matrix is not a NumPy array for L{layer}H{head}, sample {sample}, token {token_pos}. Type: {type(matrix_np)}")
                 skipped_count += 1
                 continue

            activation_vector_np = matrix_np.squeeze()

            if activation_vector_np.shape != (activation_dim,):
                 if activation_vector_np.size == activation_dim:
                     activation_vector_np = activation_vector_np.reshape(activation_dim)
                 else:
                      # logging.warning(f"Skipping entry due to unexpected activation vector shape: {activation_vector_np.shape} for L{layer}H{head}, sample {sample}, token {token_pos}. Expected ({activation_dim},). Original matrix shape: {matrix_np.shape}")
                      shape_errors += 1
                      skipped_count += 1
                      continue

            # Transfer the NumPy vector to CuPy GPU vector
            activation_vector_gpu = cp.asarray(activation_vector_np, dtype=dtype)

            # Assign to the 4D GPU array
            if 0 <= layer < num_layers and 0 <= head < num_heads:
                activations_4d_gpu[idx, layer, head, :] = activation_vector_gpu
                processed_count += 1
            else:
                 # logging.warning(f"Skipping entry due to out-of-bounds layer/head index: L{layer}, H{head}")
                 skipped_count += 1

        except cp.cuda.memory.OutOfMemoryError:
            logging.error("GPU Out of Memory during vector transfer/assignment. Aborting.")
            return None # Or raise error
        except Exception as e:
            # Log less verbosely during loop
            # logging.warning(f"Skipping entry due to error extracting/transferring vector: {e} for L{layer}H{head}, sample {sample}, token {token_pos}. Matrix shape was {getattr(matrix_np, 'shape', 'N/A')}")
            transfer_errors += 1
            skipped_count += 1
            continue

    # Log summary after loop
    logging.info(f"Aggregation complete. Processed {processed_count} entries.")
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} entries (Shape errors: {shape_errors}, Transfer/other errors: {transfer_errors}, Index/Type issues: {skipped_count - shape_errors - transfer_errors}). Check warnings above for details.")
    logging.info(f"Aggregation and GPU transfer time: {time.time() - start_time:.2f} seconds")

    if processed_count == 0:
        logging.error("No valid entries were processed during aggregation. Returning None.")
        return None

    # Verify memory is freed (optional)
    # mpool = cp.get_default_memory_pool()
    # logging.info(f"GPU memory used: {mpool.used_bytes()} bytes")
    # logging.info(f"GPU memory total: {mpool.total_bytes()} bytes")

    return activations_4d_gpu

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess activation data from DataFrame format to a 4D CuPy array on the GPU.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Python configuration file (e.g., key_prediction/config_linear.py) to load input path, matrix type, and precision.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output 4D CuPy array pickle file (e.g., activations_4d_gpu.pkl).")

    args = parser.parse_args()

    try:
        # 1. Load Config
        logging.info(f"Loading configuration from: {args.config}")
        CONFIG = load_config(args.config)
        input_file = CONFIG['file_path']
        matrix_type = CONFIG['matrix_type']
        precision = CONFIG.get('precision', 'float64') # Get precision from config
        logging.info(f"Input data file: {input_file}")
        logging.info(f"Matrix type: {matrix_type}")
        logging.info(f"GPU Precision: {precision}")

        # 2. Load Data into CPU DataFrame
        df_original_cpu = load_data_cpu(input_file)

        # Check matrix_type existence after loading
        if matrix_type not in df_original_cpu.columns:
            raise ValueError(f"Error: Matrix column '{matrix_type}' not found in loaded data. Available columns: {df_original_cpu.columns.tolist()}")

        # 3. Process Data on CPU (Expand Tokens, keep matrices as NumPy)
        expanded_df_cpu, activation_dim = process_data_cpu(df_original_cpu, matrix_type)

        # Free original large DataFrame if possible
        del df_original_cpu
        import gc
        gc.collect()

        # 4. Aggregate into 4D GPU Array (includes transfers)
        activations_4d_gpu = aggregate_to_4d_gpu(expanded_df_cpu, matrix_type, activation_dim, precision)

        # Free expanded CPU DataFrame
        del expanded_df_cpu
        gc.collect()


        # 5. Save Output (CuPy array)
        if activations_4d_gpu is not None:
            logging.info(f"Saving 4D CuPy array to {args.output}...")
            # Saving CuPy array directly using pickle might require specific handling
            # or converting back to NumPy temporarily just for saving if issues arise.
            # Let's try saving directly first.
            try:
                with open(args.output, 'wb') as f_out:
                    # Use cp.save or pickle; pickle is more standard for arbitrary objects
                    pickle.dump(activations_4d_gpu, f_out)
                logging.info("Successfully saved the 4D CuPy array.")
            except Exception as save_err:
                 logging.error(f"Error saving CuPy array directly with pickle: {save_err}")
                 logging.info("Attempting to save by converting to NumPy first...")
                 try:
                     activations_4d_np = cp.asnumpy(activations_4d_gpu)
                     with open(args.output, 'wb') as f_out:
                         pickle.dump(activations_4d_np, f_out)
                     logging.info("Successfully saved the 4D array after converting to NumPy.")
                     del activations_4d_np # Free temporary NumPy array
                 except Exception as np_save_err:
                     logging.error(f"Error saving NumPy converted array: {np_save_err}")
                     logging.error("Failed to save the output array.")
                     exit(1)

            # Free GPU memory explicitly if needed
            del activations_4d_gpu
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        else:
            logging.error("Aggregation resulted in None. Output file was not saved.")
            exit(1)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        exit(1)
    except ValueError as e:
        logging.error(f"Error: {e}")
        exit(1)
    except cp.cuda.memory.OutOfMemoryError:
        logging.error("GPU Out of Memory error during the process.")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())
        exit(1)

    logging.info("GPU Preprocessing finished.") 