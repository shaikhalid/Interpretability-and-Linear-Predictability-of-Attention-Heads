import sys
import os

# Add the parent directory of visualisation_helper to the Python path
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
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Added for color manipulation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression # Added import
from sklearn.metrics.pairwise import cosine_similarity # Added for finding best match
from sklearn.manifold import TSNE # Added for t-SNE plots

# Import the pastel theme
from visualisation_helper.pastel_theme import PLOT_STYLE, PASTEL_COLORS # Removed PASTEL_COLORS import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading (Copied from states_distance_plots.py) ---
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
            # config['matrix_type'] = 'q_matrix' # Example
            # logging.info(f"Automatically setting matrix_type to 'q_matrix' based on file name: {file_path}")
             config['matrix_type'] = 'k_matrix'
             logging.warning(f"Could not determine matrix type (K/V/Q) from filename. Defaulting to 'k_matrix'. Please specify 'matrix_type' in config if incorrect.")
        else:
            config['matrix_type'] = 'k_matrix' # Default fallback
            logging.warning(f"Defaulting matrix_type to 'k_matrix'. Please specify 'matrix_type' in config if incorrect.")


    return config

# --- Data Loading (Copied from states_distance_plots.py) ---
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
                        logging.warning(f"Loaded unexpected data type: {type(data_chunk)}")
                        loaded_data.append(data_chunk)
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
        logging.error("Ensure the pickle file contains data suitable for DataFrame creation (e.g., list of dicts).")
        raise

    if 'batch_id' in df.columns:
        df = df.rename(columns={'batch_id': 'sample_number'})
    if 'request_id' in df.columns:
        df = df.drop(columns=['request_id'])

    if matrix_type not in df.columns:
        potential_cols = [col for col in ['q_matrix', 'k_matrix', 'v_matrix'] if col in df.columns]
        if potential_cols:
             logging.warning(f"Config specified matrix '{matrix_type}', but it wasn't found. Found potential alternatives: {potential_cols}. Using '{potential_cols[0]}'.")
             matrix_type = potential_cols[0]
        else:
            raise ValueError(f"Error: Matrix column '{matrix_type}' not found. Available columns: {df.columns.tolist()}")

    return df, matrix_type

# --- Data Processing (Copied and adapted from states_distance_plots.py) ---
# Note: Prefill/Decode logic remains the same as we need token-level data first.
def process_prefill_data(prefill_df, matrix_type):
    """Process prefill data by expanding matrices into single tokens."""
    expanded_prefill_rows = []
    if not prefill_df.empty:
        logging.info(f"Processing {len(prefill_df)} prefill rows...")
        for _, row in prefill_df.iterrows():
            matrix = row[matrix_type]
            if 'num_tokens_in_matrix' in row and pd.notna(row['num_tokens_in_matrix']):
                 num_prefill_tokens = int(row['num_tokens_in_matrix'])
            elif hasattr(matrix, 'shape') and len(matrix.shape) > 1:
                 num_prefill_tokens = matrix.shape[1]
            else:
                 logging.warning(f"Could not determine number of prefill tokens for sample {row.get('sample_number', 'N/A')}, layer {row.get('layer_id', 'N/A')}, head {row.get('head_id', 'N/A')}. Skipping row.")
                 continue

            if hasattr(matrix, 'shape') and len(matrix.shape) == 2:
                 matrix = np.expand_dims(matrix, axis=0)
            elif not (hasattr(matrix, 'shape') and len(matrix.shape) >= 3):
                 logging.warning(f"Unexpected matrix shape {getattr(matrix, 'shape', 'N/A')} for prefill sample {row.get('sample_number', 'N/A')}, layer {row.get('layer_id', 'N/A')}, head {row.get('head_id', 'N/A')}. Skipping row.")
                 continue

            if matrix.shape[1] != num_prefill_tokens:
                 logging.warning(f"Mismatch between 'num_tokens_in_matrix' ({num_prefill_tokens}) and matrix shape ({matrix.shape[1]}) for sample {row.get('sample_number', 'N/A')}. Using matrix shape.")
                 num_prefill_tokens = matrix.shape[1]

            base_info_cols = list(row.index.drop([matrix_type], errors='ignore'))
            base_info_cols = [c for c in base_info_cols if c not in ['num_tokens_in_matrix', 'matrix_shape']]
            base_info = row[base_info_cols]

            for token_idx in range(num_prefill_tokens):
                try:
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
        else:
            base_cols = list(prefill_df.columns.drop([matrix_type], errors='ignore'))
            base_cols = [c for c in base_cols if c not in ['num_tokens_in_matrix', 'matrix_shape']]
            final_cols = base_cols + [matrix_type, 'token_position', 'prefill']
            expanded_prefill_df = pd.DataFrame(columns=final_cols)

        prefill_prompts_df = prefill_df
        if 'num_decode_tokens' in prefill_df.columns:
            prefill_prompts_df = prefill_df[prefill_df['num_decode_tokens'] == 0]
        if not prefill_prompts_df.empty:
            prefill_prompts_df['num_tokens_in_matrix_calc'] = prefill_prompts_df[matrix_type].apply(lambda x: x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else 0)
            sample_n_map = prefill_prompts_df.groupby('sample_number')['num_tokens_in_matrix_calc'].first()
            sample_n_map = sample_n_map.fillna(0).astype(int)
        else:
             sample_n_map = pd.Series(dtype=int)


    else:
        base_cols = list(prefill_df.columns.drop([matrix_type], errors='ignore'))
        base_cols = [c for c in base_cols if c not in ['num_tokens_in_matrix', 'matrix_shape']]
        final_cols = base_cols + [matrix_type, 'token_position', 'prefill']
        expanded_prefill_df = pd.DataFrame(columns=final_cols)
        sample_n_map = pd.Series(dtype=int)

    return expanded_prefill_df, sample_n_map


def process_decode_data(decode_df, sample_n_map, matrix_type):
    """Process decode data by calculating token positions."""
    if not decode_df.empty:
        logging.info(f"Processing {len(decode_df)} decode rows...")
        if 'sample_number' not in decode_df.columns:
             logging.error("Missing 'sample_number' column in decode data.")
             return pd.DataFrame(columns=decode_df.columns.tolist() + ['token_position', 'prefill'])

        if 'num_decode_tokens' not in decode_df.columns:
             logging.error("Missing 'num_decode_tokens' column in decode data.")
             raise ValueError("Missing 'num_decode_tokens' required for decode processing.")


        decode_df['N'] = decode_df['sample_number'].map(sample_n_map).fillna(0).astype(int)
        decode_df['token_position'] = (decode_df['num_decode_tokens'] - 1) + decode_df['N']
        decode_df['prefill'] = False

        def check_decode_matrix(matrix):
             if hasattr(matrix, 'shape') and len(matrix.shape) >= 2:
                 if matrix.shape[1] != 1:
                     logging.warning(f"Decode matrix has unexpected token dimension: {matrix.shape}. Expected 1 token.")
                 elif len(matrix.shape) == 1:
                     pass
                 return matrix
             else:
                 logging.warning(f"Unexpected data type or shape for decode matrix: {type(matrix)}")
                 return matrix

        decode_df = decode_df.drop(columns=['N', 'num_tokens_in_matrix', 'matrix_shape'], errors='ignore')
    return decode_df


def process_data(df, matrix_type):
    """Main function to process the DataFrame and expand matrices."""
    start_time = time.time()
    logging.info("Starting data processing to expand matrices...")
    df_processed = df.copy()

    required_cols = [matrix_type, 'sample_number']
    if 'num_decode_tokens' not in df_processed.columns:
         logging.warning("'num_decode_tokens' column not found. Decode token positions might be inaccurate if prefill exists.")

    df_processed['matrix_shape'] = df_processed[matrix_type].apply(lambda x: getattr(x, 'shape', None))
    df_processed['num_tokens_in_matrix'] = df_processed[matrix_type].apply(
        lambda x: x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else 1
    )

    is_prefill = df_processed['num_tokens_in_matrix'] > 1
    prefill_df = df_processed[is_prefill].copy()
    decode_df = df_processed[~is_prefill].copy()
    logging.info(f"Identified {len(prefill_df)} potential prefill rows and {len(decode_df)} potential decode rows based on matrix token count.")

    expanded_prefill_df, sample_n_map = process_prefill_data(prefill_df, matrix_type)
    decode_df = process_decode_data(decode_df, sample_n_map, matrix_type)

    common_columns = list(set(expanded_prefill_df.columns) & set(decode_df.columns))
    essential_cols = ['sample_number', 'layer_id', 'head_id', matrix_type, 'prefill', 'token_position']
    for col in essential_cols:
        if col not in common_columns and (col in expanded_prefill_df.columns or col in decode_df.columns):
            common_columns.append(col)

    for col in common_columns:
        if col not in expanded_prefill_df.columns:
            expanded_prefill_df[col] = pd.NA
        if col not in decode_df.columns:
            decode_df[col] = pd.NA

    # Combine valid dataframes, handling empty cases
    valid_dfs = []
    if not expanded_prefill_df.empty:
        valid_dfs.append(expanded_prefill_df[common_columns])
    if not decode_df.empty:
        valid_dfs.append(decode_df[common_columns])

    if not valid_dfs:
        logging.error("No data available after processing prefill and decode steps.")
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=essential_cols)

    expanded_df = pd.concat(valid_dfs, ignore_index=True)

    for col in ['sample_number', 'layer_id', 'head_id', 'token_position']:
         if col in expanded_df.columns:
             expanded_df[col] = pd.to_numeric(expanded_df[col], errors='coerce')

    expanded_df.dropna(subset=['sample_number', 'layer_id', 'head_id', 'token_position'], inplace=True)

    for col in ['sample_number', 'layer_id', 'head_id', 'token_position']:
         if col in expanded_df.columns and not expanded_df[col].empty :
             expanded_df[col] = expanded_df[col].astype(int)

    # Ensure the matrix column exists even if the dataframe is empty after processing
    if matrix_type not in expanded_df.columns:
        expanded_df[matrix_type] = np.nan # Or appropriate placeholder


    if not expanded_df.empty:
         expanded_df = expanded_df.sort_values(by=['sample_number', 'layer_id', 'head_id', 'token_position'], ignore_index=True)


    logging.info(f"Data processing time: {time.time() - start_time:.2f} seconds")
    logging.info(f"Original DF shape: {df.shape}")
    logging.info(f"Expanded DF shape (token level): {expanded_df.shape}")

    return expanded_df

# --- Helper Function for Color Shading --- # MODIFIED
def _adjust_brightness(hex_color, factor):
    """Adjusts the brightness of a hex color.
    factor > 1 lightens, factor < 1 darkens.
    """
    try:
        rgb = mcolors.hex2color(hex_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[2] = np.clip(hsv[2] * factor, 0.0, 1.0) # Adjust Value (brightness)
        new_rgb = mcolors.hsv_to_rgb(hsv)
        return mcolors.rgb2hex(new_rgb)
    except ValueError:
        logging.warning(f"Could not parse hex color: {hex_color}. Using original.")
        return hex_color

def _create_yellowish_shade(base_color, saturation_factor=1.2, brightness_factor=0.95):
    """Creates a more yellowish or orangish shade of the base color by adjusting hue and saturation."""
    try:
        rgb = mcolors.hex2color(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        # Use more orangish-yellow hue (between orange ~0.1 and yellow ~0.16)
        target_hue = 0.12  # More orangish-yellow (closer to orange than before)
        hsv[0] = target_hue * 0.85 + hsv[0] * 0.15  # Stronger shift toward orange-yellow
        # Adjust saturation and brightness
        hsv[1] = np.clip(hsv[1] * saturation_factor, 0.0, 1.0)  # Increase saturation
        hsv[2] = np.clip(hsv[2] * brightness_factor, 0.0, 1.0)  # Adjust brightness
        new_rgb = mcolors.hsv_to_rgb(hsv)
        return mcolors.rgb2hex(new_rgb)
    except ValueError:
        logging.warning(f"Could not parse hex color: {base_color}. Using original.")
        return base_color


# --- Vector Extraction and Plotting ---
def plot_3d_vectors(vec1, vec2, vec_pred, output_path, layer_head_1, layer_head_2, matrix_type_name, sample_num, token_pos):
    """Plots three 3D vectors (vec1, vec2, vec_pred) using matplotlib with themed styling."""
    if vec1.size < 3 or vec2.size < 3 or vec_pred.size < 3:
        logging.error(f"Vectors must have at least 3 dimensions to plot in 3D. Got shapes {vec1.shape}, {vec2.shape}, and {vec_pred.shape}")
        return

    vec1_3d = vec1[:3]
    vec2_3d = vec2[:3]
    vec_pred_3d = vec_pred[:3]

    # Normalize the 3D vectors for visualization
    norm1_3d = np.linalg.norm(vec1_3d)
    norm2_3d = np.linalg.norm(vec2_3d)
    norm_pred_3d = np.linalg.norm(vec_pred_3d)

    vec1_3d = vec1_3d / norm1_3d if norm1_3d > 0 else vec1_3d
    vec2_3d = vec2_3d / norm2_3d if norm2_3d > 0 else vec2_3d
    vec_pred_3d = vec_pred_3d / norm_pred_3d if norm_pred_3d > 0 else vec_pred_3d

    # --- Apply Theme Settings --- # MODIFIED to use white background with specific plane colors
    plt.rcParams['font.family'] = PLOT_STYLE['font_family']
    # Define specific colors based on the reference image
    color_plane_x = '#E6BF83'  # Orange/tan for the left wall (YZ plane)
    color_plane_y = '#D9A066'  # Slightly different orange/tan for the back wall (XZ plane)
    color_plane_z = '#FFFACD'  # Light yellow/cream for the floor (XY plane)
    color_grid = '#A9A9A9'     # Darker grey for grid lines for better visibility
    color_pane_edge = '#696969' # Dim grey for pane edges

    # Increase figure width to prevent cutting off elements
    fig = plt.figure(figsize=(14, 9))
    fig.set_facecolor('white') # Set figure background to white

    # Add more padding to prevent cut-off
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    # Make sure there's sufficient padding around the plot
    # Increased margins further
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
    
    ax.set_facecolor('white') # Set general axes background to white

    # Adjust position to ensure y-axis labels are visible
    # Increased camera distance further
    ax.dist = 12
    
    # Additional label padding to prevent overlap with tick labels
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    # Set individual pane colors using the defined hex codes
    ax.xaxis.pane.set_facecolor(color_plane_x)
    ax.yaxis.pane.set_facecolor(color_plane_y)
    ax.zaxis.pane.set_facecolor(color_plane_z)
    ax.xaxis.pane.set_edgecolor(color_pane_edge) # Using darker grey for edges on pale background
    ax.yaxis.pane.set_edgecolor(color_pane_edge)
    ax.zaxis.pane.set_edgecolor(color_pane_edge)

    # Grid settings - use defined grid color
    ax.grid(True, color=color_grid, # Use defined grid color
            linestyle=PLOT_STYLE['grid_linestyle'],
            linewidth=PLOT_STYLE['grid_linewidth'])

    # Plot vectors as arrows from origin
    # Use specific bright colors for vectors from the theme
    vector_color1 = PASTEL_COLORS.get('bright_blue', '#0000FF')  # Bright Blue for Input (with fallback)
    vector_color2 = PASTEL_COLORS.get('bright_red', '#FF0000')   # Bright Red for Actual (with fallback)
    vector_color_pred = PASTEL_COLORS.get('bright_green', '#00FF00') # Bright Green for Predicted (with fallback)

    # Increase arrow thickness - use full line_width from theme or even thicker
    arrow_thickness = PLOT_STYLE['line_width'] * 1.2  # Make arrows 20% thicker than standard line width
    
    ax.quiver(0, 0, 0, vec1_3d[0], vec1_3d[1], vec1_3d[2], color=vector_color1, 
              label=f'L{layer_head_1[0]}H{layer_head_1[1]} (Input Key State)', # Updated label
              arrow_length_ratio=0.12,  # Slightly larger arrow heads
              linewidth=arrow_thickness,
              zorder=10) # Ensure vectors are drawn on top
    
    ax.quiver(0, 0, 0, vec2_3d[0], vec2_3d[1], vec2_3d[2], color=vector_color2, 
              label=f'L{layer_head_2[0]}H{layer_head_2[1]} (Actual Key State)', # Updated label
              arrow_length_ratio=0.12,  # Slightly larger arrow heads
              linewidth=arrow_thickness,
              zorder=10) # Ensure vectors are drawn on top

    ax.quiver(0, 0, 0, vec_pred_3d[0], vec_pred_3d[1], vec_pred_3d[2], color=vector_color_pred,
              label=f'L{layer_head_2[0]}H{layer_head_2[1]} (Predicted Key State)', # Updated label
              arrow_length_ratio=0.12,
              linewidth=arrow_thickness,
              linestyle='dashed', # Dashed line for predicted
              zorder=10) # Ensure vectors are drawn on top

    # Determine plot limits based on vector coordinates, now including predicted vector
    max_abs_val = max(np.abs(vec1_3d).max(), np.abs(vec2_3d).max(), np.abs(vec_pred_3d).max())
    if max_abs_val == 0: max_abs_val = 1.0 # Avoid zero limits if vectors are zero
    limit_val = max_abs_val * 1.2 # Add padding

    ax.set_xlim([-limit_val, limit_val])
    ax.set_ylim([-limit_val, limit_val])
    ax.set_zlim([-limit_val, limit_val])

    # Apply font styles to labels and title
    ax.set_xlabel('Dimension 1', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Dimension 2', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_zlabel('Dimension 3', fontsize=PLOT_STYLE['label_fontsize'])
    # Updated title format
    title_str = f"First 3 Dimensions of {matrix_type_name} States\n"
    title_str += "GSM 8k on Llama3.1 8b\n"
    title_str += f"Sample {sample_num} - Token {token_pos}"
    ax.set_title(title_str, fontsize=PLOT_STYLE['title_fontsize'])

    # Apply font styles to tick labels
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])

    # Optional: Customize tick label colors if needed, otherwise they inherit default
    # tick_color = PLOT_STYLE.get('text_color', 'black') # Use theme text color or default black
    # ax.tick_params(axis='x', colors=tick_color)
    # ax.tick_params(axis='y', colors=tick_color)
    # ax.tick_params(axis='z', colors=tick_color)

    ax.legend(fontsize=PLOT_STYLE['label_fontsize'] * 0.8) # Slightly smaller legend font
    
    # Don't use tight_layout() here as it can cause issues with 3D plots
    # plt.tight_layout()

    # Ensure output path has a .pdf extension
    output_path_pdf = os.path.splitext(output_path)[0] + '.pdf'

    logging.info(f"Saving 3D vector plot as PDF (vector format for papers) to: {output_path_pdf}")
    # Save as PDF for high quality vector output with custom padding
    # Increased pad_inches significantly
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.5)
    plt.close(fig) # Close the figure to free memory

def plot_tsne_vectors(vec1, vec2, vec_pred, output_path, layer_head_1, layer_head_2, matrix_type_name, sample_num, token_pos, seed=42):
    """Plots three separate t-SNE visualizations of the dimensions for each vector (vec1, vec2, vec_pred)."""
    # Ensure vectors are 1D
    if vec1.ndim > 1 and vec1.shape[0] == 1:
        vec1 = vec1.flatten()
    if vec2.ndim > 1 and vec2.shape[0] == 1:
        vec2 = vec2.flatten()
    if vec_pred.ndim > 1 and vec_pred.shape[0] == 1:
        vec_pred = vec_pred.flatten()
    
    # Check dimensionality
    if vec1.size < 10 or vec2.size < 10 or vec_pred.size < 10:
        logging.warning(f"Vectors too small for meaningful t-SNE visualization: {vec1.size}, {vec2.size}, {vec_pred.size}. Need at least 10 dimensions.")
        return
    
    # Create three separate figures, one for each vector type
    vector_data = [
        {'name': 'input', 'vector': vec1, 'layer_head': layer_head_1, 'color': PASTEL_COLORS.get('bright_blue', '#0000FF')},
        {'name': 'target', 'vector': vec2, 'layer_head': layer_head_2, 'color': PASTEL_COLORS.get('bright_red', '#FF0000')},
        {'name': 'predicted', 'vector': vec_pred, 'layer_head': layer_head_2, 'color': PASTEL_COLORS.get('bright_green', '#00FF00')}
    ]
    
    for data in vector_data:
        # Reshape the vector for t-SNE
        # For t-SNE, we need to represent each dimension as a separate data point with features
        # We'll use a 2D representation where each dimension is a point
        vector = data['vector']
        n_dims = vector.size
        
        # Create a feature matrix: each dimension is a data point, features are derived from position and value
        X = np.zeros((n_dims, 3))  # 3 features: dimension index, value, normalized position
        
        # Feature 1: Dimension index
        X[:, 0] = np.arange(n_dims)
        
        # Feature 2: Actual value from the vector
        X[:, 1] = vector
        
        # Feature 3: Normalized position in the vector (0-1)
        X[:, 2] = np.linspace(0, 1, n_dims)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, n_dims//4), 
                    n_iter=1000, learning_rate='auto', init='pca')
        try:
            embedded_dims = tsne.fit_transform(X)
        except Exception as e:
            logging.error(f"t-SNE failed for {data['name']} vector: {e}")
            try:
                logging.info(f"Retrying t-SNE for {data['name']} with init='random'")
                tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, n_dims//4),
                            n_iter=1000, learning_rate='auto', init='random')
                embedded_dims = tsne.fit_transform(X)
            except Exception as e_retry:
                logging.error(f"t-SNE retry failed for {data['name']} vector: {e_retry}. Skipping plot.")
                continue
        
        # Create the plot
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Create a scatter plot where:
        # - Each point represents one dimension of the vector
        # - Color intensity based on the actual value (normalized)
        
        # Normalize values for coloring
        values = vector
        vmin, vmax = np.min(values), np.max(values)
        if vmin == vmax:  # Handle constant vector
            normalized_values = np.zeros_like(values)
        else:
            normalized_values = (values - vmin) / (vmax - vmin)
        
        # Create a colormap based on the vector type color
        base_color = data['color']
        cmap = plt.cm.get_cmap('coolwarm')
        
        # Scatter plot with color based on the actual value
        scatter = ax.scatter(embedded_dims[:, 0], embedded_dims[:, 1], 
                           c=values, cmap='coolwarm',
                           s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Dimension Value', fontsize=PLOT_STYLE['label_fontsize'] * 0.9)
        
        # Optionally annotate some points with their dimension index
        # Only annotate a subset to avoid clutter - highest/lowest values
        if n_dims > 10:
            # Find indices of top and bottom values
            n_annotate = min(5, n_dims // 10)  # Annotate top/bottom 5 or fewer for smaller vectors
            top_indices = np.argsort(values)[-n_annotate:]
            bottom_indices = np.argsort(values)[:n_annotate]
            indices_to_annotate = np.concatenate([top_indices, bottom_indices])
            
            for idx in indices_to_annotate:
                ax.annotate(f'{idx}', (embedded_dims[idx, 0], embedded_dims[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='black', fontweight='bold')
        else:
            # Annotate all points for small vectors
            for idx in range(n_dims):
                ax.annotate(f'{idx}', (embedded_dims[idx, 0], embedded_dims[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='black', fontweight='bold')
        
        # Add labels and title
        ax.set_xlabel('t-SNE Dimension 1', fontsize=PLOT_STYLE['label_fontsize'])
        ax.set_ylabel('t-SNE Dimension 2', fontsize=PLOT_STYLE['label_fontsize'])
        
        layer_head_str = f"L{data['layer_head'][0]}H{data['layer_head'][1]}"
        title_str = f"t-SNE Visualization of {matrix_type_name} {data['name'].capitalize()} Vector Dimensions\n"
        title_str += f"{layer_head_str} - Sample {sample_num} - Token {token_pos}\n"
        title_str += f"({n_dims} dimensions)"
        
        ax.set_title(title_str, fontsize=PLOT_STYLE['title_fontsize'])
        
        # Style grid and ticks
        ax.grid(True, color=PLOT_STYLE.get('grid_color', '#DDDDDD'), 
                linestyle=PLOT_STYLE['grid_linestyle'], 
                linewidth=PLOT_STYLE['grid_linewidth'])
        ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])
        
        # Save the plot
        output_name_parts = os.path.splitext(output_path)
        output_path_tsne = f"{output_name_parts[0]}_{data['name']}_tsne{output_name_parts[1]}"
        
        logging.info(f"Saving t-SNE plot for {data['name']} vector to: {output_path_tsne}")
        plt.savefig(output_path_tsne, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the first 3 dimensions of activation vectors for two layer-head pairs at a random token step.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Python configuration file (e.g., key_prediction/config_linear.py).")
    parser.add_argument("--num_pairs", type=int, default=5,
                        help="Number of random layer-head pairs (L1 < L2) to visualize.")
    parser.add_argument("--output", type=str, required=True,
                        help="Base path/prefix to save the output 3D plot images (e.g., vector_viz). Extension will be added.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results.")
    # Add optional arguments if needed, e.g., --matrix_type, --token_position

    args = parser.parse_args()
    parser.description = "Visualize predicted vs actual activation vectors for N random layer-head pairs (L1 < L2)."

    # Set random seed if provided
    if args.seed is not None:
        logging.info(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)

    try:
        # 1. Load Config
        logging.info(f"Loading configuration from: {args.config}")
        CONFIG = load_config(args.config)
        input_file = CONFIG['file_path']
        matrix_type = CONFIG['matrix_type'] # Use matrix_type from config
        matrix_type_name = matrix_type.replace('_matrix','').upper()

        logging.info(f"Input data file: {input_file}")
        logging.info(f"Matrix type: {matrix_type} ({matrix_type_name})")
        logging.info(f"Attempting to visualize {args.num_pairs} random pairs.")
        logging.info(f"Output prefix: {args.output}")

        # 2. Load Data
        df_original, matrix_type = load_data(input_file, matrix_type) # Update matrix_type if needed

        # 3. Process Data (Expand Tokens)
        expanded_df = process_data(df_original, matrix_type)
        if expanded_df.empty:
            logging.error("Expanded DataFrame is empty after processing. Cannot proceed.")
            exit(1)

        # Ensure matrix column contains numpy arrays
        if not expanded_df[matrix_type].apply(lambda x: isinstance(x, np.ndarray)).all():
            logging.warning(f"Not all entries in '{matrix_type}' column are NumPy arrays. Check data processing.")
            # Add conversion/filtering if needed, or exit
            valid_mask = expanded_df[matrix_type].apply(lambda x: isinstance(x, np.ndarray))
            logging.info(f"Filtering out {len(expanded_df) - valid_mask.sum()} rows with non-array data.")
            expanded_df = expanded_df[valid_mask].copy()
            if expanded_df.empty:
                 logging.error(f"No valid NumPy arrays found in '{matrix_type}' column after filtering.")
                 exit(1)

        # --- Identify and Select Layer-Head Pairs ---
        unique_layers = sorted(expanded_df['layer_id'].unique())
        unique_heads = sorted(expanded_df['head_id'].unique())
        logging.info(f"Found {len(unique_layers)} layers and {len(unique_heads)} heads in the data.")

        all_pairs = []
        # Ensure we have at least two layers to form pairs
        if len(unique_layers) >= 2:
            # Generate adjacent layer pairs (l, l+1)
            unique_layers_set = set(unique_layers) # Faster lookups
            layer_pairs = [(l, l + 1) for l in unique_layers if (l + 1) in unique_layers_set]
            # Combine with all head combinations
            for l1, l2 in layer_pairs:
                for h1 in unique_heads:
                    for h2 in unique_heads:
                        all_pairs.append(((l1, h1), (l2, h2)))
        else:
            logging.warning("Not enough unique layers (< 2) found to create adjacent layer pairs (L, L+1).")

        if not all_pairs:
            logging.error("No valid adjacent layer-head pairs (L, L+1) could be generated. Exiting.")
            exit(1)

        logging.info(f"Generated {len(all_pairs)} potential adjacent layer-head pairs (L, L+1). Selecting up to {args.num_pairs} to process.")

        # Select N random pairs (or fewer if not enough available)
        if args.num_pairs == 5: # Check if default value is used (or explicitly set to 5)
             logging.info("Defaulting to visualize the specific pair L0H6 -> L1H3 because --num_pairs was not specified or set to 5.")
             selected_pairs = [((0, 6), (1, 3))]
             # Check if the required layers and heads exist in the loaded data
             unique_layers_set = set(unique_layers) # Convert for efficient lookup
             unique_heads_set = set(unique_heads)   # Convert for efficient lookup
             required_layers = {0, 1}
             required_heads = {3, 6}
             missing_layers = required_layers - unique_layers_set
             missing_heads = required_heads - unique_heads_set
             if missing_layers or missing_heads:
                 error_msg = "The default pair L0H6 -> L1H3 cannot be visualized because required components are missing from the data:"
                 if missing_layers:
                     error_msg += f" Missing Layers: {missing_layers}."
                 if missing_heads:
                     error_msg += f" Missing Heads: {missing_heads}."
                 logging.error(error_msg)
                 exit(1)
             # Verify the pair (0,6) -> (1,3) is actually an adjacent pair (optional but good check)
             # if ((0, 6), (1, 3)) not in all_pairs:
             #    logging.warning("The specific pair L0H6->L1H3 was selected by default but might not be present in the generated list of adjacent pairs.")
             #    # Depending on desired behavior, could exit or continue hoping filter works. Let's continue.

        else: # User specified a non-default number of pairs
             logging.info(f"Selecting {args.num_pairs} random adjacent pairs as requested.")
             num_to_select = min(args.num_pairs, len(all_pairs))
             if num_to_select < args.num_pairs:
                 logging.warning(f"Requested {args.num_pairs} pairs, but only {num_to_select} unique adjacent pairs are available.")
             if num_to_select == 0:
                 logging.error("No available adjacent pairs to select randomly.")
                 exit(1)
             selected_pairs = random.sample(all_pairs, num_to_select)
        # --- End Pair Selection ---

        logging.info(f"Selected pairs to process: {selected_pairs}")
        # --- End Pair Selection ---

        # --- Process Each Selected Pair ---
        processed_count = 0
        results_to_plot = [] # Store results here

        for pair_idx, ((l1, h1), (l2, h2)) in enumerate(selected_pairs):
            logging.info(f"\n--- Processing Pair {pair_idx+1}/{len(selected_pairs)}: L{l1}H{h1} -> L{l2}H{h2} ---")
            layer_head_1 = (l1, h1)
            layer_head_2 = (l2, h2)

            # 4. Filter for the two specified layer-head pairs
            df1 = expanded_df[(expanded_df['layer_id'] == l1) & (expanded_df['head_id'] == h1)].copy()
            df2 = expanded_df[(expanded_df['layer_id'] == l2) & (expanded_df['head_id'] == h2)].copy()

            logging.info(f"Found {len(df1)} entries for L{l1}H{h1}")
            logging.info(f"Found {len(df2)} entries for L{l2}H{h2}")

            if df1.empty or df2.empty:
                 logging.warning(f"Skipping pair L{l1}H{h1} -> L{l2}H{h2}: Could not find data for one or both states.")
                 continue # Skip to the next pair

            # 5. Find common sample/token positions
            merged_df = pd.merge(
                df1[['sample_number', 'token_position', matrix_type]],
                df2[['sample_number', 'token_position', matrix_type]],
                on=['sample_number', 'token_position'],
                suffixes=('_1', '_2')
            )
            logging.info(f"Found {len(merged_df)} matching token positions across samples for the two states.")

            if merged_df.empty:
                 logging.warning(f"Skipping pair L{l1}H{h1} -> L{l2}H{h2}: No matching sample/token positions found.")
                 continue # Skip to the next pair

            # 7.5 Train a Linear Regression model to predict vec2 from vec1
            logging.info("Preparing data for Linear Regression...")
            # Extract all pairs from merged_df for training
            X_lr_raw = merged_df[matrix_type + '_1'].tolist()
            y_lr_raw = merged_df[matrix_type + '_2'].tolist()
            # sample_numbers = merged_df['sample_number'].tolist() # Not needed for training
            # token_positions = merged_df['token_position'].tolist()

            # Flatten each matrix in the lists
            # Filter out None values resulting from potential malformed data during extraction
            X_lr_flat = [np.asarray(m).flatten() for m in X_lr_raw if m is not None and np.asarray(m).ndim > 0]
            y_lr_flat = [np.asarray(m).flatten() for m in y_lr_raw if m is not None and np.asarray(m).ndim > 0]

            # Store original indices corresponding to the flattened vectors
            original_indices = [i for i, (m1, m2) in enumerate(zip(X_lr_raw, y_lr_raw)) if m1 is not None and np.asarray(m1).ndim > 0 and m2 is not None and np.asarray(m2).ndim > 0]

            # Ensure all vectors have the same dimension after flattening
            # And remove pairs where flattening might have failed or dimensions mismatch
            if not X_lr_flat or not y_lr_flat:
                logging.warning(f"Skipping pair L{l1}H{h1} -> L{l2}H{h2}: No valid vectors found after initial flattening.")
                continue

            target_shape_x = X_lr_flat[0].shape
            target_shape_y = y_lr_flat[0].shape

            valid_indices_filtered = [i for i, (x, y) in enumerate(zip(X_lr_flat, y_lr_flat)) if x.shape == target_shape_x and y.shape == target_shape_y]

            if len(valid_indices_filtered) < len(X_lr_flat):
                logging.warning(f"Pair L{l1}H{h1} -> L{l2}H{h2}: Removed {len(X_lr_flat) - len(valid_indices_filtered)} samples due to inconsistent shapes after flattening.")

            if not valid_indices_filtered:
                logging.warning(f"Skipping pair L{l1}H{h1} -> L{l2}H{h2}: Not enough valid data to train Linear Regression model after shape filtering.")
                continue

            # Filter based on valid shapes and update original indices mapping
            X_lr = np.array([X_lr_flat[i] for i in valid_indices_filtered])
            y_lr = np.array([y_lr_flat[i] for i in valid_indices_filtered])
            # Map the filtered indices back to the original merged_df indices
            final_original_indices = [original_indices[i] for i in valid_indices_filtered]

            logging.info(f"Training Linear Regression model on {len(X_lr)} valid samples for L{l1}H{h1} -> L{l2}H{h2}...")
            lr_model = LinearRegression()
            lr_model.fit(X_lr, y_lr)
            logging.info("Linear Regression model trained.")

            # Predict ALL vec2 vectors using the trained model
            y_pred_lr = lr_model.predict(X_lr)
            logging.info(f"Generated predictions for all {len(y_pred_lr)} valid token steps.")

            # Calculate cosine similarity for each actual vs predicted pair
            similarities = []
            for i in range(len(y_lr)):
                # Reshape for cosine_similarity function (expects 2D arrays)
                actual = y_lr[i].reshape(1, -1)
                predicted = y_pred_lr[i].reshape(1, -1)
                similarity = cosine_similarity(actual, predicted)[0][0] # Get the single similarity value
                similarities.append(similarity)

            # Find the index of the highest similarity
            best_match_index_in_filtered = np.argmax(similarities)
            max_similarity = similarities[best_match_index_in_filtered]

            # Get the corresponding original index in merged_df
            best_match_original_index = final_original_indices[best_match_index_in_filtered]
            selected_row = merged_df.iloc[best_match_original_index] # Use iloc with the original index

            # Extract the vectors and info for the best match
            vec1 = X_lr[best_match_index_in_filtered]
            vec2 = y_lr[best_match_index_in_filtered]
            vec2_pred = y_pred_lr[best_match_index_in_filtered]

            sample_num = selected_row['sample_number']
            token_pos = selected_row['token_position']
            sample_token_info = f"Best Match (Cos Sim: {max_similarity:.3f}) - Sample: {sample_num}, Token Pos: {token_pos}"
            logging.info(f"Selected token step with highest cosine similarity: {sample_token_info}")

            # Define output filename for this specific pair
            output_filename = f"{args.output}_L{l1}H{h1}_to_L{l2}H{h2}.pdf" # Added .pdf extension here

            # Store results for potential plotting later
            results_to_plot.append({
                'max_similarity': max_similarity,
                'vec1': vec1,
                'vec2': vec2,
                'vec2_pred': vec2_pred,
                'output_filename': output_filename,
                'layer_head_1': layer_head_1,
                'layer_head_2': layer_head_2,
                'matrix_type_name': matrix_type_name,
                'sample_num': sample_num, # Store sample_num
                'token_pos': token_pos   # Store token_pos
            })

            processed_count += 1
        # --- End Pair Processing Loop ---

        logging.info(f"\nSuccessfully processed {processed_count} out of {len(selected_pairs)} selected adjacent pairs.")

        # --- Select Top 3 and Plot ---
        if not results_to_plot:
            logging.warning("No pairs were successfully processed, nothing to plot.")
        else:
            # Sort results by max_similarity descending
            results_to_plot.sort(key=lambda x: x['max_similarity'], reverse=True)

            num_plots = min(3, len(results_to_plot))
            logging.info(f"Plotting the top {num_plots} pairs based on highest cosine similarity...")

            for i in range(num_plots):
                result = results_to_plot[i]
                logging.info(f"  Plotting {i+1}: {result['output_filename']} (Max Cos Sim: {result['max_similarity']:.4f})")
                plot_3d_vectors(
                    result['vec1'],
                    result['vec2'],
                    result['vec2_pred'],
                    result['output_filename'],
                    result['layer_head_1'],
                    result['layer_head_2'],
                    result['matrix_type_name'],
                    result['sample_num'], # Pass sample_num
                    result['token_pos']  # Pass token_pos
                )
                # Also plot t-SNE for the same result
                # Construct a different output filename for t-SNE plot
                base_output_filename, ext = os.path.splitext(result['output_filename'])
                tsne_output_filename = base_output_filename + '_tsne' + ext # e.g. vector_viz_L0H0_to_L1H1_tsne.pdf

                logging.info(f"  Plotting t-SNE for: {tsne_output_filename} (Max Cos Sim: {result['max_similarity']:.4f})")
                plot_tsne_vectors(
                    result['vec1'],
                    result['vec2'],
                    result['vec2_pred'],
                    tsne_output_filename, # Use the new filename for t-SNE
                    result['layer_head_1'],
                    result['layer_head_2'],
                    result['matrix_type_name'],
                    result['sample_num'],
                    result['token_pos'],
                    seed=args.seed if args.seed is not None else 42 # Pass seed for reproducibility
                )
        # --- End Plotting ---

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
    except ValueError as e:
        logging.error(f"Error: {e}")
    except KeyError as e:
         logging.error(f"Configuration or data structure error: Missing key {e}")
         exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())
        exit(1)

    logging.info("Script finished.")
