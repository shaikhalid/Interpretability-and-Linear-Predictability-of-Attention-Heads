import torch
import math
import argparse
import networkx as nx
import numpy as np
from collections import defaultdict
import re
import os
import itertools
from transformers import AutoModelForCausalLM, AutoConfig
from scipy.stats import pearsonr, spearmanr, kendalltau

def principal_angles_with_overlap(A: torch.Tensor,
                                  B: torch.Tensor,
                                  tol: float = 1e-6):
    """
    Calculates principal angles and related metrics between two subspaces A and B.

    A: (m, k1), B: (m, k2) with k2 <= k1 <= m. Subspace basis matrices.
    tol: threshold for considering sigma_i == 1.

    Returns:
      mean_angle_deg    float       mean(principal angles) in degrees. Lower is better
      max_angle_deg     float       max(principal angle) in degrees    Lower is better
      sum_squared_sines float       = sum_i (1 - sigma_i^2)            Lower is better
      dim_overlap       int         = count(sigma_i >= 1 - tol)        Higher is better
    """
    # Ensure inputs are float tensors for SVD
    A = A.float()
    B = B.float()

    # 1) orthonormal bases
    try:
        QA, _ = torch.linalg.qr(A, mode='reduced')    # (m, k1)
        QB, _ = torch.linalg.qr(B, mode='reduced')    # (m, k2)
    except torch.linalg.LinAlgError as e:
        print(f"QR decomposition failed: {e}")
        print(f"Input shapes: A={A.shape}, B={B.shape}")
        # Handle error, e.g., return NaNs or default values
        return float('nan'), float('nan'), float('nan'), 0


    # 2) cross‐inner product
    M = QA.T @ QB                                 # (k1, k2)

    # 3) canonical correlations (singular values)
    try:
        # Ensure M is float for SVD
        sigma = torch.linalg.svdvals(M.float())           # (min(k1, k2),)
    except torch.linalg.LinAlgError as e:
         print(f"SVD failed: {e}")
         print(f"M shape: {M.shape}, M dtype: {M.dtype}")
         # Handle error
         return float('nan'), float('nan'), float('nan'), 0

    # Clamp sigma values to be within [0, 1] to avoid domain errors in acos
    sigma = sigma.clamp(0.0, 1.0)

    # 4) principal angles (theta)
    # Add a small epsilon to avoid acos(1.0) numerical issues if any sigma is exactly 1
    # Although clamp should handle > 1, epsilon handles potential exact 1.0 instability
    epsilon = 1e-7
    theta = torch.acos(sigma.clamp(0.0, 1.0 - epsilon)) # (min(k1, k2),)

    # 5) Sum of squared sines (1 - cos^2(theta) = sin^2(theta))
    sum_squared_sines = torch.sum(1.0 - sigma**2).item()

    # 6) Overlap dimension (count angles close to 0, i.e., sigma close to 1)
    dim_overlap = int((sigma >= (1.0 - tol)).sum().item())

    # 7) Interpretable summaries
    if theta.numel() == 0: # Handle case where one subspace might be zero-dimensional after QR
         mean_angle_deg = float('nan')
         max_angle_deg = float('nan')
    else:
         mean_angle_deg = float(theta.mean().item() * 180.0 / math.pi)
         max_angle_deg  = float(theta.max().item()  * 180.0 / math.pi)


    return mean_angle_deg, max_angle_deg, sum_squared_sines, dim_overlap

def parse_node_label(label):
    """Parses '(layer, head)' format label."""
    match = re.match(r'\((\d+),\s*(\d+)\)', label)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_and_filter_gml(gml_path, r2_attribute='weight'):
    """Loads GML, filters for intra-layer edges, and returns R2 values."""
    print(f"Loading GML file: {gml_path}")
    try:
        G = nx.read_gml(gml_path, label='id')
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"Error reading GML file {gml_path}: {e}")
        return None

    r2_values = {}
    skipped_inter_layer = 0
    skipped_missing_attr = 0
    skipped_parse_error = 0
    edges_processed = 0

    print(f"Filtering for intra-layer edges and extracting '{r2_attribute}'...")
    for u_id, v_id, data in G.edges(data=True):
        edges_processed += 1
        u_label = G.nodes[u_id].get('label', '')
        v_label = G.nodes[v_id].get('label', '')

        u_layer, u_head = parse_node_label(u_label)
        v_layer, v_head = parse_node_label(v_label)

        if u_layer is None or v_layer is None or u_head is None or v_head is None:
            # print(f"Warning: Could not parse labels for edge ({u_label}, {v_label}). Skipping.")
            skipped_parse_error += 1
            continue

        if u_layer == v_layer:
            if r2_attribute in data:
                try:
                    r2 = float(data[r2_attribute])
                    # Store R2 for the pair, ensure order doesn't matter for lookup
                    key = tuple(sorted((u_head, v_head)))
                    r2_values[(u_layer, key[0], key[1])] = r2
                except (ValueError, TypeError):
                     # print(f"Warning: Invalid R2 value '{data[r2_attribute]}' for edge ({u_label}, {v_label}). Skipping.")
                     skipped_missing_attr +=1 # Count as missing/invalid attribute
            else:
                # print(f"Warning: Missing R2 attribute '{r2_attribute}' for edge ({u_label}, {v_label}). Skipping.")
                skipped_missing_attr += 1
        else:
            skipped_inter_layer += 1

    print(f"Processed {edges_processed} edges.")
    print(f"Found {len(r2_values)} intra-layer edges with valid '{r2_attribute}'.")
    if skipped_inter_layer > 0:
        print(f"Skipped {skipped_inter_layer} inter-layer edges.")
    if skipped_missing_attr > 0:
        print(f"Skipped {skipped_missing_attr} edges due to missing/invalid '{r2_attribute}'.")
    if skipped_parse_error > 0:
        print(f"Skipped {skipped_parse_error} edges due to label parsing errors.")

    if not r2_values:
         print("Warning: No valid intra-layer R2 values found.")
         return None

    return r2_values

def load_llm(model_name_or_path):
    """Loads a Hugging Face causal LM."""
    print(f"Loading model: {model_name_or_path}")
    try:
        # Load config first to get model details
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if available
            device_map='auto', # Automatically distribute across GPUs if available
            trust_remote_code=True,
            cache_dir = '../.cache'
        )
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
        return model, config
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        return None, None

def extract_head_k_projections(model, config, layer_idx):
    """Extracts K-projection weights for each head in a given layer."""
    try:
        # Accessing weights - common pattern, might need adjustment for specific models
        # e.g., model.model.layers / model.transformer.h etc.
        layer = model.model.layers[layer_idx]
        k_proj_weight = layer.self_attn.k_proj.weight # Shape: (num_heads * head_dim, hidden_size)

        hidden_size = config.hidden_size
        num_heads = config.num_key_value_heads # Use num_key_value_heads for K/V
        head_dim = config.hidden_size // config.num_attention_heads # head_dim often relates to Q, K, V dims
        # Verify expected shape based on config
        expected_shape = (num_heads * head_dim, hidden_size)
        if k_proj_weight.shape != expected_shape:
             print(f"Warning: Layer {layer_idx} K-proj weight shape mismatch. Got {k_proj_weight.shape}, expected {expected_shape}. Using actual shape.")
             # Adjust num_heads or head_dim based on actual weight shape if possible
             actual_total_dim = k_proj_weight.shape[0]
             # Assuming hidden_size is correct, try to infer num_heads * head_dim
             # This part might need refinement based on model architecture variations
             if actual_total_dim % hidden_size == 0 : # Unlikely, k_proj maps TO head dims
                 pass # Shape is likely (out_dim, in_dim)
             elif k_proj_weight.shape[1] == hidden_size: # More likely (num_heads*head_dim, hidden_size)
                 head_dim = actual_total_dim // num_heads # Infer head_dim if num_heads seems right
                 print(f"Adjusted head_dim based on weight shape to {head_dim}")
             else:
                  print(f"Error: Cannot reconcile K-proj weight shape {k_proj_weight.shape} with config hidden_size {hidden_size}.")
                  return None, 0, 0 # Return None if shape is totally unexpected


        # Transpose to (hidden_size, num_heads * head_dim) for splitting
        k_proj_transposed = k_proj_weight.t().contiguous() # Use .t() for 2D transpose

        # Split into list of head tensors: List[(hidden_size, head_dim)]
        head_k_projections = list(torch.split(k_proj_transposed, head_dim, dim=1))

        # Verify number of heads extracted matches config
        if len(head_k_projections) != num_heads:
             print(f"Warning: Layer {layer_idx} - Extracted {len(head_k_projections)} heads, but config specifies {num_heads} key/value heads.")
             # Potentially adjust num_heads for iteration if mismatch is confirmed/expected for this model
             # num_heads = len(head_k_projections) # Uncomment carefully

        print(f"Layer {layer_idx}: Extracted {len(head_k_projections)} K-projection heads, each shape {head_k_projections[0].shape}")
        return head_k_projections, num_heads, head_dim

    except AttributeError as e:
        print(f"Error accessing K-projection weights for layer {layer_idx}: {e}. Check model structure.")
        return None, 0, 0
    except Exception as e:
        print(f"Unexpected error extracting weights for layer {layer_idx}: {e}")
        return None, 0, 0

def compute_rank_linear_metrics(x, y):
    """
    Compute three association metrics between two arrays/lists x and y:
      - Pearson's r
      - Spearman's rho
      - Kendall's tau

    Returns a dict with metric names as keys and (statistic, p-value) tuples.
    """
    # ensure numpy arrays and flatten
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    results = {}
    # 1. Pearson
    try:
        pearson_stat, pearson_p = pearsonr(x, y)
        results['pearson_r'] = (pearson_stat, pearson_p)
    except ValueError as e:
        print(f"Could not calculate Pearson correlation: {e}")
        results['pearson_r'] = (float('nan'), float('nan'))

    # 2. Spearman
    try:
        spearman_stat, spearman_p = spearmanr(x, y)
        results['spearman_rho'] = (spearman_stat, spearman_p)
    except ValueError as e:
        print(f"Could not calculate Spearman correlation: {e}")
        results['spearman_rho'] = (float('nan'), float('nan'))

    # 3. Kendall
    try:
        kendall_stat, kendall_p = kendalltau(x, y)
        results['kendall_tau'] = (kendall_stat, kendall_p)
    except ValueError as e:
        print(f"Could not calculate Kendall correlation: {e}")
        results['kendall_tau'] = (float('nan'), float('nan'))

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Calculate principal angles between LLM head K-projections and correlate with GML R2 values."
    )
    parser.add_argument("gml_file", help="Path to the input GML file.")
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., 'meta-llama/Meta-Llama-3.1-8B').")
    parser.add_argument("-a", "--attribute", default='weight', help="GML edge attribute for R2 value (default: 'weight').")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for dimension overlap calculation (default: 1e-6).")

    args = parser.parse_args()

    # 1. Load GML and R2 values
    r2_values = load_and_filter_gml(args.gml_file, args.attribute)
    if r2_values is None:
        print("Exiting due to GML loading/processing errors.")
        return 1

    # 2. Load LLM
    model, config = load_llm(args.model_id)
    if model is None or config is None:
        print("Exiting due to model loading errors.")
        return 1

    # Determine model parameters (handle potential variations in config names)
    try:
        num_layers = config.num_hidden_layers
        # Use num_key_value_heads for K/V projections, common in grouped-query attention
        num_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads # Head dim often based on total attention heads

        print(f"Model Config: Layers={num_layers}, KV Heads={num_kv_heads}, Head Dim={head_dim}")
    except AttributeError as e:
        print(f"Error accessing model configuration parameters (num_layers, num_key_value_heads, hidden_size, num_attention_heads): {e}")
        return 1


    # 3. Calculate Metrics and Collect Results
    results = {
        "mean_angle": [], "max_angle": [], "sum_sines": [], "dim_overlap": [], "r2": []
    }
    processed_pairs = 0

    print("\nCalculating principal angles for intra-layer head pairs...")
    for layer_idx in range(num_layers):
        print(f"--- Processing Layer {layer_idx} ---")
        head_k_projections, n_heads_layer, _ = extract_head_k_projections(model, config, layer_idx)

        if head_k_projections is None:
            print(f"Skipping layer {layer_idx} due to weight extraction errors.")
            continue

        # Ensure the number of heads used matches what was extracted
        if n_heads_layer != num_kv_heads:
             print(f"Note: Using {n_heads_layer} heads found in layer {layer_idx} instead of config's {num_kv_heads}.")
        
        num_heads_to_iterate = n_heads_layer # Use the actual number of heads found

        # Iterate through unique pairs of heads within the layer
        for head1_idx, head2_idx in itertools.combinations(range(num_heads_to_iterate), 2):
            # Ensure head indices are ordered for lookup in r2_values
            key = tuple(sorted((head1_idx, head2_idx)))
            r2_key = (layer_idx, key[0], key[1])

            if r2_key in r2_values:
                processed_pairs += 1
                r2 = r2_values[r2_key]
                A = head_k_projections[head1_idx]
                B = head_k_projections[head2_idx]

                # Move tensors to CPU for calculation if they aren't already
                # Calculations might be faster on CPU if matrices are not huge
                A_cpu = A.detach().cpu()
                B_cpu = B.detach().cpu()

                mean_angle, max_angle, sum_sines, dim_ov = principal_angles_with_overlap(A_cpu, B_cpu, tol=args.tol)

                # Check for NaN results from calculation
                if math.isnan(mean_angle) or math.isnan(max_angle) or math.isnan(sum_sines):
                     print(f"Warning: NaN result for pair L{layer_idx} H{head1_idx}-H{head2_idx}. Skipping.")
                     continue


                results["r2"].append(r2)
                results["mean_angle"].append(mean_angle)
                results["max_angle"].append(max_angle)
                results["sum_sines"].append(sum_sines)
                results["dim_overlap"].append(dim_ov)
            # else:
            #     # Optional: Report pairs found in model but not GML
            #     # print(f"Debug: R2 value not found for L{layer_idx} H{head1_idx}-H{head2_idx}")
            #     pass

    print(f"\nProcessed {processed_pairs} head pairs with corresponding R2 values.")

    if processed_pairs < 2: # Need at least 2 points for correlation
        print("Not enough data points (< 2) with both metrics and R2 values to calculate correlation.")
        return 1

    # 4. Calculate and Print Correlations
    print("\n--- Correlations with R2 ---")
    metrics_to_correlate = ["mean_angle", "max_angle", "sum_sines", "dim_overlap"]
    
    # Convert lists to numpy arrays for correlation calculation
    r2_array = np.array(results["r2"])
    if np.std(r2_array) == 0:
        print("Warning: R2 values have zero standard deviation. Cannot calculate meaningful correlations.")
        # Allow proceeding to calculate for metrics, but Pearson/Spearman/Kendall with R2 will likely fail or be NaN.

    for metric in metrics_to_correlate:
        metric_array = np.array(results[metric])

        # Check for NaNs in metric array
        if np.isnan(metric_array).any():
            print(f"\nWarning: NaNs found in '{metric}' data. Skipping correlation calculations for this metric.")
            continue
        
        # Check for zero standard deviation in metric array
        if np.std(metric_array) == 0:
            print(f"\nWarning: '{metric}' values have zero standard deviation. Cannot calculate meaningful correlations for this metric.")
            # Continue, but correlations will likely be NaN or fail.

        print(f"\nCorrelations for R2 vs {metric}:")

        # Use the new function to get all three correlations
        correlation_results = compute_rank_linear_metrics(r2_array, metric_array)

        for corr_name, (stat, p_val) in correlation_results.items():
            if math.isnan(stat):
                print(f"  {corr_name+':':<15} Result is NaN (likely due to zero variance or other issues)")
            else:
                print(f"  {corr_name+':':<15} Statistic = {stat:>+.4f}, p-value = {p_val:.4e}")

    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
# Removed original single function structure
# Removed original return values (now handled in main)

