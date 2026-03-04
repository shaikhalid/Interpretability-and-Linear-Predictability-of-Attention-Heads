#!/usr/bin/env python3
import torch
import math
import argparse
import numpy as np
from collections import defaultdict
import itertools
from transformers import AutoModelForCausalLM, AutoConfig
import sys # Added for sys.exit
import matplotlib.pyplot as plt # Added for plotting
import networkx as nx # Added for GML parsing
import re # Added for parsing GML labels

# --- Copied and potentially modified functions from principle_angle.py ---

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
        # Perform QR on the device of A and B
        QA, _ = torch.linalg.qr(A, mode='reduced')    # (m, k1)
        QB, _ = torch.linalg.qr(B, mode='reduced')    # (m, k2)
    except torch.linalg.LinAlgError as e:
        # print(f"QR decomposition failed: {e}")
        # print(f"Input shapes: A={A.shape}, B={B.shape}, Device={A.device}")
        return float('nan'), float('nan'), float('nan'), 0

    # 2) cross‐inner product
    M = QA.T @ QB                                 # (k1, k2)

    # 3) canonical correlations (singular values)
    try:
        # Ensure M is float for SVD, perform on the device of M
        sigma = torch.linalg.svdvals(M.float())           # (min(k1, k2),)
    except torch.linalg.LinAlgError as e:
        #  print(f"SVD failed: {e}")
        #  print(f"M shape: {M.shape}, M dtype: {M.dtype}, Device={M.device}")
         return float('nan'), float('nan'), float('nan'), 0

    # Clamp sigma values to be within [0, 1] to avoid domain errors in acos
    sigma = sigma.clamp(0.0, 1.0)

    # 4) principal angles (theta)
    # Add a small epsilon to avoid acos(1.0) numerical issues if any sigma is exactly 1
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


def load_llm(model_name_or_path, revision):
    """Loads a Hugging Face causal LM, initializing randomly if revision is 'random'."""
    print(f"Loading config for: {model_name_or_path}")
    try:
        # Load config first regardless of revision
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if revision == "random":
            print(f"Initializing model {model_name_or_path} with random weights.")
            # Initialize model from config (random weights)
            model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch.bfloat16, # Still use bfloat16 if desired
                trust_remote_code=True
            )
            # Manually move to device as from_config doesn't have device_map
            # Try CUDA, fall back to CPU if not available or if model is too large for single GPU
            # (Note: 'auto' device_map is more complex with from_config)
            try:
                 model.to('cuda')
                 print("Random model placed on CUDA.")
            except Exception as e:
                 print(f"Could not move random model to CUDA ({e}), using CPU.")
                 model.to('cpu')

        else:
            print(f"Loading pre-trained model: {model_name_or_path}, Revision: {revision}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if available
                device_map='auto', # Automatically distribute across GPUs if available
                trust_remote_code=True,
                revision=revision,
                cache_dir = '../.cache' # Assumes cache relative to workspace root
            )
            # device_map='auto' handles moving to device for pretrained
            print("Pre-trained model loaded.")

        model.eval() # Set to evaluation mode
        print("Model ready for evaluation.")
        return model, config

    except Exception as e:
        print(f"Error during model loading/initialization for {model_name_or_path} (Revision: {revision}): {e}")
        return None, None

def extract_head_k_projections(model, config, layer_idx):
    """Extracts K-projection weights for each head in a given layer."""
    try:
        # Accessing weights - adjust path if necessary for the specific model
        # Common paths: model.model.layers, model.transformer.h, etc.
        layer = model.model.layers[layer_idx]
        # Determine the correct attribute name for K-projection weights
        if hasattr(layer.self_attn, 'k_proj') and layer.self_attn.k_proj is not None:
            k_proj_weight = layer.self_attn.k_proj.weight
        # Add other potential attribute names if needed for different architectures
        # elif hasattr(layer.self_attn, 'key_proj'):
        #     k_proj_weight = layer.self_attn.key_proj.weight
        else:
            print(f"Error: Could not find K-projection weight attribute in layer {layer_idx}. Tried 'k_proj'.")
            return None, 0, 0

        hidden_size = config.hidden_size
        # Use num_key_value_heads if available (GQA/MQA), otherwise num_attention_heads
        num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = hidden_size // config.num_attention_heads # Q, K, V head dim often derived from total attention heads

        # Verify expected shape: (num_kv_heads * head_dim, hidden_size)
        expected_shape = (num_heads * head_dim, hidden_size)
        if k_proj_weight.shape != expected_shape:
             # Attempt to handle shape mismatch (e.g., fused QKV) - needs careful checking
             print(f"Warning: Layer {layer_idx} K-proj weight shape mismatch. Got {k_proj_weight.shape}, expected {expected_shape}. Attempting to proceed.")
             # Infer num_heads * head_dim from the first dimension if possible
             actual_out_dim = k_proj_weight.shape[0]
             if k_proj_weight.shape[1] == hidden_size:
                 if actual_out_dim % head_dim == 0:
                     inferred_heads = actual_out_dim // head_dim
                     if inferred_heads != num_heads:
                         print(f"Adjusting number of heads for layer {layer_idx} based on weight shape from {num_heads} to {inferred_heads}")
                         num_heads = inferred_heads
                     # else shape matches expectation after adjustment check
                 else:
                     print(f"Error: Cannot reconcile K-proj weight shape {k_proj_weight.shape} with hidden_size {hidden_size} and head_dim {head_dim}")
                     return None, 0, 0
             else:
                  print(f"Error: K-proj weight input dimension {k_proj_weight.shape[1]} does not match hidden_size {hidden_size}")
                  return None, 0, 0


        # Transpose to (hidden_size, num_heads * head_dim) for splitting
        k_proj_transposed = k_proj_weight.t().contiguous() # Use .t() for 2D transpose

        # Split into list of head tensors: List[(hidden_size, head_dim)]
        # Ensure split happens correctly even if num_heads was adjusted
        actual_split_dim = k_proj_transposed.shape[1]
        if actual_split_dim == num_heads * head_dim:
             head_k_projections = list(torch.split(k_proj_transposed, head_dim, dim=1))
        else:
             print(f"Error: Transposed K-proj dim {actual_split_dim} doesn't match inferred {num_heads} * {head_dim}")
             return None, 0, 0

        # Verify number of heads extracted matches num_heads (potentially adjusted)
        if len(head_k_projections) != num_heads:
             print(f"Critical Error: Layer {layer_idx} - Extracted {len(head_k_projections)} heads, expected {num_heads} after potential adjustments.")
             return None, 0, 0

        # print(f"Layer {layer_idx}: Extracted {len(head_k_projections)} K-projection heads, each shape {head_k_projections[0].shape}")
        return head_k_projections, num_heads, head_dim

    except AttributeError as e:
        print(f"Error accessing K-projection weights for layer {layer_idx}: {e}. Check model structure (e.g., model.model.layers[...].self_attn.k_proj).")
        return None, 0, 0
    except Exception as e:
        print(f"Unexpected error extracting weights for layer {layer_idx}: {e}")
        return None, 0, 0


# --- Main logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Calculate best principal angles between LLM head K-projections within each layer."
    )
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., 'meta-llama/Meta-Llama-3.1-8B').")
    # Allow multiple revisions
    parser.add_argument("--revisions", nargs='+', default=["main"], help="Space-separated list of Hugging Face model revisions (default: main).")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for dimension overlap calculation (default: 1e-6).")
    parser.add_argument("--gml_file", type=str, required=True, help="Path to the GML file containing head connections and r2 weights.")
    parser.add_argument("--num_gml_partners", type=int, default=5, help="Number of top GML partners to select for subspace A (default: 5).")

    args = parser.parse_args()

    # --- Load GML data once ---
    print(f"Loading GML data from: {args.gml_file}")
    gml_connections_data = parse_gml_connections(args.gml_file)
    if not gml_connections_data:
        print("Error: GML data could not be loaded or is empty. Exiting.")
        sys.exit(1)


    # --- Loop over each revision ---
    # Data structures to store results across all revisions for plotting
    processed_revisions = []
    all_revision_overall_averages = defaultdict(list)

    for revision in args.revisions:
        print(f"\n===== Processing Revision: {revision} =====")

        # 1. Load LLM for the current revision
        model, config = load_llm(args.model_id, revision)
        if model is None or config is None:
            print(f"Skipping revision {revision} due to model loading errors.")
            continue # Move to the next revision

        # Determine model parameters
        try:
            num_layers = config.num_hidden_layers
            # Use num_key_value_heads if available (GQA/MQA), default to num_attention_heads
            num_kv_heads_config = getattr(config, 'num_key_value_heads', config.num_attention_heads)
            # Head dim calculation might need adjustment based on specific architecture details
            head_dim_config = config.hidden_size // config.num_attention_heads

            print(f"Model Config: Layers={num_layers}, Config KV Heads={num_kv_heads_config}, Config Head Dim={head_dim_config}")
        except AttributeError as e:
            print(f"Error accessing model configuration parameters for revision {revision}: {e}")
            continue # Move to the next revision


        # 2. Calculate Metrics and Collect Best Results Per Layer for this revision
        # Structure: {metric_name: {layer_idx: [list of best values for this layer]}}
        # Reset metrics for each revision
        layer_best_metrics = defaultdict(lambda: defaultdict(list))

        print(f"Calculating principal angles using GML-defined head groups (Revision: {revision}, Top {args.num_gml_partners} partners)...")
        for layer_idx in range(num_layers):
            # print(f"--- Processing Layer {layer_idx} ---")
            head_k_projections, n_heads_layer, head_dim_layer = extract_head_k_projections(model, config, layer_idx)

            if head_k_projections is None:
                print(f"Skipping layer {layer_idx} (Revision: {revision}) due to weight extraction errors.")
                continue
            if n_heads_layer == 0: # Check if n_heads_layer is 0
                print(f"Skipping layer {layer_idx} (Revision: {revision}): No heads found ({n_heads_layer}).")
                continue


            # Iterate through each head as the 'anchor' (Subspace B)
            for anchor_head_idx in range(n_heads_layer):
                B_proj = head_k_projections[anchor_head_idx] # Shape: (hidden_size, head_dim_layer)

                # Get top K partner head indices from GML data
                top_partner_indices = get_top_k_partners(layer_idx, anchor_head_idx, gml_connections_data, k=args.num_gml_partners)

                actual_partner_k_projections_list = []
                if top_partner_indices: # Ensure there are potential partners from GML
                    for p_idx in top_partner_indices:
                        if 0 <= p_idx < n_heads_layer:
                            # GML parsing should already ensure partner_idx != anchor_head_idx
                            actual_partner_k_projections_list.append(head_k_projections[p_idx])
                        else:
                            print(f"Warning: Layer {layer_idx}, Anchor {anchor_head_idx}: GML partner index {p_idx} is out of bounds for {n_heads_layer} actual heads. Ignoring this partner.")
                
                # Check if we have the required number of valid partners
                if len(actual_partner_k_projections_list) < args.num_gml_partners:
                    # print(f"Layer {layer_idx} Head {anchor_head_idx}: Found only {len(actual_partner_k_projections_list)} valid GML partners, require {args.num_gml_partners}. Skipping.")
                    continue
                
                # Take exactly num_gml_partners if more were somehow collected (should not happen with get_top_k_partners)
                A_proj_list = actual_partner_k_projections_list[:args.num_gml_partners]

                # Stack the K-projections of the selected partners to form Subspace A
                # Each is (hidden_size, head_dim_layer)
                # A_proj will be (hidden_size, num_gml_partners * head_dim_layer)
                A_proj = torch.cat(A_proj_list, dim=1)


                # Calculate principal angles
                mean_angle, max_angle, sum_sines, dim_ov = principal_angles_with_overlap(A_proj, B_proj, tol=args.tol)

                if any(math.isnan(m) for m in [mean_angle, max_angle, sum_sines]):
                    # print(f"Warning: NaN result for L{layer_idx} H_anchor{anchor_head_idx} vs its top {args.num_gml_partners} GML partners. Skipping.")
                    continue

                # Store the metrics for this anchor head's comparison
                layer_best_metrics["mean_angle"][layer_idx].append(mean_angle)
                layer_best_metrics["max_angle"][layer_idx].append(max_angle)
                layer_best_metrics["sum_sines"][layer_idx].append(sum_sines)
                layer_best_metrics["dim_overlap"][layer_idx].append(dim_ov)


        # 3. Calculate Averages, Calculate Overall Averages, and Save Results for this revision
        print(f"--- Results for Revision: {revision} ---")

        # Dictionary to store layer averages for this revision
        revision_layer_averages = defaultdict(dict)
        # Dictionary to store overall averages for this revision
        revision_overall_averages = {}
        metric_names = ["mean_angle", "max_angle", "sum_sines", "dim_overlap"]

        for metric in metric_names:
            # print(f"Metric: {metric}") # Print within file saving later
            layer_averages_for_metric = [] # Collect averages for overall calculation
            if not layer_best_metrics[metric]:
                # print("  No data collected for this metric.")
                revision_layer_averages[metric] = {} # Ensure key exists even if empty
                continue

            for layer_idx in sorted(layer_best_metrics[metric].keys()):
                best_values_for_layer = layer_best_metrics[metric][layer_idx]
                if best_values_for_layer:
                    # Filter out potential inf/-1 initial values
                    valid_values = [v for v in best_values_for_layer if not (math.isinf(v) or v == -1)]
                    if valid_values:
                        average_best = np.mean(valid_values)
                        revision_layer_averages[metric][layer_idx] = average_best
                        layer_averages_for_metric.append(average_best)
                        # print(f"  Layer {layer_idx:<3}: Avg Best = {average_best:>8.4f} (from {len(valid_values)} heads)") # Print within file saving
                    else:
                        # print(f"  Layer {layer_idx:<3}: No valid best values recorded.")
                        revision_layer_averages[metric][layer_idx] = None # Indicate no valid data
                else:
                     # print(f"  Layer {layer_idx:<3}: No best values recorded.")
                     revision_layer_averages[metric][layer_idx] = None # Indicate no data

            # Calculate overall average for this metric for this revision
            if layer_averages_for_metric:
                revision_overall_averages[metric] = np.mean(layer_averages_for_metric)
            else:
                revision_overall_averages[metric] = None # Indicate no overall average possible


        # --- Save results to file ---
        # Clean revision name for filename (replace slashes etc.)
        safe_revision_name = revision.replace('/', '_').replace('\\', '_')
        output_filename = f"{safe_revision_name}_metrics.txt"
        try:
            with open(output_filename, 'w') as f:
                f.write(f"Results for Model: {args.model_id}, Revision: {revision}\n")
                f.write("=" * 40 + "\n\n")

                f.write("--- Average Best Intra-Layer Metrics ---\n")
                for metric in metric_names:
                    f.write(f"Metric: {metric}\n")
                    if not revision_layer_averages[metric]:
                        f.write("  No data collected for this metric.\n")
                    else:
                        for layer_idx in sorted(revision_layer_averages[metric].keys()):
                            avg_val = revision_layer_averages[metric][layer_idx]
                            if avg_val is not None:
                                # Get count for reporting (re-calculate for clarity, could optimize later)
                                valid_values_count = len([v for v in layer_best_metrics[metric][layer_idx] if not (math.isinf(v) or v == -1)])
                                f.write(f"  Layer {layer_idx:<3}: Avg Best = {avg_val:>8.4f} (from {valid_values_count} heads)\n")
                            else:
                                # Check if the layer existed but had no valid values vs no data at all
                                if layer_idx in layer_best_metrics[metric]:
                                     f.write(f"  Layer {layer_idx:<3}: No valid best values recorded.\n")
                                else:
                                     f.write(f"  Layer {layer_idx:<3}: No best values recorded.\n")
                    f.write("\n") # Add space between metrics

                f.write("--- Overall Average Metrics (across layers) ---\n")
                for metric in metric_names:
                    overall_avg = revision_overall_averages.get(metric)
                    if overall_avg is not None:
                         f.write(f"  {metric}: {overall_avg:>8.4f}\n")
                    else:
                         f.write(f"  {metric}: N/A (no valid layer data)\n")
                f.write("\n")

            print(f"Results for revision '{revision}' saved to '{output_filename}'")

        except IOError as e:
            print(f"Error writing results for revision {revision} to file {output_filename}: {e}")

        # Optional: Clean up model from memory if needed, especially for large models
        del model
        del config
        torch.cuda.empty_cache() # Clear GPU cache between revisions

        # Store results for plotting
        processed_revisions.append(revision) # Store the name of the revision processed
        for metric in metric_names:
             # Store the overall average for this revision, use np.nan if it was None
             overall_avg = revision_overall_averages.get(metric)
             all_revision_overall_averages[metric].append(overall_avg if overall_avg is not None else np.nan)

    print("\n===== All revisions processed. =====")

    # --- Generate Plot ---
    if processed_revisions:
        print("Generating plot...")
        plot_filename = "overall_metrics_comparison.png"
        try:
            plt.figure(figsize=(12, 8))
            metric_names = list(all_revision_overall_averages.keys())

            for metric in metric_names:
                # Plot only if there's data for this metric
                if metric in all_revision_overall_averages:
                    averages = all_revision_overall_averages[metric]
                    # Plotting points where data exists (non-NaN)
                    plt.plot(processed_revisions, averages, marker='o', linestyle='-', label=metric)

            plt.xlabel("Revision")
            plt.ylabel("Overall Average Value")
            plt.title(f"Overall Average Metrics per Revision (Model: {args.model_id})")
            plt.xticks(rotation=45, ha='right') # Rotate labels if many revisions
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout() # Adjust layout
            plt.savefig(plot_filename)
            print(f"Plot saved to '{plot_filename}'")
        except Exception as e:
            print(f"Error generating plot: {e}")
    else:
        print("No revisions were processed successfully, skipping plot generation.")


    print("Analysis complete.")
    return 0


# --- GML Helper Functions ---
def parse_gml_connections(gml_file_path: str):
    """
    Parses a GML file to extract intra-layer head connections based on r2 metric.

    Args:
        gml_file_path: Path to the GML file.

    Returns:
        A dictionary: layer_connections[layer_idx][source_head_idx] = sorted_list_of[(target_head_idx, r2_weight)]
        Returns None if parsing fails or GML is not structured as expected.
    """
    try:
        graph = nx.read_gml(gml_file_path, label='id') # Use 'id' attribute from GML as node ID in networkx
    except Exception as e:
        print(f"Error reading GML file {gml_file_path}: {e}")
        return None

    node_id_to_layer_head = {} # Map GML node ID to (layer, head) tuple
    for node_id, data in graph.nodes(data=True):
        label = data.get('label')
        if label:
            match = re.match(r'\((\d+),(\d+)\)', label)
            if match:
                layer = int(match.group(1))
                head = int(match.group(2))
                node_id_to_layer_head[str(node_id)] = (layer, head) # GML node IDs might be strings even if numeric
            else:
                print(f"Warning: Could not parse GML label '{label}' for node ID {node_id}. Skipping this node.")
        else:
            print(f"Warning: GML Node ID {node_id} has no label. Skipping this node.")

    # Structure: layer_connections[layer_idx][source_head_idx] = list_of[(target_head_idx, r2_weight), ...]
    layer_head_connections = defaultdict(lambda: defaultdict(list))

    for source_node_id_nx, target_node_id_nx, edge_data in graph.edges(data=True):
        # networkx might use the original GML IDs or its own internal IDs depending on how graph was read.
        # Ensure we use string versions if node_id_to_layer_head keys are strings.
        source_node_id_str = str(source_node_id_nx)
        target_node_id_str = str(target_node_id_nx)

        if edge_data.get('metric') == 'r2':
            r2_weight = edge_data.get('weight')
            if r2_weight is None:
                # print(f"Warning: Edge {source_node_id_str}-{target_node_id_str} has metric 'r2' but no weight. Skipping.")
                continue

            source_info = node_id_to_layer_head.get(source_node_id_str)
            target_info = node_id_to_layer_head.get(target_node_id_str)

            if source_info and target_info:
                source_layer, source_head = source_info
                target_layer, target_head = target_info

                if source_layer == target_layer:  # Intra-layer connections
                    if source_head != target_head: # Do not connect a head to itself as a "partner"
                        layer_head_connections[source_layer][source_head].append((target_head, float(r2_weight)))
            # else:
                # print(f"Warning: Could not find layer/head info for edge between {source_node_id_str} and {target_node_id_str}. Skipping edge.")

    # Sort connections by r2_weight in descending order
    for layer_idx in layer_head_connections:
        for source_head_idx in layer_head_connections[layer_idx]:
            layer_head_connections[layer_idx][source_head_idx].sort(key=lambda x: x[1], reverse=True)

    if not layer_head_connections:
        print("Warning: No valid intra-layer 'r2' connections found in GML data.")
    return layer_head_connections


def get_top_k_partners(layer_idx: int, anchor_head_idx: int, gml_connections: dict, k: int = 5):
    """
    Retrieves the top k partner head indices for a given anchor head from GML data.

    Args:
        layer_idx: The layer index of the anchor head.
        anchor_head_idx: The index of the anchor head.
        gml_connections: Parsed GML connection data from parse_gml_connections.
        k: The number of top partners to retrieve.

    Returns:
        A list of up to k partner head indices.
    """
    if not gml_connections:
        return []
    
    connections_for_anchor = gml_connections.get(layer_idx, {}).get(anchor_head_idx, [])
    
    # Extract just the head indices from the (target_head_idx, r2_score) tuples
    partner_indices = [conn[0] for conn in connections_for_anchor]
    
    return partner_indices[:k]


if __name__ == "__main__":
    sys.exit(main()) 