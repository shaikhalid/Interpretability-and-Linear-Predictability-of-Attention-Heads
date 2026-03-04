import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import re
import unicodedata # Import unicodedata
import math

# Helper function (similar to gml_to_tikz.py) to extract layer index
def get_node_layer_from_data(node_id: Any, node_data: Dict[str, Any], attr_name: str = 'layer') -> int:
    """
    Extracts the layer index from node data, trying attribute first, then parsing label.

    Args:
        node_id: The identifier of the node (used for error messages).
        node_data: The dictionary of attributes for the node.
        attr_name: The attribute name to check for the layer index.

    Returns:
        The integer layer index.

    Raises:
        ValueError: If the layer cannot be determined.
    """
    # node_id = node_data.get('id', 'UNKNOWN') # REMOVED - node_id is now passed directly

    # 1. Try the specified layer attribute first
    layer = node_data.get(attr_name)

    # 2. If attribute is missing, try parsing the label `"(L, N)"`
    if layer is None:
        label = node_data.get('label')
        if label:
            label = label.strip() # Strip leading/trailing whitespace
            try:
                # Normalize the string to handle potential hidden characters
                normalized_label = unicodedata.normalize('NFKC', label)
            except TypeError:
                normalized_label = label

            # --- Parse using string manipulation instead of regex ---
            if normalized_label.startswith('(') and normalized_label.endswith(')') and ',' in normalized_label:
                # Remove parentheses
                content = normalized_label[1:-1].strip()
                # Split by comma and take first part
                parts = content.split(',')
                try:
                    layer = int(parts[0].strip())
                except (ValueError, IndexError):
                    raise ValueError(f"Node {node_id} has label '{label}' but could not extract layer number from first component.")
            else:
                # Direct checks for special cases
                if normalized_label == '(0,0)':
                    layer = 0
                elif normalized_label == '(0,1)':
                    layer = 0
                else:
                    # Fall back to error if not parseable
                    raise ValueError(f"Node {node_id} is missing '{attr_name}' and its label '{label}' isn't in the expected format '(L, N)'.")
        else:
            raise ValueError(f"Node {node_id} is missing the '{attr_name}' attribute and has no label for inference.")

    # 3. Convert the found layer to int
    try:
        return int(layer)
    except ValueError:
        raise ValueError(f"Node {node_id}'s derived layer value ('{layer}') is not an integer.")

def create_weighted_density_groups(gml_file_path: str, min_group_size: int = 2, max_group_size: int = 6, weight_attr: str = 'weight', layer_attr: str = 'layer') -> List[List[int]]:
    """
    Creates layer groups based on weighted connection density and size constraints.

    Reads a GML file, calculates the sum of edge weights between adjacent layers,
    and iteratively splits the largest groups at the weakest connection points until
    all groups satisfy the minimum and maximum size constraints.

    Args:
        gml_file_path: Path to the input GML file.
        min_group_size: The minimum allowed number of layers in a group (inclusive). Defaults to 2.
        max_group_size: The maximum allowed number of layers in a group (inclusive). Defaults to 6.
        weight_attr: The GML edge attribute containing the weight (e.g., 'weight').
        layer_attr: The GML node attribute containing the layer index (e.g., 'layer').

    Returns:
        A list of lists, where each inner list is [start_layer, end_layer] (inclusive)
        representing a group, sorted by start_layer.

    Raises:
        FileNotFoundError: If the GML file cannot be found.
        ValueError: If the GML data is invalid, layers cannot be determined,
                    or if constraints are impossible to meet.
        nx.NetworkXError: For general GML parsing errors.
    """
    try:
        G = nx.read_gml(gml_file_path, label='id')
        print(f"Successfully loaded GML graph from: {gml_file_path}")
    except FileNotFoundError:
        print(f"Error: GML file not found at {gml_file_path}")
        raise
    except Exception as e:
        print(f"Error reading GML file {gml_file_path}: {e}")
        raise # Re-raise other potential NetworkX errors

    # 1. Group nodes by layer
    nodes_by_layer = defaultdict(list)
    node_layer_map = {} # Store node_id -> layer mapping for quick lookup
    min_layer, max_layer = float('inf'), float('-inf')

    try:
        for node_id, data in G.nodes(data=True):
            layer = get_node_layer_from_data(node_id, data, layer_attr)
            nodes_by_layer[layer].append(node_id)
            node_layer_map[node_id] = layer
            min_layer = min(min_layer, layer)
            max_layer = max(max_layer, layer)

        if min_layer == float('inf'):
            raise ValueError("No nodes found or layers could not be determined.")

        num_layers = max_layer - min_layer + 1
        print(f"Found {num_layers} layers (from {min_layer} to {max_layer}).")

        if min_group_size <= 0:
            raise ValueError("Minimum group size must be positive.")
        if max_group_size < min_group_size:
            raise ValueError("Maximum group size cannot be less than minimum group size.")
        if num_layers < min_group_size:
             print(f"Warning: Total number of layers ({num_layers}) is less than min_group_size ({min_group_size}). Returning a single group.")
             return [[min_layer, max_layer]]

    except ValueError as e:
        print(f"Error processing node layers: {e}")
        raise

    # 2. Calculate inter-layer connection strengths
    inter_layer_strength = {} # Map: layer_index -> strength between layer i and i+1

    for i in range(min_layer, max_layer): # Iterate through potential connection boundaries
        layer_i_nodes = set(nodes_by_layer.get(i, []))
        layer_i_plus_1_nodes = set(nodes_by_layer.get(i + 1, []))

        if not layer_i_nodes or not layer_i_plus_1_nodes:
            inter_layer_strength[i] = 0.0
            continue

        current_strength = 0.0
        for u, v, data in G.edges(data=True):
            u_layer = node_layer_map.get(u)
            v_layer = node_layer_map.get(v)

            if (u_layer == i and v_layer == i + 1) or (u_layer == i + 1 and v_layer == i):
                try:
                    weight = float(data.get(weight_attr, 0.0))
                    current_strength += weight
                except (ValueError, TypeError):
                    print(f"Warning: Invalid weight '{data.get(weight_attr)}' for edge ({u},{v}). Treating as 0.")
                    current_strength += 0.0

        inter_layer_strength[i] = current_strength

    # Ensure we have strengths for all potential splits
    if len(inter_layer_strength) != num_layers - 1:
         print(f"Warning: Expected {num_layers - 1} inter-layer strengths, but calculated {len(inter_layer_strength)}. There might be missing layers or unexpected structure.")
         for i in range(min_layer, max_layer):
             if i not in inter_layer_strength:
                 inter_layer_strength[i] = 0.0

    # 3. Determine optimal splits using dynamic programming to satisfy group size constraints
    n_layers = max_layer - min_layer + 1
    # If the total number of layers is within the max_group_size, return a single group
    if n_layers <= max_group_size:
        print(f"Total number of layers ({n_layers}) is within max_group_size ({max_group_size}); returning single group.")
        return [[min_layer, max_layer]]
    # Build list of inter-layer strengths in ascending layer order
    strength_list = [inter_layer_strength[i] for i in range(min_layer, max_layer)]
    inf = math.inf
    dp = [inf] * n_layers
    prev = [None] * n_layers
    # Dynamic programming to find cut positions minimizing total cut strength
    for j in range(n_layers):
        for k in range(min_group_size, max_group_size + 1):
            start_j = j - k + 1
            if start_j < 0:
                continue
            if start_j == 0:
                cost = 0.0
            else:
                cost = dp[start_j - 1] + strength_list[start_j - 1]
            if cost < dp[j]:
                dp[j] = cost
                prev[j] = start_j - 1
    if dp[n_layers - 1] == inf:
        raise ValueError(f"Could not partition {n_layers} layers into groups of size between {min_group_size} and {max_group_size}.")
    # Reconstruct groups from DP solution
    groups = []
    j = n_layers - 1
    while j >= 0:
        i = prev[j]
        start_offset = 0 if i < 0 else i + 1
        end_offset = j
        groups.append([min_layer + start_offset, min_layer + end_offset])
        j = i
    groups.reverse()

    print(f"Generated {len(groups)} groups.")
    return groups

if __name__ == "__main__":
    gml_path = 'gmls/thought_graph_mmlu_pro_math_key.gml' # <--- Replace with your GML file path
    try:
        # Define constraints
        min_layers_per_group = 3
        max_layers_per_group = 6
        weighting_attribute = 'weight' # or 'R2' or other relevant attribute

        print(f"\nRunning grouping with min_size={min_layers_per_group}, max_size={max_layers_per_group}, weight='{weighting_attribute}'")
        density_groups = create_weighted_density_groups(
            gml_path,
            min_group_size=min_layers_per_group,
            max_group_size=max_layers_per_group,
            weight_attr=weighting_attribute
        )
        # Print all groups on a single line
        print(f"\nFinal Weighted Density Groups: {density_groups}")

        # Optional: Validate group sizes
        valid = True
        for start, end in density_groups:
            size = end - start + 1
            if not (min_layers_per_group <= size <= max_layers_per_group):
                print(f"  Validation Error: Group [{start}, {end}] has size {size}, violating constraints [{min_layers_per_group}, {max_layers_per_group}]")
                valid = False
        if valid:
            print("  Group sizes validated successfully.")

    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
        import traceback
        print(traceback.format_exc())
