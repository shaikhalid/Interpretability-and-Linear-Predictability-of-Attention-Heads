import networkx as nx
import json
import logging
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import math # Import math for checking float values if needed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
parser = argparse.ArgumentParser(description='Find reference heads in a graph.')
parser.add_argument('--graph_file_path', type=str, default='gmls/thought_graph_humaneval_key.gml', help='Input graph file')
parser.add_argument('--output_json_path', type=str, default='ref_target_heads.json', help='Output file for results')
parser.add_argument('--output_plot_file', type=str, default='ref_target_graph.png', help='Output file for visualization')
parser.add_argument('--min_in_degree_for_target', type=int, default=4, help='Minimum incoming connections to be considered a target (must be > 2)')
parser.add_argument('--min_refs_for_valid_target', type=int, default=4, help='Minimum reference heads required to validate a target')
parser.add_argument('--max_refs_per_target', type=int, default=6, help='Maximum reference heads to associate with a single target (set to None for no limit)')
parser.add_argument('--max_target_heads', type=int, default=10000, help='Maximum number of target heads allowed after promotion')
parser.add_argument('--target_num_heads', type=float, default=None, help='Target fraction of heads (e.g., 0.5 for 50%%). Defaults to 0.5 if not specified.')
args = parser.parse_args()

GRAPH_FILE_PATH = args.graph_file_path
OUTPUT_JSON_PATH = args.output_json_path
OUTPUT_PLOT_FILE = args.output_plot_file
MIN_IN_DEGREE_FOR_TARGET = args.min_in_degree_for_target
MIN_REFS_FOR_VALID_TARGET = args.min_refs_for_valid_target
MAX_REFS_PER_TARGET = args.max_refs_per_target
MAX_TARGET_HEADS = args.max_target_heads
TARGET_NUM_HEADS = args.target_num_heads

# --- Functions ---

def load_graph(filepath):
    """Loads a graph from a GML file."""
    logging.info(f"Loading graph from {filepath}...")
    try:
        graph = nx.read_gml(filepath)
        # Convert node labels back to tuples if they were stored as strings
        # GML standard doesn't directly support tuples as node labels, NetworkX might convert them
        relabelled_nodes = {}
        needs_relabel = False
        for node in graph.nodes():
            if isinstance(node, str):
                 try:
                     # Attempt to evaluate the string as a tuple literal (e.g., "(1, 5)")
                     potential_tuple = eval(node)
                     if isinstance(potential_tuple, tuple) and len(potential_tuple) == 2 and all(isinstance(i, int) for i in potential_tuple):
                         relabelled_nodes[node] = potential_tuple
                         needs_relabel = True
                     else:
                         logging.warning(f"Node label '{node}' is a string but not a valid (int, int) tuple string. Keeping as string.")
                         relabelled_nodes[node] = node # Keep as is if conversion fails
                 except Exception as e: # Catch broader exceptions during eval
                      logging.warning(f"Node label '{node}' could not be parsed as a tuple due to error: {e}. Keeping as string.")
                      relabelled_nodes[node] = node # Keep as is if conversion fails
            elif isinstance(node, (int, float)) or not (isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], int) and isinstance(node[1], int)):
                 # Handle cases where nodes might be single numbers or invalid tuples directly
                 logging.warning(f"Node label '{node}' is not in the expected (int, int) tuple format. Attempting conversion or skipping.")
                 # Optionally try to convert if it makes sense, or just mark for removal/error
                 # For now, let's keep it and let the validation step catch it.
                 relabelled_nodes[node] = node
            else:
                 relabelled_nodes[node] = node # Keep valid tuple nodes as they are

        if needs_relabel:
             graph = nx.relabel_nodes(graph, relabelled_nodes, copy=True)
             logging.info("Relabelled string node labels to tuples where possible.")

        # Ensure all nodes are actually tuples after potential relabelling
        invalid_nodes = [n for n in graph.nodes() if not (isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], int) and isinstance(n[1], int))]
        if invalid_nodes:
             logging.error(f"Graph contains nodes that are not in (Layer, Head) tuple format after loading/relabeling: {invalid_nodes[:5]}...")
             # Option 1: Remove invalid nodes
             # graph.remove_nodes_from(invalid_nodes)
             # logging.warning(f"Removed {len(invalid_nodes)} invalid nodes.")
             # Option 2: Raise an error
             raise ValueError("Graph contains invalid node formats. Cannot proceed.")


        logging.info(f"Graph loaded successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph
    except FileNotFoundError:
        logging.error(f"Graph file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading or processing graph: {e}")
        return None


def find_ref_target_heads(graph):
    """
    Simplified algorithm for selecting reference and target heads in a directed graph.
    1. Identify candidate targets: nodes with in-degree >= MIN_IN_DEGREE_FOR_TARGET.
    2. Greedily build a set S of up to MAX_TARGET_HEADS targets, ensuring all constraints are met at each step.
    3. Reference heads are all nodes not in S with out-degree > 0.
    4. For each target in S, assign its reference heads as its predecessors not in S, capped at MAX_REFS_PER_TARGET if specified.
    Args:
        graph (nx.DiGraph): The thought graph with nodes as (Layer, Head) tuples.
    Returns:
        tuple: (results_dict, set of reference heads, set of target heads)
    """
    if not graph or graph.number_of_nodes() == 0:
        logging.warning("Graph is empty or None. Cannot find heads.")
        return {}, set(), set()

    nodes = list(graph.nodes())
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())

    # 1. Candidate targets: nodes with in-degree >= MIN_IN_DEGREE_FOR_TARGET
    candidate_targets = {n for n in nodes if in_degrees.get(n, 0) >= MIN_IN_DEGREE_FOR_TARGET}
    S = set()  # Selected targets

    def num_refs(node, current_S):
        # Number of predecessors of node that are NOT in S
        return len([p for p in graph.predecessors(node) if p not in current_S])

    def num_succ_in_S(node, current_S):
        # Number of successors of node that are in S
        return len([s for s in graph.successors(node) if s in current_S])

    # 2. Greedy selection loop
    while len(S) < MAX_TARGET_HEADS:
        candidates = []
        # Filter candidates ensuring validity constraints and collect layer info
        for c in candidate_targets - S:
            # Check if the candidate itself would be valid if added
            c_num_refs = num_refs(c, S)
            if c_num_refs < MIN_REFS_FOR_VALID_TARGET:
                continue  # Skip if adding this candidate makes it invalid

            # Check if adding this candidate invalidates any existing targets in S
            valid = True
            for t in S:
                # Check refs for existing target t if c were added to S
                if num_refs(t, S | {c}) < MIN_REFS_FOR_VALID_TARGET:
                    valid = False
                    break
            
            if valid:
                # Store (layer, num_succ_in_S, candidate) for sorting
                # Nodes are assumed to be (layer, head) tuples
                layer = c[0] if isinstance(c, tuple) and len(c) == 2 else -1 # Default layer if format unexpected
                candidates.append((layer, num_succ_in_S(c, S), c))

        if not candidates:
            break  # No valid candidate to add

        # Sort candidates: prioritize higher layers (descending), then minimal successors in S (ascending)
        candidates.sort(key=lambda x: (-x[0], x[1])) 
        
        # Choose the best candidate based on the new sorting
        chosen_c = candidates[0][2] 
        S.add(chosen_c)

    # 3. Reference heads: all nodes not in S with out-degree > 0
    ref_heads = {n for n in nodes if n not in S and out_degrees.get(n, 0) > 0}

    # 4. For each target, assign its reference heads (predecessors not in S, capped)
    results = {}
    for idx, t in enumerate(sorted(S)):
        refs = [p for p in graph.predecessors(t) if p not in S]
        # Optionally cap the number of refs
        if MAX_REFS_PER_TARGET is not None and len(refs) > MAX_REFS_PER_TARGET:
            # Sort by out-degree ascending (lowest first)
            refs = sorted(refs, key=lambda x: out_degrees.get(x, 0))[:MAX_REFS_PER_TARGET]
        # Use string key in the format 'layer_head' (e.g., '12_3')
        if isinstance(t, tuple) and len(t) == 2:
            layer_head_key = f"{t[0]}_{t[1]}"
        else:
            layer_head_key = str(t)
        results[layer_head_key] = {
            'target_heads': [t],
            'ref_heads': refs
        }

    return results, ref_heads, S

def save_results(results, filepath):
    """Saves the results dictionary to a JSON file."""
    logging.info(f"Saving results to {filepath}...")
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info("Results saved successfully.")
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")

# --- Visualization Function ---
def visualize_graph(graph, ref_heads, target_heads, output_path):
    """Visualizes the thought graph with colored nodes for targets and references."""
    if not graph or not graph.nodes:
        logging.warning("Graph is empty or None, skipping visualization.")
        return

    logging.info(f"Visualizing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges...")
    plt.figure(figsize=(30, 30)) # Increased size further for potentially better layout

    # Position nodes in layers
    pos = {}
    layer_nodes = defaultdict(list)
    all_layers = set()
    all_head_indices = set()

    # Validate nodes during position calculation
    valid_nodes = []
    for node in graph.nodes():
         if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], int) and isinstance(node[1], int):
             layer, head = node
             layer_nodes[layer].append(head)
             all_layers.add(layer)
             all_head_indices.add(head)
             valid_nodes.append(node)
         else:
             logging.warning(f"Skipping node {node} during visualization positioning due to invalid format.")

    if not valid_nodes:
        logging.error("No valid nodes found for visualization.")
        plt.close()
        return

    max_layer = max(all_layers) if all_layers else 0
    # Ensure layers are contiguous for x-axis ticks if needed, find min/max
    min_layer = min(all_layers) if all_layers else 0
    sorted_head_indices = sorted(list(all_head_indices))
    max_head = max(all_head_indices) if all_head_indices else 0

    # Calculate positions for valid nodes
    for layer in range(min_layer, max_layer + 1):
        # Sort heads within the layer for consistent vertical placement
        heads_in_layer = sorted(list(set(layer_nodes[layer]))) # Use set to ensure unique heads
        for head_index, head in enumerate(heads_in_layer):
            node = (layer, head)
            # Position based on layer and normalized head index within the layer
            # Use actual head index for y-position to maintain vertical alignment across layers
            pos[node] = (layer, -head) # Negative to put lower head indices (0) at the top

    # Determine node colors based on the *final* classification for VALID nodes
    node_colors = []
    nodes_to_draw = []
    node_labels = {}
    for node in graph.nodes():
        if node in pos: # Only consider nodes for which we calculated a position
            nodes_to_draw.append(node)
            if node in target_heads: # Final targets
                node_colors.append('lightgreen')
            elif node in ref_heads: # Final references
                node_colors.append('orange')
            else: # Final others (neither ref nor target)
                node_colors.append('skyblue')
            node_labels[node] = f"{node[0]},{node[1]}" # Format labels as L,H

    if not nodes_to_draw:
         logging.error("No nodes to draw after position calculation and validation.")
         plt.close()
         return

    # Improve visualization options
    options = {
        "font_size": 8,
        "node_size": 400, # Slightly smaller nodes if graph is dense
        "node_color": node_colors, # Use the list of colors
        "edge_color": "gray",
        "linewidths": 1,
        "width": 0.4, # Thinner edges
        "with_labels": True,
        "labels": node_labels,
        "nodelist": nodes_to_draw, # Explicitly provide the list of nodes to draw
        "arrows": True,
        "arrowstyle": "-|>",
        "arrowsize": 7, # Slightly smaller arrows
    }

    try:
        # Draw only the edges connected to the nodes we are drawing
        edges_to_draw = [(u, v) for u, v in graph.edges() if u in pos and v in pos]
        nx.draw(graph, pos, edgelist=edges_to_draw, **options)
    except Exception as e:
        logging.error(f"Error during networkx drawing: {e}")
        plt.close()
        return


    # Get the unique head indices for y-ticks
    y_positions = sorted([-h for h in all_head_indices], reverse=True) # Sort descending numerically (-0 > -1 > -2)
    y_labels = sorted([h for h in all_head_indices]) # Corresponding ascending labels (0, 1, 2)

    plt.title("Reference/Target Head Identification Graph", fontsize=16, pad=20)
    plt.xlabel("Layer Index", fontsize=14, labelpad=15)
    plt.ylabel("Head Index", fontsize=14, labelpad=15)
    # Adjust x-ticks to cover the actual range of layers present
    plt.xticks(range(min_layer, max_layer + 1), fontsize=10)
    # Adjust y-ticks if there are many heads to avoid clutter
    if len(y_positions) > 40: # Example threshold
        tick_step = math.ceil(len(y_positions) / 20) # Show approx 20 labels
        plt.yticks(y_positions[::tick_step], y_labels[::tick_step], fontsize=9)
    else:
        plt.yticks(y_positions, y_labels, fontsize=9)

    plt.grid(axis='both', linestyle='--', alpha=0.6)
    # plt.gca().invert_yaxis() # Already handled by using negative head index in pos

    # Add a legend for node colors reflecting the final classification
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Final Target ({len(target_heads)})', markersize=10, markerfacecolor='lightgreen'),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Final Reference ({len(ref_heads)})', markersize=10, markerfacecolor='orange'),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Other (Final)', markersize=10, markerfacecolor='skyblue') # Calculate 'other' count if needed
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05))


    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=300) # Higher DPI
        logging.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save graph visualization: {e}")
    finally:
        plt.close() # Close the plot figure


# --- Main Execution ---

def main():
    initial_graph = load_graph(GRAPH_FILE_PATH)
    if initial_graph is None:
        return

    total_heads = initial_graph.number_of_nodes()
    if total_heads == 0:
        logging.warning("Graph is empty. Cannot proceed with threshold search.")
        return

    default_target_fraction = 0.5
    target_num_target_heads = total_heads * default_target_fraction 

    if TARGET_NUM_HEADS is not None:
        if 0.0 < TARGET_NUM_HEADS <= 1.0:
            target_num_target_heads = total_heads * TARGET_NUM_HEADS
            logging.info(f"Using specified target fraction of heads: {TARGET_NUM_HEADS:.2f}, resulting in approx {target_num_target_heads:.2f} target heads.")
        else:
            logging.warning(f"Specified target_num_heads fraction ({TARGET_NUM_HEADS}) is invalid (must be > 0.0 and <= 1.0). Defaulting to {default_target_fraction*100}%% of total_heads.")
    
    logging.info(f"Aiming for approximately {target_num_target_heads:.2f} target heads (Total heads: {total_heads}).")

    low_threshold = 0.0
    high_threshold = 1.0
    best_threshold = 0.5 # Default if search fails
    final_results = {}
    final_ref_heads_set = set()
    final_target_heads_set = set()
    graph_for_visualization = initial_graph.copy() # Keep a copy for final visualization

    max_iterations = 30 # Max iterations for binary search
    found_suitable_threshold = False

    for iteration in range(max_iterations):
        current_threshold = (low_threshold + high_threshold) / 2
        logging.info(f"Iteration {iteration + 1}/{max_iterations}: Testing threshold = {current_threshold:.4f}")

        # Create a working copy of the graph for this iteration
        current_graph = initial_graph.copy()

        # Filter out edges below the current_threshold before processing
        edges_to_remove = [(u, v) for u, v, d in current_graph.edges(data=True) if d.get('weight', 0) < current_threshold]
        if edges_to_remove:
            # logging.info(f"Removing {len(edges_to_remove)} edges with weight < {current_threshold:.4f}")
            current_graph.remove_edges_from(edges_to_remove)

        results, ref_heads, target_heads = find_ref_target_heads(current_graph)
        num_target_heads = len(target_heads)
        logging.info(f"Threshold {current_threshold:.4f} -> Found {num_target_heads} target heads.")

        # Check if the number of target heads is within the desired range
        if abs(num_target_heads - target_num_target_heads) <= 1:
            logging.info(f"Found suitable threshold {current_threshold:.4f} with {num_target_heads} target heads.")
            best_threshold = current_threshold
            final_results = results
            final_ref_heads_set = ref_heads
            final_target_heads_set = target_heads
            graph_for_visualization = current_graph.copy() # Save this graph state
            found_suitable_threshold = True
            break
        elif num_target_heads < target_num_target_heads:
            # Too few target heads, means threshold is too high (too many edges removed)
            # Lower the threshold to include more edges
            high_threshold = current_threshold
        else:
            # Too many target heads, means threshold is too low (too few edges removed)
            # Increase the threshold to remove more edges
            low_threshold = current_threshold

        if (high_threshold - low_threshold) < 1e-4: # Convergence criterion
            logging.info(f"Binary search converged. Using last valid or best threshold {best_threshold:.4f}.")
            if not found_suitable_threshold: # If no exact match, use the one that was closest or last tested that produced results
                 best_threshold = current_threshold # Or some other logic to pick best_threshold
                 final_results = results
                 final_ref_heads_set = ref_heads
                 final_target_heads_set = target_heads
                 graph_for_visualization = current_graph.copy()
            break
    else: # Max iterations reached
        logging.warning(f"Max iterations reached for binary search. Using best found threshold: {best_threshold:.4f}")
        if not found_suitable_threshold and not final_results: # If loop finished without finding anything
            # Fallback: re-run with the best_threshold if results weren't stored from the last iteration
            logging.info(f"Re-running with best_threshold {best_threshold:.4f} to get final sets.")
            current_graph = initial_graph.copy()
            edges_to_remove = [(u, v) for u, v, d in current_graph.edges(data=True) if d.get('weight', 0) < best_threshold]
            if edges_to_remove:
                current_graph.remove_edges_from(edges_to_remove)
            final_results, final_ref_heads_set, final_target_heads_set = find_ref_target_heads(current_graph)
            graph_for_visualization = current_graph.copy()


    if not final_results:
        logging.warning("No valid reference/target head pairs found after binary search.")
    else:
        logging.info(f"Using threshold {best_threshold:.4f}. Identified {len(final_results)} final target head configurations.")

    save_results(final_results, OUTPUT_JSON_PATH)

    # Calculate the number of heads actually used as references for the final targets
    used_ref_heads_in_final = set()
    for data in final_results.values():
        if 'ref_heads' in data:
             used_ref_heads_in_final.update(data['ref_heads'])
        else:
            logging.warning(f"Target configuration {data.get('target_heads', 'N/A')} in final results is missing 'ref_heads'.")

    # Ensure the final sets are based on graph nodes that exist
    valid_graph_nodes = set(graph_for_visualization.nodes()) # Use the graph state corresponding to best_threshold
    final_ref_heads_set = final_ref_heads_set.intersection(valid_graph_nodes)
    final_target_heads_set = final_target_heads_set.intersection(valid_graph_nodes)

    # Final count of nodes that are neither ref nor target
    final_other_heads = valid_graph_nodes - final_ref_heads_set - final_target_heads_set
    other_heads_count = len(final_other_heads)

    logging.info(f"Summary (Final with threshold {best_threshold:.4f}): Found {len(final_ref_heads_set)} reference heads, "
                 f"{len(final_target_heads_set)} target heads, "
                 f"and {other_heads_count} other heads.")
    # Check intersection against the *final* validated reference heads
    used_and_final_refs = used_ref_heads_in_final.intersection(final_ref_heads_set)
    logging.info(f"Out of the final reference heads, {len(used_and_final_refs)} were used to connect to final targets.")
    logging.info(f"Total final target configurations saved: {len(final_results)}")

    # Visualize the graph with final colored nodes
    visualize_graph(graph_for_visualization, final_ref_heads_set, final_target_heads_set, OUTPUT_PLOT_FILE)


if __name__ == "__main__":
    main()