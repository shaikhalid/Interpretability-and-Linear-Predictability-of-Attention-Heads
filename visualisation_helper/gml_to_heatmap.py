#!/usr/bin/env python3
# gml_to_heatmap.py
# ==================
#
# Reads a Graph Modeling Language (GML) file, organizes nodes into layers,
# and generates a heatmap where the color intensity of each node position
# is based on the sum of R^2 values (or other specified edge weights)
# associated with it (incoming, outgoing, or all).
#
# Usage:
# ------
# python gml_to_heatmap.py <input_gml_file> [options]
#
# Common Options:
#   -o, --output    : Output image file path (e.g., heatmap.png).
#   -a, --attribute : GML edge attribute for R^2/weight values (default: 'weight').
#   --layer-attr  : GML node attribute for layer index (default: 'layer').
#   --aggregation : How to aggregate weights for a node ('incoming', 'outgoing', 'all').
#                   Default: 'incoming'.
#   --colormap    : Matplotlib colormap for the heatmap (default: 'viridis').
#
# Requirements:
# -------------
# - Python 3
# - NetworkX library (`pip install networkx`)
# - Matplotlib library (`pip install matplotlib`)
# - NumPy library (`pip install numpy`)

import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import re
import math

# Attempt to import a custom plot style, with a fallback.
try:
    # Assuming PLOT_STYLE might be in a file relative to this script's parent.
    # Adjust path if PLOT_STYLE is structured differently.
    # For this example, we'll define a simple fallback directly.
    # from .pastel_theme import PLOT_STYLE # If pastel_theme.py is in the same directory
    PLOT_STYLE = {
        'background_color': 'white',
        'font_family': 'sans-serif',
        'title_fontsize': 14, # No longer used for plot title, but kept for other potential uses
        'label_fontsize': 34, # Changed default label size
        'tick_fontsize': 18, # Changed default tick font size to 22
    }
    # If you have a shared visualisation_helper.pastel_theme, you might use:
    # parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # if parent_dir not in sys.path:
    #     sys.path.insert(0, parent_dir)
    # from visualisation_helper.pastel_theme import PLOT_STYLE

except ImportError:
    print("Warning: Could not import a custom PLOT_STYLE. Using basic default styles.")
    PLOT_STYLE = {
        'background_color': 'white',
        'font_family': 'sans-serif',
        'title_fontsize': 14,
        'label_fontsize': 12,
        'tick_fontsize': 10,
    }

def get_node_layer(node_data_dict, node_id_for_error, attr_name='layer'):
    """
    Determines the layer of a node.
    First, tries the specified layer attribute.
    If missing, tries parsing a label of the format "(L,N)".
    """
    layer = node_data_dict.get(attr_name)

    if layer is None:
        label = node_data_dict.get('label')
        if label:
            # Ensure label is treated as a string for regex
            match = re.match(r'\(\s*(\d+)\s*,\s*\d+\s*\)', str(label))
            if match:
                layer = match.group(1)
            else:
                raise ValueError(
                    f"Node {node_id_for_error} is missing '{attr_name}' and its label '{label}' "
                    f"doesn\'t match pattern '(L, N)'."
                )
        else:
            raise ValueError(
                f"Node {node_id_for_error} is missing the '{attr_name}' attribute and has no label for inference."
            )
    try:
        return int(layer)
    except ValueError:
        raise ValueError(
            f"Node {node_id_for_error}'s derived layer value ('{layer}') is not an integer."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Generate a heatmap of a graph from a GML file based on aggregated R^2/edge weights."
    )
    parser.add_argument("input_gml", help="Input GML file path.")
    parser.add_argument(
        "-o", "--output",
        help="Output image file path (e.g., heatmap.png). Default: <input_basename>_heatmap.png",
        default=None
    )
    parser.add_argument(
        "-a", "--attribute",
        default='weight',
        help="GML edge attribute for R^2/weight values (default: 'weight')."
    )
    parser.add_argument(
        "--layer-attr",
        default='layer',
        help="GML node attribute for layer index (default: 'layer')."
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title for the plot. Default: No title."
    )
    parser.add_argument(
        "--aggregation",
        default='incoming',
        choices=['incoming', 'outgoing', 'all'],
        help="How to aggregate weights for a node: 'incoming' (sum of weights of edges pointing to the node), "
             "'outgoing' (sum of weights of edges from the node), or 'all' (sum of weights of all incident edges). "
             "Default: 'incoming'."
    )
    parser.add_argument(
        "--colormap",
        default='viridis',
        help="Matplotlib colormap for the heatmap (e.g., 'viridis', 'hot', 'coolwarm', 'YlGnBu'). Default: 'viridis'."
    )

    args = parser.parse_args()

    # Setup Matplotlib style
    # plt.style.use('seaborn-v0_8-pastel') # Example style
    plt.rcParams['font.family'] = PLOT_STYLE.get('font_family', 'sans-serif')
    plt.rcParams['figure.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['axes.facecolor'] = PLOT_STYLE.get('background_color', 'white')

    # Read GML file
    try:
        G = nx.read_gml(args.input_gml, label='id')
    except Exception as e:
        print(f"Error reading GML file {args.input_gml}: {e}")
        sys.exit(1)

    if not G.nodes:
        print(f"Error: GML file {args.input_gml} contains no nodes.")
        sys.exit(1)

    # Group nodes by layer
    nodes_by_layer = defaultdict(list)
    processing_errors = []
    print("Processing nodes and determining layers...")
    for node_id, data in G.nodes(data=True):
        try:
            layer = get_node_layer(data, str(node_id), args.layer_attr)
            nodes_by_layer[layer].append(node_id)
        except ValueError as e:
            processing_errors.append(str(e))

    if processing_errors:
        print("Error processing nodes:")
        for msg in processing_errors:
            print(f"- {msg}")
        sys.exit(1)

    if not nodes_by_layer:
        print("Error: No layers could be determined from the GML file.")
        sys.exit(1)

    # Calculate heat values for each node
    print(f"Calculating heat values for nodes based on '{args.aggregation}' aggregation of '{args.attribute}'...")
    node_heat_values = {}
    stats = {'edges_considered_for_sum': 0, 'edges_missing_attr': 0, 'edges_invalid_value': 0}

    for node_v_id in G.nodes():
        current_sum_val = 0.0
        contributing_edges_count = 0  # Initialize counter for contributing edges
        
        edges_to_process = []
        if args.aggregation == 'incoming':
            edges_to_process = G.in_edges(node_v_id, data=True)
        elif args.aggregation == 'outgoing':
            edges_to_process = G.out_edges(node_v_id, data=True)
        elif args.aggregation == 'all':
            # This will sum weights of in-edges and out-edges.
            # If a self-loop (v,v) exists, its weight will be counted twice.
            edges_to_process.extend(G.in_edges(node_v_id, data=True))
            edges_to_process.extend(G.out_edges(node_v_id, data=True))
        
        for u_edge, v_edge, edge_data in edges_to_process:
            stats['edges_considered_for_sum'] += 1
            if args.attribute in edge_data:
                try:
                    value = float(edge_data[args.attribute])
                    current_sum_val += value
                    contributing_edges_count += 1  # Increment counter
                except (ValueError, TypeError):
                    stats['edges_invalid_value'] += 1
                    # print(f"Warning: Edge ({u_edge}, {v_edge}) has invalid value for '{args.attribute}': {edge_data[args.attribute]}.")
            else:
                stats['edges_missing_attr'] += 1
                # print(f"Warning: Edge ({u_edge}, {v_edge}) is missing attribute '{args.attribute}'.")
        
        # Calculate average: if count is zero, average is zero to avoid division by zero.
        if contributing_edges_count > 0:
            node_heat_values[node_v_id] = current_sum_val / contributing_edges_count
        else:
            node_heat_values[node_v_id] = 0.0  # Or np.nan if preferred, but 0 is simpler for now
    
    if stats['edges_missing_attr'] > 0:
         print(f"Info: {stats['edges_missing_attr']} edge instances were missing the '{args.attribute}' attribute during aggregation.")
    if stats['edges_invalid_value'] > 0:
         print(f"Info: {stats['edges_invalid_value']} edge instances had non-numeric values for '{args.attribute}' during aggregation.")

    # Prepare heatmap matrix
    print("Preparing heatmap matrix...")
    sorted_layer_numbers = sorted(nodes_by_layer.keys())
    num_layers_plot = len(sorted_layer_numbers)

    if num_layers_plot == 0:
        print("Error: No layers with nodes found to plot.")
        sys.exit(1)

    for layer_num in sorted_layer_numbers:
        # Sort nodes within each layer by their ID (converted to string for robust sorting)
        nodes_by_layer[layer_num] = sorted(nodes_by_layer[layer_num], key=lambda x: str(x))

    max_nodes_in_any_layer = max(len(n_list) for n_list in nodes_by_layer.values()) if nodes_by_layer else 0
    
    if max_nodes_in_any_layer == 0:
        print("Error: No nodes found in any layer to plot.")
        sys.exit(1)

    heatmap_matrix = np.full((max_nodes_in_any_layer, num_layers_plot), np.nan)
    min_heat_val, max_heat_val = float('inf'), float('-inf')
    has_valid_heat_value = False

    for layer_plot_idx, actual_layer_num in enumerate(sorted_layer_numbers):
        nodes_in_this_layer = nodes_by_layer[actual_layer_num]
        for node_idx_in_layer, node_id in enumerate(nodes_in_this_layer):
            heat = node_heat_values.get(node_id, np.nan)
            if not np.isnan(heat):
                 min_heat_val = min(min_heat_val, heat)
                 max_heat_val = max(max_heat_val, heat)
                 has_valid_heat_value = True
            heatmap_matrix[node_idx_in_layer, layer_plot_idx] = heat
    
    if not has_valid_heat_value:
        print("Warning: All calculated heat values are NaN or no heat values were computed. Heatmap may be empty or uninformative.")
        min_heat_val = 0 # Default for colorbar if no valid data
        max_heat_val = 1 # Default for colorbar

    # Plot heatmap
    print("Generating heatmap plot...")
    fig_width = max(10, num_layers_plot * 0.6)
    fig_height = max(8, max_nodes_in_any_layer * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    current_cmap = plt.get_cmap(args.colormap)
    current_cmap.set_bad(color='lightgray') # Color for NaN values

    im = ax.imshow(heatmap_matrix, cmap=current_cmap, aspect='auto', interpolation='nearest',
                   vmin=0,  # Fix vmin to 0
                   vmax=1)   # Fix vmax to 1

    # Add grid - REMOVED
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Updated colorbar label
    cbar_label_text = "Ref Heads" # As per "It can be ref heads"
    cbar_label = f"R2 values"
    cbar = plt.colorbar(im, label=cbar_label, ax=ax, pad=0.03)
    cbar.ax.tick_params(labelsize=PLOT_STYLE.get('tick_fontsize', 18))
    cbar.set_label(cbar_label, fontsize=PLOT_STYLE.get('label_fontsize', 34), fontweight='bold', labelpad=15)

    # Remove plot title
    # plot_title = args.title if args.title else f"Node Heatmap: Average of '{args.attribute}' ({args.aggregation})"
    # ax.set_title(plot_title, fontsize=PLOT_STYLE.get('title_fontsize', 14))

    ax.set_xticks(np.arange(num_layers_plot))
    ax.set_xticklabels([str(l) for l in sorted_layer_numbers], rotation=0, ha="center") # Changed L{l} to str(l), rotation to 0, ha to center
    ax.set_xlabel("Layer", fontsize=PLOT_STYLE.get('label_fontsize', 34), fontweight='bold') # Explicitly set size 34

    if max_nodes_in_any_layer > 20:
        step = math.ceil(max_nodes_in_any_layer / 10)
        y_ticks = np.arange(0, max_nodes_in_any_layer, step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([int(val) for val in y_ticks]) # Ensure integer labels
    elif max_nodes_in_any_layer > 0:
        y_ticks = np.arange(max_nodes_in_any_layer)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([int(val) for val in y_ticks]) # Ensure integer labels


    ax.set_ylabel("Head", fontsize=PLOT_STYLE.get('label_fontsize', 34), fontweight='bold') # Changed Y-axis label to "Head"
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 22)) # Use updated tick_fontsize

    # Ensure full frame (all spines visible)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    fig.tight_layout()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_basename = os.path.splitext(os.path.basename(args.input_gml))[0]
        output_path = f"{input_basename}_heatmap.png" # Default to PNG

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            plt.close(fig)
            sys.exit(1)
    
    try:
        plt.savefig(output_path, facecolor=fig.get_facecolor())
        print(f"Heatmap saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving heatmap to {output_path}: {e}")
    finally:
        plt.close(fig)

    print(f"\nSummary of Heat Calculation (now using averages):")
    print(f"  Edge instances considered for aggregation: {stats['edges_considered_for_sum']}")
    if stats['edges_missing_attr'] > 0:
        print(f"  Edge instances missing '{args.attribute}': {stats['edges_missing_attr']}")
    if stats['edges_invalid_value'] > 0:
        print(f"  Edge instances with invalid '{args.attribute}': {stats['edges_invalid_value']}")
    if has_valid_heat_value:
        print(f"  Min calculated average heat value (non-NaN): {min_heat_val:.4f}")
        print(f"  Max calculated average heat value (non-NaN): {max_heat_val:.4f}")
    else:
        print("  No valid heat values were computed.")
        
    sys.exit(0)

if __name__ == "__main__":
    main() 