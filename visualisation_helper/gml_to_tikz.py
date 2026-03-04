#!/usr/bin/env python3
# gml_to_tikz.py
# ==================
#
# Converts a Graph Modeling Language (GML) file representing a layered graph,
# typically a neural network, into TikZ code suitable for LaTeX documents.
#
# Functionality:
# -------------
# 1.  Reads a GML graph file using the NetworkX library.
# 2.  Identifies nodes and groups them into layers based on a specified node
#     attribute (e.g., 'layer') or by parsing node labels like "(L, N)".
# 3.  Determines node types ('input', 'hidden', 'output') based on a specified
#     attribute (e.g., 'type') or infers them based on layer position (min, max,
#     intermediate layers), unless inference is disabled.
# 4.  Calculates node positions for TikZ rendering:
#     - Layers are spaced horizontally (controlled by LAYER_SPACING).
#     - Nodes within a layer are spaced vertically and centered (controlled by
#       NODE_SPACING).
# 5.  Generates TikZ code for drawing nodes with styles based on their type
#     (input, hidden, output). Node labels from the GML 'label' attribute are
#     used if available, otherwise, the node ID is used.
# 6.  Generates TikZ code for drawing edges between nodes.
#     - Edges can optionally be filtered based on a 'weight' attribute; edges
#       with weights below a threshold are skipped.
#     - By default (as modified), it draws ALL edges present in the GML file that
#       pass the weight threshold. It does *not* restrict edges to only connect
#       adjacent layers unless the relevant code section is uncommented.
# 7.  Adds layer titles above the nodes (e.g., "L0", "L1", ...).
# 8.  Outputs the complete TikZ picture environment, either to stdout or to a
#     specified output file.
#
# Usage:
# ------
# python gml_to_tikz.py <input_gml_file> [options] > <output_tex_file>
#
# Common Options:
#   -o, --output : Specify an output file instead of stdout.
#   -s, --scale  : Apply a scaling factor to all positions.
#   -t, --threshold : Set a minimum weight threshold for drawing edges.
#   --layer-attr  : GML attribute name for the layer index (default: 'layer').
#   --type-attr   : GML attribute name for the node type (default: 'type').
#   --no-infer-types: Disable automatic node type inference.
#
# Requirements:
# -------------
# - Python 3
# - NetworkX library (`pip install networkx`)

import argparse
import networkx as nx
import re
import math
from collections import defaultdict

# TikZ styles
TIKZ_OPTIONS = r"""[
    >=stealth,
    shorten >=1pt,
    auto,
    node distance=2cm,
    on grid % Places nodes on integer coordinates, might help alignment
]"""

# TikZ Layer setup
TIKZ_LAYER_SETUP = r"""%
    \pgfdeclarelayer{edges}
    \pgfdeclarelayer{nodes}
    \pgfsetlayers{edges,nodes}
"""

# Old style definition for reference
# TIKZ_OPTIONS = r"""[
#     >=stealth,
#     shorten >=1pt,
#     auto,
#     node distance=2cm,
#     input/.style={circle, draw, fill=orange!80, minimum size=8pt, inner sep=1pt},
#     hidden/.style={circle, draw, fill=blue!30, minimum size=8pt, inner sep=1pt},
#     output/.style={circle, draw, fill=blue!70, minimum size=8pt, inner sep=1pt},
#     edge_style/.style={draw, color=gray, -latex}
# ]"""

LAYER_SPACING = 0.2 # Horizontal spacing between layers
NODE_SPACING = 0.5  # Vertical spacing between nodes within a layer
TITLE_Y_OFFSET = 0.2 # Vertical offset for layer titles above nodes

def sanitize_name(name_str):
    """Converts a node ID string to a TikZ-safe name."""
    # Keep basic structure for tuples like (0,0) -> _0_0_
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name_str)
    # Ensure it doesn't start with a number if it's not purely numeric
    if not sanitized.isnumeric() and sanitized[0].isdigit():
        sanitized = 'n' + sanitized
    # Avoid empty or problematic names
    if not sanitized:
        sanitized = 'node_' + str(hash(name_str))[:8]
    return sanitized

def get_node_layer(node_data, attr_name='layer'):
    node_id = node_data.get('id', 'UNKNOWN') # Use GML id for error messages

    # 1. Try the specified layer attribute first
    layer = node_data.get(attr_name)

    # 2. If attribute is missing, try parsing the label `"(L,N)"`
    if layer is None:
        label = node_data.get('label')
        if label:
            match = re.match(r'\(\s*(\d+)\s*,\s*\d+\s*\)', label)
            if match:
                layer = match.group(1)
                # print(f"Inferred layer {layer} for node {node_id} from label '{label}'") # Debugging
            else:
                # Label exists but doesn't match the expected format
                 raise ValueError(f"Node {node_id} is missing '{attr_name}' and its label '{label}' doesn\'t match pattern \'(L, N)\'.")
        else:
            # No layer attribute and no label to parse
            raise ValueError(f"Node {node_id} is missing the '{attr_name}' attribute and has no label for inference.")

    # 3. Convert the found layer (either from attr or label) to int
    try:
        return int(layer)
    except ValueError:
        raise ValueError(f"Node {node_id}'s derived layer value ('{layer}') is not an integer.")

def get_node_style():
    """Returns a fixed style for all nodes."""
    return "circle, fill=yellow!70!brown!60, minimum size=4pt, inner sep=1pt"

def main():
    parser = argparse.ArgumentParser(description="Convert GML graph (with layers) to TikZ code")
    parser.add_argument("input", help="Input GML file path")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)", default=None)
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Overall scale factor for positions")
    parser.add_argument("-t", "--threshold", type=float, default=0.50, help="Filter out edges with weight strictly less than this threshold (default: disabled)")
    parser.add_argument("--layer-attr", default='layer', help="GML node attribute for layer index (default: 'layer')")
    parser.add_argument("--type-attr", default='type', help="GML node attribute for type ('input', 'hidden', 'output') (default: 'type')")
    parser.add_argument("--no-infer-types", action="store_true", help="Do not infer node types if type attribute is missing")

    args = parser.parse_args()

    # Read GML file
    try:
        G = nx.read_gml(args.input, label='id') # Use 'id' for node identifiers if 'label' is for display
    except Exception as e:
        print(f"Error reading GML file {args.input}: {e}")
        return 1

    # Group nodes by layer and determine node types
    nodes_by_layer = defaultdict(list)
    node_data = {}
    min_layer, max_layer = float('inf'), float('-inf')
    missing_attrs = []

    for n, data in G.nodes(data=True):
        node_data[n] = data
        try:
            layer = get_node_layer(data, args.layer_attr)
            nodes_by_layer[layer].append(n)
            min_layer = min(min_layer, layer)
            max_layer = max(max_layer, layer)

            # Determine node type
            node_type = data.get(args.type_attr)
            if not node_type and not args.no_infer_types:
                # Inference logic will run after finding min/max layers
                pass
            elif node_type not in ['input', 'hidden', 'output']:
                 if not args.no_infer_types: # Only warn if type was expected or inference failed
                     print(f"Warning: Node {n} has invalid type '{node_type}'. Will attempt inference or default.")
                     node_type = None # Allow inference to try
                 else:
                      print(f"Warning: Node {n} has invalid type '{node_type}'. Defaulting to 'hidden'.")
                      node_type = 'hidden' # Default if no inference allowed and type is bad

            node_data[n]['_type'] = node_type # Store type internally

        except ValueError as e:
            missing_attrs.append(str(e))
        except KeyError: # Should be caught by get_node_layer now, but keep for safety
             missing_attrs.append(f"Node {n} data access error (unexpected).")


    if missing_attrs:
        print("Error: The following nodes have missing or invalid attributes:")
        for msg in missing_attrs:
            print(f"- {msg}")
        return 1

    if min_layer == float('inf'):
        print("Error: No nodes found or no layers detected.")
        return 1

    # Final pass for type inference
    if not args.no_infer_types:
        for n, data in node_data.items():
            if not data.get('_type'): # Infer if type wasn't provided or was invalid
                layer = get_node_layer(data, args.layer_attr)
                if layer == min_layer:
                    node_data[n]['_type'] = 'input'
                elif layer == max_layer:
                    node_data[n]['_type'] = 'output'
                else:
                    node_data[n]['_type'] = 'hidden'


    # Calculate node positions
    positions = {}
    max_nodes_in_layer = 0
    layer_node_counts = {}
    sorted_layers = sorted(nodes_by_layer.keys())

    for layer in sorted_layers:
        nodes = sorted(nodes_by_layer[layer]) # Sort nodes for consistent vertical order
        nodes_by_layer[layer] = nodes # Update with sorted list
        count = len(nodes)
        layer_node_counts[layer] = count
        max_nodes_in_layer = max(max_nodes_in_layer, count)

    title_y = ( (max_nodes_in_layer -1 ) / 2.0 * NODE_SPACING + TITLE_Y_OFFSET ) * args.scale

    for layer_idx, layer in enumerate(sorted_layers):
        nodes = nodes_by_layer[layer]
        count = layer_node_counts[layer]
        x = layer_idx * LAYER_SPACING * args.scale
        for i, node_id in enumerate(nodes):
            # Center nodes vertically within the layer
            y = -(i - (count - 1) / 2.0) * NODE_SPACING * args.scale
            positions[node_id] = (x, y)


    # Generate TikZ
    lines = []
    lines.append(r"\begin{tikzpicture}" + TIKZ_OPTIONS)
    lines.append(TIKZ_LAYER_SETUP) # Add layer setup commands

    # Add Layer Titles - moved inside the nodes layer below
    # for layer_idx, layer in enumerate(sorted_layers):
    #      x_pos = layer_idx * LAYER_SPACING * args.scale
    #      layer_title = f"L{layer}"
    #      lines.append(fr"  \node[anchor=center, font=\tiny] at ({x_pos:.2f},{title_y:.2f}) {{{layer_title}}};")


    # 1. Pre-calculate node styles and colors -> Removed, using fixed style now
    # node_styles = {}
    # node_colors = {}
    # for node_id in G.nodes():
    #     data = node_data[node_id]
    #     layer = get_node_layer(data, args.layer_attr)
    #     style = get_layer_style(layer, min_layer, max_layer) # OLD dynamic style
    #     node_styles[node_id] = style
    #     # Extract color for edge styling
    #     if "fill=" in style:
    #         color_info = style.split("fill=")[1].split(",")[0]
    #         node_colors[node_id] = color_info
    #     else:
    #         node_colors[node_id] = "gray" # Default edge color if node has no fill

    # 2. Draw Nodes and Layer Titles (define them first, place on 'nodes' layer)
    lines.append(r"  \begin{pgfonlayer}{nodes}")

    # Add Layer Titles (simple version - now explicitly on 'nodes' layer)
    for layer_idx, layer in enumerate(sorted_layers):
         x_pos = layer_idx * LAYER_SPACING * args.scale
         layer_title = f"L{layer}"
         lines.append(fr"    \node[anchor=center, font=\fontsize{3}{4}\selectfont] at ({x_pos:.2f},{title_y:.2f}) {{{layer_title}}};") # Smaller text

    fixed_node_style = get_node_style() # Get the fixed style once
    for node_id, (x, y) in positions.items():
        # style = node_styles.get(node_id, "circle, fill=gray") # Use pre-calculated style -> OLD
        name = sanitize_name(str(node_id))
        lines.append(fr"    \node[{fixed_node_style}] ({name}) at ({x:.2f},{y:.2f}) {{}}; % Node defined here") # Use fixed style
    lines.append(r"  \end{pgfonlayer}")

    # 3. Draw Edges (after nodes are defined, place on 'edges' layer)
    lines.append(r"  \begin{pgfonlayer}{edges}")
    drawn_edges = 0
    skipped_edges = 0
    edge_style = "draw, color=brown!70!black!30, opacity=0.7, line width=0.2pt, -latex" # Finer edges
    for u, v, edge_data in G.edges(data=True):
         # Skip edges below the weight threshold
        if args.threshold is not None:
            try:
                weight = float(edge_data.get("weight", 1.0)) # Ensure weight is float
                if weight < args.threshold:
                    skipped_edges += 1
                    continue
            except (ValueError, TypeError):
                print(f"Warning: Could not interpret weight '{edge_data.get('weight')}' for edge ({u}, {v}). Skipping threshold check for this edge.")

        # Get the color of the source node -> Removed
        # source_color = node_colors.get(u, "gray") # Use pre-calculated color

        name_u = sanitize_name(str(u))
        name_v = sanitize_name(str(v))
        # lines.append(fr"    \path[draw, color={source_color}, -latex] ({name_u}) edge ({name_v});") # OLD edge drawing
        lines.append(fr"    \path[{edge_style}] ({name_u}) edge ({name_v});") # Use fixed edge style
        drawn_edges += 1
    lines.append(r"  \end{pgfonlayer}")


    lines.append(r"\end{tikzpicture}")
    output = "\n".join(lines)

    print(f"Generated TikZ code: {len(positions)} nodes, {drawn_edges} edges drawn, {skipped_edges} edges skipped.")

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Output written to {args.output}")
        except IOError as e:
            print(f"Error writing to output file {args.output}: {e}")
            return 1
    else:
        print(output)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 