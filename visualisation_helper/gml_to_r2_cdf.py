#!/usr/bin/env python3
# gml_to_r2_cdf.py
# ==================
#
# Reads a Graph Modeling Language (GML) file, extracts R^2 values from the
# edges, and plots the Cumulative Distribution Function (CDF) of these values.
# By default, saves the plot as a PDF in a 'plots/' directory.
#
# Usage:
# ------
# python gml_to_r2_cdf.py <input_gml_file> [options]
#
# Common Options:
#   -o, --output : Specify a custom output PDF file path. Overrides the default
#                  saving location and name (plots/input_basename_attr_cdf.pdf).
#   -a, --attribute: GML edge attribute name for the R^2 value (default: 'weight').
#   -t, --title  : Title for the plot.
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
import sys # Added for path manipulation
import os  # Added for path manipulation
import itertools # Added for color cycling
import glob # Added for finding GML files
import seaborn as sns # Added for KDE plots
import logging # Added for consistency with the new plot style source

# Add parent directory to sys.path for relative imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Define the new PLOT_STYLE based on theme_ref_code.py (from recompute_r2_from_top_n.py)
# This PLOT_STYLE will be used throughout the script.
PLOT_STYLE = {
    'background_color': 'white',
    'colors_list': [ # Updated with more distinct colors
        '#1f77b4',  # Muted Blue
        '#ff7f0e',  # Safety Orange
        '#2ca02c',  # Cooked Asparagus Green
        '#d62728',  # Brick Red
        '#9467bd',  # Muted Purple
        '#8c564b',  # Chestnut Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Medium Gray
        '#bcbd22',  # Curry Yellow-Green
        '#17becf'   # Blue-Teal
    ],
    'hatches_list': ['\\\\\\\\', '-', '//', '', 'x', ''], # Corrected: '\\\\\\\\' in string literal becomes '\\\\' in Python code
    'line_color_default': '#1f77b4', # Updated to match the first color in the new list
    'marker_color_default': '#1f77b4', # Updated to match the first color in the new list
    'opacity_default': 0.8,
    'opacity_fill_kde': 0.4,
    'grid_linestyle': '--',
    'grid_linewidth': 0.5,
    'grid_alpha': 0.7,
    'font_family': 'sans-serif',
    'title_fontsize': 32,
    'label_fontsize': 32,
    'tick_fontsize': 28,
    'legend_fontsize': 20,
    'fontweight': 'bold',
    'plot_edgecolor': 'black',
    'spine_top_visible': True,
    'spine_right_visible': True,
}

# Setup logging (optional, but good practice if we use it later or for consistency)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Commented out as logging isn't used elsewhere in this script yet.

def get_layer_from_label(label_str):
    """
    Extracts the layer number from a GML node label string.
    Assumes label format like "(layer, head)", e.g., "(0,3)".
    Returns the layer number as an int, or None if parsing fails.
    """
    if not isinstance(label_str, str):
        return None
    try:
        # Remove parentheses and split by comma
        parts = label_str.strip("()").split(',')
        if len(parts) == 2:
            return int(parts[0].strip())
    except (ValueError, AttributeError):
        # Handle cases where parsing fails or label_str is not as expected
        return None
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Generate CDF and KDE plots of R^2 values from GML files found in subdirectories of a given input directory. Each subdirectory is expected to contain one GML file, and the subdirectory's name will be used as the label in the plot."
    )
    # Changed 'inputs' to 'input_directory' and expect a single path
    parser.add_argument("input_directory", help="Path to the main directory containing subdirectories with GML files.")
    parser.add_argument(
        "-o", "--output",
        help="Custom output base file path (e.g., 'my_plots/custom_name'). Extensions (.pdf, .png) and '_kde' suffix for KDE plots will be appended. If omitted, defaults to 'plots/<input_directory_name>_<attribute>_cdf'.",
        default=None
    )
    parser.add_argument("-a", "--attribute", default='weight', help="GML edge attribute containing the R^2 value (default: 'weight')")
    parser.add_argument("-t", "--title", default=None, help="Base title for the plots (default: 'Edge <Attribute> Values'). 'CDF of' or 'KDE of' will be prepended.")
    # Removed "-l, --labels" argument

    # Add mutually exclusive group for --intra and --inter
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--intra", action="store_true", help="Consider only intra-layer connections (edges where source and target nodes are in the same layer).")
    group.add_argument("--inter", action="store_true", help="Consider only inter-layer connections (edges where source and target nodes are in different layers).")

    args = parser.parse_args()

    # --- Discover GML files and their labels ---
    gml_entries = [] # List of {'path': path_to_gml, 'label': subdirectory_name}
    
    base_input_dir = args.input_directory
    if not os.path.isdir(base_input_dir):
        print(f"Error: Input path '{base_input_dir}' is not a directory or does not exist.")
        return 1

    print(f"Scanning subdirectories in '{base_input_dir}' for GML files...")
    
    sub_items = sorted(os.listdir(base_input_dir)) # Sort for consistent processing order

    for item_name in sub_items:
        item_path = os.path.join(base_input_dir, item_name)
        if os.path.isdir(item_path):
            potential_label = item_name # Subdirectory name as label
            
            # Look for .gml files in this subdirectory
            # Using '*.gml' to find any GML file.
            gml_files_found = glob.glob(os.path.join(item_path, '*.gml'))
            
            if not gml_files_found:
                print(f"Warning: No GML files found in subdirectory '{item_path}'. Skipping.")
                continue
            
            # Sort found GML files for deterministic behavior if multiple exist
            gml_files_found.sort()
            selected_gml_file = gml_files_found[0] # Take the first one

            if len(gml_files_found) > 1:
                print(f"Warning: Multiple GML files found in '{item_path}'. Using '{selected_gml_file}'. Found: {gml_files_found}")
            
            gml_entries.append({'path': selected_gml_file, 'label': potential_label})
            print(f"  Found GML: '{selected_gml_file}' with label '{potential_label}'")

    if not gml_entries:
        print(f"Error: No GML files could be processed from the subdirectories of '{base_input_dir}'.")
        return 1

    # --- Process each discovered GML file ---
    all_results = [] # Store results for each file: (label, r2_values, cdf)
    global_stats = {'total_edges_processed': 0, 'total_edges_missing': 0, 'total_edges_invalid': 0}

    print(f"\nProcessing {len(gml_entries)} GML file(s) from subdirectories...")

    for entry in gml_entries:
        input_file = entry['path']
        current_plot_label = entry['label']
        
        print(f"\n--- Processing file: {input_file} (Label: {current_plot_label}) ---")
        try:
            G = nx.read_gml(input_file, label='id')
        except Exception as e:
            print(f"Error reading GML file {input_file}: {e}. Skipping this file.")
            continue

        # Use defaultdict to store R^2 values for each head node
        head_r2_map = defaultdict(list)
        edges_missing_attr = 0
        edges_invalid_value = 0

        print(f"Reading edges, extracting '{args.attribute}' and finding max per head node...")
        for u, v, data in G.edges(data=True): # v is the head node (target)
            # --- Layer filtering logic ---
            if args.intra or args.inter:
                try:
                    u_label = G.nodes[u].get('label')
                    v_label = G.nodes[v].get('label')

                    if u_label is None or v_label is None:
                        # print(f"Warning: Node {u} or {v} missing label. Skipping edge ({u}, {v}) for layer filtering.")
                        continue

                    u_layer = get_layer_from_label(u_label)
                    v_layer = get_layer_from_label(v_label)

                    if u_layer is None or v_layer is None:
                        # print(f"Warning: Could not parse layer from label for node {u} ('{u_label}') or {v} ('{v_label}'). Skipping edge ({u}, {v}).")
                        continue
                    
                    if args.intra and u_layer != v_layer:
                        continue # Skip if --intra and layers are different
                    if args.inter and u_layer == v_layer:
                        continue # Skip if --inter and layers are the same
                except KeyError as e:
                    # This might happen if a node involved in an edge is not in G.nodes (should not happen for valid GML)
                    # print(f"Warning: Node {e} not found in graph nodes while trying to get label. Skipping edge ({u}, {v}).")
                    continue
            # --- End Layer filtering logic ---
            
            if args.attribute in data:
                try:
                    r2_value = float(data[args.attribute])
                    head_r2_map[v].append(r2_value) # Store R^2 for this specific head node
                    global_stats['total_edges_processed'] += 1 # Count only if attribute exists and is processed
                except (ValueError, TypeError):
                    edges_invalid_value += 1
                    global_stats['total_edges_invalid'] += 1
            else:
                edges_missing_attr += 1
                global_stats['total_edges_missing'] += 1

        # Select the best (max) R^2 for each head node
        selected_r2_values = []
        if not head_r2_map:
            print(f"Warning: No head nodes found with attribute '{args.attribute}' in {input_file}.")
        else:
            for head_node, r2_list_for_head in head_r2_map.items():
                if r2_list_for_head: # Ensure the list is not empty
                    selected_r2_values.append(max(r2_list_for_head))
        
        if not selected_r2_values:
            print(f"Warning: No valid max '{args.attribute}' values found for any head node in {input_file}. Skipping plot for this file.")
            print(f"Summary for {input_file}: Edges missing attribute (overall): {edges_missing_attr}, Edges with invalid value (overall): {edges_invalid_value}")
            continue

        print(f"Found {len(selected_r2_values)} head nodes with max '{args.attribute}' values in {input_file}.")
        if edges_missing_attr > 0:
            print(f"Note: {edges_missing_attr} edges (overall) were missing the '{args.attribute}' attribute.")
        if edges_invalid_value > 0:
            print(f"Note: {edges_invalid_value} edges (overall) had non-numeric values for '{args.attribute}'.")

        # Sort the selected max R^2 values
        r2_values_sorted = np.sort(np.array(selected_r2_values))
        cdf_y = np.arange(1, len(r2_values_sorted) + 1) / len(r2_values_sorted)
        
        # Store selected max R2 values for KDE plot as well
        all_results.append({
            'label': current_plot_label,
            'x_cdf': r2_values_sorted,
            'y_cdf': cdf_y,
            'raw_r2_values': list(selected_r2_values) # Use the selected max R2 values
        })

    if not all_results:
        print("\nError: No valid data found in any of the processed GML files. Cannot generate plots.")
        print(f"Overall Summary: Total Edges Processed: {global_stats['total_edges_processed']}, Total Missing Attribute: {global_stats['total_edges_missing']}, Total Invalid Value: {global_stats['total_edges_invalid']}")
        return 1

    print(f"\nOverall Summary: Plotting data for {len(all_results)} file(s).")
    print(f"Total Edges Processed: {global_stats['total_edges_processed']}, Total Missing Attribute: {global_stats['total_edges_missing']}, Total Invalid Value: {global_stats['total_edges_invalid']}")

    # --- Generate CDF Plot ---
    print("\nGenerating CDF plot...")
    base_plot_title = args.title if args.title else f"Max R² per Head ({args.attribute}) Values"
    cdf_plot_title = f"CDF of {base_plot_title}"

    # Apply new plot style using rcParams
    plt.rcParams['font.family'] = PLOT_STYLE.get('font_family', 'sans-serif')
    plt.rcParams['axes.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['figure.facecolor'] = PLOT_STYLE.get('background_color', 'white')
    plt.rcParams['axes.spines.top'] = PLOT_STYLE.get('spine_top_visible', True)
    plt.rcParams['axes.spines.right'] = PLOT_STYLE.get('spine_right_visible', True)
    plt.rcParams['axes.grid'] = True # Enable grid
    plt.rcParams['grid.linestyle'] = PLOT_STYLE.get('grid_linestyle', '--')
    plt.rcParams['grid.linewidth'] = PLOT_STYLE.get('grid_linewidth', 0.5)
    plt.rcParams['grid.alpha'] = PLOT_STYLE.get('grid_alpha', 0.7)

    fig_cdf = plt.figure(figsize=(12, 8)) # Increased figure size for larger fonts
    ax_cdf = plt.gca()
    # fig_cdf.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Handled by rcParams
    # ax_cdf.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Handled by rcParams

    color_cycler_cdf = itertools.cycle(PLOT_STYLE['colors_list'])
    marker_style = '.' # Default marker style from original script, PLOT_STYLE doesn't specify 'marker_style'

    for result in all_results:
        plot_color = next(color_cycler_cdf)
        ax_cdf.plot(result['x_cdf'], result['y_cdf'], marker=marker_style, linestyle='none',
                    color=plot_color, label=result['label'], markerfacecolor=plot_color,
                    markeredgecolor=plot_color, # Changed from PLOT_STYLE.get('plot_edgecolor', 'black')
                    alpha=PLOT_STYLE.get('opacity_default', 0.8), # Use new opacity
                    markersize=8) # Increased marker size

    ax_cdf.set_xlabel("R² Value", fontsize=PLOT_STYLE.get('label_fontsize', 12), fontweight=PLOT_STYLE.get('fontweight', 'bold'))
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=PLOT_STYLE.get('label_fontsize', 12), fontweight=PLOT_STYLE.get('fontweight', 'bold'))
    ax_cdf.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 10))
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_xlim(left=-0.05, right=1.0) # Set xlim similar to recompute_r2 script
    ax_cdf.grid(True, linestyle=PLOT_STYLE.get('grid_linestyle', '--'), linewidth=PLOT_STYLE.get('grid_linewidth', 0.5), alpha=PLOT_STYLE.get('grid_alpha', 0.7)) # Re-enable grid for this plot
    ax_cdf.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 22))

    # --- Generate KDE Plot ---
    print("\nGenerating KDE plot...")
    kde_plot_title = f"KDE of {base_plot_title}"
    
    fig_kde = plt.figure(figsize=(12, 8)) # Increased figure size
    ax_kde = plt.gca()
    # fig_kde.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Handled by rcParams
    # ax_kde.set_facecolor(PLOT_STYLE.get('background_color', 'white')) # Handled by rcParams

    color_cycler_kde = itertools.cycle(PLOT_STYLE['colors_list'])

    for result in all_results:
        if not result['raw_r2_values']:
            print(f"Skipping KDE for {result['label']} as it has no max R2 values for head nodes.")
            continue
        plot_color = next(color_cycler_kde)
        sns.kdeplot(data=result['raw_r2_values'], ax=ax_kde, label=result['label'],
                    color=plot_color, 
                    fill=True, 
                    alpha=PLOT_STYLE.get('opacity_fill_kde', 0.4), # Use new opacity for fill
                    linewidth=1.5,
                    edgecolor=None) # Remove edgecolor for cleaner KDE

    ax_kde.set_xlabel("R² Value", fontsize=PLOT_STYLE.get('label_fontsize', 12), fontweight=PLOT_STYLE.get('fontweight', 'bold'))
    ax_kde.set_ylabel("Density", fontsize=PLOT_STYLE.get('label_fontsize', 12), fontweight=PLOT_STYLE.get('fontweight', 'bold'))
    ax_kde.tick_params(axis='both', which='major', labelsize=PLOT_STYLE.get('tick_fontsize', 10))
    ax_kde.set_xlim(left=-0.05, right=1.0) # Set xlim similar to recompute_r2 script
    ax_kde.set_ylim(0, 5.0) # Limit y-axis to 5
    ax_kde.grid(True, linestyle=PLOT_STYLE.get('grid_linestyle', '--'), linewidth=PLOT_STYLE.get('grid_linewidth', 0.5), alpha=PLOT_STYLE.get('grid_alpha', 0.7)) # Re-enable grid for this plot
    if any(res['raw_r2_values'] for res in all_results): 
        ax_kde.legend(fontsize=PLOT_STYLE.get('legend_fontsize', 22))

    # --- Determine output paths ---
    if args.output:
        output_base_path_no_ext = os.path.splitext(args.output)[0]
        output_target_dir_from_arg = os.path.dirname(output_base_path_no_ext)
        if output_target_dir_from_arg and not os.path.exists(output_target_dir_from_arg):
            print(f"Creating output directory specified in --output: {output_target_dir_from_arg}")
            try:
                os.makedirs(output_target_dir_from_arg, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {output_target_dir_from_arg}: {e}")
                plt.close(fig_cdf)
                if 'fig_kde' in locals(): plt.close(fig_kde)
                return 1
    else:
        default_output_dir = "plots"
        if not os.path.exists(default_output_dir):
            print(f"Creating default output directory: {default_output_dir}")
            try:
                os.makedirs(default_output_dir)
            except OSError as e:
                print(f"Error creating directory {default_output_dir}: {e}")
                plt.close(fig_cdf)
                if 'fig_kde' in locals(): plt.close(fig_kde)
                return 1
        
        # New default filename logic based on input_directory name
        input_dir_name_part = os.path.basename(os.path.normpath(args.input_directory))
        # Sanitize the directory name for use in a filename
        sanitized_input_dir_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in input_dir_name_part)
        if not sanitized_input_dir_name: # Handle cases like "/" or "." becoming empty
            sanitized_input_dir_name = "plot_output"

        default_filename_stem = f"{sanitized_input_dir_name}_{args.attribute}_cdf"
        output_base_path_no_ext = os.path.join(default_output_dir, default_filename_stem)

    # Paths for CDF plots
    output_path_cdf_pdf = output_base_path_no_ext + ".pdf"
    output_path_cdf_png = output_base_path_no_ext + ".png"

    # Paths for KDE plots (append _kde to the base stem)
    output_path_kde_pdf = output_base_path_no_ext + "_kde.pdf"
    output_path_kde_png = output_base_path_no_ext + "_kde.png"

    # --- Save Plots ---
    cdf_pdf_saved = False
    cdf_png_saved = False
    kde_pdf_saved = False # Initialize for KDE
    kde_png_saved = False # Initialize for KDE

    # Save CDF PDF
    print(f"Saving CDF PDF plot to: {output_path_cdf_pdf}")
    try:
        fig_cdf.savefig(output_path_cdf_pdf, format='pdf', bbox_inches='tight', facecolor=fig_cdf.get_facecolor())
        print(f"CDF PDF plot saved successfully.")
        cdf_pdf_saved = True
    except Exception as e:
        print(f"Error saving CDF PDF plot to {output_path_cdf_pdf}: {e}")

    # Save CDF PNG
    print(f"Saving CDF PNG plot to: {output_path_cdf_png}")
    try:
        fig_cdf.savefig(output_path_cdf_png, format='png', bbox_inches='tight', facecolor=fig_cdf.get_facecolor(), dpi=300)
        print(f"CDF PNG plot saved successfully.")
        cdf_png_saved = True
    except Exception as e:
        print(f"Error saving CDF PNG plot to {output_path_cdf_png}: {e}")
    
    plt.close(fig_cdf) # Close CDF figure

    # Save KDE PDF (only if KDE plot was generated)
    if 'fig_kde' in locals() and any(res['raw_r2_values'] for res in all_results):
        print(f"Saving KDE PDF plot to: {output_path_kde_pdf}")
        try:
            fig_kde.savefig(output_path_kde_pdf, format='pdf', bbox_inches='tight', facecolor=fig_kde.get_facecolor())
            print(f"KDE PDF plot saved successfully.")
            kde_pdf_saved = True
        except Exception as e:
            print(f"Error saving KDE PDF plot to {output_path_kde_pdf}: {e}")

        # Save KDE PNG
        print(f"Saving KDE PNG plot to: {output_path_kde_png}")
        try:
            fig_kde.savefig(output_path_kde_png, format='png', bbox_inches='tight', facecolor=fig_kde.get_facecolor(), dpi=300)
            print(f"KDE PNG plot saved successfully.")
            kde_png_saved = True
        except Exception as e:
            print(f"Error saving KDE PNG plot to {output_path_kde_png}: {e}")
        
        plt.close(fig_kde) # Close KDE figure
    elif 'fig_kde' in locals(): # KDE figure was created but no data plotted
        plt.close(fig_kde) 
        # If no data was plotted, we consider KDE "saving" successful as there's nothing to save.
        kde_pdf_saved = True 
        kde_png_saved = True


    final_return_code = 0
    if not (cdf_pdf_saved and cdf_png_saved):
        print("One or more CDF plot saving operations failed.")
        final_return_code = 1
    
    # Check KDE saving only if it was attempted (i.e., there was data)
    if any(res['raw_r2_values'] for res in all_results): # Check if KDE plotting was meaningful
        if not (kde_pdf_saved and kde_png_saved):
            print("One or more KDE plot saving operations failed.")
            final_return_code = 1
    
    if final_return_code == 0:
        print("\nAll plots saved successfully.")
    
    return final_return_code

if __name__ == "__main__":
    import sys
    sys.exit(main())
