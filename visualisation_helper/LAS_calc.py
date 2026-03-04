import os
import glob
import argparse
import sys
from typing import Tuple, Dict, List

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Plotting style from recompute_r2_from_top_n.py
PLOT_STYLE = {
    'background_color': 'white',
    'colors_list': [  # Updated with more distinct colors
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
    'hatches_list': ['\\\\', '-', '//', '', 'x', ''],
    'line_color_default': '#1f77b4',
    'marker_color_default': '#1f77b4',
    'opacity_default': 0.8,
    'opacity_fill_kde': 0.4,
    'grid_linestyle': '--',
    'grid_linewidth': 0.5,
    'grid_alpha': 0.7,
    'font_family': 'sans-serif',
    'title_fontsize': 25,
    'label_fontsize': 25,
    'tick_fontsize': 20,
    'legend_fontsize': 22,
    'fontweight': 'bold',
    'plot_edgecolor': 'black',
    'spine_top_visible': True,
    'spine_right_visible': True,
}

# ---------- Helper utilities -------------------------------------------------

def parse_node_label(label: str) -> Tuple[int, int]:
    """Parse a node label of the form "(layer, head)" -> (layer, head)."""
    try:
        layer_str, head_str = label.strip("() ").split(",")
        return int(layer_str), int(head_str)
    except Exception:
        return None, None


def get_best_incoming_edges_for_heads(G: nx.Graph, n: int = 1) -> List[Tuple[int, int, float]]:
    """Return the N heaviest incoming edges for every head node in *G*.

    A *head node* is recognised by a label formatted as ``(layer, head)``.
    The function returns a list of tuples  ``(u, v, weight)`` with **unique**
    undirected edges.

    Args:
        G: The graph to analyze
        n: Number of top edges to return per head (default 1)
    """
    heads = {}
    for node, data in G.nodes(data=True):
        lbl = data.get("label", "")
        layer, head = parse_node_label(lbl)
        if layer is not None:
            heads[node] = (layer, head)

    best_for_node: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {}
    for u, v, data in G.edges(data=True):
        w = data.get("weight")
        if w is None:
            continue
        w = float(w)
        # Consider each endpoint *if* it is a recognised head node
        for node in (u, v):
            if node in heads:
                if node not in best_for_node:
                    best_for_node[node] = []
                best_for_node[node].append((w, (u, v)))
    
    # Sort edges by weight and take top n
    for node in best_for_node:
        best_for_node[node].sort(reverse=True)
        best_for_node[node] = best_for_node[node][:n]

    # keep unique undirected edges with their weight
    unique_edges = {}
    for edges_list in best_for_node.values():
        for weight, (u, v) in edges_list:
            key = tuple(sorted((u, v)))
            if key not in unique_edges or weight > unique_edges[key][2]:
                unique_edges[key] = (u, v, weight)
    return list(unique_edges.values())


def count_layer_connectivity(gml_path: str, n: int = 1) -> Tuple[int, int]:
    """Return (intra_count, inter_count) for *gml_path* considering top N edges.

    The function looks at the strongest N incoming edges per head node and classifies
    them as intra‑layer or inter‑layer based solely on the layers of the two heads,
    without applying any weight threshold.

    Args:
        gml_path: Path to the GML file
        n: Number of top edges to consider per head (default 1)
    """
    try:
        G = nx.read_gml(gml_path, label="id")
    except Exception:
        return None, None

    intra = inter = 0
    best_edges = get_best_incoming_edges_for_heads(G, n)
    node_layers = {n: parse_node_label(G.nodes[n]["label"])[0] for n in G.nodes if "label" in G.nodes[n]}
    for u, v, _ in best_edges:
        if node_layers.get(u) is None or node_layers.get(v) is None:
            continue
        if node_layers[u] == node_layers[v]:
            intra += 1
        else:
            inter += 1
    return intra, inter


# ---------- Plot helpers ------------------------------------------------------

def _apply_plot_style(ax):
    fig = plt.gcf()
    fig.set_facecolor(PLOT_STYLE['background_color'])
    ax.set_facecolor(PLOT_STYLE['background_color'])
    ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
    ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])


def plot_counts(steps: List[str], intra: List[int], inter: List[int], out_dir: str, n: int = 1):
    plt.figure(figsize=(12, 6))
    x = range(len(steps))
    plt.plot(x, intra, "o-", label="Intra‑layer", linewidth=4.0, color=PLOT_STYLE['colors_list'][0])
    plt.plot(x, inter, "s--", label="Inter‑layer", linewidth=4.0, color=PLOT_STYLE['colors_list'][1])

    plt.xticks(x, steps)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Dominant Connections", fontsize=PLOT_STYLE['label_fontsize'], fontweight='bold')
    plt.xlabel("Training step", fontsize=PLOT_STYLE['label_fontsize'], fontweight='bold')
    plt.grid(True, linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])
    plt.legend(fontsize=PLOT_STYLE['legend_fontsize'])
    plt.tight_layout()

    _apply_plot_style(plt.gca())

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"inter_intra_counts_N{n}.png")
    pdf = os.path.join(out_dir, f"inter_intra_counts_N{n}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved plots to {png} and {pdf}")
    plt.close()


def plot_percentages(steps: List[str], intra_pct: List[float], inter_pct: List[float], out_dir: str, n: int = 1):
    plt.figure(figsize=(12, 6))
    x = range(len(steps))
    plt.plot(x, intra_pct, "o-", label="Intra‑layer (%)", linewidth=4.0, color=PLOT_STYLE['colors_list'][0])
    plt.plot(x, inter_pct, "s--", label="Inter‑layer (%)", linewidth=4.0, color=PLOT_STYLE['colors_list'][1])

    plt.xticks(x, steps)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Dominant Connections (%)", fontsize=PLOT_STYLE['label_fontsize'], fontweight='bold')
    plt.xlabel("Training step", fontsize=PLOT_STYLE['label_fontsize'], fontweight='bold')
    plt.grid(True, linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])
    plt.legend(fontsize=PLOT_STYLE['legend_fontsize'])
    plt.tight_layout()

    _apply_plot_style(plt.gca())

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"inter_intra_percentages_N{n}.png")
    pdf = os.path.join(out_dir, f"inter_intra_percentages_N{n}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved percentage plots to {png} and {pdf}")
    plt.close()

# ---------- Main --------------------------------------------------------------

def format_step_label(step_num: int) -> str:
    """Format step number into k/M notation for specific values."""
    if step_num == 1000:
        return "1k"
    elif step_num == 50000:
        return "50k"
    elif step_num == 100000:
        return "100k"
    elif step_num == 250000:
        return "250k"
    elif step_num == 928646:
        return "1M"
    return str(step_num)

def extract_step_from_name(name: str) -> int:
    """Return leading integer in *name* (e.g. "1000 step" -> 1000)."""
    parts = name.strip().split()
    return int(parts[0]) if parts and parts[0].isdigit() else float("inf")


def main():
    p = argparse.ArgumentParser(
        description="Plot intra & inter‑layer connectivity (strongest incoming edge per head) across training steps."
    )
    p.add_argument("--input_dir", required=True, help="Root directory containing sub‑directories with GML files.")
    p.add_argument("--output_dir", default="plots", help="Directory to save plots.")
    p.add_argument("--ignore_key", action="store_true", help="Ignore *_key.gml files.")
    args = p.parse_args()

    steps: List[str] = []
    
    # For N=1
    intra_counts_n1: List[int] = []
    inter_counts_n1: List[int] = []
    
    # For N=5
    intra_counts_n5: List[int] = []
    inter_counts_n5: List[int] = []

    for sub in sorted(os.listdir(args.input_dir), key=extract_step_from_name):
        sub_path = os.path.join(args.input_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        # Prefer value file, fall back to key if allowed
        gml_pattern = "*_value.gml"
        files = glob.glob(os.path.join(sub_path, gml_pattern))
        if not files and not args.ignore_key:
            files = glob.glob(os.path.join(sub_path, "*_key.gml"))
        if not files:
            print(f"[WARN] No GML found in {sub}")
            continue
        gml_path = files[0]  # take first match
        
        # Process N=1
        intra_n1, inter_n1 = count_layer_connectivity(gml_path, n=1)
        if intra_n1 is None:
            print(f"[ERROR] Failed to process {gml_path}")
            continue
            
        # Process N=5
        intra_n5, inter_n5 = count_layer_connectivity(gml_path, n=5)
        if intra_n5 is None:
            print(f"[ERROR] Failed to process {gml_path} for N=5")
            continue

        raw_step = extract_step_from_name(sub)
        steps.append(format_step_label(raw_step))
        
        # Store data for N=1
        intra_counts_n1.append(intra_n1)
        inter_counts_n1.append(inter_n1)
        
        # Store data for N=5
        intra_counts_n5.append(intra_n5)
        inter_counts_n5.append(inter_n5)
        
        print(f"{sub} (N=1): intra={intra_n1}, inter={inter_n1}")
        print(f"{sub} (N=5): intra={intra_n5}, inter={inter_n5}")

    if not steps:
        print("No valid data collected – nothing to plot.")
        sys.exit(1)

    # Plot for N=1
    plot_counts(steps, intra_counts_n1, inter_counts_n1, args.output_dir, n=1)

    # Compute percentages for N=1
    intra_pct_n1 = []
    inter_pct_n1 = []
    for intra, inter in zip(intra_counts_n1, inter_counts_n1):
        total = intra + inter
        if total == 0:
            intra_pct_n1.append(0)
            inter_pct_n1.append(0)
        else:
            intra_pct_n1.append(intra / total * 100)
            inter_pct_n1.append(inter / total * 100)

    plot_percentages(steps, intra_pct_n1, inter_pct_n1, args.output_dir, n=1)
    
    # Plot for N=5
    plot_counts(steps, intra_counts_n5, inter_counts_n5, args.output_dir, n=5)

    # Compute percentages for N=5
    intra_pct_n5 = []
    inter_pct_n5 = []
    for intra, inter in zip(intra_counts_n5, inter_counts_n5):
        total = intra + inter
        if total == 0:
            intra_pct_n5.append(0)
            inter_pct_n5.append(0)
        else:
            intra_pct_n5.append(intra / total * 100)
            inter_pct_n5.append(inter / total * 100)

    plot_percentages(steps, intra_pct_n5, inter_pct_n5, args.output_dir, n=5)


if __name__ == "__main__":
    main()
