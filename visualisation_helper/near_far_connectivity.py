import ast
import statistics as stats
from pathlib import Path
import networkx as nx


def _layer_head(label: str) -> tuple[int, int]:
    """
    Convert a node label like "(3,12)" → (layer=3, head=12).
    """
    return ast.literal_eval(label)   # safer than split/strip


def analyze_connectivity(gml_path: str | Path) -> None:
    """
    Compute summary statistics for the directed graph stored in *gml_path*.
    """
    G = nx.read_gml(gml_path)

    # ------------------------------------------------------------------
    # Build a mapping id  → (layer, head) for quick lookup
    # ------------------------------------------------------------------
    id2layer = {}
    skipped_nodes_parsing = []
    for node_id in G.nodes:
        try:
            # Assume node ID itself is the string "(layer,head)"
            layer_head_tuple = _layer_head(str(node_id))
            id2layer[node_id] = layer_head_tuple
        except (ValueError, SyntaxError):
            # Log nodes whose IDs couldn\'t be parsed
            skipped_nodes_parsing.append(node_id)
            # Optional: Add fallback to label if needed later

    if skipped_nodes_parsing:
        print(f"Warning: Could not parse layer/head info directly from the node ID for {len(skipped_nodes_parsing)} nodes (showing up to 5): {skipped_nodes_parsing[:5]}...")

    # Determine the maximum layer index
    if not id2layer:
        print("Error: No nodes could be mapped to layers. Cannot proceed.")
        return # Or handle appropriately
    max_layer = max(layer for layer, head in id2layer.values())

    # ------------------------------------------------------------------
    # Gather edge‑level information
    # ------------------------------------------------------------------
    weights = []
    # Initialize sums for weights instead of counts
    total_weight_sum = 0.0
    same_layer_weight_sum = 0.0
    inter_layer_weight_sum = 0.0
    # Initialize counts for strong connections
    same_layer_strong_connections = 0
    inter_layer_strong_connections = 0
    # Initialize total counts for same/inter layer edges
    same_layer_total_edges = 0
    inter_layer_total_edges = 0

    close2_weight_sum = 0.0
    far2_weight_sum = 0.0
    close4_weight_sum = 0.0
    far4_weight_sum = 0.0

    for u, v, data in G.edges(data=True):
        try:
            w = float(data["weight"])
        except (KeyError, ValueError):
            print(f"Warning: Edge ({u}, {v}) missing or has non-numeric weight '{data.get('weight')}', skipping.")
            continue # Skip edges without valid weights

        weights.append(w)
        total_weight_sum += w # Accumulate total weight

        # Check if source or target node was skipped during ID parsing
        if u not in id2layer or v not in id2layer:
             print(f"Warning: Skipping edge ({u}, {v}) because one or both nodes couldn't be mapped to a layer.")
             continue

        src_layer, _ = id2layer[u]
        tgt_layer, _ = id2layer[v]
        delta = tgt_layer - src_layer          # positive ⇒ forward in depth

        if delta == 0:
            same_layer_weight_sum += w
            same_layer_total_edges += 1
            if w > 0.5:
                same_layer_strong_connections += 1
        else:
            inter_layer_weight_sum += w
            inter_layer_total_edges += 1
            if w > 0.5:
                inter_layer_strong_connections += 1

            # "close" and "far" windows (ignore backward edges, i.e. delta ≤ 0)
            if delta > 0:
                if 1 <= delta <= 2:
                    close2_weight_sum += w
                elif delta >= 3:
                    far2_weight_sum += w

                if 1 <= delta <= 4:
                    close4_weight_sum += w
                elif delta >= 5:
                    far4_weight_sum += w

    total_edges = len(weights) # Keep total edge count for stats

    # ------------------------------------------------------------------
    # Calculate Normalization Factors based on max_layer
    # ------------------------------------------------------------------
    norm_close2 = 0
    norm_far2 = 0
    norm_close4 = 0
    norm_far4 = 0

    # Same/Inter layer normalizers
    norm_same_layer = max_layer + 1
    norm_inter_layer = (max_layer + 1) * max_layer if max_layer >= 0 else 0

    if max_layer >= 1:
        norm_close2 = max(0, 2 * max_layer - 1) # Delta 1 + Delta 2 connections

    if max_layer >= 3:
        # Sum of (l-2) for l from 3 to max_layer => Sum of j for j from 1 to max_layer-2
        norm_far2 = (max_layer - 2) * (max_layer - 1) // 2

    if max_layer == 1:
        norm_close4 = 1
    elif max_layer == 2:
        norm_close4 = 3 # (L0->L1) + (L0->L2, L1->L2)
    elif max_layer >= 3:
        # Sum of min(l, 4) for l from 1 to max_layer
        # 1 + 2 + 3 + 4*(max_layer-3) = 6 + 4*max_layer - 12 = 4*max_layer - 6
        norm_close4 = 4 * max_layer - 6

    if max_layer >= 5:
        # Sum of (l-4) for l from 5 to max_layer => Sum of j for j from 1 to max_layer-4
        norm_far4 = (max_layer - 4) * (max_layer - 3) // 2

    # Calculate normalized average weights
    avg_close2 = (close2_weight_sum / norm_close2) if norm_close2 > 0 else None
    avg_far2 = (far2_weight_sum / norm_far2) if norm_far2 > 0 else None
    avg_close4 = (close4_weight_sum / norm_close4) if norm_close4 > 0 else None
    avg_far4 = (far4_weight_sum / norm_far4) if norm_far4 > 0 else None

    # Handle None for summation when calculating total normalized weights for window percentages
    val_avg_close2 = avg_close2 if avg_close2 is not None else 0
    val_avg_far2 = avg_far2 if avg_far2 is not None else 0
    val_avg_close4 = avg_close4 if avg_close4 is not None else 0
    val_avg_far4 = avg_far4 if avg_far4 is not None else 0

    total_norm_2_window = val_avg_close2 + val_avg_far2
    total_norm_4_window = val_avg_close4 + val_avg_far4

    # Calculate percentages based on normalized weights within each window
    pct_norm_close2 = (val_avg_close2 / total_norm_2_window * 100) if total_norm_2_window > 0 else 0
    pct_norm_far2 = (val_avg_far2 / total_norm_2_window * 100) if total_norm_2_window > 0 else 0
    pct_norm_close4 = (val_avg_close4 / total_norm_4_window * 100) if total_norm_4_window > 0 else 0
    pct_norm_far4 = (val_avg_far4 / total_norm_4_window * 100) if total_norm_4_window > 0 else 0

    # Calculate normalized average weights for Same/Inter
    avg_same_layer = (same_layer_weight_sum / norm_same_layer) if norm_same_layer > 0 else None
    avg_inter_layer = (inter_layer_weight_sum / norm_inter_layer) if norm_inter_layer > 0 else None

    # Calculate percentage of strong connections within each type
    pct_strong_same = (same_layer_strong_connections / same_layer_total_edges * 100) if same_layer_total_edges > 0 else 0.0
    pct_strong_inter = (inter_layer_strong_connections / inter_layer_total_edges * 100) if inter_layer_total_edges > 0 else 0.0

    # ------------------------------------------------------------------
    # Print the report
    # ------------------------------------------------------------------
    # pct function is for global percentages of total_weight_sum
    def pct(weight_sum: float) -> str:
        if total_weight_sum == 0: return "  N/A " # Handle division by zero
        return f"{(weight_sum / total_weight_sum * 100):5.2f}%"

    # Helper for printing already-normalized average value
    def format_avg_value(avg_val):
        if avg_val is not None:
            return f"{avg_val:10.4f}"
        else:
            return f"{'N/A':>10}"

    print(f"File: {Path(gml_path).name}\n"
          "==============================")
    print(f"Total edges processed       : {total_edges}")
    print(f"Total edge weight sum       : {total_weight_sum:.6f}")
    if weights: # Avoid errors if no valid weights found
        print(f"Mean edge weight (r²)       : {stats.fmean(weights):.6f}")
        print(f"Median edge weight          : {stats.median(weights):.6f}")
        print(f"Min / Max edge weight       : {min(weights):.6f} / {max(weights):.6f}\n")
    else:
        print("Mean/Median/Min/Max edge weight : N/A (no valid weights found)\\n")

    print("Edge Type Breakdown (Sum | Avg Weight per Conn Type | % of total weight sum | # Edges > 0.5 r² | % of Type > 0.5 r²)")
    print(f"(Normalization factors: Same={norm_same_layer}, Inter={norm_inter_layer})")
    print("--------------------------------------------------------------------------------------------------------------------")
    print(f"Same‑layer                  : {same_layer_weight_sum:10.4f} | {format_avg_value(avg_same_layer)} | {pct(same_layer_weight_sum)} | {same_layer_strong_connections:8d} | {pct_strong_same:17.2f}%")
    print(f"Inter‑layer                 : {inter_layer_weight_sum:10.4f} | {format_avg_value(avg_inter_layer)} | {pct(inter_layer_weight_sum)} | {inter_layer_strong_connections:8d} | {pct_strong_inter:17.2f}%\\n")

    print("Inter‑layer distance windows (Sum | Avg Weight per Conn Type, excluding same‑layer & backward)")
    print("  % Total Weight: Percentage of total graph weight sum.")
    print("  % Window Avg  : Percentage relative to the sum of Avg Weights within each window.")
    print(f"(Normalization factors: C2={norm_close2}, F2={norm_far2}, C4={norm_close4}, F4={norm_far4})")
    print("                                                            Sum |   Avg/Conn | % Total Weight | % Window Avg")
    print("  • 2‑layer window:")
    print(f"      Close  (≤2)           : {close2_weight_sum:10.4f} | {format_avg_value(avg_close2)} | {pct(close2_weight_sum)} | {pct_norm_close2:5.2f}%")
    print(f"      Far    (≥3)           : {far2_weight_sum:10.4f} | {format_avg_value(avg_far2)} | {pct(far2_weight_sum)} | {pct_norm_far2:5.2f}%")
    print("  • 4‑layer window:")
    print(f"      Close  (≤4)           : {close4_weight_sum:10.4f} | {format_avg_value(avg_close4)} | {pct(close4_weight_sum)} | {pct_norm_close4:5.2f}%")
    print(f"      Far    (≥5)           : {far4_weight_sum:10.4f} | {format_avg_value(avg_far4)} | {pct(far4_weight_sum)} | {pct_norm_far4:5.2f}%")


if __name__ == "__main__":
    analyze_connectivity("visualisation_helper/thought_graph_allenai_c4_l8b_key.gml")   # ← replace with your file
