# Make a plot of the overlap dimensions for all layers for each checkpoint
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import torch
from transformers import AutoModelForCausalLM
import argparse
import os

# Plotting style from falcon310b_truthful_qa_plot.py
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
    'title_fontsize': 34,
    'label_fontsize': 34,
    'tick_fontsize': 24,
    'legend_fontsize': 26,
    'fontweight': 'bold',
    'plot_edgecolor': 'black',
    'spine_top_visible': True,
    'spine_right_visible': True,
}

# ---------- Plot helpers ------------------------------------------------------

def _apply_plot_style(ax):
    fig = plt.gcf()
    fig.set_facecolor(PLOT_STYLE['background_color'])
    ax.set_facecolor(PLOT_STYLE['background_color'])
    ax.spines['top'].set_visible(PLOT_STYLE['spine_top_visible'])
    ax.spines['right'].set_visible(PLOT_STYLE['spine_right_visible'])
    ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])
    # Potentially set font family for ticks if needed, e.g.
    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontname(PLOT_STYLE['font_family'])


# Define checkpoints to skip
skip_checkpoints = [
    "stage1-step50000-tokens210B",
    "stage1-step119000-tokens500B",
    "stage1-step250000-tokens1049B",
    "stage1-step500000-tokens2098B",
    "stage1-step599000-tokens2513B",
]

# ---------- New: Checkpoint metadata & overlap computation helpers ----------

# Mapping from checkpoint revision to shorter alias (optional)
revision_to_alias = {
    "stage1-step150-tokens1B": "o7b1b",
    "stage1-step1000-tokens5B": "o7b5b",
    "stage1-step10000-tokens42B": "o7b42b",
    "stage1-step101000-tokens424B": "o7b424b",
    "stage1-step928646-tokens3896B": "o7b3896b",
}

# Sorted list of checkpoints by training step
olmo2_checkpoints = sorted(
    revision_to_alias.keys(),
    key=lambda x: int(re.search(r"step(\d+)", x).group(1))
)


# ---------- Linear-algebra helpers ------------------------------------------------


def total_overlap_dim(W_list):
    """Compute total sub-space overlap dimension among a list of column matrices.

    For matrices W_i with orthonormal (or arbitrary) columns, the dimension of the
    intersection of their column spaces equals ∑ rank(W_i) − rank(concat(W_i)).
    """
    # Convert to CPU / fp32 for stable rank computation
    W_list = [w.to(torch.float32).cpu() for w in W_list]

    sum_ranks = sum(int(torch.linalg.matrix_rank(w)) for w in W_list)
    W_cat = torch.cat(W_list, dim=1)
    rank_cat = int(torch.linalg.matrix_rank(W_cat))

    return sum_ranks - rank_cat


# ---------- Core computation: generate overlap data like the notebook ----------


def compute_overlap_results(attention_state: str):
    """Return a dict  {checkpoint -> {layer_idx -> overlap_dim}} for a projection.

    The logic mirrors experimentation/subspace_alignment.ipynb but is executed on-the-fly so
    that JSON files are no longer required.
    """
    assert attention_state in {"q", "k", "v"}, "attention_state must be one of 'q', 'k', or 'v'"

    overall_results = {}

    for ckpt in olmo2_checkpoints:
        print(f"\n>>> Processing checkpoint: {ckpt}  ({attention_state}_proj)")

        # Load model shard (FP16 keeps VRAM lower). Switch to CPU if no CUDA.
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMo-2-1124-7B",
            revision=ckpt,
            low_cpu_mem_usage=True,
            cache_dir="../../cache",
            torch_dtype=torch.float16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        n_heads = model.config.num_attention_heads

        layer_overlaps = {}

        for layer_idx, layer in enumerate(model.model.layers):
            # Obtain the relevant projection weight
            proj_weight = getattr(layer.self_attn, f"{attention_state}_proj").weight  # (out, in)
            proj_t = proj_weight.transpose(0, 1)  # shape (in, out)

            head_dim = proj_t.shape[1] // n_heads

            head_matrices = [
                proj_t[:, i * head_dim : (i + 1) * head_dim]
                for i in range(n_heads)
            ]

            overlap_dim = total_overlap_dim(head_matrices)
            layer_overlaps[layer_idx] = overlap_dim

        # Store & clean up
        overall_results[ckpt] = layer_overlaps

        # Free GPU/CPU memory ASAP
        del model
        torch.cuda.empty_cache()

    return overall_results


# ---------- Updated data-processing helpers ----------------------------------


def get_avg_overlap_data(overall_results):
    """Compute (steps, avg_overlap) from the results dict."""
    checkpoint_steps = []
    checkpoint_avgs = []

    for checkpoint, layers_data in overall_results.items():
        if checkpoint in skip_checkpoints:
            continue
        step_match = re.search(r"step(\d+)", checkpoint)
        if not step_match:
            continue
        step_num = int(step_match.group(1))

        avg_overlap = sum(layers_data.values()) / len(layers_data)
        checkpoint_steps.append(step_num)
        checkpoint_avgs.append(avg_overlap)

    # Sort by step
    order = np.argsort(checkpoint_steps)
    return [checkpoint_steps[i] for i in order], [checkpoint_avgs[i] for i in order]


def create_plots(overall_results, attention_state: str, output_dir: str):
    """Replicates the two-panel plotting logic using an in-memory results dict."""

    # -------- Plot 1: per-layer overlap for each checkpoint -----------------
    plt.figure(figsize=(12, 8))

    for checkpoint, layers_data in overall_results.items():
        if checkpoint in skip_checkpoints:
            continue
        try:
            layer_indices = sorted(layers_data.keys())
            overlap_values = [layers_data[idx] for idx in layer_indices]

            color_index = list(overall_results.keys()).index(checkpoint) % len(PLOT_STYLE["colors_list"])

            step_match = re.search(r"step(\d+)", checkpoint)
            legend_label = f"Step {step_match.group(1)}" if step_match else checkpoint

            plt.plot(
                layer_indices,
                overlap_values,
                marker="o",
                linestyle="-",
                label=legend_label,
                color=PLOT_STYLE["colors_list"][color_index],
                linewidth=3.5,
                markersize=12,
            )
        except KeyError as e:
            print(f"Error processing checkpoint {checkpoint}: {e}")
            continue

    plt.xlabel("Layer Index", fontsize=PLOT_STYLE["label_fontsize"], fontfamily=PLOT_STYLE["font_family"], fontweight=PLOT_STYLE["fontweight"])
    plt.ylabel("Overlap Dimension", fontsize=PLOT_STYLE["label_fontsize"], fontfamily=PLOT_STYLE["font_family"], fontweight=PLOT_STYLE["fontweight"])
    plt.grid(True, linestyle=PLOT_STYLE["grid_linestyle"], linewidth=PLOT_STYLE["grid_linewidth"], alpha=PLOT_STYLE["grid_alpha"])
    plt.legend(loc="best", fontsize=PLOT_STYLE["legend_fontsize"])
    plt.ylim(0, 250)
    _apply_plot_style(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attention_state}_proj_overlap_dimensions_by_layer.pdf"), dpi=300)
    plt.close()

    # -------- Plot 2: avg overlap vs steps ----------------------------------
    plt.figure(figsize=(12, 8))
    steps, avgs = get_avg_overlap_data(overall_results)

    plt.plot(
        steps,
        avgs,
        marker="o",
        linestyle="-",
        linewidth=3.5,
        markersize=12,
        color=PLOT_STYLE["colors_list"][0],
    )
    plt.xscale("log")
    plt.xlabel("Training Steps", fontsize=PLOT_STYLE["label_fontsize"], fontfamily=PLOT_STYLE["font_family"], fontweight=PLOT_STYLE["fontweight"])
    plt.ylabel("Avg Overlap Dimension", fontsize=PLOT_STYLE["label_fontsize"], fontfamily=PLOT_STYLE["font_family"], fontweight=PLOT_STYLE["fontweight"])
    plt.grid(True, linestyle=PLOT_STYLE["grid_linestyle"], linewidth=PLOT_STYLE["grid_linewidth"], alpha=PLOT_STYLE["grid_alpha"])
    _apply_plot_style(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attention_state}_proj_avg_overlap_vs_steps.pdf"), dpi=300)
    plt.close()


def create_combined_avg_plot(k_results, q_results, v_results, output_dir: str):
    plt.figure(figsize=(12, 8))

    k_steps, k_avgs = get_avg_overlap_data(k_results)
    q_steps, q_avgs = get_avg_overlap_data(q_results)
    v_steps, v_avgs = get_avg_overlap_data(v_results)

    plt.plot(k_steps, k_avgs, marker='o', linestyle='-', linewidth=6.0, markersize=14,
             color=PLOT_STYLE['colors_list'][0], label='K Projection')
    plt.plot(q_steps, q_avgs, marker='s', linestyle='-', linewidth=6.0, markersize=14,
             color=PLOT_STYLE['colors_list'][1], label='Q Projection')
    plt.plot(v_steps, v_avgs, marker='^', linestyle='-', linewidth=6.0, markersize=14,
             color=PLOT_STYLE['colors_list'][2], label='V Projection')

    plt.xscale('log')
    plt.xlabel('Training Steps', fontsize=PLOT_STYLE['label_fontsize'], fontfamily=PLOT_STYLE['font_family'], fontweight=PLOT_STYLE['fontweight'])
    plt.ylabel('Avg Overlap Dimension', fontsize=PLOT_STYLE['label_fontsize'], fontfamily=PLOT_STYLE['font_family'], fontweight=PLOT_STYLE['fontweight'])
    plt.grid(True, linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])
    plt.legend(loc='best', fontsize=PLOT_STYLE['legend_fontsize'])
    _apply_plot_style(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_avg_overlap_vs_steps.pdf'), dpi=300)
    plt.close()


# ---------- Execute full pipeline -------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute overlap-dimension plots for OLMo checkpoints")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory in which to save the generated plot PDFs")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    k_results = compute_overlap_results('k')
    q_results = compute_overlap_results('q')
    v_results = compute_overlap_results('v')

    create_plots(k_results, 'k', args.output_dir)
    create_plots(q_results, 'q', args.output_dir)
    create_plots(v_results, 'v', args.output_dir)

    create_combined_avg_plot(k_results, q_results, v_results, args.output_dir)

    print(f"All plots created successfully in {args.output_dir}!")
