import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model_name = "llama_70b"

SKIP_SEQ_LENGTHS = [512, 2048]

# Read the CSV data
data = pd.read_csv(f"./data/{model_name}_pp_scaling_ttft.csv")

# Extract sequence lengths and pipeline parallel degrees
def extract_info(column_name):
    parts = column_name.split("_")
    seq_length = int(parts[0].split("-")[1])
    pp_degree = int(parts[3].split("-")[1])
    return seq_length, pp_degree

# Prepare data for plotting
plot_data = {}
for column in data.columns:
    if column.startswith("l-") and column.endswith("prefill_e2e_time_median"):
        seq_length, pp_degree = extract_info(column)
        if seq_length not in plot_data:
            plot_data[seq_length] = {}
        plot_data[seq_length][pp_degree] = data[column].iloc[0]

# add the 10M data manually for now
plot_data[1024 * 10] = {}
if model_name == "llama_7b":
    plot_data[1024 * 10][4] = 2410
    plot_data[1024 * 10][8] = 1209
    plot_data[1024 * 10][16] = 639
elif model_name == "llama_70b":
    plot_data[1024 * 10][16] = 3060

# filter out the sequence lengths to skip
plot_data = {seq_length: plot_data[seq_length] for seq_length in plot_data if seq_length not in SKIP_SEQ_LENGTHS}

print(plot_data)

# Sort sequence lengths and pipeline parallel degrees
seq_lengths = sorted(plot_data.keys())
pp_degrees = sorted(set(degree for lengths in plot_data.values() for degree in lengths.keys()).union({1}))

fig, axes = plt.subplots(1, len(seq_lengths), figsize=(3.7 * len(seq_lengths), 3.5), sharey=False)
fig.subplots_adjust(wspace=0.4)

width = 0.7

# Define colors and hatches
# COLORS = ["chocolate", "cadetblue", "forestgreen", "darkviolet", "crimson"]
COLORS =['chocolate', 'cadetblue', 'olive', 'darkslategray', 'darkkhaki', 'darkgoldenrod']
# HATCHES = ['/', '\\', 'x', '+', '*']
HATCHES =  ['\\\\', '-', '//', '', 'x', '']
OPACITY = 0.7

def to_length(seq_length):
    if seq_length < 1024:
        return f"{seq_length}K"
    return f"{seq_length//1024}M"

for i, seq_length in enumerate(seq_lengths):
    ax = axes[i] if len(seq_lengths) > 1 else axes
    x = np.arange(len(pp_degrees))
    
    values = [plot_data[seq_length].get(pp, 0) for pp in pp_degrees]
    
    for j, (pp, value) in enumerate(zip(pp_degrees, values)):
        color = COLORS[j % len(COLORS)]
        hatch = HATCHES[j % len(HATCHES)]
        bar = ax.bar(j, value, width, color=color, alpha=OPACITY, hatch=hatch, 
                     label=f'PP Degree: {pp}' if i == 0 else "", edgecolor='black')
        
        if value == 0:
            ax.plot(j, ax.get_ylim()[0], 'rx', markersize=10)
        
        if pp > 1:
            base_pp = min(plot_data[seq_length].keys())
            if base_pp < pp and plot_data[seq_length].get(base_pp, 0) > 0 and pp in plot_data[seq_length]:
                efficiency = (plot_data[seq_length][base_pp] / value) / (pp / base_pp) * 100
                # ax.text(j, value + 3 * plot_data[seq_length][base_pp] / 100, f"{efficiency:.1f}%",
                #         ha='center', va='bottom', fontsize=12, rotation=90)

                optimal_value = plot_data[seq_length][base_pp] / (pp / base_pp)
                ax.plot([j - width/2, j + width/2], [optimal_value, optimal_value],
                        color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_title(f"Sequence Length: {to_length(seq_length)}", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pp_degrees, fontsize=14, rotation=45)
    ax.set_xlabel("SPP Degree", fontsize=16, fontweight="bold")
    
    if i == 0:
        ax.set_ylabel("Prefill Latency (s)", fontsize=16, fontweight="bold")
    
    ax.grid(True, linestyle="--", axis='y', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.1)

plt.tight_layout()
plt.savefig(f"./outputs/{model_name}_pp_ttft_scaling.png", bbox_inches='tight', dpi=300)
plt.savefig(f"./outputs/{model_name}_pp_ttft_scaling.pdf", bbox_inches='tight')