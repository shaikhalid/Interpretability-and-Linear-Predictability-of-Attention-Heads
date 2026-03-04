import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

# ------------------------------------------------------------------
# PLOT_STYLE (updated to match avg_r2_accross_N.py)
# ------------------------------------------------------------------
PLOT_STYLE = {
    'background_color': 'white',
    'colors_list': [
        '#FF7070',  # Darker Pastel Red (for Key)
        '#FFA050',  # Darker Pastel Orange (for Query)
        '#A993C0',  # Darker Pastel Purple (for Value)
        '#8FB1D9',  # Darker Pastel Blue
        '#7BC86C',  # Darker Pastel Green
        '#B08078',  # Darker Pastel Brown
        '#F098B8',  # Darker Pastel Pink
        '#B0B0B0',  # Darker Pastel Gray
        '#C8C86C',  # Darker Pastel Yellow-Green
        '#7FC0D0'   # Darker Pastel Blue-Teal
    ],
    'hatches_list': ['\\\\', '-', '//', '', 'x', ''],
    'line_color_default': '#1f77b4',
    'marker_color_default': '#1f77b4',
    'opacity_default': 1.0,
    'opacity_fill_kde': 0.4,
    'grid_linestyle': '--',
    'grid_linewidth': 0.5,
    'grid_alpha': 0.7,
    'font_family': 'sans-serif',
    'title_fontsize': 34,
    'label_fontsize': 34,
    'tick_fontsize': 28,
    'legend_fontsize': 26,
    'fontweight': 'bold',
    'plot_edgecolor': 'black',
    'spine_top_visible': True,
    'spine_right_visible': True,
}

# Apply PLOT_STYLE to matplotlib.rcParams
plt.rcParams['font.family'] = PLOT_STYLE['font_family']
plt.rcParams['axes.facecolor'] = PLOT_STYLE['background_color']
plt.rcParams['figure.facecolor'] = PLOT_STYLE['background_color']
plt.rcParams['axes.spines.top'] = PLOT_STYLE['spine_top_visible']
plt.rcParams['axes.spines.right'] = PLOT_STYLE['spine_right_visible']
plt.rcParams['axes.grid'] = True  # Enable grid
plt.rcParams['grid.linestyle'] = PLOT_STYLE['grid_linestyle']
plt.rcParams['grid.linewidth'] = PLOT_STYLE['grid_linewidth']
plt.rcParams['grid.alpha'] = PLOT_STYLE['grid_alpha']
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 1.  Load the stats  ───────────────────────────────────────────────
# ------------------------------------------------------------------
# If you saved the export exactly as shown earlier:
df = pd.read_json("visualisation_helper/parsed_r2_stats.json", orient="records")

# ------------------------------------------------------------------
# 2.  Prepare data for plotting (ensure correct dtypes & states)
# ------------------------------------------------------------------
# Keep all relevant attention states and ensure correct dtypes
plot_df = (
    df.copy()
      .astype({"Num_Ref": int, "Mean_R2": float})
)
# Standardize Attention_State names to uppercase for consistency (e.g., QUERY, KEY, VALUE)
plot_df["Attention_State"] = plot_df["Attention_State"].str.upper()
# Filter for only Q, K, V states if other states exist and are not needed
valid_states = ["QUERY", "KEY", "VALUE"]
plot_df = plot_df[plot_df["Attention_State"].isin(valid_states)]

# ------------------------------------------------------------------
# 3.  Plot – one bar plot per dataset, showing Q, K, V for different Num_Ref
# ------------------------------------------------------------------

# Define abbreviations for legend
ATTENTION_STATE_ABBREVIATIONS = {
    "QUERY": "Query",
    "KEY": "Key",
    "VALUE": "Value"
}

# Map attention states to their respective colors from PLOT_STYLE
ATTENTION_STATE_COLORS = {
    "KEY": PLOT_STYLE['colors_list'][0],     # Red for Key
    "QUERY": PLOT_STYLE['colors_list'][1],   # Orange for Query
    "VALUE": PLOT_STYLE['colors_list'][2]    # Purple for Value
}

# Iterate over each unique model and dataset combination
for model_name, model_grp in plot_df.groupby("Model"):
    for dataset_name, dataset_grp in model_grp.groupby("Dataset"):
        plt.figure(figsize=(12, 8))  # New figure for each dataset within each model
        
        # Get unique num_ref values for this dataset
        num_refs = sorted(dataset_grp["Num_Ref"].unique())
        
        # Dictionary to store R2 values by attention state and num_ref
        r2_by_state = {state: [] for state in valid_states}
        
        # Gather data for each num_ref and attention state
        for num_ref in num_refs:
            num_ref_data = dataset_grp[dataset_grp["Num_Ref"] == num_ref]
            
            for state in valid_states:
                state_data = num_ref_data[num_ref_data["Attention_State"] == state]
                if not state_data.empty:
                    r2_by_state[state].append(state_data["Mean_R2"].values[0])
                else:
                    # If there's no data for this state, use 0
                    r2_by_state[state].append(0)
        
        # Plot bars for each attention state across all num_refs
        bar_width = 0.25
        x = np.arange(len(num_refs))
        
        # Plot bars with consistent color assignment like in avg_r2_accross_N.py
        plt.bar(x - bar_width, r2_by_state["KEY"], width=bar_width, 
                label=ATTENTION_STATE_ABBREVIATIONS["KEY"], 
                color=ATTENTION_STATE_COLORS["KEY"],
                alpha=PLOT_STYLE['opacity_default'])
        
        plt.bar(x, r2_by_state["QUERY"], width=bar_width, 
                label=ATTENTION_STATE_ABBREVIATIONS["QUERY"], 
                color=ATTENTION_STATE_COLORS["QUERY"],
                alpha=PLOT_STYLE['opacity_default'])
        
        plt.bar(x + bar_width, r2_by_state["VALUE"], width=bar_width, 
                label=ATTENTION_STATE_ABBREVIATIONS["VALUE"], 
                color=ATTENTION_STATE_COLORS["VALUE"],
                alpha=PLOT_STYLE['opacity_default'])
        
        # Set x-axis labels and other settings
        plt.xlabel('Number of Reference Heads (N)', fontsize=PLOT_STYLE['label_fontsize'], fontweight=PLOT_STYLE['fontweight'])
        plt.ylabel("Mean $R^2$", fontsize=PLOT_STYLE['label_fontsize'], fontweight=PLOT_STYLE['fontweight'])
        plt.xticks(x, num_refs, fontsize=PLOT_STYLE['tick_fontsize'])
        plt.tick_params(axis='both', which='major', labelsize=PLOT_STYLE['tick_fontsize'])
        plt.ylim(0, 1)
        plt.grid(True, linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])
        plt.legend(fontsize=PLOT_STYLE['legend_fontsize'])
        plt.tight_layout()
        
        # Save the plot
        safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
        safe_dataset_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
        plt.savefig(f"plots/bar_r2_{safe_model_name.lower()}_{safe_dataset_name.lower()}_qkv.pdf", 
                    facecolor=plt.gcf().get_facecolor(), bbox_inches='tight')
        plt.savefig(f"plots/bar_r2_{safe_model_name.lower()}_{safe_dataset_name.lower()}_qkv.png", 
                    facecolor=plt.gcf().get_facecolor(), bbox_inches='tight', dpi=300)
        print(f"Saved plots to: plots/bar_r2_{safe_model_name.lower()}_{safe_dataset_name.lower()}_qkv.pdf/png")
        plt.close()

# print("Bar plots generated successfully.")
