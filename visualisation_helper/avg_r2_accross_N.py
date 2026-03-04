import matplotlib.pyplot as plt
import numpy as np
import os

# Define the PLOT_STYLE based on the style used in recompute_r2_from_top_n.py
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

# Apply plot style using rcParams
plt.rcParams['font.family'] = PLOT_STYLE.get('font_family', 'sans-serif')
plt.rcParams['axes.facecolor'] = PLOT_STYLE.get('background_color', 'white')
plt.rcParams['figure.facecolor'] = PLOT_STYLE.get('background_color', 'white')
plt.rcParams['axes.spines.top'] = PLOT_STYLE.get('spine_top_visible', True)
plt.rcParams['axes.spines.right'] = PLOT_STYLE.get('spine_right_visible', True)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = PLOT_STYLE.get('grid_linestyle', '--')
plt.rcParams['grid.linewidth'] = PLOT_STYLE.get('grid_linewidth', 0.5)
plt.rcParams['grid.alpha'] = PLOT_STYLE.get('grid_alpha', 0.7)

# Data extracted from the image
num_reference_heads = [1, 2, 3, 4, 5]
key_values = [0.70, 0.78, 0.82, 0.84, 0.86]
value_values = [0.56, 0.68, 0.73, 0.77, 0.80]
query_values = [0.64, 0.72, 0.76, 0.79, 0.81]

bar_width = 0.25
x = np.arange(len(num_reference_heads))

# Create the plot
fig = plt.figure(figsize=(12, 8))  # Increased figure size

# Use colors from PLOT_STYLE['colors_list']
plt.bar(x - bar_width, key_values, width=bar_width, label='Key', 
       color=PLOT_STYLE['colors_list'][0], alpha=PLOT_STYLE['opacity_default'])
plt.bar(x, query_values, width=bar_width, label='Query', 
       color=PLOT_STYLE['colors_list'][1], alpha=PLOT_STYLE['opacity_default'])
plt.bar(x + bar_width, value_values, width=bar_width, label='Value', 
       color=PLOT_STYLE['colors_list'][2], alpha=PLOT_STYLE['opacity_default'])

# Add labels and formatting with the new style
plt.xlabel('Number of Reference Heads (N)', fontsize=PLOT_STYLE['label_fontsize'], fontweight=PLOT_STYLE['fontweight'])
plt.ylabel('Mean $R^2$', fontsize=PLOT_STYLE['label_fontsize'], fontweight=PLOT_STYLE['fontweight'])
# Title removed as requested
plt.xticks(x, num_reference_heads, fontsize=PLOT_STYLE['tick_fontsize'])
plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'])
plt.ylim(0.5, 0.9)
plt.legend(fontsize=PLOT_STYLE['legend_fontsize'])
plt.grid(True, linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])

plt.tight_layout()

# Define output paths for saving
output_dir = "plots"  # User confirmed this directory always exists
output_base_filename = "avg_r2_across_N"
output_path_pdf = os.path.join(output_dir, f"{output_base_filename}.pdf")
output_path_png = os.path.join(output_dir, f"{output_base_filename}.png")

# Save the plots
print(f"Saving PDF plot to: {output_path_pdf}")
fig.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())

print(f"Saving PNG plot to: {output_path_png}")
fig.savefig(output_path_png, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)

# Display the plot
plt.show()
