import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Plotting style from LAS_calc.py
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
    'tick_fontsize': 20,
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

# Percent of heads predicted
percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# Convert percentages to compression ratios (1/(1-p/100))
compression_ratios = [1, 1.11, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5, 10]

# Corresponding accuracies for Truthful QA
accuracies_truthful_qa = [
    0.3802,
    0.36474908200734396,
    0.37209302325581395,
    0.36964504283965727,
    0.34516523867809057,
    0.35006119951040393,
    0.35107711138310894,
    0.3243574051407589,
    0.32313341493268055,
    0.2729498164014688,
]

# Corresponding accuracies for MMLU Stem
accuracies_mmlu_stem = [
    0.684744687599112,
    0.6799873136695211,
    0.6682524579765303,
    0.6514430700919759,
    0.6238503013003489,
    0.619092927370758,
    0.5543926419283223,
    0.47954329210275926,
    0.41294005708848713,
    0.29624484617824296,
]

# Corresponding accuracies for Winogrande
accuracies_winogrande = [
    0.7119179163378059,
    0.7119179163378059,
    0.7056037884767167,
    0.7008681925808997,
    0.7087608524072613,
    0.7142857142857143,
    0.6953433307024467,
    0.6842936069455406,
    0.6629834254143646,
    0.5785319652722968,
]

# Plot
plt.figure(figsize=(10, 7)) # Increased figure size for better readability

# Convert to numpy arrays for spline interpolation
x = np.array(percentages)  # Still using percentages internally for interpolation
y_truthful_qa = np.array(accuracies_truthful_qa)
y_mmlu_stem = np.array(accuracies_mmlu_stem)
y_winogrande = np.array(accuracies_winogrande)

# Create a new x-axis for a smoother curve
x_smooth = np.linspace(x.min(), x.max(), 300)

# Create the spline object for Truthful QA (cubic spline by default, k=3)
spl_truthful_qa = make_interp_spline(x, y_truthful_qa, k=3)
y_smooth_truthful_qa = spl_truthful_qa(x_smooth)

# Create the spline object for MMLU Stem (cubic spline by default, k=3)
spl_mmlu_stem = make_interp_spline(x, y_mmlu_stem, k=3)
y_smooth_mmlu_stem = spl_mmlu_stem(x_smooth)

# Create the spline object for Winogrande (cubic spline by default, k=3)
spl_winogrande = make_interp_spline(x, y_winogrande, k=3)
y_smooth_winogrande = spl_winogrande(x_smooth)

plt.plot(x_smooth, y_smooth_truthful_qa, color=PLOT_STYLE['colors_list'][0], label='Truthful QA')
plt.plot(x_smooth, y_smooth_mmlu_stem, color=PLOT_STYLE['colors_list'][1], label='MMLU Stem')
plt.plot(x_smooth, y_smooth_winogrande, color=PLOT_STYLE['colors_list'][2], label='Winogrande')

# Add baseline dotted lines
plt.axhline(y=accuracies_truthful_qa[0], color=PLOT_STYLE['colors_list'][0], linestyle='--', linewidth=1)
plt.axhline(y=accuracies_mmlu_stem[0], color=PLOT_STYLE['colors_list'][1], linestyle='--', linewidth=1)
plt.axhline(y=accuracies_winogrande[0], color=PLOT_STYLE['colors_list'][2], linestyle='--', linewidth=1)

# Find accuracies at 50%
accuracy_truthful_qa_50 = accuracies_truthful_qa[percentages.index(50)]
accuracy_mmlu_stem_50 = accuracies_mmlu_stem[percentages.index(50)]
accuracy_winogrande_50 = accuracies_winogrande[percentages.index(50)]

# Calculate absolute accuracy difference
abs_diff_truthful_qa = accuracy_truthful_qa_50 - accuracies_truthful_qa[0]
abs_diff_mmlu_stem = accuracy_mmlu_stem_50 - accuracies_mmlu_stem[0]
abs_diff_winogrande = accuracy_winogrande_50 - accuracies_winogrande[0]

# Annotate difference for Truthful QA
plt.annotate(
    f'{"+" if abs_diff_truthful_qa >= 0 else ""}{abs_diff_truthful_qa * 100:.2f}',  # Multiply by 100
    xy=(50, accuracy_truthful_qa_50),
    xytext=(52, (accuracies_truthful_qa[0] + accuracy_truthful_qa_50) / 2), # Move text closer
    ha='left',
    va='center',
    fontsize=PLOT_STYLE['tick_fontsize'], # Increased fontsize (removed the -6)
    fontweight='bold', # Bold text
    color='black' # Black text for annotation
)
plt.annotate( # Restoring bidirectional arrow for Truthful QA
    '',
    xy=(50, accuracy_truthful_qa_50),
    xytext=(50, accuracies_truthful_qa[0]),
    arrowprops=dict(arrowstyle='<->', linestyle='--', color='gray', shrinkA=0, shrinkB=0, linewidth=1.5)
)

# Annotate difference for MMLU Stem
plt.annotate(
    f'{"+" if abs_diff_mmlu_stem >= 0 else ""}{abs_diff_mmlu_stem * 100:.2f}',  # Multiply by 100
    xy=(50, accuracy_mmlu_stem_50),
    xytext=(48, (accuracies_mmlu_stem[0] + accuracy_mmlu_stem_50) / 2), # Move text closer
    ha='right',
    va='center',
    fontsize=PLOT_STYLE['tick_fontsize'], # Increased fontsize (removed the -6)
    fontweight='bold', # Bold text
    color='black' # Black text for annotation
)
plt.annotate( # Restoring bidirectional arrow for MMLU Stem
    '',
    xy=(50, accuracy_mmlu_stem_50),
    xytext=(50, accuracies_mmlu_stem[0]),
    arrowprops=dict(arrowstyle='<->', linestyle='--', color='gray', shrinkA=0, shrinkB=0, linewidth=1.5)
)

# Annotate difference for Winogrande
plt.annotate(
    f'{"+" if abs_diff_winogrande >= 0 else ""}{abs_diff_winogrande * 100:.2f}',  # Multiply by 100
    xy=(50, accuracy_winogrande_50),
    xytext=(52, (accuracies_winogrande[0] + accuracy_winogrande_50) / 2), # Removed + 0.05 offset
    ha='left',
    va='center',
    fontsize=PLOT_STYLE['tick_fontsize'], # Increased fontsize (removed the -6)
    fontweight='bold', # Bold text
    color='black' # Black text for annotation
)
plt.annotate( # Restoring bidirectional arrow for Winogrande
    '',
    xy=(50, accuracy_winogrande_50),
    xytext=(50, accuracies_winogrande[0]),
    arrowprops=dict(arrowstyle='<->', linestyle='--', color='gray', shrinkA=0, shrinkB=0, linewidth=1.5)
)

plt.xlabel("Compression Ratio", fontsize=PLOT_STYLE['label_fontsize'], fontfamily=PLOT_STYLE['font_family'], fontweight=PLOT_STYLE['fontweight'])
plt.ylabel("Accuracy", fontsize=PLOT_STYLE['label_fontsize'], fontfamily=PLOT_STYLE['font_family'], fontweight=PLOT_STYLE['fontweight'])

# Only display selected compression ratios: 1, 2, 5, and 10
selected_percentages = [0, 50, 80, 90]  # These correspond to 1x, 2x, 5x, 10x
selected_labels = ['1x', '2x', '5x', '10x']
plt.xticks(selected_percentages, selected_labels, fontsize=PLOT_STYLE['tick_fontsize'], fontfamily=PLOT_STYLE['font_family'])

plt.yticks(fontsize=PLOT_STYLE['tick_fontsize'], fontfamily=PLOT_STYLE['font_family'])
plt.ylim(0, 0.75) # Adjusted y-axis limit
plt.grid(linestyle=PLOT_STYLE['grid_linestyle'], linewidth=PLOT_STYLE['grid_linewidth'], alpha=PLOT_STYLE['grid_alpha'])
plt.tight_layout()

plt.legend(fontsize=PLOT_STYLE['legend_fontsize'])

_apply_plot_style(plt.gca())

plt.show()
plt.savefig("visualisation_helper/falcon310b_truthful_qa_plot.pdf")
