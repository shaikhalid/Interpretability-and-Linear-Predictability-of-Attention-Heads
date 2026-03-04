"""
Defines a pastel color theme for plots.
"""

PASTEL_COLORS = {
    "blue": "#a1c9f4",
    "orange": "#ffb482",
    "green": "#8de5a1",
    "red": "#ff9f9b",
    "purple": "#d0bbff",
    "brown": "#debb9b",
    "pink": "#fab0e4",
    "gray": "#cfcfcf",
    "yellow": "#fff M8", # Changed to lighter yellow - might need adjustment
    "cyan": "#99d8c9",
    "pale_yellow": "#FFFFF0", # Made pale yellow lighter (Ivory)
    "text_grey": "#4F4F4F",  # Added slightly grey color for text
    # Bright Colors
    "bright_blue": "#0000FF",
    "bright_red": "#FF0000",
    "bright_green": "#00FF00"
}

# You can define specific colors and styles for plot elements
PLOT_STYLE = {
    # Colors
    "line_color": PASTEL_COLORS["orange"],
    "marker_color": PASTEL_COLORS["orange"],
    "background_color": PASTEL_COLORS["pale_yellow"],
    # "text_color": PASTEL_COLORS["text_grey"], # Removed custom text color

    # Line Styles
    "line_width": 2.5,

    # Font Styles
    "font_family": "sans-serif",
    # "font_weight": "light", # Changed from semibold to light
    "title_fontsize": 20,  # Was 18 + 2
    "label_fontsize": 18,  # Was 16 + 2
    "tick_fontsize": 16,   # Was 14 + 2

    # Grid Styles
    "grid_linestyle": "--",
    "grid_linewidth": 0.5,

    # Spine Styles
    "spine_top_visible": False,
    "spine_right_visible": False,

    # Add more style elements as needed
} 