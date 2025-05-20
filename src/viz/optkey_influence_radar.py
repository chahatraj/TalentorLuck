import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import os

# ========== CONFIG ==========
axis = "religion"
model = "qwen_32b"

input_dir = f'../../outputs/closed_ended/actor_actor/{axis}/{model}'
output_dir = f'../../figs/closed_ended/actor_actor/{axis}/{model}/radar/test'
os.makedirs(output_dir, exist_ok=True)

files = {
    "Success": f"{input_dir}/closed_ended_both_success_{model}_{axis}_all_1_runs.csv",
    "Failure": f"{input_dir}/closed_ended_both_failure_{model}_{axis}_all_1_runs.csv"
}

option_labels = {
    "opt1_higheffort": "High Effort",
    "opt2_highability": "High Ability",
    "opt3_easytask": "Easy Task",
    "opt4_goodluck": "Good Luck",
    "opt1_loweffort": "Low Effort",
    "opt2_lowability": "Low Ability",
    "opt3_difficulttask": "Difficult Task",
    "opt4_badluck": "Bad Luck"
}

domain_label_map = {
    "sports": "Sports",
    "education": "Education",
    "workplace": "Workplace",
    "healthcare": "Healthcare",
    "technology": "Technology",
    "media": "Media",
    "law_and_policy": "Law & Policy",
    "economics": "Economics",
    "environment": "Environment",
    "art_and_leisure": "Arts & Leisure"
}

# Aesthetic-matched attribution color palette
colors = ["#9880bf", "#008080", "#e37317", "#CD5C5C"]  # Effort, Ability, Task, Luck

# ========== LOAD AND AGGREGATE ==========
# def get_option_means(filepath, mode):
#     df = pd.read_csv(filepath)
#     if mode == "Success":
#         opt_cols = ['optX1_higheffort', 'optX2_highability', 'optX3_easytask', 'optX4_goodluck']
#     else:
#         opt_cols = ['optY1_loweffort', 'optY2_lowability', 'optY3_difficulttask', 'optY4_badluck']

#     domain_means = df.groupby('domain')[opt_cols].mean().reset_index()
#     option_means_by_domain = {
#         option_labels[col]: dict(zip(domain_means['domain'], domain_means[col]))
#         for col in opt_cols
#     }
#     return option_means_by_domain

def get_option_means(filepath, mode):
    df = pd.read_csv(filepath)
    
    # Dynamically determine which columns are available
    if mode == "Success":
        if "optX1_higheffort" in df.columns:
            opt_cols = ['optX1_higheffort', 'optX2_highability', 'optX3_easytask', 'optX4_goodluck']
        else:
            opt_cols = ['optX1_loweffort', 'optX2_lowability', 'optX3_difficulttask', 'optX4_badluck']
    else:
        opt_cols = ['optY1_loweffort', 'optY2_lowability', 'optY3_difficulttask', 'optY4_badluck']

    # Use only domains that exist in both success/failure files
    df = df[df['domain'].isin(domain_label_map.keys())]

    domain_means = df.groupby('domain')[opt_cols].mean().reset_index()
    
    # Map column names to readable labels (fallback to raw col name if not in map)
    option_means_by_domain = {
        option_labels.get(col, col): dict(zip(domain_means['domain'], domain_means[col]))
        for col in opt_cols
    }
    return option_means_by_domain


data_success = get_option_means(files["Success"], "Success")
data_failure = get_option_means(files["Failure"], "Failure")

# ========== SHARED RADAR CONFIG ==========
raw_categories = sorted(next(iter(data_success.values())).keys())
categories = [domain_label_map.get(cat, cat.title()) for cat in raw_categories]

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ========== PLOT ==========
fig, axs = plt.subplots(1, 2, figsize=(22, 12), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

for ax, (mode, option_data) in zip(axs, [("Success", data_success), ("Failure", data_failure)]):
    for i, (label, score_dict) in enumerate(option_data.items()):
        values = [score_dict[c] for c in raw_categories]
        values += values[:1]
        ax.plot(angles, values, label=label, linewidth=4, color=colors[i], marker='o', markersize=15)
        ax.fill(angles, values, alpha=0.3, color=colors[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=25, fontweight='semibold')

    # Adjust x-tick positions inward
    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_position((x, y + 0.05))

    ax.set_ylim(0, 0.45)
    ax.set_yticks([0.1, 0.2, 0.3])
    ax.set_yticklabels(['0.1', '0.2', '0.3'], fontsize=16, fontweight='medium')
    ax.tick_params(axis='y', left=False, labelsize=25)
    ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.6)

    # Set polar spine
    ax.spines['polar'].set_color("lightgray")
    ax.spines['polar'].set_linestyle("--")
    ax.spines['polar'].set_linewidth(1.2)

    # Title
    ax.set_title(f"{mode}", fontsize=40, weight='bold', pad=30)

# === Legend ===
attribution_labels = ["Effort", "Ability", "Difficulty", "Luck"]
legend_handles = [
    Line2D([0], [0], color=color, marker='o', linestyle='-', linewidth=3.5, markersize=15)
    for color in colors
]

legend_font = FontProperties(weight='bold', size=40)
title_font = FontProperties(weight='bold', size=42)

fig.legend(
    handles=legend_handles,
    labels=attribution_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.07),
    ncol=4,
    frameon=False,
    prop=legend_font,
    title="",
    title_fontproperties=title_font
)

plt.tight_layout()
plt.savefig(f"{output_dir}/radar_combined_{axis}_{model}.pdf", dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Radar subplot saved: {output_dir}/radar_combined_{axis}_{model}.pdf")
