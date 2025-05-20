import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import os

# ========== CONFIG ==========
axis = "religion"
model = "qwen_32b"  # or "gemma_3_27b_it", "llama3_3_70b_it"
input_dir = f'../../outputs/closed_ended/single_actor/{axis}/{model}'
output_dir = f'../../figs/closed_ended/single_actor/{axis}/{model}/radar'
os.makedirs(output_dir, exist_ok=True)

files = {
    "Success": f"{input_dir}/closed_ended_success_{model}_{axis}_all_1_runs.csv",
    "Failure": f"{input_dir}/closed_ended_failure_{model}_{axis}_all_1_runs.csv"
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

# Custom refined colors (pastel + bold tones)
# colors = ["#C3B1E1", "#008080", "#FFB347", "#CD5C5C"]
colors = ["#9880bf", "#008080", "#e37317", "#CD5C5C"]

# ========== LOAD AND AGGREGATE ==========
def get_option_means(filepath, mode):
    df = pd.read_csv(filepath)
    if mode == "Success":
        opt_cols = ['opt1_higheffort', 'opt2_highability', 'opt3_easytask', 'opt4_goodluck']
    else:
        opt_cols = ['opt1_loweffort', 'opt2_lowability', 'opt3_difficulttask', 'opt4_badluck']

    domain_means = df.groupby('domain')[opt_cols].mean().reset_index()
    option_means_by_domain = {
        option_labels[col]: dict(zip(domain_means['domain'], domain_means[col]))
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
        label.set_position((x, y - 0.05))

    ax.set_ylim(0, 0.35)
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

# Legend on the right
# axs[1].legend(
#     loc='center left',
#     bbox_to_anchor=(1.2, 0.5),
#     fontsize=18,
#     frameon=False,
#     title="Attribution Type",
#     title_fontsize=20
# )

# Create dummy handles for success and failure legends


# success_labels = ["High Effort", "High Ability", "Easy Task", "Good Luck"]
# failure_labels = ["Low Effort", "Low Ability", "Difficult Task", "Bad Luck"]
# handles_success = [Line2D([0], [0], color=c, lw=3.5) for c in colors]
# handles_failure = [Line2D([0], [0], color=c, lw=3.5) for c in colors]


# Success legend (left)
# Success legend (left)
# fig.legend(
#     handles=handles_success,
#     labels=success_labels,
#     loc='lower center',
#     bbox_to_anchor=(0.26, -0.07),
#     ncol=4,
#     fontsize=30,
#     title="Success",
#     # title_fontsize=22,
#     frameon=False,
#     prop={'weight': 'bold', 'size':15},           # ✅ Make legend labels bold
#     title_fontproperties={'weight': 'bold', 'size':15}  # ✅ Make title bold
# )

# # Failure legend (right)
# fig.legend(
#     handles=handles_failure,
#     labels=failure_labels,
#     loc='lower center',
#     bbox_to_anchor=(0.74, -0.07),
#     ncol=4,
#     fontsize=30,
#     title="Failure",
#     # title_fontsize=22,
#     frameon=False,
#     prop={'weight': 'bold', 'size':15},           # ✅ Make legend labels bold
#     title_fontproperties={'weight': 'bold', 'size':15}  # ✅ Make title bold
# )


attribution_labels = ["Effort", "Ability", "Difficulty", "Luck"]
# legend_handles = [Line2D([0], [0], color=c, lw=3.5) for c in colors]
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
    title="Attribution Type",
    title_fontproperties=title_font
)

plt.tight_layout()
plt.savefig(f"{output_dir}/radar_combined_{axis}_{model}.pdf", dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Radar subplot saved: {output_dir}/radar_combined_{axis}_{model}.pdf")
