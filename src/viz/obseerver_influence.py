import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import wilcoxon
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# === Custom colormap ===
colors = ["#9c213e", "#E6E6E6", "#046d70"]  # Lavender → Gray → Teal
delta_cmap = LinearSegmentedColormap.from_list("LavenderTealDiverge", colors, N=256)

# === CONFIG ===
axis = "nationality"
model = "qwen_32b"

success_path = f"../../outputs/closed_ended/actor_observer/{axis}/{model}/closed_ended_success_{model}_{axis}_all_1_runs.csv"
failure_path = f"../../outputs/closed_ended/actor_observer/{axis}/{model}/closed_ended_failure_{model}_{axis}_all_1_runs.csv"
output_dir = f"../../figs/closed_ended/actor_observer/{axis}/{model}/dimension_influence"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
df_success = pd.read_csv(success_path)
df_failure = pd.read_csv(failure_path)

df_success["outcome"] = "Success"
df_success["metric"] = (
    df_success["opt1_higheffort"] + df_success["opt2_highability"]
    - df_success["opt3_easytask"] - df_success["opt4_goodluck"]
)

df_failure["outcome"] = "Failure"
df_failure["metric"] = (
    df_failure["opt1_loweffort"] + df_failure["opt2_lowability"]
    - df_failure["opt3_difficulttask"] - df_failure["opt4_badluck"]
)

df = pd.concat([df_success, df_failure], ignore_index=True)

# === Label cleaner ===
def clean_label(label):
    label = str(label).strip().lower()
    label = re.sub(r'^(a|an)\s+', '', label)
    label = label.replace("person", "").strip()
    label = label.replace("middle-eastern", "middle\neastern")
    label = label.replace("native american", "native\namerican")
    label = label.replace("russia", "russian")
    label = re.sub(r'\s+', ' ', label)
    label = label.title()
    return '\n'.join(label.split())

# === Attribution keys ===
success_keys = [
    "y_opt1_higheffort", "y_opt2_highability",
    "y_opt3_easytask", "y_opt4_goodluck"
]
failure_keys = [
    "y_opt1_loweffort", "y_opt2_lowability",
    "y_opt3_difficulttask", "y_opt4_badluck"
]

# === Generate heatmaps ===
for outcome in ["Success", "Failure"]:
    y_keys = success_keys if outcome == "Success" else failure_keys
    for y_key in y_keys:
        df_ok = df[(df["outcome"] == outcome) & (df["y_opt_key"] == y_key)].copy()
        if df_ok.empty:
            continue

        for domain in ["ALL"] + sorted(df_ok["domain"].dropna().unique()):
            df_domain = df_ok if domain == "ALL" else df_ok[df_ok["domain"] == domain]
            if df_domain.empty:
                continue

            # pivot = df_domain.pivot_table(
            #     index="dimension1", columns="dimension2", values="metric", aggfunc="mean"
            # ).round(2)

            # if pivot.empty:
            #     continue

            # abs_max = np.abs(pivot.values).max()
            # color_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

            # plt.figure(figsize=(20, 20))
            # ax = sns.heatmap(
            #     pivot,
            #     annot=True,
            #     fmt=".2f",
            #     cmap=delta_cmap,
            #     norm=color_norm,
            #     linewidths=1,
            #     linecolor='black',
            #     cbar_kws={"pad": 0.02},
            #     annot_kws={"fontsize": 18}
            # )
            pivot = df_domain.pivot_table(
                index="dimension1", columns="dimension2", values="metric", aggfunc="mean"
            ).round(2)

            if pivot.empty:
                continue

            # === Compute p-values and annotations ===
            pval_matrix = pd.DataFrame(index=pivot.index, columns=pivot.columns)
            annotations = pivot.copy().astype(str)

            for i in pivot.index:
                for j in pivot.columns:
                    cell_subset = df_domain[
                        (df_domain["dimension1"] == i) & (df_domain["dimension2"] == j)
                    ]
                    if len(cell_subset) >= 5:
                        try:
                            _, pval = wilcoxon(cell_subset["metric"], zero_method="zsplit")
                            pval_matrix.loc[i, j] = pval
                            if pval < 0.05:
                                annotations.loc[i, j] = f"$\\bf{{{annotations.loc[i, j]}}}$"
                        except:
                            pval_matrix.loc[i, j] = np.nan
                    else:
                        pval_matrix.loc[i, j] = np.nan

            # === Plot ===
            abs_max = np.abs(pivot.values).max()
            color_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

            plt.figure(figsize=(20, 20))
            ax = sns.heatmap(
                pivot,
                annot=annotations,
                fmt="",
                cmap=delta_cmap,
                norm=color_norm,
                linewidths=1,
                linecolor='black',
                cbar_kws={"pad": 0.02},
                annot_kws={"fontsize": 18}
            )

            for i, row in enumerate(pivot.index):
                for j, col in enumerate(pivot.columns):
                    p = pval_matrix.loc[row, col]
                    if pd.notna(p) and p < 0.05:
                        ax.annotate('♥', xy=(j + 0.7, i + 0.2), xytext=(8, 8),
                                    textcoords='offset points', color='black',
                                    fontsize=18, ha='center', va='top', annotation_clip=False)

            # Tick styling
            xticklabels = [clean_label(label.get_text()) for label in ax.get_xticklabels()]
            yticklabels = [clean_label(label.get_text()) for label in ax.get_yticklabels()]
            ax.set_xticklabels(xticklabels, rotation=90, ha='right', fontsize=20, fontweight='semibold')
            ax.set_yticklabels(yticklabels, rotation=0, fontsize=22, fontweight='semibold')

            ax.tick_params(axis='x', width=1.5, pad=5)
            ax.tick_params(axis='y', width=1.5)
            ax.collections[0].colorbar.ax.tick_params(labelsize=18, width=2)
            ax.collections[0].colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

            # Border styling
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)

            # Title and labels
            key_clean = y_key.replace("y_opt", "").replace("_", " ").title()
            domain_str = "All Domains" if domain == "ALL" else domain.title()
            ax.set_title(f"{outcome} — {key_clean} — {domain_str}", fontsize=16)
            ax.set_xlabel("Observer Identity", fontsize=14)
            ax.set_ylabel("Actor Identity", fontsize=14)
            plt.tight_layout()

            fname = f"{axis}_{model}_{outcome.lower()}_{y_key}_{domain}_heatmap.pdf"
            out_path = os.path.join(output_dir, fname)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved: {out_path}")
