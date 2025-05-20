import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# === CONFIG ===
axis = "race"  # change to "nationality" or "religion" as needed
model = "aya_expanse_8b"
success_path = f"../../outputs/closed_ended/actor_observer/{axis}/{model}/closed_ended_success_{model}_{axis}_all_1_runs.csv"
failure_path = f"../../outputs/closed_ended/actor_observer/{axis}/{model}/closed_ended_failure_{model}_{axis}_all_1_runs.csv"
name_gender_json = f"../../data/names/{axis}.json"
output_dir = f"../../figuresforpaper/actor_observer/{axis}/{model}"
os.makedirs(output_dir, exist_ok=True)

# === Load ===
df_success = pd.read_csv(success_path)
df_failure = pd.read_csv(failure_path)

# === Annotate outcome and compute metric ===
df_success["outcome"] = "Success"
df_success["metric"] = df_success["opt1_higheffort"] + df_success["opt2_highability"] - df_success["opt3_easytask"] - df_success["opt4_goodluck"]

df_failure["outcome"] = "Failure"
df_failure["metric"] = df_failure["opt1_loweffort"] + df_failure["opt2_lowability"] - df_failure["opt3_difficulttask"] - df_failure["opt4_badluck"]

df = pd.concat([df_success, df_failure], ignore_index=True)

# === Load name-to-gender mapping ===
with open(name_gender_json) as f:
    name_gender_dict = json.load(f)

# === Infer observer gender from name2 ===
def infer_gender(name, dim2):
    dim2_key = dim2.lower().strip()
    if dim2_key not in name_gender_dict:
        return "unknown"
    male_names = set(n.lower() for n in name_gender_dict[dim2_key]["male_names"])
    female_names = set(n.lower() for n in name_gender_dict[dim2_key]["female_names"])
    name_lower = name.lower()
    if name_lower in male_names:
        return "male"
    elif name_lower in female_names:
        return "female"
    else:
        return "unknown"

df["observer_gender"] = df.apply(lambda row: infer_gender(row["name2"], row["dimension2"]), axis=1)
df["gender_pair"] = df["gender"] + " → " + df["observer_gender"]

# === Clean label helper ===
def clean_label(label):
    label = str(label).strip().lower()
    label = re.sub(r'^(a|an)\s+', '', label)
    label = label.replace("person", "").strip()
    label = re.sub(r'\s+', ' ', label)
    return '\n'.join(label.title().split())

# === Color map ===
colors = ["#9c213e", "#E6E6E6", "#046d70"]
delta_cmap = LinearSegmentedColormap.from_list("LavenderTealDiverge", colors, N=256)

# === y_opt_key list ===
success_keys = ["y_opt1_higheffort", "y_opt2_highability", "y_opt3_easytask", "y_opt4_goodluck"]
failure_keys = ["y_opt1_loweffort", "y_opt2_lowability", "y_opt3_difficulttask", "y_opt4_badluck"]

# === Plotting loop ===
for outcome in ["Success", "Failure"]:
    y_keys = success_keys if outcome == "Success" else failure_keys
    for y_key in y_keys:
        filtered = df[(df["outcome"] == outcome) & (df["y_opt_key"] == y_key)]
        if filtered.empty:
            continue
        for gender_pair in filtered["gender_pair"].unique():
            g_df = filtered[filtered["gender_pair"] == gender_pair]
            for domain in g_df["domain"].unique():
                sub_df = g_df[g_df["domain"] == domain]
                if sub_df.empty:
                    continue

                pivot = sub_df.pivot_table(
                    index="dimension2", columns="dimension1", values="metric", aggfunc="mean"
                ).round(2)

                if pivot.empty:
                    continue

                abs_max = np.abs(pivot.values).max()
                color_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap=delta_cmap, norm=color_norm,
                                 linewidths=1, linecolor='black', cbar_kws={"pad": 0.02},
                                 annot_kws={"fontsize": 14})

                ax.set_xticklabels([clean_label(l.get_text()) for l in ax.get_xticklabels()],
                                   rotation=90, ha='right', fontsize=16, fontweight='semibold')
                ax.set_yticklabels([clean_label(l.get_text()) for l in ax.get_yticklabels()],
                                   rotation=0, fontsize=16, fontweight='semibold')

                ax.tick_params(axis='x', width=1.5, pad=5)
                ax.tick_params(axis='y', width=1.5)
                ax.collections[0].colorbar.ax.tick_params(labelsize=14, width=2)
                ax.collections[0].colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(2)

                title = f"{gender_pair} | {outcome} | {y_key.replace('y_opt', '').replace('_', ' ').title()} | {domain}"
                plt.title(title, fontsize=16)
                plt.xlabel("Actor Identity", fontsize=14)
                plt.ylabel("Observer Identity", fontsize=14)
                plt.tight_layout()

                # Save
                filename = f"{axis}_{model}_{gender_pair.replace(' ', '')}_{outcome}_{y_key}_{domain}.pdf"
                out_path = os.path.join(output_dir, filename)
                plt.savefig(out_path, bbox_inches="tight")
                plt.close()

print("✅ All actor-observer heatmaps saved.")
