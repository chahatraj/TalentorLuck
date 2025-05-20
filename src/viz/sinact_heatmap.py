import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import numpy as np
import os

# ========== CONFIGURATION ==========
axis = "religion"  # or "religion"
model = "aya_expanse_8b"  # or "gemma_3_27b_it", "llama3_3_70b_it"
mode = "failure"  # or "success"

# File path (same as in original barplot code)
csv_path = f'../../outputs/closed_ended/single_actor/{axis}/{model}/closed_ended_{mode}_{model}_{axis}_all_1_runs.csv'
df = pd.read_csv(csv_path)

# Output directory
output_dir = f'../../figs/closed_ended/single_actor/{axis}/{model}/heatmap'
os.makedirs(output_dir, exist_ok=True)

# ========== STYLING ==========
sns.set_context("paper")
sns.set(style='whitegrid')

# Custom blended colormap
vlag_r = plt.get_cmap("vlag_r")
spectral = plt.get_cmap("Spectral")
vlag_colors = vlag_r(np.linspace(0, 1, 256))
spectral_colors = spectral(np.linspace(0, 1, 256))
blended_colors = 0.6 * vlag_colors + 0.4 * spectral_colors
blended_cmap_r = ListedColormap(blended_colors[::-1])

# ========== SELECT COLUMNS ==========
if mode == "success":
    opt_cols = ['opt1_higheffort', 'opt2_highability', 'opt3_easytask', 'opt4_goodluck']
else:
    opt_cols = ['opt1_loweffort', 'opt2_lowability', 'opt3_difficulttask', 'opt4_badluck']

nationalities = sorted(df['religion'].dropna().unique())
genders = sorted(df['gender'].dropna().unique())
domains = sorted(df['domain'].dropna().unique())

# ========== COMBINED HEATMAP ==========
all_grouped = (
    df.groupby(['religion', 'gender'])[opt_cols]
    .mean()
    .reset_index()
    .melt(id_vars=['religion', 'gender'], var_name='option', value_name='prob')
)

all_pivot = all_grouped.pivot_table(
    index='religion',
    columns=['option', 'gender'],
    values='prob'
).reindex(nationalities)

plt.figure(figsize=(14, 10))
ax = sns.heatmap(all_pivot, annot=True, cmap=blended_cmap_r, center=0.25, cbar=True)

# Custom x-tick labels
new_labels = [
    f"{opt.replace('opt', '').split('_')[1].capitalize()} ({gender[0].upper()})"
    for opt, gender in all_pivot.columns
]
ax.set_xticklabels(new_labels, rotation=45, ha='right')

# Draw border boxes around gender pairs
n_nationalities = len(all_pivot)
for i in range(0, len(all_pivot.columns), 2):
    rect = patches.Rectangle(
        (i, 0), 2, n_nationalities,
        linewidth=2,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(rect)

plt.title(f'Average Attribution Probabilities by religion and Gender (All Domains)\nMode: {mode.capitalize()}')
plt.xlabel('Option – Gender')
plt.ylabel('religion')
plt.tight_layout()
plt.savefig(f'{output_dir}/{mode}_heatmap_all_domains.png')
plt.close()

# ========== PER-DOMAIN HEATMAPS ==========
for domain in domains:
    domain_df = df[df['domain'] == domain]
    grouped = (
        domain_df.groupby(['religion', 'gender'])[opt_cols]
        .mean()
        .reset_index()
        .melt(id_vars=['religion', 'gender'], var_name='option', value_name='prob')
    )

    pivot = grouped.pivot_table(
        index='religion',
        columns=['option', 'gender'],
        values='prob'
    ).reindex(nationalities)

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(pivot, annot=True, cmap=blended_cmap_r, center=0.25, cbar=True)

    new_labels = [
        f"{opt.replace('opt', '').split('_')[1].capitalize()} ({gender[0].upper()})"
        for opt, gender in pivot.columns
    ]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

    n_nationalities = len(pivot)
    for i in range(0, len(pivot.columns), 2):
        rect = patches.Rectangle(
            (i, 0), 2, n_nationalities,
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.title(f'Average Attribution Probabilities by religion and Gender\nDomain: {domain.capitalize()} | Mode: {mode.capitalize()}')
    plt.xlabel('Option – Gender')
    plt.ylabel('religion')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{mode}_heatmap_{domain}.png')
    plt.close()

print(f"✅ Heatmaps saved to {output_dir}")
