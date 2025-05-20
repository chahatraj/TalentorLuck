import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from scipy.stats import wilcoxon
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Define custom colormap
colors = ["#9c213e", "#E6E6E6", "#046d70"]  # Lavender → Gray → Teal
delta_cmap = LinearSegmentedColormap.from_list("LavenderTealDiverge", colors, N=256)
#9c213e
# === CONFIG ===
axis = "race"  # or 'nationality', 'religion'
model = "aya_expanse_8b"  # "gemma_3_27b_it", "llama3_3_70b_it", "qwen_32b"

success_path = f"../../outputs/closed_ended/actor_actor/{axis}/{model}/closed_ended_both_success_{model}_{axis}_all_1_runs.csv"
failure_path = f"../../outputs/closed_ended/actor_actor/{axis}/{model}/closed_ended_both_failure_{model}_{axis}_all_1_runs.csv"
succfail_path = f"../../outputs/closed_ended/actor_actor/{axis}/{model}/closed_ended_success_failure_{model}_{axis}_all_1_runs.csv"

# === Load ===
df_succ = pd.read_csv(success_path)
df_fail = pd.read_csv(failure_path)
df_succfail = pd.read_csv(succfail_path)

def clean_label(label):
    label = str(label).strip().lower()
    label = re.sub(r'^(a|an)\s+', '', label)
    label = label.replace("middle eastern", "middle eastern")
    label = label.replace("native american", "native american")
    label = label.replace("person", "").strip()
    label = re.sub(r'\s+', ' ', label)    
    label = label.title()
    return '\n'.join(label.split())



# === Compute effect metric ===
df_succ['metric_X'] = df_succ['optX1_higheffort'] + df_succ['optX2_highability'] - df_succ['optX3_easytask'] - df_succ['optX4_goodluck']
df_succ['metric_Y'] = df_succ['optY1_higheffort'] + df_succ['optY2_highability'] - df_succ['optY3_easytask'] - df_succ['optY4_goodluck']
df_succ['outcome'] = "Success"

df_fail['metric_X'] = df_fail['optX1_loweffort'] + df_fail['optX2_lowability'] - df_fail['optX3_difficulttask'] - df_fail['optX4_badluck']
df_fail['metric_Y'] = df_fail['optY1_loweffort'] + df_fail['optY2_lowability'] - df_fail['optY3_difficulttask'] - df_fail['optY4_badluck']
df_fail['outcome'] = "Failure"

df_succfail['metric_X'] = df_succfail['optX1_higheffort'] + df_succfail['optX2_highability'] - df_succfail['optX3_easytask'] - df_succfail['optX4_goodluck']
df_succfail['metric_Y'] = df_succfail['optY1_loweffort'] + df_succfail['optY2_lowability'] - df_succfail['optY3_difficulttask'] - df_succfail['optY4_badluck']
df_succfail['outcome'] = "Success-Failure"


# === Combine ===
df = pd.concat([df_succ, df_fail, df_succfail], ignore_index=True)


# === Compute delta ===
df['delta_metric'] = df['metric_X'] - df['metric_Y']
df['delta_metric'] = df['delta_metric'].round(2)


# === Filter: only unequal dimension pairs ===
df = df[df['dimension1'] != df['dimension2']]

# === Output dir ===
output_dir = f'../../figs/closed_ended/actor_actor/{axis}/{model}'
os.makedirs(output_dir, exist_ok=True)

# === For each gender_pair and outcome, plot delta heatmap ===
for gender in df['gender_pair'].unique():
    for outcome in ['Success', 'Failure', 'Success-Failure']:

        subset = df[(df['gender_pair'] == gender) & (df['outcome'] == outcome)]

        pivot = subset.pivot_table(
            index='dimension2',
            columns='dimension1',
            values='delta_metric',
            aggfunc='mean'
        ).round(2)

        # Reorder for consistency
        pivot = pivot.reindex(sorted(pivot.index), axis=0)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        if pivot.empty:
            print(f"⚠️ Skipping plot for {gender} | {outcome} — no data available.")
            continue

        
        # Create pivot and p-value matrices
        delta_pivot = subset.pivot_table(
            index='dimension2',
            columns='dimension1',
            values='delta_metric',
            aggfunc='mean'
        ).round(2)

        pval_matrix = pd.DataFrame(index=delta_pivot.index, columns=delta_pivot.columns)

        # Compute per-cell Wilcoxon test
        for d2 in delta_pivot.index:
            for d1 in delta_pivot.columns:
                cell_subset = df[
                    (df['gender_pair'] == gender) &
                    (df['outcome'] == outcome) &
                    (df['dimension1'] == d1) &
                    (df['dimension2'] == d2)
                ]
                if len(cell_subset) >= 5:
                    try:
                        _, pval = wilcoxon(cell_subset['metric_X'], cell_subset['metric_Y'])
                        pval_matrix.loc[d2, d1] = pval
                    except:
                        pval_matrix.loc[d2, d1] = np.nan
                else:
                    pval_matrix.loc[d2, d1] = np.nan


        # Plot
        plt.figure(figsize=(10, 8))
        # sns.heatmap(pivot, annot=True, cmap='vlag', center=0, fmt=".2f", linewidths=0.5)
        # Determine color range based on max abs value in pivot
        abs_max = np.abs(pivot.values).max()
        color_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        annotations = pivot.copy().astype(str)
        for i in pivot.index:
            for j in pivot.columns:
                p = pval_matrix.loc[i, j]
                if pd.notna(p) and p < 0.05:
                    annotations.loc[i, j] = f"$\\bf{{{annotations.loc[i, j]}}}$"


        # Plot with custom style
        ax = sns.heatmap(pivot, annot=annotations, fmt="", cmap=delta_cmap, norm=color_norm,
                 linewidths=1, linecolor='black',
                 cbar_kws={"pad": 0.02}, annot_kws={"fontsize": 18})


        # Tick styling
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14, fontweight='medium')
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, fontweight='medium')
        # Clean and format tick labels
        xticklabels = [clean_label(label.get_text()) for label in ax.get_xticklabels()]
        yticklabels = [clean_label(label.get_text()) for label in ax.get_yticklabels()]
        ax.set_xticklabels(xticklabels, rotation=90, ha='right', fontsize=20, fontweight='semibold')
        ax.set_yticklabels(yticklabels, rotation=0, fontsize=20, fontweight='semibold')

        ax.tick_params(axis='x', width=1.5, pad=5)
        ax.tick_params(axis='y', width=1.5)
        ax.collections[0].colorbar.ax.tick_params(labelsize=18, width=2)
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))


        # Border styling
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        for i, row in enumerate(pivot.index):
            for j, col in enumerate(pivot.columns):
                p = pval_matrix.loc[row, col]
                if pd.notna(p) and p < 0.05:
                    ax.annotate('♥', xy=(j + 0.7, i + 0.2), xytext=(8, 8),
                                textcoords='offset points', color='black',
                                fontsize=18, ha='center', va='top', annotation_clip=False)


        plt.title(f"{gender} | {outcome} | Δd = d_X - d_Y")
        plt.xlabel("Dimension1 (X)")
        plt.ylabel("Dimension2 (Y)")
        plt.tight_layout()
        out_path = f"{output_dir}/{axis}_{model}_{gender}_{outcome}_bias_barplot.pdf"
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
        plt.close()

print("✅ Δd heatmaps saved successfully.")


# === Also plot domain-wise heatmaps ===
for gender in df['gender_pair'].unique():
    for outcome in ['Success', 'Failure', 'Success-Failure']:
        for domain in df['domain'].unique():
            subset = df[(df['gender_pair'] == gender) & (df['outcome'] == outcome) & (df['domain'] == domain)]

            pivot = subset.pivot_table(
                index='dimension2',
                columns='dimension1',
                values='delta_metric',
                aggfunc='mean'
            ).round(2)

            # Reorder
            pivot = pivot.reindex(sorted(pivot.index), axis=0)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            if pivot.empty:
                print(f"⚠️ Skipping domain plot for {gender} | {outcome} | {domain} — no data.")
                continue


            # Create pivot and p-value matrices
            delta_pivot = subset.pivot_table(
                index='dimension2',
                columns='dimension1',
                values='delta_metric',
                aggfunc='mean'
            ).round(2)

            pval_matrix = pd.DataFrame(index=delta_pivot.index, columns=delta_pivot.columns)

            # Compute per-cell Wilcoxon test
            for d2 in delta_pivot.index:
                for d1 in delta_pivot.columns:
                    cell_subset = df[
                        (df['gender_pair'] == gender) &
                        (df['outcome'] == outcome) &
                        (df['dimension1'] == d1) &
                        (df['dimension2'] == d2)
                    ]
                    if len(cell_subset) >= 5:
                        try:
                            _, pval = wilcoxon(cell_subset['metric_X'], cell_subset['metric_Y'])
                            pval_matrix.loc[d2, d1] = pval
                        except:
                            pval_matrix.loc[d2, d1] = np.nan
                    else:
                        pval_matrix.loc[d2, d1] = np.nan

            # Plot
            plt.figure(figsize=(10, 8))
            # sns.heatmap(pivot, annot=True, cmap='vlag', center=0, fmt=".2f", linewidths=0.5)
            # Determine color range based on max abs value in pivot
            abs_max = np.abs(pivot.values).max()
            color_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

            annotations = pivot.copy().astype(str)
            for i in pivot.index:
                for j in pivot.columns:
                    p = pval_matrix.loc[i, j]
                    if pd.notna(p) and p < 0.05:
                        annotations.loc[i, j] = f"$\\bf{{{annotations.loc[i, j]}}}$"


            # Plot with custom style
            ax = sns.heatmap(pivot, annot=annotations, fmt="", cmap=delta_cmap, norm=color_norm,
                 linewidths=1, linecolor='black',
                 cbar_kws={"pad": 0.02}, annot_kws={"fontsize": 18})


            # Tick styling
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14, fontweight='medium')
            # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, fontweight='medium')
            # Clean and format tick labels
            xticklabels = [clean_label(label.get_text()) for label in ax.get_xticklabels()]
            yticklabels = [clean_label(label.get_text()) for label in ax.get_yticklabels()]
            ax.set_xticklabels(xticklabels, rotation=90, ha='right', fontsize=20, fontweight='semibold')
            ax.set_yticklabels(yticklabels, rotation=0, fontsize=22, fontweight='semibold')

            ax.tick_params(axis='x', width=1.5, pad=5)
            ax.tick_params(axis='y', width=1.5)
            ax.collections[0].colorbar.ax.tick_params(labelsize=18, width=2)
            ax.collections[0].colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))


            # Border styling
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)

            for i, row in enumerate(pivot.index):
                for j, col in enumerate(pivot.columns):
                    p = pval_matrix.loc[row, col]
                    if pd.notna(p) and p < 0.05:
                        ax.annotate('♥', xy=(j + 0.7, i + 0.2), xytext=(8, 8),
                                    textcoords='offset points', color='black',
                                    fontsize=18, ha='center', va='top', annotation_clip=False)


            plt.title(f"{gender} | {outcome} | {domain} | Δd = d_X - d_Y")
            plt.xlabel("Dimension1 (X)")
            plt.ylabel("Dimension2 (Y)")
            plt.tight_layout()

            out_path = f"{output_dir}/{axis}_{model}_{gender}_{outcome}_{domain}_bias_barplot.pdf"
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            plt.close()
