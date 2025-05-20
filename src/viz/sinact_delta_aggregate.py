import pandas as pd
import seaborn as sns
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_1samp
import os

axis = "nationality" # religion, nationality
model= "qwen_32b" # "qwen_32b" # "aya_expanse_8b" # gemma_3_27b_it, llama3_3_70b_it
# mode = "success"

# File paths
success_path = f'../../outputs/closed_ended/single_actor/{axis}/{model}/closed_ended_success_{model}_{axis}_all_1_runs.csv'
failure_path = f'../../outputs/closed_ended/single_actor/{axis}/{model}/closed_ended_failure_{model}_{axis}_all_1_runs.csv'


# Load and tag data
df_success = pd.read_csv(success_path)
df_success['outcome'] = 'Success'

df_failure = pd.read_csv(failure_path)
df_failure['outcome'] = 'Failure'

# 👉 Add this block to compute `metric` for success and failure
df_success['metric'] = (
    df_success['opt1_higheffort'] + df_success['opt2_highability']
    - df_success['opt3_easytask'] - df_success['opt4_goodluck']
)

df_failure['metric'] = (
    df_failure['opt1_loweffort'] + df_failure['opt2_lowability']
    - df_failure['opt3_difficulttask'] - df_failure['opt4_badluck']
)

# Combine both
# Pivot to get d_success and d_failure side by side
# Pivot to align success and failure for each (religion, gender, sample)
df_success['metric_type'] = 'success'
df_failure['metric_type'] = 'failure'

df_combined = pd.concat([df_success, df_failure], ignore_index=True)

# Add sample_id to ensure rows match between success and failure
df_combined['sample_id'] = df_combined.groupby(['nationality', 'gender', 'metric_type']).cumcount()

# Pivot so each row has both success and failure
pivot_detailed = df_combined.pivot_table(
    index=['nationality', 'gender', 'sample_id'],
    columns='metric_type',
    values='metric'
).dropna().reset_index()

pivot_detailed['delta'] = pivot_detailed['success'] - pivot_detailed['failure']

delta_significance = {}
for key, group in pivot_detailed.groupby(['nationality', 'gender']):
    if len(group) > 1:
        stat, p = ttest_1samp(group['delta'], 0)
        delta_significance[key] = (group['delta'].mean(), p)

# For barplot: average delta per (nationality, gender)
pivot_df = pivot_detailed.groupby(['nationality', 'gender'])['delta'].mean().reset_index()


grouped = df_combined.groupby(['outcome', 'nationality', 'gender'])

significance_dict = {}  # (outcome, nationality, gender) → (mean, p)
for key, group in grouped:
    if len(group) > 1:
        stat, p = ttest_1samp(group['metric'], 0)
        mean = group['metric'].mean()
        significance_dict[key] = (mean, p)

aggregated_df = df_combined.groupby(['outcome', 'nationality', 'gender'])['metric'].mean().reset_index()


# Output directory
output_dir = f'../../figs/closed_ended/single_actor/{axis}/{model}'
os.makedirs(output_dir, exist_ok=True)

# Set Seaborn style
sns.set(style='whitegrid')

# Boost font sizes across all elements
sns.set_context("paper", font_scale=2.4)

# # You override text weight and tick label size after
# mpl.rcParams['font.weight'] = 560
# mpl.rcParams['axes.titleweight'] = 560
# mpl.rcParams['axes.labelweight'] = 560
# # mpl.rcParams['xtick.labelsize'] = 14
# # mpl.rcParams['ytick.labelsize'] = 14


mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.weight'] = 'bold'  # Or 'semibold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'

# palette = {'male': '#00BFC4', 'female': '#DC143C'}  # Aqua Blue & Crimson Red

palette = {
    'male': '#00BFC4',     # Aqua Blue
    'female': '#E78CA2'    # Soft Crimson Rose
}

# Ensure consistent order of nationalities
nationality_order = df_combined['nationality'].dropna().unique().tolist()
label_map_race = {
        "white person": "White\nperson",
        "black person": "Black\nperson",
        "east asian": "East\nAsian",
        "south asian": "South\nAsian",
        "middle eastern": "Middle\nEastern",
        "hispanic": "Hispanic"
    }

label_map_religion = {
        "christian": "Christian",
        "muslim": "Muslim",
        "hindu": "Hindu",
        "buddhist": "Buddhist",
        "sikh": "Sikh",
        "jewish": "Jewish"
    }

label_map_nationality = {
        "American": "American",
        "African": "African",
        "Chinese": "Chinese",
        "Indian": "Indian",
        "Japanese": "Japanese",
        "Korean": "Korean",
        "French": "French",
        "German": "German",
        "Italian": "Italian",
        "Greek": "Greek",
        "British": "British",
        "Russia": "Russian",
        "Latin American": "Latin\nAmerican",
        "Middle-Eastern": "Middle\nEastern",
        "Australian": "Australian"
    }

# Clean and map x-axis labels for display
cleaned_names = [n.replace("an ", "").replace("a ", "") for n in nationality_order]
if axis == "race":
    short_labels = [label_map_race.get(n, n) for n in cleaned_names]
elif axis == "religion":
    short_labels = [label_map_religion.get(n, n) for n in cleaned_names]
elif axis == "nationality":
    short_labels = [label_map_nationality.get(n, n) for n in cleaned_names]
else:
    short_labels = nationality_order  # fallback

# Get all unique domains
domains = df_combined['domain'].dropna().unique()

fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
# fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=pivot_df,
    x='nationality',
    y='delta',
    hue='gender',
    order=nationality_order,
    palette=palette,
    capsize=0.1,
    edgecolor='black',
    linewidth=2.0,
    ax=ax,
    errorbar=None,
)

ax.set_xticks(range(len(nationality_order)))
ax.set_xticklabels(short_labels, rotation=90, ha='center')

# Set axis limits
bar_tops = [p.get_y() + p.get_height() for p in ax.patches if isinstance(p, plt.Rectangle)]
bar_bottoms = [p.get_y() for p in ax.patches if isinstance(p, plt.Rectangle)]
bar_extents = bar_tops + bar_bottoms
if bar_extents:
    max_abs = max(abs(v) for v in bar_extents)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_yticks(np.linspace(-max_abs, max_abs, 5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

for p in ax.patches:
    if not isinstance(p, plt.Rectangle):
        continue
    x = p.get_x() + p.get_width() / 2
    y = p.get_y() + p.get_height()
    nationality_idx = int(p.get_x() // 1)
    religion = nationality_order[nationality_idx]
    gender = 'male' if p.get_facecolor()[:3] == mpl.colors.to_rgb(palette['male']) else 'female'
    key = (religion, gender)
    if key in delta_significance and delta_significance[key][1] < 0.05:
        p.set_hatch('*')

ax.axhline(0, color='black', linewidth=1.5)
ax.set_title('', pad=15)
ax.set_ylabel('')
ax.set_xlabel('')
ax.legend(title='', loc='best')

for spine in ax.spines.values():
    spine.set_linewidth(2.0)
    spine.set_color('black')

ax.yaxis.grid(True, linestyle='--', linewidth=1.2, color='gray', alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.2)
out_path = f"{output_dir}/{axis}_{model}_delta_bias_barplot.pdf"
plt.savefig(out_path, format='pdf', bbox_inches='tight')
plt.close()
print(f"✅ Saved delta plot: {out_path}")

