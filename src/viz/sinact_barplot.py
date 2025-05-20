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
df_combined = pd.concat([df_success, df_failure], ignore_index=True)

significance_dict = {}  # (domain, outcome, nationality, gender) → (mean, p)
grouped = df_combined.groupby(['domain', 'outcome', 'nationality', 'gender'])
for key, group in grouped:
    if len(group) > 1:  # t-test requires at least 2 samples
        stat, p = ttest_1samp(group['metric'], 0)
        mean = group['metric'].mean()
        significance_dict[key] = (mean, p)

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

cleaned_names = [n.replace("an ", "").replace("a ", "") for n in nationality_order]
short_labels = [label_map_nationality.get(n, n) for n in cleaned_names]

# Get all unique domains
domains = df_combined['domain'].dropna().unique()

for domain in domains:
    domain_df = df_combined[df_combined['domain'] == domain]
    
    # Order and label setup
    nationality_order = domain_df['nationality'].dropna().unique().tolist()
    # short_labels = [n.replace("an ", "").replace("a ", "") for n in nationality_order]
    cleaned_names = [n.replace("an ", "").replace("a ", "") for n in nationality_order]
    short_labels = [label_map_nationality.get(n, n) for n in cleaned_names]
    
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=False)
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

    for i, outcome in enumerate(['Success', 'Failure']):
        subset = domain_df[domain_df['outcome'] == outcome]

        # max_abs = max(abs(subset['metric'].min()), abs(subset['metric'].max()))
        # axes[i].set_ylim(-max_abs, max_abs)
        # Get bar heights after plot is drawn
        sns.barplot(
            data=subset,
            x='nationality',
            y='metric',
            hue='gender',
            order=nationality_order,
            # errorbar=('ci', 95),
            errorbar=None,
            err_kws={'linewidth': 2.0},
            palette=palette,
            capsize=0.1,
            edgecolor='black',
            linewidth=2.0,
            ax=axes[i],
        )

        if i == 1:
            # Grab the categorical axis positions
            xticks = axes[i].get_xticks()
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels(short_labels, rotation=90, ha='center')

        # if i == 0:
        #     # Grab the categorical axis positions
        #     xticks = axes[i].get_xticks()
        #     axes[i].set_xticks(xticks)
        #     axes[i].set_xticklabels(short_labels, rotation=90, ha='center')

        if i == 0:
            axes[i].tick_params(axis='x', labelbottom=False)  # Hide top x labels

        for p in axes[i].patches:
            if not isinstance(p, plt.Rectangle):
                continue
            # Get x, y position
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height()

            # Infer category info from bar color order (slow but safe)
            nationality_idx = int(p.get_x() // 1)  # each bar is 1 unit apart
            nationality = nationality_order[nationality_idx]
            gender = 'male' if p.get_facecolor()[:3] == mpl.colors.to_rgb(palette['male']) else 'female'
            domain_name = domain
            outcome_name = outcome

            key = (domain_name, outcome_name, nationality, gender)
            if key in significance_dict:
                _, p_val = significance_dict[key]
                if p_val < 0.05:
                    p.set_hatch('*')
                    # p.set_edgecolor('gray')   # hatch color
                    # p.set_linewidth(1.0)

                # if p_val < 0.05:
                #     stars = '*'
                # else:
                #     stars = ''

                # if stars:
                #     axes[i].text(x, y + 0.01, stars, ha='center', va='bottom', fontsize=20, weight='bold')


        bar_tops = [p.get_y() + p.get_height() for p in axes[i].patches if isinstance(p, plt.Rectangle)]
        bar_bottoms = [p.get_y() for p in axes[i].patches if isinstance(p, plt.Rectangle)]
        bar_extents = bar_tops + bar_bottoms
        if bar_extents:
            max_abs = max(abs(v) for v in bar_extents)
            axes[i].set_ylim(-max_abs, max_abs)
            axes[i].set_yticks(np.linspace(-max_abs, max_abs, 5))
            axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


        # axes[1].set_xticklabels(short_labels, rotation=50, ha='right')
        axes[i].axhline(0, color='gray', linestyle='--', linewidth=1)
        # axes[i].set_title(f'{outcome} – {domain.capitalize()}', loc='center', pad=15)
        clean_domain = domain.replace('_', ' ').capitalize()
        axes[i].set_title(f'{outcome} – {clean_domain}', loc='center', pad=15)
        axes[i].axhline(0, color='black', linewidth=1.5)
        axes[i].set_ylabel('')
        if i == 0:
            axes[i].legend(title='', loc='best')
        else:
            axes[i].get_legend().remove()

    # ticks = np.arange(len(nationality_order))
    # axes[1].set_xticks(ticks)
    # axes[1].set_xticklabels(short_labels, rotation=50, ha='right')

    # axes[1].set_xticks(axes[1].get_xticks())  # use actual x positions
    # axes[1].set_xticklabels(short_labels, rotation=50, ha='right')

    axes[1].set_xticks(range(len(nationality_order)))
    # axes[1].set_xticklabels(short_labels, rotation=50, ha='right')



    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    # axes[0].set_xticklabels([])
    # axes[0].set_xlabel('')
    # axes[1].tick_params(axis='x', labelrotation=50)

    for ax in axes:
        ax.yaxis.grid(True, linestyle='--', linewidth=1.2, color='gray', alpha=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.06, bottom=0.2)
    plt.subplots_adjust(left=0.06, bottom=0.2, hspace=0.25)

    # Save with domain in filename
    out_path = f"{output_dir}/{axis}_{model}_bias_barplot_{domain}.pdf"
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {out_path}")