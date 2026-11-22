"""Fig. 7: perplexity of LLaMA-3 8B (INT8) across memory compression ratios
(weights + sparse-format metadata) for SIGMA, SpaceA, SparsePIM, pSyncPIM and
BlockPIM. The dashed line is the dense (DensePIM) perplexity.

Input : data/pruning/ppl_llama3.csv
Output: fig7_ppl_memory.png (this directory)
"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ====================== 1. Load data (INT8 only) ======================
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, '../../../data/pruning/ppl_llama3.csv'))
df = df[(df['DataType'] == 'INT8') & (df['model'] == '8B')].copy()

# ====================== 2. Name mapping ======================
name_map = {
    'csr-only': 'SIGMA',
    'spaceA': 'SpaceA',
    'sparsepim': 'SparsePIM',
    'psyncpim': 'pSyncPIM',
    'blockpim': 'BlockPIM',
    'DensePIM': 'DensePIM'
}

df['Unit_Label'] = df['policy'].map(name_map)
df = df.dropna(subset=['Unit_Label'])

# Baseline PPL (DensePIM)
baseline_row = df[df['Unit_Label'] == 'DensePIM']
if not baseline_row.empty:
    baseline_ppl = baseline_row['ppl'].values[0]
else:
    baseline_ppl = 9.549263

# ====================== 3. Colors, order and hatches ======================
unit_order = ['SIGMA', 'SpaceA', 'SparsePIM', 'pSyncPIM', 'BlockPIM']

color_mapping = {
    'SIGMA':    '#F59E0B',
    'DensePIM': '#9CA3AF',
    'SparsePIM':'#059669',
    'pSyncPIM': '#10B981',
    'SpaceA':   '#D97706',
    'BlockPIM': '#374151',
}

# Hatch only SIGMA (NPU-based)
hatch_mapping = {
    'SIGMA':    '///',
    'SpaceA':   '',
    'SparsePIM':'',
    'pSyncPIM': '',
    'BlockPIM': '',
    'DensePIM': ''
}

# ====================== 4. x-axis (compression ratio) ======================
memory_ratios   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ratio_labels    = [f"{int(r*100)}%" for r in memory_ratios]

# ====================== 5. Style ======================
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman']
plt.rcParams['font.size'] = 24

# ====================== 6. Plot (broken y-axis) ======================
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(15, 4), gridspec_kw={'height_ratios': [1, 2]})
fig.subplots_adjust(hspace=0.1)

n_units   = len(unit_order)
n_ratios  = len(memory_ratios)
width     = 0.15
group_gap = 0.2
x_pos = np.arange(n_ratios) * (n_units * width + group_gap)

for r_idx, ratio in enumerate(memory_ratios):
    for u_idx, unit in enumerate(unit_order):
        row = df[(df['compression'].notna()) & (np.isclose(df['compression'], ratio, atol=1e-5)) & (df['Unit_Label'] == unit)]
        
        if row.empty or pd.isna(row['ppl'].values[0]):
            continue
            
        ppl_val = row['ppl'].values[0]
        x = x_pos[r_idx] + (u_idx - n_units / 2 + 0.5) * width

        # Draw the bar on both axes
        for ax in (ax_top, ax_bottom):
            ax.bar(
                x, ppl_val, width,
                color     = color_mapping[unit],
                edgecolor = 'black',
                linewidth = 0.8,
                hatch     = hatch_mapping[unit],
                zorder    = 3,
            )

# Dense baseline PPL on both axes
for ax in (ax_top, ax_bottom):
    ax.axhline(y=baseline_ppl, color='black', linestyle='--',
               linewidth=1.5, zorder=4, label=f'Baseline PPL ({baseline_ppl:.2f})')

# ====================== 7. Axes (y-limits and cut marks) ======================
ax_top.set_ylim(10000, 60000)
ax_bottom.set_ylim(0, 6000)

# Hide the spines between the two axes
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(labeltop=False, bottom=False)
ax_bottom.xaxis.tick_bottom()

# Cut marks
d_x = 0.015
d_y_top = 0.04
d_y_bot = 0.02

kwargs_top = dict(transform=ax_top.transAxes, color='black', clip_on=False, linewidth=1.5)
ax_top.plot((-d_x, +d_x), (-d_y_top, +d_y_top), **kwargs_top)
ax_top.plot((1 - d_x, 1 + d_x), (-d_y_top, +d_y_top), **kwargs_top)

kwargs_bot = dict(transform=ax_bottom.transAxes, color='black', clip_on=False, linewidth=1.5)
ax_bottom.plot((-d_x, +d_x), (1 - d_y_bot, 1 + d_y_bot), **kwargs_bot)
ax_bottom.plot((1 - d_x, 1 + d_x), (1 - d_y_bot, 1 + d_y_bot), **kwargs_bot)

# x-axis label and ticks
ax_top.set_yticks([40000])
ax_bottom.set_yticks([0,4000])
ax_bottom.set_xticks(x_pos)
ax_bottom.set_xticklabels(ratio_labels, fontsize=24)
ax_bottom.set_xlabel('Compression Ratio', fontsize=24, labelpad=8)

# Shared y-axis label
fig.supylabel('Perplexity', fontsize=24, x=0.05, y=0.6)

ax_top.tick_params(axis='y', labelsize=22)
ax_bottom.tick_params(axis='y', labelsize=22)
ax_bottom.tick_params(axis='x', labelsize=24)

for ax in (ax_top, ax_bottom):
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

# ====================== 8. Legend ======================
legend_elements = [
    mpatches.Patch(facecolor=color_mapping[u], edgecolor='black', hatch=hatch_mapping[u], label=u)
    for u in unit_order
]

fig.legend(
    handles        = legend_elements,
    loc            = 'upper center',
    bbox_to_anchor = (0.52, 1.05),
    ncol           = 5,
    frameon        = False,
    fontsize       = 22,
)

plt.tight_layout()
output_filename = os.path.join(HERE, 'fig7_ppl_memory.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Saved: {output_filename}")

# ====================== 9. BlockPIM vs. baselines (printed table) ======================
print("\n" + "="*70)
print("[BlockPIM PPL Comparison vs Baselines (INT8)]")
print("="*70)

table_data = []
for ratio in memory_ratios:
    bp_row = df[(df['compression'].notna()) & (np.isclose(df['compression'], ratio, atol=1e-5)) & (df['Unit_Label'] == 'BlockPIM')]
    if bp_row.empty or pd.isna(bp_row['ppl'].values[0]): continue
    bp_val = bp_row['ppl'].values[0]

    for unit in unit_order:
        if unit == 'BlockPIM': continue
        other_row = df[(df['compression'].notna()) & (np.isclose(df['compression'], ratio, atol=1e-5)) & (df['Unit_Label'] == unit)]
        if other_row.empty or pd.isna(other_row['ppl'].values[0]): continue
        other_val = other_row['ppl'].values[0]

        table_data.append({
            'Ratio': ratio,
            'Baseline_Unit': unit,
            'Base_PPL': round(other_val, 2),
            'BlockPIM_PPL': round(bp_val, 2),
            'PPL_Diff': round(other_val - bp_val, 2),
            'Ratio(Base/BP)': f"{other_val / bp_val:.2f}x" if bp_val > 0 else "N/A"
        })

if table_data:
    comp_df = pd.DataFrame(table_data)
    print(comp_df.to_string(index=False))
print("="*70 + "\n")
