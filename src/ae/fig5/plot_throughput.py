"""Fig. 5: normalized throughput across batch sizes (1, 16, 64) and sparsity
(0/30/50/80 %): (a) LLaMA-3 8B, (b) LLaMA-3 70B, both without KV-cache pruning,
(c) LLaMA-3 8B with KV-cache pruning. Normalized to DensePIM at batch size 1.

Input : data/simulator/summary_sparsity_final_bs{1,16,64}_{8b,80b}_kv{0,1}.csv
Output: fig5_throughput.png (this directory)
"""
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Patch

# ─── Paths and name mapping ────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, '../../../data/simulator')
BS_LIST  = [1, 16, 64]

name_map = {
    'NPU':       'NPU',
    'SIGMA_NPU': 'SIGMA',
    'Dense_PIM': 'DensePIM',
    'pSyncPIM':  'pSyncPIM',
    'SpaceA':    'SpaceA',
    'SparsePIM': 'SparsePIM',
    'BlockPIM':  'BlockPIM',
    'StackPIM':  'BlockPIM',
}

# ─── Subplots: (model, kv_mode, title); 80b = LLaMA-3 70B (80 layers) ────────
# (model_name, kv_mode, title)
SUBPLOTS_CONFIG = [
    ('8b',  0, 'LLaMA3-8B'),
    ('80b', 0, 'LLaMA3-70B'), 
    ('8b',  1, 'LLaMA3-8B (KV pruning)'),
]

def load_data(model, kv_mode):
    data = {}
    for bs in BS_LIST:
        path = os.path.join(BASE_DIR, f'summary_sparsity_final_bs{bs}_{model}_kv{kv_mode}.csv')
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[Warning] {path} not found.")
            data[bs] = pd.DataFrame()
            continue
            
        df = df[df['DataType'] == 'INT8'].copy()
        df['Unit'] = df['Unit'].map(name_map)
        
        layers = 32 if model == '8b' else 80
        df['Throughput'] = bs / (df['Latency'] * layers)
        data[bs] = df
    return data

# ─── Style ─────────────────────────────────────────────────────────────────
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman']
plt.rcParams['font.size'] = 14

color_mapping = {
    'NPU':      '#60A5FA',
    'SIGMA':    '#F59E0B',
    'DensePIM': '#9CA3AF',
    'SpaceA':   '#D97706',
    'SparsePIM':'#059669',
    'pSyncPIM': '#10B981',
    'BlockPIM': '#374151',
}

# Hatch only the NPU-based designs (NPU, SIGMA)
hatch_mapping = {
    'NPU':      '///',
    'SIGMA':    '///',
    'DensePIM': '',
    'SpaceA':   '',
    'SparsePIM':'',
    'pSyncPIM': '',
    'BlockPIM': ''
}

groups = ['Sparsity 0%', 'Sparsity 30%', 'Sparsity 50%', 'Sparsity 80%']
# DensePIM is placed right before BlockPIM
group_units = {
    'Sparsity 0%': [('NPU', 0.0), ('SIGMA', 0.0), ('SpaceA', 0.0), ('SparsePIM', 0.0), ('pSyncPIM', 0.0), ('DensePIM', 0.0), ('BlockPIM', 0.0)],
    'Sparsity 30%': [('SIGMA', 0.3), ('SpaceA', 0.3), ('SparsePIM', 0.3), ('pSyncPIM', 0.3), ('BlockPIM', 0.3)],
    'Sparsity 50%': [('SIGMA', 0.5), ('SpaceA', 0.5), ('SparsePIM', 0.5), ('pSyncPIM', 0.5), ('BlockPIM', 0.5)],
    'Sparsity 80%': [('SIGMA', 0.8), ('SpaceA', 0.8), ('SparsePIM', 0.8), ('pSyncPIM', 0.8), ('BlockPIM', 0.8)]
}

# ─── Plot ──────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(30, 3)) 
fig.subplots_adjust(bottom=0.28, wspace=0.05) 

width = 1.4
pad_sp = 0.6
pad_bs = 1.5

def plot_subplot(ax, data_dict, title, sub_idx):
    start_x = 3.0
    current_x = start_x
    
    sp_cells, bs_cells = [], []
    
    for b_idx, bs in enumerate(BS_LIST):
        start_bs = current_x
        df = data_dict.get(bs, pd.DataFrame())
        
        for g_idx, group in enumerate(groups):
            start_sp = current_x
            units = group_units[group]
            num_bars = len(units)
            
            for i, (unit, sparsity_val) in enumerate(units):
                if not df.empty:
                    row = df[(df['Unit'] == unit) & (np.isclose(df['Sparsity'], sparsity_val, atol=1e-6))]
                    if row.empty:
                        row = df[df['Unit'] == unit]
                    val = row['Throughput'].values[0] if not row.empty else 0
                else:
                    val = 0
                
                pos = current_x + i * width + width / 2
                
                ax.bar(pos, val, width, color=color_mapping.get(unit, 'grey'), 
                       edgecolor='black', linewidth=0.1, hatch=hatch_mapping.get(unit, ''), zorder=3)
            
            current_x += num_bars * width
            end_sp = current_x
            sp_cells.append(((start_sp + end_sp) / 2, group.replace('Sparsity ', '')))
            
            if g_idx < len(groups) - 1:
                current_x += pad_sp
                
        end_bs = current_x
        bs_cells.append(((start_bs + end_bs) / 2, f"bs={bs}"))
        
        if b_idx < len(BS_LIST) - 1:
            sep_x = current_x + pad_bs / 2
            ax.axvline(x=sep_x, color='gray', linestyle='--', alpha=0.5, zorder=0)
            current_x += pad_bs
    
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    y_level_sp = -0.06
    y_level_bs = -0.16
    
    header_x = start_x - 1.0
    
    # 'Sparsity' row label only on subplot (a)
    if sub_idx == 0:
        ax.text(header_x, y_level_sp-0.04, 'Sparsity', ha='right', va='center', transform=ax.get_xaxis_transform(), fontsize=14)
    
    for cx, label in sp_cells:
        ax.text(cx, y_level_sp-0.04, label, ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=13)
    for cx, label in bs_cells:
        ax.text(cx, y_level_bs-0.06, label, ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=14)
    
    # y-axis label only on subplot (a)
    if sub_idx == 0:
        ax.set_ylabel('Norm. Throughput', fontsize=15, labelpad=5)
        
    ax.set_xlim(header_x - 0.5, current_x + 0.5)
    
    # Fixed y-axis range and ticks per subplot
    if sub_idx == 0:
        ax.set_ylim(0, 20)
        ax.set_yticks([0, 10, 20])
    elif sub_idx == 1:
        ax.set_ylim(0, 40)
        ax.set_yticks([0, 20, 40])
    elif sub_idx == 2:
        ax.set_ylim(0, 50)
        ax.set_yticks([0, 25, 50])
        
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # (a)/(b)/(c) caption below each subplot
    caption = f"({chr(97 + sub_idx)})"
    ax.text(0.5, -0.38, caption, transform=ax.transAxes, ha='center', va='center', fontsize=20)

# Render each subplot
for idx, (model, kv_mode, title) in enumerate(SUBPLOTS_CONFIG):
    data_dict = load_data(model, kv_mode)
    
    # Normalization (Speedup over DensePIM BS=1)
    norm_factor = 1.0
    if 1 in data_dict and not data_dict[1].empty:
        df_bs1 = data_dict[1]
        dense_pim_tp = df_bs1[(df_bs1['Unit'] == 'DensePIM') & (np.isclose(df_bs1['Sparsity'], 0.0, atol=1e-6))]['Throughput'].values[0]
        norm_factor = dense_pim_tp
    
    for bs in BS_LIST:
        if bs in data_dict and not data_dict[bs].empty:
            data_dict[bs]['Throughput'] /= norm_factor
    # Print the normalized values that are plotted
    print(f"\n{'='*50}")
    print(f"[{title}] Normalized Data to Plot")
    print(f"{'='*50}")
    for bs in BS_LIST:
        if bs in data_dict and not data_dict[bs].empty:
            print(f"\n>>> Batch Size: {bs}")
            display_cols = ['Unit', 'Sparsity', 'Latency', 'Throughput']
            print(data_dict[bs][display_cols].to_string(index=False))
    # ===================================
            
    plot_subplot(axs[idx], data_dict, title, idx)

# ─── Shared legend ─────────────────────────────────────────────────────────
all_units = ['NPU', 'SIGMA', 'SpaceA', 'SparsePIM', 'pSyncPIM', 'DensePIM', 'BlockPIM']

legend_elements = [Patch(facecolor=color_mapping[u], edgecolor='black', hatch=hatch_mapping.get(u, ''), label=u) for u in all_units]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), 
           ncol=7, frameon=False, fontsize=16, columnspacing=1.5)

outfile = os.path.join(HERE, 'fig5_throughput.png')
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"Saved plot: {outfile}")
