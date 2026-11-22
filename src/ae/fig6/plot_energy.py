"""Fig. 6: normalized energy of LLaMA-3 8B across sparsity levels for batch
size 1 and 32 (KV-cache pruning on), split into static, dynamic compute and
dynamic I/O energy, normalized to DensePIM at 0 % sparsity.

Inputs: data/simulator/summary_sparsity_final_bs{1,32}_8b_kv1.csv
        data/verilog/int8/{dense,csr,coo,dense_stackpim_32}.out (power)
Output: fig6_energy.png (this directory)
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os
import re
import argparse

# ─── Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot Energy vs Sparsity for BS=1 and BS=32.")
parser.add_argument('--model', type=str, default='8b', choices=['8b', '80b'], 
                    help="Model to plot: '8b' for LLaMA3-8B or '80b' for LLaMA3-70B")
args = parser.parse_args()

TARGET_MODEL = args.model
model_label_str = 'LLaMA3-8B' if TARGET_MODEL == '8b' else 'LLaMA3-70B'

# ====================== 1. Power from synthesis reports ======================
HERE = os.path.dirname(os.path.abspath(__file__))
VERILOG_POWER_BASE = os.path.join(HERE, '../../../data/verilog')
POWER_CACHE = {}

def parse_power(filepath):
    if not os.path.exists(filepath): return 0.0, 0.0
    with open(filepath, 'r') as f:
        content = f.read()
        match = re.search(r"Total\s+([\d\.e\-\+]+)\s+([\d\.e\-\+]+)\s+([\d\.e\-\+]+)", content)
        if match: return float(match.group(1)) + float(match.group(2)), float(match.group(3))
    return 0.0, 0.0

def get_power(unit):
    dtype_dir = 'int8'
    if unit in ('NPU', 'SIGMA_NPU', 'SIGMA'):
        return 7.2, 0.9
    elif unit == 'StackPIM' or unit == 'BlockPIM':
        filename = f'{VERILOG_POWER_BASE}/{dtype_dir}/dense_stackpim_32.out'
    elif unit == 'SpaceA':
        filename = f'{VERILOG_POWER_BASE}/{dtype_dir}/csr.out'
    elif unit in ('SparsePIM', 'pSyncPIM'):
        filename = f'{VERILOG_POWER_BASE}/{dtype_dir}/coo.out'
    else:
        filename = f'{VERILOG_POWER_BASE}/{dtype_dir}/dense.out'

    if filename in POWER_CACHE:
        return POWER_CACHE[filename]
    d, s = parse_power(filename)
    POWER_CACHE[filename] = (d, s)
    return d, s

# ====================== 2. Load data and compute energy ======================
name_map = {
    'NPU': 'NPU',
    'SIGMA_NPU': 'SIGMA',
    'Dense_PIM': 'DensePIM',
    'pSyncPIM': 'pSyncPIM',
    'SpaceA': 'SpaceA',
    'SparsePIM': 'SparsePIM',
    'BlockPIM': 'BlockPIM',
    'StackPIM': 'BlockPIM'
}

BATCH_SIZES = [1, 32]
KV_MODE = 1
SPARSITIES = [0.0, 0.3, 0.5, 0.8]

data_dict = {}

for bs in BATCH_SIZES:
    path = os.path.join(HERE, f'../../../data/simulator/summary_sparsity_final_bs{bs}_{TARGET_MODEL}_kv{KV_MODE}.csv')
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[Warning] {path} not found.")
        continue
        
    df = df[df['DataType'] == 'INT8'].copy()
    df['Unit'] = df['Unit'].map(name_map)
    
    layers = 32 if TARGET_MODEL == '8b' else 80
    
    dyn_comp_energies = []
    dyn_io_energies = []
    sta_energies = []
    
    for idx, row in df.iterrows():
        d_p, s_p = get_power(row['Unit'])
        if row['Unit'] not in ['NPU', 'SIGMA']:
            s_p += 0.052538 # static power of the host NPU, added to every PIM design
            
        lat = row['Latency']
        comp_lat = row['Compute_Latency']
        io_lat = row['IO_Latency']
        
        # Calculate ratio
        total_sub_lat = comp_lat + io_lat
        if total_sub_lat > 0:
            comp_ratio = comp_lat / total_sub_lat
            io_ratio = io_lat / total_sub_lat
        else:
            comp_ratio, io_ratio = 0.5, 0.5
            
        total_dyn = d_p * lat * layers
        dyn_comp = total_dyn * comp_ratio
        dyn_io = total_dyn * io_ratio
        
        total_sta = s_p * lat * layers
        
        dyn_comp_energies.append(dyn_comp)
        dyn_io_energies.append(dyn_io)
        sta_energies.append(total_sta)
        
    df['Dynamic_Compute_Energy'] = dyn_comp_energies
    df['Dynamic_IO_Energy'] = dyn_io_energies
    df['Dynamic_Energy'] = df['Dynamic_Compute_Energy'] + df['Dynamic_IO_Energy']
    df['Static_Energy'] = sta_energies
    df['Total_Energy'] = df['Dynamic_Energy'] + df['Static_Energy']
    data_dict[bs] = df
import matplotlib.colors as mcolors

# ====================== 3. Style ======================
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman']
plt.rcParams['font.size'] = 26

color_mapping = {
    'NPU':      '#60A5FA',  
    'SIGMA':    '#F59E0B',
    'DensePIM': '#9CA3AF',
    'SparsePIM':'#059669',
    'pSyncPIM': '#10B981',
    'SpaceA':   '#D97706',
    'BlockPIM': '#374151',
}

# Fill opacity of the three energy components
ALPHA_STATIC   = 0.5   # hatched
ALPHA_COMPUTE  = 0.15
ALPHA_IO       = 1.0

unit_order = ['NPU', 'SIGMA', 'SpaceA', 'SparsePIM', 'pSyncPIM', 'DensePIM', 'BlockPIM']
sub_labels = ['(a) bs=1', '(b) bs=32']

fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
fig.subplots_adjust(bottom=0.25)
width = 0.11
x_pos = np.arange(len(SPARSITIES))

# Global y-axis range
global_max_val = 0
for bs in BATCH_SIZES:
    if bs not in data_dict: continue
    df_sub = data_dict[bs]
    dense_data = df_sub[(df_sub['Unit'] == 'DensePIM') & (np.isclose(df_sub['Sparsity'], 0.0, atol=1e-6))]
    if dense_data.empty: continue
    norm_base = dense_data['Total_Energy'].values[0]
    for u in unit_order:
        u_data = df_sub[df_sub['Unit'] == u]
        if not u_data.empty:
            global_max_val = max(global_max_val, (u_data['Total_Energy'] / norm_base).max())

global_max_ylim = int(np.ceil(global_max_val / 5.0) * 5.0)
if global_max_ylim == 0: global_max_ylim = 5

for idx, bs in enumerate(BATCH_SIZES):
    if bs not in data_dict: continue
    ax = axes[idx]
    df_sub = data_dict[bs]
    
    dense_data = df_sub[(df_sub['Unit'] == 'DensePIM') & (np.isclose(df_sub['Sparsity'], 0.0, atol=1e-6))]
    if dense_data.empty:
        norm_base = 1.0
    else:
        norm_base = dense_data['Total_Energy'].values[0]

    # Print the normalized values that are plotted
    print(f"\n{'='*70}")
    print(f"[{model_label_str}] Batch Size: {bs} - Normalized Energy Data")
    print(f"{'='*70}")
    
    df_print = df_sub.copy()
    df_print['Norm_Static']  = df_print['Static_Energy'] / norm_base
    df_print['Norm_Dyn_Comp'] = df_print['Dynamic_Compute_Energy'] / norm_base
    df_print['Norm_Dyn_IO']   = df_print['Dynamic_IO_Energy'] / norm_base
    df_print['Norm_Total']   = df_print['Total_Energy'] / norm_base
    
    df_print = df_print.sort_values(by=['Sparsity', 'Unit'])
    display_cols = ['Unit', 'Sparsity', 'Norm_Static', 'Norm_Dyn_Comp', 'Norm_Dyn_IO', 'Norm_Total']
    print(df_print[display_cols].to_string(index=False))

    for i, unit in enumerate(unit_order):
        unit_data = df_sub[df_sub['Unit'] == unit].set_index('Sparsity').reindex(SPARSITIES).fillna(0)
        
        static_norm  = unit_data['Static_Energy']  / norm_base
        dyn_comp_norm = unit_data['Dynamic_Compute_Energy'] / norm_base
        dyn_io_norm   = unit_data['Dynamic_IO_Energy'] / norm_base

        pos = x_pos + (i - len(unit_order) / 2) * width + width / 2
        base_color = color_mapping[unit]

        # Only the face color is transparent; edges and hatches stay solid black.
        
        # 1. Static (hatched)
        ax.bar(pos, static_norm, width,
               facecolor=mcolors.to_rgba(base_color, alpha=ALPHA_STATIC), 
               hatch='\\\\', edgecolor='black', linewidth=1.2, zorder=3)
        
        # 2. Dynamic compute
        ax.bar(pos, dyn_comp_norm, width, bottom=static_norm,
               facecolor=mcolors.to_rgba(base_color, alpha=ALPHA_COMPUTE), 
               edgecolor='black', linewidth=1.2, zorder=3)
        
        # 3. Dynamic I/O
        ax.bar(pos, dyn_io_norm, width, bottom=static_norm + dyn_comp_norm,
               facecolor=mcolors.to_rgba(base_color, alpha=ALPHA_IO), 
               edgecolor='black', linewidth=1.2, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_ylim(0, global_max_ylim)
    axes[0].set_ylim(0, 18)
    axes[0].set_yticks([0, 9, 18])
    axes[1].set_ylim(0, 10)
    axes[1].set_yticks([0, 5, 10])
    
    if idx == 0:
        ax.set_ylabel('Norm. Energy', fontsize=24)
        
    ax.set_xticklabels([f"{int(s*100)}%" for s in SPARSITIES], fontsize=24)
    
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.set_xlabel('Sparsity', labelpad=1, fontsize=22)
    ax.text(0.5, -0.48, sub_labels[idx], transform=ax.transAxes,
            ha='center', va='top', fontsize=24)

    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

# ====================== 4. Legend ======================
legend_elements  = [Patch(facecolor=color_mapping[u], edgecolor='black', label=u)
                    for u in unit_order]
legend_elements += [Patch(facecolor='white', edgecolor='white', label=' ')]

legend_elements += [
    Patch(facecolor=mcolors.to_rgba('gray', alpha=ALPHA_STATIC), edgecolor='black', hatch='\\\\', label='Static'),
    Patch(facecolor=mcolors.to_rgba('gray', alpha=ALPHA_COMPUTE), edgecolor='black', label='Dyn. Compute'),
    Patch(facecolor=mcolors.to_rgba('gray', alpha=ALPHA_IO), edgecolor='black', label='Dyn. I/O'),
]

fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 1.12), ncol=6, frameon=False, fontsize=20, columnspacing=1.0)

plt.tight_layout()
outfile = os.path.join(HERE, 'fig6_energy.png' if TARGET_MODEL == '8b' else f'fig6_energy_{TARGET_MODEL}.png')
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"Saved: {outfile}")
