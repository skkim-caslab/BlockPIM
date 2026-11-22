"""Fig. 1(a): perplexity of pruned LLaMA-3 8B vs. memory compression ratio
for CSR/COO-only and hybrid CSR/COO+Dense designs.

Input : data/pruning/ppl_llama3.csv
Output: fig1a_ppl_compression.png (this directory)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PPL_CSV = os.path.join(HERE, '../../../data/pruning/ppl_llama3.csv')

# ====================== Fonts and Styles ======================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['lines.linewidth'] = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='8B', choices=['8B', '70B'], help='Model to plot (8B or 70B)')
    args = parser.parse_args()

    # ====================== Data Preparation ======================
    # x-axis is compression ratio
    target_compression = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    df_ppl = pd.read_csv(PPL_CSV)
    df_ppl = df_ppl[(df_ppl['DataType'] == 'INT8') & (df_ppl['model'] == args.model)].copy()

    def get_ppl(policy, comp):
        if policy == 'DensePIM':
            sub = df_ppl[df_ppl['policy'] == 'DensePIM']
            if sub.empty: return None
            return sub['ppl'].min()

        # A hybrid design may also run fully sparse; take the best option.
        policies_to_check = [policy]
        if policy == 'csr+dense':
            policies_to_check.append('csr-only')
        elif policy == 'coo+dense':
            policies_to_check.append('coo-only')

        sub = df_ppl[(df_ppl['policy'].isin(policies_to_check)) & (np.isclose(df_ppl['compression'], comp))]
        if sub.empty: return None
        return sub['ppl'].min()

    # ====================== Plotting ======================
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0))

    ppl_configs    = ['DensePIM', 'csr+dense', 'coo+dense', 'csr-only', 'coo-only',]
    ppl_labels     = ['DensePIM', 'CSR+Dense', 'COO+Dense', 'CSR-Only', 'COO-Only', ]

    ppl_colors     = {'DensePIM': 'gray', 'csr+dense': '#D97706', 'coo+dense': '#059669', 'csr-only': '#FBBF24', 'coo-only': '#34D399', 'blockpim': '#2563EB'}
    # Same markers as Fig. 1(b) (plot_area_time.py)
    ppl_markers    = {'DensePIM': 's',       'csr+dense': 'P',       'coo+dense': 'X',       'csr-only': 'o', 'coo-only': '^', 'blockpim': 'D'}
    ppl_linestyles = {'DensePIM': '-',      'csr+dense': '-',      'coo+dense': '-',       'csr-only': '-', 'coo-only': '-', 'blockpim': '-'}

    for cfg, label in zip(ppl_configs, ppl_labels):
        x_vals_b, y_vals = [], []
        if cfg == 'DensePIM':
            # DensePIM is a flat line across all compression ratios
            val = get_ppl(cfg, 0)
            if val is not None:
                x_vals_b = target_compression
                y_vals = [val] * len(target_compression)
        else:
            for comp in target_compression:
                val = get_ppl(cfg, comp)
                if val is not None:
                    x_vals_b.append(comp)
                    y_vals.append(val)

        if y_vals:
            ax.plot(x_vals_b, y_vals,
                    color=ppl_colors[cfg],
                    marker=ppl_markers[cfg],
                    linestyle=ppl_linestyles[cfg],
                    linewidth=2, markersize=8, markeredgecolor='k',
                    label=label)

    ax.set_ylabel('Perplexity')
    ax.set_yscale('log')

    if args.model == '8B':
        ax.set_ylim(bottom=8, top=50000)
    else:
        ax.set_ylim(bottom=6, top=2000)

    # Format x-axis as percentage, 90 to 0
    ax.set_xlim(0.95, 0.05)
    ax.set_xticks(target_compression)
    ax.set_xticklabels([f'{v*100:.0f}' for v in target_compression], fontsize=14)
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=12, ncol=3, frameon=False)

    ax.text(0.5, -0.15, 'Compression Ratio (%)', transform=ax.transAxes, ha='center', fontsize=14)

    plt.subplots_adjust(bottom=0.20, top=0.75)

    suffix = '' if args.model == '8B' else '_70b'
    out_name = os.path.join(HERE, f'fig1a_ppl_compression{suffix}.png')
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_name}")


if __name__ == '__main__':
    main()
