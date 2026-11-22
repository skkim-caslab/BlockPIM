"""Re-run the SparseGPT pruning + C4 perplexity points of Fig. 1(a) and Fig. 7.

Every row of data/pruning/ppl_llama3.csv is one (design, compression ratio)
point. Baseline rows carry the exact llama.py command in the `cmd` column.
BlockPIM rows prune 32-byte blocks of INT8 weights at sparsity = compression + 10/256
(the bitmap overhead, rounded to 3 decimals), e.g. for c = 0.1:
    python llama.py meta-llama/Llama-3.1-8B c4 --sparsity 0.139 --blocksize 128
        --group_bytes 32 --wbits 8
The perplexity reported is the last one llama.py prints (C4).

Usage:
    python run_ppl.py --model 8B --policy blockpim            # print commands
    python run_ppl.py --model 8B --policy spaceA --run        # run them
    python run_ppl.py --model 8B --run --out my_ppl.csv       # all 8B rows
"""
import argparse
import os
import re
import shlex
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PPL_CSV = os.path.join(HERE, '../../data/pruning/ppl_llama3.csv')
HF_MODEL = {'8B': 'meta-llama/Llama-3.1-8B', '70B': 'meta-llama/Llama-3.1-70B'}
BITMAP_OVERHEAD = 10.0 / 256.0


def command_for(row):
    if isinstance(row['cmd'], str) and row['cmd'].strip():
        return row['cmd'].strip()
    if row['policy'] == 'blockpim' and pd.notna(row['compression']):
        if row['compression'] == 0.0:
            s = 0.0
        else:
            s = round(row["compression"] + BITMAP_OVERHEAD, 3)
        return (f"python llama.py {HF_MODEL[row['model']]} c4 --sparsity {s} "
                f"--blocksize 128 --group_bytes 32 --wbits 8")
    if row['policy'] == 'DensePIM':
        return f"python llama.py {HF_MODEL[row['model']]} c4 --sparsity 0.0 --blocksize 128 --wbits 8"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['8B', '70B'], default='8B')
    ap.add_argument('--policy', default=None,
                    help='DensePIM, blockpim, spaceA, sparsepim, psyncpim, csr-only, coo-only, csr+dense, coo+dense')
    ap.add_argument('--only-reported', action='store_true',
                    help='skip rows whose perplexity is empty in the CSV (not used by any figure)')
    ap.add_argument('--run', action='store_true', help='execute the commands (needs a GPU)')
    ap.add_argument('--out', default='ppl_rerun.csv', help='where --run writes its results')
    args = ap.parse_args()

    df = pd.read_csv(PPL_CSV)
    df = df[df['model'] == args.model]
    if args.policy:
        df = df[df['policy'] == args.policy]
    if args.only_reported:
        df = df[df['ppl'].notna()]

    results = []
    for _, row in df.iterrows():
        cmd = command_for(row)
        if cmd is None:
            continue
        if not args.run:
            print(cmd)
            continue
        print(f'>>> {cmd}', flush=True)
        argv = shlex.split(cmd)
        argv[0] = sys.executable
        out = subprocess.run(argv, cwd=HERE, capture_output=True, text=True).stdout
        found = re.findall(r'Perplexity:\s*([0-9.eE+\-]+)', out)
        ppl = float(found[-1]) if found else float('nan')
        print(f'    ppl = {ppl}  (reported: {row["ppl"]})', flush=True)
        results.append({**row.to_dict(), 'ppl_rerun': ppl})
        pd.DataFrame(results).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
