"""Table I: metadata overhead of CSR and COO for LLaMA-3 up_proj layers.

Per stored non-zero weight:
  CSR : value bits + column index bits           (row pointers amortized)
  COO : value bits + row index bits + column index bits
with index widths ceil(log2(dimension)). The metadata ratio is
index bits / total bits.

Output: printed table and table1_metadata_overhead.csv (this directory)
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (model, layer, rows, cols): up_proj weight is [intermediate x hidden]
LAYERS = [
    ('LLaMA-3 8B', 'up_proj', 14336, 4096),
    ('LLaMA-3 70B', 'up_proj', 28672, 8192),
]
PRECISIONS = [('INT8', 8), ('INT4', 4)]


def main():
    rows = []
    for model, layer, n_rows, n_cols in LAYERS:
        ridx = math.ceil(math.log2(n_rows))
        cidx = math.ceil(math.log2(n_cols))
        for prec, vbits in PRECISIONS:
            csr = vbits + cidx
            coo = vbits + ridx + cidx
            rows.append({
                'model': model, 'layer': layer,
                'dims': f'{n_rows}x{n_cols}', 'ridx_bits': ridx, 'cidx_bits': cidx,
                'precision': prec,
                'csr_total_bits': csr, 'csr_metadata_pct': round(100 * cidx / csr, 1),
                'coo_total_bits': coo, 'coo_metadata_pct': round(100 * (ridx + cidx) / coo, 1),
            })

    print(f"{'Model (Layer)':<24} {'Dimensions':<26} {'Prec':<5} {'CSR / COO total bits (metadata ratio)'}")
    print('-' * 90)
    for r in rows:
        dims = f"{r['dims']} (Ridx: {r['ridx_bits']}b, Cidx: {r['cidx_bits']}b)"
        print(f"{r['model'] + ' (' + r['layer'] + ')':<24} {dims:<26} {r['precision']:<5} "
              f"{r['csr_total_bits']} ({r['csr_metadata_pct']}%) / {r['coo_total_bits']} ({r['coo_metadata_pct']}%)")

    out = os.path.join(HERE, 'table1_metadata_overhead.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
