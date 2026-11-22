"""Batch-size x sparsity sweep used for Fig. 5 (throughput) and Fig. 6 (energy).

For one (batch size, model) pair this runs simulate.py once per
(unit, sparsity, KV-pruning mode) and writes
    ../../data/simulator/summary_sparsity_final_bs{bs}_{model}_kv{0,1}.csv

Usage: python run_sparsity_sweep.py --bs 1 --model 8b
"""
import argparse
import csv
import subprocess
import os
import re
import sys

MODEL_CONFIGS = {
    '8b':  {'dim': 4096, 'heads': 32},
    '80b': {'dim': 8192, 'heads': 64},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--model', type=str, choices=['8b', '80b'], required=True)
    args = parser.parse_args()

    bs, model = args.bs, args.model
    dim, heads = MODEL_CONFIGS[model]['dim'], MODEL_CONFIGS[model]['heads']


    units_data = [
        ('NPU', '-', 'dense'),
        ('SIGMA_NPU', 'CSR', 'sigma'),
        ('Dense_PIM', '-', 'dense'),
        ('SpaceA', 'CSR', 'csr'),
        ('SparsePIM', 'COO', 'coo'),
        ('pSyncPIM', 'COO+comp', 'coo_comp'),
        ('StackPIM', 'Bitmap', 'stackpim')
    ]
    
    sparsities = [0.0, 0.3, 0.5, 0.8]
    kv_modes = [0, 1]

    for kv_mode in kv_modes:
        outfile = f'../../data/simulator/summary_sparsity_final_bs{bs}_{model}_kv{kv_mode}.csv'
        with open(outfile, 'w', newline='') as f_out:
            fieldnames = ['Unit', 'Type', 'DataType', 'Sparsity', 'Calculated_Sparsity', 'Overhead', 'Memory_Ratio', 'Memory_MB', 'Throughput', 'Latency', 'Compute_Latency', 'IO_Latency']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for s in sparsities:
                for unit, fmt_type, config_key in units_data:
                    dtype = 'int8'
                    elem_size = 1
                    
                    # Calculate Overhead
                    BITMAP_OVERHEAD_CONST = 10.0 / 256.0
                    if unit == 'StackPIM':
                        overhead = 1.0 + BITMAP_OVERHEAD_CONST
                    elif unit == 'SIGMA_NPU':
                        import math
                        data_bits = 8 # int8
                        cidx_bits = math.ceil(math.log2(dim))
                        overhead = (data_bits + cidx_bits) / float(data_bits)
                    elif fmt_type == 'CSR':
                        overhead = 5.0
                    elif fmt_type == 'COO':
                        overhead = 9.0
                    elif fmt_type == 'COO+comp':
                        overhead = 6.0
                    else:
                        overhead = 1.0
                    
                    # Sparsity rules
                    sim_s = s
                    if unit in ['NPU', 'Dense_PIM']:
                        sim_s = 0.0 # Dense units always dense
                    
                    # Calc Memory Ratio
                    if unit == 'StackPIM':
                        ratio = 1.0 - sim_s + BITMAP_OVERHEAD_CONST
                    else:
                        ratio = (1.0 - sim_s) * overhead
                    
                    dense_mb = (dim * dim * elem_size) / (1024 * 1024)
                    target_mem = dense_mb * ratio
                    
                    if bs == 1:
                        config_file = f"./Tiles/test_tile/double_512_N_large_{config_key}_{dtype.lower()}.json"
                    else:
                        config_file = f"./Tiles/test_tile/double_512_bs{bs}_{config_key}_{dtype.lower()}.json"
                    
                    cmd = f"{sys.executable} -u simulate.py 1 {config_file} {dim} {heads} --datatype {dtype} --sparsity {sim_s} --bs {bs}"
                    
                    if unit in ['NPU', 'SIGMA_NPU']:
                        cmd += " --npu_only"
                    
                    # Set environment variable for KV cache pruning
                    env = os.environ.copy()
                    env["KV_CACHE_PRUNING"] = str(kv_mode)
                    
                    result = subprocess.run(cmd.split(), env=env, capture_output=True, text=True)
                    match = re.search(r"Latency,\s*([0-9\.eE\-\+]+)", result.stdout)
                    match_comp = re.search(r"Compute_Latency,\s*([0-9\.eE\-\+]+)", result.stdout)
                    match_io = re.search(r"IO_Latency,\s*([0-9\.eE\-\+]+)", result.stdout)
                    
                    if match:
                        lat = float(match.group(1))
                    else:
                        print(f"  [WARN] Latency not found in stdout.")
                        if result.stderr.strip(): print(f"  [STDERR] {result.stderr.strip()[:300]}")
                        lat = 0.0
                        
                    comp_lat = float(match_comp.group(1)) if match_comp else 0.0
                    io_lat = float(match_io.group(1)) if match_io else 0.0
                        
                    throughput = bs / lat if lat > 0 else 0.0
                    
                    print(f"  Processed KV={kv_mode} {unit:12s}  s={s:.2f}  ratio={ratio:.4f}  lat={lat:.6f}")
                    
                    writer.writerow({
                        'Unit': unit,
                        'Type': fmt_type,
                        'DataType': 'INT8',
                        'Sparsity': s,
                        'Calculated_Sparsity': sim_s,
                        'Overhead': overhead,
                        'Memory_Ratio': ratio,
                        'Memory_MB': target_mem,
                        'Throughput': throughput,
                        'Latency': lat,
                        'Compute_Latency': comp_lat,
                        'IO_Latency': io_lat
                    })
        print(f"[DONE] Saved: {outfile}")

if __name__ == '__main__':
    main()
