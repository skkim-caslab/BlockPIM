#!/bin/bash
# Regenerates data/simulator/summary_sparsity_final_bs{BS}_{MODEL}_kv{0,1}.csv
# (Fig. 5 and Fig. 6 inputs) for every batch size and model.
# 8b  = LLaMA-3 8B  (d_model 4096, 32 heads, 32 layers)
# 80b = LLaMA-3 70B (d_model 8192, 64 heads, 80 layers)
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
for bs in 1 16 32 64; do
  for model in 8b 80b; do
    $PYTHON run_sparsity_sweep.py --bs $bs --model $model
  done
done
