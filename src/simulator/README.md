# BlockPIM Simulator (LLMCompass-based)

A cycle-level NPU + PIM simulator extended from
[LLMCompass](https://github.com/PrincetonUniversity/LLMCompass) (Zhang et al.,
ISCA 2024; BSD 3-Clause, see `LICENSE`). It simulates one decoder block of a
LLaMA-3 model during token generation: non-linear operators run on the NPU and,
in PIM mode, the matrix multiplications run on the PIM units.

| File / directory | Role |
|------------------|------|
| `simulate.py` | Simulates one decoder block and prints `Latency`, `Compute_Latency` and `IO_Latency` (seconds) |
| `run_sparsity_sweep.py` | Fig. 5/6 sweep for one (batch size, model): 7 designs x 4 sparsities x KV pruning off/on |
| `run_all.sh` | Runs the sweep for batch sizes 1/16/32/64 and both models |
| `hardware_model/` | NPU, PIM and SIGMA device definitions (INT8) |
| `software_model/` | Operator models; sparse-format (CSR/COO/bitmap) PIM matmul timing is in `sw_matmul.py` |
| `Tiles/test_tile/` | Per-design tile/PIM configuration (`pim_config`: `dense`, `csr`, `coo`, `psyncpim`, `stackpim`, `sigma`) |
| `systolic_array_model/` | Systolic-array look-up tables from LLMCompass |

`stackpim` is the internal name of BlockPIM; the simulator output files keep it
(`Unit = StackPIM`), and the plotting scripts relabel it as BlockPIM.

## Designs

| `Unit` in the CSV | Paper label | Device | Tile config |
|-------------------|-------------|--------|-------------|
| `NPU`       | NPU       | NPU (`--npu_only`) | `*_dense_int8.json` |
| `SIGMA_NPU` | SIGMA     | NPU (`--npu_only`) | `*_sigma_int8.json` |
| `Dense_PIM` | DensePIM  | PIM | `*_dense_int8.json` |
| `SpaceA`    | SpaceA    | PIM | `*_csr_int8.json` |
| `SparsePIM` | SparsePIM | PIM | `*_coo_int8.json` |
| `pSyncPIM`  | pSyncPIM  | PIM | `*_coo_comp_int8.json` |
| `StackPIM`  | BlockPIM  | PIM | `*_stackpim_int8.json` |

`KV_CACHE_PRUNING=1` in the environment models 80% KV-cache pruning: the
attention matmuls (`q_mul_k`, `a_mul_v`) read 20% of the KV cache, scaled by
each format's metadata overhead. `run_sparsity_sweep.py` writes both settings
(`_kv0` / `_kv1`); Fig. 5(c) and Fig. 6 use `_kv1`.

## Usage

```bash
# One design point: LLaMA-3 8B (d_model 4096, 32 heads), BlockPIM, 50% sparsity, batch 1
python simulate.py 1 ./Tiles/test_tile/double_512_N_large_stackpim_int8.json 4096 32 \
    --datatype int8 --sparsity 0.5 --bs 1
# Latency, 0.0007780301768480441

# All Fig. 5/6 inputs (writes data/simulator/summary_sparsity_final_bs*_*_kv*.csv)
bash run_all.sh
```

`simulate.py` writes its detailed per-operator log to `logs/` and caches
per-operator results in `configs/`; both are scratch output.
