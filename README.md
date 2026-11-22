# BlockPIM: Enabling Sparse LLM Inference on Dense PIM Architectures

[![Paper](https://img.shields.io/badge/IEEE%20CAL-2026-blue)](https://ieeexplore.ieee.org/document/11616633/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FLCA.2026.3715204-blue)](https://doi.org/10.1109/LCA.2026.3715204)

This repository contains the simulator, pruning code, RTL and figure scripts of **BlockPIM** (IEEE CAL 2026).

> **BlockPIM: Enabling Sparse LLM Inference on Dense PIM Architectures**<br>
> Seulki Kim<sup>1</sup>, Yewon Jang<sup>2</sup>, Yunhyeong Jeon<sup>1</sup>, Kyeonghyeon Ryu<sup>1</sup>, Daehoon Kim<sup>2</sup><br>
> <sup>1</sup>DGIST, <sup>2</sup>Yonsei University<br>
> *IEEE Computer Architecture Letters*, vol. 25, no. 2, pp. 311–314, July 2026<br>
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/11616633/) [[DOI]](https://doi.org/10.1109/LCA.2026.3715204) [[Project Page]](https://caslab-yonsei.github.io/publications/cal26-2-skim/)

## Abstract

Large Language Models (LLMs) have intensified the memory wall during inference,
where performance is increasingly limited by memory bandwidth rather than
computation. While Processing-In-Memory (PIM) and model compression techniques
mitigate this issue, existing sparse PIM designs rely on CSR/COO formats that
assume ultra-high sparsity (e.g., > 90%), which is difficult to achieve in
practical LLMs without significant accuracy loss. Moreover, fine-grained indexing
introduces substantial metadata overhead, diminishing the benefits of pruning,
especially under low-bit quantization. To address this mismatch, we propose
BlockPIM, a PIM architecture that employs hardware-aligned coarse-grained pruning
to efficiently execute sparse LLMs on standard dense computation units. BlockPIM
replaces per-element indices with compact bitmaps and contiguous non-zero blocks,
enabling sparse workloads to be processed directly on dense execution pipelines
without complex index decoding or irregular control flow. Experimental results
show that BlockPIM achieves up to 8.0× and 3.6× speedups over the NPU and
state-of-the-art sparse PIM baselines, respectively, while reducing energy
consumption by up to 95.2% and providing superior memory efficiency under
practical accuracy constraints.

## Repository Structure

```text
BlockPIM/
├── src/
│   ├── simulator/     # LLMCompass-based NPU + PIM simulator
│   ├── pruning/       # SparseGPT-based 32-byte block pruning + perplexity
│   ├── verilog/       # RTL of the PIM processing units + Yosys/OpenSTA flow
│   └── ae/            # Figure and table scripts (fig1, table1, fig5, fig6, fig7)
├── data/
│   ├── simulator/     # Simulator results (Fig. 5, Fig. 6)
│   ├── pruning/       # C4 perplexity of the pruned LLaMA-3 models (Fig. 1(a), Fig. 7)
│   └── verilog/int8/  # Yosys/OpenSTA area, timing and power reports
├── requirements.txt
└── CITATION.cff
```

Details: [`src/simulator/README.md`](src/simulator/README.md), [`src/pruning/README.md`](src/pruning/README.md), [`src/verilog/README.md`](src/verilog/README.md).
In the simulator files, `stackpim` / `StackPIM` is the internal name of BlockPIM, and `80b` denotes LLaMA-3 70B (80 decoder layers).

## Getting Started

### Requirements

- Python 3.9+ and the packages in [`requirements.txt`](requirements.txt) (includes `scalesim==2.0.2`)
- Pruning only: a CUDA GPU and access to `meta-llama/Llama-3.1-8B` on Hugging Face
- Synthesis only: Yosys and OpenSTA on `PATH` (reports were produced with Yosys 0.53+101 and OpenSTA 2.7.0); the ASAP7 library is included in `src/verilog/libs/`
- Optional: the Times New Roman font (otherwise Matplotlib falls back to its default serif font)

### Setup

```bash
pip install -r requirements.txt
export BLOCKPIM_HOME=$(pwd)
```

## Reproducing the Paper Figures and Tables

The figures and tables are redrawn from the result data in `data/`; each script writes its output
next to itself. Figs. 2–4 are design diagrams and have no script.

| Figure / Table | Script | Data |
|---|---|---|
| Fig. 1(a) | `src/ae/fig1/plot_ppl_compression.py` | `data/pruning/` |
| Fig. 1(b) | `src/ae/fig1/plot_area_time.py` | `data/verilog/int8/` |
| Table I | `src/ae/table1/metadata_overhead.py` | computed analytically |
| Fig. 5 | `src/ae/fig5/plot_throughput.py` | `data/simulator/` |
| Fig. 6 | `src/ae/fig6/plot_energy.py` | `data/simulator/`, `data/verilog/int8/` |
| Fig. 7 | `src/ae/fig7/plot_ppl_memory.py` | `data/pruning/` |

```bash
cd $BLOCKPIM_HOME/src/ae
for s in fig1/plot_ppl_compression.py fig1/plot_area_time.py table1/metadata_overhead.py \
         fig5/plot_throughput.py fig6/plot_energy.py fig7/plot_ppl_memory.py; do python3 $s; done
```

## Regenerating the Data (optional)

### Simulation

`run_all.sh` simulates the seven designs (NPU, SIGMA, DensePIM, SpaceA, SparsePIM, pSyncPIM,
BlockPIM) at sparsity 0/30/50/80%, with and without KV-cache pruning, for batch sizes 1/16/32/64
and both models, and overwrites `data/simulator/summary_sparsity_final_bs*_*_kv*.csv`.

```bash
cd $BLOCKPIM_HOME/src/simulator
bash run_all.sh
python3 run_sparsity_sweep.py --bs 1 --model 8b   # or a single (batch size, model) pair
```

Runtime: about 20 minutes on a laptop CPU (448 simulations).

### Pruning

Prunes LLaMA-3.1 with SparseGPT (32-byte INT8 blocks, `--group_bytes 32`, at sparsity = compression
ratio + 10/256 for the bitmap) and evaluates perplexity on WikiText-2, PTB and C4; the C4 value is used. `data/pruning/ppl_llama3.csv` lists every point of Fig. 1(a) and Fig. 7 with
the command that produced it; the pruned checkpoints are not included. Re-run values depend on the
calibration data and seed. See [`src/pruning/README.md`](src/pruning/README.md).

```bash
cd $BLOCKPIM_HOME/src/pruning
python3 run_ppl.py --model 8B --only-reported            # print the commands
python3 run_ppl.py --model 8B --policy blockpim --run
```

`--run` executes the commands and writes `ppl_rerun.csv`.

Runtime: roughly half an hour per 8B point (estimate) on one 24 GB GPU; the 70B points need more GPU memory.

### Hardware Synthesis

Synthesizes the DensePIM, CSR, COO and BlockPIM processing units with Yosys (ASAP7) and analyzes
them with OpenSTA at 940 MHz; overwrites `data/verilog/int8/{dense,csr,coo,dense_stackpim_32}.out`.

```bash
cd $BLOCKPIM_HOME/src/verilog && bash run_all.sh
```

Runtime: a few minutes.

## Citation

```bibtex
@article{kim2026blockpim,
  author  = {Kim, Seulki and Jang, Yewon and Jeon, Yunhyeong and Ryu, Kyeonghyeon and Kim, Daehoon},
  title   = {{BlockPIM}: Enabling Sparse {LLM} Inference on Dense {PIM} Architectures},
  journal = {IEEE Computer Architecture Letters},
  volume  = {25},
  number  = {2},
  pages   = {311--314},
  year    = {2026},
  doi     = {10.1109/LCA.2026.3715204}
}
```

## Acknowledgements

The simulator is built on [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass)
(BSD 3-Clause, `src/simulator/LICENSE`) and [ScaleSim](https://github.com/scalesim-project/scale-sim-v2).
The pruning code is based on [SparseGPT](https://github.com/IST-DASLab/sparsegpt)
(Apache 2.0, `src/pruning/LICENSE`). Hardware evaluation uses
[Yosys](https://github.com/YosysHQ/yosys), [OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA)
and the [ASAP7](https://github.com/The-OpenROAD-Project/asap7) PDK.
