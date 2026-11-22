# Block Pruning (SparseGPT-based)

This directory contains the pruning code for the perplexity experiments of
Fig. 1(a) and Fig. 7. It is a modified copy of
[SparseGPT](https://github.com/IST-DASLab/sparsegpt) (Frantar and Alistarh,
ICML 2023), distributed under its original license (`LICENSE`, Apache 2.0).

## Changes relative to SparseGPT

- `sparsegpt.py`: **hardware-aligned block pruning** (`--group_bytes`, in bytes
  of INT8 weights). Weights are ranked by the SparseGPT saliency
  (`w^2 / [H^-1]_ii^2`); going down the ranking, the whole block that contains each highly ranked weight is kept
  until the target density is reached. Blocks are therefore retained by the
  presence of critical weights, not by an average/median score.
- `llama.py`: LLaMA-3 support, `--group_bytes`, per-layer sparsity
  (`--layer-sparsities`, used for the CSR/COO+Dense hybrids), and INT8
  weight quantization through `--wbits 8`.
- `datautils.py`: Hugging Face dataset names updated (`Salesforce/wikitext`)
  and LLaMA-3 tokenizer loading through `AutoTokenizer`.
- `run_ppl.py`: re-runs the points listed in `data/pruning/ppl_llama3.csv`.

## Block format

With `--group_bytes 32`, the pruning unit is a 32-byte block of 32 INT8 weights,
matching the 32-byte DRAM access granularity and the PIM MAC width. Each block
is either kept entirely or pruned entirely; the kept blocks are stored
contiguously and a bitmap (one bit per block) records which blocks are present.
A target memory compression ratio `c` therefore corresponds to a pruning ratio
of `c + 10/256` (the bitmap overhead used in the paper).

## Requirements

`torch`, `transformers`, `datasets` and one GPU with enough memory
to hold one decoder layer plus its calibration activations (LLaMA-3.1 8B:
one 24 GB GPU, e.g. RTX A5000; the 70B points need a larger GPU or several
GPUs). Access to the gated
`meta-llama/Llama-3.1-8B` / `-70B` checkpoints on Hugging Face is required.

## Usage

```bash
# Dense baseline (INT8)
python llama.py meta-llama/Llama-3.1-8B c4 --sparsity 0.0 --blocksize 128 --wbits 8

# BlockPIM: 10% memory compression = 13.9% pruning (10/256 bitmap overhead), 32-byte blocks
python llama.py meta-llama/Llama-3.1-8B c4 --sparsity 0.139 --blocksize 128 --group_bytes 32 --wbits 8

# CSR+Dense hybrid: first 11 layers dense, the rest pruned
python llama.py meta-llama/Llama-3.1-8B c4 --layer-sparsities '0-10:0.0,*:0.6610' --blocksize 128 --wbits 8

# Print (or run with --run) every command behind Fig. 1(a) / Fig. 7
python run_ppl.py --model 8B --only-reported
```

`llama.py` prints the perplexity on WikiText-2, PTB and C4, in that order; the
figures use the C4 value (the last `Perplexity:` line). Perplexity after pruning
depends on the calibration samples and the random seed (`--seed`,
`--nsamples`), so re-running a point can give values that differ from those
in `data/pruning/ppl_llama3.csv`. Run times were not
logged; as a rough estimate, one 8B point (pruning 32 layers plus three
evaluations) takes on the order of half an hour on a single 24 GB GPU.

## Citation (SparseGPT)

```bibtex
@article{frantar-sparsegpt,
  title   = {{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot},
  author  = {Elias Frantar and Dan Alistarh},
  year    = {2023},
  journal = {arXiv preprint arXiv:2301.00774}
}
```
