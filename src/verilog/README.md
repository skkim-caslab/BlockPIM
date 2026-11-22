# PIM Processing Modules (RTL)

RTL of the per-bank processing units compared in Fig. 1(b), plus the scripts
that synthesize them (Yosys, ASAP7 7 nm) and report timing and power
(OpenSTA, 940 MHz). Fig. 1(b) reads the area and critical-path delay, and
Fig. 6 reads the power, from the reports in `data/verilog/int8/`.

| Directory | Top module | Design |
|-----------|------------|--------|
| `dense/` | `dense_spmv_8bank` | DensePIM: 32-byte dense MAC per bank |
| `csr/` | `csr_spmv_8bank` | CSR processing unit (SpaceA-style) |
| `coo/` | `coo_spmv_8bank` | COO processing unit (SparsePIM/pSyncPIM-style) |
| `dense_stackpim/` | `dense_stackpim_spmv_8bank` | BlockPIM: dense MAC + bitmap Find-First-One (`bitmap_ffo`) |

Each directory holds the RTL (`<name>.v`), the Yosys script
(`synth_<name>.ys`) and the OpenSTA script (`sta.tcl`).
Shared files: `config/constraints_940MHz.sdc` (clock) and `libs/asap7.lib`
(ASAP7 standard cells, merged TT library).

## Running

Install [Yosys](https://github.com/YosysHQ/yosys) and
[OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) so that `yosys` and
`sta` are on `PATH` (the reports in `data/verilog/int8/` were produced with
Yosys 0.53+101 and OpenSTA 2.7.0; other versions may give slightly different
area and timing), then:

```bash
bash run_all.sh
```

The script overwrites `data/verilog/int8/{dense,csr,coo,dense_stackpim_32}.out`
(Yosys `stat` area report followed by the OpenSTA `report_checks` and
`report_power` output). The plotting scripts apply the 10x DRAM-process area
penalty and the per-format MAC counts on top of these reports.
