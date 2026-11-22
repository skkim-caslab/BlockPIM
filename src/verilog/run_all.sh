#!/bin/bash
# Synthesizes the four PIM processing modules with Yosys (ASAP7) and runs
# OpenSTA at 940 MHz. Each module's area, timing and power report is written
# to data/verilog/int8/<module>.out, which Fig. 1(b) and Fig. 6 read.
# Requires `yosys` and `sta` (OpenSTA) on PATH; binaries placed in this
# directory are picked up as well.
set -e
cd "$(dirname "$0")"
export PATH="$(pwd):$PATH"

dirs=(dense csr coo dense_stackpim)

for d in "${dirs[@]}"; do
    echo "===== Running in $d ====="
    (
        cd "$d"
        yosys -s synth_${d}.ys
        sta -exit sta.tcl
    )
done
