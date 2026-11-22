read_liberty ../libs/asap7.lib
read_verilog synthesized_csr.v
link_design csr_spmv_8bank
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/verilog/int8/csr.out
report_power >> ../../../data/verilog/int8/csr.out
