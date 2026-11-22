read_liberty ../libs/asap7.lib
read_verilog synthesized_coo.v
link_design coo_spmv_8bank
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/verilog/int8/coo.out
report_power >> ../../../data/verilog/int8/coo.out
