read_liberty ../libs/asap7.lib
read_verilog synthesized_dense_stackpim.v
link_design dense_stackpim_spmv_8bank
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/verilog/int8/dense_stackpim_32.out
report_power >> ../../../data/verilog/int8/dense_stackpim_32.out
