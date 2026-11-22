# Clock: 940 MHz (period 1.0638 ns)
create_clock -name clk -period 1.0638 [get_ports clk]
set_output_delay -clock clk 2 [all_outputs]
set_load 1 [all_outputs]
