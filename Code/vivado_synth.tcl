# Read your exported Verilog
foreach f [glob ./my_mlp_project/solution1/syn/verilog/*.v] {
    read_verilog $f
}

# Synthesis
synth_design -top mlp_inference

# Place & Route
place_design
route_design

# Reports
report_timing_summary
report_utilization

exit
