from software_model.sw_transformer import(
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)

from software_model.sw_utils import data_type_dict, Tensor
from hardware_model.hw_system import system_dict
import sys


def generate_cycle_table(log_path, mode="pim"):
    op_stats = {}
    current_op = None
    
    # Device mapping 
    if mode == "npu_only":
        # In NPU-only mode, everything is NPU
        default_device = 'NPU'
        device_map = {
            'layernorm(MHA)': 'NPU',
            'q_projection': 'NPU',
            'k_projection': 'NPU',
            'v_projection': 'NPU',
            'q_mul_k': 'NPU',
            'softmax': 'NPU',
            'a_mul_v': 'NPU',
            'w0_projection': 'NPU',
            'all_reduce(MHA)': 'NPU',
            'layernorm(FFN)': 'NPU',
            'w1_projection': 'NPU',
            'gelu': 'NPU',
            'w2_projection': 'NPU',
            'all_reduce(FFN)': 'NPU',
            'flash_attention': 'NPU'
        }
    else:
        # In PIM mode, specific ops are on PIM
        # User requested all_reduce to be on PIM in PIM mode
        device_map = {
            'layernorm(MHA)': 'NPU',
            'q_projection': 'PIM',
            'k_projection': 'PIM',
            'v_projection': 'PIM',
            'q_mul_k': 'PIM',
            'softmax': 'NPU',
            'a_mul_v': 'PIM',
            'w0_projection': 'PIM',
            'all_reduce(MHA)': 'PIM',
            'layernorm(FFN)': 'NPU',
            'w1_projection': 'PIM',
            'gelu': 'NPU',
            'w2_projection': 'PIM',
            'all_reduce(FFN)': 'PIM',
            'flash_attention': 'PIM'
        }
    
    # Ordered list for display
    op_order = [
        'layernorm(MHA)', 'q_projection', 'k_projection', 'v_projection', 
        'q_mul_k', 'softmax', 'a_mul_v', 'flash_attention', 'w0_projection', 
        'all_reduce(MHA)', 'layernorm(FFN)', 'w1_projection', 'gelu', 
        'w2_projection', 'all_reduce(FFN)'
    ]

    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("simulating "):
                    raw_op = line.replace("simulating ", "").split("(")[0].strip()
                    
                    # Map to canonical names
                    if "layernorm(MHA)" in line: current_op = "layernorm(MHA)"
                    elif "q projection" in line: current_op = "q_projection"
                    elif "k projection" in line: current_op = "k_projection"
                    elif "v projection" in line: current_op = "v_projection"
                    elif "flash attention" in line: current_op = "flash_attention"
                    elif "q_mul_k" in line: current_op = "q_mul_k"
                    elif "softmax" in line: current_op = "softmax"
                    elif "a_mul_v" in line: current_op = "a_mul_v"
                    elif "w0_projection" in line: current_op = "w0_projection"
                    elif "all_reduce(MHA)" in line: current_op = "all_reduce(MHA)"
                    elif "layernorm(FFN)" in line: current_op = "layernorm(FFN)"
                    elif "w1_projection" in line: current_op = "w1_projection"
                    elif "gelu" in line: current_op = "gelu"
                    elif "w2_projection" in line: current_op = "w2_projection"
                    elif "all_reduce(FFN)" in line: current_op = "all_reduce(FFN)"
                    else: current_op = raw_op

                    if current_op not in op_stats:
                        op_stats[current_op] = {'compute': 0, 'io': 0, 'device': device_map.get(current_op, 'UNKNOWN')}
                
                if current_op:
                    if "compute cycle(X1)" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            val = float(parts[1].strip())
                            op_stats[current_op]['compute'] += val
                            
                    if "io cycle(X2)" in line:
                         parts = line.split(":")
                         if len(parts) > 1:
                            val = float(parts[1].strip())
                            op_stats[current_op]['io'] += val
                            
    except Exception as e:
        print(f"Error parsing log for stats: {e}")
        return

    # Print Table
    print("\n" + "="*85)
    print(f"{'Operation':<25} {'Device':<10} {'NPU Comp':<12} {'NPU IO':<12} {'PIM Comp':<12} {'PIM IO':<12}")
    print("-" * 85)
    
    calc_npu_compute = 0
    calc_npu_io = 0
    calc_pim_compute = 0
    calc_pim_io = 0
    
    # Filter and sort ops that were actually found
    found_ops = [op for op in op_order if op in op_stats]
    # Add any others found but not in ordered list (fallback)
    found_ops += [op for op in op_stats if op not in found_ops]
    
    for op_name in found_ops:
        stat = op_stats[op_name]
        npu_compute = stat['compute'] if stat['device'] == 'NPU' else 0
        npu_io = stat['io'] if stat['device'] == 'NPU' else 0
        pim_compute = stat['compute'] if stat['device'] == 'PIM' else 0
        pim_io = stat['io'] if stat['device'] == 'PIM' else 0
        
        print(f"{op_name:<25} {stat['device']:<10} {npu_compute:<12.0f} {npu_io:<12.0f} {pim_compute:<12.0f} {pim_io:<12.0f}")
        
        calc_npu_compute += npu_compute
        calc_npu_io += npu_io
        calc_pim_compute += pim_compute
        calc_pim_io += pim_io
        
    print("-" * 85)
    print(f"{'TOTAL':<25} {'':<10} {calc_npu_compute:<12.0f} {calc_npu_io:<12.0f} {calc_pim_compute:<12.0f} {calc_pim_io:<12.0f}")
    print("="*85 + "\n")


if __name__ == "__main__":
    bs = 1
    INITFLAG = False

    if "--bs" in sys.argv:
        idx = sys.argv.index("--bs")
        if idx + 1 < len(sys.argv):
            bs = int(sys.argv[idx + 1])

    if len(sys.argv) > 3 and sys.argv[-1] == "--init":
        INITFLAG = True
    
    output_token_length = int(sys.argv[1])
    config_file = sys.argv[2]
    model_dim = int(sys.argv[3])
    head_num = int(sys.argv[4])
#    INITFLAG = "--init" in sys.argv

    sr_data_type = "int8"
    if "--datatype" in sys.argv:
        idx = sys.argv.index("--datatype")
        if idx + 1 < len(sys.argv):
            sr_data_type = sys.argv[idx+1]

    s = 1024
    if len(sys.argv) < 3:
        print("Usage: python simulate.py <seq_len> <tile config_file> [dimension] [head_num] [--datatype int8/fp16/fp32]")
        sys.exit(1)

    import os
    from datetime import datetime
    
    # Setup logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    import json
    
    config_name = sys.argv[2].split('/')[-1].replace('.json', '')
    mode_str = "npu_only" if "--npu_only" in sys.argv else "pim"
    
    # Determine PIM config type from JSON
    pim_config_type = "csr" # default
    try:
        with open(config_file, 'r') as f:
            tile_config = json.load(f)
            # Scan for any non-default config
            for key, val in tile_config.items():
                if 'pim_config' in val:
                    if val['pim_config'] == 'dense':
                        pim_config_type = 'dense'
                        break
                    elif val['pim_config'] == 'coo':
                        pim_config_type = 'coo'
                    elif val['pim_config'] == 'psyncpim':
                        pim_config_type = 'psyncpim'
    except Exception as e:
        print(f"Warning: Could not read config for filename generation: {e}")

    # Parse --tag argument for logging
    tag_str = ""
    if "--tag" in sys.argv:
        try:
            tag_idx = sys.argv.index("--tag")
            if tag_idx + 1 < len(sys.argv):
                tag_str = f"_{sys.argv[tag_idx+1]}"
        except ValueError:
            pass

    log_file = f"{log_dir}/log_{mode_str}_{pim_config_type}_{config_name}{tag_str}.txt"
    
    # Redirect stdout
    sys.stdout = open(log_file, 'w')
    
    # Redirect stdout
    sys.stdout = open(log_file, 'w')
    
    if "--sigma" in sys.argv:
        TPU_system = system_dict["SIGMA_"+sr_data_type]
        print("Using SIGMA System")
    else:
        TPU_system = system_dict["NPU_"+sr_data_type]
    
    # Check for mode argument
    mode = "pim"
    if "--npu_only" in sys.argv:
        mode = "npu_only"
    
    if mode == "npu_only":
        PIM_system = TPU_system
        print("Running in NPU only mode")
    else:
        PIM_system = system_dict["PIM_"+sr_data_type]
        print("Running in PIM mode")

    # Check for sparsity argument
    sparsity = 0.0
    for i, arg in enumerate(sys.argv):
        if arg == "--sparsity":
            sparsity = float(sys.argv[i+1])
            pass

    if INITFLAG == True:
        model = TransformerBlockInitComputationTP(
            d_model = model_dim,
            n_heads=head_num,  #1.3B 16, 6.7B 32
            device_count=1,
            data_type=data_type_dict[sr_data_type],
            system=TPU_system,
            config_file = config_file,
            use_flash_attention=False,
        )
        _ = model(
            Tensor([bs, s, model_dim], data_type_dict[sr_data_type]) 
        )
        latency = model.compile_and_simulate(TPU_system, PIM_system, sparsity=sparsity)
        print(f"Latency, {latency}", file=sys.__stdout__)
        import sys as _sys
        print(f"Compute_Latency, {getattr(_sys, 'total_compute_latency', 0.0)}", file=_sys.__stdout__)
        print(f"IO_Latency, {getattr(_sys, 'total_io_latency', 0.0)}", file=_sys.__stdout__)
    else:        
        model = TransformerBlockAutoRegressionTP(
            d_model = model_dim,
            n_heads=head_num,  #1.3B 16, 6.7B 32
            device_count=1,
            data_type=data_type_dict[sr_data_type],
            system=TPU_system,
            config_file = config_file,
            use_flash_attention=False,
        )
        _ = model(
            Tensor([bs, 1, model_dim], data_type_dict[sr_data_type]), s + output_token_length
        )
        latency = model.compile_and_simulate(TPU_system, PIM_system, sparsity=sparsity)
        print(f"Latency, {latency}", file=sys.__stdout__)
        import sys as _sys
        print(f"Compute_Latency, {getattr(_sys, 'total_compute_latency', 0.0)}", file=_sys.__stdout__)
        print(f"IO_Latency, {getattr(_sys, 'total_io_latency', 0.0)}", file=_sys.__stdout__)

    # Generate and print per-operation stats table from log
    sys.stdout.flush()
    generate_cycle_table(log_file, mode=mode)

