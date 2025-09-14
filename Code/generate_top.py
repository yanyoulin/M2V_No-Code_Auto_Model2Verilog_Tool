import re
import os
import sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_code/default_output"
config_path = os.path.join(output_dir, 'config.h')
common_path = os.path.join(output_dir, 'common.h')
output_path = os.path.join(output_dir, 'top.cpp')

layer_list = []
dim_map = {}

# 讀取 config.h 中的層類型、名稱、維度
with open(config_path, 'r') as f:
    for line in f:
        if '#define LAYER' in line and '_TYPE' in line:
            layer_type = line.strip().split()[-1]
            idx = int(re.search(r'LAYER(\d+)_TYPE', line).group(1))
            while len(layer_list) <= idx:
                layer_list.append({})
            layer_list[idx]['type'] = layer_type
        elif '#define LAYER' in line and '_NAME' in line:
            layer_name = line.strip().split()[-1]
            idx = int(re.search(r'LAYER(\d+)_NAME', line).group(1))
            layer_list[idx]['name'] = layer_name
        elif '#define LAYER' in line and '_DIM' in line:
            idx = int(re.search(r'LAYER(\d+)_DIM', line).group(1))
            dim = int(line.strip().split()[-1])
            dim_map[idx] = dim

# 找出最大維度以安全宣告 buffer
max_dim = max(dim_map.values())

DIM = dim_map[0]
with open(common_path, 'r') as f:
    for line in f:
        if '#define DIM' in line:
            dim = int(line.strip().split()[-1])
            DIM = dim

if len(dim_map) != len(layer_list):
    raise ValueError("每層必須在 config.h 中定義 _DIM")

with open(output_path, 'w') as f_out:
    f_out.write('#include "top.h"\n#include "dense.h"\n#include "gelu.h"\n#include "relu.h"\n#include "softmax.h"\n#include "weights.h"\n#include "config.h"\n\n')
    f_out.write('void mlp_inference(data_t input[DIM], data_t output[FF_DIM]) {\n')
    f_out.write('#pragma HLS array_partition variable=input complete\n')
    f_out.write('#pragma HLS array_partition variable=output complete\n\n')
    f_out.write(f'    data_t buffer0[{max_dim}];\n')
    f_out.write(f'    data_t buffer1[{max_dim}];\n')
    f_out.write('    data_t* buffers[] = {buffer0, buffer1};\n')
    f_out.write('    data_t* cur = input;\n')
    f_out.write('    data_t* next = buffer0;\n')
    f_out.write('    int toggle = 0;\n\n')

    for i, layer in enumerate(layer_list):
        lname = layer['name']
        ltype = layer['type']

        if ltype == 'Dense':
            if i == 0:
                in_dim = DIM
            else:
                for j in range(i - 1, -1, -1):
                    if layer_list[j]['type'] in ['Dense', 'Activation', 'Flatten']:
                        in_dim = dim_map[j]
                        break
            out_dim = dim_map[i]
        elif ltype == 'Activation':
            in_dim = dim_map[i - 1]
            out_dim = in_dim
        else:
            for j in range(i - 1, -1, -1):
                if layer_list[j]['type'] in ['Dense', 'Activation', 'Flatten']:
                    in_dim = dim_map[j]
                    break
            out_dim = dim_map[i]

        cur = 'input' if i == 0 else 'buffers[toggle ^ 1]'
        next = 'buffers[toggle]'

        # 插入 pragma pipeline 條件
        macro_pipeline = f'CONFIG_PIPELINE_{lname.upper()}'
        f_out.write(f'#ifdef {macro_pipeline}\n#pragma HLS PIPELINE\n#endif\n')

        # 層函數呼叫
        if ltype == 'Dense':
            f_out.write(f'    dense<{in_dim}, {out_dim}>({cur}, {lname}_kernel, {lname}_bias, {next});\n')
        elif ltype == 'Activation':
            lname_lower = lname.lower()
            if 'relu' in lname_lower:
                f_out.write(f'    relu<{in_dim}>({cur}, {next});\n')
            elif 'gelu' in lname_lower:
                f_out.write(f'    for (int i = 0; i < {in_dim}; i++) {{\n')
                f_out.write(f'#pragma HLS UNROLL\n        {next}[i] = gelu({cur}[i]);\n    }}\n')
            elif 'softmax' in lname_lower:
                f_out.write(f'    softmax<{in_dim}>({cur}, {next});\n')
            else:
                raise ValueError(f"Unsupported activation type: {lname}")
        elif ltype == 'Flatten':
            f_out.write('    // Flatten layer: no-op in HLS\n')
            continue
        else:
            raise ValueError(f"Unsupported layer type: {ltype}")

        f_out.write('    toggle ^= 1;\n\n')

    f_out.write('    for (int i = 0; i < FF_DIM; i++) {\n')
    f_out.write('        output[i] = buffers[toggle ^ 1][i];\n')
    f_out.write('    }\n')
    f_out.write('}\n')
