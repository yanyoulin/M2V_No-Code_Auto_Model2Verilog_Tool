import tensorflow as tf
import numpy as np
import os
import sys
import json

output_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_code/default_output"

model = tf.keras.models.load_model(os.path.join(output_dir, 'model.h5'))

weights_dict = {}
network_structure = []
input_dim = None
output_dim = None

for layer in model.layers:
    lname = layer.name
    if isinstance(layer, tf.keras.layers.InputLayer):
        input_dim = layer.input_shape[-1]
    elif isinstance(layer, tf.keras.layers.Dense):
        network_structure.append(('Dense', lname))
        weights = layer.get_weights()
        weights_dict[f"{lname}_kernel"] = weights[0].T  # shape (in, out)
        weights_dict[f"{lname}_bias"] = weights[1]
    elif isinstance(layer, tf.keras.layers.Flatten):
        network_structure.append(('Flatten', lname))
    elif isinstance(layer, tf.keras.layers.Activation):
        network_structure.append(('Activation', lname))
    elif hasattr(layer, 'activation'):
        act_fn = layer.activation.__name__
        if act_fn in ['relu', 'softmax', 'gelu']:
            network_structure.append(('Activation', lname))

if input_dim is None:
    input_dim = model.input_shape[-1]
output_dim = model.output_shape[-1]

print("Parsed layers:")
for i, (typ, name) in enumerate(network_structure):
    print(f"  Layer {i}: {typ} ({name})")

# Write weights.h
def array_to_c_array(name, arr):
    if arr.ndim == 2:
        arr = arr.flatten(order='C')
        out = f"const data_t {name}[{arr.size}] = {{\n    "
        out += ", ".join([f"data_t({x:.8f})" for x in arr]) + "\n};\n\n"
    elif arr.ndim == 1:
        out = f"const data_t {name}[{arr.shape[0]}] = {{\n    "
        out += ", ".join([f"data_t({x:.8f})" for x in arr]) + "\n};\n\n"
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return out

with open(os.path.join(output_dir, 'weights.h'), 'w') as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    f.write("#include \"common.h\"\n\n")
    for k, arr in weights_dict.items():
        f.write(array_to_c_array(k, arr))
    f.write("#endif\n")

# Write config.h
with open(os.path.join(output_dir, 'config.h'), 'w') as f:
    f.write("#ifndef CONFIG_H\n#define CONFIG_H\n\n")
    f.write(f"#define NUM_LAYERS {len(network_structure)}\n\n")
    for i, (typ, name) in enumerate(network_structure):
        f.write(f"#define LAYER{i}_TYPE {typ}\n")
        f.write(f"#define LAYER{i}_NAME {name}\n\n")
        
        layer_obj = model.get_layer(name)
        if isinstance(layer_obj, tf.keras.layers.Dense):
            dim = layer_obj.units
        elif isinstance(layer_obj, tf.keras.layers.Activation):
            # 對 Activation 層，輸出維度 = 前一層的輸出
            if i > 0 and isinstance(model.get_layer(network_structure[i - 1][1]), tf.keras.layers.Dense):
                dim = model.get_layer(network_structure[i - 1][1]).units
            else:
                dim = output_dim
        else:
            try:
                dim = layer_obj.output_shape[-1]
            except AttributeError:
                dim = output_dim

        f.write(f"#define LAYER{i}_DIM {dim}\n\n")
    f.write("#endif\n")

# Write common.h based on real I/O
with open(os.path.join(output_dir, 'common.h'), 'w') as f:
    f.write("#ifndef COMMON_H\n#define COMMON_H\n\n")
    f.write("#include <ap_fixed.h>\n#include <hls_math.h>\n\n")
    f.write(f"#define DIM {input_dim}\n")
    last_dense = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense = layer
            break
    if last_dense is not None:
        ff_dim = last_dense.units
    else:
        ff_dim = output_dim  # fallback（保底）

    f.write(f"#define FF_DIM {ff_dim}\n\n")
    
    f.write("typedef ap_fixed<16,6> data_t;\n\n")
    f.write("#endif\n")

# Save model_arch.json
with open("model_arch.json", "w") as f:
    json.dump(network_structure, f)