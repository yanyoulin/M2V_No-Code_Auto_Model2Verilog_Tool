import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import os
import sys
import json

def compare_outputs(output_dir):
    model_path = os.path.join(output_dir, "model.h5")
    input_path = os.path.join(output_dir, "example_input.txt")
    hls_output_path = os.path.join(output_dir, "my_mlp_project", "solution1", "csim", "build", "output.txt")
    label_path = os.path.join(output_dir, "example_label.txt")

    for path in [model_path, input_path, hls_output_path, label_path]:
        if not os.path.exists(path):
            print(f"Missing required file: {path}")
            return

    # 讀取資料
    model = load_model(model_path)

    # 讀取輸入資料
    x = np.loadtxt(input_path).reshape(1, -1)
    '''
    print("\nKeras intermediate layer outputs:")
    intermediate_outputs = []

    
    layer_input = x
    for i, layer in enumerate(model.layers):
        layer_output = layer(layer_input)
        layer_input = layer_output  # 為下一層輸入
        print(f"Layer {i} ({layer.name}) output shape: {layer_output.shape}")
        print(layer_output.numpy().flatten())
        intermediate_outputs.append(layer_output.numpy().flatten())'''

    keras_out = model.predict(x, verbose=0)[0]
    hls_out = np.loadtxt(hls_output_path)
    true_label = int(open(label_path).read().strip())

    # 輸出 softmax 比較
    #print("\nSoftmax output comparison:")
    #for i, (k, h) in enumerate(zip(keras_out, hls_out)):
        #print(f"[{i}] Keras: {k:.6f} | HLS: {h:.6f} | Δ = {abs(k - h):.6f}")

    # 比對結果
    keras_pred = np.argmax(keras_out)
    hls_pred = np.argmax(hls_out)

    #print("\nKeras predicted class:", keras_pred)
    #print("HLS predicted class:  ", hls_pred)
    #print("Ground Truth Label:   ", true_label)

    match = keras_pred == hls_pred
    correct = hls_pred == true_label

    #print("Match status: ", "Match with Keras" if match else "Mismatch with Keras")
    #print("Prediction correctness: ", "HLS correct" if correct else "HLS wrong")
    
    result = {
        "softmax": [
            {"index": i, "keras": float(keras_out[i]), "hls": float(hls_out[i]), "delta": abs(keras_out[i] - hls_out[i])}
            for i in range(len(keras_out))
        ],
        "keras_pred": int(np.argmax(keras_out)),
        "hls_pred": int(np.argmax(hls_out)),
        "label": int(true_label),
        "match": keras_pred == hls_pred,
        "correct": hls_pred == true_label
    }
    # 確保所有值都是 JSON 可序列化的型別
    result["match"] = bool(result["match"])
    result["correct"] = bool(result["correct"])

    # 儲存到 output_dir/result.json
    json_path = os.path.join("result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Result written to {json_path}")

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "generated_code/latest_run"
    compare_outputs(out_dir)