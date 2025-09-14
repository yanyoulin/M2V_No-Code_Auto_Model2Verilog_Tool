from flask import Flask, render_template, request, jsonify
import os
import subprocess
import json
import shutil
import re
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

latest_timestamp_path = "latest_timestamp.txt"

STATIC_FILES = [
    "dense.h", "gelu.h", "relu.h", "top.h", "test.cpp",
    "softmax.h"
]


def get_latest_timestamp():
    if os.path.exists(latest_timestamp_path):
        with open(latest_timestamp_path, 'r') as f:
            return f.read().strip()
    return ""

def set_latest_timestamp(ts):
    with open(latest_timestamp_path, 'w') as f:
        f.write(ts)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("generated_code", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    set_latest_timestamp(timestamp)
    for f in STATIC_FILES:
        shutil.copy(f, os.path.join(output_dir, f))

    if request.headers.get('X-Use-Example') == 'true':
        subprocess.run(['python', 'train_test.py', output_dir])
        subprocess.run(['python', 'read_model.py', output_dir])
        subprocess.run(['python', 'generate_top.py', output_dir])
        subprocess.run(['python', 'generate_build_prj.py', output_dir])
        return {'status': 'example used'}

    file = request.files['model']
    if file.filename.endswith('.h5'):
        h5_path = os.path.join(output_dir, 'model.h5')
        file.save(h5_path)
        subprocess.run(['python', 'read_model.py', output_dir], check=True)
        subprocess.run(['python', 'generate_top.py', output_dir])
        subprocess.run(['python', 'generate_build_prj.py', output_dir])
        return {'status': 'uploaded'}

    return {'status': 'fail'}, 400

@app.route('/list_all_folders')
def list_all_folders():
    try:
        folders = sorted(os.listdir("generated_code"))
        return jsonify({"folders": folders})
    except Exception:
        return jsonify({"folders": []})

@app.route('/set_timestamp')
def set_timestamp():
    ts = request.args.get("ts")
    if not ts:
        return {"status": "error", "message": "Missing timestamp"}, 400
    if not os.path.exists(os.path.join("generated_code", ts)):
        return {"status": "error", "message": "Folder not found"}, 404
    set_latest_timestamp(ts)
    return {"status": "ok"}

@app.route('/get_arch', methods=['GET'])
def get_arch():
    arch_path = os.path.join("model_arch.json")
    if not os.path.exists(arch_path):
        return {"error": "model_arch.json not found"}, 404

    with open(arch_path, 'r') as f:
        arch = json.load(f)
    return {'layers': arch}

import re

@app.route('/submit_hls', methods=['POST'])
def submit_hls():
    ts = get_latest_timestamp()
    if not ts:
        return {"status": "error", "detail": "No recent timestamp found"}

    output_dir = os.path.join("generated_code", ts)
    layer_config_path = os.path.join(output_dir, "layer_config.json")
    config_h_path = os.path.join(output_dir, "config.h")

    # 1. 儲存使用者 HLS 設定到 layer_config.json
    data = request.get_json()
    with open(layer_config_path, "w") as f:
        json.dump(data, f, indent=2)

    # 2. 讀取原始 config.h
    if os.path.exists(config_h_path):
        with open(config_h_path, "r") as f:
            lines = f.readlines()
    else:
        lines = ["#ifndef CONFIG_H", "#define CONFIG_H", "#endif"]

    # 3. 移除舊的 CONFIG_REUSE_ / CONFIG_PIPELINE_ / typedef 行
    param_pattern = re.compile(r'CONFIG_REUSE_|CONFIG_PIPELINE_|typedef\s+.*data_.*_t')
    clean_lines = []
    for line in lines:
        if param_pattern.search(line):
            continue
        clean_lines.append(line.rstrip())

    # 4. 尋找 #endif 的 index，插入前處填入參數
    try:
        endif_idx = clean_lines.index("#endif")
    except ValueError:
        # 若沒有 #endif 就加在最後
        endif_idx = len(clean_lines)
        clean_lines.append("#endif")

    # 5. 構造新參數段
    param_lines = [""]
    for lname, params in data.items():
        reuse = params.get("reuse", 1)
        pipeline = params.get("pipeline", "Disabled")
        precision = params.get("precision", "ap_fixed<16,6>")

        param_lines.append(f"#define CONFIG_REUSE_{lname.upper()} {reuse}")
        if pipeline == "Enabled":
            param_lines.append(f"#define CONFIG_PIPELINE_{lname.upper()}")
        param_lines.append(f"typedef {precision} data_{lname}_t;")

    # 6. 插入參數段到 #endif 之前
    final_lines = clean_lines[:endif_idx] + param_lines + clean_lines[endif_idx:]

    # 7. 寫回 config.h
    with open(config_h_path, "w") as f:
        f.write("\n".join(final_lines) + "\n")

    # 8. 重新產生 top.cpp
    try:
        subprocess.run(["python", "generate_top.py", output_dir], check=True)
    except subprocess.CalledProcessError as e:
        return {"status": "error", "detail": "Failed to generate top.cpp:\n" + e.stderr.decode()}

    return {"status": "updated"}

@app.route('/preview_config')
def preview_config():
    ts = get_latest_timestamp()
    if not ts:
        return "No config.h found."
    path = os.path.join("generated_code", ts, "config.h")
    if not os.path.exists(path):
        return "No config.h found."
    with open(path, "r") as f:
        return f.read()

@app.route("/run_hls")
def run_hls():
    ts = get_latest_timestamp()
    output_dir = os.path.join("generated_code", ts)
    with open("hls_run.log", "w") as f:
        subprocess.Popen(
            ["vitis_hls", "-f", "build_prj.tcl"],
            cwd=output_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    return "Started HLS"

@app.route("/run_vivado")
def run_vivado():
    ts = get_latest_timestamp()
    output_dir = os.path.join("generated_code", ts)
    tcl_src = "vivado_synth.tcl"
    tcl_dst = os.path.join(output_dir, "vivado_synth.tcl")

    # 複製 tcl 檔案進去 output_dir
    if not os.path.exists(tcl_dst):
        shutil.copy(tcl_src, tcl_dst)
    with open("vivado_run.log", "w") as f:
        subprocess.Popen(
            ["vivado", "-mode", "batch", "-source", "vivado_synth.tcl"],
            cwd=output_dir,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    return "Started Vivado"

@app.route("/log_hls")
def log_hls():
    if os.path.exists("hls_run.log"):
        with open("hls_run.log") as f:
            return f.read()
    return "No HLS log yet."

@app.route("/log_vivado")
def log_vivado():
    if os.path.exists("vivado_run.log"):
        with open("vivado_run.log") as f:
            return f.read()
    return "No Vivado log yet."

@app.route("/get_utilization")
def get_utilization():
    rpt_path = "my_mlp_project/solution1/syn/report/csynth.rpt"
    if not os.path.exists(rpt_path):
        return "Utilization report not found."

    dsp = lut = ff = bram = "N/A"
    with open(rpt_path) as f:
        for line in f:
            if "DSP" in line:
                dsp = line.split(":")[-1].strip()
            elif "LUT" in line:
                lut = line.split(":")[-1].strip()
            elif "FF" in line:
                ff = line.split(":")[-1].strip()
            elif "BRAM" in line:
                bram = line.split(":")[-1].strip()

    return f"DSP: {dsp}, LUT: {lut}, FF: {ff}, BRAM: {bram}"

@app.route("/view_rpt")
def view_rpt():
    file = request.args.get("file", "")
    rpt_map = {
        "csynth": "my_mlp_project/solution1/syn/report/csynth.rpt",
        "csynth_size": "my_mlp_project/solution1/syn/report/csynth_design_size.rpt",
        "util": "vivado.log",
        "synth": "my_mlp_project/solution1/syn/report/synth_1.rpt"
    }
    if file not in rpt_map:
        return f"Unknown report type: {file}", 400

    ts = get_latest_timestamp()
    if not ts:
        return "No recent timestamp folder found.", 404

    path = os.path.join("generated_code", ts, rpt_map[file])
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return f"{file}.rpt not found at path: {path}", 404

@app.route("/check_status")
def check_status():
    result = {}

    def check_hls():
        try:
            with open("hls_run.log") as f:
                log = f.read()
            return "C/RTL co-simulation finished: PASS" in log
        except:
            return False

    def check_vivado():
        try:
            with open("vivado_run.log") as f:
                log = f.read()
            return "synth_design" in log and "Completed successfully" in log
        except:
            return False

    result["hls"] = check_hls()
    result["vivado"] = check_vivado()
    return result

@app.route("/list_generated_code")
def list_generated_code():
    try:
        base = "generated_code"
        latest_dir = sorted(os.listdir(base))[-1]
        files = os.listdir(os.path.join(base, latest_dir))
        return json.dumps({"dir": latest_dir, "files": files})
    except Exception:
        return json.dumps({"dir": "", "files": []})

@app.route('/view_file')
def view_file():
    dir = request.args.get('dir')
    file = request.args.get('file')
    path = os.path.join('generated_code', dir, file)
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return f"{file} not found in {dir}", 404

@app.route("/run_compare")
def run_compare():
    try:
        base = "generated_code"
        latest_dir = sorted(os.listdir(base))[-1]
        output_dir = os.path.join(base, latest_dir)

        # 先執行 compare.py 產生 result.json（在根目錄）
        subprocess.run(['python', 'compare.py', output_dir])

        # 從根目錄讀取 result.json
        json_path = "result.json"
        with open(json_path, "r") as f:
            result = json.load(f)

        return jsonify(result)
    except Exception as e:
        print("/run_compare error:", e)
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
