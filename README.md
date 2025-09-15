# M2V_No-Code_Auto_Model2Verilog_Tool

## Overview
Deep learning models are typically stored in .h5 format after training, which is a high-level software representation not directly executable by FPGA hardware. Manually constructing hardware designs from such models requires advanced expertise in hardware description languages, creating a barrier for software developers who wish to deploy their trained models to FPGA platforms.<br>
To address this challenge, we conducted research and ultimately developed a tool that enables seamless deployment of trained models onto FPGA platforms.<br>

## About Project
Our project M2V (Model2Verilog) provides a no-code, GUI-driven system that automatically converts trained machine learning models into Verilog code executable on FPGA. By offering a user-friendly interface, our tool enables seamless deployment of AI models to hardware accelerators, reducing the complexity of software–hardware integration.<br>

### Pre-work
**HLS4ML study** (learn how other people convert model to Verilog):<br>
**Vitis HLS study** (learn how this tool can be used to support model-to-hardware conversion):<br>

### Features
- No-Code Workflow: Users can upload a .h5 model and adjust parameters via GUI without requiring Verilog or HLS knowledge.
- Automatic Conversion: Backend scripts translate the model into HLS C++ code and execute Vitis HLS flow for synthesis and simulation.
- Hardware Deployment: Generates Verilog code compatible with FPGA platforms (test on xczu7ev-ffvc1156-2-e and pass).
- Visualization: Displays model structure, conversion status, and simulation results.
- Downloadable Package: Provides Verilog code, reports (.rpt), and simulation logs in a compressed package.
- Model Support: Supports models with common activation functions such as ReLU, GELU, and Softmax.
<br>
![image]() <br>

### Project Flowchart
1. Writing Sorce Codes:<br>
Source codes include data type definitions, Dense layers, GELU, and ReLU activation. The architecture is implemented using Fixed-Point, fully unrolled, and pipelined execution.<br>
2. Weights & Config:<br>
Traverse the entire .h5 file, identify & collect the Kernel and Bias of each layer, and create a network structure list that records the type of each layer (such as Dense layer or Activation layer).<br>
3. Top Function:<br>
Read config.h, organize an inference process based on the function of each layer (such as fully connected layers or activation functions). The final generated inference framework can directly correspond to the hardware design process.<br>
4. build_prj.tcl:<br>
This script is responsible for automating the construction of a HLS project, covering steps such as project creation, design file integration, clock and target device, inference simulation, hardware synthesis, and C-to-RTL comparison.<br>
5. Run ALL!!:<br>
Write a Python script to run the entire process of converting a model into hardware. The script should sequentially complete model training, weight and structure extraction, inference program generation, and High-Level Synthesis (HLS), enabling the entire design workflow to be executed with a single command.<br>

### System Architecture
![image]() <br>
The project is divided into frontend and backend components:
- Frontend
  - Upload .h5 model and set conversion parameters.
  - Visualize model structure and monitor conversion progress.
  - Provide downloadable results package (Verilog code + reports).
- Backend
  - Parse the model architecture (input dimensions, Dense layers, activation types).
  - Automatically generate weight.h, config.h, and the top function for HLS project.
  - Execute Vitis HLS flow

## Experiments & Results
1. MNIST MLP Model
Architecture: 784 → Dense(32) → ReLU → Dense(10) → Softmax <br>
Result: HLS inference output matches Keras predictions with error < 0.005% <br>
2. Scikit-learn Digits Model
Simplified model successfully converted and executed.<br>
<br>
These experiments validate that our system can faithfully reproduce the results of the original software models on FPGA-generated hardware implementations.<br>

## Contributions
- Developed a code-free conversion tool, lowering the entry barrier for hardware deployment of ML models.
- Enabled users without FPGA/Verilog knowledge to experiment with AI hardware acceleration.
- Automated the end-to-end workflow (conversion, synthesis, simulation, and packaging).
- Compared outputs between Keras and HLS versions for validation.
- Designed for extensibility, supporting additional architectures and functions.

## Challenges
- Limited Model Support: Current version does not fully support CNNs and Transformer architectures.
- Compatibility: Requires further testing with datasets and formats beyond TensorFlow and Scikit-learn.
- Optimization: Automatic pragma insertion and resource utilization tuning remain ongoing challenges.
- Performance Gap: Compared to HLS4ML, our tool requires further improvement in efficiency and resource usage.

## Demo
[Deme Video]([https://github.com](https://youtu.be/IdfV-gUlPQM))

## Comparison with HLS4ML
![image]() <br>

## Team
Students: 林彥佑, 曾若恩 <br>
Advisor: Prof. 陳添福 <br>


