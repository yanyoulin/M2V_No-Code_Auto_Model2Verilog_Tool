# 專案更新紀錄

## 一、程式邏輯與架構修正

1. **主要調整**：
   - `generate_top.py` 移除 `#pragma HLS UNROLL` 指令，以解決因 array partition 所引發的錯誤。
   - `dense.h` 及 `dense.cpp` 中的 `weight` 宣告由二維陣列 `data_t weight[FF_DIM][DIM]` 改為一維 `data_t weight[FF_DIM * DIM]`，以符合合成限制與提升相容性。
   - 修改了 `read_model.py` 的程式邏輯，解決原先必須要以 Dense 層開始跟結束的預設，以及可以依照正確的順序讀取每一層。
   - 此修改使專案能成功完成 C-SIM 階段。

## 二、訓練與測試資料產生 (`train_test`) 模組

目前共有三個版本：
- `train_test_v1.py`：隨機生成亂數。
- `train_test_v2.py`：【本版本】使用 sklearn 的 Digits 資料集，降維後僅取前 4 維特徵。可快速產生模型並適用於驗證 CSIM 輸出是否一致。
- `train_test.py`(第三版)：使用 MNIST 資料進行訓練，精度較高（Keras 準確率約 90% 以上），但進行 CSIM 測試需耗時約 10 分鐘。
    > 撰寫第三版的 `train_test.py` 是為了驗證 Keras 跟 HLS 的表現與準確率無關，在測試時使用第二版即可，因為不論 Keras 準確率多少，Keras 跟 HLS 的輸出數值應該都要一致。

## 三、結果驗證模組新增與改寫

- 新增 `compare.py`：用以比較 Keras 與 HLS 輸出結果的差異。
- 修改 `run_all.py` 以及 `test.cpp`：整合測試流程，方便自動比對模型結果。

## 四、專案流程優化

- 為加快 CSIM 測試流程，已將 `generate_build_prj.py` 中的 `csynth_design`, `cosim_design`, `export_design` 註解，使流程只進行 CSIM 階段，節省時間以專注輸出比對。

## 五、激活函數的修正與測試

- `gelu` 與 `softmax` 的實作有所修正，目前仍在觀察結果準確性，待後續進一步驗證。

## 六、數值精度設定測試

- 嘗試調整 `ap_fixed` 精度（在 `read_model.py` 中撰寫 `common.h` 的段落 `typedef ap_fixed<16,6>`），但目前未觀察到顯著差異，推測問題可能非來自精度設定。

## 七、MNIST 測試模型輸出比較結果

以下為 `train_test.py`（MNIST）產生的 Softmax 輸出與 HLS 的比較結果：
```
Softmax output comparison:
[0] Keras: 0.000003 | HLS: 0.000000 | Δ = 0.000003
[1] Keras: 0.000001 | HLS: 0.108398 | Δ = 0.108397
[2] Keras: 0.000303 | HLS: 0.112305 | Δ = 0.112002
[3] Keras: 0.000287 | HLS: 0.106445 | Δ = 0.106158
[4] Keras: 0.000000 | HLS: 0.112305 | Δ = 0.112305
[5] Keras: 0.000000 | HLS: 0.114258 | Δ = 0.114258
[6] Keras: 0.000000 | HLS: 0.116211 | Δ = 0.116211
[7] Keras: 0.999388 | HLS: 0.108398 | Δ = 0.890990
[8] Keras: 0.000002 | HLS: 0.108398 | Δ = 0.108396
[9] Keras: 0.000017 | HLS: 0.108398 | Δ = 0.108381

Keras predicted class: 7
HLS predicted class:   6
Ground Truth Label:    7
Match status:  Mismatch with Keras
Prediction correctness:  HLS wrong
```
