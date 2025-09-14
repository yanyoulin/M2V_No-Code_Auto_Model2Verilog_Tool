#include "top.h"
#include <iostream>
#include <fstream>
#include <cmath>

int main() {
    data_t input[DIM];
    data_t output[FF_DIM];

    std::ifstream fin("../../../../example_input.txt");
    if (!fin.is_open()) {
        std::cerr << "Failed to open input file: example_input.txt\n";
        return 1;
    }

    // 輸入資料
    for (int i = 0; i < DIM; ++i) {
        float val;
        fin >> val;
        input[i] = static_cast<data_t>(val);
    }
    fin.close();

    // 執行推論
    mlp_inference(input, output);

    // 輸出結果到 output.txt
    std::ofstream fout("output.txt");
    if (!fout.is_open()) {
        std::cerr << "Failed to open output file: output.txt\n";
        return 1;
    }

    for (int i = 0; i < FF_DIM; ++i) {
        fout << output[i].to_double() << "\n";
    }
    fout.close();

    std::cout << "Inference finished. Output written to output.txt\n";
    return 0;
}
