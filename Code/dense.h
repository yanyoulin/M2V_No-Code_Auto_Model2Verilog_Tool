#ifndef DENSE_H
#define DENSE_H

#include "common.h"

template<int IN_DIM, int OUT_DIM>
void dense(
    data_t input[IN_DIM],
    const data_t weight[OUT_DIM * IN_DIM],
    const data_t bias[OUT_DIM],
    data_t output[OUT_DIM]
) {
#pragma HLS array_partition variable=input complete
#pragma HLS array_partition variable=output complete
#pragma HLS array_partition variable=weight complete
#pragma HLS array_partition variable=bias complete
#pragma HLS PIPELINE II=1

    for (int i = 0; i < OUT_DIM; i++) {
#pragma HLS UNROLL
        data_t acc = bias[i];
        for (int j = 0; j < IN_DIM; j++) {
#pragma HLS UNROLL
            acc += weight[i * IN_DIM + j] * input[j];
        }
        output[i] = acc;
    }
}

#endif
