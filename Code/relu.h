#ifndef RELU_H
#define RELU_H

#include "common.h"

template<int LEN>
void relu(data_t input[LEN], data_t output[LEN]) {
#pragma HLS array_partition variable=input complete
#pragma HLS array_partition variable=output complete

    for (int i = 0; i < LEN; i++) {
#pragma HLS UNROLL
        output[i] = (input[i] > 0) ? input[i] : data_t(0);
    }
}

#endif
