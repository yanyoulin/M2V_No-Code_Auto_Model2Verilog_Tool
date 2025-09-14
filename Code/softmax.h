#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "common.h"
#include <hls_math.h>

template<int LEN>
void softmax(data_t input[LEN], data_t output[LEN]) {
#pragma HLS INLINE off

    data_t max_val = input[0];
    for (int i = 1; i < LEN; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    data_t exp_sum = 0;
    data_t exp_val[LEN];

    for (int i = 0; i < LEN; i++) {
#pragma HLS UNROLL
        exp_val[i] = hls::exp(input[i] - max_val);
        exp_sum += exp_val[i];
    }

    for (int i = 0; i < LEN; i++) {
#pragma HLS UNROLL
        output[i] = exp_val[i] / exp_sum;
    }
}

#endif
