#ifndef GELU_H
#define GELU_H

#include "common.h"
#include <hls_math.h>

inline data_t gelu(data_t x) {
    const data_t a = data_t(0.044715);
    const data_t sqrt_2_over_pi = data_t(0.7978845608);
    data_t x3 = x * x * x;
    data_t tanh_input = sqrt_2_over_pi * (x + a * x3);
    data_t tanh_out = hls::tanh(tanh_input);
    return data_t(0.5) * x * (1 + tanh_out);
}

#endif
