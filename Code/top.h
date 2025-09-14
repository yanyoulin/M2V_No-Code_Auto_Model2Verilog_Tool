#ifndef TOP_H
#define TOP_H

#include "common.h"
#include "dense.h"
#include "gelu.h"
#include "relu.h"
#include "weights.h"
#include "softmax.h"

void mlp_inference(data_t input[DIM], data_t output[FF_DIM]);

#endif
