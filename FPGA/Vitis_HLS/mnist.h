#ifndef MNIST_H
#define MNIST_H

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "dnn_params.h"
#include "ap_fixed.h"
#include <stdint.h>
#include <limits.h>

typedef ap_axis<32,2,5,6> stream_data;

#define MATRIX_DIM	28
#define MATRIX_UNFOLD	MATRIX_DIM * MATRIX_DIM
#define RESULT_VECTOR_LENGTH 10
#define HIDDEN_LAYER_LENGTH 64

#define HIDDEN_BIAS_START	MATRIX_UNFOLD * HIDDEN_LAYER_LENGTH
#define RESULT_WEIGHT_START HIDDEN_BIAS_START + HIDDEN_LAYER_LENGTH
#define RESULT_BIAS_START	RESULT_WEIGHT_START + (HIDDEN_LAYER_LENGTH * RESULT_VECTOR_LENGTH)

int32_t calculate_relu(int32_t val);

#endif
