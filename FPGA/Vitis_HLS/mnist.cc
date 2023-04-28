#include "mnist.h"

void mnist_dnn(hls::stream<stream_data> &in, hls::stream<stream_data> &out)
{
#pragma HLS INTERFACE mode=axis port=in
#pragma HLS INTERFACE mode=axis port=out
#pragma HLS INTERFACE mode=s_axilite port=return

	stream_data tmp;
	int32_t input_layer[MATRIX_UNFOLD];
	int32_t hidden_layer[HIDDEN_LAYER_LENGTH];
	int32_t result_layer[RESULT_VECTOR_LENGTH];

#pragma HLS ARRAY_PARTITION variable=input_layer dim=1 type=block factor=MATRIX_DIM
#pragma HLS ARRAY_PARTITION variable=hidden_layer type=complete
#pragma HLS ARRAY_PARTITION variable=result_layer type=complete

	int copy_index = 0;
	while(1)
	{
		in.read(tmp);

		// Copy into a temporary buffer matrix
		input_layer[copy_index] = (int32_t) tmp.data.to_int();
		copy_index++;

		if (tmp.last) break;
	}

	// Perform the NN calculations: Step 1 = Input layer
	int index = 0;
	for (int i = 0; i < HIDDEN_LAYER_LENGTH; i++)
	{
		hidden_layer[i] = 0;
#pragma HLS unroll factor=4
		for (int j = 0; j < MATRIX_UNFOLD; j++)
		{
			hidden_layer[i] += ((int32_t) input_layer[j]) * ((int32_t) weights_0[index]);
			index++;
		}
	}

	// Add biases
	for (int i = 0; i < HIDDEN_LAYER_LENGTH; i++)
	{
#pragma HLS unroll factor=4
		hidden_layer[i] = calculate_relu((hidden_layer[i] + biases_0[i]));
	}


	// Step 2 = Result layer
	index = 0;
	for (int i = 0; i < RESULT_VECTOR_LENGTH; i++)
	{
#pragma HLS pipeline off
#pragma HLS unroll factor=4
		result_layer[i] = 0;
		for (int j = 0; j < HIDDEN_LAYER_LENGTH; j++)
		{
			result_layer[i] += hidden_layer[j] * ((int32_t) weights_1[index]);
			index++;
		}
	}

	// Add biases
	for (int i = 0; i < RESULT_VECTOR_LENGTH; i++)
	{
#pragma HLS unroll factor=4
		result_layer[i] += biases_1[i];
	}


	// DMA back the result vector
	tmp.last = false;
	for (int entry = 0; entry < RESULT_VECTOR_LENGTH; entry++)
	{
		if (entry == (RESULT_VECTOR_LENGTH - 1))
		{
			tmp.last = true;
		}
		tmp.data = result_layer[entry];
		out.write(tmp);
	}

}

inline int32_t calculate_relu(int32_t val)
{
	if (val < 0)
	{
		return 0;
	}
	else
	{
		return val;
	}
}
