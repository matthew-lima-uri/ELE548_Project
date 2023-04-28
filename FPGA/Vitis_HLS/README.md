# ELE-548 Final Project: FPGA Python implementation

mnist.cc: Main logic code for the FPGA implementation. mnist_dnn() function interfaces with Vivado to take image data as an input stream and output the result vector as an output stream.

mnist.h: Header file for mnist.cc. Contains simple #defines, #includes, and one typedef for the stream data type.

dnn_params.h: DNN model parameters stored as const arrays. Copied from the Windows Implementation arrays.h file and re-named for clarity.

MNIST_HLS.zip: Zipped project directory for the entire HLS project. NOTE: DOES NOT CONTAIN THE ABOVE FILES; THESE NEED TO BE IMPORTED INTO THE PROJECT.