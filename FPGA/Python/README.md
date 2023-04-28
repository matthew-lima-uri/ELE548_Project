# ELE-548 Final Project: FPGA Python implementation

FPGA Evaluator.ipynb: Jupyter Notebook for evaluating the FPGA HW and SW implementations

MNIST DNN Accelerator.ipynb: Initial testing environment for the FPGA HW implementation. Performs inference on a single image and displays both the image and the prediction

mnist_driver.py: Python driver to interface with the DNN accelerator hardware

mnist_images.npy: Numpy array of the MNIST dataset stored as uint8

mnist_results.npy: Numpy array of the MNIST dataset expected outputs stored as uint8

hidden_layer_weights.npy: Numpy array of the weights of the hidden layer of the trained DNN model stored as int8. Used to evaluate the software implementation of the DNN

hidden_layer_biases.npy: Numpy array of the biases of the hidden layer of the trained DNN model stored as int32. Used to evaluate the software implementation of the DNN

output_layer_weights.npy: Numpy array of the weights of the output layer of the trained DNN model stored as int8. Used to evaluate the software implementation of the DNN

output_layer_biases.npy: Numpy array of the biases of the output layer of the trained DNN model stored as int32. Used to evaluate the software implementation of the DNN