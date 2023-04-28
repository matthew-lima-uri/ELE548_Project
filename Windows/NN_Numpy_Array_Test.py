import numpy as np
import tensorflow as tf
import time

# Load the input data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)
x_train = x_train.astype("uint8")
x_test = x_test.astype("uint8")
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
combined_x = np.concatenate((x_train, x_test))
combined_y = np.concatenate((y_train, y_test))
with open('mnist_results.npy', 'rb') as f:
    results = np.load(f)

# Load the weights and biases
hidden_weights = np.load('hidden_layer_weights.npy')
hidden_biases = np.load('hidden_layer_biases.npy')
output_weights = np.load('output_layer_weights.npy')
output_biases = np.load('output_layer_biases.npy')

def relu(vec):
    return np.maximum(vec, 0)



def inference(image):

    # Flatten the input
    image = image.flatten()

    # Calculate the hidden layer output
    hidden_layer = np.matmul(image.astype(np.int32), hidden_weights.T.astype(np.int32)) + hidden_biases
    hidden_layer = relu(hidden_layer)

    # Calculate the output layer
    output_layer = np.matmul(hidden_layer.astype(np.int32), output_weights.T.astype(np.int32)) + output_biases

    return np.argmax(output_layer)


def calculate_quantized(images):
    total_correct_quantized = 0
    total_incorrect_quantized = 0
    q_start = time.time()
    for i, img in enumerate(images):
        dnn_guess_quantized = inference(img)
        if dnn_guess_quantized == results[i]:
            total_correct_quantized = total_correct_quantized + 1
        else:
            total_incorrect_quantized = total_incorrect_quantized + 1
        if i % 10000 == 0:
            q_end = time.time()
            print(str(i) + " quantized images processed!")
            print("Quantized time taken: " + str(q_end - q_start))
            q_start = q_end
    return total_correct_quantized, total_incorrect_quantized



total_correct_quantized, total_incorrect_quantized = calculate_quantized(images=combined_x)
print("Accuracy of the quantized model: " + str((total_correct_quantized / len(combined_x)) * 100) + "%")
print("Total mis-predicted of the quantized model: " + str(total_incorrect_quantized))
