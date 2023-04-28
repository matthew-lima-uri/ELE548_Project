import numpy as np

hidden_weights = np.load('hidden_layer_weights.npy')
hidden_biases = np.load('hidden_layer_biases.npy')
output_weights = np.load('output_layer_weights.npy')
output_biases = np.load('output_layer_biases.npy')


def numpy_to_c_array(name, np_array, dtype="int8_t"):
    c_array = f"const {dtype} {name}[] {{"
    for i, elem in enumerate(np_array.flatten()):
        if i % 10 == 0:
            c_array += "\n    "
        c_array += f"{int(elem)}, "
    c_array = c_array.rstrip(", ") + "};\n"
    c_array += f"const unsigned int {name}_len = {np_array.size};\n"
    return c_array


hidden_weights_c = numpy_to_c_array("weights_0", hidden_weights)
hidden_biases_c = numpy_to_c_array("biases_0", hidden_biases, dtype="int32_t")
output_weights_c = numpy_to_c_array("weights_1", output_weights)
output_biases_c = numpy_to_c_array("biases_1", output_biases, dtype="int32_t")

with open("arrays.h", "w") as f:
    f.write(hidden_weights_c)
    f.write(hidden_biases_c)
    f.write(output_weights_c)
    f.write(output_biases_c)
