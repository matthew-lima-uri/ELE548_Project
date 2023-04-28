import tensorflow as tf

# Load and preprocess dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("uint8")
x_test = x_test.astype("uint8")

# Load the float32 model
model = tf.keras.models.load_model('mnist.h5')

# Create a generator function for the representative dataset
def representative_dataset_gen():
    for data in tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1).take(100):
        yield [data[0]]


# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
tflite_quant_model = converter.convert()

# Save the quantized model
with open('mnist_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

