import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import random
from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.python.framework import convert_to_constants as _convert_to_constants

"""
Load the MNIST data
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 60,000 training data and 10,000 test data of 28x28 pixel images
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
fig = plt.figure(figsize=(9,9))
for i in range(9):
  plt.subplot(3,3,i+1)
  num = random.randint(0, len(x_train))
  plt.imshow(x_train[num], cmap="gray", interpolation=None)
  plt.title(y_train[num])
plt.tight_layout()


# Create a model with a 28x28 pixel input vector
#    -> 1 hidden layer of 64 nodes
#    -> 10 categories of outputs (digits 0-9)
def create_model():
    created_model = Sequential()
    created_model.add(Flatten(input_shape=(28 * 28,), dtype=tf.uint8))
    created_model.add(Dense(64, activation="relu", use_bias=True))
    created_model.add(Dense(10, activation="softmax", use_bias=True))
    created_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return created_model


model = create_model()
model.summary()

"""
Setup training and test data, then train the model
"""
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
x_train = x_train.astype("uint8")
x_test = x_test.astype("uint8")
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
# Train and save the model
model.fit(x=x_train, y=y_train, batch_size=256, epochs=25, verbose=1)
scores = model.evaluate(x_test, y_test, verbose=2)
print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])


tf.keras.backend.set_learning_phase(False)
func = _saving_utils.trace_model_call(model)
concrete_func = func.get_concrete_function()
frozen_func = _convert_to_constants.convert_variables_to_constants_v2(concrete_func)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

model.save("mnist.h5")
