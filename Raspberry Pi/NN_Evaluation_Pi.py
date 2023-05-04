import warnings
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import threading
import os
# import wmi
import time
import psutil
import resource
from queue import Queue
#import pythoncom

# Ignore TF warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
tf.get_logger().setLevel('ERROR')
print("Tensorflow Version: " + tf.version.VERSION)

# Prevent memory fragmentation on the GPU. Although the system has one, this could be scaled to multi-GPU systems
# This is not a problem for the CPU because it has 64GB of memory on the target system
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", str(len(gpus)))

"""
Load the necessary data
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
combined_x = np.concatenate((x_train, x_test))
combined_y = np.concatenate((y_train, y_test))
model = tf.keras.models.load_model('mnist.h5')
batch=64

# Helper function
def within_std(arr, val):
    std = np.std(arr)
    count = 0
    for num in arr:
        if (val < num < (val + std)) or ((val - std) < num < val):
            count = count + 1
    return count


"""
Accuracy test
"""


def calculate_accuracy():
    scores = model.evaluate(x_test, y_test, verbose=2)
    return scores[1] * 100


print("\nBeginning accuracy test.")
accuracy_score = []
for i in range(10):
    accuracy_score.append(calculate_accuracy())
average_accuracy = sum(accuracy_score) / len(accuracy_score)
print("Number of accuracy samples collected: {}".format(len(accuracy_score)))
print("Average accuracy: {} %".format(average_accuracy))
print("Maximum accuracy: {} %".format(np.max(accuracy_score)))
print("Standard deviation of accuracy: {} %".format(np.std(accuracy_score)))
print("Number of samples within 1 standard deviation of the mean accuracy: {}".format(within_std(accuracy_score, average_accuracy)))
print("Number of entries within 1 standard deviation of the maximum accuracy: {}".format(within_std(accuracy_score, np.max(accuracy_score))))


"""
Performance test
"""


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


total_flops = get_flops(model)
print("The FLOPs is:{}".format(total_flops), flush=True)


def calculate_performance():
    p_start = time.time_ns()
    model.predict(combined_x, batch_size=batch)
    p_duration = time.time_ns() - p_start
    performance = ((total_flops * len(combined_x)) / (p_duration / 10**9)) / 10**9
    return performance


print("\nBeginning performance test.")
# set maximum memory limit to 4GB
# resource.setrlimit(resource.RLIMIT_DATA, (4 * 1024 * 1024 * 1024, resource.RLIM_INFINITY))

performance_score = []
for i in range(100):
    performance_score.append(calculate_performance())
average_performance_score = sum(performance_score) / len(performance_score)
print("Number of performance samples collected: {}".format(len(performance_score)))
print("Average performance: {} GFLOPS".format(average_performance_score))
print("Maximum performance: {} GFLOPS".format(np.max(performance_score)))
print("Standard deviation of performance: {} GFLOPS".format(np.std(performance_score)))
print("Number of samples within 1 standard deviation of the mean performance: {}".format(within_std(performance_score, average_performance_score)))
print("Number of entries within 1 standard deviation of the maximum performance: {}".format(within_std(performance_score, np.max(performance_score))))


"""
Power efficiency test
"""

def get_cpu_package_power():
    return psutil.sensors_temperatures()['cpu_thermal'][0].current

def get_gpu_power():
    power_command = 'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits'
    power_output = os.popen(power_command).read().strip()
    return float(power_output)


def get_power(device_type):
    if device_type == 'CPU':
        cpu_power = get_cpu_package_power()
        if cpu_power is not None:
            return cpu_power
        else:
            raise Exception("OHM not running!")
    else:
        return get_gpu_power()


def measure_power(device, power_queue, power_data):
    while not power_queue.empty():
        device_type = device.device_type.upper()
        power = get_power(device_type)
        power_data.append((device_type, power))
        time.sleep(0.01)

# Get performance device and create large dataset for testing
devices = tf.config.list_physical_devices()
selected_device = devices[0]

# Create thread management variables
nominal_queue = Queue()
nominal_queue.put(1)
nominal_data = []
power_queue = Queue()
power_queue.put(1)
power_data = []

# Create the threads
nominal_thread = threading.Thread(target=measure_power, args=(selected_device, nominal_queue, nominal_data))
power_thread = threading.Thread(target=measure_power, args=(selected_device, power_queue, power_data))

# Get nominal power draw data
print("Gathering nominal power draw")
x = input("Press enter to start nominal power draw test")
nominal_thread.start()
time.sleep(60)
nominal_queue.get()
nominal_thread.join()

# Get DNN power draw data
print("Gathering DNN power draw")
x = input("Press enter to start DNN power draw test")
start_time = time.time()
power_thread.start()
while time.time() - start_time < 60:
    model.predict(combined_x, batch_size=batch)
power_queue.get()
power_thread.join()
print("Performance test ended.\n")

# Analyze performance results
nominal_power = [p[1] for p in nominal_data]
power = [p[1] for p in power_data]
nominal_average_power = sum(nominal_power) / len(nominal_power)
average_power = sum(power) / len(power)
print("Number of nominal power draw samples collected: {}".format(len(nominal_power)))
print("Average nominal power draw: {} W".format(nominal_average_power))
print("Maximum nominal power draw: {} W".format(np.max(nominal_power)))
print("Standard deviation of nominal power draw: {} W".format(np.std(nominal_power)))
print("Number of samples within 1 standard deviation of the mean nominal power draw: {}".format(within_std(nominal_power, nominal_average_power)))
print("Number of entries within 1 standard deviation of the maximum nominal power draw: {}".format(within_std(nominal_power, np.max(nominal_power))))
print("Number of DNN power draw samples collected: {}".format(len(power)))
print("Average DNN power draw: {} W".format(average_power))
print("Maximum DNN power draw: {} W".format(np.max(power)))
print("Standard deviation of DNN power draw: {} W".format(np.std(power)))
print("Number of samples within 1 standard deviation of the mean DNN power draw: {}".format(within_std(power, average_power)))
print("Number of entries within 1 standard deviation of the maximum DNN power draw: {}".format(within_std(power, np.max(power))))

