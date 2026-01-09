import tensorflow as tf
import timeit
import os

tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

def generate_matrix(shape, device, dtype=tf.float32):
    with tf.device(device):
        return tf.random.normal(shape, dtype=dtype)

def benchmark(device, num_runs=500):
    # 预热所有设备
    _ = generate_matrix([1000, 1000], device)
    
    def run():
        a = generate_matrix([10000, 1000], device)
        b = generate_matrix([1000, 2000], device)
        return tf.reduce_sum(tf.matmul(a, b)).numpy()
    
    return timeit.timeit(run, number=num_runs)

if __name__ == "__main__":
    devices = []
    if tf.config.list_physical_devices('GPU'):
        devices.append('/gpu:0')
    devices.append('/cpu:0')  

    for device in devices:
        time_taken = benchmark(device, num_runs=500)
        print(f"{device.upper()} Time (500 runs): {time_taken:.4f} seconds")