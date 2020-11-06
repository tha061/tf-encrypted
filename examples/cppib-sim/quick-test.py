import logging
import sys

import absl
import silence_tensorflow.auto
import tensorflow as tf

import tf_encrypted as tfe

if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
else:
    config = tfe.LocalConfig([
        'cpp-inputter',
        'cpp-server',
        'cts-inputter',
        'cts-server',
        'cape-triple-provider',
    ])

tfe.set_config(config)
tfe.set_protocol(
    tfe.protocol.Pond(
        server_0=config.get_player('cpp-server'),
        server_1=config.get_player('cts-server'),
        triple_source=config.get_player('cape-triple-provider'),
    )
)

def build_cpp_inputter(num_features, num_rows):
    @tfe.local_computation('cpp-inputter')
    def cpp_inputter():
        return tf.random.uniform(
            minval=-0.5, maxval=0.5, shape=[num_features, num_rows], dtype=tf.float32,
        )  # this is the (p, n) matrix (B from slides)
    return cpp_inputter

def build_cts_inputter(num_rows, num_outputs):
    @tfe.local_computation('cts-inputter')
    def cts_inputter():
        return tf.random.uniform(
            minval=-0.5, maxval=0.5, shape=[num_rows, num_outputs], dtype=tf.float32,
        )
    return cts_inputter

@tfe.local_computation('cpp-inputter')
def cpp_receiver(matmul_result):
    # print(matmul_result.shape)
    return tf.print(matmul_result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    NUM_ROWS = 10000
    NUM_FEATURES = 10
    NUM_OUTPUTS = 1

    cpp_inputter = build_cpp_inputter(NUM_FEATURES, NUM_ROWS)
    cts_inputter = build_cts_inputter(NUM_ROWS, NUM_OUTPUTS)

    cpp_input = cpp_inputter()
    cts_input = cts_inputter()
    matrix_prod = tfe.matmul(cpp_input, cts_input)
    result_op = cpp_receiver(matrix_prod)
    
    with tfe.Session() as sess:
        sess.run(result_op, tag="quick-test")
