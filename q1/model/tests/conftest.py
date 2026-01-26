import pytest
import tensorflow as tf


@pytest.fixture
def batch_binary_input():
    # For featureless.MainNetwork (B, opt_size)
    return tf.constant([
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1],
    ], dtype=tf.float32)


@pytest.fixture
def sequence_input():
    # For featurebased.MainNetwork (B, n, d)
    B, n, d = 3, 10, 5
    X = tf.random.normal((B, n, d))
    lengths = tf.constant([10, 7, 4], dtype=tf.int32)  # valid lengths
    return X, lengths
