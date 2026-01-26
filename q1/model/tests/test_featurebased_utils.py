import tensorflow as tf
import os 
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(project_root)
from featurebased import make_valid_mask, masked_softmax


def test_make_valid_mask_without_mode():
    n = 5
    lengths = tf.constant([3, 5], dtype=tf.int32)  # batch of 2
    mask = make_valid_mask(n, lengths)  # (2, 5)

    # first row: first 3 True, rest False
    expected0 = [True, True, True, False, False]
    # second row: all 5 valid
    expected1 = [True, True, True, True, True]

    assert mask.shape == (2, 5)
    assert (mask[0].numpy() == expected0).all()
    assert (mask[1].numpy() == expected1).all()


def test_masked_softmax_basic():
    scores = tf.constant([[1.0, 2.0, 3.0, 4.0],
                          [1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
    lengths = tf.constant([2, 4], dtype=tf.int32)

    log_probs = masked_softmax(scores, lengths)   # (2, 4)
    probs = tf.exp(log_probs)

    # for first row, only first 2 positions valid
    row0 = probs[0].numpy()
    assert abs(row0[0] + row0[1] - 1.0) < 1e-5
    assert row0[2] < 1e-6 and row0[3] < 1e-6

    # for second row, all 4 valid
    row1 = probs[1].numpy()
    assert abs(row1.sum() - 1.0) < 1e-5
