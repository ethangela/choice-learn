import tensorflow as tf
import os 
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(project_root)
from featureless import ExaResBlock, QuaResBlock, MainNetwork


def test_exa_res_block_shape(batch_binary_input):
    B, opt_size = batch_binary_input.shape
    hidden_dim = 8

    block = ExaResBlock(input_dim=opt_size, hidden_dim=hidden_dim)
    z_prev = tf.random.normal((B, hidden_dim))
    e0 = batch_binary_input

    out = block((z_prev, e0))

    assert out.shape == (B, hidden_dim)
    assert not tf.reduce_any(tf.math.is_nan(out))


def test_qua_res_block_shape(batch_binary_input):
    B, d = batch_binary_input.shape

    block = QuaResBlock(d)
    x = tf.random.normal((B, d))

    out = block(x)

    assert out.shape == (B, d)
    assert not tf.reduce_any(tf.math.is_nan(out))


def test_main_network_featureless_forward(batch_binary_input):
    opt_size = batch_binary_input.shape[1]
    depth = 4
    resnet_width = 16
    block_types = ["exa", "qua", "exa"]  # length = depth-1

    model = MainNetwork(
        opt_size=opt_size,
        depth=depth,
        resnet_width=resnet_width,
        block_types=block_types,
    )

    probs, logits = model(batch_binary_input)

    # shape checks
    assert probs.shape == (batch_binary_input.shape[0], opt_size)
    assert logits.shape == (batch_binary_input.shape[0], opt_size)

    # rows of probs should sum ~1
    row_sums = tf.reduce_sum(probs, axis=-1)
    assert tf.reduce_all(tf.abs(row_sums - 1.0) < 1e-5)

    # mask logic: positions where e == 0 should have negligible probability
    mask = tf.equal(batch_binary_input, 1.0)
    probs_np = probs.numpy()
    # probabilities at masked-out positions should be very small
    assert (probs_np[~mask.numpy()] < 1e-3).all()

    # no NaNs
    assert not tf.reduce_any(tf.math.is_nan(probs))
    assert not tf.reduce_any(tf.math.is_nan(logits))
