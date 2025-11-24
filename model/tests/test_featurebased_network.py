import tensorflow as tf
import os 
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(project_root)
from featurebased import NonlinearTransformation, MainNetwork, make_valid_mask


def test_nonlinear_transformation_shape(sequence_input):
    X, _ = sequence_input
    B, n, embed_in = X.shape
    H = 4
    embed = 16

    layer = NonlinearTransformation(H=H, embed=embed, dropout=0.1)
    # but the layer assumes input last dim == `embed`, so we project first:
    proj = tf.keras.layers.Dense(embed)
    X_embed = proj(X)

    out = layer(X_embed, training=False)

    assert out.shape == (B, n, H, embed)
    assert not tf.reduce_any(tf.math.is_nan(out))


def test_main_network_featurebased_forward(sequence_input):
    X, lengths = sequence_input
    B, n, d = X.shape
    H = 4
    L = 2
    embed = 16

    model = MainNetwork(
        n=n,
        input_dim=d,
        H=H,
        L=L,
        embed=embed,
        dropout=0.1,
    )

    log_probs, logits = model(X, lengths, training=False)

    # shapes
    assert log_probs.shape == (B, n)
    assert logits.shape == (B, n)

    # exp(log_probs) should sum ~1 over valid positions for each row
    probs = tf.exp(log_probs)
    valid = make_valid_mask(n, lengths)  # (B, n)
    probs_valid_sum = tf.reduce_sum(
        tf.where(valid, probs, 0.0), axis=-1
    )
    assert tf.reduce_all(tf.abs(probs_valid_sum - 1.0) < 1e-5)

    # invalid positions should have near-zero probability
    probs_np = probs.numpy()
    valid_np = valid.numpy()
    assert (probs_np[~valid_np] < 1e-6).all()

    # no NaNs
    assert not tf.reduce_any(tf.math.is_nan(log_probs))
    assert not tf.reduce_any(tf.math.is_nan(logits))
