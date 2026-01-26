import tensorflow as tf
from tensorflow.keras import layers, Model

def make_valid_mask(n, lengths, mode='without'):
    """
    lengths: int32/int64 tensor of shape (B,)
    returns: bool mask of shape (B, n)
    """
    # row = [0,1,...,n-1] shaped (1, n)
    row = tf.range(n)[tf.newaxis, :] # (1, n)
    lengths = lengths[:, tf.newaxis] # (B, 1)

    if mode == 'with':
        # (row < lengths) OR (row == n-1)
        return tf.logical_or(row < lengths,
                             tf.equal(row, n - 1))
    else:
        return row < lengths


def masked_softmax(scores, lengths):
    """
    scores: (B, n)
    lengths: (B,)
    returns: log-softmax over valid positions -> (B, n)
    """
    n = tf.shape(scores)[1]
    valid = make_valid_mask(n, lengths)                   # (B, n)

    very_neg = tf.constant(-1e9, dtype=scores.dtype)
    masked_scores = tf.where(valid, scores, very_neg)

    # log-softmax along last axis
    log_probs = tf.nn.log_softmax(masked_scores, axis=-1)
    return log_probs


class NonlinearTransformation(layers.Layer):
    def __init__(self, H, embed=128, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.H = H
        self.embed = embed
        self.fc1 = layers.Dense(embed * H)
        self.fc2 = layers.Dense(embed)
        self.dropout = layers.Dropout(dropout)
        self.enc_norm = layers.LayerNormalization(axis=-1)

    def call(self, X, training=False):
        """
        X: (B, n, embed)
        returns: (B, n, H, embed)
        """
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]

        # (B, n, embed*H) -> (B, n, H, embed)
        X = self.fc1(X)
        X = tf.reshape(X, (B, n, self.H, self.embed))

        X = tf.nn.relu(X)
        X = self.dropout(X, training=training)

        # Dense acts on last dim, broadcasting over (B, n, H)
        X = self.fc2(X)
        X = self.enc_norm(X)

        return X  # (B, n, H, embed)


class MainNetwork(Model):
    def __init__(self, n, input_dim, H, L, embed=128, dropout=0.0, **kwargs):
        """
        n: max sequence length (only needed to build masks in call)
        input_dim: feature dimension of input
        H: number of heads
        L: number of layers
        embed: embedding dimension
        """
        super().__init__(**kwargs)
        self.n = n
        self.H = H
        self.L = L
        self.embed = embed

        # Basic encoder: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        self.basic_encoder = tf.keras.Sequential([
            layers.Dense(embed),
            layers.ReLU(),
            layers.Dropout(dropout),

            layers.Dense(embed),
            layers.ReLU(),
            layers.Dropout(dropout),

            layers.Dense(embed),
        ])

        self.enc_norm = layers.LayerNormalization(axis=-1)

        # L separate aggregate_linear layers (Dense(embed -> H))
        self.aggregate_linear = [
            layers.Dense(H) for _ in range(L)
        ]

        # L nonlinear blocks
        self.nonlinear_blocks = [
            NonlinearTransformation(H, embed, dropout) for _ in range(L)
        ]

        # Final linear to scalar per position
        self.final_linear = layers.Dense(1)


    def call(self, X, lengths, training=False):
        """
        X: (B, n, input_dim)
        lengths: (B,) valid length for each sequence
        returns: log_probs (B, n), logits (B, n)
        """
        B = tf.shape(X)[0]
        n = tf.shape(X)[1]

        # Basic encoder + layer norm
        Z = self.basic_encoder(X, training=training) # (B, n, embed)
        Z = self.enc_norm(Z)
        X_work = tf.identity(Z)

        lengths_f = tf.cast(lengths, Z.dtype)  # for division

        for fc, nt in zip(self.aggregate_linear, self.nonlinear_blocks):
            # Aggregate linear: (B, n, embed) -> (B, n, H)
            A = fc(Z) # (B, n, H)

            # mean over valid positions
            # sum over axis=1, then divide by lengths
            sum_A = tf.reduce_sum(A, axis=1) # (B, H)
            mu = sum_A / tf.expand_dims(lengths_f, -1) # (B, H)

            # reshape to (B, 1, H, 1) for broadcasting
            Z_bar = mu[:, tf.newaxis, :, tf.newaxis] # (B, 1, H, 1)

            # Nonlinear transform: (B, n, embed) -> (B, n, H, embed)
            phi = nt(X_work, training=training)  # (B, n, H, embed)

            # Mask invalid positions
            valid = make_valid_mask(n, lengths) # (B, n)
            valid = tf.cast(valid, phi.dtype)
            phi = phi * valid[:, :, tf.newaxis, tf.newaxis]

            # (phi * Z_bar).sum over heads, then / H and residual-add to Z
            interaction = tf.reduce_sum(phi * Z_bar, axis=2) / self.H  # (B, n, embed)
            Z = Z + interaction

        # Final logits
        logits = self.final_linear(Z) # (B, n, 1)
        logits = tf.squeeze(logits, axis=-1) # (B, n)

        log_probs = masked_softmax(logits, lengths) # (B, n)
        return log_probs, logits



# Example dummy run
if __name__ == "__main__":
    B = 3
    n = 10
    d = 5
    H = 4
    L = 2
    embed = 16

    model = MainNetwork(n=n, input_dim=d, H=H, L=L, embed=embed, dropout=0.1)

    X = tf.random.normal((B, n, d))
    lengths = tf.constant([10, 7, 4], dtype=tf.int32)

    log_probs, logits = model(X, lengths, training=False)
    print(log_probs.shape, logits.shape)   # (3, 10) (3, 10)
