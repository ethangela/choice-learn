import tensorflow as tf


'''Residual blocks'''
class ExaResBlock(tf.keras.layers.Layer):
    """
        z_next = W_main( z_prev * (W_act e0) ) + z_prev
    """
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.linear_main = tf.keras.layers.Dense(
            hidden_dim, use_bias=False
        )
        self.linear_act = tf.keras.layers.Dense(
            hidden_dim, use_bias=False, input_shape=(input_dim,)
        )

    def call(self, inputs):
        z_prev, e0 = inputs
        a = self.linear_act(e0)          
        m = z_prev * a               
        u = self.linear_main(m)        
        return u + z_prev     


class QuaResBlock(tf.keras.layers.Layer):
    """
        z_next = W_qua(z^2) + z
    """
    def __init__(self, d, **kwargs):
        super().__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(
            d, use_bias=False, input_shape=(d,)
        )

    def call(self, x):
        u = self.linear(tf.math.square(x))
        return u + x


'''Main network'''
class MainNetwork(tf.keras.Model):
    def __init__(self, opt_size: int, depth: int, resnet_width: int, block_types, **kwargs):
        """
        opt_size = input dimension 
        depth = total # of layers including input/output
        resnet_width = hidden dimension (same as resnet_width)
        block_types = list of length depth-1, entries: "exa" or "qua"
        """
        super().__init__(**kwargs)
        assert len(block_types) == depth - 1

        self.opt_size = opt_size
        self.resnet_width = resnet_width

        # input / output linear layers (bias=False)
        self.in_lin = tf.keras.layers.Dense(
            resnet_width, use_bias=False, input_shape=(opt_size,)
        )
        self.out_lin = tf.keras.layers.Dense(
            opt_size, use_bias=False, input_shape=(resnet_width,)
        )

        # build residual blocks
        self.blocks = []
        for t in block_types:
            if t == "exa":
                self.blocks.append(ExaResBlock(opt_size, resnet_width))
            elif t == "qua":
                self.blocks.append(QuaResBlock(resnet_width))
            else:
                raise ValueError(f"Unknown block type {t}")

    def call(self, e, training=False):
        """
        e: (B, opt_size)

        returns:
            probs : (B, opt_size)
            logits: (B, opt_size)
        """
        # mask
        mask = tf.equal(e, tf.cast(1, e.dtype)) # bool, (B, opt_size)
        e0 = tf.identity(e) # original input
        z = self.in_lin(e0) # (B, resnet_width)

        # Pass through blocks
        for b in self.blocks:
            if isinstance(b, ExaResBlock):
                z = b((z, e0))
            else:  # QuaResBlock
                z = b(z)

        logits = self.out_lin(z) # (B, opt_size)
        neg_large = tf.constant(-1e9, dtype=logits.dtype)
        masked_logits = tf.where(mask, logits, neg_large)

        probs = tf.nn.softmax(masked_logits, axis=-1)

        return probs, masked_logits
