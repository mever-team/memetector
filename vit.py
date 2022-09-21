import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({"patch_size": self.patch_size})
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.class_token = tf.Variable(
            initial_value=tf.zeros_initializer()(
                shape=(1, 1, projection_dim), dtype="float32"
            ),
            trainable=True,
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({"projection_dim": self.projection_dim})
        return config

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.class_token, [batch_size, 1, self.projection_dim]),
            dtype=patch.dtype,
        )
        return tf.concat([cls_broadcasted, self.projection(patch)], 1)


class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PositionalEmbedding, self).__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"projection_dim": self.projection_dim, "num_patches": self.num_patches}
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        batch_size = tf.shape(patch)[0]
        embs = tf.cast(
            tf.broadcast_to(
                self.position_embedding(positions),
                [batch_size, self.num_patches + 1, self.projection_dim],
            ),
            dtype=patch.dtype,
        )
        return patch + embs


def vit(
    input_shape,
    patch_size,
    projection_dim,
    num_patches,
    transformer_layers,
    num_heads,
    transformer_units,
    mlp_head_units,
):
    # Get inputs
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(projection_dim)(patches)
    # Add positional embedding
    z = PositionalEmbedding(num_patches, projection_dim)(encoded_patches)
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(z)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, z])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        z = layers.Add()([x3, x2])

    # Use class token representation
    y = layers.LayerNormalization(epsilon=1e-6)(z)[:, 0, :]
    y = layers.Dropout(0.5)(y)
    # Add MLP.
    features = mlp(y, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    output = layers.Dense(1, activation="sigmoid")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=output)
    return model


def vita(
    input_shape,
    patch_size,
    projection_dim,
    num_patches,
    transformer_layers,
    num_heads,
    transformer_units,
    mlp_head_units,
):
    # Get inputs
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(projection_dim)(patches)
    # Add positional embedding
    z = PositionalEmbedding(num_patches, projection_dim)(encoded_patches)
    Z = []
    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(z)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, z])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        z = layers.Add()([x3, x2])
        if i % 2 == 0:
            Z.append(z)

    # Use class token representation
    y = layers.LayerNormalization(epsilon=1e-6)(z)[:, 0, :]
    y = layers.Dropout(0.5)(y)
    # Attend back to the transformer's previous states
    attention = layers.Dense(1)
    G = []
    for i, z in enumerate(Z):
        c = attention(
            tf.concat(
                [tf.stack([y for _ in range(num_patches)], axis=1), z[:, 1:, :]],
                axis=-1,
            )
        )
        alpha = tf.nn.softmax(c, axis=1)
        g = tf.reduce_sum(alpha * z[:, 1:, :], axis=1)
        G.append(g)
    # Add MLP.
    features = mlp(tf.concat(G, axis=-1), hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    output = layers.Dense(1, activation="sigmoid")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=output)
    return model
