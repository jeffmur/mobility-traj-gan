"""models/gan.py
GAN model
"""
import tensorflow as tf
from tensorflow.keras import Model, initializers, layers, regularizers


class LSTMAdversarial:
    """LSTM-based GAN for generating synthetic trajectories.
    Based on LSTM-TrajGAN
    https://github.com/GeoDS/LSTM-TrajGAN/blob/master/model.py
    """

    def __init__(self, latent_dim, max_length):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.vocab_size = dict(lat_lon=2, day=7, hour=24, mask=1)
        self.scale_factor = 1

    def build_generator(self):
        """Build the generator network."""
        # Input Layer
        inputs = []
        # Embedding Layer
        embeddings = []
        noise = layers.Input(shape=(self.latent_dim,), name="input_noise")
        mask = layers.Input(shape=(self.max_length, 1), name="input_mask")
        for key, val in self.vocab_size:
            if key == "mask":
                inputs.append(mask)
            elif key == "lat_lon":
                i = layers.Input(shape=(self.max_length, val), name="input_" + key)
                unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = layers.Dense(
                    units=64,
                    activation="relu",
                    use_bias=True,
                    kernel_initializer=initializers.he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_latlon = [d(x) for x in unstacked]
                e = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = layers.Input(shape=(self.max_length, val), name="input_" + key)
                unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = layers.Dense(
                    units=val,
                    activation="relu",
                    use_bias=True,
                    kernel_initializer=initializers.he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_attr = [d(x) for x in unstacked]
                e = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        inputs.append(noise)

        # Feature Fusion Layer
        concat_input = layers.Concatenate(axis=2)(embeddings)
        unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = layers.Dense(
            units=100,
            use_bias=True,
            activation="relu",
            kernel_initializer=initializers.he_uniform(seed=1),
            name="emb_trajpoint",
        )
        dense_outputs = [d(layers.Concatenate(axis=1)([x, noise])) for x in unstacked]
        emb_traj = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)

        # LSTM Modeling Layer (many-to-many)
        lstm_cell = layers.LSTM(
            units=100,
            batch_input_shape=(None, self.max_length, 100),
            return_sequences=True,
            recurrent_regularizer=regularizers.l1(0.02),
        )(emb_traj)

        # Outputs
        outputs = []
        for key, val in self.vocab_size:
            if key == "mask":
                output_mask = layers.Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            elif key == "lat_lon":
                output = layers.TimeDistributed(
                    layers.Dense(2, activation="tanh"), name="output_latlon"
                )(lstm_cell)
                output_stretched = layers.Lambda(lambda x: x * self.scale_factor)(output)
                outputs.append(output_stretched)
            else:
                output = layers.TimeDistributed(
                    layers.Dense(val, activation="softmax"), name="output_" + key
                )(lstm_cell)
                outputs.append(output)

        return Model(inputs=inputs, outputs=outputs)

    def build_discriminator(self):
        """Build the discriminator adversary."""
        inputs = []
        embeddings = []
        for key, val in self.vocab_size:
            if key == "lat_lon":
                i = layers.Input(shape=(self.max_length, val), name="input_" + key)

                unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = layers.Dense(
                    units=64,
                    use_bias=True,
                    activation="relu",
                    kernel_initializer=initializers.he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_latlon = [d(x) for x in unstacked]
                e = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            elif key != "mask":
                i = layers.Input(shape=(self.max_length, val), name="input_" + key)
                unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = layers.Dense(
                    units=val,
                    use_bias=True,
                    activation="relu",
                    kernel_initializer=initializers.he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_attr = [d(x) for x in unstacked]
                e = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)

        # Feature Fusion Layer
        concat_input = layers.Concatenate(axis=2)(embeddings)
        unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = layers.Dense(
            units=100,
            use_bias=True,
            activation="relu",
            kernel_initializer=initializers.he_uniform(seed=1),
            name="emb_trajpoint",
        )
        dense_outputs = [d(x) for x in unstacked]
        emb_traj = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)

        # LSTM Modeling Layer (many-to-one)
        lstm_cell = layers.LSTM(units=100, recurrent_regularizer=regularizers.l1(0.02))(emb_traj)

        # Output
        sigmoid = layers.Dense(1, activation="sigmoid")(lstm_cell)

        return Model(inputs=inputs, outputs=sigmoid)
