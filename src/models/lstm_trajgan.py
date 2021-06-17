"""models/lstm_trajgan.py
Reimplementation of LSTM-TrajGAN
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers, losses, regularizers


def traj_loss(real_traj, gen_traj):
    """Novel trajectory loss from LSTM-TrajGAN paper"""

    def loss(y_true, y_pred):
        traj_length = K.sum(real_traj[3], axis=1)

        bce_loss = losses.binary_crossentropy(y_true, y_pred)

        # TODO: replace this with a Masking layer in the model
        masked_latlon_full = K.sum(
            K.sum(
                tf.multiply(
                    tf.multiply((gen_traj[0] - real_traj[0]), (gen_traj[0] - real_traj[0])),
                    tf.concat([real_traj[3] for x in range(2)], 2),
                ),
                axis=1,
            ),
            axis=1,
            keepdims=True,
        )
        masked_latlon_mse = K.sum(tf.math.divide(masked_latlon_full, traj_length))

        ce_category = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            gen_traj[1], real_traj[1]
        )
        ce_day = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(gen_traj[2], real_traj[2])
        ce_hour = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(gen_traj[3], real_traj[3])

        ce_category_masked = tf.multiply(ce_category, K.sum(real_traj[3], axis=2))
        ce_day_masked = tf.multiply(ce_day, K.sum(real_traj[3], axis=2))
        ce_hour_masked = tf.multiply(ce_hour, K.sum(real_traj[3], axis=2))

        ce_category_mean = K.sum(tf.math.divide(ce_category_masked, traj_length))
        ce_day_mean = K.sum(tf.math.divide(ce_day_masked, traj_length))
        ce_hour_mean = K.sum(tf.math.divide(ce_hour_masked, traj_length))

        p_bce = 1
        p_latlon = 10
        p_cat = 1
        p_day = 1
        p_hour = 1

        return (
            bce_loss * p_bce
            + masked_latlon_mse * p_latlon
            + ce_category_mean * p_cat
            + ce_day_mean * p_day
            + ce_hour_mean * p_hour
        )

    return loss


class LSTMAdversarial:
    """LSTM-based GAN for generating synthetic trajectories.
    Based on LSTM-TrajGAN
    https://github.com/GeoDS/LSTM-TrajGAN/blob/master/model.py
    """

    def __init__(self, optimizer, latent_dim, max_length, vocab_size=None):
        self.latent_dim = latent_dim
        self.max_length = max_length
        if vocab_size is None:
            vocab_size = dict(lat_lon=2, day=7, hour=24, mask=1)
        self.vocab_size = vocab_size
        self.scale_factor = 1
        self.generator = None
        self.discriminator = None
        self.combined = None
        self.optimizer = optimizer
        self.build_gan()

    def _build_embeddings(self):
        inputs = []
        embeddings = []
        for key, val in self.vocab_size.items():
            if key != "mask":
                if key == "lat_lon":
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
        return inputs, embeddings

    def build_generator(self):
        """Build the generator network."""
        inputs, embeddings = self._build_embeddings()
        mask = layers.Input(shape=(self.max_length, 1), name="input_mask")
        inputs.append(mask)
        noise = layers.Input(shape=(self.latent_dim,), name="input_noise")
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
        for key, val in self.vocab_size.items():
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
        inputs, embeddings = self._build_embeddings()

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

    def build_gan(self):
        self.generator = self.build_generator()
        # The trajectory generator takes real trajectories and noise as inputs
        noise = layers.Input(shape=(self.latent_dim,), name="input_noise")
        inputs = []
        for key in self.vocab_size:
            i = layers.Input(shape=(self.max_length, self.vocab_size[key]), name="input_" + key)
            inputs.append(i)
        inputs.append(noise)
        gen_trajs = self.generator(inputs)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"]
        )

        self.discriminator.trainable = False
        pred = self.discriminator(gen_trajs[:3])  # 3 input features
        self.combined = Model(inputs, pred)
        self.combined.compile(loss=traj_loss(inputs, gen_trajs), optimizer=self.optimizer)

    @staticmethod
    def draw_real(x, n):
        """Draw a batch of random real examples from the dataset."""
        idxs = np.random.permutation(x.shape[0])[0:n]
        y = np.ones((n, 1))
        return x[idxs], y

    @staticmethod
    def draw_fake(generator, latent_dim: int, n: int):
        """Draw a batch of fake images from the latent space."""
        inputs = np.random.randn(latent_dim * n).reshape(n, latent_dim)
        x = generator.predict(inputs)
        y = np.zeros((n, 1))
        return x, y

    def train(self, X_train, epochs, batch_size, sample_interval=10):
        """Fit the model to training data."""
        for epoch in range(1, epochs + 1):

            random_indices = np.random.permutation(X_train[0].shape[0])

            num_batches = np.ceil(random_indices.shape[0] / batch_size).astype(np.int)

            for i in range(num_batches):

                # Select a random batch of real trajectories
                idx = random_indices[batch_size * i : batch_size * (i + 1)]

                # Ground truths for real trajectories and synthetic trajectories
                real_bc = np.ones((idx.shape[0], 1))
                syn_bc = np.zeros((idx.shape[0], 1))

                # Random noise
                noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))

                real_trajs_bc = []
                real_trajs_bc.append(X_train[0][idx])  # latlon
                real_trajs_bc.append(X_train[1][idx])  # day
                real_trajs_bc.append(X_train[2][idx])  # hour
                real_trajs_bc.append(X_train[3][idx])  # mask
                real_trajs_bc.append(noise)  # random noise

                # Generate a batch of synthetic trajectories
                gen_trajs_bc = self.generator.predict(real_trajs_bc)

                # Train the discriminator
                # No mask and noise are used
                d_loss_real = self.discriminator.train_on_batch(real_trajs_bc[:3], real_bc)
                d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:3], syn_bc)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

                # Train the generator
                # Mask and noise are used
                noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))
                real_trajs_bc[5] = noise
                g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)

                # Print the losses
                print(
                    "[Epoch %d/%d] [Batch %d/%d] D Loss: %f | G Loss: %f"
                    % (epoch, epochs, i + 1, num_batches, d_loss[0], g_loss)
                )
            # Save the params
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print("Model params saved to the disk.")

    def save_checkpoint(self, epoch):
        self.combined.save_weights("training_params/C_model_" + str(epoch) + ".h5")
        self.generator.save_weights("training_params/G_model_" + str(epoch) + ".h5")
        self.discriminator.save_weights("training_params/D_model_" + str(epoch) + ".h5")
        print("Training Params Saved")
