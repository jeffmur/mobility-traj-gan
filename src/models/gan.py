"""models/gan.py
GAN model
Rewrite of LSTM-TrajGAN for TF2
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers, losses, regularizers

SEED = 11


def traj_loss(real_traj, gen_traj):
    """Combined GAN loss"""

    def loss(y_true, y_pred):
        # Binary cross-entropy loss for real/fake trajectory labeling by discriminator
        bce_loss = losses.binary_crossentropy(y_true, y_pred)
        # mean squared error loss for
        lat_lon_true, lat_lon_pred = (real_traj[:, :, 0:2], gen_traj[:, :, 0:2])
        mse_loss = losses.mean_squared_error(lat_lon_true, lat_lon_pred)


def build_inputs_latlon(timesteps, dense_units):
    i = layers.Input(shape=(timesteps, 2), name="input_latlon")
    mask = layers.Masking()(i)
    unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(mask)
    d = layers.Dense(
        units=dense_units,
        activation="relu",
        use_bias=True,
        kernel_initializer=initializers.he_uniform(seed=SEED),
        name="embed_latlon",
    )
    dense_latlon = [d(x) for x in unstacked]
    e = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
    return (i, e)


def build_inputs_cat(timesteps, levels, feature_name):
    i = layers.Input(shape=(timesteps, levels), name="input_" + feature_name)
    mask = layers.Masking()(i)
    unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(mask)
    d = layers.Dense(
        units=levels,
        activation="relu",
        use_bias=True,
        kernel_initializer=initializers.he_uniform(seed=SEED),
        name="emb_" + feature_name,
    )
    dense_attr = [d(x) for x in unstacked]
    e = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
    return (i, e)


def build_inputs(timesteps, dense_units):
    """Build the multiple input layer for both the generator and the discriminator."""

    # TODO: add Masking layer to input

    latlon_input, latlon_embed = build_inputs_latlon(timesteps, dense_units)
    day_input, day_embed = build_inputs_cat(timesteps, 7, "day")
    hour_input, hour_embed = build_inputs_cat(timesteps, 24, "hour")
    inputs = [latlon_input, day_input, hour_input]
    embeddings = [latlon_embed, day_embed, hour_embed]
    concat_input = layers.Concatenate(axis=2)(embeddings)
    return (inputs, concat_input)


def build_generator(
    timesteps, latlon_dense_units=64, concat_dense_units=100, lstm_units=100, latent_dim=100
):
    # Add random noise input
    inputs, concat_input = build_inputs(timesteps, latlon_dense_units)
    noise_input = layers.Input(shape=(latent_dim,), name="input_noise")
    inputs.append(noise_input)
    unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
    d = layers.Dense(
        units=concat_dense_units,
        use_bias=True,
        activation="relu",
        kernel_initializer=initializers.he_uniform(seed=1),
        name="emb_trajpoint",
    )
    dense_outputs = [d(layers.Concatenate(axis=1)([x, noise_input])) for x in unstacked]
    emb_traj = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)

    lstm_cell = layers.LSTM(
        units=lstm_units,
        batch_input_shape=(None, timesteps, latent_dim),
        return_sequences=True,
        recurrent_regularizer=regularizers.l1(0.02),
    )(emb_traj)

    latlon_output = layers.TimeDistributed(
        layers.Dense(2, activation="tanh"), name="output_latlon"
    )(lstm_cell)

    day_output = layers.TimeDistributed(layers.Dense(7, activation="softmax"), name="output_day")(
        lstm_cell
    )

    hour_output = layers.TimeDistributed(
        layers.Dense(24, activation="softmax"), name="output_hour"
    )(lstm_cell)
    outputs = [latlon_output, day_output, hour_output]
    return Model(inputs=inputs, outputs=outputs)
