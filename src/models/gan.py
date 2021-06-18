"""models/gan.py
GAN model
Rewrite of LSTM-TrajGAN for TF2
"""
import csv
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, initializers, layers, losses, regularizers

SEED = 11


def build_inputs_latlon(timesteps, dense_units):
    """Build input layers for lat-lon features."""
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
    """Build input layers for categorical features."""
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


def build_inputs(timesteps, latlon_dense_units, concat_dense_units, latent_dim, noise=False):
    """Build the multiple input layer for both the generator and the discriminator."""
    latlon_input, latlon_embed = build_inputs_latlon(timesteps, latlon_dense_units)
    day_input, day_embed = build_inputs_cat(timesteps, 7, "day")
    hour_input, hour_embed = build_inputs_cat(timesteps, 24, "hour")
    inputs = [latlon_input, day_input, hour_input]
    embeddings = [latlon_embed, day_embed, hour_embed]
    concat_input = layers.Concatenate(axis=2)(embeddings)
    unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)

    d = layers.Dense(
        units=concat_dense_units,
        activation="relu",
        kernel_initializer=initializers.he_uniform(seed=1),
        name="emb_trajpoint",
    )
    if noise:
        noise_input = layers.Input(shape=(latent_dim,), name="input_noise")
        inputs.append(noise_input)
        dense_outputs = [d(layers.Concatenate(axis=1)([x, noise_input])) for x in unstacked]
    else:
        dense_outputs = [d(x) for x in unstacked]
    emb_traj = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
    return (inputs, emb_traj)


def build_generator(
    timesteps,
    latlon_dense_units,
    concat_dense_units,
    lstm_units,
    latent_dim,
    lstm_reg,
):
    """Build the generator network."""
    # Add random noise input
    inputs, emb_traj = build_inputs(
        timesteps, latlon_dense_units, concat_dense_units, latent_dim, noise=True
    )

    lstm_cell = layers.LSTM(
        units=lstm_units,
        batch_input_shape=(None, timesteps, latent_dim),
        return_sequences=True,
        recurrent_regularizer=regularizers.l1(lstm_reg),
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
    return Model(inputs=inputs, outputs=outputs, name="generator")


def build_discriminator(
    timesteps, latlon_dense_units, concat_dense_units, lstm_units, latent_dim, lstm_reg
):
    """Build the discriminator network."""
    inputs, emb_traj = build_inputs(timesteps, latlon_dense_units, concat_dense_units, latent_dim)

    # LSTM Modeling Layer (many-to-one)
    lstm_cell = layers.LSTM(units=lstm_units, recurrent_regularizer=regularizers.l1(lstm_reg))(
        emb_traj
    )

    # Output
    output = layers.Dense(1, activation="sigmoid")(lstm_cell)

    return Model(inputs=inputs, outputs=output, name="discriminator")


def build_gan(
    optimizer,
    timesteps,
    latlon_dense_units=64,
    concat_dense_units=100,
    lstm_units=100,
    latent_dim=100,
    lstm_reg=0.02,
):
    """Build the full GAN network"""
    gen = build_generator(
        timesteps, latlon_dense_units, concat_dense_units, lstm_units, latent_dim, lstm_reg
    )
    dis = build_discriminator(
        timesteps, latlon_dense_units, concat_dense_units, lstm_units, latent_dim, lstm_reg
    )
    dis.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    dis.trainable = False
    pred = dis(gen.outputs)
    gan = Model(inputs=gen.inputs, outputs=[pred, *(gen.outputs)])
    gan.compile(
        optimizer=optimizer,
        loss=[
            "binary_crossentropy",  # discriminator real/fake trajectory predictions
            "mean_squared_error",  # lat-lon predictions
            "categorical_crossentropy",  # day of week predictions
            "categorical_crossentropy",  # hour of day predictions
        ],
        loss_weights=[1.0, 10.0, 1.0, 1.0],  # 10x weight on lat-lon loss
    )
    return gen, dis, gan


def draw_real(dataset, n, channels: int = 1):
    """Draw a batch of random real images from the dataset."""
    idxs = np.random.permutation(dataset.shape[0])[0:n]
    y = np.ones((n, 1))
    return dataset[idxs], y


def draw_z(latent_dim, n):
    """Draw points from the latent space."""
    return np.random.randn(latent_dim * n).reshape(n, latent_dim)


def draw_fake(generator, inputs, latent_dim: int, n: int):
    """Draw a batch of fake images from the latent space."""
    x = generator.predict(inputs)
    y = np.zeros((n, 1))
    return x, y


def write_csv(exp_name, start_time, epoch, **kwargs):
    """Save the provided labels by appending to a CSV file."""
    fieldnames = ["epoch", *kwargs.keys()]
    row = {"epoch": epoch, **kwargs}
    csv_path = f"experiments/{exp_name}/{start_time}/history.csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if os.path.isfile(csv_path):
        with open(csv_path, "a") as label_file:
            writer = csv.DictWriter(label_file, fieldnames=fieldnames)
            writer.writerow(row)
    else:
        with open(csv_path, "w") as label_file:
            writer = csv.DictWriter(label_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def train(exp_name, gen, dis, gan, x_train, x_valid, epochs=200, batch_size=256, latent_dim=100):
    """Train the GAN."""
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    n_examples = x_train.shape[0]
    n_batches = np.ceil(n_examples / batch_size).astype(int)
    for i in range(1, epochs + 1):
        for j in range(1, n_batches + 1):
            x_real, y_real = draw_real(x_train, batch_size // 2)

            # split the x variables into separate input arrays.
            # lat-lon = 2 features
            # one-hot encoded day of week = 7 features
            # one-hot encoded hour of day = 24 features
            x_split = [x_real[:, :, 0:2], x_real[:, :, 2:9], x_real[:, :, 9:]]

            # generate synthetic examples
            gen_input = draw_z(latent_dim, batch_size)  # TODO: check this is the right size
            x_fake, y_fake = draw_fake(gen, [*x_split, gen_input], latent_dim, batch_size // 2)
            x = np.vstack((x_real, x_fake))
            y = np.vstack((y_real, y_fake))

            # shuffle the real and synthetic examples together
            idx = np.random.permutation(x.shape[0])
            x = x[idx]
            y = y[idx]

            # train the discriminator
            dis_res = dis.train_on_batch(x, y, return_dict=True)
            dis_loss = dis_res["loss"]
            dis_acc = dis_res["accuracy"]

            # train the combined model
            gen_labels = np.ones((batch_size, 1))
            # "one-sided label smoothing": use 0.9 instead of 1 as real labels
            # gen_labels = np.full((batch_size, 1), 0.9)
            gen_loss = gan.train_on_batch(gen_input, gen_labels)
            print(
                " ".join(
                    f"""Epoch {i}/{epochs}, Batch {j}/{n_batches}, dis_loss={dis_loss:.3f},
                    dis_acc={dis_acc:.2f}, gen_loss={gen_loss:.3f}
                    """.splitlines()
                ),
                end="\r",
            )
        print()
        x_valid_real, y_valid_real = draw_real(x_valid, batch_size // 2)
        x_valid_split = [x_valid_real[:, :, 0:2], x_valid_real[:, :, 2:9], x_valid_real[:, :, 9:]]
        gen_valid_input = draw_z(latent_dim, batch_size // 2)
        x_valid_fake, y_valid_fake = draw_fake(
            gen, [*x_valid_split, gen_valid_input], latent_dim, batch_size // 2
        )
        dis_valid_r_loss, dis_valid_r_acc = dis.evaluate(x_valid_real, y_valid_real)
        dis_valid_f_loss, dis_valid_f_acc = dis.evaluate(x_valid_fake, y_valid_fake)
        print(
            " ".join(
                f"""Experiment {exp_name} epoch{i:04d}
                loss real: {dis_valid_r_loss:.3f} fake: {dis_valid_f_loss:.3f}
                accuracy real: {dis_valid_r_acc:.2f}, fake: {dis_valid_f_acc:.2f}""".splitlines()
            )
        )
        rounding = 3
        # log learning curves to a CSV file.
        write_csv(
            exp_name,
            start_time,
            i,
            val_discriminator_accuracy_real=round(dis_valid_r_acc, rounding),
            val_discriminator_loss_real=round(dis_valid_r_loss, rounding),
            val_discriminator_accuracy_fake=round(dis_valid_f_acc, rounding),
            val_discriminator_loss_fake=round(dis_valid_f_loss, rounding),
        )
