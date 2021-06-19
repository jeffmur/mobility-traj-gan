"""models/gan.py
GAN model
Rewrite of LSTM-TrajGAN for TF2
"""
import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import (Model, initializers, layers, losses, optimizers,
                              regularizers, utils)
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    i = layers.Input(shape=(timesteps, levels), name=f"input_{feature_name}")
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


def build_inputs(
    timesteps, latlon_dense_units, concat_dense_units, latent_dim, vocab_sizes, noise=False
):
    """Build the multiple input layer for both the generator and the discriminator."""
    latlon_input, latlon_embed = build_inputs_latlon(timesteps, latlon_dense_units)
    inputs = [latlon_input]
    embeddings = [latlon_embed]
    for key, val in vocab_sizes.items():
        cat_input, cat_embed = build_inputs_cat(timesteps, val, key)
        inputs.append(cat_input)
        embeddings.append(cat_embed)
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
    vocab_sizes,
):
    """Build the generator network."""
    # Add random noise input
    inputs, emb_traj = build_inputs(
        timesteps, latlon_dense_units, concat_dense_units, latent_dim, vocab_sizes, noise=True
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
    outputs = [latlon_output]
    for key, val in vocab_sizes.items():
        out = layers.TimeDistributed(layers.Dense(val, activation="softmax"), name=f"output_{key}")(
            lstm_cell
        )
        outputs.append(out)
    return Model(inputs=inputs, outputs=outputs, name="generator")


def build_discriminator(
    timesteps, latlon_dense_units, concat_dense_units, lstm_units, latent_dim, lstm_reg, vocab_sizes
):
    """Build the discriminator network."""
    inputs, emb_traj = build_inputs(
        timesteps, latlon_dense_units, concat_dense_units, latent_dim, vocab_sizes
    )
    lstm_cell = layers.LSTM(units=lstm_units, recurrent_regularizer=regularizers.l1(lstm_reg))(
        emb_traj
    )
    output = layers.Dense(1, activation="sigmoid")(lstm_cell)
    return Model(inputs=inputs, outputs=output, name="discriminator")


def build_gan(
    optimizer,
    timesteps,
    vocab_sizes,
    latlon_dense_units=64,
    concat_dense_units=100,
    lstm_units=100,
    latent_dim=100,
    lstm_reg=0.02,
):
    """Build the full GAN network"""
    gen = build_generator(
        timesteps,
        latlon_dense_units,
        concat_dense_units,
        lstm_units,
        latent_dim,
        lstm_reg,
        vocab_sizes,
    )
    dis = build_discriminator(
        timesteps,
        latlon_dense_units,
        concat_dense_units,
        lstm_units,
        latent_dim,
        lstm_reg,
        vocab_sizes,
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
            *["categorical_crossentropy" for _ in vocab_sizes],
        ],
        loss_weights=[1.0, 10.0, 1.0, 1.0],  # 10x weight on lat-lon loss
    )
    return gen, dis, gan


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


def split_inputs(x, vocab_sizes):
    """Split NumPy array into separate feature arrays."""
    x_split = [x[:, :, 0:2]]  # lat-lon feature
    start = 2
    for key, val in vocab_sizes.items():
        x_split.append(x[:, :, start : (start + val)])
        start = start + val
    return x_split


def train(
    exp_name,
    gen,
    dis,
    gan,
    x_train,
    x_valid,
    vocab_sizes,
    epochs=200,
    batch_size=32,
    latent_dim=100,
):
    """Train the GAN."""
    start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    n_examples = x_train.shape[0]
    n_batches = np.ceil(n_examples / batch_size).astype(int)
    random_idx = np.random.permutation(x_train.shape[0])
    for i in range(1, epochs + 1):
        for j in range(1, n_batches + 1):
            idx = random_idx[batch_size * i : batch_size * (i + 1)]
            x_real = x_train[idx]
            y_real = np.ones((idx.shape[0], 1))
            # split the x variables into separate input arrays.
            # lat-lon = 2 features
            # one-hot encoded day of week = 7 features
            # one-hot encoded hour of day = 24 features
            x_split = split_inputs(x_real, vocab_sizes)
            # generator input is conditioned on real data with addition of Gaussian noise
            noise = np.random.normal(0, 1, (idx.shape[0], latent_dim))
            x_fake = np.concatenate(gen.predict([*x_split, noise]), axis=2)
            y_fake = np.zeros((batch_size, 1))
            # shuffle the real and synthetic examples together
            x = np.vstack((x_real, x_fake))
            y = np.vstack((y_real, y_fake))
            idx_shuf = np.random.permutation(x.shape[0])
            x = x[idx_shuf]
            y = y[idx_shuf]
            x_split = split_inputs(x, vocab_sizes)
            # train the discriminator
            dis_res = dis.train_on_batch(x_split, y, return_dict=True)
            dis_loss = dis_res["loss"]
            dis_acc = dis_res["accuracy"]
            # train the combined model
            gen_labels = np.ones((batch_size * 2, 1))
            # "one-sided label smoothing": use 0.9 instead of 1 as real labels
            # gen_labels = np.full((batch_size, 1), 0.9)
            noise = np.random.normal(0, 1, (x.shape[0], latent_dim))
            # TODO: get the shapes right for loss function calculation
            gen_loss = gan.train_on_batch([*x_split, noise], [gen_labels, *x_split])
            print(
                " ".join(
                    f"""Epoch {i}/{epochs}, Batch {j}/{n_batches}, dis_loss={dis_loss:.3f},
                    dis_acc={dis_acc:.2f}, gen_loss={gen_loss:.3f}
                    """.splitlines()
                ),
                end="\r",
            )
        print()
        # validation at end of epoch
        # y_val_real = np.ones((x_valid.shape[0], 1))
        # y_val_fake = np.zeros((x_valid.shape[0], 1))
        # x_val_split = split_inputs(x_valid, vocab_sizes)
        # noise = np.random.normal(0, 1, (x_valid.shape[0], latent_dim))
        # gen_input = [*x_val_split, noise]
        # x_val_fake = gen.predict(gen_input)
        # dis_val_r_loss, dis_valid_r_acc = dis.evaluate(x_val_split, y_val_real)
        # dis_valid_f_loss, dis_valid_f_acc = dis.evaluate(x_val_fake, y_val_fake)
        # print(
        #     " ".join(
        #         f"""Experiment {exp_name} epoch{i:04d}
        #         loss real: {dis_val_r_loss:.3f} fake: {dis_valid_f_loss:.3f}
        #         accuracy real: {dis_valid_r_acc:.2f}, fake: {dis_valid_f_acc:.2f}""".splitlines()
        #     )
        # )
        # # TODO: add early stopping
        # gen.save(f"experiments/{exp_name}/{start_time}/{i:04d}")
        # rounding = 3
        # # log learning curves to a CSV file.
        # write_csv(
        #     exp_name,
        #     start_time,
        #     i,
        #     val_discriminator_accuracy_real=round(dis_valid_r_acc, rounding),
        #     val_discriminator_loss_real=round(dis_val_r_loss, rounding),
        #     val_discriminator_accuracy_fake=round(dis_valid_f_acc, rounding),
        #     val_discriminator_loss_fake=round(dis_valid_f_loss, rounding),
        # )


def run():
    """Run an experiment with a given set of hyperparameters."""
    os.makedirs("experiments", exist_ok=True)
    exp_name = "000"
    optimizer = optimizers.Adam(0.001, 0.5)
    timesteps = 144
    vocab_sizes = {"day": 7, "hour": 24, "category": 10}
    gen, dis, gan = build_gan(optimizer, timesteps, vocab_sizes)
    # from CSV
    # df = pd.read_csv("data/dev_train_encoded_final.csv")
    # cat_frames = []
    # for key, val in vocab_sizes.items():
    #     cat = pd.DataFrame(
    #         utils.to_categorical(df[key], num_classes=val),
    #         columns=[f"{key}_{i}" for i in range(0, val)],
    #     )
    #     cat_frames.append(cat)
    # df = pd.concat([df.drop(["tid", "label", *vocab_sizes.keys()], axis=1), *cat_frames], axis=1)
    # x = df.to_numpy()

    ## From saved npy
    x = np.load("data/final_train.npy", allow_pickle=True)
    x = x[0 : (len(vocab_sizes) + 1)]
    # Padding zero to reach the maxlength
    x = np.concatenate(
        [pad_sequences(f, timesteps, padding="post", dtype="float64") for f in x], axis=2
    )

    n = x.shape[0]
    idx = np.random.permutation(n)
    valid_split = 0.1
    valid_n = np.ceil(n * valid_split).astype(int)
    valid_idx, train_idx = idx[:valid_n], idx[valid_n:]
    x_train, x_valid = x[train_idx, :], x[valid_idx, :]
    train(exp_name, gen, dis, gan, x_train, x_valid, vocab_sizes)
    return gen, dis, gan
