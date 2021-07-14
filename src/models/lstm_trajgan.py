"""models/lstm_trajgan.py
GAN model
Rewrite of LSTM-TrajGAN for TF2
"""
import csv
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers, optimizers, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.models.base import Generative, log_start, log_end
from src.datasets import Dataset
from src.processors import GPSNormalizer

SEED = 11
LOG = logging.Logger(__name__)
LOG.setLevel(logging.DEBUG)

# Masked Loss from LSTM-TrajGAN
def traj_loss(real_traj, gen_traj, mask, latlon_weight=10.0):
    """Novel trajectory loss from LSTM-TrajGAN paper"""
    traj_length = K.sum(mask, axis=1)
    masked_latlon_full = K.sum(
        K.sum(
            tf.multiply(
                tf.multiply((gen_traj[0] - real_traj[0]), (gen_traj[0] - real_traj[0])),
                tf.concat([mask for x in range(2)], 2),
            ),
            axis=1,
        ),
        axis=1,
        keepdims=True,
    )
    masked_latlon_mse = K.sum(tf.math.divide(masked_latlon_full, traj_length))
    cat_losses = []
    for idx in range(1, len(real_traj)):
        ce_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            real_traj[idx], gen_traj[idx]
        )
        ce_loss_masked = tf.multiply(ce_loss, K.sum(mask, axis=2))
        ce_mean = K.sum(tf.math.divide(ce_loss_masked, traj_length))
        cat_losses.append(ce_mean)

    total_loss = masked_latlon_mse * latlon_weight + K.sum(cat_losses)

    return total_loss


def build_inputs_latlon(timesteps: int, dense_units: int):
    """Build input layer for the lat-lon features.

    Parameters
    ----------
    timesteps : int
        The number of time steps per trajectory.
    dense_units : int
        The number of dense units for lat-lon embedding.
    """

    i = layers.Input(shape=(timesteps, 2), name="input_latlon")
    unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(i)
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
    unstacked = layers.Lambda(lambda x: tf.unstack(x, axis=1))(i)
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
    timesteps,
    latlon_dense_units,
    concat_dense_units,
    latent_dim,
    vocab_sizes,
    noise=False,
    mask=False,
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
    if mask:
        inputs.append(layers.Input(shape=(timesteps, 1), name="input_mask"))
    emb_traj = layers.Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
    return (inputs, emb_traj)


def build_generator(
    timesteps: int,
    latlon_dense_units: int,
    concat_dense_units: int,
    lstm_units: int,
    latent_dim: int,
    lstm_reg: float,
    vocab_sizes: Dict[str, int],
):
    """Build the generator network.

    Parameters
    ----------
    timesteps : int
        The number of time steps per trajectory.
    latlon_dense_units : int
        The number of dense units in the latitude/longitude embedding
        layer.
    concat_dense_units : int
        The number of dense units in the concatenated feature fusion
        layer.
    lstm_units : int
        The number of units in the LSTM layer.
    latent_dim : int
        The dimension of the latent vector space.
    lstm_reg : float
        The L1 regularization strength for the LSTM units.
    vocab_sizes : dict
        A dictionary of each categorical feature name and its number
        of distinct values.
    """

    # Add random noise input
    inputs, emb_traj = build_inputs(
        timesteps,
        latlon_dense_units,
        concat_dense_units,
        latent_dim,
        vocab_sizes,
        noise=True,
        mask=True,
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
    # pass the mask through so the loss function can use it
    outputs.append(layers.Lambda(lambda x: x, name="output_mask")(inputs[-1]))
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
    # Compile discriminator with masked BCE loss. Mask is last output of generator
    dis.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    dis.trainable = False

    # The trajectory generator takes real trajectories and noise as inputs
    # inputs = [layers.Input(shape=(timesteps, 2), name="input_latlon")]
    # for key, val in vocab_sizes.items():
    #     inputs.append(layers.Input(shape=(timesteps, val), name="input_" + key))
    # inputs.append(layers.Input(shape=(latent_dim,), name="input_noise"))
    # inputs.append(layers.Input(shape=(timesteps, 1), name="input_mask"))
    # gen_trajs = gen(inputs)
    # y_pred = dis(gen_trajs[:-1])
    # mask = inputs[-1]
    # gan = Model(inputs, y_pred)
    # gan.add_loss(traj_loss(inputs[:-2], gen_trajs[:-1], mask))
    ##
    y_pred = dis(gen.outputs[:-1])
    gan = Model(gen.inputs, y_pred)
    mask = gen.inputs[-1]
    gan.add_loss(traj_loss(gen.inputs[:-2], gen.outputs[:-1], mask))
    gan.compile(optimizer=optimizer, loss="binary_crossentropy")
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
    for val in vocab_sizes.values():
        x_split.append(x[:, :, start : (start + val)])
        start += val
    # append mask
    x_split.append(x[:, :, [-1]])
    return x_split


def train_model(
    exp_name,
    gen,
    dis,
    gan,
    x_train,
    x_valid,
    vocab_sizes,
    epochs=200,
    batch_size=256,
    latent_dim=100,
    patience=10,
    start_time=None,
):
    """Train the GAN."""
    if start_time is None:
        start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    n_examples = x_train.shape[0]
    n_batches = np.ceil(n_examples / batch_size).astype(int)
    random_idx = np.random.permutation(x_train.shape[0])
    best_loss = 10000
    no_improvement = 0
    for i in range(1, epochs + 1):
        for j in range(1, n_batches + 1):
            idx = random_idx[batch_size * (j - 1) : batch_size * (j)]
            this_batch_size = idx.shape[0]
            x_real = x_train[idx]
            y_real = np.ones((this_batch_size, 1))
            # split the x variables into separate input arrays.
            # lat-lon = 2 features
            # one-hot encoded day of week = 7 features
            # one-hot encoded hour of day = 24 features
            x_split = split_inputs(x_real, vocab_sizes)
            # generator input is conditioned on real data with addition of Gaussian noise
            noise = np.random.normal(0, 1, (this_batch_size, latent_dim))
            # pass generations (dropping input mask) to discriminator
            gen_inputs = x_split
            gen_inputs.insert(-1, noise)
            x_fake_split = gen.predict(gen_inputs)
            y_fake = np.zeros((this_batch_size, 1))
            # train the discriminator
            dis_loss_real, dis_acc_real = dis.train_on_batch(x_split[:-2], y_real)
            dis_loss_fake, dis_acc_fake = dis.train_on_batch(x_fake_split[:-1], y_fake)
            dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_fake)
            dis_acc = 0.5 * np.add(dis_acc_real, dis_acc_fake)
            # train the combined model
            gen_labels = np.ones((this_batch_size, 1))
            # "one-sided label smoothing": use 0.9 instead of 1 as real labels
            # gen_labels = np.full((batch_size, 1), 0.9)
            gen_loss = gan.train_on_batch(gen_inputs, gen_labels, return_dict=True)["loss"]
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
        y_val_real = np.ones((x_valid.shape[0], 1))
        y_val_fake = np.zeros((x_valid.shape[0], 1))
        x_val_split = split_inputs(x_valid, vocab_sizes)
        noise = np.random.normal(0, 1, (x_valid.shape[0], latent_dim))
        gen_val_input = x_val_split
        gen_val_input.insert(-1, noise)
        gen_val_labels = np.ones((x_valid.shape[0], 1))
        gen_val_loss = gan.evaluate(gen_val_input, gen_val_labels, return_dict=True)["loss"]
        x_val_fake = gen.predict(gen_val_input)
        dis_val_r_loss, dis_val_r_acc = dis.evaluate(x_val_split[:-2], y_val_real)
        dis_val_f_loss, dis_val_f_acc = dis.evaluate(x_val_fake[:-1], y_val_fake)
        print(
            " ".join(
                f"""Experiment {exp_name} epoch{i:04d}
                G val loss: {gen_val_loss:.3f}
                D val loss real: {dis_val_r_loss:.3f} fake: {dis_val_f_loss:.3f}
                D val accuracy real: {dis_val_r_acc:.2f}, fake: {dis_val_f_acc:.2f}""".splitlines()
            )
        )
        rounding = 3
        # log learning curves to a CSV file.
        write_csv(
            exp_name,
            start_time,
            i,
            val_gen_loss=round(gen_val_loss, rounding),
            val_dis_acc_real=round(dis_val_r_acc, rounding),
            val_dis_loss_real=round(dis_val_r_loss, rounding),
            val_dis_acc_fake=round(dis_val_f_acc, rounding),
            val_dis_loss_fake=round(dis_val_f_loss, rounding),
        )
        # early stopping and checkpointing
        if gen_loss < best_loss:
            best_loss = gen_loss
            gen.save(f"experiments/{exp_name}/{start_time}/{i:04d}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break


def run():
    """Run an experiment with a given set of hyperparameters."""
    exp_name = "trajgan_000"
    os.makedirs(f"experiments/{exp_name}", exist_ok=True)
    optimizer = optimizers.Adam(0.001, 0.5)
    timesteps = 144
    epochs = 200
    vocab_sizes = {"day": 7, "hour": 24, "category": 10}
    gen, dis, gan = build_gan(optimizer, timesteps, vocab_sizes)
    # from CSV
    # import pandas as pd
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
    # x = x[0 : (len(vocab_sizes) + 1)]
    # Padding zero to reach the maxlength
    x = np.concatenate(
        [pad_sequences(f, timesteps, padding="pre", dtype="float32") for f in x], axis=2
    )

    n = x.shape[0]
    idx = np.random.permutation(n)
    valid_split = 0.1
    valid_n = np.ceil(n * valid_split).astype(int)
    valid_idx, train_idx = idx[:valid_n], idx[valid_n:]
    x_train, x_valid = x[train_idx, :], x[valid_idx, :]
    train_model(exp_name, gen, dis, gan, x_train, x_valid, vocab_sizes, epochs=epochs)
    return gen, dis, gan


class LSTMTrajGAN(Generative):
    """An LSTM-based Trajectory Generative Adversarial Network.

    Based on: Rao, J., Gao, S.*, Kang, Y. and Huang, Q. (2020).
    LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection.
    In the Proceedings of the 11th International Conference on Geographic

    Information Science (GIScience 2021), pp. 1-16.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.vocab_sizes = dataset.get_vocab_sizes()
        self.gps_normalizer = GPSNormalizer()
        self.encoders = []
        self.timesteps = None
        self.learning_rate = None
        self.momentum = None
        self.latent_dim = None
        self.batch_size = None
        self.test_size = None
        self.patience = None
        self.gen = None
        self.dis = None
        self.gan = None
        self.trained_epochs = 0

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split the dataset into train and test sets.

        Use a group shuffle split to assign each trajectory to either
        the train set or the test set.

        Parameters
        ----------
        test_size : float
            The ratio of the data that should be assigned to the test set.
        """
        train_inds, test_inds = next(
            GroupShuffleSplit(test_size=test_size, n_splits=2).split(
                df, groups=df[self.dataset.trajectory_column]
            )
        )

        train_set = df.iloc[train_inds]
        test_set = df.iloc[test_inds]
        return train_set, test_set

    def preprocess(
        self,
        data: pd.DataFrame,
        train: bool = True,
        max_len_qtile: float = 0.95,
        padding: str = "pre",
    ):
        """Transform trajectories DataFrame to a NumPy tensor with zero padding."""
        df = data.copy()
        latlon_cols = [self.dataset.lat_column, self.dataset.lon_column]
        if train:
            self.gps_normalizer.fit(df.loc[:, latlon_cols])
        df.loc[:, latlon_cols] = self.gps_normalizer.transform(df.loc[:, latlon_cols])

        encoded_features = []
        for feature in self.vocab_sizes:
            encoder = OneHotEncoder(sparse=False)
            if train:
                encoder.fit(df[[feature]])
            feat_enc = pd.DataFrame(
                encoder.transform(df[[feature]]),
                columns=[f"{feature}_{i}" for i in range(0, self.vocab_sizes[feature])],
            )
            self.encoders.append(encoder)
            encoded_features.append(feat_enc)
        df = df.reset_index().drop("index", errors="ignore", axis=1)
        df = df.drop(self.vocab_sizes.keys(), axis=1)
        df = pd.concat([df, *encoded_features], axis=1)
        tids = df[self.dataset.trajectory_column].unique()
        tid_groups = df.groupby(self.dataset.trajectory_column).groups
        tid_dfs = [df.iloc[g] for g in tid_groups.values()]
        # label is a y-variable, split it into a separate vector
        labels = np.array([tdf[self.dataset.label_column].values[0] for tdf in tid_dfs])
        if train:
            # get percentile of sequence length
            x_lengths = sorted([len(l) for l in tid_dfs])
            pct_idx = round(len(x_lengths) * max_len_qtile)
            maxlen = x_lengths[pct_idx]
            self.timesteps = maxlen
        x_nested = [tdf.iloc[:, 2:].to_numpy() for tdf in tid_dfs]
        x_pad = pad_sequences(
            x_nested,
            maxlen=self.timesteps,
            padding=padding,
            truncating=padding,
            value=0.0,
            dtype="float32",
        )
        # add mask vector
        mask = np.expand_dims((x_pad[:, :, 0] != 0.0).astype("float32"), axis=2)
        x_pad = np.concatenate([x_pad, mask], axis=2)
        return x_pad, labels, tids

    def postprocess(self, x: np.array, y: np.array, tids: np.array):
        """Convert the tensor representation back to a data frame of GPS records."""
        tids = np.repeat(np.expand_dims(tids, axis=1), x.shape[1], axis=1).reshape(
            x.shape[0] * x.shape[1]
        )
        labels = np.repeat(np.expand_dims(y, axis=1), x.shape[1], axis=1).reshape(
            x.shape[0] * x.shape[1]
        )
        x_res = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        df = pd.DataFrame(x_res)
        df[self.dataset.label_column] = labels
        df[self.dataset.trajectory_column] = tids
        df = df[df[0] + df[1] != 0.0].reset_index()
        latlon_cols = [self.dataset.lat_column, self.dataset.lon_column]
        df.loc[:, latlon_cols] = self.gps_normalizer.inverse_transform(df[[0, 1]])

        # Reverse one-hot encoding on categorical features
        col_pos = 2
        for idx, feature in enumerate(self.vocab_sizes.keys()):
            encoder = self.encoders[idx]
            df[feature] = encoder.inverse_transform(
                df[list(range(col_pos, col_pos + self.vocab_sizes[feature]))]
            )
            col_pos += self.vocab_sizes[feature]

        return df.iloc[:, df.columns.get_loc(self.dataset.label_column) :]

    def train(
        self,
        epochs: int = 200,
        batch_size: int = 256,
        latent_dim: int = 100,
        learning_rate: float = 0.001,
        momentum: float = 0.5,
        test_size: float = 0.2,
        patience: int = 10,
    ):
        """Train this model on a dataset."""
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.test_size = test_size
        self.patience = patience
        optimizer = optimizers.Adam(learning_rate, momentum)
        df = self.dataset.to_trajectories(min_points=10)
        train_set, _ = self.train_test_split(df, test_size=test_size)
        x, _, _ = self.preprocess(train_set)
        # Train-valid split
        n = x.shape[0]
        idx = np.random.permutation(n)
        valid_split = 0.1
        valid_n = np.ceil(n * valid_split).astype(int)
        valid_idx, train_idx = idx[:valid_n], idx[valid_n:]
        x_train, x_valid = x[train_idx, :], x[valid_idx, :]
        # build the network
        self.gen, self.dis, self.gan = build_gan(optimizer, self.timesteps, self.vocab_sizes)
        exp_name = f"{type(self).__name__}"
        hparams = dict(epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
        start_time = log_start(LOG, exp_name, **hparams)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        exp_path = Path(f"experiments/{exp_name}/{start_time_str}")
        train_model(
            exp_name,
            self.gen,
            self.dis,
            self.gan,
            x_train,
            x_valid,
            self.vocab_sizes,
            patience=patience,
            start_time=start_time_str,
            **hparams,
        )
        self.trained_epochs += epochs
        log_end(LOG, exp_name, start_time)
        self.save(f"{exp_path}/saved_model")

    def save(self, save_path: os.PathLike):
        """Serialize the model to a directory on disk."""
        os.makedirs(save_path, exist_ok=True)
        save_path = Path(save_path)
        pickle.dump(self.dataset, save_path / "dataset.pkl")
        hparams = dict(
            latent_dim=self.latent_dim,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
        if self.trained_epochs > 0:
            pickle.dump(self.encoders, save_path / "encoders.pkl")
            pickle.dump(self.gps_normalizer, save_path / "gps_normalizer.pkl")
            self.gen.save(save_path / "generator_model")
            self.dis.save(save_path / "discriminator_model")
            self.gan.save(save_path / "gan_model")
            train_state = dict(
                patience=self.patience,
                trained_epochs=self.trained_epochs,
                timesteps=self.timesteps,
                vocab_sizes=self.vocab_sizes,
            )
            pickle.dump(train_state, save_path / "train_state.pkl")
        pickle.dump(hparams, save_path / "hparams.pkl")

        return self

    @classmethod
    def restore(cls, save_path: os.PathLike):
        """Restore the model from a checkpoint on disk."""
        save_path = Path(save_path)
        dataset = pickle.load(save_path / "dataset.pkl")
        model = cls(dataset)
        model.gen = load_model(save_path / "generator_model")
        model.dis = load_model(save_path / "discriminator_model")
        model.gan = load_model(save_path / "gan_model")
        model.encoders = pickle.load(save_path / "encoders.pkl")
        model.gps_normalizer = pickle.load(save_path / "gps_normalizer.pkl")
        hparams = pickle.load(save_path / "hparams.pkl")
        train_state = pickle.load(save_path / "train_state.pkl")
        for key, val in {**hparams, **train_state}:
            setattr(model, key, val)
        return model
