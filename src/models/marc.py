"""marc.py

TensorFlow 2 reimplementation of the Multi-Aspect Trajectory Classifier model
from: https://github.com/bigdata-ufsc/petry-2020-marc
"""
import geohash2 as gh
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from tensorflow.keras import Model, initializers, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.datasets import Dataset
from src.processors import GPSGeoHasher

base32 = [
    *[str(i) for i in range(0, 10)],
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "j",
    "k",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
binary = [np.asarray(list("{0:05b}".format(x)), dtype=int) for x in range(0, len(base32))]
base32toBin = dict(zip(base32, binary))


def bin_geohash(lat, lon, precision=15):
    hashed = gh.encode(lat, lon, precision)
    return np.concatenate([base32toBin[x] for x in hashed])


def build_classifier(
    optimizer, vocab_size, timesteps, num_classes, embedder_size=100, hidden_units=100, dropout=0.5
):
    """Build the classifier Keras model."""
    inputs = []
    embeddings = []

    for key, val in vocab_size.items():
        if key == "lat_lon":
            i = layers.Input(shape=(timesteps, val), name="input_" + key)
            e = layers.Dense(
                units=embedder_size,
                kernel_initializer=initializers.he_uniform(),
                name="emb_" + key,
            )(i)
        else:
            i = layers.Input(shape=(timesteps,), name="input_" + key)
            e = layers.Embedding(
                vocab_size[key], embedder_size, input_length=timesteps, name="emb_" + key
            )(i)
        inputs.append(i)
        embeddings.append(e)

    hidden_input = layers.Concatenate(axis=2)(embeddings)

    hidden_dropout = layers.Dropout(dropout)(hidden_input)

    rnn_cell = layers.LSTM(units=hidden_units, recurrent_regularizer=regularizers.l1(0.02))(
        hidden_dropout
    )

    rnn_dropout = layers.Dropout(dropout)(rnn_cell)

    softmax = layers.Dense(
        units=num_classes, kernel_initializer=initializers.he_uniform(), activation="softmax"
    )(rnn_dropout)

    classifier = Model(inputs=inputs, outputs=softmax)

    classifier.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["acc", "top_k_categorical_accuracy"],
    )
    return classifier


class MARC:
    """An LSTM-based trajectory classifier network.

    Based on: May Petry, L., Leite Da Silva, C., Esuli, A., Renso, C., and Bogorny, V. (2020).
    MARC: a robust method for multiple-aspect trajectory classification via space, time,
    and semantic embeddings.
    International Journal of Geographical Information Science, 34(7), 1428-1450.
    """

    def __init__(self, dataset: Dataset, geohash_precision: int = 8):
        self.dataset = dataset
        self.geohasher = GPSGeoHasher(precision=geohash_precision)
        self.geohash_dim = geohash_precision * 5
        self.y_encoder = OneHotEncoder(sparse=False)
        self.encoders = []
        self.vocab_sizes = None
        self.timesteps = None
        self.num_classes = None
        self.optimizer = None
        self.classifier = None

    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split the dataset into train and test sets.

        Use a stratified split on label to balance label classes across the
        test and train sets.

        Parameters
        ----------
        test_size : float
            The ratio of the data that should be assigned to the test set.
        """
        ids = df[[self.dataset.label_column, self.dataset.trajectory_column]].drop_duplicates()
        ids = ids.groupby("label").filter(lambda x: len(x) > 1)
        split = StratifiedShuffleSplit(test_size=test_size, n_splits=2).split(
            ids, ids[self.dataset.label_column]
        )
        train_tids, test_tids = next(split)
        train_set = df[df[self.dataset.trajectory_column].isin(train_tids)]
        test_set = df[df[self.dataset.trajectory_column].isin(test_tids)]
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
            self.geohasher.fit(df.loc[:, latlon_cols])
            self.y_encoder.fit(df.loc[:, [self.dataset.label_column]])

        latlon_hashed = pd.DataFrame(
            self.geohasher.transform(df.loc[:, latlon_cols].to_numpy()),
            columns=[f"latlon_{i}" for i in range(self.geohash_dim)],
        )
        encoded_features = []
        vocab_sizes = self.dataset.get_vocab_sizes()
        self.vocab_sizes = {**vocab_sizes, "lat_lon": self.geohash_dim}
        for feature in vocab_sizes:
            encoder = OrdinalEncoder()
            if train:
                encoder.fit(df[[feature]])
            feat_enc = pd.DataFrame(
                encoder.transform(df[[feature]]),
                columns=[feature],
            )
            self.encoders.append(encoder)
            encoded_features.append(feat_enc)
        df = df.reset_index().drop("index", errors="ignore", axis=1)
        df = df.drop(vocab_sizes.keys(), axis=1)
        df = df.drop(latlon_cols, axis=1)
        df = pd.concat([df, latlon_hashed, *encoded_features], axis=1)
        tids = df[self.dataset.trajectory_column].unique()
        tid_groups = df.groupby(self.dataset.trajectory_column).groups
        tid_dfs = [df.iloc[g] for g in tid_groups.values()]
        labels = self.y_encoder.transform(df.loc[:, [self.dataset.label_column]])
        self.num_classes = labels.shape[1]
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
        # split into one input per feature
        x_split = []
        for i, key in enumerate(self.vocab_sizes):
            if key == "lat_lon":
                x_split.append(x_pad[:, :, i : i + self.geohash_dim])
            else:
                x_split.append(x_pad[:, :, i])
        return x_split, labels, tids

    def train(
        self, optimizer=None, epochs: int = 200, batch_size: int = 64, test_size: float = 0.2
    ):
        """Train this model on the dataset."""
        if optimizer is None:
            optimizer = optimizers.Adam()
        self.optimizer = optimizer
        df = self.dataset.to_trajectories(min_points=10)
        train_set, _ = self.train_test_split(df, test_size=test_size)
        x_train, y_train, _ = self.preprocess(train_set)
        self.classifier = build_classifier(
            self.optimizer, self.vocab_sizes, self.timesteps, self.num_classes
        )

        self.classifier.fit(
            x=x_train,
            y=y_train,
            validation_split=0.1,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[],  # TODO: Early Stopping callback
        )
