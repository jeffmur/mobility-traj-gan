"""marc.py

TensorFlow 2 reimplementation of the Multi-Aspect Trajectory Classifier model
from: https://github.com/bigdata-ufsc/petry-2020-marc
"""
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from tensorflow.keras import Model, callbacks, initializers, layers, optimizers, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config
from src.datasets import Dataset
from src.models.base import TrajectoryModel, log_end, log_start
from src.processors import GPSGeoHasher

LOG = logging.Logger(__name__)


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


class MARC(TrajectoryModel):
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
        self.y_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoders = []
        self.vocab_sizes = None
        self.timesteps = None
        self.num_classes = None
        self.learning_rate = None
        self.momentum = None
        self.batch_size = None
        self.patience = None
        self.classifier = None
        self.test_size = None
        self.trained_epochs = 0

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
        split = StratifiedShuffleSplit(
            test_size=test_size, n_splits=2, random_state=config.SEED
        ).split(ids, ids[self.dataset.label_column])
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
        if train:
            self.encoders = []
        for i, feature in enumerate(vocab_sizes):
            if train:
                encoder = OrdinalEncoder()
                encoder.fit(df[[feature]])
                self.encoders.append(encoder)
            encoder = self.encoders[i]
            feat_enc = pd.DataFrame(
                encoder.transform(df[[feature]]),
                columns=[feature],
            )
            encoded_features.append(feat_enc)
        df = df.reset_index().drop("index", errors="ignore", axis=1)
        df = df.drop(vocab_sizes.keys(), axis=1)
        df = df.drop(latlon_cols, axis=1)
        df = pd.concat([df, latlon_hashed, *encoded_features], axis=1)
        tids = df[self.dataset.trajectory_column].unique()
        tid_groups = df.groupby(self.dataset.trajectory_column).groups
        tid_dfs = [df.iloc[g] for g in tid_groups.values()]
        labels = np.array([tdf[self.dataset.label_column].values[0] for tdf in tid_dfs]).reshape(
            -1, 1
        )
        labels = self.y_encoder.transform(labels)
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
        self,
        epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        test_size: float = 0.2,
        patience: int = 10,
    ):
        """Train this model on the dataset."""
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.test_size = test_size
        self.patience = patience
        exp_name = f"{type(self).__name__}_{type(self.dataset).__name__}"
        start_time = log_start(LOG, exp_name, batch_size=batch_size)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        exp_path = Path(f"experiments/{exp_name}/{start_time_str}")
        self.test_size = test_size
        optimizer = optimizers.Adam(learning_rate, momentum)
        df = self.dataset.to_trajectories(min_points=10)
        train_set, _ = self.train_test_split(df, test_size=test_size)
        train_set, valid_set = self.train_test_split(train_set, test_size=0.1)
        x_train, y_train, _ = self.preprocess(train_set)
        x_val, y_val, _ = self.preprocess(valid_set, train=False)
        self.classifier = build_classifier(
            optimizer, self.vocab_sizes, self.timesteps, self.num_classes
        )

        history = self.classifier.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[
                callbacks.ModelCheckpoint(
                    exp_path / "checkpoints",
                    save_weights_only=True,
                    monitor="val_loss",
                    save_best_only=True,
                ),
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    verbose=1,
                    restore_best_weights=True,
                ),
                callbacks.CSVLogger(exp_path / "history.csv", separator=",", append=True),
                callbacks.TensorBoard(log_dir=exp_path / "tensorboard", histogram_freq=1),
            ],
        )
        self.trained_epochs += epochs
        log_end(LOG, exp_name, start_time)
        self.save(f"{exp_path}/saved_model")
        return history

    def predict(self, df: pd.DataFrame):
        """Predict the labels of a dataset."""
        x, _, _ = self.preprocess(df, train=False)
        return self.classifier.predict(x, batch_size=self.batch_size)

    def evaluate(self, df: pd.DataFrame):
        """Evaluate the model performance on the test set."""
        x, y, _ = self.preprocess(df, train=False)
        return self.classifier.evaluate(x, y, batch_size=self.batch_size)

    def save(self, save_path: os.PathLike):
        """Serialize the model to a directory on disk."""
        os.makedirs(save_path, exist_ok=True)
        save_path = Path(save_path)
        joblib.dump(self.dataset, save_path / "dataset.pkl")
        hparams = dict(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            geohash_dim=self.geohash_dim,
        )
        if self.trained_epochs > 0:
            joblib.dump(self.encoders, save_path / "encoders.pkl")
            joblib.dump(self.geohasher, save_path / "geohasher.pkl")
            joblib.dump(self.y_encoder, save_path / "y_encoder.pkl")
            self.classifier.save(save_path / "classifier_model")
            train_state = dict(
                trained_epochs=self.trained_epochs,
                patience=self.patience,
                timesteps=self.timesteps,
                vocab_sizes=self.vocab_sizes,
                num_classes=self.num_classes,
                test_size=self.test_size,
            )
            joblib.dump(train_state, save_path / "train_state.pkl")
        joblib.dump(hparams, save_path / "hparams.pkl")

        return self

    @classmethod
    def restore(cls, save_path: os.PathLike):
        """Restore the model from a checkpoint on disk."""
        save_path = Path(save_path)
        dataset = joblib.load(save_path / "dataset.pkl")
        model = cls(dataset)
        model.classifier = load_model(save_path / "classifier_model")
        model.encoders = joblib.load(save_path / "encoders.pkl")
        model.geohasher = joblib.load(save_path / "geohasher.pkl")
        model.y_encoder = joblib.load(save_path / "y_encoder.pkl")
        hparams = joblib.load(save_path / "hparams.pkl")
        train_state = joblib.load(save_path / "train_state.pkl")
        for key, val in {**hparams, **train_state}.items():
            setattr(model, key, val)
        return model
