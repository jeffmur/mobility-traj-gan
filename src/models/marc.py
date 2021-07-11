"""marc.py

TensorFlow 2 reimplementation of the Multi-Aspect Trajectory Classifier model
from: https://github.com/bigdata-ufsc/petry-2020-marc
"""
import geohash2 as gh
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import Model, initializers, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


def get_trajectories(df_train, df_test, tid_col="tid", label_col="label", geo_precision=8):
    df = df_train.copy().append(df_test)
    tids_train = df_train[tid_col].unique()
    keys = list(df_train.keys())
    vocab_size = {}
    keys.remove(tid_col)
    num_classes = df[label_col].nunique()
    lat_lon = False

    if "lat" in keys and "lon" in keys:
        keys.remove("lat")
        keys.remove("lon")
        lat_lon = True

    for attr in keys:
        # This is a bad practice because encoder is being fitted on the test set
        lab_enc = LabelEncoder()
        df[attr] = lab_enc.fit_transform(df[attr])

        if attr != label_col:
            vocab_size[attr] = max(df[attr]) + 1

    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    idx_train = []
    idx_test = []
    max_length = 0

    if lat_lon:
        x.append([])

    for idx, tid in enumerate(df[tid_col].unique()):
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, keys].values)

        for i, feature in enumerate(features):
            x[i].append(feature)

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj["lat"].values[i]
                lon = traj["lon"].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))
            x[-1].append(loc_list)

        label = traj[label_col].iloc[0]
        y.append(label)

        if tid in tids_train:
            idx_train.append(idx)
        else:
            idx_test.append(idx)

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append("lat_lon")
        vocab_size["lat_lon"] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    x = [np.asarray(f) for f in x]
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()

    x_train = np.asarray([f[idx_train] for f in x])
    y_train = y[idx_train]
    x_test = np.asarray([f[idx_test] for f in x])
    y_test = y[idx_test]

    x_train_pad = [pad_sequences(f, max_length, padding="pre") for f in x_train]
    x_test_pad = [pad_sequences(f, max_length, padding="pre") for f in x_test]

    return (vocab_size, num_classes, max_length, x_train_pad, y_train, x_test_pad, y_test)


def build_classifier(
    vocab_size, timesteps, num_classes, embedder_size=100, hidden_units=100, dropout=0.5
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
    opt = optimizers.Adam(lr=0.001)

    classifier.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["acc", "top_k_categorical_accuracy"],
    )
    return classifier


def train(train_df, test_df, epochs):
    """Train the MARC trajectory classifier model."""
    vocab_size, num_classes, timesteps, x_train, y_train, x_test, y_test = get_trajectories(
        train_df, test_df
    )
    model = build_classifier(vocab_size, timesteps, num_classes)
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=64,
        shuffle=True,
        epochs=epochs,
        callbacks=[],  # TODO: Early Stopping callback
    )


class MARC:
    """An LSTM-based trajectory classifier network.

    Based on: May Petry, L., Leite Da Silva, C., Esuli, A., Renso, C., and Bogorny, V. (2020).
    MARC: a robust method for multiple-aspect trajectory classification via space, time,
    and semantic embeddings.
    International Journal of Geographical Information Science, 34(7), 1428-1450.
    """
