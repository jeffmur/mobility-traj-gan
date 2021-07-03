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


def get_trajectories(df, tid_col="tid", label_col="label", geo_precision=8):
    keys = list(df.keys())
    vocab_size = {}
    keys.remove(tid_col)
    num_classes = len(set(df[label_col]))
    count_attr = 0
    lat_lon = False

    if "lat" in keys and "lon" in keys:
        keys.remove("lat")
        keys.remove("lon")
        lat_lon = True
        count_attr += geo_precision * 5

    for attr in keys:
        df[attr] = LabelEncoder().fit_transform(df[attr])
        vocab_size[attr] = max(df[attr]) + 1

        if attr != label_col:
            values = len(set(df[attr]))
            count_attr += values

    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    max_length = 0

    if lat_lon:
        x.append([])

    for tid in set(df[tid_col]):
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

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append("lat_lon")
        vocab_size["lat_lon"] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    x = [np.asarray(f) for f in x]
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()

    return (vocab_size, num_classes, max_length, x, y)


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
    vocab_size, num_classes, timesteps, x_train, y_train = get_trajectories(train_df)
    _, _, _, x_test, y_test = get_trajectories(test_df)
    model = build_classifier(vocab_size, timesteps, num_classes)
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=64,
        shuffle=True,
        epochs=epochs,
        verbose=0,
        callbacks=[],  # TODO: Early Stopping callback
    )
