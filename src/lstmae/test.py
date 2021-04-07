# -*- coding: utf-8 -*-
"""
@authors: arshi, jeffmur, alexkyllo
"""
import os
import numpy as np
import tensorflow.keras as keras
import pickle
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from src.lstmae.clustering_layer import ClusteringLayer
from src.lstmae.plots import produce_plot
from tensorflow.keras.utils import plot_model
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans

# TODO: replace path hardcoding with configs
processed_dir = "D:/0/data_processed"
save_dir = "D:/0/project/model"
devel_n_file = "devel_n2.pkl"
test_n_file = "test_n2.pkl"
train_n_file = "train_n2.pkl"
devel_o_file = "devel_o2.pkl"
test_o_file = "test_o2.pkl"
train_o_file = "train_o2.pkl"

os.chdir(processed_dir)

print("Reading train and test sets.")

with open(devel_n_file, "rb") as input:
    devel_n = np.array(pickle.load(input), dtype=object)

with open(test_n_file, "rb") as input:
    test_n = np.array(pickle.load(input), dtype=object)

with open(train_n_file, "rb") as input:
    train_n = np.array(pickle.load(input), dtype=object)

with open(devel_o_file, "rb") as input:
    devel_o = np.array(pickle.load(input), dtype=object)

with open(test_o_file, "rb") as input:
    test_o = np.array(pickle.load(input), dtype=object)

with open(train_o_file, "rb") as input:
    train_o = np.array(pickle.load(input), dtype=object)

print("Reading files done.")


def target_distribution(
    q,
):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


x_v = np.append(devel_n, devel_o, axis=0)
print(np.shape(x_v[1, 0]))
x_ = np.append(train_n, train_o, axis=0)
# y_ = np.append(test_n, test_o, axis=0)
# y = np.append(np.zeros(len(train_n)), np.ones(len(train_o)), axis=0)
y = []
y_test = []
y_val = []
for i in range(len(train_n)):
    y.append([1, 0])
for i in range(len(train_o)):
    y.append([0, 1])
y = np.asarray(y)

for i in range(len(test_n)):
    y_test.append([1, 0])
for i in range(len(test_o)):
    y_test.append([0, 1])

for i in range(len(devel_n)):
    y_val.append([1, 0])
for i in range(len(devel_o)):
    y_val.append([0, 1])

y_val = np.asarray(y_val)

x_test = np.append(test_n, test_o, axis=0)

x_t = []
x = []
x_val = []

# fix X val
l = np.shape(x_v[:, 0])[0]
for i in range(l):
    x_val.append(x_v[i, 0])

x_val = np.asarray(x_val)
# fix X
l = np.shape(x_[:, 0])[0]

for i in range(l):
    x.append(x_[i, 0])

x = np.asarray(x)

l = np.shape(x_test[:, 0])[0]
for i in range(l):
    x_t.append(x_test[i, 0])

x_t = np.asarray(x_t)
print(np.shape(x_t))
print(np.shape(x))
print(np.shape(x_val))
x = np.einsum("kli->kil", x)
x_t = np.einsum("kli->kil", x_t)
x_val = np.einsum("kli->kil", x_val)
timesteps = np.shape(x)[1]  # 121
n_features = np.shape(x)[2]  # 1765

print("Creating model.")


class MyCallback(Callback):
    def __init__(self, alpha, beta, alphaIter, betaIter, epochIter):
        self.alpha = alpha
        self.beta = beta
        self.alphaIter = alphaIter
        self.betaIter = betaIter
        self.epochIter = epochIter

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.epochIter == 0) and (epoch != 0):
            K.set_value(self.alpha, self.alphaIter)
            K.set_value(self.beta, self.beta - K.get_value(self.betaIter))
            if self.beta < 0:
                K.set_value(self.beta, 0)


earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

val_set = (x_val, [y_val, x_val])

epochs = 500
batch_size = 128
# optimizer = SGD(lr=0.01, momentum=0.9)
# optimizer = Adam(0.1, beta_1=0.9, beta_2=0.001, amsgrad=True)
# optimizer = Adam(0.001, beta_1=0.9, amsgrad=True)
optimizer = "adam"
# if optimizer == 'adam':
#     opt_name = 'adam'
# else:
#     opt_name = 'sgd'
# optimizer = 'adam'
# optimizer = SGD(lr=1, momentum=0.9)
# opt_name = 'SGD'
opt_name = "adam"


model_dir = "D:/0/project/model"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
for gamma in [0, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 10]:
    # if gamma == 0:
    #     epochs = 300
    # else:
    #     epochs = 1000
    tf.keras.backend.clear_session()
    # for gamma in [2]:
    print("Setting Up Model for training")
    print(gamma)
    model_name = (
        "Gamma("
        + str(gamma)
        + ")-Optim("
        + opt_name
        + ")-BatchSize("
        + str(batch_size)
        + ")"
    )

    model = 0

    inputs = encoder = decoder = hidden = clustering = output = 0

    inputs = keras.Input(shape=(timesteps, n_features))
    encoder = LSTM(64, activation="tanh")(inputs)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(200, activation="relu")(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder_out = Dense(30, activation=None, name="encoder_out")(encoder)
    clustering = ClusteringLayer(n_clusters=2, name="clustering", alpha=0.05)(
        encoder_out
    )
    hidden = RepeatVector(timesteps, name="Hidden")(encoder_out)
    decoder = Dense(200, activation="relu")(hidden)
    decoder = Dense(64, activation="relu")(decoder)
    decoder = LSTM(64, activation="tanh", return_sequences=True)(decoder)
    output = TimeDistributed(Dense(n_features), name="decoder_out")(decoder)

    kmeans = KMeans(n_clusters=2, n_init=100)

    #
    encoder_model = Model(inputs=inputs, outputs=encoder_out)
    kmeans.fit(encoder_model.predict(x))

    model = Model(inputs=inputs, outputs=[clustering, output])

    clustering_model = Model(inputs=inputs, outputs=clustering)

    plot_model(model, show_shapes=True)
    model.summary()

    alpha = K.variable(0)
    beta = K.variable(1)
    alphaIter = K.variable(gamma)
    betaIter = K.variable(0)
    epochIter = K.variable(30)

    callbacks = MyCallback(alpha, beta, alphaIter, betaIter, epochIter)

    # model.compile(loss={'clustering': 'kld', 'decoder_out': 'mse'},
    #               loss_weights=[alpha, beta], optimizer=optimizer)
    model.compile(
        loss={"clustering": "kld", "decoder_out": "mse"},
        loss_weights=[gamma, 1],
        optimizer=optimizer,
    )
    model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])
    os.chdir("D:/0/project/model")
    plot_model(
        model,
        to_file="model1.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="LR",
        dpi=200,
    )
    print("Model compiled.")
    print("Training Starting:")
    train_history = model.fit(
        x,
        y={"clustering": y, "decoder_out": x},
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size,
        verbose=2,
    )

    q, _ = model.predict(x, verbose=0)
    q_t, _ = model.predict(x_t, verbose=0)
    p = target_distribution(q)
    y_pred = np.argmax(q, axis=1)
    y_arg = np.argmax(y, axis=1)
    y_pred_test = np.argmax(q_t, axis=1)
    y_arg_test = np.argmax(y_test, axis=1)
    # acc = np.sum(y_pred == y_arg).astype(np.float32) / y_pred.shape[0]
    # testAcc = np.sum(y_pred_test == y_arg_test).astype(np.float32) / y_pred_test.shape[0]
    acc = np.round(accuracy_score(y_arg, y_pred), 5)
    testAcc = np.round(accuracy_score(y_arg_test, y_pred_test), 5)
    nmi = np.round(normalized_mutual_info_score(y_arg, y_pred), 5)
    nmi_test = np.round(
        normalized_mutual_info_score(y_arg_test, y_pred_test), 5
    )
    ari = np.round(adjusted_rand_score(y_arg, y_pred), 5)
    ari_test = np.round(adjusted_rand_score(y_arg_test, y_pred_test), 5)
    print("====================")
    print("====================")
    print("====================")
    print("====================")
    print("Train accuracy")
    print(acc)
    print("Test accuracy")
    print(testAcc)
    print("NMI")
    print(nmi)
    print("ARI")
    print(ari)
    print("====================")
    print("====================")
    print("====================")
    print("====================")
    os.chdir(model_dir)
    pklName = model_name + ".pkl"

    saved_format = {
        "history": train_history.history,
        "gamma": gamma,
        "lr": K.eval(model.optimizer.lr),
        "batch": batch_size,
        "accuracy": acc,
        "nmi": nmi,
        "ari": ari,
        "nmi_test": nmi_test,
        "ari_test": ari_test,
    }
    # saved_format = [train_history.history, gamma, K.eval(model.optimizer.lr), batch_size]
    # with open(pklName, 'wb') as out_file:
    #     pickle.dump(train_history.history, out_file, pickle.HIGHEST_PROTOCOL)
    with open(pklName, "wb") as out_file:
        pickle.dump(saved_format, out_file, pickle.HIGHEST_PROTOCOL)

    print("Saving model.")
    save_name = "D:/0/project/model/" + model_name
    # model.save(save_name)

    produce_plot(model_name, train_history.history, gamma, testAcc)
