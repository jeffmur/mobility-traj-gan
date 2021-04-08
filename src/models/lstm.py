from tensorflow.keras import layers, Model
from tensorflow.keras import backend


class LSTMAutoEncoder:
    def __init__(
        self,
        optimizer,
        loss,
        n_subjects: int,
        n_timesteps: int,
        n_features: int,
        lstm_units: int,
    ):
        """"""
        # Input tensor is ragged in the number of time steps
        inputs = layers.Input(shape=(n_subjects, n_timesteps, n_features))
        encoder = layers.LSTM(lstm_units)(inputs)
        hidden = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(lstm_units, return_sequences=True)(hidden)
        outputs = layers.TimeDistributed(layers.Dense(n_subjects, n_features))(decoder)
        encoder_model = Model(inputs=inputs, outputs=outputs)
        encoder_model.compile(optimizer=optimizer, loss=loss)
        self.model = encoder_model

    def fit(self, x, y, epochs, initial_epoch=0, callbacks=None, verbose=2):
        """Train the model."""
        return self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=verbose,
        )