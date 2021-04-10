from tensorflow.keras import layers, Model


class LSTMAutoEncoder:
    def __init__(
        self,
        optimizer,
        loss,
        n_timesteps: int,
        n_features: int,
        lstm_units: int,
    ):
        """"""
        # Input tensor is ragged in the number of time steps
        inputs = layers.Input(shape=(n_timesteps, n_features))
        masking = layers.Masking(mask_value=-1)(inputs)
        encoder = layers.LSTM(lstm_units)(masking)
        hidden = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(lstm_units, return_sequences=True)(hidden)
        outputs = layers.TimeDistributed(layers.Dense(n_features))(decoder)
        encoder_model = Model(inputs=inputs, outputs=outputs)
        encoder_model.compile(optimizer=optimizer, loss=loss)
        self.model = encoder_model

    def fit(self, x, y, batch_size, epochs, initial_epoch=0, callbacks=None, verbose=2):
        """Train the model."""
        return self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=verbose,
        )