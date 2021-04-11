from tensorflow.keras import layers, Model


class LSTMAutoEncoder:
    def __init__(
        self,
        optimizer,
        loss,
        n_timesteps: int,
        n_features: int,
        lstm_units: int,
        mask_value: int = 0,
        activation="relu",
        metrics=None,
    ):
        """"""
        # Input tensor is ragged in the number of time steps
        inputs = layers.Input(shape=(n_timesteps, n_features))
        masking = layers.Masking(mask_value=mask_value)(inputs)
        # TODO: Decide whether to use Embedding or One-Hot encoding, or normalized raw GPS coordinates
        encoder = layers.LSTM(lstm_units, activation=activation)(masking)
        hidden = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(
            lstm_units, activation=activation, return_sequences=True
        )(hidden)
        outputs = layers.TimeDistributed(
            layers.Dense(n_features, activation="softmax")
        )(decoder)
        encoder_model = Model(inputs=inputs, outputs=outputs)
        encoder_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model = encoder_model

    def fit(
        self,
        x,
        y,
        batch_size,
        epochs,
        initial_epoch=0,
        validation_split=0.0,
        callbacks=None,
        verbose=2,
    ):
        """Train the model."""
        return self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
