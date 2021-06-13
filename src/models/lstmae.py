"""models/lstmae.py
"""
from tensorflow.keras import Model, layers


class LSTMAutoEncoder:
    """An LSTM Autoencoder model. Later this can be expanded to include prediction,
    clustering, and reidentification outputs."""

    def __init__(
        self,
        optimizer,
        loss,
        n_timesteps: int,
        n_feature_values: int,
        lstm_units: int,
        metrics=None,
    ):
        """Constructor"""
        inputs = layers.Input(shape=(n_timesteps, 1))
        encoder_1 = layers.LSTM(lstm_units, return_sequences=True)(inputs)
        encoder_2 = layers.LSTM(lstm_units // 2)(encoder_1)
        hidden = layers.RepeatVector(n_timesteps)(encoder_2)
        decoder_1 = layers.LSTM(
            lstm_units // 2,
            return_sequences=True,
        )(hidden)
        decoder_2 = layers.LSTM(lstm_units, return_sequences=True)(decoder_1)
        outputs = layers.TimeDistributed(layers.Dense(n_feature_values, activation="softmax"))(
            decoder_2
        )
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

    def predict(self, x):
        return self.model.predict(x)
