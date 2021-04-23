from tensorflow.keras import layers, Model


class LSTMBottleneck(layers.Layer):
    """https://stackoverflow.com/a/59313862/1834892"""

    def __init__(self, lstm_units, time_steps, **kwargs):
        self.lstm_units = lstm_units
        self.time_steps = time_steps
        self.lstm_layer = layers.LSTM(lstm_units, return_sequences=False)
        self.repeat_layer = layers.RepeatVector(time_steps)
        super().__init__(**kwargs)

    def call(self, inputs):
        # just call the two initialized layers
        return self.repeat_layer(self.lstm_layer(inputs))

    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask


class LSTMAutoEncoder:
    def __init__(
        self,
        optimizer,
        loss,
        n_timesteps: int,
        n_feature_values: int,
        embedding_length: int,
        lstm_units: int,
        activation="tanh",
        metrics=None,
    ):
        """"""
        inputs = layers.Input(shape=(n_timesteps,))
        embedding = layers.Embedding(
            n_feature_values,
            embedding_length,
            input_length=n_timesteps,
            mask_zero=True,
        )(inputs)
        # TODO: Flatten or concatenate the embeddings?
        # masking = layers.Masking(mask_value=mask_value)(inputs)
        # encoder = LSTMBottleneck(lstm_units, n_timesteps)(masking)
        # encoder = LSTMBottleneck(

        encoder = layers.LSTM(lstm_units, input_shape=(n_timesteps, 1))(
            embedding
        )
        hidden = layers.RepeatVector(n_timesteps)(encoder)
        decoder = layers.LSTM(
            lstm_units,
            activation=activation,
            return_sequences=True,
        )(hidden)
        outputs = layers.TimeDistributed(
            layers.Dense(n_feature_values, activation="softmax")
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
