import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


class Decoder:
    """
    Decoder class that creates a neural network model to decode a fixed-length representation (latent variable) into a
    target sequence.
    """
    def __init__(self, length_sequence, n_features, n_latent):
        """
        Initializes the decoder model.

        Parameters
        ----------
        length_sequence : int
            The length of the target sequence.
        n_features : int
            The number of features in the target sequence.
        n_latent : int
            The size of the latent space (i.e., the dimensionality of the encoded representation).
        """
        latent_inputs = Input(shape=(n_latent,))
        x = RepeatVector(length_sequence)(latent_inputs)
        x = LSTM(32, return_sequences=True)(x)
        x = LSTM(32, return_sequences=True)(x)
        output = TimeDistributed(Dense(n_features))(x)
        self.model = Model(latent_inputs, output, name='Decoder-Model')
