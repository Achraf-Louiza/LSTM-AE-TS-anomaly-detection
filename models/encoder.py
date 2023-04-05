import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

class Encoder:
    """
    Encoder class that creates a neural network model to encode input sequences into a fixed-length representation
    (latent variable) that can be used as input to a decoder network to generate a target sequence.
    """
    def __init__(self, length_sequence, n_features, n_latent):
        """ 
        Initializes the encoder model.

        Parameters
        ----------
        length_sequence : int
            The length of the input sequence.
        n_features : int
            The number of features in the input sequence.
        n_latent : int
            The size of the latent space (i.e., the dimensionality of the encoded representation).
        """
        inputs = Input(shape=(length_sequence, n_features))
        x = LSTM(32)(inputs)
        output = Dense(n_latent)(x)
        self.model = Model(inputs, output, name='Encoder-Model')
        