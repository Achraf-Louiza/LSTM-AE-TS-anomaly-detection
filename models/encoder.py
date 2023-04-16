import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Conv1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.models import Model

class LSTMEncoder:
    """
    Encoder class that creates an LSTM neural network model to encode input sequences into a fixed-length representation
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
        x = LSTM(32, return_sequences=True)(inputs)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        output = Dense(n_latent)(x)
        self.model = Model(inputs, output, name='Encoder-Model')
        
class ConvEncoder:
    """
    Encoder class that creates a Conv1D neural network model to encode input sequences into a fixed-length representation
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
        
        x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
        x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        
        x = Conv1D(filters=32, kernel_size=8, activation='relu', padding='same')(inputs)
        x = Conv1D(filters=32, kernel_size=8, activation='relu', padding='same')(inputs)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        
        output = Dense(n_latent)(x)
        
        self.model = Model(inputs, output, name='Encoder-Model')