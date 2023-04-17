import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Conv1DTranspose, Reshape, BatchNormalization, Dropout, UpSampling1D
from tensorflow.keras.models import Model


class LSTMDecoder:
    """
    Decoder class that creates an LSTM neural network model to decode a fixed-length representation (latent variable) into a
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
        x = Dense(length_sequence * n_features)(latent_inputs)
        x = Reshape((length_sequence, n_features))(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        x = LSTM(32, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = LSTM(32, return_sequences=True)(x)
        output = TimeDistributed(Dense(n_features))(x)
        self.model = Model(latent_inputs, output, name='Decoder-Model')
      
class ConvDecoder:
    """
    Decoder class that creates a Conv1D neural network model to decode a fixed-length representation (latent variable) into a
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
        
        x = Dense(int(length_sequence/8) * n_features)(latent_inputs)
        x = Reshape((int(length_sequence/8), n_features))(x)
        
        x = Conv1DTranspose(filters=128, kernel_size=5, activation='relu', strides=2, padding='same')(x)
        x = Conv1DTranspose(filters=128, kernel_size=5, activation='relu', strides=2, padding='same')(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        
        x = Conv1DTranspose(filters=64, kernel_size=7, activation='relu', padding='same', strides=2)(x)
        x = Conv1DTranspose(filters=64, kernel_size=7, activation='relu', padding='same', strides=1)(x)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        
        output = Conv1DTranspose(filters=1, kernel_size=3, activation='relu', padding='same', strides=1)(x)
        
        self.model = Model(latent_inputs, output, name='Decoder-Model')
