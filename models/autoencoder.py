import tensorflow as tf
from tensorflow.keras.models import Model


class Autoencoder:
    """
    Autoencoder class that creates a neural network model to encode and decode input sequences.
    """
    def __init__(self, encoder_model, decoder_model):
        """
        Initializes the autoencoder model.

        Parameters
        ----------
        encoder_model : tensorflow.keras.models.Model
            The encoder model to use for encoding input sequences.
        decoder_model : tensorflow.keras.models.Model
            The decoder model to use for decoding latent variables into output sequences.
        """
        inputs = encoder_model.inputs
        latent = encoder_model(inputs)
        outputs = decoder_model(latent)
        self.model = Model(inputs, outputs, name='Autoencoder-Model')
