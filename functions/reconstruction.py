import pandas as pd
import numpy as np

def reconstruct_cycles(W, length, sub_length=12, gap=4):
    """
    Reconstruct cycles from a given 3D numpy array.
    
    Parameters
    ----------
    W : numpy.ndarray
        A 3D numpy array of shape (n, m, k), where n, m, and k are the number of cycles, 
        sub-sequences, and features respectively. 
    length : int
        Length of the original sequence from which W was generated.
    sub_length : int, optional
        Length of each sub-sequence used to generate W. Default is 12.
    gap : int, optional
        Gap between sub-sequences used to generate W. Default is 4.
    
    Returns
    -------
    numpy.ndarray
        A 3D numpy array of shape (n, m, k) representing the reconstructed cycles
    """
    shape = W.shape
    n_sub_sequences = (length - sub_length) // gap + 1  # Calculate the number of sub-sequences needed to reconstruct the cycles
    cycles = []  # Initialize an empty list to store the reconstructed cycles
    for i in range(0, shape[0], n_sub_sequences):
        # Extract a subset of the input array that contains n_sub_sequences sub-sequences, starting from index i
        S = W[i:i+n_sub_sequences, :, :]
        cycle = S[0, :gap, :]  # Initialize the first part of the cycle with the first sub-sequence
        for j in range(1, n_sub_sequences-1):
            cycle = np.concatenate([cycle, S[j, :gap, :]], axis=0)  # Concatenate the remaining sub-sequences to the cycle
        cycle = np.concatenate([cycle, S[n_sub_sequences-1, :, :]], axis=0)  # Add the final sub-sequence to the cycle
        cycles.append(cycle)  # Add the reconstructed cycle to the list of cycles
    cycles = np.stack(cycles, axis=0)  # Stack the list of cycles into a 3D numpy array
    return cycles  # Return the reconstructed cycles