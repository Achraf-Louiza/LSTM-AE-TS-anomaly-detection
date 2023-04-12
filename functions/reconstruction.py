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
            cycle = np.concatenate([cycle, np.mean(np.stack([S[k, (j-k)*gap:(j-k+1)*gap, :] for k in range(j-1, j+1)], axis=0), axis=0)], axis=0)  # Average over present cycles in each overlapping gap
        cycle = np.concatenate([cycle, S[n_sub_sequences-1, :, :]], axis=0)  # Add the final sub-sequence to the cycle
        cycles.append(cycle)  # Add the reconstructed cycle to the list of cycles
    cycles = np.stack(cycles, axis=0)  # Stack the list of cycles into a 3D numpy array
    return cycles  # Return the reconstructed cycles

def calc_avg_mae_per_cycle(original, reconstruction):
    """
    Calculate the average mean absolute error (MAE) per cycle between a set of
    original 3D sequences and their reconstructions.

    Parameters
    ----------
    original : array_like
        A 3D array of original sequences with shape (num_seqs, seq_len, num_features).
    reconstruction : array_like
        A 3D array of reconstructed sequences with shape (num_seqs, seq_len, num_features).

    Returns
    -------
    mae_per_cycle : ndarray
        A 1D array of average MAE per cycle with length equal to seq_len.
    """
    mae_per_cycle = np.mean(np.abs(original - reconstruction), axis=(1, 2))
    return mae_per_cycle
