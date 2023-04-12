import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_cycle(df, col_anomaly='isAnomaly', col_x='value'):
    """
    Plot a random normal cycle and a random cycle with anomaly.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the cycles to plot.
    col_anomaly : str, optional
        Column name indicating whether a cycle has anomaly, by default 'isAnomaly'.
    col_x : str, optional
        Column name indicating the feature to plot, by default 'value'.
    """
    # Select cycles without anomaly
    df_ok = df[df[col_anomaly] == 1]
    id_ok = random.choice(df_ok.id.unique())
    y_ok = df_ok[df_ok.id == id_ok][col_x].tolist()

    # Select cycles with anomaly
    df_n_ok = df[df[col_anomaly] == -1]
    id_n_ok = random.choice(df_n_ok.id.unique())
    y_n_ok = df_n_ok[df_n_ok.id == id_n_ok][col_x].tolist()

    # Define x axis for plotting
    x = range(len(y_ok))

    # Create subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot the first subplot (normal cycle)
    axs[0].plot(x, y_ok)
    axs[0].set_title('Normal cycle')

    # Plot the second subplot (cycle with anomaly)
    axs[1].plot(x, y_n_ok)
    axs[1].set_title('Anomaly')

    # Add a title to the figure
    fig.suptitle('Randomly visualize normal & abnormal cycles')

    # Show the plot
    plt.show()


def plot_windowed_cycle(df, W, cycle=0, gap=4):
    """
    Plots first cycle and its overlapping windowed version.

    Parameters
    ----------
    df : numpy.array
        Dataframe containing the original cycles.
    W : numpy.array
        Dataframe containing the windowed data.
    gap : int, optional
        Number of timestamp to jump when windowing a cycle, by default 4.
    """
    # Original cycle length
    length_sequence = df.shape[1]
    
    # Windowed data shape
    length_sub_sequence = W.shape[1]
    n_sub_sequence = (length_sequence-length_sub_sequence) // gap + 1
    
    # Create subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # Plot the original cycle 
    for i in range(n_sub_sequence):
        axs[1].plot(range(i*length_sub_sequence+cycle*n_sub_sequence, (i+1)*length_sub_sequence+cycle*n_sub_sequence), 
                    W[cycle*n_sub_sequence:(cycle+1)*n_sub_sequence][i, :, 0])
    axs[1].set_title('Overlapped due to windowing cycle')
    
    # Plot the windowed cycle
    axs[0].plot(df[cycle, :, 0])
    axs[0].set_title('Original cycle')

    plt.show()

def plot_reconstruction(original, reconstruction, cycle):
    """
    Plots reconstructed cycle in comparison to the original one.

    Parameters
    ----------
    original : np.Array
        A 3D array containing original multivariate cycles
    reconstruction : np.Array
        A 3D array containing reconstructed multivariate cycles
    cycle : int
        an integer representing the index of the cycle to plot
        
    """
    
    fig = plt.figure(figsize=(10, 4))
    plt.plot(original[cycle], color='blue', label='Original cycle')
    plt.plot(reconstruction[cycle], color='red', label='Reconstructed cycle')
    plt.title('Reconstructed cycle over original using LSTM-AE')
    plt.legend()
    plt.show()

def plot_mae_per_cycle(mae_per_cycle, qualitative_var):
    """
    Plot a bar chart of the average mean absolute error (MAE) per cycle,
    colored by a qualitative variable.

    Parameters
    ----------
    mae_per_cycle : array_like
        A 1D array of average MAE per cycle.
    qualitative_var : array_like
        A 1D array of qualitative variable with length equal to the number of cycles.
        This variable is used to color the bars based on the category of the data.
    """
    # Define the x-axis tick labels as the cycle numbers (starting from 1)
    xticks = np.arange(len(mae_per_cycle)) + 1

    # Define x axis
    x = np.array(range(len(qualitative_var)))
    
    # Differentiate between normal cycles and anomalies
    x_normal = x[qualitative_var==1]
    x_anomaly = x[qualitative_var==-1]
    y_normal = mae_per_cycle[qualitative_var==1]
    y_anomaly = mae_per_cycle[qualitative_var==-1]
    
    # Create the bar plot with labeled axes
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x_normal, y_normal, label='normal')
    ax.bar(x_anomaly, y_anomaly, label='anomaly')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Average MAE')
    ax.legend()
    plt.show()
    
def plot_loss(history):
    """
    Plots the train and validation loss functions for each epoch in the given Keras history object.

    Parameters
    ----------
    history : keras.callbacks.History
        A history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """
    # Get the train and validation loss values for each epoch
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the train and validation loss functions
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()