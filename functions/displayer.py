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
    Plot first cycle and its overlapping windowed version.

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
    
    # Plot the first subplot (normal cycle)
    for i in range(n_sub_sequence):
        axs[1].plot(range(i*length_sub_sequence+cycle*n_sub_sequence, (i+1)*length_sub_sequence+cycle*n_sub_sequence), 
                    W[cycle*n_sub_sequence:(cycle+1)*n_sub_sequence][i, :, 0])
    axs[1].set_title('Overlapped due to windowing cycle')
    
    # Plot the second subplot (cycle with anomaly)
    axs[0].plot(df[cycle, :, 0])
    axs[0].set_title('Original cycle')

    plt.show()
