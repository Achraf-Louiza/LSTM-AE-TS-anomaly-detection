import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class DataProcessor:
    """
    A class for preprocessing data for use in a machine learning model.
    """

    def __init__(self, x_train, x_test, col_names, id='id'):
        """
        Initializes a new DataProcessor object.

        Parameters:
            x_train (pandas.DataFrame): The training data.
            x_test (pandas.DataFrame): The testing data.
            col_names (list): The names of the columns to scale.
            id (str): The name of the id column (default is 'id').
        """
        self.x_train = x_train
        self.x_test = x_test
        self.col_names = col_names
        self.id = id
    
    def scale_data(self):
        """
        Scales the training and testing data using StandardScaler.
        """
        # Create a new StandardScaler object
        self.scaler = StandardScaler()
        
        # Fit the scaler to the training data using the specified column names
        self.scaler.fit(self.x_train[self.col_names])
        
        # Scale the training data
        self.s_train = self.scaler.transform(self.x_train[self.col_names])
        self.s_train = pd.DataFrame(self.s_train, columns=self.col_names)
        self.s_train[self.id] = self.x_train[self.id]
        
        # Scale the testing data
        self.s_test = self.scaler.transform(self.x_test[self.col_names])
        self.s_test = pd.DataFrame(self.s_test, columns=self.col_names)
        self.s_test[self.id] = self.x_test[self.id]
    
    def reshape_data(self):
        """
        Reshapes the scaled data.
        """
        # Group the scaled training data by the specified id column and convert to a list of numpy arrays
        self.s_train = np.stack(self.s_train.groupby([self.id]).apply(lambda A: A[self.col_names].to_numpy()))
        
        # Group the scaled testing data by the specified id column and convert to a list of numpy arrays
        self.s_test = np.stack(self.s_test.groupby([self.id]).apply(lambda A: A[self.col_names].to_numpy()))
        
    def windowing(self, length_sub_sequence = 12, gap=4):
        """
        Splits the time series data into subsequences with overlapping windows.

        Parameters:
            length_sub_sequence (int): The length of each subsequence (default is 10).
            gap (int): The gap between each subsequence (default is 4).

        Returns:
            None
        """
        # Get the initial shapes of the training and testing data
        train_shape = self.s_train.shape
        test_shape = self.s_test.shape
        length_cycle = train_shape[1]
        n_features = train_shape[2]
        n_seq_train = train_shape[0]
        n_seq_test = test_shape[0]
        
        # Initialize empty lists to hold the new training and testing data
        train = []
        test = []
        
        # Loop over each possible window in the time series data
        for t in range(0, length_cycle-length_sub_sequence+1, gap):
            # Get the subsequence of the training data and append it to the list
            train_seg = self.s_train[:, t:t + length_sub_sequence, :]
            train.append(train_seg)
            
            # Get the subsequence of the testing data and append it to the list
            test_seg = self.s_test[:, t:t + length_sub_sequence, :]
            test.append(test_seg)
        
        # Stack the list of training and testing data to create a new array with shape 
        # (number_subsequences, length_subsequence, n_features)
        self.w_train = np.stack(train, axis=1)
        shapes = self.w_train.shape
        self.w_train = np.reshape(self.w_train, (shapes[0]*shapes[1], shapes[2], shapes[3]))
        self.w_test = np.stack(test, axis=1)
        shapes = self.w_test.shape 
        self.w_test = np.reshape(self.w_test, (shapes[0]*shapes[1], shapes[2], shapes[3]))

        
        
