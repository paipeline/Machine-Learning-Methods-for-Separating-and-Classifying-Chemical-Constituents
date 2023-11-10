
__author__      = "Chen Li & Pai Peng"
__copyright__   = "Copyright 2023"

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from Src.baseline_removal import baseline_correct_compressed_data

def process_data(file_path, m, k):
    '''
    This method smooths the data and picks out all peaks

    :param file_path: path to the excel file
    :param m: number of peaks to pick
    :param k: window size for smoothing
    :return: a DataFrame with x, y, peaks
    '''
    data_file = pd.read_excel(file_path)

    # Getting X values
    # one to one correspondence between x and y

    x_val = data_file.iloc[:, 0]
    y_val_dic = {}

    total_peaks = np.zeros_like(data_file.iloc[:, 1]) # Initialize the total peaks vector with 0s

    for col in data_file.columns[1:]: # iterate through all columns except the first one
        # Smoothing using a sliding window
        y_val = data_file[col]
        #y_val = baseline_correct_compressed_data(y_val)
        y_smooth = y_val.rolling(window=k).mean() #smooth y would have shape (n-k+1,)

        # Normalize Data
        n_min = y_smooth.min()
        n_max = y_smooth.max()
        n_y_smooth = (y_smooth - n_min) / (n_max - n_min)

        # Find all peaks with prominence 0.02, by index
        idx, _ = find_peaks(n_y_smooth, prominence=0.02) #idx  is a list of indices of peaks
        binary_p_vector = np.zeros_like(n_y_smooth)
        binary_p_vector[idx] = 1  #assign 1 to where peaks are in x-axis

        # According to the methods on the paper, sum up all peaks over all recordings to find CPs
        total_peaks += binary_p_vector
        y_val_dic[col] = pd.Series(n_y_smooth)  # Convert to Series

    # Get the top M peaxks
    peak_indices = np.argsort(total_peaks)[::-1][:m]
    lx_m = np.zeros_like(total_peaks)
    lx_m[peak_indices] = total_peaks[peak_indices]

    # Create the DataFrame with Series objects
    data = {'x': x_val, 'y': y_val_dic, 'peaks': lx_m}

    return data, peak_indices
    
# Test
if __name__ == "__main__":
    file_path = "Data/SERS 1ANTH+1PYR.xlsx"  # Provide the actual file path
    m = 5  # Number of peaks to pick
    k = 3  # Window size for smoothing
    data = process_data(file_path, m, k)[0]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # then print your data
    print(data['peaks'])

    # data format:
    # data = {'x': x_val, 'y': pd.Series(y_val_dic), 'peaks': lx_m}

