
__author__      = "Chen Li & Pai Peng"
__copyright__   = "Copyright 2023"

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from dataPreprocess import process_data
from repeat import increase_m_and_repeat
from maxPooling import max_pool
def visualize_data(data, col_names, sample_num, after_clusters):
    '''
    This function plots each series with its peaks and cluster boundaries

    :param data: data dictionary
    :param col_names: List of names of the columns to plot
    :param sample_num: Number of samples
    :param after_clusters: dictionary of clusters
    '''
    plt.figure(figsize=(10, 6))

    colors = cm.rainbow(np.linspace(0, 1, sample_num))

    # Plot each column
    for col, color in zip(range(1, sample_num), colors):
        plt.plot(data['x'], data['y'][col], color=color, label=f"col: {col} smoothed y")


    for col, color in zip(range(1, sample_num), colors):
        peaks = data['peaks'] > 0
        plt.plot(data['x'][peaks], data['y'][col][peaks], "x", color=color)
    
    # Plot each cluster
    for cluster, color in zip(after_clusters.keys(), colors):
        mid_points_and_values = after_clusters[cluster]
        mid_points = [point for point, value in mid_points_and_values]
        max_values = [value for point, value in mid_points_and_values]

        plt.plot(data['x'][mid_points], max_values, "o", color=color)

        # Plot cluster boundaries
        min_index, max_index = min(mid_points), max(mid_points)
        plt.axvline(x=data['x'][min_index], color=color, linestyle='--')  # start of cluster
        plt.axvline(x=data['x'][max_index], color=color, linestyle='--')  # end of cluster


    # set a -- line in y = 0
    plt.plot(data['x'], np.zeros(len(data['x'])), "--", color="black", label="y = 0")
    plt.title("data Visualization")
    plt.xlabel("Raman Shift (cm-1)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.show()


# Test
if __name__ == "__main__":
    # file_path = "data/SERS 1ANTH+1PYR.xlsx"  # Provide the actual file path
    file_path = "data\SERS 2BaA+1ANTH.xlsx"
    m = 30  # Number of peaks to pick
    k = 3  # Window size for smoothing
    clustering_threshold = 100
    epoch = 1
    dct_cluster,data = increase_m_and_repeat(file_path, m,clustering_threshold ,k, epoch)
    data = process_data(file_path, m, k) #fix CANNOT PROCESS the returned data from increase_m_and_repeat
    col_names = data["y"]
    sample_num = len(col_names) - 1  # minus 1 because the first column is 'x'
    after_clusters = max_pool(data, dct_cluster)
    visualize_data(data, col_names, sample_num,after_clusters)

    #data format:
    #data = {'x': x, 'y': y, 'peaks': peaks, 'smoothed_y': smoothed_y,
    # "clusters": {"indices","count"}}