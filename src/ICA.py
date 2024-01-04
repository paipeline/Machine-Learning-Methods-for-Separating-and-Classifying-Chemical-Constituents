
__author__      = "Chen Li & Pai Peng"
__copyright__   = "Copyright 2023"

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler

from Src.clusterPeaks import cluster_peaks
from Src.repeat import increase_m_and_repeat
from Src.dataPreprocess import process_data
from Src.maxPooling import max_pool


def perform_ica(after_clusters, n_components=2): # without considering the mid_points
    """
    This function performs Independent Component Analysis (ICA) on the provided data.

    :param after_clusters: The data to perform ICA on, in the form of a dictionary of clusters.
    :param n_components: The number of independent components to calculate. (default is 2)
    :return: Transformed mid-points and max-values after ICA
    """
    mid_points = []
    max_values = []
    all_transformed = []
    for col in after_clusters.keys():
        mid_points_cluster = [mid for mid, value in after_clusters[col]] 
        max_values_cluster = [value for mid, value in after_clusters[col]]

        mid_points.append(mid_points_cluster) # TODO 所有sample 都是共通的
        max_values.append(max_values_cluster)


    ica = FastICA(n_components=n_components, max_iter=1000, tol=0.01)
    max_values_transformed = ica.fit_transform(max_values)

    #TODO 未完成
    """
    d = len(mid_points[0])
    vector = np.zeros_like()

    for mid_points_cluster, max_values_cluster in zip(mid_points, max_values_transformed):
        for mid_point, value in zip(mid_points_cluster, max_values_cluster):
            mid_point = int(mid_point)
            vector[mid_point] = value
        """

    return max_values_transformed, mid_points




def plot_ica_results(max_values_transformed):
    """
    This function plots the results of Independent Component Analysis (ICA).

    :param mid_points_transformed: The transformed mid-points after ICA
    :param max_values_transformed: The transformed max-values after ICA
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))



    ax[0].scatter(max_values_transformed[:, 0], max_values_transformed[:, 1], color='blue')
    ax[0].set_title('ICA on Max Values')
    ax[0].set_xlabel('Component 1')
    ax[0].set_ylabel('Component 2')

    plt.tight_layout()
    plt.show()


def plot_ica_components(max_values_transformed):
    """
    This function plots each component separately after the Independent Component Analysis (ICA).

    :param max_values_transformed: The transformed max-values after ICA
    """
    num_components = max_values_transformed.shape[1]

    fig, axs = plt.subplots(num_components, 1, figsize=(10, 10))

    for i in range(num_components):
        axs[i].scatter(range(max_values_transformed.shape[0]), max_values_transformed[:, i], color='blue')
        axs[i].set_title(f'ICA Component {i+1}')
        axs[i].set_xlabel('Observation')
        axs[i].set_ylabel('Component Value')

    plt.tight_layout()
    plt.show()


def calculate_similarity(spectrum1, spectrum2):
    scaler = MinMaxScaler()
    spectrum1_scaled = scaler.fit_transform(spectrum1)
    spectrum2_scaled = scaler.fit_transform(spectrum2)

    similarity = np.dot(spectrum1_scaled, spectrum2_scaled)
    return similarity




# Test
if __name__ == "__main__":
        # file_path = "Data/SERS 1ANTH+1PYR.xlsx"  # Provide the actual file path
    file_path = "Data\SERS 2BaA+1ANTH.xlsx"
    m = 30  # Number of peaks to pick
    k = 3  # Window size for smoothing
    clustering_threshold = 100
    epoch = 1
    dct_cluster,data = increase_m_and_repeat(file_path, m,clustering_threshold ,k, epoch)
    data = process_data(file_path, m, k)
    col_names = data["y"]
    sample_num = len(col_names) - 1  # minus 1 because the first column is 'x'
    after_clusters = max_pool(data, dct_cluster)
    max_values_transformed = perform_ica(after_clusters, n_components=2)
    plot_ica_results(max_values_transformed)
    plot_ica_components(max_values_transformed)


