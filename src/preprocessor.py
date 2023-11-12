import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Tuple, List, Dict


class Preprocessor:

    def __init__(self, file_path: str, m: int, threshold: int, mat_iteration: int, windows_size: int):
        """
        Simple fetch all parameters.

        :param file_path: The file points to the data (directory)
        :param m: The required number of clusters
        :param threshold: Clustering threshold
        :param mat_iteration: Max number of iteration if num of clusters never reaches m
        :param windows_size: the size of the rolling window
        """
        self.file_path = file_path
        self.m = m
        self.threshold = threshold
        self.max_it = mat_iteration
        self.window = windows_size

    @staticmethod
    def raw_data_process(file_path: str, m: int, window_size: int) -> Tuple[Dict[str, pd.Series], List[int]]:
        """
        This method takes raw Ramen Shift data from pre-given files, smooth them using a specific
        size rolling mean window, and extract peaks with prominence of 0.02, after all data is
        normalized to [0, 1].
        """

        # Load the Excel file into a DataFrame

        data_file = pd.read_excel(file_path)

        # Extract the x-values (assuming they are in the first column)
        x_values = data_file.iloc[:, 0]
        # y_value consists of multiple different testing samples
        y_values_dict = {}

        # We need to count peaks of prominence of 0.02, and sum over all cols
        total_peaks = np.zeros_like(x_values)

        # Loop over each column (excluding the first one, which has x-values)
        for col in data_file.columns[1:]:
            y_values = data_file[col]
            # Smooth the y-values using a rolling window
            y_smooth = y_values.rolling(window=window_size).mean()
            # Normalize the smoothed y-values
            normalized_y_smooth = (y_smooth - y_smooth.min()) / (y_smooth.max() - y_smooth.min())
            # Detect peaks in the normalized y-values, and save them into a binary vector
            peak_indices, _ = find_peaks(normalized_y_smooth, prominence=0.02)
            binary_peak_vector = np.zeros_like(normalized_y_smooth)
            binary_peak_vector[peak_indices] = 1

            # Sum over all cols and save smoothed y-value
            total_peaks += binary_peak_vector
            y_values_dict[col] = normalized_y_smooth

        # Extract the top 'm' prominent peaks
        top_peaks = total_peaks.argsort()[::-1][:m]
        extracted_peaks = np.zeros_like(total_peaks)
        extracted_peaks[top_peaks] = total_peaks[top_peaks]

        return {'x': x_values, 'y': y_values_dict, 'peaks': extracted_peaks}, top_peaks

    @staticmethod
    def cluster_peaks(data: dict, threshold: int, indices: list) -> dict:
        """
        Clusters peaks that are within the given threshold.
        """
        sorted_indices = np.sort(indices)
        clusters = []
        current_cluster = [sorted_indices[0]]

        # Group indices that are close to each other into clusters
        for index in sorted_indices[1:]:
            if index - current_cluster[0] <= threshold:
                current_cluster.append(index)
            else:
                clusters.append(current_cluster)
                current_cluster = [index]
        clusters.append(current_cluster)

        cluster_data = {}
        # For each cluster, compute how many peaks it contains
        for i, cluster in enumerate(clusters):
            cluster_data[i] = {
                'indices': cluster,
                'count': sum(data['peaks'][index] for index in cluster)
            }

        data['peaks'] = cluster_data

        return data

    def max_pool(self, data: dict, cluster_data: dict) -> dict:
        """
        Perform max pooling on the data to get the maximum value in each cluster.
        """
        mid_points = {}
        pooled_vectors = {}

        # For each cluster, compute its middle point, start, and end
        for label, info in cluster_data.items():
            indices = info['indices']
            start, end = min(indices), max(indices)
            mid_point = (start + end) // 2
            mid_points[label] = [mid_point, start, end]

        # For each y-value recording, get the maximum value in each cluster
        for col, data_recording in data['y'].items():
            pooled_vector = [
                (mid_points[label][0], max(data_recording[start:end + 1]))
                for label, (mid_point, start, end) in mid_points.items()
            ]
            pooled_vectors[col] = pooled_vector

        return pooled_vectors

    def increase_m_and_repeat(self, file_path: str, m: int, threshold: int, window_size: int,
                              max_iterations: int,
                              increase_rate: float = 0.2) -> Tuple[Dict[int, Dict[str, int]], Dict[str, pd.Series]]:
        """
        Iteratively increase the m to extract and process the data until the
        number of clusters exceed m or the maximum number of iterations is reached.
        """
        # Fetch Data
        data, peak_indices = self.raw_data_process(file_path, m, window_size)
        data = self.cluster_peaks(data, threshold, peak_indices)
        m_cluster = len(data['peaks'])
        m_star = m
        iterations = 1

        # Iteration till we have equal or more than m clusters, or max_it reaches.
        while m_cluster < m and iterations < max_iterations:
            m_star += int(m * increase_rate)
            data, peak_indices = self.raw_data_process(file_path, m_star, window_size)
            data = self.cluster_peaks(data, threshold, peak_indices)
            m_cluster = len(data['peaks'])
            iterations += 1

        # Sort the clusters by their count and keep only the top ones
        sorted_clusters = sorted(data['peaks'].items(), key=lambda x: x[1]['count'], reverse=True)
        top_clusters = dict(sorted_clusters[:m])

        # print(type(top_clusters))
        return top_clusters, data

    def run(self) -> dict:
        clusters, data = self.increase_m_and_repeat(self.file_path, self.m, self.threshold, self.window, self.max_it)
        return self.max_pool(data=data, cluster_data=clusters)
