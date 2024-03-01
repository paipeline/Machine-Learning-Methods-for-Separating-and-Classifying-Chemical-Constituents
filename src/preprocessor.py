import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Tuple, List, Dict
from BaselineRemoval import BaselineRemoval


class Preprocessor:

    def __init__(self, file_path: str, m: int, threshold: int, mat_iteration: int, window: int, prominence:float, c = False, baseline_removal = False):
        """
        Simple fetch all parameters.

        :param file_path: The file points to the data (directory)
        :param m: The required number of clusters
        :param threshold: Clustering threshold
        :param mat_iteration: Max number of iteration if num of clusters never reaches m
        :param window: the size of the rolling window
        """
        self.prominence = prominence
        self.file_path = file_path
        self.m = m
        self.threshold = threshold
        self.max_it = mat_iteration
        self.window = window
        self.baseline_removal = baseline_removal
    def load_raw_data(self,file_path: str) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        Load raw data from an Excel file.
        """
        data_file = pd.read_excel(file_path)
        x_values = data_file.iloc[:, 0]
        y_values_dict = {col: data_file[col] for col in data_file.columns[1:]}
        if self.baseline_removal:
            for col, y_values in y_values_dict.items():
                baseObj = BaselineRemoval(y_values)
                y_values_dict[col] = baseObj.ZhangFit()
            print("*Baseline-window-normalize-peaks")
        else:
            print("*RAW-window-normalize-peaks")

            
        return x_values, y_values_dict

    def apply_rolling_window(self, y_values_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Apply a rolling window to smooth the y-values.
        """
        print("raw-*WINDOW-normalize-peaks")
        smoothed_dict = {}
        for col, y_values in y_values_dict.items():
            if isinstance(y_values, pd.Series):
                smoothed_dict[col] = y_values.rolling(window=self.window).mean()
            else:
                # If it's not a Pandas Series, you can convert it to one
                smoothed_dict[col] = pd.Series(y_values).rolling(window=self.window).mean()
        return smoothed_dict 
    def normalize_data(self,y_values_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Normalize the smoothed y-values.
        """
        print("raw-window-*NORMALIZE-peaks")
        return {col: (y_values - y_values.min()) / (y_values.max() - y_values.min()) 
                for col, y_values in y_values_dict.items()}

    def find_peaks(self,y_values_dict: Dict[str, pd.Series], x_values: pd.Series, m: int) -> Tuple[Dict[str, pd.Series], List[int]]:
        """
        Detect peaks in the normalized y-values.
        """
        total_peaks = np.zeros_like(x_values)
        for col, y_values in y_values_dict.items():
            peak_indices, _ = find_peaks(y_values, prominence=self.prominence)
            binary_peak_vector = np.zeros_like(y_values)
            binary_peak_vector[peak_indices] = 1
            total_peaks += binary_peak_vector

        top_peaks = total_peaks.argsort()[::-1][:m]
        extracted_peaks = np.zeros_like(total_peaks)
        extracted_peaks[top_peaks] = total_peaks[top_peaks]

        data_with_peaks = {
            'x': x_values,
            'y': y_values_dict,
            'peaks': extracted_peaks
        }
        print("raw-window-normalize-*PEAKS")
        print(" ")
        return data_with_peaks, top_peaks

    
    def raw_data_process(self,m_star) -> Tuple[Dict[str, pd.Series], List[int]]:
        """
        This method takes raw Ramen Shift data from pre-given files, smooth them using a specific
        size rolling mean window, and extract peaks with prominence (of default 0.02), after all data is
        normalized to [0, 1].
        """
        x_values, y_values_dict = self.load_raw_data(self.file_path)
        y_smoothed = self.apply_rolling_window(y_values_dict)
        y_normalized = self.normalize_data(y_smoothed)
        return self.find_peaks(y_normalized, x_values, m_star)
    
    
    def cluster_peaks(self, data: dict, threshold: int, indices: list) -> dict:
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

    def increase_m_and_repeat(self, file_path: str, m: int, threshold: int, window: int,
                              max_iterations: int,
                              increase_rate: float = 0.2) -> Tuple[Dict[int, Dict[str, int]], Dict[str, pd.Series]]:
        """
        Iteratively increase the m to extract and process the data until the
        number of clusters exceed m or the maximum number of iterations is reached.
        """
        # Fetch Data
        print("--init--")
        data, peak_indices = self.raw_data_process(m)
        data = self.cluster_peaks(data, threshold, peak_indices)
        m_cluster = len(data['peaks'])
        m_star = m
        iterations = 1

        # Iteration till we have equal or more than m clusters, or max_it reaches.
        while m_cluster < m and iterations < max_iterations:
            m_star += int(m * increase_rate) # more PEAKS
            data, peak_indices = self.raw_data_process(m_star)
            data = self.cluster_peaks(data, threshold, peak_indices)
            m_cluster = len(data['peaks'])
            print("------iter:", iterations, "m star:", m_star," length cluster:", m_cluster, "-------")
            iterations += 1

        # Sort the clusters by their count and keep only the top ones
        sorted_clusters = sorted(data['peaks'].items(), key=lambda x: x[1]['count'], reverse=True)
        top_clusters = dict(sorted_clusters[:m])

        
        return top_clusters, data

    def run(self) -> dict:
        clusters, data = self.increase_m_and_repeat(self.file_path, self.m, self.threshold, self.window, self.max_it)
        return self.max_pool(data=data, cluster_data=clusters)
