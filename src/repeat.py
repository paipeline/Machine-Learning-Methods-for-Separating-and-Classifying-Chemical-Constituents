"""repeat.py This repeats the step 1-5 until num of clusters >= M"""
__author__ = "Chen Li & Pai Peng"
__copyright__ = "Copyright 2023"

from Src.dataPreprocess import process_data
from Src.clusterPeaks import cluster_peaks
import os


# Uses the algorithm described in the Method section, iterate until m_c >= m_sta
# Uses the algorithm described in the Method section, iterate until m_c >= m_sta
def increase_m_and_repeat(file_path, m, threshold, k, max_it, m_rate=0.2):
    m_c = 0
    m_sta = m  # just M stays the same
    data, peak_indices= process_data(file_path, m, k)
    data = cluster_peaks(data, threshold, peak_indices)
    m_c = len(data['peaks'])
    iterations = 1

    while m_c < m_sta and iterations < max_it:
        m += int(m_sta * m_rate)

        data, peak_indices= process_data(file_path, m, k)
        data = cluster_peaks(data, threshold, peak_indices)
        m_c = len(data['peaks'])
        iterations += 1


    # get the top M clusters (then we have estimated range of CP locations, the next step is described by
    # "Compressing Spectra to Lower Dimensions." section in the paper
    sorted_clusters = sorted(data['peaks'].items(), key=lambda x: x[1]['count'], reverse=True)
    top_clusters = dict(sorted_clusters[:m_sta])
    """
    Top Cluster 里面是 top M cluster 基于他们的 aggregated count。
    Data 里面是xvalue， yvalue， 和所有clusters。

    具体使用：

    top_cluster[CLUSTER_LABEL]['indices'] - indices of all peaks within a cluster label (这个不是x value）

    top_cluster[CLUSTER_LABEL]['count'] - aggregated Count
    data['x'] - xvalued (可以使用 indice 来调取 peak 的 x 轴的值）
    data['y'] - smooth 过后的 所有 recording
    data['peaks'] - 所有的peaks，具体access方式同 top_clusters
    """
    return top_clusters, data



#test

if __name__ == "__main__":
    lib = 'Data'
    data_dic = {}
    m = 20 # Hyperparamer (it suppose to be 10C, but I have no idea how to make it happen)
    clustering_threshold = 100 # Hyper parameter
    K = 3 # the number of clusters
    maxIt = 20
    for file in os.listdir(lib):
        if file.endswith('.xlsx'):
            file_path = os.path.join(lib, file)
            key = os.path.splitext(file)[0]
            top_clusters, data = increase_m_and_repeat(file_path, m, clustering_threshold, K, maxIt)
            data_dic[key] = [top_clusters, data]
            
