"""clusterPeaks.py This clusters all peaks by their distance being D from each other"""

__author__      = "Chen Li"
__copyright__   = "Copyright 2023"

from sklearn.cluster import DBSCAN
import numpy as np


def cluster_peaks(data, d_t, indices):
    '''
    This method clusters all peaks by their distance being D from each other

    @param data: data dictionary
    @param d_t: threshold distance
    @return: data dictionary with peaks clustered
    '''

    def create_clusters(dt):
        indice = np.sort(indices)

        clusters = []
        current_cluster = [indice[0]]

        for idx in indice[1:]:
            if idx - current_cluster[0] <= dt:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]

        clusters.append(current_cluster)

        return clusters

    # Cluster using custom function
    clusters = create_clusters(d_t)

    cluster_dict = {}

    # Populate the cluster_dict
    for i, cluster in enumerate(clusters):
        cluster_dict[i] = {}
        cluster_dict[i]['indices'] = cluster
        cluster_dict[i]['count'] = sum(data['peaks'][index] for index in cluster)

    data['peaks'] = cluster_dict

    return data


