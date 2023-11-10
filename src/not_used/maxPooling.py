"""maxPooling to make the resulting x M-sparse"""

__author__ = "Chen Li"
__copyright__ = "Copyright 2023"


def max_pool(data, cluster):
    """
    This method performs max pooling on the data.

    @param data: data dictionary
    @param cluster: cluster dictionary
    @return: data dictionary with max pooled data with corresponding midpoint

    """
    mid_points = {}
    pooled_vectors = {}

    for label, info in cluster.items():
        indices = info['indices']
        smallest, largest = min(indices), max(indices)
        mid_point = (smallest + largest) // 2
        mid_points[label] = [mid_point, smallest, largest]

    # Perform max pooling
    for col, data_recording in data['y'].items():
        pooled_vector = [
            (
                mid_points[label][0],
                max(data_recording[start:end + 1])
            )
            for label, (mid_point, start, end) in mid_points.items()
        ]

        pooled_vectors[col] = pooled_vector

    return pooled_vectors

"""
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
    print(len(data["y"]))
    col_names = data["y"]
    sample_num = len(col_names) - 1  # minus 1 because the first column is 'x'
    print(max_pool(data, dct_cluster))
    #data format:
    #data = {'x': x, 'y': y, 'peaks': peaks, 'smoothed_y': smoothed_y,
    # "clusters": {"indices","count"}}











 ## version without midpoint
def max_pool(data, cluster):

    mid_points = {}
    pooled_vectors = {}

    for label in cluster:
        indices = cluster[label]['indices']
        smallest = min(indices)
        largest = max(indices)
        mid_point = int((smallest + largest) / 2)
        mid_points[label] = [mid_point, smallest, largest]

    # Perform max pooling
    for col in data['y']:
        data_recording = data['y'][col]
        pooled_vector = np.zeros_like(data_recording)

        for label in cluster:
            start = mid_points[label][1]
            end = mid_points[label][2] + 1
            max_value = -1

            # Find the max value in the cluster
            for i in range(start, end):
                if data_recording[i] > max_value:
                    max_value = data_recording[i]

            pooled_vector[mid_points[label][0]] = max_value

        pooled_vector = pooled_vector[pooled_vector != 0]

    return pooled_vectors

store the midpoint as a pair to each max_value: (mid,max value)    
"""