# Use this to make 10 figs that shows each step of preprocessing
import os

import numpy as np

from src.preprocessor import Preprocessor

if __name__ == "__main__":
    files = ['SERS 1BaP+2ANTH.xlsx', 'SERS 2ANTH+1PYR.xlsx', 'SERS 1BaP+1PYR.xlsx', 'ImidaclopridTapwaterdata.xlsx',
             'SERS 1ANTH+1PYR+1BaP+2BaA.xlsx', 'SERS 1BaP+2BaA.xlsx', 'SERS 1ANTH+2PYR+1BaP+1BaA.xlsx',
             'SERS 5ANTH+1PYR.xlsx', 'SERS 2ANTH+1PYR+1BaP+1BaA.xlsx', 'SERS BaA.xlsx']
    dir = '../raw_dataset'

    global_raw_data = []
    global_sliced_data = []
    global_baseline_removed = []
    global_normalized = []
    top_cluster_s = []



    for file in os.listdir(dir):
        if file in files:
            file_path = os.path.join(dir, file)
            processor = Preprocessor(file_path, m=10, threshold=20, mat_iteration=100, window=50, prominence=0.02)
            raw_data, sliced_data, baseline_removed_data, normalized_vals, top_clusters, data, max =  processor.outside_repeat(file_path, m=10, threshold=20, max_iterations=100, window=50)

            x, y = raw_data
            global_raw_data.append([np.array(x), np.array(y.get(0))])
            global_sliced_data.append(sliced_data[0])
            global_baseline_removed.append(baseline_removed_data[0])
            global_normalized.append(normalized_vals[0])

            top_cluster_s.append(top_clusters)

            print(max)
