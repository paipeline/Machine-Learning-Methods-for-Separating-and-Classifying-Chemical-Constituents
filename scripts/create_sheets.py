import os

import numpy as np
from BaselineRemoval import BaselineRemoval

from src.preprocessor import Preprocessor
import pandas as pd


def data_to_df(data):
    df_dict = {}
    max_length = 0
    for key, val in data.items():
        x = val['x']
        y = val['y']
        lengths = [len(arr) for arr in y]
        max_length = max(max_length, *lengths, len(x))

        df_dict[f"X Value-{key}"] = np.pad(x, (0, max_length - len(x)), 'constant', constant_values=np.nan)

        for i, y_values in enumerate(y):
            df_dict[f"{key}-{i}"] = np.pad(y_values, (0, max_length - len(y_values)), 'constant',
                                           constant_values=np.nan)

    return pd.DataFrame(df_dict)


if __name__ == "__main__":
    raw_data = {}
    dir = '../raw_dataset'
    for file in os.listdir(dir):
        if file.endswith(".xlsx"):
            file_path = os.path.join(dir, file)
            processor = Preprocessor(file_path, m=10, threshold=20, mat_iteration=100, window=50, prominence=0.02)
            raw_x, raw_y = processor.load_raw_data(file_path)
            raw_data[file] = {'x': raw_x, "y":raw_y}

    sliced_data = {}
    baseline_removed_data = {}
    normalized_val = {}

    for item in raw_data.items():
        name = item[0]
        data = item[1]
        x = data['x']
        new_y_values = []
        basline_removed_y_value = []
        normalized_ys = []
        final_sliced_x = None
        for y_values in data['y'].values():
            sliced = [(x_val, y_val) for x_val, y_val in zip(x, y_values) if 350 <= x_val <= 2000 and not np.isnan(y_val)]

            sliced_x, sliced_y = zip(*sliced)
            if final_sliced_x == None:
                final_sliced_x = sliced_x
            new_y_values.append(sliced_y)
            print(len(sliced_y))
            baseObj = BaselineRemoval(sliced_y)
            baseline_removed_y = baseObj.IModPoly()
            basline_removed_y_value.append(baseline_removed_y)

            normalized_y = (baseline_removed_y - min(baseline_removed_y)) / (max(baseline_removed_y) - min(baseline_removed_y))
            normalized_ys.append(normalized_y)
        sliced_data[name] = {'x' : final_sliced_x, 'y' : new_y_values}
        baseline_removed_data[name] = {'x' : final_sliced_x, 'y' : basline_removed_y_value}
        normalized_val[name] = {'x' : final_sliced_x, 'y' : normalized_ys}

    df_sliced = data_to_df(sliced_data)
    df_baseline_removed = data_to_df(baseline_removed_data)
    df_normalized = data_to_df(normalized_val)

    with pd.ExcelWriter('processed_data.xlsx') as writer:
        df_sliced.to_excel(writer, sheet_name='Sliced', index=False)
        df_baseline_removed.to_excel(writer, sheet_name='Baseline-Removed', index=False)
        df_normalized.to_excel(writer, sheet_name='Normalized', index=False)
