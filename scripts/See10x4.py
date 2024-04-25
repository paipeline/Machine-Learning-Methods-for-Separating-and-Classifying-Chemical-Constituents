import os
import matplotlib.pyplot as plt
import pandas as pd
from src.preprocessor import Preprocessor
from BaselineRemoval import BaselineRemoval

#RAW

# Since we want 10 different cols, I use 10 different files




raw_data = []
dir = '../raw_dataset' # change this as needed

#print(os.listdir('../raw_dataset'))
files = ['SERS 1BaP+2ANTH.xlsx', 'SERS 2ANTH+1PYR.xlsx', 'SERS 1BaP+1PYR.xlsx', 'ImidaclopridTapwaterdata.xlsx',
         'SERS 1ANTH+1PYR+1BaP+2BaA.xlsx', 'SERS 1BaP+2BaA.xlsx', 'SERS 1ANTH+2PYR+1BaP+1BaA.xlsx',
         'SERS 5ANTH+1PYR.xlsx', 'SERS 2ANTH+1PYR+1BaP+1BaA.xlsx', 'SERS BaA.xlsx'] # change the dataset as needed as well
for file in os.listdir(dir):
    if file in files:
        file_path = os.path.join(dir, file)
        processor = Preprocessor(file_path,m=10, threshold=20,mat_iteration=100,window=50, prominence=0.02)
        raw_x, raw_y = processor.load_raw_data(file_path)
        if file == 'ImidaclopridTapwaterdata.xlsx':
            single_raw_y = raw_y[0]
        else:
            single_raw_y = raw_y[5]
        raw_data.append([raw_x, single_raw_y])

# Ploting
plt.figure(figsize=(100, 50))
plt.suptitle('Raw / Sliced / BaselineRemoved / Normalized')

for i, (raw_x, raw_y) in enumerate(raw_data, start=1):
    plt.subplot(10, 4, 4*i - 3)
    plt.plot(raw_x, raw_y)
    plt.title(files[i-1])
    plt.xlabel('xVal')
    plt.ylabel('yVal')
    plt.grid(True)

sliced_data = []
for (xval, yval) in raw_data:
    sliced = [(x, y) for x, y in zip(xval, yval) if 350 <= x <= 2000]
    sliced_x, sliced_y = zip(*sliced)
    sliced_data.append([sliced_x, sliced_y])

# Plot
for i, (sliced_x, sliced_y) in enumerate(sliced_data, start=1):
    plt.subplot(10, 4, 4*i-2)
    plt.plot(sliced_x, sliced_y)
    plt.title(files[i-1])
    plt.xlabel('xVal')
    plt.ylabel('yVal')
    plt.grid(True)


# Basline Removal:
baseline_removed_data = []
for val in sliced_data:
    x, y = val
    baseObj = BaselineRemoval(y)
    baseline_removed_y = baseObj.ZhangFit()
    baseline_removed_data.append([x, baseline_removed_y])
    
for i, (x, y) in enumerate(baseline_removed_data, start=1):
    plt.subplot(10, 4, 4*i - 1)
    plt.plot(x, y)
    plt.title(files[i-1])
    plt.xlabel('xVal')
    plt.ylabel('yVal')
    plt.grid(True)

# Normalization

normalized_val = []
for val in baseline_removed_data:
    x, y = val
    normalized_y = (y - min(y)) / (max(y) - min(y))
    normalized_val.append([x, normalized_y])

for i, (x, y) in enumerate(normalized_val, start=1):
    plt.subplot(10, 4, 4*i)
    plt.plot(x, y)
    plt.title(files[i-1])
    plt.xlabel('xVal')
    plt.ylabel('yVal')
    plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('../fig/Comparison RAW SLICED BASELINE_REMOVED NORMALIZED')
plt.show()

for file in os.listdir(dir):
    for file in files:
        file_path = os.path.join(dir, file)
        processor = Preprocessor(file_path,m=10, threshold=20,mat_iteration=100,window=50, prominence=0.02,baseline_removal=True)
        pre_pro.increase_m_and_repeat(self.file_path, self.m, self.threshold, self.window, self.max_it)

#1. make excel file




