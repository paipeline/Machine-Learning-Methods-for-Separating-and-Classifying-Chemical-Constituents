import os
from preprocessor import Preprocessor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Set your working directory and initialize the Preprocessor
os.chdir("C:\\Users\\23675\\OneDrive - UW-Madison\\Desktop\\research final")
pre_processor = Preprocessor(file_path='raw_dataset\SERS 2BaA+1ANTH.xlsx',
                             m=10, threshold=20, mat_iteration=100, windows_size=10, prominence=0.02)

# Load raw data using the pre_processor instance
x_values, y_values_dict = pre_processor.load_raw_data(pre_processor.file_path)

# Calculate the mean of the raw data
y_mean_raw = pd.DataFrame(y_values_dict).mean(axis=1)

# Define the window sizes
window_sizes = range(5, 201)

# Plotting
fig, axes = plt.subplots(len(window_sizes) + 1, 1, figsize=(12, 6 * (len(window_sizes) + 1)))  # rows = len(window_sizes) + 1, columns = 1

# Raw Data Plot
axes[0].plot(x_values, y_mean_raw, label='Raw Data', color='blue')
axes[0].set_title('Raw Data')
axes[0].set_xlabel('X-Values (Raman Shift)')
axes[0].set_ylabel('Y-Values (Intensity)')
axes[0].grid(True)

# Create a function to update the plot for each frame of the animation
def update_plot(frame):
    window_size = frame + 5  # Calculate the window size for the current frame
    axes[1:].clear()  # Clear the axes for rolling window plots

    # Iterate over the window sizes and generate plots
    for i, size in enumerate(range(1, frame + 2)):
        # Apply a rolling window to each recording
        for sample_name, y_values in y_values_dict.items():
            y_rolling = y_values.rolling(window=size).mean()

            # Rolling Window Plot
            axes[i + 1].plot(x_values, y_rolling, label=f'{sample_name} (Window Size: {size})', linewidth=2)

        axes[i + 1].set_title(f'Rolling Window Applied to Each Recording (Window Size: {size})')
        axes[i + 1].set_xlabel('X-Values (Raman Shift)')
        axes[i + 1].set_ylabel('Y-Values (Intensity)')
        axes[i + 1].grid(True)

    plt.tight_layout()

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=len(window_sizes), interval=200)

# Show the animation
plt.show()