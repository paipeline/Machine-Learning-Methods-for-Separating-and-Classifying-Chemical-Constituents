import os
print("New current working directory:", os.getcwd())

import numpy as np
import re
from src.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split

class PAHRatio:
    """
    A class to process and prepare data for predicting PAH ratios.

    Attributes:
    - lib (str): Directory containing the dataset.
    - m, clustering_threshold, K, maxIt (int): Parameters for the Preprocessor.
    - inputs (list): List to store processed input data.
    - labels (list): List to store labels corresponding to the inputs.
    - model_inputs (np.array): Numpy array to store the final input data for the model.
    - model_labels (np.array): Numpy array to store the final labels for the model.
    """

    def __init__(self, lib, m=10, clustering_threshold=20, windows_size = 50, K=5, maxIt=100,prominence = 0.02,baseline_removal =False):
        """
        Initializes the PAHRatio class with the given parameters.
        """
        self.window = windows_size
        self.lib = lib
        self.m = m
        self.clustering_threshold = clustering_threshold
        self.K = K
        self.maxIt = maxIt
        self.inputs = []
        self.labels = []
        self.pesticide_inputs = []
        self.pesticide_labels = []
        self.model_inputs = None
        self.model_labels = None
        self.prominence =prominence
        self.baseline_removal = baseline_removal

    def seperate_letter_num(self, string):
        """
        Separates letters and numbers from a string.

        Parameters:
        - string (str): The string to be separated.

        Returns:
        - tuple: A tuple containing the separated number and letter.
        """
        number = re.findall(r'\d+', string)
        letter = re.findall(r'[A-Za-z]+', string)
        return number[0], letter[0]

    def get_pah_ratios(self, file_name):

        if file_name == "ImidaclopridTapwaterdata.xlsx":
            return np.array([0, 0, 0, 0])

        ratios = {'ANTH': 0, 'PYR': 0, 'BaP': 0, 'BaA': 0}
        file_name = file_name.replace("SERS ", "")
        file_name = file_name.replace(".xlsx", "")

        if "+" not in file_name:
            ratios[file_name] = 1
        else:
            compounds = file_name.split("+")
            for compound in compounds:
                num, name = self.seperate_letter_num(compound)
                ratios[name] = int(num)

        return np.array(list(ratios.values()))

    def process_files(self):
        for file in os.listdir(self.lib):
            if file.endswith('.xlsx'):
                file_path = os.path.join(self.lib, file)
                pre_pro = Preprocessor(file_path=file_path, m=self.m, threshold = self.clustering_threshold, mat_iteration= self.maxIt, window = self.window, prominence = self.prominence, baseline_removal = self.baseline_removal)
                print(f"{file} gets processed successfully")
                pooled_vectors = pre_pro.run()

                label = self.get_pah_ratios(file)
                for col in pooled_vectors:
                    self.inputs.append(pooled_vectors[col])
                    self.labels.append(label)
        # Temporarily commented out for debugging purposes
        """
        for file in os.listdir(self.lib):
            if file.endswith('.xlsx'):
                file_path = os.path.join(self.lib, file)
                
                pre_pro = Preprocessor(file_path, self.m, self.clustering_threshold, self.maxIt, self.K, self.prominence)
                pooled_vectors = pre_pro.run()
                label = self.get_pah_ratios(file)
                print(f"--- loading {file} ---")
                for col in pooled_vectors:
                    if file != "ImidaclopridTapwaterdata.xlsx": # for PAH
                        self.inputs.append(pooled_vectors[col])
                        self.labels.append(label)
                    else: # for pesticide
                        self.pesticide_inputs.append(pooled_vectors[col])
                        self.pesticide_labels.append(label)
                print(f"/t{file} gets processed successfully")
        """
                        
    def prepare_model_data(self, test_size=0.1):
        """
        Prepares the model data by converting inputs and labels to numpy arrays.
        """
        print(test_size)
        self.model_inputs = np.array(self.inputs)
        self.model_labels = np.stack(self.labels)
        #print("Model Inputs Shape:", self.model_inputs.shape)
        #print("Model Labels Shape:", self.model_labels.shape)
        if test_size > 0:
            self.train_inputs, self.test_inputs, self.train_labels, self.test_labels = train_test_split(
                self.model_inputs, self.model_labels, test_size=test_size, random_state=42)

    def save_processed_data(self):
        """
        Saves the processed data to numpy files.
        """
        np.save('dataset/model_inputs.npy', self.train_inputs)
        np.save('dataset/model_labels.npy', self.train_labels)
        np.save('dataset/test_inputs.npy', self.test_inputs)
        np.save('dataset/test_labels.npy', self.test_labels)
        print("Processed PAH model data saved successfully")
        np.save('dataset\pesticide_inputs.npy', np.array(self.pesticide_inputs))
        np.save('dataset\pesticide_labels.npy', np.stack(self.pesticide_labels))
        print("Processed Pesticide model data saved successfully")

    def run(self):
        """
        Runs the processes for file processing, data preparation, and saving the data.
        """
        self.process_files()
        self.prepare_model_data()
        self.save_processed_data()

# Example usage of the class
if __name__ == '__main__':
    pah_predictor = PAHRatio(lib='raw_dataset', m=10, clustering_threshold=20, K=5, maxIt=100, prominence =0.2,windows_size = 50)
    pah_predictor.run()
