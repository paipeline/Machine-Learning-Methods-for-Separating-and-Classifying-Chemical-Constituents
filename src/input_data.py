import os
import numpy as np
import re
from preprocessor import Preprocessor

class PAHRatio:

    def __init__(self, lib, m=10, clustering_threshold=20, K=5, maxIt=100):
        self.lib = lib
        self.m = m
        self.clustering_threshold = clustering_threshold
        self.K = K
        self.maxIt = maxIt
        self.inputs = []
        self.labels= []
        self.model_inputs = None
        self.model_labels = None
    def seperate_letter_num(self, string):
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
                pre_pro = Preprocessor(file_path, self.m, self.clustering_threshold, self.maxIt, self.K)
                print(f"{file} gets processed successfully")
                pooled_vectors = pre_pro.run()

                label = self.get_pah_ratios(file)
                for col in pooled_vectors:
                    self.inputs.append(pooled_vectors[col])
                    self.labels.append(label)

    def prepare_model_data(self):
        self.model_inputs = np.array(self.inputs)
        self.model_labels = np.stack(self.labels)

    def save_processed_data(self):
        np.save('dataset\model_inputs.npy', self.model_inputs)
        np.save('dataset\model_labels.npy', self.model_labels)
        print(f"{file} gets processed successfully")
    def run(self):
        self.process_files()
        self.prepare_model_data()
        self.save_processed_data()

# Example Usage:
if __name__ == '__main__':
    pah_predictor = PAHRatio(lib='raw dataset', m=10, clustering_threshold=20, K=5, maxIt=100)
    pah_predictor.run()
