import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=400, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(400)
        self.conv2 = nn.Conv1d(in_channels=400, out_channels=200, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(200)

        self.flatten = nn.Flatten()

        # Calculate the size of the flattened layer
        flattened_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def _get_conv_output(self, shape):
        batch_size = 1
        input_tensor = torch.rand(batch_size, *shape)
        output_feat = self._forward_features(input_tensor)
        flattened_size = output_feat.data.view(batch_size, -1).size(1)
        return flattened_size

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, 70)
        self.fc4 = nn.Linear(70, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class UNetModel(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(UNetModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 16, 3, padding='same')
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 64, 3, padding='same')
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv1d(64, 16, 3, padding='same')
        self.output_conv = nn.Conv1d(16, 1, 1)  # Adjust the output channels as needed

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = self.pool(x1)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.output_conv(x)
        return x
def get_rf_model():
    random_forest = RandomForestRegressor(n_estimators=200)
    return random_forest
def get_svr_model():
    svr_model = MultiOutputRegressor(SVR(kernel='rbf'))
    return svr_model

