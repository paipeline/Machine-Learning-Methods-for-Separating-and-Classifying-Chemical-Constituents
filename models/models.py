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
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=256, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
        self.bn4 = nn.BatchNorm1d(32)

        # Calculate the size of the flattened layer
        flattened_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(flattened_size, 268)
        self.fc2 = nn.Linear(268, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 4)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.output(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, 64)
        self.fc4 = nn.Linear(64, output_dim)
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

