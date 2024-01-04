from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models.models import CNNModel,DNNModel, get_rf_model, get_svr_model,UNetModel  # Custom model imports
from src.utils import plot_metrics, print_metrics  # Utility functions for metrics and plotting
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
"""
train_evaluate.py

This script contains functions for training and evaluating different machine learning models,
including a custom CNN model built with PyTorch, and other models like Random Forest and SVR from scikit-learn.
The script utilizes k-fold cross-validation for assessing model performance.

Functions:
- trainh_model: Trains model
- evaluate_model: Evaluates a model
- perform_kfold_cv: Performs k-fold cross-validation for a model
- train_and_evaluate_all_models: Orchestrates the training and evaluation of all specified models
"""
def train_model(model, train_loader, criterion, optimizer, epochs):
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)
    return losses


def evaluate_model(model, test_loader):
    """ Evaluates a model by calculating the mean squared error over a test dataset. """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += torch.nn.functional.mse_loss(output, target, reduction='sum').item()
    test_loss /= len(test_loader.dataset)
    return test_loss

def perform_kfold_cv(X, Y, model, k=5):
    """ 
    Performs k-fold cross-validation on a model and returns the average mean squared error.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    mses = []  # Store mean squared error for each fold

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Create dataloaders for training and testing
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()

        # Train and evaluate the model
        train_model(model, train_loader, criterion, optimizer, epochs=15)
        mse = evaluate_model(model, test_loader)
        mses.append(mse)
        # Further metrics like MAE, R2 can be calculated and appended here

    avg_mse = sum(mses) / len(mses)
    return avg_mse  # Return the average MSE over all folds

def plot_predictions_vs_actual(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Labels")
    plt.ylabel("Predicted Labels")
    plt.title(title)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # Reference line
    plt.show()
def plot_learning_curve(losses, title, ylabel='Loss', xlabel='Epochs'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
"""

train_and_evaluate_all_models

Trains and evaluates multiple machine learning models, including both PyTorch-based neural networks
(CNN, DNN, U-Net) and traditional machine learning models from scikit-learn (Random Forest, SVR).
The function employs k-fold cross-validation to assess the performance of each model.

Parameters:
- train_X: Training data features
- train_Y: Training data labels or targets
- test_X: Testing data features
- test_Y: Testing data labels or targets

The function assumes the input data for PyTorch models (CNN, DNN, U-Net) is in a format compatible
with PyTorch (e.g., tensors), and for scikit-learn models (RF, SVR), the data is in a format 
compatible with scikit-learn (e.g., numpy arrays).

The function trains each model using the provided training data and evaluates them on the testing data,
calculating the mean squared error (MSE) for each fold of the cross-validation and then averaging it.

The function prints the average MSE for each model, giving insights into their comparative performance.

Returns:
- None: The function prints the average MSE for each model but does not return any values.
"""

def train_and_evaluate_all_models(train_X, train_Y, test_X, test_Y):
    model_metrics = {}
    fold_losses = {}  # To store losses for each fold
    kf = KFold(n_splits=5, shuffle=True, random_state=30)
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)
    scaler = StandardScaler()
    scaler.fit(train_X) 
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    original_shape = (-1, 2, 10)
    cnn_train_X = train_X.reshape(original_shape)
    cnn_test_X = test_X.reshape(original_shape)

    for model_name, get_model in [('CNN', CNNModel)]:
        mses = []
        for train_index, test_index in kf.split(cnn_train_X):
            X_train, X_test = cnn_train_X[train_index], cnn_train_X[test_index]
            y_train, y_test = train_Y[train_index], train_Y[test_index]

            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            model = get_model(input_shape=original_shape[1:])
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.MSELoss()

            train_model(model, train_loader, criterion, optimizer, epochs=10)
            mse = evaluate_model(model, test_loader)
            mses.append(mse)

        avg_mse = sum(mses) / len(mses)
        model_metrics[model_name] = avg_mse

    # Models to train
    models_to_train = {
        'DNN': DNNModel,
        'RF': get_rf_model,
        'SVR': get_svr_model
    }
    fold_mses = {}
    for model_name, get_model in models_to_train.items():
        mses = []
        fold_losses[model_name] = []
        fold_mses[model_name] = []  # Initialize list for traditional models' MSEs

        for fold, (train_index, test_index) in enumerate(kf.split(train_X)):
            X_train, X_test = train_X[train_index], train_X[test_index]
            y_train, y_test = train_Y[train_index], train_Y[test_index]

            if model_name in ['DNN']:
                # Training for deep learning models
                model = get_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
                optimizer = torch.optim.Adam(model.parameters())
                criterion = torch.nn.MSELoss()

                train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
                test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                losses = train_model(model, train_loader, criterion, optimizer, epochs=10)
                mse = evaluate_model(model, test_loader)
                mses.append(mse)
                fold_losses[model_name].append(losses)

            else:
                # Training for traditional models
                model = get_model()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                mses.append(mse)
                fold_mses[model_name].append(mse)
        avg_mse = sum(mses) / len(mses)
        model_metrics[model_name] = avg_mse

    # Plotting losses and MSEs
    for model_name, losses in fold_losses.items():
        plt.figure(figsize=(12, 6))
        for fold, fold_loss in enumerate(losses):
            plt.plot(fold_loss, label=f'Fold {fold+1}')
        plt.title(f'Training Losses for {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    for model_name, mse in model_metrics.items():
        print(f"Average MSE for {model_name}: {mse}")

# Call the function with your data
# train_and_evaluate_all_models(train_X, train_Y, test_X, test_Y)
