import keras.src.activations
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src import regularizers
from keras.src.applications.densenet import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from prettytable import PrettyTable
from PIL import Image, ImageDraw
from keras import backend

def custom_loss(y_true, y_pred):
    overestimation_penalty_coefficient = 10
    error = float(y_pred) - float(y_true)
    loss = backend.square(error)
    over_penalty_loss = overestimation_penalty_coefficient * loss
    return backend.mean(backend.switch(error > 0, over_penalty_loss, loss), axis=-1)

""" This is not working since it is non-differentiable
def custom_activation(x):
    return backend.round(keras.src.activations.relu(x))
    """

def get_cnn_model():
    # This model is subject to change
    cnn_model = Sequential([
        layers.Conv1D(filters=400, kernel_size=2, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='relu'),
    ])

    cnn_model.compile(optimizer='adam', loss=custom_loss)

    return cnn_model

def get_rf_model():
    random_forest = RandomForestRegressor(n_estimators=10)
    return random_forest

def get_dnn_model():

    dnn_model = Sequential([
        layers.Dense(1024, activation='relu', input_shape=[dnn_X_train.shape[1]]),
        layers.Dropout(0.4),
        layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
        layers.Dense(70, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
        layers.Dense(4, activation='relu')
    ])

    dnn_model.compile(optimizer='adam', loss=custom_loss)

    return dnn_model

def get_svr_model():
    svr_model = MultiOutputRegressor(SVR(kernel='rbf'))
    return svr_model


def plot_metrics(metric_name, data):
    plt.figure(figsize=(10, 6))
    for model_name, metric in data.items():
        plt.plot(metric[metric_name], label = f"{model_name} - {metric_name}")

    plt.title(f"{metric_name} for models")
    plt.xlabel("folds")
    plt.ylabel(metric_name)
    plt.grid()
    plt.legend()
    plt.savefig(f"{metric_name}.png")
    plt.show()


def print_metrics(mae, mse, model_name, r2, err):
    print("==================================")
    print(f"model: {model_name}")
    print(f"mae avg: {mae}")
    print(f"r2 avg: {r2}")
    print(f"mse avg: {mse}")
    print(f"Error Rate: {err}")


def perform_kfold_cv(X, Y, get_model, model_name, additional_test_X, additional_text_Y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    maes, r2s, mses, err = [], [], [], []
    add_maes, add_r2s, add_mses, add_err = [], [], [], []

    for train_index, text_index in kf.split(X):
        X_train, X_test = X[train_index], X[text_index]
        y_train, y_test = Y[train_index], Y[text_index]
        model = get_model()

        if model_name == "CNN" or model_name == "DNN":
            model.fit(X_train, y_train, epochs=50, batch_size=5)
        else:
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        add_pred = model.predict(additional_test_X)

        # This can use a custom threshold (can be obtained from gridSearch)
        rounded_pred = np.round(pred)
        rounded_add_pred = np.round(add_pred)

        error = np.mean(np.abs(rounded_pred - y_test) > 0)
        add_error = np.mean(np.abs(rounded_add_pred - additional_text_Y) > 0)
        add_maes.append(mean_absolute_error(additional_text_Y, add_pred))
        add_mses.append(mean_squared_error(additional_text_Y, add_pred))
        add_r2s.append(r2_score(additional_text_Y, add_pred))
        maes.append(mean_absolute_error(y_test, pred))
        mses.append(mean_squared_error(y_test, pred))
        r2s.append(r2_score(y_test, pred))
        err.append(error)
        add_err.append(add_error)

    print_metrics(np.mean(maes), np.mean(mses), model_name, np.mean(r2s), np.mean(err))
    print_metrics(np.mean(add_maes), np.mean(add_mses), model_name + " - Pesticide", np.mean(add_r2s), np.mean(add_err))

    return {
        'MAE': maes,
        'MSE': mses,
        'R2': r2s,
        'ErrorRate': err,
        'Pesticide_MAE': add_maes,
        'Pesticide_MSE': add_mses,
        'Pesticide_R2': add_r2s,
        'Pesticide_ErrorRate': add_err
    }



if __name__ == "__main__":
    """
    prev_data = PAHRatio('Data', m=10, clustering_threshold=20, K=5, maxIt=100)
    prev_data.process_files()
    prev_data.prepare_model_data()

    train_X = np.array(prev_data.cnn_inputs)
    train_Y = np.array(prev_data.cnn_labels)
    test_data = PAHRatio('test_data', m=10, clustering_threshold=20, K=5, maxIt=100)
    test_data.process_files()
    test_data.prepare_model_data()
    test_X = np.array(test_data.cnn_inputs)
    test_Y = np.array(test_data.cnn_labels)
    """
    train_X = np.load('train_X.npy')
    train_Y = np.load('train_Y.npy')
    test_X = np.load('test_X.npy')
    test_Y = np.load('test_Y.npy')
            
    dnn_X_train = train_X.reshape(train_X.shape[0], -1)
    dnn_X_test = test_X.reshape(test_X.shape[0], -1)
    dnn_scaler = StandardScaler()
    dnn_scaler.fit(dnn_X_train) 
    dnn_X_train = dnn_scaler.transform(dnn_X_train)
    dnn_X_test = dnn_scaler.transform(dnn_X_test)

    model_metrics = {}
    model_metric["U"] = perform_kfold_cv(dnn_X_train, train_Y,get_unet_model,"U-net",dnn_X_test,test_Y)
    model_metrics['CNN'] = perform_kfold_cv(train_X, train_Y, get_cnn_model, "CNN", test_X, test_Y)
    model_metrics['DNN'] = perform_kfold_cv(dnn_X_train, train_Y, get_dnn_model, "DNN", dnn_X_test, test_Y)
    model_metrics['SVR'] = perform_kfold_cv(dnn_X_train, train_Y, get_svr_model, "SVR", dnn_X_test, test_Y)
    model_metrics['RF'] = perform_kfold_cv(dnn_X_train, train_Y, get_rf_model, "RF", dnn_X_test, test_Y)
    keys = model_metrics['CNN'].keys()

    for key in keys:
        plot_metrics(key, model_metrics)