import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# from prettytable import PrettyTable
from src.preprocessor import Preprocessor
from src.input_data import PAHRatio


# This is a arbitray seed for setting initial weights
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)

def custom_loss(y_true, y_pred):
    overestimation_penalty_coefficient = 30
    error = float(y_pred) - float(y_true)
    loss = backend.square(error)
    over_penalty_loss = overestimation_penalty_coefficient * loss
    return backend.mean(backend.switch(error > 0, over_penalty_loss, loss), axis=-1)

""" This is not working since it is non-differentiable
def custom_activation(x):
    return backend.round(keras.src.activations.relu(x))
    """

def get_cnn_model():
    # Fix initial weights to make results reproducible
    weight_initializer = keras.initializers.GlorotUniform(seed=RANDOM_SEED)
    # This model is subject to change
    cnn_model = Sequential([
        layers.Conv1D(filters=400, kernel_size=2, activation='tanh',
                      input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=weight_initializer,
                      bias_initializer=initializers.Zeros()),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Conv1D(filters=200, activation='tanh', kernel_size=3, kernel_initializer=weight_initializer,
                      bias_initializer=initializers.Zeros()),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1024, activation='tanh', kernel_initializer=weight_initializer,
                     bias_initializer=initializers.Zeros()),
        layers.Dense(512, activation='relu', kernel_initializer=weight_initializer,
                     bias_initializer=initializers.Zeros()),
        layers.Dense(128, activation='relu', kernel_initializer=weight_initializer,
                     bias_initializer=initializers.Zeros()),
        layers.Dropout(0.2, seed=RANDOM_SEED),
        layers.Dense(4, activation='relu', kernel_initializer=weight_initializer,
                     bias_initializer=initializers.Zeros()),
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    cnn_model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])

    return cnn_model

def get_rf_model():
    random_forest = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED)
    return random_forest

def get_dnn_model():
    # Fix initial weights to make results reproducible
    weight_initializer = keras.initializers.GlorotUniform(seed=RANDOM_SEED)
    dnn_model = Sequential([
        layers.Dense(1024, activation='relu', input_shape=[dnn_X_train.shape[1]], kernel_initializer=weight_initializer,
                    bias_initializer=initializers.Zeros()),
        layers.Dense(512, activation='relu', input_shape=[dnn_X_train.shape[1]], kernel_initializer=weight_initializer,
                     bias_initializer=initializers.Zeros()),
        layers.Dropout(0.4, seed=RANDOM_SEED),
        layers.Dense(100, activation='relu', kernel_initializer=weight_initializer,
                      bias_initializer=initializers.Zeros()),
        layers.Dense(70, activation='relu', kernel_regularizer=regularizers.l2(0.1),
                     kernel_initializer=weight_initializer,
                     bias_initializer=initializers.Zeros()
                     ),
        layers.Dense(4, activation='relu', kernel_initializer=weight_initializer,
                    bias_initializer=initializers.Zeros())
    ])
    dnn_model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])

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


def perform_kfold_cv(X, Y, get_model, model_name, additional_test_X, additional_text_Y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    maes, r2s, mses, err = [], [], [], []
    add_maes, add_r2s, add_mses, add_err = [], [], [], []
    accuracys = []

    for train_index, text_index in kf.split(X):
        X_train, X_test = X[train_index], X[text_index]
        y_train, y_test = Y[train_index], Y[text_index]
        model = get_model()

        if model_name == "CNN" or model_name == "DNN":
            history = model.fit(X_train, y_train, epochs=50, batch_size=11, shuffle=True)
            accuracys.append(history.history['accuracy'])
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
        'ErrorRate': err,
        'Pesticide_MAE': add_maes,
        'Pesticide_MSE': add_mses,
        'Pesticide_ErrorRate': add_err
    }, np.mean(np.array(accuracys), axis=0)


if __name__ == "__main__":
    prev_data = PAHRatio('Data', m=10, clustering_threshold=20, K=5, maxIt=100,window_size = 50)
    prev_data.process_files()
    prev_data.prepare_model_data()

    train_X = np.array(prev_data.cnn_inputs)
    train_Y = np.array(prev_data.cnn_labels)
    test_data = PAHRatio('test_data', m=10, clustering_threshold=20, K=5, maxIt=100,window_size = 50)
    test_data.process_files()
    test_data.prepare_model_data()
    test_X = np.array(test_data.cnn_inputs)
    test_Y = np.array(test_data.cnn_labels)

    train_X = np.load('train_X.npy')
    train_Y = np.load('train_Y.npy')
    test_X = np.load('test_X.npy')
    test_Y = np.load('test_Y.npy')

    # The following code are used for testing RF, CNN, DNN, and SVR models, to generate figures used in paper.
    # To reproduce results, just uncomment all the following code and run them.
    
    print(np.array(train_X).shape)

    # This section reshapes 3D data for cnn input into 2D to fit into a standard scalar
    # and then the input is reshaped back into its original shape
    cnn_input_reshape = train_X.reshape(train_X.shape[0], -1)
    cnn_test_input_reshape = test_X.reshape(test_X.shape[0], -1)
    # print(np.array(cnn_input_reshape).shape)
    cnn_scalar = StandardScaler()
    cnn_scalar.fit_transform(cnn_input_reshape)
    cnn_scalar.transform(cnn_test_input_reshape)
    cnn_train_input = np.array(cnn_input_reshape).reshape(train_X.shape)
    cnn_test_input = np.array(np.array(cnn_test_input_reshape).reshape(test_X.shape))

    dnn_X_train = train_X.reshape(train_X.shape[0], -1)
    dnn_X_test = test_X.reshape(test_X.shape[0], -1)
    dnn_scaler = StandardScaler()
    dnn_scaler.fit(dnn_X_train)
    dnn_X_train = dnn_scaler.transform(dnn_X_train)
    dnn_X_test = dnn_scaler.transform(dnn_X_test)

    model_metrics = {}
    accuracy = {}

    model_metrics['CNN'], accuracy['CNN'] = perform_kfold_cv(cnn_train_input, train_Y, get_cnn_model, "CNN", cnn_test_input, test_Y)

    model_metrics['DNN'], accuracy['DNN'] = perform_kfold_cv(dnn_X_train, train_Y, get_dnn_model, "DNN", dnn_X_test, test_Y)

    plt.figure(figsize=(10, 6))
    plt.plot(accuracy['CNN'], color='r', label='CNN')
    plt.plot(accuracy['DNN'], c='b', label='DNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("learningCurve.png")
    plt.show()

    model_metrics['SVR'], accuracy['SVR'] = perform_kfold_cv(dnn_X_train, train_Y, get_svr_model, "SVR", dnn_X_test, test_Y)
    model_metrics['RF'], accuracy['RF'] = perform_kfold_cv(dnn_X_train, train_Y, get_rf_model, "RF", dnn_X_test, test_Y)

    # print(model_metrics)
    col_names = model_metrics['CNN'].keys()
    table = PrettyTable()
    table.field_names = np.hstack((np.array(['model_name']), np.array(list(col_names))))

    # Iteratively create a table of avg all metrics
    for model in model_metrics.items():
        avg_metrics = []
        for it in model[1].items():
            avg_metrics.append(round(float(np.mean(it[1])), ndigits=4))

        model_name = [model[0]]
        model_name.extend(avg_metrics)
        table.add_row(model_name)

    # print(table)
    # This is to put the table into an image
    table_str = table.get_string()
    image = Image.new('RGB', (1000, 300), color=(255, 255, 255))
    drawing = ImageDraw.Draw(image)
    drawing.text(xy=(10, 10), text=table_str, fill=(0, 0, 0))
    image.save("FinalPaperUseCOMPARISON.png")

    keras.utils.plot_model(get_cnn_model(), to_file='cnn_model.png', show_shapes=True)
    keras.utils.plot_model(get_dnn_model(), to_file='dnn_model.png', show_shapes=True) 
    