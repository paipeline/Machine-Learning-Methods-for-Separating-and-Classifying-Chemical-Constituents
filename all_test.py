from keras.src.activations import relu
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src import regularizers
from keras.src.applications.densenet import layers
from keras.src.initializers import initializers
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from prettytable import PrettyTable
from PIL import Image, ImageDraw
from keras import backend
import keras
import random
from src.input_data import PAHRatio


RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
metrics = {}
#keras.utils.set_random_seed(RANDOM_SEED)

def custom_loss(y_true, y_pred):
    # This is a hyperparameter to be tunned
    overestimation_penalty_coefficient = 30
    error = float(y_pred) - float(y_true)
    loss = backend.square(error)
    over_penalty_loss = overestimation_penalty_coefficient * loss
    return backend.mean(backend.switch(error > 0, over_penalty_loss, loss), axis=-1)

""" This is not working since it is non-differentiable
def custom_activation(x):
    return backend.round(keras.src.activations.relu(x))
    """

"""
Note the following models has been modified to the newest. All changes are determined by manual testing, 
comparing, and tuning. It is now deterministic, which means for given input, the model will always end up in
the same ending state and produce the same testing result.
"""

def get_cnn_model(*args):
    # Fix initial weights to make results reproducible
    weight_initializer = keras.initializers.GlorotUniform(seed=RANDOM_SEED)
    # This model is already updated to the newest
    cnn_model = Sequential([ 
        layers.Conv1D(filters=400, kernel_size=2, activation='tanh',
                      input_shape=(args[0].shape[1], args[0].shape[2]), kernel_initializer=weight_initializer,
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

def get_rf_model(*args):
    random_forest = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED)
    return random_forest

def get_dnn_model(*args):
    dnn_X_train = args[0]
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

def get_svr_model(*args):
    svr_model = MultiOutputRegressor(SVR(kernel='rbf'))
    return svr_model





def print_metrics(mae, mse, model_name, r2, err):
    print("==================================")
    print(f"model: {model_name}")
    print(f"mae avg: {mae}")
    print(f"r2 avg: {r2}")
    print(f"mse avg: {mse}")
    print(f"Error Rate: {err}")
    results_df = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2', 'ErrorRate'])


def final_test(model_name):
    train_X = np.load('./dataset/seperate_test_data/pah/pah_train_X.npy')
    train_Y = np.load('./dataset/seperate_test_data/pah/pah_train_Y.npy')
    test_X = np.load('./dataset/seperate_test_data/pah/pah_test_X.npy')
    test_Y = np.load('./dataset/seperate_test_data/pah/pah_test_Y.npy')
    pes_X = np.load('./dataset/seperate_test_data/pes/pes_X.npy')
    pes_Y = np.load('./dataset/seperate_test_data/pes/pes_Y.npy')

    dnn_X_train = train_X.reshape(train_X.shape[0], -1)
    dnn_X_pes = pes_X.reshape(pes_X.shape[0], -1)
    dnn_test_x = test_X.reshape(test_X.shape[0], -1)
    dnn_scaler = StandardScaler()
    dnn_scaler.fit(dnn_X_train) 
    dnn_X_train = dnn_scaler.transform(dnn_X_train)
    dnn_X_pes = dnn_scaler.transform(dnn_X_pes)
    dnn_X_test = dnn_scaler.transform(dnn_test_x)

    # This section reshapes 3D data for cnn input into 2D to fit into a standard scalar
    # and then the input is reshaped back into its original shape
    cnn_input_reshape = train_X.reshape(train_X.shape[0], -1)
    cnn_test_input_reshape = test_X.reshape(test_X.shape[0], -1)
    cnn_pes = pes_X.reshape(pes_X.shape[0], -1)
    # print(np.array(cnn_input_reshape).shape)
    cnn_scalar = StandardScaler()
    cnn_scalar.fit_transform(cnn_input_reshape)
    cnn_scalar.transform(cnn_test_input_reshape)
    cnn_train_input = np.array(cnn_input_reshape).reshape(train_X.shape)
    cnn_test_input = np.array(np.array(cnn_test_input_reshape).reshape(test_X.shape))
    cnn_pes_input = np.array(np.array(cnn_pes).reshape(pes_X.shape))
    
    model = get_cnn_model(cnn_train_input) if model_name == "CNN" else get_dnn_model(dnn_X_train) if model_name == "DNN" else get_svr_model(dnn_X_train) if model_name == "SVR" else get_rf_model(dnn_X_train)
    if model_name == "CNN" or model_name == "DNN":
        model.fit(train_X, train_Y, epochs=50, batch_size=4)
    else:
        model.fit(train_X, train_Y)

    prediction_pah = model.predict(test_X)
    prediction_pes = model.predict(pes_X)
    rounded_pah_prediction = np.round(prediction_pah)
    rounded_pes_prediction = np.round(prediction_pes)
    pah_error = np.mean(np.abs(rounded_pah_prediction - test_Y) > 0)
    pes_error = np.mean(np.abs(rounded_pes_prediction - pes_Y) > 0)

    precise_difference_pah_test = np.sum(np.abs(prediction_pah - test_Y))
    precise_difference_pes_test = np.sum(np.abs(prediction_pes - pes_Y))

    # load metrics into a dictionary to be used for plotting later
    metrics[model_name] = [pah_error, pes_error, precise_difference_pah_test, precise_difference_pes_test]
    print("**********************************")
    print(f"pah Error Rate of {model_name}: {pah_error}")
    print(f"pes Error Rate of {model_name}: {pes_error}")
    print(f"The exact error amount of pah test data accross all data of {model_name}: {precise_difference_pah_test}")
    print(f"The exact error amount of pes test data accross all data of {model_name}: {precise_difference_pes_test}")
    print("**********************************")

def perform_kfold_cv(X, Y, get_model, model_name, pes_x, pex_y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    maes, r2s, mses, err = [], [], [], []
    add_maes, add_r2s, add_mses, add_err = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model = get_model()

        if model_name == "CNN" or model_name == "DNN":
            model.fit(X_train, y_train, epochs=50, batch_size=4)
        else:
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        add_pred = model.predict(pes_x)

        # This can use a custom threshold (can be obtained from gridSearch)
        rounded_pred = np.round(pred)
        rounded_add_pred = np.round(add_pred)

        error = np.mean(np.abs(rounded_pred - y_test) > 0)
        add_error = np.mean(np.abs(rounded_add_pred - pex_y) > 0)
        add_maes.append(mean_absolute_error(pex_y, add_pred))
        add_mses.append(mean_squared_error(pex_y, add_pred))
        add_r2s.append(r2_score(pex_y, add_pred))
        maes.append(mean_absolute_error(y_test, pred))
        mses.append(mean_squared_error(y_test, pred))
        r2s.append(r2_score(y_test, pred))
        err.append(error)
        add_err.append(add_error)

        mae_avg = np.mean(maes)
        mse_avg = np.mean(mses)
        r2_avg = np.mean(r2s)
        err_avg = np.mean(err)

        results_df = results_df.append({'Model': model_name, 'MAE': mae_avg, 'MSE': mse_avg, 'R2': r2_avg, 'ErrorRate': err_avg}, ignore_index=True)
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

"""
This is using our original hyperparameters. Future improvements can be made by fine tuning these parameters.
This is preprocessed with rolling windows and without baseline-removal.
"""
def create_import_data(m = 10,windows_size = 50,clustering_threshold = 20,K = 5, maxIt =100,baseline_removal = True):
    # PAH
    pah_data_path = "./dataset/raw/pah"
    pah_data = PAHRatio(pah_data_path, m,windows_size, clustering_threshold, K, maxIt, baseline_removal)
    pah_data.process_files()
    pah_data.prepare_model_data()
    # Here I am using model_inputs (with no portion left out) to reproduce results from previous attempts.
    # In theory, we need to use .train_inputs and .train_labels.
    pah_train_X = np.array(pah_data.train_inputs)
    pah_train_Y = np.array(pah_data.train_labels)
    # TODO Need to be adjusted to use the training set for comparing preprocessing steps.
    pah_test_X = np.array(pah_data.test_inputs)
    pah_test_Y = np.array(pah_data.test_labels)

    # Pesticides 
    pes_data_path = "./dataset/raw/pes"
    pes_data = PAHRatio(pes_data_path, m,windows_size, clustering_threshold, K, maxIt) 
    pes_data.process_files()
    pes_data.prepare_model_data(test_size=0) # No split for pes since they are all used for testing
    pes_X = np.array(pes_data.model_inputs)
    pes_Y = np.array(pes_data.model_labels)
    np.save("./dataset/seperate_test_data/pah/pah_train_X.npy", pah_train_X)
    np.save("./dataset/seperate_test_data/pah/pah_train_Y.npy", pah_train_Y)
    np.save("./dataset/seperate_test_data/pah/pah_test_X.npy", pah_test_X)
    np.save("./dataset/seperate_test_data/pah/pah_test_Y.npy", pah_test_Y)
    np.save("./dataset/seperate_test_data/pes/pes_X.npy", pes_X)
    np.save("./dataset/seperate_test_data/pes/pes_Y.npy", pes_Y)


def plot_results(metrics):
    # Error Rates Plot
    plt.figure(figsize=(12, 6))
    model_names = list(metrics.keys())
    pah_errors = [metrics[model]['pah_error'] for model in model_names]
    pes_errors = [metrics[model]['pes_error'] for model in model_names]
    
    x = list(range(len(model_names)))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar([i - width/2 for i in x], pah_errors, width, label='PAH Error')
    rects2 = ax.bar([i + width/2 for i in x], pes_errors, width, label='Pesticide Error')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error Rates')
    ax.set_title('Error Rates by Model and Data Type')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    fig.tight_layout()
    
    plt.savefig("temp_error_rates.png")  # Save the plot to the current working directory
    plt.show()

    # Precise Differences Plot
    plt.figure(figsize=(12, 6))
    pah_precise_differences = [metrics[model]['pah_precise_difference'] for model in model_names]
    pes_precise_differences = [metrics[model]['pes_precise_difference'] for model in model_names]

    fig, ax = plt.subplots()
    rects1 = ax.bar([i - width/2 for i in x], pah_precise_differences, width, label='PAH Precise Difference')
    rects2 = ax.bar([i + width/2 for i in x], pes_precise_differences, width, label='Pesticide Precise Difference')

    ax.set_ylabel('Precise Differences')
    ax.set_title('Precise Differences by Model and Data Type')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    fig.tight_layout()

    plt.savefig("temp_precise_differences.png")  # Save the plot to the current working directory
    plt.show()




def run():
    # This is for creating data usable for models
    #create_import_data()

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
    train_X = np.load('./dataset/seperate_test_data/pah/pah_train_X.npy')
    train_Y = np.load('./dataset/seperate_test_data/pah/pah_train_Y.npy')
    test_X = np.load('./dataset/seperate_test_data/pah/pah_test_X.npy')
    test_Y = np.load('./dataset/seperate_test_data/pah/pah_test_Y.npy')
    pes_X = np.load('./dataset/seperate_test_data/pes/pes_X.npy')
    pes_Y = np.load('./dataset/seperate_test_data/pes/pes_Y.npy')

    # Normalization is key to great result
    dnn_X_train = train_X.reshape(train_X.shape[0], -1)
    dnn_X_pes = pes_X.reshape(pes_X.shape[0], -1)
    dnn_test_x = test_X.reshape(test_X.shape[0], -1)
    dnn_scaler = StandardScaler()
    dnn_scaler.fit(dnn_X_train) 
    dnn_X_train = dnn_scaler.transform(dnn_X_train)
    dnn_X_pes = dnn_scaler.transform(dnn_X_pes)
    dnn_X_test = dnn_scaler.transform(dnn_test_x)

    # This section reshapes 3D data for cnn input into 2D to fit into a standard scalar
    # and then the input is reshaped back into its original shape
    cnn_input_reshape = train_X.reshape(train_X.shape[0], -1)
    cnn_test_input_reshape = test_X.reshape(test_X.shape[0], -1)
    cnn_pes = pes_X.reshape(pes_X.shape[0], -1)
    # print(np.array(cnn_input_reshape).shape)
    cnn_scalar = StandardScaler()
    cnn_scalar.fit_transform(cnn_input_reshape)
    cnn_scalar.transform(cnn_test_input_reshape)
    cnn_train_input = np.array(cnn_input_reshape).reshape(train_X.shape)
    cnn_test_input = np.array(np.array(cnn_test_input_reshape).reshape(test_X.shape))
    cnn_pes_input = np.array(np.array(cnn_pes).reshape(pes_X.shape))

    return cnn_train_input, train_Y, cnn_test_input, test_Y, cnn_pes_input, pes_Y, dnn_X_train, train_Y, dnn_X_test, test_Y, dnn_X_pes, pes_Y

    """
    model_metrics = {}
    # model_metric["U"] = perform_kfold_cv(dnn_X_train, train_Y,get_unet_model,"U-net",dnn_X_test,test_Y)

    model_metrics['CNN'] = perform_kfold_cv(cnn_train_input, train_Y, get_cnn_model, "CNN", cnn_pes_input, pes_Y)
    model_metrics['DNN'] = perform_kfold_cv(dnn_X_train, train_Y, get_dnn_model, "DNN", dnn_X_pes, pes_Y)
    model_metrics['SVR'] = perform_kfold_cv(dnn_X_train, train_Y, get_svr_model, "SVR", dnn_X_pes, pes_Y)
    model_metrics['RF'] = perform_kfold_cv(dnn_X_train, train_Y, get_rf_model, "RF", dnn_X_pes, pes_Y)
    keys = model_metrics['CNN'].keys()
    for key in keys:
        plot_metrics(key, model_metrics)

    # For producing the comparison table, the result we previously had is using all data to do k-fold, lefting out
    # no testing data.
    # TODO Comparing preprocessing w or w/o baseline_removal & w or w/o rolling window

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
 
    # This is to put the table into an image
    table_str = table.get_string()
    image = Image.new('RGB', (1000, 300), color=(255, 255, 255))
    drawing = ImageDraw.Draw(image)
    drawing.text(xy=(10, 10), text=table_str, fill=(0, 0, 0))
    image.save("./fig/comparison_chart.png")
    """


if __name__ == "__main__":
    create_import_data(m = 10,windows_size = 1,clustering_threshold = 20,K = 5, maxIt =100)
    run()
    create_import_data(m = 10,windows_size = 50,clustering_threshold = 20,K = 5, maxIt =100)
    run()
    results_df.to_csv('learning_results_.csv', index=False)

    create_import_data(m = 10,windows_size = 1,clustering_threshold = 20,K = 5, maxIt =100,baseline_removal = True)
    run()
    create_import_data(m = 10,windows_size = 50,clustering_threshold = 20,K = 5, maxIt =100,baseline_removal = True)
    run()
    results_df.to_csv('learning_results_withBaselineRemoval.csv', index=False)




    final_test(cnn_train_input, train_Y, cnn_test_input, test_Y, cnn_pes_input, pes_Y, get_cnn_model(), "CNN")
    final_test(dnn_X_train, train_Y, dnn_X_test, test_Y, dnn_X_pes, pes_Y, get_dnn_model(), "DNN")
    final_test(dnn_X_train, train_Y, dnn_X_test, test_Y, dnn_X_pes, pes_Y, get_svr_model(), "SVR")
    final_test(dnn_X_train, train_Y, dnn_X_test, test_Y, dnn_X_pes, pes_Y, get_rf_model(), "RF")
    plot_results(metrics)
