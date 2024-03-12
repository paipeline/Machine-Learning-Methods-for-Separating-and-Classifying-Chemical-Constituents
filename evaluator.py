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
#keras.utils.set_random_seed(RANDOM_SEED)
class evaluator:
    def __init__(self, **kwargs):
        # Default parameter values
        default_params = {
            'm': 10,
            'windows_size': 50,
            'clustering_threshold': 20,
            'K': 5,
            'maxIt': 100,
            'baseline_removal': True
        }
        
        
        # Update defaults with any kwargs provided
        self.params = {**default_params, **kwargs}
        
        if kwargs:  # If there are any keyword arguments
            self.create_import_data(
                m=self.params['m'],
                windows_size=self.params['windows_size'],
                clustering_threshold=self.params['clustering_threshold'],
                K=self.params['K'],
                maxIt=self.params['maxIt'],
                baseline_removal=self.params['baseline_removal']
            )
        self.metrics = {}
        # Load the data
        self.train_X = np.load('./dataset/seperate_test_data/pah/pah_train_X.npy')
        self.train_Y = np.load('./dataset/seperate_test_data/pah/pah_train_Y.npy')
        self.test_X = np.load('./dataset/seperate_test_data/pah/pah_test_X.npy')
        self.test_Y = np.load('./dataset/seperate_test_data/pah/pah_test_Y.npy')
        self.pes_X = np.load('./dataset/seperate_test_data/pes/pes_X.npy')
        self.pes_Y = np.load('./dataset/seperate_test_data/pes/pes_Y.npy')

        self.dnn_X_train = self.train_X.reshape(self.train_X.shape[0], -1)
        self.dnn_X_pes = self.pes_X.reshape(self.pes_X.shape[0], -1)
        self.dnn_test_x = self.test_X.reshape(self.test_X.shape[0], -1)
        self.dnn_scaler = StandardScaler()
        self.dnn_scaler.fit(self.dnn_X_train) 
        self.dnn_X_train = self.dnn_scaler.transform(self.dnn_X_train)
        self.dnn_X_pes = self.dnn_scaler.transform(self.dnn_X_pes)
        self.dnn_X_test = self.dnn_scaler.transform(self.dnn_test_x)
        # This section reshapes 3D data for cnn input into 2D to fit into a standard scalar
        # and then the input is reshaped back into its original shape
        self.cnn_input_reshape = self.train_X.reshape(self.train_X.shape[0], -1)
        self.cnn_test_input_reshape = self.test_X.reshape(self.test_X.shape[0], -1)
        self.cnn_pes = self.pes_X.reshape(self.pes_X.shape[0], -1)
        # print(np.array(cnn_input_reshape).shape)
        cnn_scalar = StandardScaler()
        cnn_scalar.fit_transform(self.cnn_input_reshape)
        cnn_scalar.transform(self.cnn_test_input_reshape)
        self.cnn_train_input = np.array(self.cnn_input_reshape).reshape(self.train_X.shape)
        self.cnn_test_input = np.array(np.array(self.cnn_test_input_reshape).reshape(self.test_X.shape))
        self.cnn_pes_input = np.array(np.array(self.cnn_pes).reshape(self.pes_X.shape))
        
    def custom_loss(self,y_true, y_pred):
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

    def get_cnn_model(self):
        # Fix initial weights to make results reproducible
        weight_initializer = keras.initializers.GlorotUniform(seed=RANDOM_SEED)
        # This model is already updated to the newest
        cnn_model = Sequential([ 
            layers.Conv1D(filters=400, kernel_size=2, activation='tanh',
                        input_shape=(self.cnn_train_input.shape[1], self.cnn_train_input.shape[2]), kernel_initializer=weight_initializer,
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

        cnn_model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=['accuracy'])

        return cnn_model

    def get_rf_model(self):
        random_forest = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED)
        return random_forest

    def get_dnn_model(self):
        # Fix initial weights to make results reproducible
        weight_initializer = keras.initializers.GlorotUniform(seed=RANDOM_SEED)
        dnn_model = Sequential([
            layers.Dense(1024, activation='relu', input_shape=[self.dnn_X_train.shape[1]], kernel_initializer=weight_initializer,
                        bias_initializer=initializers.Zeros()),
            layers.Dense(512, activation='relu', input_shape=[self.dnn_X_train.shape[1]], kernel_initializer=weight_initializer,
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
        dnn_model.compile(optimizer='adam', loss=self.custom_loss, metrics=['accuracy'])

        return dnn_model

    def get_svr_model(self):
        svr_model = MultiOutputRegressor(SVR(kernel='rbf'))
        return svr_model





    def print_metrics(self,mae, mse, model_name, r2, err):
        print("==================================")
        print(f"model: {model_name}")
        print(f"mae avg: {mae}")
        print(f"r2 avg: {r2}")
        print(f"mse avg: {mse}")
        print(f"Error Rate: {err}")
        results_df = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2', 'ErrorRate'])


    def final_test(self,model_name):
        
        model = self.get_cnn_model() if model_name == "CNN" else self.get_dnn_model() if model_name == "DNN" else self.get_svr_model() if model_name == "SVR" else self.get_rf_model()
        if model_name == "CNN":
            model.fit(self.cnn_train_input, self.train_Y, epochs=50, batch_size=4)
            prediction_pah = model.predict(self.cnn_test_input)
            prediction_pes = model.predict(self.cnn_pes_input)
        elif model_name == "DNN":
            model.fit(self.dnn_X_train, self.train_Y, epochs=50, batch_size=4)
            prediction_pah = model.predict(self.dnn_X_test)
            prediction_pes = model.predict(self.dnn_X_pes)
        else:
            model.fit(self.dnn_X_train, self.train_Y)
            prediction_pah = model.predict(self.dnn_X_test)
            prediction_pes = model.predict(self.dnn_X_pes)

        rounded_pah_prediction = np.round(prediction_pah)
        rounded_pes_prediction = np.round(prediction_pes)
        pah_error = np.mean(np.abs(rounded_pah_prediction - self.test_Y) > 0)
        pes_error = np.mean(np.abs(rounded_pes_prediction - self.pes_Y) > 0)
        precise_difference_pah_test = np.sum(np.abs(prediction_pah - self.test_Y))
        precise_difference_pes_test = np.sum(np.abs(prediction_pes - self.pes_Y))
        if model_name not in self.metrics:
            self.metrics[model_name] = {'pah_error': pah_error, 'pes_error': pes_error, 'precise_difference_pah': precise_difference_pah_test, 'precise_difference_pes': precise_difference_pes_test}
                
        # load metrics into a dictionary to be used for plotting later
        print("**********************************")
        print(f"pah Error Rate of {model_name}: {pah_error}")
        print(f"pes Error Rate of {model_name}: {pes_error}")
        print(f"The exact error amount of pah test data accross all data of {model_name}: {precise_difference_pah_test}")
        print(f"The exact error amount of pes test data accross all data of {model_name}: {precise_difference_pes_test}")
        print("**********************************")

    def perform_kfold_cv(self,X, Y, get_model, model_name, pes_x, pex_y, k=5):
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
    def create_import_data(self, m = 10,windows_size = 50,clustering_threshold = 20,K = 5, maxIt =100,baseline_removal = True):
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


    def plot_results(self):
        # Error Rates Plot
        plt.figure(figsize=(12, 6))
        model_names = list(self.metrics.keys())
        pah_errors = [self.metrics[model]['pah_error'] for model in model_names]
        pes_errors = [self.metrics[model]['pes_error'] for model in model_names]
        
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
        pah_precise_differences = [self.metrics[model]['precise_difference_pah'] for model in model_names]
        pes_precise_differences = [self.metrics[model]['precise_difference_pes'] for model in model_names]

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
