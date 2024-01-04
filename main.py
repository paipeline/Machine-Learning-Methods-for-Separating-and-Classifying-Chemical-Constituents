from train_evaluate import train_and_evaluate_all_models
import numpy as np

if __name__ == "__main__":
    train_val_X = np.load('dataset/model_inputs.npy')
    train_val_Y = np.load('dataset/model_labels.npy')

    test_X_Pesticide = np.load('dataset/pesticide_inputs.npy')
    test_Y_Pesticide = np.load('dataset/pesticide_labels.npy')
    test_X_PAH = np.load('dataset/test_inputs.npy')
    test_Y_PAH = np.load('dataset/test_labels.npy')
    print("Shape of PAH train_val X size:", train_val_X.shape)
    print("Shape of PAH train_val Y size:", train_val_Y.shape)
    print("Shape of PAH test X size:", test_X_PAH.shape)
    print("Shape of PAH tset Y size:", test_Y_PAH.shape)
    print("Shape of Pesticide test X size:", test_X_Pesticide.shape)
    print("Shape of Pesticide tset Y size:", test_Y_Pesticide.shape)
    print("-------------------------------------------------------")
    print("--- MSE on test PAH ---")
    train_and_evaluate_all_models(train_val_X, train_val_Y, test_X_PAH, test_Y_PAH)
    print("--- MSE on Pesticide ---")
    train_and_evaluate_all_models(train_val_X, train_val_Y, test_X_Pesticide, test_Y_Pesticide)

