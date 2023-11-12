from train_evaluate import train_and_evaluate_all_models
import numpy as np

if __name__ == "__main__":
    train_val_X = np.load('dataset/model_inputs.npy')
    train_val_Y = np.load('dataset/model_labels.npy')
    # test_X_PAH
    # test_Y_PAH
    test_X_pesticide = np.load('dataset/test_X.npy')
    test_Y_pesticide = np.load('dataset/test_Y.npy')
    print("Shape of PAH train_val X size:", train_val_X.shape)
    print("Shape of PAH train_val Y size:", train_val_Y.shape)
    print("Shape of Pesticide test X size:", test_X_pesticide.shape)
    print("Shape of Pesticide tset Y size:", test_Y_pesticide.shape)
    print("-------------------------------------------------------")
    train_and_evaluate_all_models(train_val_X, train_val_Y, test_X_pesticide, test_Y_pesticide)
