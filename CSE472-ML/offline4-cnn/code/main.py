
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, accuracy_score
from matplotlib import pyplot as plt

import numpy as np
import pickle

from layers import *
from model import *

np.random.seed(0)


def main():

    cnn = CNN(learning_rate=0.001)

    cnn.add(ConvolutionLayer(n_filter=6, n_channel_in=1, kernel_size=5, stride=1, padding=0))
    cnn.add(ReLUActivationLayer())
    cnn.add(MaxPoolingLayer(pool_size=2, stride=2))

    cnn.add(ConvolutionLayer(n_filter=16, n_channel_in=6, kernel_size=5, stride=1, padding=0))
    cnn.add(ReLUActivationLayer())
    cnn.add(MaxPoolingLayer(pool_size=2, stride=2))

    cnn.add(FlatteningLayer())
    
    cnn.add(DenseLayer(n_output=120))
    # cnn.add(ReLUActivationLayer())

    cnn.add(DenseLayer(n_output=84))
    # cnn.add(ReLUActivationLayer())

    cnn.add(DenseLayer(n_output=10))
    cnn.add(SoftMaxLayer())
    
    X_train, _, y_train, _ = load_dataset(test_size=0.9, shuffle=True)
    _, X_valid, _, y_valid = load_dataset(test_size=0.01, shuffle=True)
    
    print("Dataset Loaded")
    print("Train Data Size:", X_train.shape)
    print("Valid Data Size:", X_valid.shape)

    cnn.train(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        batch_size=42,
        epochs=10
    )
    
    from datetime import datetime
    pickle.dump(cnn, open(f"model.pkl", "wb"))
    print("Pickle File Saved")


if __name__ == "__main__":
    main()