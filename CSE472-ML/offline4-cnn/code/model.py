from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, accuracy_score

import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm

DATASET_DIR = "../../../../numta"
IMAGE_SHAPE = (32, 32)
# DATASET_NAME = "training-de-all-32"
DATASET_NAME = "training-a"


def load_dataset(test_size=0.2, shuffle=False):
    dataset = f"{DATASET_DIR}/{DATASET_NAME}.csv"
    df = pd.read_csv(dataset)
    
    X = df["filename"]
    y = df["digit"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    return X_train, X_valid, y_train, y_valid


def load_image_as_grayscale(image_path):    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SHAPE)
    img = (255 - img) / 255
    return np.array([ img ])


def load_image_as_rgb(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SHAPE)
    img = (255 - img.transpose(2, 0, 1)) / 255
    return np.array(img)



class CNN:

    def __init__(self, learning_rate=0.0005):
        self.layers = []
        self.learning_rate = learning_rate
    

    def add(self, layer):
        self.layers.append(layer)

    
    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32):

        print("Training Started")
        X_valid = np.array([ load_image_as_grayscale(f"{DATASET_DIR}/{DATASET_NAME}/{img_name}") for img_name in X_valid ])
        y_valid = np.array([ np.eye(10)[digit] for digit in y_valid ], dtype=int)

        for epoch in range(epochs):

            n_batch = int(np.ceil(len(X_train)/batch_size))
            loss = 0
            accu = 0

            for batch in tqdm(range(n_batch)):
                #   reading batch of image_name from CSV
                #   and loading them as grayscale
                X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
                X_batch = np.array([ load_image_as_grayscale(f"{DATASET_DIR}/{DATASET_NAME}/{img_name}") for img_name in X_batch ])

                #   reading batch of labels from CSV
                #   and converting them to one-hot encoding
                y_true = y_train[batch*batch_size : (batch+1)*batch_size]
                y_true = np.array([ np.eye(10)[digit] for digit in y_true ], dtype=int)

                #   forward pass
                y_pred = np.copy(X_batch)
                for layer in self.layers:
                    y_pred = layer.forward(y_pred)

                y_true_hot = np.argmax(y_true, axis=1)
                y_pred_hot = np.argmax(y_pred, axis=1)

                loss += log_loss(y_true, y_pred)
                accu += accuracy_score(y_true_hot, y_pred_hot)

                dl = np.copy(y_pred) - y_true
                dl /= batch_size

                #   backward pass
                for layer in reversed(self.layers):
                    dl = layer.backward(dl, self.learning_rate)

            #  training loss
            train_loss = loss / n_batch
            train_accu = accu / n_batch

            y_pred = self.predict(X_valid)
            y_valid_hot = np.argmax(y_valid, axis=1)
            y_pred_hot = np.argmax(y_pred, axis=1) 

            print("y_pred_hot")
            print(y_pred_hot)
        
            print("y_valid_hot")
            print(y_valid_hot)

            loss = log_loss(y_valid, y_pred)
            accuracy = accuracy_score(y_valid_hot, y_pred_hot)
            error = mean_squared_error(y_valid, y_pred)

            print(f"Epoch: {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Train Accuracy: {train_accu:.4f}")
            print(f"Epoch: {epoch+1}/{epochs} Validation Accuracy: {accuracy:.4f} Error: {error:.4f} CE Loss: {loss:.4f}")


    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

