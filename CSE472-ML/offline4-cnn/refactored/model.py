"""
Model file where we implement the CNN model
"""
from matplotlib import pyplot as plt
from sklearn.metrics import *
import numpy as np
import copy
import pickle
import tqdm
import os 

# Local Imports
from utils import *



class CNN:

    def __init__(self, learning_rate=0.0005):
        self.layers = []
        self.learning_rate = learning_rate
    

    def copy(self):
        return copy.deepcopy(self)


    def add(self, layer):
        self.layers.append(layer)


    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X


    def save_model(self, model_name):
        for layer in self.layers:
            if hasattr(layer, "cache"):
                layer.cache = None
        pickle.dump(self, open(model_name, "wb"))
    

    def read_model(self, file_name):
        self = pickle.load(open(file_name, "rb"))


    def train(self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32):

        print("Training Started")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Batch Size: {batch_size}")

        arr_train_loss = []
        arr_valid_loss = []
        arr_valid_accu = []
        arr_valid_f1sc = []
        arr_epochs = range(1, epochs+1)

        X_valid = np.array([ load_image_as_grayscale(f"{DATASET_DIR}/{DATASET_NAME}/{img_name}") for img_name in X_valid ])
        y_valid = np.array([ np.eye(10)[digit] for digit in y_valid ], dtype=int)

        if not os.path.exists(REPORT_FOLDER):
            os.mkdir(REPORT_FOLDER)

        for epoch in range(epochs):

            n_batch = int(np.ceil(len(X_train)/batch_size))
            train_loss = 0

            for batch in tqdm(range(n_batch)):
                X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
                X_batch = np.array([ load_image_as_grayscale(f"{DATASET_DIR}/{DATASET_NAME}/{img_name}") for img_name in X_batch ])

                y_true = y_train[batch*batch_size : (batch+1)*batch_size]
                y_true = np.array([ np.eye(10)[digit] for digit in y_true ], dtype=int)

                #   forward pass
                y_pred = np.copy(X_batch)
                for layer in self.layers:
                    y_pred = layer.forward(y_pred)

                train_loss += log_loss(y_true, y_pred)

                dl = (y_pred - y_true)
                dl /= batch_size

                #   backward pass
                for layer in reversed(self.layers):
                    dl = layer.backward(dl, self.learning_rate)

            y_pred = self.predict(X_valid)
            y_pred_hot = np.argmax(y_pred, axis=1) 
            y_valid_hot = np.argmax(y_valid, axis=1)

            # print("y_pred_hot")
            # print(y_pred_hot)        
            # print("y_valid_hot")
            # print(y_valid_hot)

            #  training loss
            train_loss = (train_loss/n_batch)

            #  validation loss
            valid_loss = log_loss(y_valid, y_pred)
            valid_accu = accuracy_score(y_valid_hot, y_pred_hot) * 100
            valid_f1   = f1_score(y_valid_hot, y_pred_hot, average='macro') * 100

            arr_train_loss.append(train_loss)
            arr_valid_loss.append(valid_loss)
            arr_valid_accu.append(valid_accu)
            arr_valid_f1sc.append(valid_f1)

            conf_mat = confusion_matrix(y_valid_hot, y_pred_hot)
            conf_plot = ConfusionMatrixDisplay(conf_mat, display_labels=[0,1,2,3,4,5,6,7,8,9])
            conf_plot.plot()

            plt.savefig(f"{REPORT_FOLDER}/confusion_matrix_{epoch+1}.png")
            plt.clf()

            print(f"[TRAIN] Epoch: {epoch+1}/{epochs} Loss: {train_loss:.4f}")
            print(f"[VALID] Epoch: {epoch+1}/{epochs} Loss: {valid_loss:.4f} Accuracy: {valid_accu:.4f} F1: {valid_f1:.4f}")


        plt.plot(arr_epochs, arr_train_loss, label="Training Loss")
        plt.plot(arr_epochs, arr_valid_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{REPORT_FOLDER}/loss.png")
        plt.clf()

        plt.plot(arr_epochs, arr_valid_accu, label="Validation Accuracy")
        plt.plot(arr_epochs, arr_valid_f1sc, label="Validation F1 Score")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{REPORT_FOLDER}/accuracy.png")
        plt.clf()
