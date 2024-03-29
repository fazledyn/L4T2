from sklearn.metrics import log_loss, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from tqdm import tqdm

import pandas as pd
import numpy as np
import pickle
import copy
import cv2
import os

np.random.seed(66)

#################################################################################################################################################
#                                                               LAYER DEFINITION
#################################################################################################################################################


class ReLUActivationLayer:

    def __str__(self) -> str:
        return "ReLU"

    def forward(self, input):
        out = np.maximum(input, 0)
        return out

    def backward(self, output, learning_rate):
        return np.where(output > 0, 1, 0)


class MaxPoolingLayer:

    def __str__(self) -> str:
        return "Maxpool"

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, input):

        self.cache = input
        batch_size, n_channel, height, width = input.shape

        output_h = int((height - self.pool_size)/self.stride + 1)
        output_w = int((width  - self.pool_size)/self.stride + 1)

        output_shape = (batch_size, n_channel, output_h, output_w)
        output = np.zeros(output_shape)

        for b in range(batch_size):
            for c in range(n_channel):
                for h in range(output_h):
                    for w in range(output_w):
                        output[b, c, h, w] = np.max(input[b, :, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size])

        return output

    def backward(self, output, learning_rate):
        batch_size, n_channel, height, width = output.shape
        input = np.zeros(self.cache.shape)

        for b in range(batch_size):
            for c in range(n_channel):
                for h in range(height):
                    for w in range(width):
                        input[b, c, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size] = np.where(input[b, c, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size] == np.max(input[b, c, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size]), output[b, c, h, w], 0)

        return input    


class FlatteningLayer:

    def __str__(self) -> str:
        return "Flatten"

    def __init__(self) -> None:
        self.cache = None

    def forward(self, input):
        self.cache = input
        return input.reshape(input.shape[0], -1)

    def backward(self, output, learning_rate):
        return output.reshape(self.cache.shape)


class DenseLayer:

    def __str__(self) -> str:
        return "Dense"

    def __init__(self, n_output):
        self.n_output = n_output
        self.weights = None
        self.biases = None
        self.cache = None

    def forward(self, input):
        
        self.cache = input        
        batch_size, n_input = input.shape

        if self.weights is None:
            self.weights = np.random.randn(n_input, self.n_output) / np.sqrt(n_input)

        if self.biases is None:
            self.biases = np.random.randn(self.n_output)

        output = np.dot(input, self.weights) + self.biases
        return output

    def backward(self, output, learning_rate):

        del_w = np.dot(np.transpose(self.cache), output)
        del_b = np.sum(output, axis=0)
        del_u = np.dot(output, np.transpose(self.weights))

        self.weights -= learning_rate * del_w
        self.biases -= learning_rate * del_b
        return del_u


class SoftMaxLayer:

    def __str__(self) -> str:
        return "Softmax"

    def forward(self, input):
        val = input - np.max(input, axis=1, keepdims=True)
        val = np.exp(val) / np.exp(val).sum(axis=1, keepdims=True)
        return val

    def backward(self, output, learning_rate):
        return output



class ConvolutionLayer:

    def __str__(self) -> str:
        return "Convolution"

    def _get_indices(self, X_shape, HF, WF, stride, pad):

        batch_size, n_channel, height, width = X_shape

        out_h = int((height + 2 * pad - HF) / stride) + 1
        out_w = int((width + 2 * pad - WF) / stride) + 1
    
        # ----Compute matrix of index i----
        level1 = np.repeat(np.arange(HF), WF)
        level1 = np.tile(level1, n_channel)
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        # ----Compute matrix of index j----
        slide1 = np.tile(np.arange(WF), HF)
        slide1 = np.tile(slide1, n_channel)

        everySlides = stride * np.tile(np.arange(out_w), out_h)
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
        d = np.repeat(np.arange(n_channel), HF * WF).reshape(-1, 1)

        return i, j, d


    def _im2col(self, X, HF, WF, stride, pad):

        X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self._get_indices(X.shape, HF, WF, stride, pad)
    
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)

        return cols


    def _col2im(self, dX_col, X_shape, HF, WF, stride, pad):

        batch_size, n_channel, height, width = X_shape
        H_padded, W_padded = height + 2 * pad, width + 2 * pad
        X_padded = np.zeros((batch_size, n_channel, H_padded, W_padded))
        
        i, j, d = self._get_indices(X_shape, HF, WF, stride, pad)
        dX_col_reshaped = np.array(np.hsplit(dX_col, batch_size))
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)

        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]
    

    def __init__(self, n_filter, n_channel_in, kernel_size, stride=1, padding=0):
        self.n_filter = n_filter
        self.k_size = kernel_size
        self.n_channel_in = n_channel_in
        self.stride = stride
        self.padding = padding
        self.cache = None

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {
            'val': np.random.randn(self.n_filter, self.n_channel_in, self.k_size, self.k_size) * np.sqrt(1. / (self.k_size)),
            'grad': np.zeros((self.n_filter, self.n_channel_in, self.k_size, self.k_size))
        }
        self.b = {
            'val': np.random.randn(self.n_filter) * np.sqrt(1. / self.n_filter),
            'grad': np.zeros((self.n_filter))
        }

    def forward(self, X):
        batch_size, _, height, width = X.shape

        new_n_channel = self.n_filter
        new_height = int((height + 2 * self.padding - self.k_size)/ self.stride) + 1
        new_width = int((width + 2 * self.padding - self.k_size)/ self.stride) + 1
        
        X_col = self._im2col(X, self.k_size, self.k_size, self.stride, self.padding)
        w_col = self.W['val'].reshape((self.n_filter, -1))
        b_col = self.b['val'].reshape(-1, 1)

        # Perform matrix multiplication.
        out = w_col @ X_col + b_col

        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, new_n_channel, new_height, new_width))
        self.cache = X, X_col, w_col
        return out


    def backward(self, dout, learning_rate):

        X, X_col, w_col = self.cache
        batch_size, _, _, _ = X.shape
        self.b['grad'] = np.sum(dout, axis=(0,2,3))

        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, batch_size))
        dout = np.concatenate(dout, axis=-1)

        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dX_col = w_col.T @ dout
        dw_col = dout @ X_col.T

        # Reshape back to image (col2im).
        dX = self._col2im(dX_col, X.shape, self.k_size, self.k_size, self.stride, self.padding)
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.n_channel_in, self.k_size, self.k_size))                
        return dX


#################################################################################################################################################
#                                                           UTILS FUNCTIONS 
#################################################################################################################################################


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


def save_model(model, model_name):
    for layer in model.layers:
        if hasattr(layer, "cache"):
            layer.cache = None
    pickle.dump(model, open(model_name, "wb"))


#################################################################################################################################################
#                                                               MODEL DEFINITION
#################################################################################################################################################

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



#################################################################################################################################################
#                                                        MAIN SCRIPT 
#################################################################################################################################################

DATASET_DIR = "../../../../numta"
IMAGE_SHAPE = (32, 32)
DATASET_NAME = "training-de-all-32"

REPORT_FOLDER = "./report"
LEARNING_RATE = 0.005


def main():
    
    cnn = CNN(learning_rate=LEARNING_RATE)

    cnn.add(ConvolutionLayer(n_filter=6, n_channel_in=1, kernel_size=5, stride=1, padding=0))
    cnn.add(ReLUActivationLayer())
    cnn.add(MaxPoolingLayer(pool_size=2, stride=2))

    cnn.add(ConvolutionLayer(n_filter=16, n_channel_in=6, kernel_size=5, stride=1, padding=0))
    cnn.add(ReLUActivationLayer())
    cnn.add(MaxPoolingLayer(pool_size=2, stride=2))

    cnn.add(FlatteningLayer())
    
    cnn.add(DenseLayer(n_output=84))
    cnn.add(DenseLayer(n_output=10))

    cnn.add(SoftMaxLayer())
    
    X_train, _, y_train, _ = load_dataset(test_size=0.97, shuffle=True)
    _, X_valid, _, y_valid = load_dataset(test_size=0.01, shuffle=True)
    
    print("Dataset Loaded:", DATASET_NAME)
    print("Train Data Size:", X_train.shape)
    print("Valid Data Size:", X_valid.shape)

    cnn.train(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        batch_size=32,
        epochs=10
    )

    print("Training Complete!")
    save_model(cnn, f"{REPORT_FOLDER}/model_lr{LEARNING_RATE}.pkl")


if __name__ == "__main__":
    main()