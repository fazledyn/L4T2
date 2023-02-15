from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import cv2


DATASET_DIR = "../../../../numta"
IMAGE_SHAPE = (32, 32)
DATASET_NAME = "training-de-all-32"

REPORT_FOLDER = "./report"
LEARNING_RATE = 0.005



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
