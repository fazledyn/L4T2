from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from train_1705066 import *
from tqdm import tqdm


import numpy as np
import pandas as pd
import pickle
import sys
import os


def main():

    cnn = CNN()
    cnn = pickle.load(open('1705066_model.pickle', 'rb'))
    print("CNN Loaded from Pickle File!")

    input_folder = sys.argv[1]
    input_files = os.listdir(input_folder)

    exts = [ "png", "jpg", "jpeg", "PNG", "JPG", "JPEG" ]
    input_files = [ file for file in input_files if file.split(".")[-1] in exts ]

    BATCH_SIZE = 32
    print(f"Prediction will be done in batches of {BATCH_SIZE}")
    n_batch = int(np.ceil(len(input_files)/BATCH_SIZE))

    y_pred = []
    for i in tqdm(range(n_batch)):
        X_batch = input_files[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        X_batch = np.array([ load_image_as_grayscale(f"{input_folder}/{filename}") for filename in X_batch ])
        y_batch = cnn.predict(X_batch)
        y_pred.extend(np.argmax(y_batch, axis=1))

    df = pd.DataFrame()
    df["FileName"] = input_files
    df["Digit"] = y_pred

    df.to_csv(f"{input_folder}/1705066_prediction.csv", index=False)
    print("CSV File Generated!")


def compare_testset(testfile, predfile):
    
        test_df = pd.read_csv(testfile)
        pred_df = pd.read_csv(predfile)
    
        test_df = test_df.sort_values(by=['filename'])
        pred_df = pred_df.sort_values(by=['FileName'])
    
        test_df = test_df.reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)

        y_true = test_df['digit'].values
        y_pred = pred_df['Digit'].values

        print("Accuracy: ", accuracy_score(y_true, y_pred))
        print("F1 Score: ", f1_score(y_true, y_pred, average='macro'))

        conf_mat = confusion_matrix(y_true, y_pred)
        conf_dis = ConfusionMatrixDisplay(conf_mat)
        conf_dis.plot()
        plt.savefig('confusion_matrix.png')

        print("Confusion Matrix Saved!")



if __name__ == '__main__':

    # if len(sys.argv) < 2:
    #     print('Usage: python test_1705066.py <test_file>')
    #     exit(0)

    # main()
    compare_testset("training-d.csv", "1705066_prediction.csv")
