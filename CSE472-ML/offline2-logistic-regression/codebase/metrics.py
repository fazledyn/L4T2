"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np


def accuracy(y_true, y_pred):
    correct = 0
    for i in range(np.size(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct/len(y_true)


def precision_score(y_true, y_pred):
    tp, fp = 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
    return tp/(tp+fp)


def recall_score(y_true, y_pred):
    tp, fn = 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
    return tp/(tp+fn)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
