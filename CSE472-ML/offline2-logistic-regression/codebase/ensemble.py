from data_handler import bagging_sampler
from linear_model import LogisticRegression
import numpy as np
from copy import copy


class BaggingClassifier:

    def __init__(self, params, n_estimator):
        self.estimators = [ LogisticRegression(params) for _ in range(n_estimator) ]
        self.n_estimator = n_estimator

    def fit(self, X, y):
        for estimator in self.estimators:
            new_X, new_y = bagging_sampler(X, y)
            estimator.fit(new_X, new_y)

    def predict(self, X):
        predictions = np.zeros(shape=(len(X), self.n_estimator), dtype=int)

        for i_est in range(self.n_estimator):
            predictions[:, i_est] = self.estimators[i_est].predict(X)

        pred_list = []
        for x in range(len(X)):
            total = 0
            for col in range(self.n_estimator):
                total += predictions[x][col]
            pred_list.append(total)

        # print(pred_list)

        y_pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            counts = np.bincount(predictions[i])
            y_pred[i] = np.argmax(counts)

        return y_pred
