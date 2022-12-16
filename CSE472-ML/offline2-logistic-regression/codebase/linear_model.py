import numpy as np
import pandas as pd


class LogisticRegression:

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


    def __init__(self, params):
        self.n_iters = params['n_iters']
        self.lr = params['lr']
        self.weights = None
        self.bias = 0


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):

            score = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(score)    

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self


    def predict_proba(self, X):
        y_prob = np.dot(X, self.weights) + self.bias
        sig_prob = self.sigmoid(y_prob)
        return sig_prob


    def predict(self, X):
        return np.where(self.predict_proba(X) > 0.5, 1, 0)