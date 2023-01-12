from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

import numpy as np


class GaussianMixtureModel:

    def __init__(self, n_comp, max_iter=100, tol=1e-3) -> None:
        
        self.n_comp = n_comp
        self.n_samp = 0
        self.n_feat = 0

        self.max_iter = max_iter
        self.tol = tol

        self.means = None
        self.covs = None
        self.weights = None
        self.phi = None


    def fit(self, X) -> None:

        self.n_samp, self.n_feat = X.shape
        self.means = np.random.rand(self.n_comp, self.n_feat)
        self.covs = np.array([np.eye(self.n_feat)] * self.n_comp)

        self.phi = np.full(shape=self.n_comp, fill_value=1/self.n_comp)
        self.weights = np.full(shape=(self.n_samp, self.n_feat), fill_value=1/self.n_comp)

        self.log_likelihood = 0

        for _ in range(self.max_iter):
            # Perform EM algorithm
            self.e_step(X)
            self.m_step(X)
            curr_likelihood = self.likelihood(X)

            if np.abs(curr_likelihood - self.log_likelihood) < self.tol:
                break
            
        self.log_likelihood = curr_likelihood


    def predict_prob(self, X):

        likelihood = np.zeros((self.n_samp, self.n_comp))

        for k in range(self.n_comp):
            distr = multivariate_normal(
                mean=self.means[k],
                cov=self.covs[k],
                allow_singular=True
            )
            likelihood[:, k] = distr.pdf(X)

            # print(f"likelihood[:, k] ->\n{likelihood[:, k]}")
            # print(f"log of  likelihood[:, k] ->\n{np.log(likelihood[:, k])}")            
            # print(f"basic likelihood ->\n{np.sum(np.log(likelihood[:, k]))}")

        num = likelihood * self.phi + 1e-6
        denum = num.sum(axis=1, keepdims=True)
        # print(f"num/denum -> {num/denum}\n")
        # print(f"shape of prob -> {(num/denum).shape}\n")
        return num / denum


    def likelihood(self, X):
        lh = 0
        for k in range(self.n_comp):
            lh += self.phi[k] * multivariate_normal.pdf(X, self.means[k], self.covs[k], allow_singular=True)
        return np.sum(np.log(lh))


    #   Update weights and phi (mean weights) 
    def e_step(self, X):
        self.weights = self.predict_prob(X)
        self.phi = self.weights.mean(axis=0)


    #   Update mu and sigma holding phi and weights fixed
    def m_step(self, X):
        for k in range(self.n_comp):
            weight = self.weights[:, [k]]
            total_weight = weight.sum()

            self.means[k] = (X * weight).sum(axis=0) / total_weight
            self.covs[k] = np.cov(
                X.T,
                aweights=(weight / total_weight).flatten(),
                bias=True
            )


    def predict(self, X):
        weights = self.predict_prob(X)
        return weights.argmax(axis=1)


    def score(self, X, k):
        distr = multivariate_normal(
            mean=self.means[k],
            cov=self.covs[k],
            allow_singular=True
        )
        return distr.pdf(X)


    def __plot_contour(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.predict(X), cmap='viridis', s=40, edgecolors='k', alpha=0.5)
        x, y = np.mgrid[
            np.min(X[:, 0]) : np.max(X[:, 0]) : .01,
            np.min(X[:, 1]) : np.max(X[:, 1]) : .01
        ]        
        positions = np.dstack((x, y))

        for k in range(self.n_comp):
            score = self.score(positions, k)
            plt.contour(x, y, score, colors='k', alpha=0.5)


    def plot(self, X):
        plt.ion()
        plt.clf()
        self.__plot_contour(X)
        plt.draw()
        plt.pause(10)
        plt.ioff()

