from supervised_model import SupervisedModel
import numpy as np
from math import floor
import matplotlib.pyplot as plt

class LinearRegressor(SupervisedModel):
    def __init__(self, alpha=0.001, lambda_=0.1, regularization='l2'):
        super().__init__()
        self.theta = None # parameters (weights)
        self.b = None # bias
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization

    def compute_loss(self, X, y):
        err_squared = (y - X.T @ self.theta - self.b) ** 2
        loss = np.sum(err_squared) / (2 * y.shape[0])

        return loss


    def compute_gradient(self, X, y):
        err = -(y - X.T @ self.theta - self.b) / (2 * y.shape[0])
        dtheta = X @ err
        db = err

        return dtheta, db

    def fit(self, X, y):
        MAX_ITER = 200
        self.theta = np.zeros((X.shape[0], 1))
        self.b = np.zeros((y.shape[0], 1))
        for k in range(MAX_ITER):
            dtheta, db = self.compute_gradient(X, y)
            self.theta = self.theta - self.alpha * dtheta
            self.b = self.b - db

            print(self.compute_loss(X, y))

    def predict(self, X):
        super.predict(X)

        return X.T @ self.theta + self.b



