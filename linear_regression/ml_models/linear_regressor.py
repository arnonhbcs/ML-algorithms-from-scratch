from supervised_model import SupervisedModel
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from linear_regressor_parameters import *

class LinearRegressor(SupervisedModel):
    def __init__(self, alpha=LEARNING_RATE, lambda_=L2_REGULARIZATION_RATE, regularization='l2', stochastic=False):
        """
        Implements the Linear Regression Algorithm
        :param alpha: Learning rate.
        :type alpha: float
        :param lambda_: Regularization parameter.
        :type lambda_: float
        :param regularization: Type of regularization to apply ('l1', 'l2', or 'None').
        :type regularization: str
        :param stochastic: Use stochastic gradient descent if True.
        :type stochastic: bool
        """
        super().__init__()
        self.theta = None  # parameters (weights)
        self.b = None  # bias
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization
        self.stochastic = stochastic

    def compute_loss(self, X, y):
        """
        Computes the model's loss function.
        
        :param X: Inputs from training examples.
        :type X: ndarray
        :param y: Outputs from training examples.
        :type y: ndarray
        :return: Computed loss value.
        :rtype: float
        """
        err_squared = (y - X.T @ self.theta - self.b) ** 2
        loss = np.sum(err_squared) / (2 * y.shape[0])

        if self.regularization == 'None':
            pass
        elif self.regularization == 'l1':
            loss += self.lambda_ * np.linalg.norm(self.theta, ord=1) / y.shape[0]
        elif self.regularization == 'l2':
            loss += self.lambda_ * (np.linalg.norm(self.theta, ord=2))**2 / (2 * y.shape[0])

        return loss

    def compute_gradient(self, X, y):
        """
        Computes the gradient of the loss function with respect to the model parameters.
        
        :param X: Inputs from training examples.
        :type X: ndarray
        :param y: Outputs from training examples.
        :type y: ndarray
        :return: Gradients of the loss with respect to theta and b.
        :rtype: tuple(ndarray, float)
        """
         
        err = -(y - X.T @ self.theta - self.b) / (2 * y.shape[0])
        dtheta = X @ err
        db = np.sum(err)

        if self.regularization == 'None':
            pass
        elif self.regularization == 'l1':
            dtheta += self.lambda_ * np.sign(self.theta) / y.shape[0]
        elif self.regularization == 'l2':
            dtheta += self.lambda_ * self.theta / y.shape[0]

        return dtheta, db

    def fit(self, X, y, verbose=True):
        """
        Trains the model using gradient descent or stochastic gradient descent.
        
        :param X: Inputs from training examples.
        :type X: ndarray
        :param y: Outputs from training examples.
        :type y: ndarray
        :param verbose: Set true to plot training history.
        :type verbose: bool
        """
        loss_vals = []
        epochs = []

        for k in range(MAX_ITER):
            dtheta, db = self.compute_gradient(X, y)
            self.theta = self.theta - self.alpha * dtheta
            self.b = self.b - self.alpha * db
            loss = self.compute_loss(X, y)
            loss_vals.append(loss)
            epochs.append(k+1)
        
        if verbose:
            plt.figure()
            plt.plot(epochs, loss_vals)
            plt.xlabel('Epoch')
            plt.ylabel('$L(\\theta, b)$')
            plt.show()

    def predict(self, X):
        """
        Predicts outputs for the given input data.
        
        :param X: Inputs for prediction.
        :type X: ndarray
        :return: Predicted outputs.n
        :rtype: ndarray
        """
        super().predict(X)

        return X.T @ self.theta + self.b