from supervised_model import SupervisedModel
import numpy as np
from logistic_regressor_parameters import *
import matplotlib.pyplot as plt

class LogisticRegressor(SupervisedModel):
    def __init__(self, alpha=LEARNING_RATE, lambda_=L2_REGULARIZATION_RATE, regularization='l2'):
        """
        Implements the Logistic Regressor Algorithm.
        """
        super().__init__()
        self.theta = None
        self.b = None
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization

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
        loss_arr = y * np.log(self.f_theta_b(X)) + (1-y) * np.log(1 - self.f_theta_b(X))
        m = y.shape[0]
        loss_val = - (1/m) * np.sum(loss_arr)
        if self.regularization == 'l1':
            pass
        elif self.regularization == 'l2':
            pass
        return loss_val

    def f_theta_b(self, X):
        """
        Computes the model activation function. Using Andrew NG's notation.
        :param X: training set inputs
        :type X: ndarray
        :rtype: float
        """
        z = X.T @ self.theta + self.b
        return 1 / (1 + np.exp(-z))

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
        m = y.shape[0]
        dtheta = (1/m) * np.sum(self.f_theta_b(X) - y) * X
        db = (1/m) * np.sum(self.f_theta_b(X) - y)
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
        self.theta = np.zeros((X.shape[0], 1))
        self.b = 0
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
        z = self.f_theta_b(X)

        if z >= THRESHOLD:
            return 1
        else:
            return 0