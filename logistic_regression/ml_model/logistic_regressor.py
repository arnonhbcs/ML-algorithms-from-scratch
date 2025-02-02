from supervised_model import SupervisedModel
import numpy as np
from logistic_regressor_parameters import *
import matplotlib.pyplot as plt

class LogisticRegressor(SupervisedModel):
    def __init__(self, alpha=LEARNING_RATE, lambda_=L2_REGULARIZATION_RATE, regularization='l2'):
        """
        Implements the Logistic Regressor Algorithm.
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization
        self.b = None
        self.W = None
    
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
        z = self.W @ X + self.b
        y_hat = self.sigmoid(z)
        m = y.shape[0]
        loss = .0
        for i in range(m):
            if y[i] == 1:
                loss += -(1/m) * np.log(y_hat[i, 0])
            elif y[i] == 0:
                loss += -(1/m) * np.log(1 - y_hat[i, 0])

        if self.regularization == 'None':
            pass
        elif self.regularization == 'l1':
            loss += self.lambda_ * np.linalg.norm(self.W, ord=1) / m
        elif self.regularization == 'l2':
            loss += self.lambda_ * np.linalg.norm(self.W, ord=2) ** 2 / (2 * m)
        return loss

    def sigmoid(self, z):
        """
        Computes the sigmoid function.
        """
        return np.exp(z) / (1 + np.exp(z))

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
        z = self.W @ X + self.b
        y_hat = self.sigmoid(z)
        dW = (1/m) * (y_hat - y) @ X.T
        db = (1/m) * (y_hat - y)

        if self.regularization == 'None':
            pass
        elif self.regularization == 'l1':
            dW += self.lambda_ * np.sign(self.W) / m
        elif self.regularization == 'l2':
             dW += self.lambda_ * self.W / m

        return dW, db
    
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
        self.W = np.random.randn(X.shape[0]) * .1
        self.W = self.W.reshape((1, X.shape[0]))
        self.b = np.random.normal(loc=0, scale=1)
        loss_vals = []
        epochs = []

        for k in range(MAX_ITER):
            dW, db = self.compute_gradient(X, y)
            self.W = self.W - self.alpha * dW
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
        z = self.W @ X + self.b
        y_hat = self.sigmoid(z)
        if y_hat >= THRESHOLD:
            return 1
        else:
            return 0