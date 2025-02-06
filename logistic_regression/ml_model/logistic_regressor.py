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
        z = X.T @ self.W + self.b
        y_hat = np.vectorize(self.sigmoid)(z)
        m = y.shape[0]
        loss = .0
        for i in range(m):
            if y[i] == 1:
                loss += -(1/m) * np.log(y_hat[i, 0] + EPSILON)
            elif y[i] == 0:
                loss += -(1/m) * np.log(1 - y_hat[i, 0] + EPSILON)

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
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

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
        m, n = X.shape
        z = X.T @ self.W + self.b
        y_hat = np.vectorize(self.sigmoid)(z)
        dW = np.zeros(self.W.shape)
        for j in range(dW.shape[0]):
            for i in range(n):
                dW[j, 0] += (1 / m) * (y_hat[i, 0] - y[i, 0]) * X[j, i]

        db = (1/m) * np.sum(y_hat - y)

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
        self.W = np.random.normal(loc=.0, scale=.1, size=X.shape[0])
        self.W = self.W.reshape((X.shape[0], 1))
        self.b = np.random.normal(loc=0, scale=.05)
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
        y_hat = np.zeros((X.shape[1], 1))
        z = X.T @ self.W + self.b
        y_hat = self.sigmoid(z)
        m = y_hat.shape[0]
        for i in range(m):
            if y_hat[i, 0] >= THRESHOLD:
                y_hat[i, 0] = 1
            else:
                y_hat[i, 0] = 0

        return y_hat
        
    def accuracy(self, X, y):
        """
        Computes model accuracy.

        :param X: real inputs from test set.
        :param y: real outputs from test set.
        :return: number ranging from 0 to 1.
        :rtype: float
        """
        y_hat = self.predict(X)
        return np.mean(y == y_hat) # computes TN and TP
    
    def precision(self, X, y):
        """
        Computes model precision.

        :param X: real inputs from test set.
        :param y: real outputs from test set.
        :return: number ranging from 0 to 1.
        :rtype: float
        """
        y_hat = self.predict(X)
        
        TP = np.sum((y_hat == 1) & (y == 1))
        FP = np.sum((y_hat == 1) & (y == 0))

        if TP + FP == 0:
            return 0.0
        
        return TP / (TP + FP)

    def recall(self, X, y):
        """
        Computes model recall.

        :param X: real inputs from test set.
        :param y: real outputs from test set.
        :return: number ranging from 0 to 1.
        :rtype: float
        """
        y_hat = self.predict(X)
        
        TP = np.sum((y_hat == 1) & (y == 1))
        FN = np.sum((y_hat == 0) & (y == 1))

        if TP + FN == 0:
            return 0.0
        
        return TP / (TP + FN)
 
    def plot_ROC_curve(self, X, y):
        """
        Plots model ROC curve.
        :param X: real inputs from test set.
        :param y: real outputs from test set.
        """
        y_hat = self.predict(X)
        TP = np.sum((y_hat == 1) & (y == 1))
        FN = np.sum((y_hat == 0) & (y == 1))
        FP = np.sum((y_hat == 1) & (y == 0))
        TN = np.sum((y_hat == 0) & (y == 0))

        TP_rate = TP / (TP + FN)
        FP_rate = FP / (FP + TN)

        x_random = np.linspace(0, 1, 100)
        y_random = x_random

        plt.figure()
        plt.plot(x_random, y_random, linestyle=':', color='red', label='Random Classifier')
        plt.plot(x=FP_rate, y=TP_rate, color='blue', label='ROC Curve')
        plt.xlabel('False positive rate.')
        plt.ylabel('True positive rate.')


