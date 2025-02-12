import numpy as np
from sklearn.model_selection import train_test_split
from supervised_model import SupervisedModel
from neural_network_parameters import *


class NeuralNetwork(SupervisedModel):
    def __init__(self, alpha=LEARNING_RATE, lambda_=L2_REGULARIZATION_RATE, regularization='l2'):
        """
        Implements the Linear Regression Algorithm.

        :param alpha: Learning rate.
        :type alpha: float
        :param lambda_: Regularization parameter.
        :type lambda_: float
        :param regularization: Type of regularization to apply ('l1', 'l2', or 'None').
        :type regularization: str
        """
        self.W = None
        self.b = None
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization

    def fit(self):
        """
        Trains the model using gradient descent or stochastic gradient descent.
        
        :param X: Inputs from training examples.
        :type X: ndarray
        :param y: Outputs from training examples.
        :type y: ndarray
        :param verbose: Set true to plot training history.
        :type verbose: bool
        """
        pass

    def predict(self, X):
        """
        Predicts outputs for the given input data.
        
        :param X: Inputs for prediction.
        :type X: ndarray
        :return: Predicted outputs.n
        :rtype: ndarray
        """
        pass
