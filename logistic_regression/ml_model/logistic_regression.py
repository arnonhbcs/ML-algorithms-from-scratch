from supervised_model import SupervisedModel
import numpy as np
from logistic_regressor_parameters import *

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
    pass

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
    pass

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
    pass

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