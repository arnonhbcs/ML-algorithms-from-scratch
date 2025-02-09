from supervised_model import SupervisedModel
import numpy as np
from logistic_regressor_parameters import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LogisticRegressor(SupervisedModel):
    def __init__(self, alpha=LEARNING_RATE, threshold=THRESHOLD, lambda_=L2_REGULARIZATION_RATE, regularization='l2'):
        """
        Implements the Logistic Regressor Algorithm.
        :param alpha: Learning rate for gradient descent.
        :type alpha: float
        :param threshold: Threshold for classification.
        :type threshold: float
        :param lambda_: Regularization strength.
        :type lambda_: float
        :param regularization: Type of regularization ('l1' or 'l2').
        :type regularization: str
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization
        self.threshold = threshold
        self.b = None
        self.W = None
    
    def sigmoid(self, z):
        """
        Computes the sigmoid function.
        :param z: Input value or array.
        :type z: float or ndarray
        :return: Sigmoid activation.
        :rtype: float or ndarray
        """
        return 1 / (1 + np.exp(-z))

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
        y_hat = self.sigmoid(z)
        m = y.shape[0]
        loss = -np.mean(y * np.log(y_hat + EPSILON) + (1 - y) * np.log(1 - y_hat + EPSILON))
        
        if self.regularization == 'l1':
            loss += self.lambda_ * np.linalg.norm(self.W, ord=1) / m
        elif self.regularization == 'l2':
            loss += self.lambda_ * np.linalg.norm(self.W, ord=2) ** 2 / (2 * m)
        
        return loss

    def compute_gradient(self, X, y):
        """
        Computes the gradient of the loss function with respect to the model parameters.
        :param X: Inputs from training examples.
        :type X: ndarray
        :param y: Outputs from training examples.
        :type y: ndarray
        :return: Gradients of the loss with respect to weights and bias.
        :rtype: tuple(ndarray, float)
        """
        m = X.shape[1]
        z = X.T @ self.W + self.b
        y_hat = self.sigmoid(z)
        dW = (X @ (y_hat - y)) / m
        db = np.mean(y_hat - y)

        if self.regularization == 'l1':
            dW += self.lambda_ * np.sign(self.W) / m
        elif self.regularization == 'l2':
            dW += self.lambda_ * self.W / m

        return dW, db
    
    def fit_sgd(self, X, y, BATCH_SIZE=BATCH_SIZE, verbose=True):
        """
        Trains the model using stochastic gradient descent.
        :param X: Inputs from training examples
        :type X: ndarray
        :param y: Outputs from training examples
        :type y: ndarray
        :param BATCH_SIZE: size of training batch
        :type BATCH_SIZE: int
        :param verbose: Set true to plot training history.
        :type verbose: bool
        """
        pass

    def fit(self, X, y, verbose=True):
        """
        Trains the model using gradient descent.
        :param X: Inputs from training examples.
        :type X: ndarray
        :param y: Outputs from training examples.
        :type y: ndarray
        :param verbose: Set true to plot training history.
        :type verbose: bool
        """
        self.W = np.random.normal(loc=0, scale=0.1, size=(X.shape[0], 1))
        self.b = np.random.normal(loc=0, scale=0.05)
        loss_vals = []
        epochs = []

        for k in range(MAX_ITER):
            dW, db = self.compute_gradient(X, y)
            self.W -= self.alpha * dW
            self.b -= self.alpha * db
            loss_vals.append(self.compute_loss(X, y))
            epochs.append(k + 1)

        if verbose:
            plt.figure()
            plt.plot(epochs, loss_vals)
            plt.xlabel('Epoch')
            plt.ylabel('$L(\theta, b)$')
            plt.show()
    
    def predict_probability(self, X):
        """
        Computes the probability for input data.
        :param X: Inputs for prediction.
        :type X: ndarray
        :return: Probability values.
        :rtype: ndarray
        """
        return self.sigmoid(X.T @ self.W + self.b)

    def predict(self, X):
        """
        Predicts binary class labels for the given input data.
        :param X: Inputs for prediction.
        :type X: ndarray
        :return: Predicted class labels.
        :rtype: ndarray
        """
        return (self.predict_probability(X) >= self.threshold).astype(int)
    
    def accuracy(self, X, y):
        """
        Computes model accuracy.
        :param X: Inputs from test set.
        :type X: ndarray
        :param y: Outputs from test set.
        :type y: ndarray
        :return: Accuracy value.
        :rtype: float
        """
        return np.mean(y == self.predict(X))
    
    def precision(self, X, y):
        """
        Computes model precision.
        :param X: Inputs from test set.
        :type X: ndarray
        :param y: Outputs from test set.
        :type y: ndarray
        :return: Precision value.
        :rtype: float
        """
        y_hat = self.predict(X)
        TP = np.sum((y_hat == 1) & (y == 1))
        FP = np.sum((y_hat == 1) & (y == 0))
        return TP / (TP + FP) if TP + FP > 0 else 0.0

    def recall(self, X, y):
        """
        Computes model recall.
        :param X: Inputs from test set.
        :type X: ndarray
        :param y: Outputs from test set.
        :type y: ndarray
        :return: Recall value.
        :rtype: float
        """
        y_hat = self.predict(X)
        TP = np.sum((y_hat == 1) & (y == 1))
        FN = np.sum((y_hat == 0) & (y == 1))
        return TP / (TP + FN) if TP + FN > 0 else 0.0
    
    def f1_score(self, X, y):
        """
        Computes model F1-score.
        :param X: Inputs from test set.
        :type X: ndarray
        :param y: Outputs from test set.
        :type y: ndarray
        :return: F1-score value.
        :rtype: float
        """
        precision = self.precision(X, y)
        recall = self.recall(X, y)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    

    
    