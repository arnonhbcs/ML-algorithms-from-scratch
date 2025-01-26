from abc import abstractmethod


class SupervisedModel:
    """
    Base class for machine learning algorithms
    """
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod    
    def fit(self, X, y):
        pass
    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def Rsquared(self, X):
        pass

