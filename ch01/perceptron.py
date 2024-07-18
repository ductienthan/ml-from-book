import numpy as np


class Perceptron(object):
    """
    Perceptron classifier
    Parameters
    eta: float
        Learning rate ([0.0,1.0])
    n_iter: int
        Passes over the training dataset
    random_state: int
        Random number generator seed for random weight initialization
    Attributes
    w_: 1d-array
        weight after fitting
    errors_: list
        Number of misclassification (updates) in each epoch
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, Y):
        """
        Fit training data
        Parameters
        X: {array_like}, shape = [n_example, n_features]
            Training vectors, where n_examples is the number of examples, and n_feature is the number of features.
        Y: array_like, shape = [n_examples]
            Target_values.
        Returns

        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        #Calculate net input
        return np.dot(X, self.w_[1:] + self.w_[0])
    def predict(self, X):
        #Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)