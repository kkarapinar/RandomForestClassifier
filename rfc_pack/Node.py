import numpy as np


class Node(object):
    def __init__(self, X, y, depth):
        self.X = X
        self.y = y
        self.depth = depth
        self.value = None
        self.feature = None
        self.left = None
        self.right = None
        self.prediction = None

    # Splits data according to the value of a specific feature
    def split_data(self, feature, value):
        rows = np.where(self.X[:, feature] >= value, True, False)

        X_right, y_right = self.X[rows], self.y[rows]
        X_left, y_left = self.X[~rows], self.y[~rows]

        return X_right, y_right, X_left, y_left

    def get_next(self, row):
        if row[self.feature] >= self.value:
            return self.right
        else:
            return self.left
