from rfc_pack.Node import Node
import numpy as np
from rfc_pack import index_measurements as im


# Decision tree
class Tree(object):
    def __init__(self, root, features, max_depth):
        self.root = root
        self.features = features
        self.max_depth = max_depth

    # Builds tree
    def build_tree(self):
        self.__split_recursive(self.root)

    # Gets prediction for a specific
    def get_prediction(self, test_data):
        node = self.root
        while node.prediction is None:
            node = node.get_next(test_data)
        return node.prediction

    # Splits nodes recursively
    def __split_recursive(self, node):
        # Init best values
        # Find the best value for some feature which gives minimum gini index

        b_score, b_value, b_feature, b_split = 0, None, None, None

        dict_list = {}
        for f in self.features:
            for X_row in node.X:
                v = X_row[f]
                key = (f, v)
                if key not in dict_list:
                    dict_list[(f, v)] = True
                    X_r, y_r, X_l, y_l = node.split_data(f, v)

                    score_right = im.gini(y_r)
                    score_left = im.gini(y_l)

                    p = len(y_r) / len(node.y)
                    score = 1 - (p * score_right) - ((1 - p) * score_left)

                    # Keep best values
                    if score > b_score:
                        b_score = score
                        b_value = v
                        b_feature = f
                        b_split = (X_r, y_r, X_l, y_l)

        # Check if splitting should be terminated
        if self.__check_termination(node, b_split):
            # Leaf node
            node.prediction = self.__calculate_prediction(node)
        else:
            node.value = b_value
            node.feature = b_feature

            # Expand tree to the right
            node_right = Node(b_split[0], b_split[1], node.depth + 1)
            node.right = node_right

            # Expand tree to the left
            node_left = Node(b_split[2], b_split[3], node.depth + 1)
            node.left = node_left

            if np.unique(node.left.y).size != 1:
                self.__split_recursive(node.left)
            else:
                node.left.prediction = self.__calculate_prediction(node.left)

            if np.unique(node_right.y).size != 1:
                self.__split_recursive(node.right)
            else:
                node.right.prediction = self.__calculate_prediction(node.right)
        node.X = None
        node.y = None

    # Checks if splitting node should stop
    def __check_termination(self, node, split):
        # Max depth is reached
        if node.depth >= self.max_depth:
            return True

        # No split occurred
        if split is None or split[0].shape[0] == 0 or split[2].shape[0] == 0:
            return True

    # Calculates prediction value of leaf node
    def __calculate_prediction(self, node):
        counts = np.bincount(node.y)
        return np.argmax(counts)
