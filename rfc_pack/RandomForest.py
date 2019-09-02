from rfc_pack.Tree import Tree
from rfc_pack.Node import Node
import numpy as np
import time
import multiprocessing as mp


class RandomForest(object):
    def __init__(self, X, y, num_trees, max_depth, sample_ratio, num_features, workers):
        self.X = X
        self.y = y
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.num_features = num_features
        self.workers = workers
        self.trees = None

    def build_forest(self):
        self.trees = mp.Manager().list()
        start = time.time()

        jobs = []
        for i in range(self.workers):
            p = mp.Process(target=self.build_forest_function_with_process, args=[round(self.num_trees / self.workers)])
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        done = time.time()
        elapsed = done - start
        print("Num of tree built :", len(self.trees))
        print("Training time:", elapsed)
        print("Build done!")

    def build_forest_function_with_process(self, *args):
        iter_num = args[0]
        for i in range(iter_num):
            sub_X, sub_y = self.__subsample()
            root = Node(sub_X, sub_y, 0)
            features = self.__random_features()

            tree = Tree(root, features, self.max_depth)
            tree.build_tree()
            self.trees.append(tree)

    def accuracy(self, test_X, test_y):
        pred_list = []
        for row_X in test_X:
            pred_list.append(self.predict(row_X))
        pred_list = np.array(pred_list)
        return (pred_list == test_y).mean()

    def predict(self, test_X):
        pred_list = []
        for tree in self.trees:
            pred_list.append(tree.get_prediction(test_X))
        counts = np.bincount(pred_list)
        return np.argmax(counts)

    def __subsample(self):
        rand_indices = np.random.choice(self.X.shape[0], round(self.sample_ratio * self.X.shape[0]), replace=False)
        return self.X[rand_indices], self.y[rand_indices]

    def __random_features(self, ):
        feature_size = self.X.shape[1]
        return np.random.choice(feature_size, self.num_features, replace=False)
