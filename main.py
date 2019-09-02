import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time
import parsers as p
from rfc_pack.RandomForest import RandomForest
import statistics


def cross_val_iterator(X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf.get_n_splits(X, y)
    return skf.split(X, y)


if __name__ == "__main__":
    filename = os.path.join('dataset', 'german_credit', 'german.data-numeric')
    parser = p.ParserNoOne()

    # filename = os.path.join('dataset', 'phone_price', 'dataset.csv')
    # parser = p.ParserNoTwo()

    X, y = parser.get_data(filename)

    n_trees = 100
    max_depth = 5
    sample_ratio = 0.5
    n_features = 10
    n_workers = 20

    acc_0 = []
    acc_1 = []

    for i, (train_indices, test_indices) in enumerate(cross_val_iterator(X, y, 10)):
        print('Bucket %d' % i)
        random_forests = RandomForest(X[train_indices], y[train_indices],
                                      n_trees, max_depth, sample_ratio, n_features, n_workers)
        random_forests.build_forest()

        acc = random_forests.accuracy(X[test_indices], y[test_indices])
        print('Acc', acc)

        start = time.time()
        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, max_features=n_features,
                                     n_jobs=n_workers)
        clf.fit(X[train_indices], y[train_indices])
        done = time.time()
        elapsed = done - start
        print('Sci-kit Learn speed', elapsed)

        acc_scikit_learn = clf.score(X[test_indices], y[test_indices])
        print('Acc', acc_scikit_learn)

        acc_0.append(acc)
        acc_1.append(acc_scikit_learn)
        print()


    print('Acc_0', statistics.mean(acc_0))
    print('Acc_1', statistics.mean(acc_1))
