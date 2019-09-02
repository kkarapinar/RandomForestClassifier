import numpy as np


def gini(labels):
    total_size = len(labels)

    label_quantities = np.unique(labels, return_counts=True)

    sum_of_probs_sq = 0
    for num_of_elem in label_quantities[1]:
        sum_of_probs_sq += (num_of_elem / total_size) ** 2
    return 1 - sum_of_probs_sq
