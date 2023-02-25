import scipy.sparse as sparse
import numpy as np

"""
@author Jack Ringer, Mike Adams,
Date: 2/25/2023
Description:
    This file contains various utility functions used in the project.
"""


def get_y_priors(mat: np.ndarray):
    """
    Get P(y_k) for all y_k in Y (set of all labels)
    :param mat: 2D sparse matrix, labels expected in last column. rows are entries
    :return: arr (np.ndarray) where arr[i-1] = P(y_i)
    """
    label_counts = np.unique(mat[:, -1], return_counts=True)
    labels, counts = label_counts[0], label_counts[1]
    n_entries = len(mat[:, 0])
    p_func = np.vectorize(lambda cnt: cnt / n_entries)
    probs = p_func(counts)
    return probs


def get_xi_in_yk(mat: np.ndarray):
    """
    Get the # of times each word (xi) in the vocabulary appears for entries
    with label yk.
    :param mat: np.ndarray, rows are entries
    :return: np.ndarray of shape (vocab_size x label_size)
    """
    return None


if __name__ == "__main__":
    mat1 = sparse.load_npz("../data/sparse_training.npz").toarray()
    priors = get_y_priors(mat1)
    print(priors)
    print(np.sum(priors))
    print(type(priors))
