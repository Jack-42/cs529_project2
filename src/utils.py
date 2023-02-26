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


def get_xi_in_yk(mat: np.ndarray, vocab_size: int):
    """
    Get the # of times each word (xi) in the vocabulary appears for entries
    with label yk.
    :param mat: np.ndarray, rows are entries. labels expected in last column
    :param vocab_size: int, number of words in vocabulary
    :return: arr, np.ndarray of shape (label_size x vocab_size) where
        arr[j][i] indicates the number of times word i appears in documents with
        label j.
    """
    label_vals = np.unique(mat[:, -1])
    n_labels = len(label_vals)
    get_rows = lambda label_val: mat[np.where(mat[:, -1] == label_val)]
    yk_rows = list(map(get_rows, label_vals))
    arr = np.zeros((n_labels, vocab_size))
    for i in range(len(label_vals)):
        lv_rows = yk_rows[i]
        # exclude document id and class
        cnts = np.sum(lv_rows, axis=0, dtype=np.int64)[1:-1]
        arr[i, :] = cnts
    return arr

def get_P_of_xi_given_yi(mat: np.ndarray, vocab_size: int, a: float):
    """
    Estimate P(X|Y) using a MAP estimate.
    :param mat: np.ndarray, rows are entries. labels expected in last column
    :param vocab_size: int, number of words in vocabulary
    :param a: float, alpha value used to smooth estimate
    :return arr, np.ndarray of shape (label_sive x vocab_size) where
        arr[j][i] indicates P(Xi | Yj)
    """
    xi_in_yk_counts = get_xi_in_yk(mat, vocab_size)  # (label_size x vocab_size)
    word_count_in_yk = np.sum(xi_in_yk_counts, axis=1)  # (label_size, )
    denom = word_count_in_yk + ((a - 1) * vocab_size)  # (label_size, )
    numer = xi_in_yk_counts + (a - 1)  # (label_size x vocab_size)
    arr = numer / denom.reshape((-1, 1))
    return arr

if __name__ == "__main__":
    mat1 = sparse.load_npz("../data/sparse_training.npz").toarray()
    print(mat1.shape)
    print(mat1[0][0])
    # mat1 = np.random.randint(0, 5, size=(5, 5))
    # mat1[:, -1] = np.arange(0, 5, 1, dtype=np.int32)
    # mat1[3, -1] = 1
    # print(mat1)
    x = get_xi_in_yk(mat1, 61188)
    print(x)
    estxi_given_yk = get_P_of_xi_given_yi(mat1, 61188, 1 / 61188)
    print(estxi_given_yk)
    print(estxi_given_yk.shape)