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
    :return: arr (np.ndarray) where arr[i-1] = P(y_i), shape (label_size,)
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


def get_P_of_xi_given_yk(mat: np.ndarray, vocab_size: int, a: float):
    """
    Estimate P(X|Y) using a MAP estimate.
    :param mat: np.ndarray, rows are entries. labels expected in last column
    :param vocab_size: int, number of words in vocabulary
    :param a: float, alpha value used to smooth estimate
    :return arr, np.ndarray of shape (label_size x vocab_size) where
        arr[k][i] indicates P(Xi | Yk)
    """
    xi_in_yk_counts = get_xi_in_yk(mat, vocab_size)  # (label_size x vocab_size)
    word_count_in_yk = np.sum(xi_in_yk_counts, axis=1)  # (label_size, )
    denom = word_count_in_yk + ((a - 1) * vocab_size)  # (label_size, )
    numer = xi_in_yk_counts + (a - 1)  # (label_size x vocab_size)
    arr = numer / denom.reshape((-1, 1))
    return arr

def entropy(probs: np.ndarray) -> float:
    """
    Estimate Shannon entropy for the given class distribution.
    :param probs: np.ndarray, class distribution of size (label_size,)
    :return ent: float, the entropy of the distribution
    """
    return -1.0 * np.sum(probs * np.log2(probs))
        
def get_num_docs_with_feat(data: np.ndarray) -> list:
    """
    Compute how many documents contain each word at least once.
    :param data: np.ndarray, document BoW data of shape (# docs, vocab_size)
    :return feats: list, feats[i] is how many times word i appears 
        in a document at least once
    """
    feats = []
    for i in range(data.shape[1]):
        feats.append(np.nonzero(data[:, i])[0].shape[0])
    return feats

def get_standardization_mean_stddev(mat: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """
    Get mean and standard deviation of features using training examples.
    :param mat: np.ndarray, rows are examples, columns are features
        (assume no id or class columns)
    :return tup, tuple[np.ndarray, np.ndarray], first contain means, 
        second contain standard deviations, each of shape (vocab_size).
    """
    means = np.mean(mat, axis=0)
    devs = np.std(mat, axis=0)
    tup = means, devs
    return tup

def standardize_features(mat: np.ndarray, means: np.ndarray, devs: np.ndarray) -> np.ndarray:
    """
    Standardize features such that each has mean 0 and standard deviation 1.
    :param mat: np.ndarray, rows are examples, columns are features
        (assume no id or class columns)
    :param means: np.ndarray, the mean of every feature given the mat data
    :param devs: np.ndarray, the standard deviation of every feature given the mat data
    :return out: np.ndarray, new mat with standardized features
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (mat - means) / devs
    
    # handle devs close to 0
    out = np.nan_to_num(out)
    return out

def get_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Get classification accuracy of predictions
    :param predicted: np.ndarray, predicted values
    :param actual: np.ndarray, actual values
    :return: float, classification accuracy
    """
    assert predicted.shape[0] == actual.shape[0]
    matches = np.where(predicted == actual)[0]
    n_correct = matches.shape[0]
    total = actual.shape[0]
    return n_correct / total

def get_confusion_matrix(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix of predictions to visualize errors
    :param predicted: np.ndarray, predicted values
    :param actual: np.ndarray, actual values
    :return: c: np.ndarray, c[i, j] is the # of times an instance with class j
        was classified as category i.
    """
    pred = predicted.reshape((predicted.shape[0],))
    actu = actual.reshape((actual.shape[0],))
    assert pred.shape[0] == actu.shape[0]
    n_classes = len(np.unique(pred))
    c = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_classes):
            iclass = i + 1
            jclass = j + 1
            truthindices = np.nonzero(actu == jclass)
            withi = np.nonzero(pred[truthindices] == iclass)
            c[i, j] = withi[0].shape[0]
    return c

if __name__ == "__main__":
    # mat1 = sparse.load_npz("../data/sparse_training.npz").toarray()
    # print(mat1.shape)
    # print(mat1[0][0])
    mat1 = np.random.randint(0, 5, size=(5, 5))
    mat1[:, -1] = np.arange(0, 5, 1, dtype=np.int32)
    mat1[3, -1] = 1
    print(mat1)
    mat2 = np.random.randint(0, 5, size=(5, 5))
    mat2[:, -1] = np.arange(0, 5, 1, dtype=np.int32)
    mat2[4, -1] = 1
    print(mat2)
    acc = get_accuracy(mat1[:, -1], mat2[:, -1])
    print(acc)
    mat2 = sparse.load_npz("../data/sparse_training.npz").toarray()
    means, devs = get_standardization_mean_stddev(mat2[:, 1:-1])
    std_mat2 = standardize_features(mat2[:, 1:-1], means, devs)
    print(std_mat2[550:660, :6])