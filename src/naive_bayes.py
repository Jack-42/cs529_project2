import numpy as np
import scipy.sparse as sparse

from utils import get_y_priors, get_P_of_xi_given_yk

"""
@author Jack Ringer, Mike Adams
Date: 2/26/2023
Description:
    Class for training and classifying using the Naive Bayes model
"""

class NaiveBayes:
    def __init__(self, k: int, n: int, a: float):
        """
        Initialize ready to train model
        :param k: int, number of classes in dataset
        :param n: int, number of attributes each example has
        :param a: float, alpha used in MAP for P(X|Y)
        """
        self.k = k
        self.n = n
        self.a = a

    def train(self, mat: np.ndarray):
        """
        Train model with given data.
        MLE used to estimate P(Y), MAP used to estimate P(X|Y)
        :param mat: np.ndarray, rows are entries. labels expected in last column
        """
        self.P_y = get_y_priors(mat)  # shape (label_size,)
        self.P_x_given_y = get_P_of_xi_given_yk(mat, self.n, self.a)  # (label_size x vocab_size)

    def classify(self, mat: np.ndarray, id_in_mat=True, class_in_mat=True) -> np.ndarray:
        """
        Evaluate model on given documents.
        :param mat: np.ndarray, rows are entries. labels expected in last column
            if class_in_mat is True
        :param id_in_mat, true if mat[:, 0] are IDs
        :param class_in_mat, true if mat[:, -1] are classes
        :return arr, np.ndarray of shape mat.shape[0] where arr[i] is predicted
            class of document i.
        """
        log_P_y = np.log2(self.P_y)
        log_P_x_given_y = np.log2(self.P_x_given_y)

        # mat.shape is (# docs, vocab_size)
        log_P_x_given_y_t = np.transpose(log_P_x_given_y)  # (vocab_size x label_size)
        start = 0
        if id_in_mat:
            start = 1
        end = mat.shape[1]
        if class_in_mat:
            end = -1
        sum = np.matmul(mat[:, start:end], log_P_x_given_y_t)  # (# docs, label_size)

        all_classes = log_P_y + sum  # (# docs, label_size)
        return np.argmax(all_classes, axis=1) + 1

if __name__ == "__main__":
    mat = sparse.load_npz("../data/sparse_training.npz").toarray()
    lab_count = len(np.unique(mat[:, -1]))
    attr_count = mat.shape[1] - 2
    nb = NaiveBayes(lab_count, attr_count, 1.0 + (1.0 / attr_count))
    nb.train(mat)
    test_mat = sparse.load_npz("../data/sparse_testing.npz").toarray()
    nb_pred = nb.classify(test_mat, id_in_mat=True, class_in_mat=False)
    output = np.zeros((test_mat.shape[0], 2), dtype=np.int64)
    output[:, 0] = test_mat[:, 0]
    output[:, 1] = nb_pred
    np.savetxt("../data/nb_basic_test_out.csv", output, fmt="%d", delimiter=",", header="id,class", comments="")

    # test acc
    from utils import get_accuracy
    nb_pred_train = nb.classify(mat[:, 1:-1], id_in_mat=False, class_in_mat=False)
    print("train acc: ", get_accuracy(nb_pred_train, mat[:, -1]))