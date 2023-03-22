import numpy as np
from tqdm import tqdm

from utils import get_num_docs_with_feat, kl_divergence
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
        self.P_x_given_y = get_P_of_xi_given_yk(mat, self.n,
                                                self.a)  # (label_size x vocab_size)

    def classify(self, mat: np.ndarray, id_in_mat=True,
                 class_in_mat=True) -> np.ndarray:
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
        log_P_x_given_y_t = np.transpose(
            log_P_x_given_y)  # (vocab_size x label_size)
        start = 0
        if id_in_mat:
            start = 1
        end = mat.shape[1]
        if class_in_mat:
            end = -1
        sum = np.matmul(mat[:, start:end],
                        log_P_x_given_y_t)  # (# docs, label_size)

        all_classes = log_P_y + sum  # (# docs, label_size)
        return np.argmax(all_classes, axis=1) + 1

    def get_best_words(self, data: np.ndarray, n: int, vocab: list):
        """
        Grab the terms which influence the Naive Bayes classifier the most
        using information gain.
        :param data: np.ndarray, rows are entries. first and last col expected
                     to be ids and labels, respectively
        :param n: int, how many best terms to collect
        :param vocab: list, a map from indices to words in the BoW model
        :return topn_words: list, a list of size n containing strings of the
            best n words.
        """

        def _calc_entry(x: int, y: int):
            other_pxgy = np.setdiff1d(self.P_x_given_y[:, x],
                                      self.P_x_given_y[y, x])
            pxgy = self.P_x_given_y[y, x]
            return np.sum(kl_divs_vect(pxgy, other_pxgy)) * self.P_y[y]

        assert len(vocab) == self.P_x_given_y.shape[1]
        kl_divs = np.zeros((self.k, self.n))
        kl_divs_vect = np.vectorize(kl_divergence)
        entry_vect = np.vectorize(_calc_entry)
        x_vals = np.arange(0, self.n)
        for label in tqdm(range(self.k)):
            kl_divs[label, :] = entry_vect(x_vals, label)
        kl_divs = np.amax(kl_divs, axis=0)
        topn_is = np.flip(np.argsort(kl_divs))[0:n]
        topn = list(map(lambda i: vocab[i], topn_is))
        freqs = get_num_docs_with_feat(data[:, 1:-1])
        topn_freqs = list(map(lambda i: freqs[i], topn_is))
        return topn, topn_freqs



