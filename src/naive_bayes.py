import numpy as np
from tqdm import tqdm

from utils import get_y_priors, get_P_of_xi_given_yk
from utils import get_num_docs_with_feat, kl_divergence

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

    def get_best_words(self, n: int, vocab: list):
        """
        Grab the terms which influence the Naive Bayes classifier the most
        using information gain.
        :param n: int, how many best terms to collect
        :param vocab: list, a map from indices to words in the BoW model
        :return topn_words: list, a list of size n containing strings of the
            best n words.
        """
        assert len(vocab) == self.P_x_given_y.shape[1]
        kl_divs = np.zeros((self.k, self.n))
        kl_divs_vect = np.vectorize(kl_divergence)
        for y in tqdm(range(self.k)):
            for x in range(self.n):
                other_pxgy = np.setdiff1d(self.P_x_given_y[:, x],
                                          self.P_x_given_y[y, x])
                pxgy = self.P_x_given_y[y, x]
                kl_divs[y][x] = np.sum(kl_divs_vect(pxgy, other_pxgy))
        kl_divs = np.amax(kl_divs, axis=0)
        topn_is = np.flip(np.argsort(kl_divs))[0:n]
        topn = list(map(lambda i: vocab[i], topn_is))
        freqs = get_num_docs_with_feat(train_data[:, 1:-1])
        topn_freqs = list(map(lambda i: freqs[i], topn_is))
        return topn, topn_freqs


if __name__ == "__main__":
    import os
    import scipy.sparse as sparse

    with open("../data/vocabulary.txt") as f:
        vocab = f.read().splitlines()

    mat = sparse.load_npz("../data/sparse_training.npz").toarray()
    split_r = 0.8  # 80% train, 20% val
    cutoff = int(len(mat[:, 0]) * split_r)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(mat)
    train_data, val_data = mat[0:cutoff], mat[cutoff:]
    lab_count = len(np.unique(mat[:, -1]))
    attr_count = mat.shape[1] - 2
    # betas = np.linspace(0.011138, 0.011145, 25)
    betas = [0.011111]
    for ib in betas:
        nb = NaiveBayes(lab_count, attr_count, 1.0 + ib)
        nb.train(train_data)
        topHundred, freqs = nb.get_best_words(100, vocab)
        print(topHundred)
        print(freqs)

    # # test acc
    # from utils import get_accuracy
    # nb_pred_train = nb.classify(mat[:, 1:-1], id_in_mat=False, class_in_mat=False)
    # print("train acc: ", get_accuracy(nb_pred_train, mat[:, -1]))

    # data = sparse.load_npz("../data/sparse_training.npz").toarray()

    # label_count = len(np.unique(data[:, -1]))
    # attr_count = data.shape[1] - 2

    # # split into train/val
    # split_r = 0.8  # 80% train, 20% val
    # cutoff = int(len(data[:, 0]) * split_r)
    # np.random.seed(12)  # for reproducibility
    # np.random.shuffle(data)
    # train_data, val_data = data[0:cutoff], data[cutoff:]

    # nb = NaiveBayes(label_count, attr_count, 1.0 + 0.090918)
    # nb.train(train_data)
    # train_pred = nb.classify(train_data, id_in_mat=True, class_in_mat=True)
    # val_pred = nb.classify(val_data, id_in_mat=True, class_in_mat=True)

    # from utils import get_confusion_matrix
    # c_mat = get_confusion_matrix(val_pred, val_data[:, -1])
    # print(c_mat)

    # from plots import plot_confusion_matrix
    # import os
    # os.makedirs("../figures", exist_ok=True)
    # save_path = "../figures/nb_090918_conf_mat_seed12.png"
    # plot_confusion_matrix(c_mat, title="NB", save_pth=save_path)

    # from utils import get_accuracy
    # print("val acc: ", get_accuracy(val_pred, val_data[:, -1]))

    # test_mat = sparse.load_npz("../data/sparse_testing.npz").toarray()
    # nb_pred = nb.classify(test_mat, id_in_mat=True, class_in_mat=False)
    # output = np.zeros((test_mat.shape[0], 2), dtype=np.int64)
    # output[:, 0] = test_mat[:, 0]
    # output[:, 1] = nb_pred
    # np.savetxt("../data/nb_beta_0_090918_test_out_seed12_train8020.csv", output, fmt="%d", delimiter=",", header="id,class", comments="")
