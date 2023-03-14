from itertools import cycle

import numpy as np
from scipy import sparse

from utils import get_accuracy

"""
@author Jack Ringer, Mike Adams
Date: 2/26/2023
Description:
    Class for training and classifying using the logistic regression model
"""


class LogReg:
    def __init__(self, k: int, n: int, lr: float, lam: float):
        """
        initialize logistic regression model
        :param k: int, number of classes in dataset
        :param n: int, number of attributes each example has
        :param lr: float, learning rate
        :param lam: float, penalty term
        """
        self.k = k
        self.labels = np.arange(1, k + 1, 1)
        self.n = n
        self.lr = lr
        self.lam = lam
        # weights matrix
        self.W = sparse.csr_matrix(np.zeros((k, n + 1)))

    def train(self, n_steps: int, data: sparse.lil_matrix, print_acc=True, val_data: sparse.lil_matrix = None, minibatch_size=None):
        """
        Method to train logistic reg model.
        :param n_steps: int, the number of weight updates to apply
        :param data: sparse.lil_matrix, matrix of size m x (n + 2) where first col is
            ids and last col is labels
        :param print_acc, whether to print accuracies to stdout at every step during training
        :param val_data: sparse.lil_matrix, test data matrix of size m' x (n' + 2), halts if acc decreases
        :param minibatch_size, number of training examples to use in each update
        :return: None
        """
        n_entries = data.shape[0]
        curr_acc = 0.0
        last_acc = 0.0
        bad_count = 0
        if not (val_data is None):
            val_n_entries = val_data.shape[0]
            last_acc = get_accuracy(self.classify(val_data[:, :-1], id_in_mat=True, class_in_mat=False).reshape((val_n_entries, 1)), val_data[:, -1])
        if print_acc:
            print("train accuracy on train data before training:  %f" % (get_accuracy(self.classify(data[:, :-1], id_in_mat=True, class_in_mat=False).reshape((n_entries, 1)), data[:, -1])))
            if not (val_data is None):
                print("train accuracy on test data before training:  %f" % (last_acc))

        data_list = [data]
        if not (minibatch_size is None):
            data_list = []
            last = np.ceil(float(n_entries) / minibatch_size).astype(int) - 1
            for i in range(0, last, minibatch_size):
                if i + minibatch_size > n_entries:
                    data_list.append(data[i:, :])
                else:
                    data_list.append(data[i:i+minibatch_size, :])
        data_cycle = cycle(data_list)

        for step in range(n_steps):
            self._weight_update(next(data_cycle))
            if not (val_data is None):
                curr_acc = get_accuracy(self.classify(val_data[:, :-1], id_in_mat=True, class_in_mat=False).reshape((val_n_entries, 1)), val_data[:, -1])
            if print_acc:
                print("train accuracy on train data at step %d:  %f" % (step, get_accuracy(self.classify(data[:, :-1], id_in_mat=True, class_in_mat=False).reshape((n_entries, 1)), data[:, -1])))
                if not (val_data is None):
                    print("train accuracy on test data at step %d:  %f" % (step, curr_acc))

            if (not (val_data is None)) and (curr_acc < last_acc):
                # stop early
                bad_count = bad_count + 1
                if bad_count > 5:
                    if print_acc:
                        print("stopped after step %d (early)" % (step))
                    return step
            else:
                bad_count = 0

            if not (val_data is None):
                last_acc = curr_acc

        return step

    def _weight_update(self, mat: sparse.lil_matrix):
        """
        Update the weight of our model using gradient ascent.
        :param mat: sparse.lil_matrix, matrix of size m x (n + 2) where first col is
            ids and last col is labels
        :return: None
        """
        x, y = mat[:, :-1], mat[:, -1]
        x[:, 0] = 1
        m = y.shape[0]

        # calculate delta, delta[j][i] = 1 if row i in mat has label j, else 0
        delta = np.zeros((self.k, m))
        for i, y_val in enumerate(self.labels):
            row = np.zeros((m,1))
            row[np.nonzero(y == y_val)] = 1
            delta[i] = row.reshape((m, ))

        delta = sparse.csr_matrix(delta)

        # get P(Y | X, W)
        probs = self._P_Y_given_X(x)

        # update
        w_change = self.lr * ((delta - probs) * x - (self.lam * self.W))
        self.W = self.W + w_change

    def _P_Y_given_X(self, x):
        xt = np.transpose(x)

        with np.errstate(over='ignore', invalid='ignore'):
            probs = np.exp((self.W * xt).toarray())
        
        # handle overflow in exp()
        probs = np.nan_to_num(probs)
        
        # last row all 1s to match equations 27-28 in Mitchell
        probs[-1, :] = 1
        # normalize columns to be valid probabilities
        with np.errstate(over="ignore", invalid="ignore"):
            col_sums = np.sum(probs, axis=0)

        # handle overflow in sum()
        col_sums = np.nan_to_num(col_sums)

        probs = probs / col_sums
        return sparse.csr_matrix(probs)  # (label_size, # docs)

    def classify(self, mat: sparse.lil_matrix, id_in_mat=True, class_in_mat=True) -> np.ndarray:
        """
        Evaluate model on given documents.
        :param mat: sparse.lil_matrix, rows are entries. labels expected in last column
            if class_in_mat is True
        :param id_in_mat, true if mat[:, 0] are IDs
        :param class_in_mat, true if mat[:, -1] are classes
        :return arr, np.ndarray of shape (mat.shape[0], 1) where arr[i] is predicted
            class of document i.
        """
        start = 0
        if id_in_mat:
            start = 1
        end = mat.shape[1]
        if class_in_mat:
            end = -1
        
        x = mat[:, start:end]
        # add column of 1s
        x = sparse.hstack([np.ones((mat.shape[0], 1)), x])

        probs = self._P_Y_given_X(x)

        return (np.argmax(probs, axis=0) + 1).reshape((mat.shape[0], 1))

if __name__ == "__main__":
    # n_classes = 3  # num classes
    # n_entries = 5  # num entries
    # v_size = 3  # vocab size
    # mat1 = np.random.randint(0, 5, size=(n_entries, v_size + 2))
    # # id col
    # mat1[:, 0] = np.arange(0, n_entries, 1, dtype=np.int32)
    # # labels col
    # for i in range(n_entries):
    #     mat1[i, -1] = np.random.randint(0, n_classes)
    # print(mat1)
    # log_reg = LogReg(k=n_classes, n=v_size, lr=0.01, lam=1)
    # print(log_reg.W)
    # log_reg.train(n_steps=10, data=sparse.lil_matrix(mat1))
    # print(log_reg.W)
    from utils import get_standardization_mean_stddev, standardize_features

    # make unhandled numerical warnings obvious
    import warnings
    warnings.filterwarnings("error")

    mat = sparse.load_npz("../data/sparse_training.npz")
    arrmat = mat.toarray()
    # split into train/val
    split_r = 0.8  # 80% train, 20% val
    cutoff = int(len(arrmat[:, 0]) * split_r)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(arrmat)
    train_data, val_data = arrmat[0:cutoff], arrmat[cutoff:]
    
    means, devs = get_standardization_mean_stddev(train_data[:, 1:-1])
    train_data[:, 1:-1] = standardize_features(train_data[:, 1:-1], means, devs)
    val_data[:, 1:-1] = standardize_features(val_data[:, 1:-1], means, devs)

    # sparsify
    train_data, val_data = sparse.lil_matrix(train_data), sparse.lil_matrix(val_data)

    lab_count = len(np.unique(arrmat[:, -1]))
    attr_count = mat.shape[1] - 2
    n_entries = mat.shape[0]
    log_reg = LogReg(k=lab_count, n=attr_count, lr=0.0001, lam=0.0001)
    import time
    t = time.perf_counter()
    log_reg.train(67, train_data, val_data=val_data, minibatch_size=None)
    log_reg.lr = 0.0000001
    log_reg.train(10000, train_data, val_data=val_data, minibatch_size=9000)
    print("Time %f" %(time.perf_counter() - t))
    test_mat = sparse.load_npz("../data/sparse_testing.npz").tolil()
    test_mat[:, 1:] = standardize_features(test_mat[:, 1:], means, devs)
    lr_pred = log_reg.classify(test_mat, id_in_mat=True, class_in_mat=False)
    output = np.zeros((test_mat.shape[0], 2), dtype=np.int64)
    output[:, 0] = test_mat[:, 0].toarray().reshape((test_mat.shape[0],))
    output[:, 1] = lr_pred.reshape((lr_pred.shape[0],))
    np.savetxt("../data/lr_basic_test_out_0001_0001_67_10000_80_20_earlyhalt_Whole_thenminibatchsize9000_lr0000001.csv", output, fmt="%d", delimiter=",", header="id,class", comments="")
