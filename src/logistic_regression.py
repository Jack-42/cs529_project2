import numpy as np

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
        self.W = np.zeros((k, n + 1))

    def train(self, n_steps: int, data: np.ndarray, print_acc=True, val_data: np.ndarray = None):
        """
        Method to train logistic reg model.
        :param n_steps: int, the number of weight updates to apply
        :param data: np.ndarray, matrix of size m x (n + 2) where first col is
            ids and last col is labels
        :param val_data: np.ndarray, test data matrix of size m' x (n' + 2), halts if acc decreases
        :return: None
        """
        if not (val_data is None):
            last_acc = get_accuracy(self.classify(val_data[:, :-1], id_in_mat=True, class_in_mat=False), val_data[:, -1])
        if print_acc:
            print("train accuracy on train data before training:  %f" % (get_accuracy(self.classify(data[:, :-1], id_in_mat=True, class_in_mat=False), data[:, -1])))
            if not (val_data is None):
                print("train accuracy on test data before training:  %f" % (last_acc))
        for step in range(n_steps):
            self._weight_update(data)
            if not (val_data is None):
                curr_acc = get_accuracy(self.classify(val_data[:, :-1], id_in_mat=True, class_in_mat=False), val_data[:, -1])
            if print_acc:
                print("train accuracy on train data at step %d:  %f" % (step, get_accuracy(self.classify(data[:, :-1], id_in_mat=True, class_in_mat=False), data[:, -1])))
                if not (val_data is None):
                    print("train accuracy on test data at step %d:  %f" % (step, curr_acc))

            if (not (val_data is None)) and (curr_acc < last_acc):
                # stop early
                if print_acc:
                    print("stopped after step %d (early)" % (step))
                return

            if not (val_data is None):
                last_acc = curr_acc

    def _weight_update(self, mat: np.ndarray):
        """
        Update the weight of our model using gradient ascent.
        :param mat: np.ndarray, matrix of size m x (n + 2) where first col is
            ids and last col is labels
        :return: None
        """
        x, y = mat[:, :-1], mat[:, -1]
        x[:, 0] = 1
        m = len(y)

        # calculate delta, delta[j][i] = 1 if row i in mat has label j, else 0
        delta = np.zeros((self.k, m))
        for i, y_val in enumerate(self.labels):
            row = np.zeros((m,))
            row[np.nonzero(y == y_val)] = 1
            delta[i] = row

        # get P(Y | X, W)
        probs = self._P_Y_given_X(x)

        # update
        w_change = self.lr * ((delta - probs) @ x - (self.lam * self.W))
        self.W = self.W + w_change

    def _P_Y_given_X(self, x):
        xt = np.transpose(x)

        with np.errstate(over='ignore', invalid='ignore'):
            probs = np.exp(self.W @ xt)
        
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
        return probs  # (label_size, # docs)

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
        start = 0
        if id_in_mat:
            start = 1
        end = mat.shape[1]
        if class_in_mat:
            end = -1
        
        x = mat[:, start:end]
        # add column of 1s
        x = np.concatenate([np.ones((mat.shape[0], 1)), x], axis=1)

        probs = self._P_Y_given_X(x)

        return np.argmax(probs, axis=0) + 1

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
    # log_reg.train(n_steps=10, data=mat1)
    # print(log_reg.W)
    import scipy.sparse as sparse
    from utils import get_standardization_mean_stddev, standardize_features

    # make unhandled numerical warnings obvious
    import warnings
    warnings.filterwarnings("error")

    mat = sparse.load_npz("../data/sparse_training.npz").toarray()
    means, devs = get_standardization_mean_stddev(mat[:, 1:-1])
    mat[:, 1:-1] = standardize_features(mat[:, 1:-1], means, devs)
    lab_count = len(np.unique(mat[:, -1]))
    attr_count = mat.shape[1] - 2
    n_entries = mat.shape[0]
    log_reg = LogReg(k=lab_count, n=attr_count, lr=0.01, lam=1)
    log_reg.train(100, mat)
    test_mat = sparse.load_npz("../data/sparse_testing.npz").toarray()
    test_mat[:, 1:] = standardize_features(test_mat[:, 1:], means, devs)
    lr_pred = log_reg.classify(test_mat, id_in_mat=True, class_in_mat=False)
    output = np.zeros((test_mat.shape[0], 2), dtype=np.int64)
    output[:, 0] = test_mat[:, 0]
    output[:, 1] = lr_pred
    np.savetxt("../data/lr_basic_test_out.csv", output, fmt="%d", delimiter=",", header="id,class", comments="")