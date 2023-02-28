import numpy as np

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
        self.labels = np.arange(0, k, 1)
        self.n = n
        self.lr = lr
        self.lam = lam
        # weights matrix
        self.W = np.zeros((k, n + 1))

    def train(self, n_steps: int, data: np.ndarray):
        """
        Method to train logistic reg model.
        :param n_steps: int, the number of weight updates to apply
        :param data: np.ndarray, matrix of size m x (n + 2) where first col is
            ids and last row is labels
        :return:
        """
        for _ in range(n_steps):
            self._weight_update(data)

    def _weight_update(self, mat: np.ndarray):
        """
        Update the weight of our model using gradient ascent.
        :param mat: np.ndarray, matrix of size m x (n + 2) where first col is
            ids and last row is labels
        :return: None
        """
        x, y = mat[:, :-1], mat[:, -1]
        x[:, 0] = 1
        m = len(y)

        # calculate delta, delta[j][i] = 1 if row i in mat has label j, else 0
        delta = np.zeros((self.k, m))
        for y_val in self.labels:
            row = np.zeros(m)
            row[np.nonzero(y == y_val)] = 1
            delta[y_val] = row

        # get P(Y | X, W)
        xt = np.transpose(x)
        probs = np.exp(self.W @ xt)
        # last row all 1s to match equations 27-28 in Mitchell
        probs[-1, :] = 1
        col_sums = np.sum(probs, axis=0)
        probs = probs / col_sums

        # update
        w_change = self.lr * ((delta - probs) @ x - (self.lam * self.W))
        self.W = self.W + w_change


if __name__ == "__main__":
    n_classes = 3  # num classes
    n_entries = 5  # num entries
    v_size = 3  # vocab size
    mat1 = np.random.randint(0, 5, size=(n_entries, v_size + 2))
    # id col
    mat1[:, 0] = np.arange(0, n_entries, 1, dtype=np.int32)
    # labels col
    for i in range(n_entries):
        mat1[i, -1] = np.random.randint(0, n_classes)
    print(mat1)
    log_reg = LogReg(k=n_classes, n=v_size, lr=0.01, lam=1)
    print(log_reg.W)
    log_reg.train(n_steps=10, data=mat1)
    print(log_reg.W)
