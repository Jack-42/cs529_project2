"""
@author Jack Ringer, Mike Adams
Date: 3/22/2023
Description:
    Contains code used to answer miscellaneous questions from the report.
"""

import numpy as np
import scipy.sparse as sparse

from naive_bayes import NaiveBayes
from utils import get_accuracy


def q7_main():
    """
    Main for answering question 7 of the report
    :return: None
    """
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
    beta = 1 / attr_count
    nb = NaiveBayes(lab_count, attr_count, 1.0 + beta)
    nb.train(train_data)
    top_100_words, top_100_freqs = nb.get_best_words(train_data, 100, vocab)
    val_pred = nb.classify(val_data, id_in_mat=True, class_in_mat=True)
    print("val acc: ", get_accuracy(val_pred, val_data[:, -1]))
    print(top_100_words)
    print(top_100_freqs)
    np.save("../results/nb_top_100_words.npy", {"words": top_100_words,
                                                "freqs": top_100_freqs})


if __name__ == "__main__":
    # q7_main()
    d = np.load("../results/nb_top_100_words.npy", allow_pickle=True).item()
    print(d['words'])
    print(d['freqs'])
