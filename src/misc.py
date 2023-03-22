"""
@author Jack Ringer, Mike Adams
Date: 3/22/2023
Description:
    Contains code used to answer miscellaneous questions from the report.
"""

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from naive_bayes import NaiveBayes
from utils import get_accuracy, get_num_docs_with_feat


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


def q8_main():
    """
    Main for answering question 8 of the report.
    :return: None
    """
    with open("../data/vocabulary.txt") as f:
        vocab = f.read().splitlines()

    # load train/val data
    mat = sparse.load_npz("../data/sparse_training.npz").toarray()
    split_r = 0.8  # 80% train, 20% val
    cutoff = int(len(mat[:, 0]) * split_r)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(mat)
    train_data, val_data = mat[0:cutoff], mat[cutoff:]

    # load top 100 words from NB (trained on train_data)
    top_100_train = np.load("../results/nb_top_100_words.npy",
                            allow_pickle=True).item()
    top_100_words = top_100_train['words']
    train_freqs = np.array(top_100_train['freqs']) / len(train_data)
    top_100_indices = list(map(lambda s: vocab.index(s), top_100_words))

    # compare frequencies from training data to val data
    val_freqs = get_num_docs_with_feat(val_data[:, 1:-1])  # exclude id, label
    val_freqs = val_freqs[top_100_indices] / len(val_data)

    test_data = sparse.load_npz("../data/sparse_testing.npz").toarray()
    test_freqs = get_num_docs_with_feat(test_data[:, 1:])  # exxlude id
    test_freqs = test_freqs[top_100_indices] / len(test_data)

    # compare prob differences

    # create plot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), sharey='row')
    axs[0].hist(train_freqs,
                bins=np.logspace(start=np.log10(10 ** -4),
                                 stop=np.log10(1), num=10))
    axs[0].set_title("Train Data")
    axs[0].set_ylabel("Count")
    axs[0].set_xlabel("Bin for log(P(X))")

    axs[1].set_title("Validation Data")
    axs[1].set_xlabel("Bin for log(P(X))")
    axs[1].hist(val_freqs,
                bins=np.logspace(start=np.log10(10 ** -4),
                                 stop=np.log10(1), num=10))

    axs[2].hist(test_freqs,
                bins=np.logspace(start=np.log10(10 ** -4),
                                 stop=np.log10(1), num=10))
    axs[2].set_title("Test Data")
    axs[2].set_xlabel("Bin for log(P(X))")
    for ax in axs:
        ax.set_xscale('log')
    plt.savefig("../figures/q8_fig.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    q8_main()
