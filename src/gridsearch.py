import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import scipy.sparse as sparse

from naive_bayes import NaiveBayes
from utils import get_accuracy

"""
@author Jack Ringer, Mike Adams
Date: 3/1/2023
Description:
    File use to carry out grid-search of params for models.
"""


def nb_main(data: np.ndarray, save_pth: str):
    """
    Carry out grid-search for Naive-Bayes model. Will save results to csv
    file.
    :param data: np.ndarray, rows are entries. labels expected in last column
        and ids in first col
    :param save_pth: str, path to save results csv file to
    :return: None
    """
    assert save_pth.endswith(".csv"), "Save file must be a csv!"
    label_count = len(np.unique(data[:, -1]))
    attr_count = data.shape[1] - 2

    # split into train/val
    split_r = 0.8  # 80% train, 20% val
    cutoff = int(len(data[:, 0]) * split_r)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(data)
    train_data, val_data = data[0:cutoff], data[cutoff:]

    beta_vals = np.linspace(.00001, 1, 100)
    results = []
    for beta in tqdm(beta_vals):
        res = {"beta": beta}
        alpha = 1 + beta
        nb = NaiveBayes(label_count, attr_count, alpha)
        nb.train(train_data)
        train_pred = nb.classify(train_data, id_in_mat=True, class_in_mat=True)
        val_pred = nb.classify(val_data, id_in_mat=True, class_in_mat=True)
        res["train_acc"] = get_accuracy(train_pred, train_data[:, -1])
        res["val_acc"] = get_accuracy(val_pred, val_data[:, -1])
        results.append(res)
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_pth)
    print("Successfully saved results to:", save_pth)


if __name__ == "__main__":
    save_dir = "../results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "nb_results.csv")
    mat = sparse.load_npz("../data/sparse_training.npz").toarray()
    nb_main(mat, save_path)
