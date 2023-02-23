import pandas as pd
import scipy.sparse as sparse
import os

"""
@author Jack Ringer, Mike Adams
Date: 2/22/2023
Description: 
    Script used to create and save sparse matrices from train/test data
"""


def create_n_save_sparse(inp_pth: str, save_pth: str):
    """
    Load full .csv file and convert to sparse matrix. Then save sparse matrix.
    :param inp_pth: str, path to input csv file
    :param save_pth: str, path to file where sparse matrix will be saved
    :return: None
    """
    assert save_pth.endswith(".npz"), "save_pth should end with .npz"
    test_df = pd.read_csv(inp_pth, header=None)
    sparse_mat = sparse.csr_matrix(test_df.values)
    sparse.save_npz(save_pth, sparse_mat)


def main():
    data_dir = "../data"

    # create sparse train data
    inp_train_pth = os.path.join(data_dir, "training.csv")
    train_save_pth = os.path.join(data_dir, "sparse_training.npz")
    create_n_save_sparse(inp_train_pth, train_save_pth)

    # create sparse test data
    inp_test_pth = os.path.join(data_dir, "testing.csv")
    test_save_pth = os.path.join(data_dir, "sparse_testing.npz")
    create_n_save_sparse(inp_test_pth, test_save_pth)


if __name__ == "__main__":
    # takes ~30 mins to finish
    main()
    # can load saved sparse matrices w/:
    # test_mat = sparse.load_npz("../data/sparse_testing.npz")
    # train_mat = sparse.load_npz("../data/sparse_training.npz")
