import os

import matplotlib.pyplot as plt
import pandas as pd

"""
@author Jack Ringer, Mike Adams
Date: 3/1/2023
Description:
    This file contains various funcs used to plot results
"""


def plot_nb_gridsearch(results_csv: str, save_pth: str = None):
    """
    Plot the results of the param search for Naive-Bayes
    :param results_csv: str, path to results.csv file
    :param save_pth: str (optional), path to save figure to
    :return: None
    """
    results = pd.read_csv(results_csv)
    plt.xscale('log')
    plt.plot(results['beta'], results['train_acc'], label='training accuracy')
    plt.plot(results['beta'], results['val_acc'], label='validation accuracy')
    plt.legend()
    plt.title('Naive Bayes accuracy for different values of \u03B2')
    plt.ylabel('Accuracy')
    plt.xlabel('\u03B2')
    if save_pth is not None:
        plt.savefig(save_pth, dpi=300)
    plt.show()


if __name__ == "__main__":
    nb_csv = "../results/nb_results.csv"
    save_path = "../figures/nb_beta_acc.png"
    plot_nb_gridsearch(nb_csv, save_path)
