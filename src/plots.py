import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
@author Jack Ringer, Mike Adams
Date: 3/1/2023
Description:
    This file contains various funcs used to plot results
"""


def plot_nb_gridsearch(results_csv_path: str, save_pth: str = None):
    """
    Plot the results of the param search for Naive-Bayes
    :param results_csv_path: str, path to results.csv file
    :param save_pth: str (optional), path to save figure to
    :return: None
    """
    results = pd.read_csv(results_csv_path)
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


def plot_lr_gridsearch(results_csv_path: str, save_pth: str = None):
    """
    Plot the results from LR gridsearch
    :param results_csv_path: str, path to results csv file
    :param save_pth: str (optional), path to save figure to
    :return: None
    """
    results = pd.read_csv(results_csv_path, index_col=0)
    keys = ["eta", "lambda", "n_iter"]
    xlabels = {"eta": '\u03B7', "lambda": '\u03BB', "n_iter": "num. iterations"}
    fig, axs = plt.subplots(nrows=1, ncols=len(keys), figsize=(16, 6), sharey='row')
    axs[0].set_ylabel('Accuracy')
    for i in range(len(keys)):
        key = keys[i]
        ax = axs[i]
        results_avg = results.groupby(key).mean()
        ax.set_xscale('log')
        ax.plot(results_avg.index, results_avg['train_acc'], label='training accuracy')
        ax.plot(results_avg.index, results_avg['val_acc'], label='validation accuracy')
        ax.legend()
        # ax.title('LR accuracy for different values of \u03B7')
        ax.set_xlabel(xlabels[key])
    if save_pth is not None:
        plt.savefig(save_pth, dpi=300)
    plt.show()


def plot_confusion_matrix(c: np.ndarray, title: str = None, save_pth: str = None):
    """
    Plot the heatmap for a confusion matrix
    :param c: np.ndarray, c[i, j] is the # of times an instance with class j
        was classified as category i
    :param title: str, title of plot
    :param save_pth: str (optional), path to save figure to
    :return: None
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(c, cmap="hot", interpolation="nearest")
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    plt.tick_params(labelbottom=False, labeltop=True)
    ticks = np.arange(0, 20, 1, dtype=np.int32)
    labels = ticks + 1
    labels = [str(l) for l in labels]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.gca().set_xticklabels(labels)
    plt.gca().set_yticklabels(labels)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    if save_pth is not None:
        plt.savefig(save_pth, dpi=300)
    plt.show()


if __name__ == "__main__":
    nb_csv = "../results/nb_results.csv"
    lr_csv = "../results/lr_results_combined1.csv"
    import os

    os.makedirs("../figures", exist_ok=True)
    save_path = "../figures/nb_beta_acc.png"
    plot_lr_gridsearch(lr_csv, None)
