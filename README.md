# CS529 Project 2
By Mike Adams and Jack Ringer

## Overview
This project uses Naive Bayes and logistic regression to classify the 20 newsgroups dataset. Each article is encoded into a Bag of Words feature vector such that the counts of each word is used.  Accuracy and confusion matrix metrics can be generated for both models. A maximum test accuracy of 88.5% and 84.8% for Naive Bayes and logistic regression, respectively, was achieved.  A Kullback–Leibler divergence measurement is implemented for estimating the most influential words to the Naive Bayes model. 

## Set up
First, create a new Python virtualenv, preferably with Python 3.7.2. Then, run the following command from the project root to install all needed packages:  
```
pip install -r requirements.txt
```

Use `gen_sparse_data.py` to generate sparse representations of the document data. Ensure these reside in the `data` directory before continuing.

## Hyperparameter search
The Naive Bayes and logistic regression models have their hyperparameter searches in the `gridsearch.py` file. To run:
1. Add `nb_main()` for Naive Bayes or `lr_main()` for logistic regression to the `if __name__ == "__main__"`block.
2. Adjust the `save_path` variable to save the results where desired.
3. Execute `python gridsearch.py` from the `src` directory.

NOTE:  Logistic regression was run on 15 machines to speed up computation. Two parameters can be passed via the command line to only process a subset of learning rates. For example, `python gridsearch.py 2 5` will slice the range of etas from index 2 until 5 (exclusive).

## Plots
For hyperparameter searches, `plots.py` contains code to plot training and validation accuracies versus hyperparameters. To run:  
1. Add `plot_nb_gridsearch(nb_csv, save_path)` for Naive Bayes or `plot_lr_gridsearch(lr_csv, save_path)` for logistic regression to the `if __name__ == "__main__"`block.
2. Adjust the `save_path` variable to save the plot where desired. `nb_csv`/`lr_csv` should point to the file saved during the hyperparameter search.
3. Execute `python plots.py` from the `src` directory. 

## Questions 4, 7, and 8
Code providing plots/information for these questions can be found in `misc.py`. To run:  
1. Add the appropriate function call for the desired question to the `if __name__ == "__main__"`block.
2. Execute `python misc.py` from the `src` directory. 

