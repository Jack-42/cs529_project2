import glob
import os

import pandas as pd

if __name__ == "__main__":
    res_dir = "../results/"
    prefix = "lr_results"
    paths = glob.glob(os.path.join(res_dir, prefix + "*csv"))
    df = pd.concat(map(pd.read_csv, paths), ignore_index= True)
    df = df.drop(columns=["Unnamed: 0"])
    df.to_csv("../results/lr_results_combined1.csv")