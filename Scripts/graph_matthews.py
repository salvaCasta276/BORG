import numpy as np
import torch
import torch.nn as nn
from Lib.config_loader import config
import pickle
import matplotlib.pyplot as plt
import pandas as pd

block_step = config["data_partition"]["block_step"]
tot_blocks = config["data_partition"]["tot_blocks"]
init_val = config["data_partition"]["init_val"]
lens = range(init_val, init_val + block_step * tot_blocks, block_step)

log_semipath = "Results/Logs/country_log_bs="

def main():
    results = []
    print(lens)
    for i in lens:
        df = pd.read_csv(log_semipath + str(i) + ".csv")
        best_model = df["eval_matthews_correlation"].max()
        results.append(best_model)

    if len(results) > 1:
        plot_data(lens, results)
    print(results)



def plot_data(x, y):
    plt.bar(x, y)
    plt.xlabel("Block size")
    plt.ylabel("Matthews Corr")
    plt.title("Matthews Corr by Block Size")

    plt.savefig("Graphs/country_mc_over_bs.pdf")

    plt.show()




if __name__ == "__main__":
    main()