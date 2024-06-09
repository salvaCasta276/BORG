import numpy as np
import torch
import torch.nn as nn
from Lib.config_loader import config
import pickle
import matplotlib.pyplot as plt

block_step = config["data_partition"]["block_step"]
init_val = config["data_partition"]["init_val"]
tot_blocks = config["data_partition"]["tot_blocks"]
lens = range(init_val, init_val + block_step * tot_blocks, block_step)

def main():
    logits = []
    for i in lens:
        with open("Results/Logits/country_logits_bs=" + str(i), "rb") as fp:
            current = list(zip(*pickle.load(fp)))
        keys = torch.tensor(torch.stack(current[0]), dtype=torch.float)
        values = torch.tensor(current[1], dtype=torch.long)
        logits.append((keys, values))

    cross_entropy = nn.CrossEntropyLoss()
    results = list(map(lambda x: cross_entropy(x[0], x[1]), logits))

    if len(results) > 1:
        plot_data(lens, results)
    print(results)


def plot_data(x, y):
    plt.bar(x, y)
    plt.xlabel("Block size")
    plt.ylabel("CE Loss")
    plt.title("Cross Entropy Loss by Block Size")

    plt.savefig("Graphs/country_ce_over_bs.pdf")

    plt.show()





if __name__ == "__main__":
    main()