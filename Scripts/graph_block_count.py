import numpy as np
import matplotlib.pyplot as plt

block_size = 25

def main():
        dataset = np.load("Datasets/set_bs=" + str(block_size) + ".npy")
        labels = dataset[1:, block_size].astype(np.float64).astype(int)
        unique, counts = np.unique(labels, return_counts=True)
        plot_data(unique, counts)

def plot_data(x, y):
    plt.bar(x, y)
    plt.xlabel("Author")
    plt.ylabel("Block count")
    plt.title("Block count per Author")

    plt.savefig("Graphs/blocks_per_auth.pdf")

    plt.show()
    

if __name__ == "__main__":
    main()