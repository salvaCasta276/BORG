import numpy as np
from Lib.corpus_parser import parse_corpus
from Lib.config_loader import config

corpus_path = "CoauthCorpus"
block_step = config["data_partition"]["block_step"]
tot_blocks = config["data_partition"]["tot_blocks"]
init_val = config["data_partition"]["init_val"]
lens = range(init_val, init_val + block_step * tot_blocks, block_step)

def main():
    for i in lens:
        dataset, drct = parse_corpus(corpus_path, i)
        print(drct)
        np.save("Datasets/coauth_set_bs=" + str(i), dataset)


if __name__ == "__main__":
    main()
