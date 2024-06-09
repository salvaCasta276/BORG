from Lib.config_loader import config
import numpy as np
from model import Model
import pickle
import pandas as pd

block_step = config["data_partition"]["block_step"]
tot_blocks = config["data_partition"]["tot_blocks"]
init_val = config["data_partition"]["init_val"]
lens = range(init_val, init_val + block_step * tot_blocks, block_step)

def main():


    for i in lens:

        dataset = np.load("Datasets/coauth_set_bs=" + str(i) + ".npy")
        np.random.shuffle(dataset)
        labels = dataset[1:, i].astype(np.float64).astype(int)
        unique, counts = np.unique(labels, return_counts=True)
        data = dataset[1:, :i]
        data = list(map(lambda x: ' '.join(x.tolist()), data))

        borg = Model(data, list(labels), len(unique))
        logits = list(borg.logits)
        results = borg.results
        print("Results:", results)
        borg.model.save_pretrained("Models/coauth_borg_bs=" + str(i))
        with open("Results/Logits/coauth_logits_bs=" + str(i), "wb") as fp:
            pickle.dump(logits, fp)

        log_history = pd.DataFrame(borg.trainer.state.log_history)
        log_history.to_csv("Results/Logs/coauth_log_bs=" + str(i) + ".csv", index=False)





if __name__ == "__main__":
    main()