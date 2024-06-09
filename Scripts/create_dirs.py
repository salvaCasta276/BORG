import os

def main():
    if not os.path.exists("Datasets"):
        os.makedirs("Datasets")
    if not os.path.exists("CountryCorpus"):
        os.makedirs("CountryCorpus")
    if not os.path.exists("Models"):
        os.makedirs("Models")
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists("Results/Logits"):
        os.makedirs("Results/Logits")
    if not os.path.exists("Results/Logs"):
        os.makedirs("Results/Logs")


if __name__ == "__main__":
    main()