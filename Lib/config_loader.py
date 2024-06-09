import yaml

file_path = "config.yaml"

def load_config():
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()