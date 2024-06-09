import pandas as pd
import os

countries = ["ar", "mx", "ur", "pr", "sa", "rd", "cu", "gu", "pe", "co", "ch"]
dir_path = "Corpus/"

def main():
    drct = os.listdir(dir_path)
    for country in countries:
        country_df = None
        for j, file_name in enumerate(drct):
            if file_name.endswith(country + ".csv"):
                current_df = pd.read_csv(dir_path + file_name)
                if country_df is None:
                    country_df = current_df
                else:
                    country_df = pd.concat([country_df, current_df], ignore_index=True)
        country_df.to_csv("CountryCorpus/" + country + ".csv", index=False)



if __name__ == "__main__":
    main()