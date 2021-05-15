import os

import pandas as pd


def create_csv_from_images():
    file_names = {"Animal": [], "Label": []}
    for (dirpath, dirnames, filenames) in os.walk("CatDogDataset"):
        for files in filenames:
            if files.find('cat'):
                file_names["Animal"].append(files)
                file_names["Label"].append('0')
            else:
                file_names["Animal"].append(files)
                file_names["Label"].append('1')
    df = pd.DataFrame(columns=('Animal', 'Label'), data=file_names)
    df.to_csv("cats_dog.csv", index=False)
    print(df.head())

if __name__ == '__main__':
    create_csv_from_images()