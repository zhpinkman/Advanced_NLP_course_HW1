from operator import methodcaller
import pandas as pd
import os

from sklearn.model_selection import train_test_split


def read_file(data_file: str):

    with open(data_file) as file:
        data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, labels = map(list, zip(*data_split))
    return texts, labels


if __name__ == "__main__":
    datasets = [file for file in os.listdir('.') if file.endswith('.txt')]

    for dataset in datasets:
        texts, labels = read_file(dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        with open(os.path.join('custom', f"{dataset}.train"), 'w') as f:
            for text, label in zip(X_train, y_train):
                f.write(f"{text}\t{label}\n")

        with open(os.path.join('custom', f"{dataset}.test"), 'w') as f:
            for text in X_test:
                f.write(f"{text}\n")

        with open(os.path.join('custom', f"{dataset}.true"), 'w') as f:
            for label in y_test:
                f.write(f"{label}\n")
