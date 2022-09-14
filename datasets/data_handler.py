from operator import methodcaller
import pandas as pd
import os

from sklearn.model_selection import train_test_split

import string

alphabet = string.ascii_lowercase


def decrypt_text(encrypted_message, key=13):
    decrypted_message = ""

    for c in encrypted_message:

        if c in alphabet:
            position = alphabet.find(c)
            new_position = (position - key) % 26
            new_character = alphabet[new_position]
            decrypted_message += new_character
        else:
            decrypted_message += c

    words = decrypted_message.split()
    words = [word[::-1] for word in words]
    return ' '.join(words)


if __name__ == "__main__":
    pass


def read_file(data_file: str):

    with open(data_file) as file:
        data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, labels = map(list, zip(*data_split))
    return texts, labels


def decrypt_questions_dataset_splits():
    with open('custom/questions.train.txt.train', 'r') as f:
        lines = f.read().splitlines()

    texts = []
    labels = []
    for line in lines:
        text, label = line.split('\t')
        texts.append(decrypt_text(text))
        labels.append(label)

    with open('custom/questions.train.txt.train', 'w') as f:
        for text, label in zip(texts, labels):
            f.write(f"{text}\t{label}\n")

    with open('custom/questions.train.txt.test', 'r') as f:
        texts = f.read().splitlines()

    texts = [decrypt_text(text) for text in texts]

    with open('custom/questions.train.txt.test', 'w') as f:
        for text in texts:
            f.write(f"{text}\n")


if __name__ == "__main__":
    """
    Split each dataset into 80 / 20 splits for train and validation to check the performance of the models on
    and save them in train, test, true extension endings files.  
    """

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
