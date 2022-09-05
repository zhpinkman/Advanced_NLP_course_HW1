""" 
    Basic feature extractor
"""
from collections import defaultdict
from operator import methodcaller
import string
from consts import STOP_WORDS


def tokenize(text):
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    text = text.lower()
    words = text.split()

    words = [word for word in words if word not in STOP_WORDS]

    return words


class Features:

    def __init__(self, data_file):
        with open(data_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))

        self.tokenized_text = [tokenize(text) for text in texts]

        self.labelset = list(set(self.labels))

    @classmethod
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features
        pass


class NB_Features(Features):

    def __init__(self, data_file):
        super().__init__(data_file=data_file)

    @classmethod
    def get_features(cls, tokenized, model):
        return tokenized
