""" 
    Basic feature extractor
"""
from collections import defaultdict
from collections import Counter, defaultdict
from email.policy import default
from operator import methodcaller
import string
from consts import STOP_WORDS


def tokenize(text: str):
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    text = text.lower()
    words = text.split()

    words = [word for word in words if word not in STOP_WORDS]

    return words


# TODO: features you can use
# TODO: happy words,
# TODO: prefixes
# TODO: negative words
# TODO: checkout the paper on thumbs up for the template on the report
# TODO: for the Odyia dataset take into account the word
# TODO: Check the weighted naive bayes model


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


class BOWFeatures(Features):

    def __init__(self, data_file):
        super().__init__(data_file=data_file)

    @classmethod
    def get_features(cls, tokenized, model):
        sents_words_counts = []
        for sent_tokenized in tokenized:
            words_counts = defaultdict(int)
            for word in sent_tokenized:
                words_counts[word] += 1
            sents_words_counts.append(
                words_counts
            )
        return sents_words_counts
