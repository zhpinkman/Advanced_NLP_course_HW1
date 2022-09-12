""" 
    Basic feature extractor
"""
from collections import Counter, defaultdict
from operator import methodcaller
import string
from consts import STOP_WORDS
import pandas as pd
from tqdm import tqdm
import numpy as np
from IPython import embed
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def tokenize(text: str):
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    text = text.lower()
    words = text.split()

    words = [word for word in words if word not in STOP_WORDS]

    return words

# TODO: happy words,
# TODO: prefixes
# TODO: negative words
# TODO: checkout the paper on thumbs up for the template on the report
# TODO: for the Odyia dataset take into account the word
# TODO: Check the weighted naive bayes model


class Features:
    def __init__(self, data_file, no_labels=False):
        with open(data_file) as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        if no_labels:
            texts = list(map(list, zip(*data_split)))[0]
            self.labels = None
            self.labelset = None
        else:
            texts, self.labels = map(list, zip(*data_split))
            self.labelset = list(set(self.labels))

        self.tokenized_text = [tokenize(text) for text in texts]


class BOWFeatures(Features):

    def __init__(self, data_file, no_labels=False, ngrams=(1, 2), all_words=None, features_means=None, features_stds=None):
        super().__init__(data_file=data_file, no_labels=no_labels)
        self.ngrams = ngrams
        self.sents_words_counts = self.get_sents_words_counts()
        self.all_words = all_words
        self.features_means = features_means
        self.features_stds = features_stds

    def get_all_words(self):
        all_words_counts = defaultdict(int)

        for words_counts in tqdm(self.sents_words_counts, leave=False):
            for word, count in words_counts.items():
                all_words_counts[word] += count

        all_words = set([
            word
            for word, count
            in all_words_counts.items()
            if count >= 10
        ])

        return {
            word: i
            for i, word
            in enumerate(all_words)
        }

    def get_features(self, text, all_words):
        features = np.zeros(shape=[len(all_words)])
        for i in range(self.ngrams[0], self.ngrams[1] + 1):
            for j in range(len(text) - i + 1):
                ngram = tuple(text[j:j + i])
                if ngram in all_words:
                    features[all_words[ngram]] = 1
        return features

    def process_features(self):
        if self.all_words is None:
            self.all_words = self.get_all_words()
        features = np.array([
            self.get_features(tokenized_text, self.all_words)
            for tokenized_text in tqdm(self.tokenized_text, leave=False)
        ])  # feature vector with size (#inputs, #vocab)

        if self.features_means is None:
            self.features_means = np.mean(features, axis=0)
            self.features_stds = np.std(features, axis=0)

        features = (features - self.features_means) / self.features_stds

        return features

    def get_sents_words_counts(self):
        sents_words_counts = []
        for sent_tokenized in self.tokenized_text:
            words_counts = defaultdict(int)
            for i in range(self.ngrams[0], self.ngrams[1] + 1):
                for j in range(len(sent_tokenized) - i + 1):
                    ngram = tuple(sent_tokenized[j:j+i])
                    words_counts[ngram] += 1
            sents_words_counts.append(
                words_counts
            )
        return sents_words_counts


class BertFeatures(Features):
    def __init__(self, data_file, no_labels):
        super().__init__(data_file=data_file, no_labels=no_labels)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_features(self, tokenized):
        sentences = [' '.join(words) for words in tokenized]
        embeddings = [self.model.encode(sentence)
                      for sentence in tqdm(sentences[:100], leave=False)]
        return embeddings


class TF_IDF_Features(Features):
    def __init__(self, data_file, no_labels=False):
        super().__init__(data_file, no_labels)
        corpus = [' '.join(words) for words in self.tokenized_text]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            stop_words=None,
            max_df=0.98,
            min_df=10
        )
        self.vectorizer.fit(corpus)

    def get_features(self, tokenized_sentences):
        sentences = [' '.join(words) for words in tokenized_sentences]
        X = self.vectorizer.transform(sentences)
        return X


if __name__ == "__main__":
    pass
    # feature_class = BertFeatures(
    #     data_file='datasets/custom/products.train.txt.train',
    #     no_labels=False
    # )
    # embeddings = feature_class.get_features(feature_class.tokenized_text)
    # embed()
