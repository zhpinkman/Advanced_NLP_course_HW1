"""
    Basic feature extractor
"""
from collections import Counter, defaultdict
from operator import methodcaller
import string
from consts import STOP_WORDS
import gensim
import pandas as pd
from tqdm import tqdm
import numpy as np
from IPython import embed
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import decrypt_text


def tokenize(text: str):
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    text = text.lower()
    words = text.split()

    words = [word for word in words if word not in STOP_WORDS]

    return words


class Features:
    def __init__(self, data_file, no_labels=False, decrypt=False):
        self.decrypt = decrypt
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

        if self.decrypt:
            self.tokenized_text = [
                tokenize(decrypt_text(text)) for text in texts]
        else:
            self.tokenized_text = [tokenize(text) for text in texts]


class BOWFeatures(Features):

    def __init__(self, data_file, no_labels=False, ngrams=(1, 2), feature_class=None, decrypt=False):
        super().__init__(data_file=data_file, no_labels=no_labels, decrypt=decrypt)
        self.ngrams = ngrams
        self.sents_words_counts = self.get_sents_words_counts()
        if feature_class is None:
            self.all_words = None
            self.features_means = None
            self.features_stds = None
        else:
            self.all_words = feature_class.all_words
            self.features_means = feature_class.features_means
            self.features_stds = feature_class.features_stds
            self.ngrams = feature_class.ngrams

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
                ngram = tuple(text[j: j + i])
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
                    ngram = tuple(sent_tokenized[j: j+i])
                    words_counts[ngram] += 1
            sents_words_counts.append(
                words_counts
            )
        return sents_words_counts


class BOWWeightedFeatures(BOWFeatures):
    def __init__(self, data_file, no_labels=False, ngrams=(1, 2), feature_class=None, decrypt=False):
        super().__init__(data_file=data_file, no_labels=no_labels,
                         ngrams=ngrams, feature_class=feature_class, decrypt=decrypt)
        self.word_happiness_scores = self.get_word_happiness_scores()

    def get_word_happiness_scores(self):
        df = pd.read_csv('Hedonometer.csv')
        words = df['Word in English'].tolist()
        words.append('OOV')
        scores = df['Happiness Score'].tolist()
        scores.append(5)
        return dict(zip(words, scores))


class Word2VecFeatures(Features):

    model = None

    def __init__(self, data_file, no_labels=False, decrypt=False):
        super().__init__(data_file=data_file, no_labels=no_labels, decrypt=decrypt)
        if self.model is None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                "GoogleNews-vectors-negative300.bin.gz", binary=True)

    def process_features(self):
        embeddings = []
        for words in tqdm(self.tokenized_text, leave=False):
            word_embeddings = []
            for word in words:
                if word in self.model:
                    word_embeddings.append(self.model[word])
            if len(word_embeddings) == 0:
                embeddings.append(np.zeros(shape=(1, 300)))
            else:
                embeddings.append(
                    np.mean(np.array(word_embeddings), axis=0).reshape(1, -1)
                )
        return np.array(embeddings).squeeze(axis=1)


class TF_IDF_Features(Features):
    def __init__(self, data_file, no_labels=False, feature_class=None, decrypt=False):
        super().__init__(data_file, no_labels, decrypt=decrypt)
        if feature_class is None:
            corpus = [' '.join(words) for words in self.tokenized_text]
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                stop_words=None,
                max_df=0.98,
                min_df=10
            )
            self.vectorizer.fit(corpus)
        else:
            self.vectorizer = feature_class.vectorizer

    def process_features(self):
        sentences = [' '.join(words) for words in self.tokenized_text]
        X = self.vectorizer.transform(sentences)
        return X.toarray()


if __name__ == "__main__":
    feature_class = Word2VecFeatures(
        data_file='datasets/custom/products.train.txt.train',
        no_labels=False
    )
    embeddings = feature_class.process_features()
    embed()
    quit()
