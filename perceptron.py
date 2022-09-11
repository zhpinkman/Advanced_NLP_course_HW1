"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
from typing import Any, Dict
from sklearn.metrics import classification_report
from collections import defaultdict
from IPython import embed
from Features import Features, BOWFeatures
from Model import *
from tqdm import tqdm
import numpy as np


class Perceptron(Model):

    @classmethod
    def get_all_words(cls, feature_class: Features):
        sents_words_counts = feature_class.get_features(
            feature_class.tokenized_text,
            None
        )
        all_words_counts = defaultdict(int)

        for words_counts in tqdm(sents_words_counts, leave=False):
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
        for word in set(text):
            if word in all_words:
                features[all_words[word]] = 1
        return features

    def forward(self, W, x):
        z = W[:, 0] + np.dot(W[:, 1:], x.T)
        return z

    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """

        feature_class = BOWFeatures(data_file=input_file)
        all_words = self.get_all_words(
            feature_class=feature_class
        )
        features = np.array([
            self.get_features(tokenized_text, all_words)
            for tokenized_text in tqdm(feature_class.tokenized_text, leave=False)
        ])  # feature vector with size (#inputs, #vocab)

        classes = feature_class.labelset
        label2index = {
            label: i
            for i, label in enumerate(classes)
        }
        labels_transformed = [label2index[label]
                              for label in feature_class.labels]
        # weight vector with size (#classes, #vocab + 1)
        W = np.zeros(shape=[len(classes), len(all_words) + 1])
        W[:, 0] = np.array([1])

        for _ in range(100):
            num_wrong_classified = 0
            for i in range(features.shape[0]):
                z = self.forward(W=W, x=features[i, :])
                y_hats = np.argmax(z, axis=0)
                if y_hats != labels_transformed[i]:
                    num_wrong_classified += 1
                    W[labels_transformed[i], 1:] += features[i, :]
                    W[y_hats, 1:] -= features[i, :]

        model = {
            'label2index': label2index,
            'all_words': all_words,
            'W': W
        }

        # Save the model
        self.save_model(model)
        return model

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """
        W = model['W']
        label2index = model['label2index']
        all_words = model['all_words']

        feature_class = BOWFeatures(data_file=input_file)

        features = np.array([
            self.get_features(tokenized_text, all_words)
            for tokenized_text in tqdm(feature_class.tokenized_text, leave=False)
        ])  # feature vector with size (#inputs, #vocab)
        predictions = []
        for i in range(features.shape[0]):
            z = self.forward(W=W, x=features[i, :])
            y_hats = np.argmax(z, axis=0)
            predictions.append(y_hats)

        index2label = {v: k for k, v in label2index.items()}
        predictions = [
            index2label[label]
            for label in predictions
        ]
        print(classification_report(feature_class.labels, predictions))

        return predictions


if __name__ == "__main__":
    # model = Perceptron(model_file="models/Perceptron_temp.model")
    # model.train(input_file="datasets/products.train.txt")

    model = Perceptron(model_file="models/Perceptron_temp.model")
    trained_model = model.load_model()

    preds = model.classify(
        input_file="datasets/products.train.txt",
        model=trained_model
    )
