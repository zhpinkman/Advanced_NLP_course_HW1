import numpy as np
import wandb
from typing import Any, Dict
from sklearn.metrics import classification_report
from collections import defaultdict
from IPython import embed
from Features import Features, BOWFeatures
from Model import *
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def softmax(x, axis=1):
    return np.exp(x)/np.sum(np.exp(x), axis=axis, keepdims=True)


class LogisticRegression(Model):

    def loss(self, X, Y, W):
        Z = - X @ W
        N = X.shape[0]
        loss = 1/N * (np.trace(X @ W @ Y.T) +
                      np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(self, X, Y, W, mu):
        Z = - X @ W
        P = softmax(Z, axis=1)
        N = X.shape[0]
        gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
        return gd

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

    def process_features(self, feature_class, all_words, features_means=None, features_stds=None):
        features = np.array([
            self.get_features(tokenized_text, all_words)
            for tokenized_text in tqdm(feature_class.tokenized_text, leave=False)
        ])  # feature vector with size (#inputs, #vocab)

        if features_means is None:
            features_means = np.mean(features, axis=0)
            features_stds = np.std(features, axis=0)

        features = (features - features_means) / features_stds

        return features, features_means, features_stds

    def train(self, input_file, **kwargs):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """

        onehot_encoder = OneHotEncoder(sparse=False)

        wandb.init(project=f"Logistic Regression Normalized Features BOW - dataset {input_file.replace('/', '')}",
                   entity="zhpinkman")

        feature_class = BOWFeatures(data_file=input_file)

        train_labels = feature_class.labels
        with open(kwargs['devlabels'], 'r') as f:
            test_labels = f.read().splitlines()

        all_words = self.get_all_words(
            feature_class=feature_class
        )
        features, features_means, features_stds = self.process_features(
            feature_class, all_words)

        test_feature_class = BOWFeatures(
            data_file=kwargs['dev'], no_labels=True)
        test_features, _, _ = self.process_features(
            feature_class=test_feature_class,
            all_words=all_words,
            features_means=features_means,
            features_stds=features_stds
        )

        classes = feature_class.labelset
        label2index = {
            label: i
            for i, label in enumerate(classes)
        }
        labels_transformed = np.array([label2index[label]
                                       for label in train_labels])
        test_labels_transformed = np.array([label2index[label]
                                            for label in test_labels])

        Y_onehot = onehot_encoder.fit_transform(
            labels_transformed.reshape(-1, 1))

        test_Y_onehot = onehot_encoder.transform(
            test_labels_transformed.reshape(-1, 1))

        W = np.zeros((features.shape[1], Y_onehot.shape[1]))
        # weight vector with size (#vocab, #classes)
        step = 0

        eta = 0.4
        mu = 0.01

        while step < kwargs['epochs']:
            step += 1
            W -= eta * self.gradient(features, Y_onehot, W, mu)
            epoch_train_loss = self.loss(features, Y_onehot, W)
            epoch_test_loss = self.loss(test_features, test_Y_onehot, W)

            wandb.log({
                'train_loss': epoch_train_loss,
                'test_loss': epoch_test_loss
            }, step=step)

        model = {
            'label2index': label2index,
            'all_words': all_words,
            'W': W,
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

        feature_class = BOWFeatures(data_file=input_file, no_labels=True)

        features = np.array([
            self.get_features(tokenized_text, all_words)
            for tokenized_text in tqdm(feature_class.tokenized_text, leave=False)
        ])  # feature vector with size (#inputs, #vocab)

        Z = - features @ W
        P = softmax(Z, axis=1)
        predictions = np.argmax(P, axis=1)

        index2label = {v: k for k, v in label2index.items()}
        predictions = [
            index2label[label]
            for label in predictions
        ]
        return predictions


if __name__ == "__main__":
    pass
    # model = LogisticRegression(
    #     model_file="models/logistic_regression_temp.model")
    # model.train(input_file="datasets/products.train.txt")

    # model = LogisticRegression(
    #     model_file="models/logistic_regression_temp.model")
    # trained_model = model.load_model()

    # preds = model.classify(
    #     input_file="datasets/products.train.txt",
    #     model=trained_model
    # )
