import numpy as np
import wandb
from typing import Any, Dict
from collections import defaultdict
from IPython import embed
from Features import BOWFeatures
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

        all_words = feature_class.get_all_words()
        features, features_means, features_stds = feature_class.process_features(
            all_words=all_words,
            features_means=None,
            features_stds=None
        )

        test_feature_class = BOWFeatures(
            data_file=kwargs['dev'], no_labels=True)
        test_features, _, _ = test_feature_class.process_features(
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
            'features_means': features_means,
            'features_stds': features_stds
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
        features_means = model['features_means']
        features_stds = model['features_stds']

        feature_class = BOWFeatures(data_file=input_file, no_labels=True)

        features, _, _ = feature_class.process_features(
            all_words=all_words,
            features_means=features_means,
            features_stds=features_stds
        )

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
