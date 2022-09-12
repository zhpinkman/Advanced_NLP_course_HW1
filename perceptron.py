"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
import wandb
from typing import Any, Dict
from collections import defaultdict
from IPython import embed
from Features import Features, BOWFeatures
from Model import *
from tqdm import tqdm
import numpy as np
from eval import eval_predictions

import warnings
warnings.filterwarnings("ignore")


class Perceptron(Model):

    def forward(self, W, x):
        z = W[:, 0] + np.dot(W[:, 1:], x.T)
        return z

    def train(self, input_file, **kwargs):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """

        # wandb.init(project=f"Perceptron Normalized Features BOW - dataset {input_file.replace('/', '')}",
        #            entity="zhpinkman")

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
        labels_transformed = [label2index[label]
                              for label in feature_class.labels]
        # weight vector with size (#classes, #vocab + 1)
        W = np.zeros(shape=[len(classes), len(all_words) + 1])
        W[:, 0] = np.array([1])

        for epoch in range(kwargs['epochs']):
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
                'W': W,
                'features_means': features_means,
                'features_stds': features_stds
            }

            train_predictions = self.classify(
                input_file=input_file,
                model=model,
                features=features
            )

            test_predictions = self.classify(
                input_file=kwargs['dev'],
                model=model,
                features=test_features
            )

            _, train_weighted_f1 = eval_predictions(
                true_labels=train_labels,
                predictions=train_predictions
            )

            _, test_weighted_f1 = eval_predictions(
                true_labels=test_labels,
                predictions=test_predictions
            )

            # wandb.log({
            #     'train_weighted_f1': train_weighted_f1,
            #     'test_weighted_f1': test_weighted_f1
            # }, step=epoch)

            # print("Epoch: {:>3} | train w-f1: ".format(
            # epoch) + f"{train_weighted_f1 * 100:.2e}" + " | Valid w-f1: " + f"{test_weighted_f1 * 100:.2e}")
        print('-----')

        # Save the model
        self.save_model(model)
        return model

    def classify(self, input_file, model, features=None):
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

        if features is None:
            feature_class = BOWFeatures(data_file=input_file, no_labels=True)

            features, _, _ = feature_class.process_features(
                all_words=all_words,
                features_means=features_means,
                features_stds=features_stds
            )

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
