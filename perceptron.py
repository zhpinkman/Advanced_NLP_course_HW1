"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
import wandb
from typing import Any, Dict
from collections import defaultdict
from IPython import embed
from Features import BOWFeatures, TF_IDF_Features, Word2VecFeatures
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

        wandb.init(project=f"Perceptron Normalized Features BOW - ngram {kwargs['ngram']} - dataset {input_file.replace('/', '')} - comment {kwargs['wandb']}",
                   entity="zhpinkman")

        if kwargs['features'] == "bow":
            feature_class = BOWFeatures(
                data_file=input_file,
                ngrams=(1, kwargs['ngram']),
                decrypt=kwargs['decrypt'] is not None
            )
        elif kwargs['features'] == 'tfidf':
            feature_class = TF_IDF_Features(
                data_file=input_file,
                decrypt=kwargs['decrypt'] is not None
            )
        elif kwargs['features'] == 'word2vec':
            feature_class = Word2VecFeatures(
                data_file=input_file,
                decrypt=kwargs['decrypt'] is not None
            )

        train_labels = feature_class.labels
        with open(kwargs['devlabels'], 'r') as f:
            test_labels = f.read().splitlines()

        features = feature_class.process_features()

        if kwargs['features'] == "bow":
            test_feature_class = BOWFeatures(
                data_file=kwargs['dev'],
                no_labels=True,
                feature_class=feature_class,
                decrypt=kwargs['decrypt'] is not None
            )
        elif kwargs['features'] == "tfidf":
            test_feature_class = TF_IDF_Features(
                data_file=kwargs['dev'],
                no_labels=True,
                feature_class=feature_class,
                decrypt=kwargs['decrypt'] is not None
            )
        elif kwargs['features'] == "word2vec":
            test_feature_class = Word2VecFeatures(
                data_file=kwargs['dev'],
                no_labels=True,
                decrypt=kwargs['decrypt'] is not None
            )

        test_features = test_feature_class.process_features()

        classes = feature_class.labelset
        label2index = {
            label: i
            for i, label in enumerate(classes)
        }
        labels_transformed = [label2index[label]
                              for label in feature_class.labels]
        # weight vector with size (#classes, #vocab + 1)
        W = np.zeros(shape=[len(classes), features.shape[1] + 1])
        W[:, 0] = np.array([1])

        for epoch in tqdm(range(kwargs['epochs']), leave=False):
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
                'W': W,
                'feature_class': feature_class
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

            wandb.log({
                'train_weighted_f1': train_weighted_f1,
                'test_weighted_f1': test_weighted_f1
            }, step=epoch)

            # print("Epoch: {:>3} | train w-f1: ".format(
            # epoch) + f"{train_weighted_f1 * 100:.2e}" + " | Valid w-f1: " + f"{test_weighted_f1 * 100:.2e}")
        print('-----')

        if kwargs['features'] in ["bow", 'wbow']:
            feature_class.sents_words_counts = None
            feature_class.tokenized_text = None

        model = {
            'label2index': label2index,
            'W': W,
            'feature_class': feature_class
        }

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
        feature_class = model['feature_class']

        if features is None:
            if type(feature_class).__name__ == "BOWFeatures":
                test_feature_class = BOWFeatures(
                    data_file=input_file,
                    no_labels=True,
                    feature_class=feature_class,
                    decrypt=feature_class.decrypt
                )
            elif type(feature_class).__name__ == "TF_IDF_Features":
                test_feature_class = TF_IDF_Features(
                    data_file=input_file,
                    no_labels=True,
                    feature_class=feature_class,
                    decrypt=feature_class.decrypt
                )
            elif type(feature_class).__name__ == "Word2VecFeatures":
                test_feature_class = Word2VecFeatures(
                    data_file=input_file,
                    no_labels=True,
                    decrypt=feature_class.decrypt
                )

            features = test_feature_class.process_features()

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
