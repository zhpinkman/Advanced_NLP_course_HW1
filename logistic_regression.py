import numpy as np
import wandb
from typing import Any, Dict
from IPython import embed
from Features import BOWFeatures, TF_IDF_Features, Word2VecFeatures
from Model import *
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def softmax(x, axis=1):
    return np.exp(x)/np.sum(np.exp(x), axis=axis, keepdims=True)


class LogisticRegression(Model):

    def loss(self, X, Y, W):
        """Compute the loss for a given batch

        Args:
            X : Feature vectors
            Y : True labels
            W : Weight vectors

        Returns:
         loss value computed
        """
        Z = - X @ W
        loss = (1/X.shape[0]) * (np.trace(X @ W @ Y.T) +
                                 np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(self, X, Y, W, lamda):
        """_summary_

        Args:
            X : Feature vectors
            Y : True labels
            W : Weight vectors
            lamda

        Returns:
            compute the gradient which follows the same procedure of calculus as "loss".
        """
        Z = - X @ W
        P = softmax(Z, axis=1)
        gd = (1/X.shape[0]) * (X.T @ (Y - P)) + 2 * lamda * W
        return gd

    def train(self, input_file, **kwargs):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model
        """

        onehot_encoder = OneHotEncoder(sparse=False)

        wandb.init(project=f"Logistic Regression Normalized Features BOW - ngram {kwargs['ngram']} - dataset {input_file.replace('/', '')}- comment {kwargs['wandb']}",
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

        lr = 0.4
        lamda = 0.01

        while step < kwargs['epochs']:
            step += 1
            W -= lr * self.gradient(features, Y_onehot, W, lamda)
            epoch_train_loss = self.loss(features, Y_onehot, W)
            epoch_test_loss = self.loss(test_features, test_Y_onehot, W)

            wandb.log({
                'train_loss': epoch_train_loss,
                'test_loss': epoch_test_loss
            }, step=step)

        if kwargs['features'] in ["bow", "wbow"]:
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
        feature_class = model['feature_class']

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
