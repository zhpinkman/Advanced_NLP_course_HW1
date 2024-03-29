"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y}
Please refer to lecture notes Chapter 4 for more details
"""

import math
from typing import Any, Dict
from tqdm import tqdm
import numpy as np
from IPython import embed
from Model import *
from Features import BOWFeatures, BOWWeightedFeatures, Features
from collections import defaultdict


class NaiveBayes(Model):

    def get_class_probs(self, feature_class: Features) -> Dict[Any, float]:
        labels = feature_class.labels
        class_probs = {
            label: len(np.where(np.array(labels) == label)[0]) / len(labels)
            for label in feature_class.labelset
        }
        return class_probs

    def get_word_counts_per_class(self, feature_class):
        word_counts_per_class = defaultdict(lambda: defaultdict(int))

        for words_counts, label in tqdm(zip(feature_class.sents_words_counts, feature_class.labels), leave=False):
            for word, count in words_counts.items():
                word_counts_per_class[label][word] += count

        word_counts_total_per_class = {
            label: np.sum(list(label_word_counts.values()))
            for label, label_word_counts
            in word_counts_per_class.items()
        }
        return word_counts_per_class, word_counts_total_per_class

    def train(self, input_file, **kwargs):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model
        """

        feature_class = BOWFeatures(
            data_file=input_file, ngrams=(1, kwargs['ngram'])
        )

        word_counts_per_class, word_counts_total_per_class = self.get_word_counts_per_class(
            feature_class=feature_class
        )

        all_classes_vocab = set([
            word
            for label_word_counts
            in word_counts_per_class.values()
            for word
            in label_word_counts.keys()
        ])

        word_probs_per_class = {
            label: {
                word: (1 + count) /
                (word_counts_total_per_class[label] + len(all_classes_vocab))
                for word, count
                in label_word_counts.items()
            }
            for label, label_word_counts
            in word_counts_per_class.items()
        }

        model = {
            "word_probs_per_class": word_probs_per_class,
            "classes": feature_class.labelset,
            "class_probs": self.get_class_probs(feature_class=feature_class),
            "word_counts_total_per_class": word_counts_total_per_class,
            "len_all_classes_vocab": len(all_classes_vocab),
            "ngram": kwargs['ngram'],
            "feature_type": kwargs['features']
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
        word_probs_per_class = model['word_probs_per_class']
        classes = model['classes']
        class_probs = model['class_probs']
        word_counts_total_per_class = model['word_counts_total_per_class']
        len_all_classes_vocab = model['len_all_classes_vocab']
        ngram = model['ngram']

        if model['feature_type'] == "bow":
            feature_class = BOWFeatures(
                data_file=input_file,
                no_labels=True,
                ngrams=(1, ngram)
            )
        elif model['feature_type'] == "wbow":
            feature_class = BOWWeightedFeatures(
                data_file=input_file,
                no_labels=True,
                ngrams=(1, ngram)
            )

        preds = []

        for words_counts in feature_class.sents_words_counts:
            result_probs = {
                label: math.log(class_probs[label])
                for label in classes
            }

            for word, count in words_counts.items():
                for label in classes:
                    if model['feature_type'] == "wbow":
                        if len(word) == 1 and word[0] in feature_class.word_happiness_scores:
                            weight = feature_class.word_happiness_scores[word[0]]
                        else:
                            weight = feature_class.word_happiness_scores['OOV']
                    else:
                        weight = count
                    if word in word_probs_per_class[label].keys():
                        result_probs[label] += weight * math.log(
                            word_probs_per_class[label][word]
                        )
                    else:
                        result_probs[label] += weight * math.log(
                            1 / (word_counts_total_per_class[label] +
                                 len_all_classes_vocab)
                        )
            predicted_label = list(result_probs.keys())[
                np.argmax(list(result_probs.values()))
            ]
            preds.append(predicted_label)

        return preds


if __name__ == "__main__":
    model = NaiveBayes(model_file="models/NB_temp.model")
    model.train(input_file="datasets/products.train.txt")
