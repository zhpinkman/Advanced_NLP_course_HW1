from perceptron import *
from naivebayes import *
from logistic_regression import LogisticRegression
import argparse
from sklearn.metrics import classification_report, f1_score


def eval_predictions(true_labels, predictions):
    return classification_report(
        y_true=true_labels,
        y_pred=predictions
    ), f1_score(
        y_true=true_labels,
        y_pred=predictions,
        average='weighted'
    )


def eval_predictions_by_file(true_labels_file: str, predictions_file: str):
    with open(true_labels_file, 'r') as f:
        true_labels = f.readlines()
    with open(predictions_file, 'r') as f:
        predictions = f.readlines()

    return eval_predictions(true_labels=true_labels, predictions=predictions)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", help="true labels path", type=str)
    parser.add_argument(
        "-p", help="predictions path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    report, weighted_f1 = eval_predictions_by_file(
        true_labels_file=args.t,
        predictions_file=args.p
    )

    print(report)
    print(f"Weighted F1: {weighted_f1}")
