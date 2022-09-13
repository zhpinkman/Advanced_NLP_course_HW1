import argparse
from logistic_regression import LogisticRegression
from naivebayes import *
from perceptron import *


def get_arguments():
    parser = argparse.ArgumentParser(description="Text Classifier Trainer")
    parser.add_argument(
        "-m", help="type of model to be trained: naivebayes, perceptron", type=str)
    parser.add_argument(
        "-i", help="path of the input file where training file is in the form <text>TAB<label>", type=str)
    parser.add_argument(
        "--dev", help="path of the input file where evaluation file is in the form <text>", type=str
    )
    parser.add_argument(
        "--devlabels", help="path of the input file where evaluation true labels file is in the form <label>", type=str
    )
    parser.add_argument(
        "--epochs", help='Number of epochs for the training stage', type=int, default=80
    )
    parser.add_argument(
        "--ngram", help="cap of the ngram getting used for the bag of words featurization", type=int, default=3
    )
    parser.add_argument(
        '--features', help="Feature used for training", default="bow", type=str
    )

    parser.add_argument(
        '--wandb', help="Wandb name when logging", default="normal", type=str
    )

    parser.add_argument(
        '--decrypt', help="whether to decrypt the content of the dataset or not",
        action=argparse.BooleanOptionalAction
    )
    # Respect the naming convention for the model: make sure to name it {nb, perceptron}.{4dim, authors, odiya, products}.model for your best models in your workplace otherwise the grading script will fail
    parser.add_argument("-o", help="path of the file where the model is saved")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.m == "naivebayes":
        model = NaiveBayes(model_file=args.o)
    elif args.m == "perceptron":
        model = Perceptron(model_file=args.o)
    elif args.m == "logistic_regression":
        model = LogisticRegression(model_file=args.o)

    else:
        # TODO Add any other models you wish to train
        model = None

    model = model.train(args.i, **vars(args))
