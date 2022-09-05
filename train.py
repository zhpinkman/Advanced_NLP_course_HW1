import argparse
from naivebayes import *
from perceptron import *


def get_arguments():
    parser = argparse.ArgumentParser(description="Text Classifier Trainer")
    parser.add_argument(
        "-m", help="type of model to be trained: naivebayes, perceptron")
    parser.add_argument(
        "-i", help="path of the input file where training file is in the form <text>TAB<label>")
    # Respect the naming convention for the model: make sure to name it {nb, perceptron}.{4dim, authors, odiya, products}.model for your best models in your workplace otherwise the grading script will fail
    parser.add_argument("-o", help="path of the file where the model is saved")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if args.m == "naivebayes":
        model = NaiveBayes(model_file=args.o)
    elif args.m == "perceptron":
        model = Perceptron(model_file=args.o)

    else:
        # TODO Add any other models you wish to train
        model = None

    model = model.train(input_file=args.i)
