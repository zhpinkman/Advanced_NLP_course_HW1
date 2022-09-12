import argparse
from logistic_regression import LogisticRegression
from naivebayes import *
from perceptron import *


def get_arguments():

    # Please do not change the naming of these command line options or delete them. You may add other options for other hyperparameters but please provide with that the default values you used
    parser = argparse.ArgumentParser(
        description="Given a saved model and unlabelled text, neural networks classifier")
    parser.add_argument(
        "-m", help="modelfile: the name/path of the model to load after training using train.py")
    parser.add_argument(
        "-i",  help="inputfile: the name/path of the test file that has to be read one text per line")
    parser.add_argument(
        "-o", help="outputfile: the name/path of the output file to be written")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    if "nb" in args.m:
        model = NaiveBayes(model_file=args.m)
    elif "perceptron" in args.m:
        model = Perceptron(model_file=args.m)
    elif "lr" in args.m:
        model = LogisticRegression(model_file=args.m)

    else:
        # TODO Add any other models you wish to evaluate
        model = None

    trained_model = model.load_model()

    preds = model.classify(args.i, trained_model)

    # Save the predictions: one label prediction per line
    with open(args.o, "w") as file:
        for pred in preds:
            file.write(pred+"\n")
