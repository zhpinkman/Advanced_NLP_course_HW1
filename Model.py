from abc import ABCMeta, abstractmethod
import pickle


class Model(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    def save_model(self, model):
        with open(self.model_file, "wb") as file:
            pickle.dump(model, file)

    def load_model(self):
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)
        return model

    @abstractmethod
    def train(self, input_file):
        pass

    @abstractmethod
    def classify(self, input_file, model):
        pass
