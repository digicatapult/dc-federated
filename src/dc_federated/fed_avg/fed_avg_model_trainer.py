from abc import ABC, abstractmethod


class FedAvgModelTrainer(object):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def load_model(self, model_file):
        pass

    @abstractmethod
    def load_model_from_state_dict(self, state_dict):
        pass

