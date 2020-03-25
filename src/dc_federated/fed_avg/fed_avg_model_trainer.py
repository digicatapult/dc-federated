"""
Absrtact class definition for implementing the FedAvg algorithm.
"""

from abc import ABC, abstractmethod


class FedAvgModelTrainer(object):
    """
    This class serves as the interface between the FedAvg algorithm implementation
    and specific domains/applications (such as MNIST) of the algorithm. See the
    module dc_federated.mnist for an example. Essentially this class contains the model
    that is being trained and has functions to train, test and get that model, all of which are
    required by the FedAvg algorithm as implemented in FedAvgServer
    and FedAvgWorker. The train function is only relevant for the workers and should implement
    a train loop for a given number of batches.
    """
    @abstractmethod
    def train(self):
        """
        This function should contain domain specific training logic.
        """
        pass

    @abstractmethod
    def test(self):
        """
        This class should contain domain specific testing logic.
        """
        pass

    @abstractmethod
    def get_model(self):
        """
        This should return the underlying model extending nn.Module.
        """
        pass

    @abstractmethod
    def load_model(self, model_file):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        model_file: io.BytesIO or similar object
            This object should contain a serilaized model.
        """
        pass

    @abstractmethod
    def load_model_from_state_dict(self, state_dict):
        """
        Loads a torch model from the corresponding state_dict. This is the
        preferred method of loading a model when the intention is to load
        the weights only.

        Parameters
        ----------

        state_dict: dict
            The state dictionary to load the model from.
        """
        pass

    @abstractmethod
    def get_per_session_train_size(self):
        """
        For a worker, returns the size of the training set used in the last train session.
        This is used by the FedAvgServer to calculate the weight assigned to the parameters
        of the corresponding worker during a FedAvg update.

        Returns
        ------

        int:
            The size of the training set in the last call to self.train().
        """
        pass
