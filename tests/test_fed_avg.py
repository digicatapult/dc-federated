
"""
Tests for the FedAvgServer logic. The FedAvgWorker is not tested here because
it is implicitly tested by the implementation of MNIST version of FedAvg.
Additionally, the logic in all the functions in FedAvgWorker is straightforward
enough that the amount of requried testing infrastructure is not justified.
"""

import io
import msgpack
import torch

from torch import nn
import torch.nn.functional as F

from dc_federated.algorithms.fed_avg import FedAvgServer, FedAvgModelTrainer
from dc_federated.backend import GLOBAL_MODEL, GLOBAL_MODEL_VERSION



class FedAvgTestModel(nn.Module):
    """
    Simple network for testing the dataset.
    """

    def __init__(self):
        super(FedAvgTestModel, self).__init__()
        self.lin = nn.Linear(10, 2)

    def forward(self, x):
        x = self.lin(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


class FedAvgTestTrainer(FedAvgModelTrainer):
    """
    Dummy trainer class for the test.

    """

    def __init__(self):

        self.model = FedAvgTestModel()

    def train(self):
        """
        Dummy train function.
        """
        pass

    def test(self):
        """
        Dummy test function
        """

    def get_model(self):
        """
        Returns the model for this trainer
        """
        return self.model

    def load_model(self, model_file):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        model_file: io.BytesIO or similar object
            This object should contain a serilaized model.

        """
        pass

    def load_model_from_state_dict(self, state_dict):
        """
        Loads a model from the given model file.

        Parameters
        -----------

        state_dict: dict
            Dictionary of parameter tensors.
        """
        self.model.load_state_dict(state_dict)

    def get_per_session_train_size(self):
        return 10


def assert_models_equal(model_1, model_2):
    for param_1, param_2 in zip(model_1.parameters(),
                                model_2.parameters()):
        assert torch.all(torch.eq(param_1.data, param_2.data))


def test_fed_avg_server():

    trainer = FedAvgTestTrainer()
    fed_avg_server = FedAvgServer(trainer, key_list_file=None)

    # test model is loaded properly
    model_dict = fed_avg_server.return_global_model()
    model_ret = torch.load(io.BytesIO(model_dict[GLOBAL_MODEL]))
    assert_models_equal(model_ret, trainer.model)

    # test that worker updates are received properly.
    dummy_worker_id_1 = "dummy_worker_id_1"
    worker_model_1 = FedAvgTestModel()
    fed_avg_server.worker_updates[dummy_worker_id_1] = None
    model_update = io.BytesIO()
    torch.save(worker_model_1, model_update)
    fed_avg_server.receive_worker_update(
        dummy_worker_id_1, msgpack.packb((15, model_update.getvalue())))
    assert_models_equal(worker_model_1, fed_avg_server.worker_updates[dummy_worker_id_1][2])

    # check that the global updates happen as expected
    dummy_worker_id_2 = "dummy_worker_id_2"
    fed_avg_server.update_lim = 2
    worker_model_2 = FedAvgTestModel()
    fed_avg_server.worker_updates[dummy_worker_id_2] = None
    model_update = io.BytesIO()
    torch.save(worker_model_2, model_update)
    fed_avg_server.receive_worker_update(
        dummy_worker_id_2, msgpack.packb((20, model_update.getvalue())))

    global_update_dict = {}
    sd_1 = worker_model_1.state_dict()
    sd_2 = worker_model_2.state_dict()
    for key in sd_1:
        global_update_dict[key] = (15 * sd_1[key] + 20*sd_2[key]) / 35

    test_global_model = FedAvgTestModel()
    test_global_model.load_state_dict(global_update_dict)

    assert_models_equal(
        fed_avg_server.global_model_trainer.model, test_global_model)
