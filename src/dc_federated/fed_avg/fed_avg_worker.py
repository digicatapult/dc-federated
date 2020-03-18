"""
Contains a single class illustrating the use of the classes in
dc_federated.backend.DCFWorker.
"""

import io
import time
from datetime import datetime
import logging

import torch

from dc_federated.utils import get_host_ip
from dc_federated.example_dcf_model.torch_nn_class import ExampleModelClass
from dc_federated.backend import DCFWorker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dc_federated.example_dcf_model.local_model')
logger.setLevel(level=logging.INFO)


class FedAvgWorker(object):
    """
    Class to implement the worker side of the FedAvg algorithm.
    """
    def __init__(self, fed_model_trainer, server_host_ip=None, server_port=None):
        self.fed_model = fed_model_trainer

        server_host_ip = get_host_ip() if not server_host_ip else server_host_ip
        server_port = 8080 if not server_port else server_port

        self.worker = DCFWorker(
            server_host_ip=server_host_ip,
            server_port=server_port
        )

        self.last_update_time = datetime(2017, 1, 1)
        self.global_model = None
        self.worker_id = None

    def get_model_update_time(self):
        """
        Queries the global model using the worker to get the last time
        the model was updated.

        Returns
        -------
        datetime:
            The datetime of the last update of the model.
        """
        return datetime.strptime(
            self.worker.get_global_model_status(),
            "%Y-%m-%d %H:%M:%S")

    def serialize_model(self):
        """
        Serializes the local model so that it can be sent over to the global
        model.

        Returns
        -------

        byte-string:
            A serialized version of the model.
        """
        model_data = io.BytesIO()
        torch.save(self.fed_model.get_model(), model_data)
        return model_data.getvalue()

    def run_worker_loop(self, wait_period_sec=1):
        """
        The main FedAvg client side loop.
        Check the server every wait_period_sec seconds to see if a new global model has
        been created. If yes, get the model and perform a train iteration with the given
        number of batches.
        """
        if not self.worker_id:
            self.worker_id = self.worker.register_worker()
            logger.info(f"Registered with FedAvg Server with worker id {self.worker_id}")

        while (True):
            self.fed_model.train()
            self.fed_model.test()
            logger.info(f"Finished training of local model for worker {self.worker_id}")

            self.worker.send_model_update(self.serialize_model())

            while self.get_model_update_time() <= self.last_update_time:
                time.sleep(wait_period_sec)

            model_binary = self.worker.get_global_model()
            if len(model_binary) > 0:
                self.fed_model.load_model(io.BytesIO(model_binary))

