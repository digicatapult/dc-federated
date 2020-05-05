"""
Contains the worker side implementation of the FedAvg algorithm.
"""

import io
import time
from datetime import datetime
import logging
import pickle

import torch

from dc_federated.utils import get_host_ip
from dc_federated.backend import DCFWorker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dc_federated.fed_avg.fed_avg_worker')
logger.setLevel(level=logging.INFO)


class FedAvgWorker(object):
    """
    Class to implement the worker side of the FedAvg algorithm.

    Parameters
    ----------

    fed_model_trainer: FedAvgModelTrainer
        The trainer for the DNN model for the worker.

    server_host_ip: str
        The ip-address of the host of the server.

    server_port: int
        The port at which the serer should listen to
    """
    def __init__(self, fed_model_trainer, server_host_ip=None, server_port=None):
        self.fed_model = fed_model_trainer

        server_host_ip = get_host_ip() if not server_host_ip else server_host_ip
        server_port = 8080 if not server_port else server_port

        self.worker = DCFWorker(
            server_host_ip=server_host_ip,
            server_port=server_port,
            global_model_status_changed_callback=self.global_model_status_changed_callback
        )

        self.last_update_time = datetime(2017, 1, 1)
        self.global_model = None
        self.worker_id = None

        self.initialize()

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
            self.worker.current_global_model_status,
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

    def train_and_test_model(self):
        """
        Run a training and testing iteration on the local model.
        """
        self.fed_model.train()
        self.fed_model.test()
        logger.info(f"Finished training of local model for worker {self.worker_id}")

    def send_model_update(self):
        """
        Sends the current model to the server.
        """
        self.worker.send_model_update(
            pickle.dumps((self.fed_model.get_per_session_train_size(),
                          self.serialize_model()))
        )
        logger.info(f"Sent model update from worker {self.worker_id} to the server.")

    def initialize(self):
        """
        Initializes this FedAvg worker by registering the worker with the server,
        getting the starting update time of the server and running a local
        train-test loop.
        """
        if not self.worker_id:
            self.worker_id = self.worker.register_worker()
            logger.info(f"Registered with FedAvg Server with worker id {self.worker_id}")

        self.last_update_time = self.get_model_update_time()
        self.train_and_test_model()
        self.send_model_update()

    def global_model_status_changed_callback(self):
        """
        Callback for when the global model status has changed. This function
        essentially ensures that the global model update time is more recent
        than the time this worker has, and if so updates the local model and
        carries out a local train and test iteration.
        """
        if self.get_model_update_time() <= self.last_update_time:
            return
        else:
            model_binary = self.worker.get_global_model()
            if len(model_binary) > 0:
                new_model = torch.load(io.BytesIO(model_binary))
                self.fed_model.load_model_from_state_dict(new_model.state_dict())
            self.train_and_test_model()
            self.send_model_update()

    def start(self):
        """
        Simply starts the DCFWorker run loop.
        """
        self.worker.run()
