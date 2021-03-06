"""
Contains the worker side implementation of the FedAvg algorithm.
"""

import io
import time
from datetime import datetime
import logging
import msgpack

import torch

from dc_federated.utils import get_host_ip
from dc_federated.backend import GLOBAL_MODEL, GLOBAL_MODEL_VERSION, WID_LEN
from dc_federated.backend import DCFWorker


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class FedAvgWorker(object):
    """
    Class to implement the worker side of the FedAvg algorithm.

    Parameters
    ----------

    fed_model_trainer: FedAvgModelTrainer
        The trainer for the DNN model for the worker.

    private_key_file: str
        Name of the private key file to authenticate the worker.
        Name of the corresponding public key file is assumed to be
        key_file + '.pub'

    server_host_ip: str
        The ip-address of the host of the server.

    server_port: int
        The port at which the serer should listen to
    """

    def __init__(self, fed_model_trainer, private_key_file, server_protocol=None, server_host_ip=None, server_port=None):
        self.fed_model = fed_model_trainer

        server_protocol = 'http' if server_protocol is None else 'https'
        server_host_ip = get_host_ip() if not server_host_ip else server_host_ip
        server_port = 8080 if not server_port else server_port

        self.worker_version_of_global_model = 0

        self.worker = DCFWorker(
            server_protocol=server_protocol,
            server_host_ip=server_host_ip,
            server_port=server_port,
            global_model_version_changed_callback=self.global_model_version_changed_callback,
            get_worker_version_of_global_model=lambda : self.worker_version_of_global_model,
            private_key_file=private_key_file
        )

        self.global_model = None
        self.worker_id = None

        self.initialize()

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
        logger.info(
            f"Finished training of local model for worker {self.worker_id[0:WID_LEN]}")

    def send_model_update(self):
        """
        Sends the current model to the server.
        """
        self.worker.send_model_update(
            msgpack.packb((self.fed_model.get_per_session_train_size(),
                          self.serialize_model()))
        )
        logger.info(
            f"Sent model update from worker {self.worker_id[0:WID_LEN]} to the server.")

    def initialize(self):
        """
        Initializes this FedAvg worker by registering the worker with the server,
        getting the starting update time of the server and running a local
        train-test loop.
        """
        if not self.worker_id:
            self.worker_id = self.worker.register_worker()
            logger.info(
                f"Registered with FedAvg Server with worker id {self.worker_id[0:WID_LEN]}")

        self.train_and_test_model()
        self.send_model_update()

    def global_model_version_changed_callback(self, model_dict):
        """
        Callback for when the global model status has changed. This function
        essentially ensures that the global model update time is more recent
        than the time this worker has, and if so updates the local model and
        carries out a local train and test iteration.

        Paramters
        ---------
        model_dict: dict
            A dictionary with the keys
            GLOBAL_MODEL: serialized global model.
            GLOBAL_MODEL_VERSION: version of the global model
        """
        if not isinstance(model_dict, dict) or \
                GLOBAL_MODEL not in model_dict or \
                GLOBAL_MODEL_VERSION not in model_dict:
            logger.error("Invalid model received from the server.")
            return

        self.worker_version_of_global_model = model_dict[GLOBAL_MODEL_VERSION]
        new_model = torch.load(io.BytesIO(model_dict[GLOBAL_MODEL]))
        self.fed_model.load_model_from_state_dict(new_model.state_dict())
        self.train_and_test_model()
        self.send_model_update()

    def start(self):
        """
        Simply starts the DCFWorker run loop.
        """
        self.worker.run()
