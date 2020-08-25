"""
Contains a single class illustrating the use of the classes in
dc_federated.backend.DCFWorker.
"""

import io
from datetime import datetime
import logging

import torch

from dc_federated.utils import get_host_ip
from dc_federated.examples.example_dcf_model.torch_nn_class import ExampleModelClass
from dc_federated.backend import DCFWorker, GLOBAL_MODEL_VERSION, GLOBAL_MODEL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ExampleLocalModel(object):
    """
    This is a simple class that illustrates how the DCFWorker class may be used to
    implement a federated local model. This talks to an ExampleGlobalModel
    object via a running DCFServer instance. For testing purposes, it writes all the
    models it creates and receives to disk.
    """
    def __init__(self, server_host_ip=None, server_port=None):
        self.local_model = ExampleModelClass()
        self.last_update_time = datetime(2017, 1, 1)

        server_host_ip = get_host_ip() if server_host_ip is None else server_host_ip
        server_port = 8080 if server_port is None else server_port

        print(f"Server host ip {server_host_ip}")

        self.worker = DCFWorker(
            server_host_ip=server_host_ip,
            server_port=server_port,
            global_model_version_changed_callback=self.global_model_status_changed_callback,
            get_worker_version_of_global_model=lambda : self.worker_version_of_global_model,
            private_key_file=None
        )

        self.global_model = None
        self.worker_version_of_global_model = -1

        # register the worker
        self.worker_id = self.worker.register_worker()
        with open(f"elm_worker_update_{self.worker_id}.torch", 'wb') as f:
            torch.save(self.local_model, f)

        # send the model update
        self.worker.send_model_update(self.serialize_model())

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
        torch.save(self.local_model, model_data)
        return model_data.getvalue()

    def global_model_status_changed_callback(self, model_dict):
        """
        Example showing a callback for change to the global model status.
        """
        logger.info(f"I got the global model version {model_dict[GLOBAL_MODEL_VERSION]}"
                    f"!! -- transforming...")
        self.global_model = torch.load(io.BytesIO(model_dict[GLOBAL_MODEL]))
        with open("elm_global_model.torch", 'wb') as f:
            torch.save(self.global_model, f)
        logger.info(self.global_model)

    def start(self):
        """
        Example showing how to start the worker - simply calls the
        DCFWorker run().
        """
        self.worker.run()
