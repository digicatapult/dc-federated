"""
Contains a single class illustrating the use of the classes in
dc_fl_demo.dc_federated.DCFWorker.
"""

import io
import time
from datetime import datetime

import torch

from dc_fl_demo.utils import get_host_ip
from dc_fl_demo.example_dcf_model.torch_nn_class import ExampleModelClass
from dc_fl_demo.dc_fed_sw import DCFWorker


class ExampleLocalModel(object):
    """
    This is a simple class that illustrates how the DCFWorker class may be used to
    implement a federated local model. This talks to an ExampleGlobalModel
    object via a running DCFServer instance. For testing purposes, it writes all the
    models it creates and receives to disk.
    """
    def __init__(self):
        self.local_model = ExampleModelClass()
        self.last_update_time = datetime(2017, 1, 1)
        self.worker = DCFWorker(
            server_host_ip=get_host_ip(),
            server_port=8080
        )

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
            self.worker.get_global_model_status().decode('UTF-8'),
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
        torch.save(self.local_model, model_data)
        return model_data.getvalue()

    def run_model(self):
        """
        Shows an example loop for the worker working with the server.
        """
        # register the worker
        self.worker_id = self.worker.register_worker()
        with open(f"elm_worker_update_{self.worker_id}.torch", 'wb') as f:
            torch.save(self.local_model, f)

        # send over the local model to the server
        self.worker.send_model_update(self.serialize_model())

        # wait until the global model has updated and then retrieve it.
        while self.get_model_update_time() <= self.last_update_time:
            time.sleep(5)

        model_binary = self.worker.get_global_model()

        if len(model_binary) > 0:
            print("I got the global model!! -- transforming...")
            self.global_model = torch.load(io.BytesIO(model_binary))
            with open("elm_global_model.torch", 'wb') as f:
                torch.save(self.global_model, f)
            print(self.global_model)
