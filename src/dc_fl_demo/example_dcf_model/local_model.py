import io
import time
from datetime import datetime

import torch

from dc_fl_demo.utils import get_host_ip
from dc_fl_demo.example_dcf_model.torch_nn_class import ExampleModelClass
from dc_fl_demo.dc_fed_sw.dc_federated import DCFWorker


class ExampleLocalModel(object):
    """
    This is a simple class that illustrates how the DCFWorker class may be used to
    implement a federated local model. This talks to the ExampleGlobalModel
    class.
    """
    def __init__(self):
        self.local_model = ExampleModelClass()
        self.last_update_time = datetime(2017, 1, 1)
        self.worker = DCFWorker(
            server_host_ip=get_host_ip(),
            server_port=8080
        )

        self.run_model()

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
        torch.save(self.local_model, model_data)
        return model_data.read()

    def run_model(self, worker_id):
        """
        Shows an example loop for the worker working with the server.
        """
        # register the worker
        self.worker_id = self.worker.register_worker()

        # send over the local model to the server
        self.worker.send_model_update(self.serialize_model())

        # wait until the global model has updated and then retrieve it.
        while self.get_model_update_time() <= self.last_update_time:
            time.sleep(5)

        global_model = self.worker.get_global_model()

        if global_model is not None:
            print("I got the global model!!")