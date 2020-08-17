"""
Contains a single class illustrating the use of the classes in
dc_federated.backend.DCFServer.
"""

import io
from datetime import datetime
import logging

import torch

from dc_federated.examples.example_dcf_model.torch_nn_class import ExampleModelClass
from dc_federated.backend import DCFServer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ExampleGlobalModel(object):
    """
    This is a simple class that illustrates how the DCFServer class may be used to
    implement a federated global model. For testing purposes, it writes all the
    models it creates and receives to disk.
    """

    def __init__(self):
        self.worker_updates = {}
        self.global_model = ExampleModelClass()
        with open("egm_global_model.torch", 'wb') as f:
            torch.save(self.global_model, f)

        self.global_model_status = str(datetime(2018, 10, 10))

        self.server = DCFServer(
            self.register_worker,
            self.unregister_worker,
            self.return_global_model,
            self.return_global_model_status,
            self.receive_worker_update,
            key_list_file=None
        )

    def register_worker(self, worker_id):
        """
        Register the given worker_id by initializing its update to None.

        Parameters
        ----------

        worker_id: int
            The id of the new worker.
        """
        logger.info(f"Example Global Model: Registering worker {worker_id}")
        self.worker_updates[worker_id] = None

    def unregister_worker(self, worker_id):
        """
        Unregister the given worker_id by removing it from updates.

        Parameters
        ----------

        worker_id: int
            The id of the worker to be removed.
        """
        logger.info(f"Example Global Model: Unregistering worker {worker_id}")
        self.worker_updates.pop(worker_id)

    def return_global_model(self):
        """
        Serializes the current global torch model and sends it back to the worker.

        Returns
        ----------

        byte-stream:
            The current global torch model.
        """
        logger.info(f"Example Global Model: returning global model")
        model_data = io.BytesIO()
        torch.save(self.global_model, model_data)
        return model_data.getvalue()

    def return_global_model_status(self):
        """
        Returns a default model update time of 2018/10/10.

        Returns
        ----------

        str:
            String format of the last model update time.
        """
        logger.info(f"Example Global Model: returning global model status")
        return self.global_model_status

    def receive_worker_update(self, worker_id, model_update):
        """
        Given an update for a worker, adds the the update to the list of updates.

        Returns
        ----------

        str:
            String format of the last model update time.
        """
        if worker_id in self.worker_updates:
            self.worker_updates[worker_id] = \
                torch.load(io.BytesIO(model_update))
            logger.info(f"Model update received from worker {worker_id}")
            logger.info(self.worker_updates[worker_id])
            with open(f"egm_worker_update_{worker_id}.torch", 'wb') as f:
                torch.save(self.worker_updates[worker_id], f)
            self.global_model_status = str(
                datetime.now().isoformat(' ', 'seconds'))
            return f"Update received for worker {worker_id}"
        else:
            return f"Unregistered worker {worker_id} tried to send an update!!"

    def start(self):
        self.server.start_server()
        self.server.start_admin_server()
