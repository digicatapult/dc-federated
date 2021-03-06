"""
Contains a single class illustrating the use of the classes in
dc_federated.backend.DCFServer.
"""

import io
from datetime import datetime
import logging

import torch

from dc_federated.examples.example_dcf_model.torch_nn_class import ExampleModelClass
from dc_federated.backend import DCFServer, create_model_dict, WID_LEN


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

        self.global_model_version = 0

        self.server = DCFServer(
            register_worker_callback=self.register_worker,
            unregister_worker_callback=self.unregister_worker,
            return_global_model_callback=self.return_global_model,
            is_global_model_most_recent=self.is_global_model_most_recent,
            receive_worker_update_callback=self.receive_worker_update,
            server_mode_safe=False,
            key_list_file=None,
            load_last_session_workers=False
        )

    def register_worker(self, worker_id):
        """
        Register the given worker_id by initializing its update to None.

        Parameters
        ----------

        worker_id: int
            The id of the new worker.
        """
        logger.info(f"Example Global Model: Registering worker {worker_id[0:WID_LEN]}")
        self.worker_updates[worker_id] = None

    def unregister_worker(self, worker_id):
        """
        Unregister the given worker_id by removing it from updates.

        Parameters
        ----------

        worker_id: int
            The id of the worker to be removed.
        """
        logger.info(f"Example Global Model: Unregistering worker {worker_id[0:WID_LEN]}")
        self.worker_updates.pop(worker_id)

    def return_global_model(self):
        """
        Serializes the current global torch model and sends it back to the worker.

        Returns
        ----------

        dict:
            The model dictionary as per the specification in DCFSever
        """
        logger.info(f"Example Global Model: returning global model")
        model_data = io.BytesIO()
        torch.save(self.global_model, model_data)
        return create_model_dict(model_data.getvalue(), self.global_model_version)

    def is_global_model_most_recent(self, model_version):
        """
        Returns a default model update time of 2018/10/10.

        Parameter
        ---------

        model_version: int

        Returns
        ----------

        str:
            String format of the last model update time.
        """
        logger.info(f"Example Global Model: checking if model version is most recent.")
        return self.global_model_version == model_version

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
            logger.info(f"Model update received from worker {worker_id[0:WID_LEN]}")
            logger.info(self.worker_updates[worker_id])
            with open(f"egm_worker_update_{worker_id}.torch", 'wb') as f:
                torch.save(self.worker_updates[worker_id], f)
            self.global_model_version += 1
            return f"Update received for worker {worker_id[0:WID_LEN]}"
        else:
            return f"Unregistered worker {worker_id[0:WID_LEN]} tried to send an update!!"

    def start(self):
        self.server.start_server()
