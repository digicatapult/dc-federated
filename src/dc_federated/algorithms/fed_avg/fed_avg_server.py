"""
Contains the implementation of the server side logic for the FedAvg algorithm.
"""

import msgpack
import io
from datetime import datetime
from collections import OrderedDict

import torch
from dc_federated.backend import DCFServerHandler as DCFServer, \
    GLOBAL_MODEL_VERSION, GLOBAL_MODEL

from dc_federated.backend._constants import *
from dc_federated.algorithms.fed_avg.fed_avg_model_trainer import FedAvgModelTrainer

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class FedAvgServer(object):
    """
    This class implements the server-side of the FedAvg algorithm using the
    dc_federated.backend package.

    Parameters
    ----------

    global_model_trainer: FedAvgModelTrainer
        The name of the python model-class for this problem.

    update_lim: int
        Number of unique updates that needs to be received before the last
        global update before we update the global model.

    key_list_file: str
        The list of public keys of valid workers. No authentication is performed
        if file not given.

    server_host_ip: str (default None)
        The hostname or IP address the server will bind to.
        If not given, it will default to the machine IP.

    server_port: int (default 8080)
        The port at which the server should listen to.

    ssl_enabled: bool (default False)
        Enable SSL/TLS for server/workers communications.

    ssl_keyfile: str
        Must be a valid path to the key file.
        This is mandatory if ssl_enabled is True, ignored otherwise.

    ssl_certfile: str
        Must be a valid path to the certificate.
        This is mandatory if ssl_enabled is True, ignored otherwise.
    """

    def __init__(self,
                 global_model_trainer,
                 key_list_file,
                 update_lim=10,
                 server_host_ip=None,
                 server_port=8080,
                 ssl_enabled=False,
                 ssl_keyfile=None,
                 ssl_certfile=None):
        logger.info(
            f"Initializing FedAvg server for model class {global_model_trainer.get_model().__class__.__name__}")

        self.worker_updates = {}
        self.global_model_trainer = global_model_trainer
        self.update_lim = update_lim

        self.last_global_model_update_timestamp = datetime(1980, 10, 10)
        self.server = DCFServer(
            register_worker_callback=self.register_worker,
            unregister_worker_callback=self.unregister_worker,
            return_global_model_callback=self.return_global_model,
            is_global_model_most_recent=self.is_global_model_most_recent,
            receive_worker_update_callback=self.receive_worker_update,
            server_mode_safe=key_list_file is not None,
            load_last_session_workers=False,
            key_list_file=key_list_file,
            server_host_ip=server_host_ip,
            server_port=server_port,
            ssl_enabled=ssl_enabled,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            model_check_interval = 1
        )

        self.unique_updates_since_last_agg = 0
        self.iteration = 0
        self.model_version = 0

    def register_worker(self, worker_id):
        """
        Register the given worker_id by initializing its update to None.

        Parameters
        ----------

        worker_id: int
            The id of the new worker.
        """
        logger.info(f"Registered worker {worker_id[0:WID_LEN]}")
        self.worker_updates[worker_id] = None

    def unregister_worker(self, worker_id):
        """
        Unregister the given worker_id by removing it from updates.

        Parameters
        ----------

        worker_id: int
            The id of the worker to be removed.
        """
        logger.info(f"Unregistered worker {worker_id[0:WID_LEN]}")
        self.worker_updates.pop(worker_id)

    def return_global_model(self):
        """
        Serializes the current global torch model, puts it in the proper
        dictionary, and sends it back.

        Returns
        ----------

        dict:
            A dictionary with keys:
            GLOBAL_MODEL: serialized global model.
            GLOBAL_MODEL_VERSION: version of the global model
        """
        model_data = io.BytesIO()
        torch.save(self.global_model_trainer.get_model(), model_data)

        return {
            GLOBAL_MODEL: model_data.getvalue(),
            GLOBAL_MODEL_VERSION: self.model_version
        }

    def is_global_model_most_recent(self, model_version):
        """
        Returns a default model update time of 2018/10/10.

        Parameters
        ----------

        model_version: int
            The version of most recent global model that the
            worker has.

        Returns
        ----------

        str:
            String format of the last model update time.
        """
        return self.model_version == model_version

    def receive_worker_update(self, worker_id, model_update):
        """
        Given an update for a worker, adds its update to the dictionary of updates.
        It also agg_model() to update the global model if necessary.

        Returns
        ----------

        str:
            String format of the last model update time.
        """
        if worker_id in self.worker_updates:
            # update the number of unique updates received
            if self.worker_updates[worker_id] is None or \
                    self.worker_updates[worker_id][0] < self.last_global_model_update_timestamp:
                self.unique_updates_since_last_agg += 1
            update_size, model_bytes = msgpack.unpackb(model_update)
            self.worker_updates[worker_id] = (
                datetime.now(),
                update_size,
                torch.load(io.BytesIO(model_bytes))
            )
            logger.info(f"Model update from worker {worker_id[0:WID_LEN]} accepted.")
            if self.agg_model():
                self.global_model_trainer.test()
            return f"Update received for worker {worker_id[0:WID_LEN]}"
        else:
            logger.warning(
                f"Unregistered worker {worker_id[0:WID_LEN]} tried to send an update.")
            return f"Please register before sending an update."

    def agg_model(self):
        """
        Updates the global model by aggregating all the most recent updates
        from the workers, assuming that the number of unique updates received
        since the last global model update is above the threshold.
        """
        if self.unique_updates_since_last_agg < self.update_lim:
            return False

        logger.info("Updating the global model.\n")

        def agg_params(key, state_dicts, update_sizes):
            agg_val = state_dicts[0][key] * update_sizes[0]
            for sd, sz in zip(state_dicts[1:], update_sizes[1:]):
                agg_val = agg_val + sd[key] * sz
            agg_val = agg_val / sum(update_sizes)
            return torch.tensor(agg_val.cpu().clone().numpy())

        # gather the model-updates to use for the update
        state_dicts_to_update_with = []
        update_sizes = []
        # each item in the worker_updates dictionary contains a
        # (timestamp update, update-size, model)
        for wi in self.worker_updates:
            if self.worker_updates[wi][0] > self.last_global_model_update_timestamp:
                state_dicts_to_update_with.append(
                    self.worker_updates[wi][2].state_dict())
                update_sizes.append(self.worker_updates[wi][1])

        # now update the global model
        global_model_dict = OrderedDict()
        for key in state_dicts_to_update_with[0].keys():
            global_model_dict[key] = agg_params(
                key, state_dicts_to_update_with, update_sizes)

        self.global_model_trainer.load_model_from_state_dict(global_model_dict)

        self.last_global_model_update_timestamp = datetime.now()
        self.unique_updates_since_last_agg = 0
        self.iteration += 1
        self.model_version += 1

        return True

    def start(self):
        self.server.start_server()
