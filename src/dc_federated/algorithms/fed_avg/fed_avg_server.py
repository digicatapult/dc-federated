"""
Contains the implementation of the server side logic for the FedAvg algorithm.
"""

import pickle

import io
from datetime import datetime
import logging
from collections import OrderedDict

import torch
from dc_federated.backend import DCFServer
from dc_federated.algorithms.fed_avg.fed_avg_model_trainer import FedAvgModelTrainer


logging.basicConfig(level=logging.INFO)
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
    """
    def __init__(self, global_model_trainer, key_list_file, update_lim=10, ):
        logger.info(f"Initializing FedAvg server for model class {global_model_trainer.get_model().__class__.__name__}")

        self.worker_updates = {}
        self.global_model_trainer = global_model_trainer
        self.update_lim = update_lim

        self.last_global_model_update_timestamp = datetime(1980, 10, 10)
        self.server = DCFServer(
            self.register_worker,
            self.return_global_model,
            self.return_global_model_status,
            self.receive_worker_update,
            key_list_file=key_list_file
        )

        self.unique_updates_since_last_agg = 0
        self.iteration = 0

    def register_worker(self, worker_id):
        """
        Register the given worker_id by initializing its update to None.

        Parameters
        ----------

        worker_id: int
            The id of the new worker.
        """
        logger.info(f"Registered worker {worker_id}")
        self.worker_updates[worker_id] = None

    def return_global_model(self):
        """
        Serializes the current global torch model and sends it back to the worker.

        Returns
        ----------

        byte-stream:
            The current global torch model.
        """
        model_data = io.BytesIO()
        torch.save(self.global_model_trainer.get_model(), model_data)
        return model_data.getvalue()

    def return_global_model_status(self):
        """
        Returns a default model update time of 2018/10/10.

        Returns
        ----------

        str:
            String format of the last model update time.
        """
        return str(self.last_global_model_update_timestamp.isoformat(' ', 'seconds'))

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
            update_size, model_bytes = pickle.loads(model_update)
            self.worker_updates[worker_id] = (datetime.now(), update_size,
                                              torch.load(io.BytesIO(model_bytes)))
            logger.info(f" Model update received from worker {worker_id}")
            if self.agg_model():
                self.global_model_trainer.test()
            return f"Update received for worker {worker_id}"
        else:
            logger.warning(f" Unregistered worker {worker_id} tried to send an update.")
            return f"Please register before sending an update."

    def agg_model(self):
        """
        Updates the global model by aggregating all the most recent updates
        from the workers, assuming that the number of unique updates received
        since the last global model update is above the threshold.
        """
        if self.unique_updates_since_last_agg < self.update_lim:
            return False

        logger.info(" Updating the global model.\n")

        def agg_params(key, state_dicts, update_sizes):
            agg_val = state_dicts[0][key] * update_sizes[0]
            for sd, sz  in zip(state_dicts[1:], update_sizes[1:]):
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
                state_dicts_to_update_with.append(self.worker_updates[wi][2].state_dict())
                update_sizes.append(self.worker_updates[wi][1])

        # now update the global model
        global_model_dict = OrderedDict()
        for key in state_dicts_to_update_with[0].keys():
            global_model_dict[key] = agg_params(key, state_dicts_to_update_with, update_sizes)

        self.global_model_trainer.load_model_from_state_dict(global_model_dict)

        self.last_global_model_update_timestamp = datetime.now()
        self.unique_updates_since_last_agg = 0
        self.iteration += 1

        return True

    def start(self):
        self.server.start_server()
