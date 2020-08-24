"""
Defines the core worker class for the federated learning.
Abstracts away the lower level worker logic from the federated
machine learning logic.
"""
import pickle
import time
from datetime import datetime
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder

import requests
from dc_federated.backend._constants import *
from dc_federated.backend.backend_utils import is_valid_model_dict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class DCFWorker(object):
    """
    This class implements a worker API for the DCFServer

    Parameters
    ----------

        server_host_ip: str
            The ip-address of the host of the server.

        server_port: int
            The port at which the serer should listen to

        global_model_version_changed_callback: dict -> ()
            The callback to run if server status has changed. The function
            is expected to take a dictionary with two entries:
            GLOBAL_MODEL: serialized version of the global model.
            GLOBAL_MODEL_VERSION: str giving the version of the current
            global model.

        worker_version_of_global_model: () -> object
            This function is expected to return the version of the last global
            model that the worker received.

        private_key_file: str
            Name of the private key to use to authenticate the worker to the server.
            No authentication is performed if a None is passed.  Name of the
            corresponding public key file is assumed to be key_file + '.pub'

        polling_wait_period: int
            The number of seconds to wait before polling the server
            for status information.
    """
    def __init__(
            self,
            server_host_ip,
            server_port,
            global_model_version_changed_callback,
            worker_version_of_global_model,
            private_key_file,
            polling_wait_period=1):
        self.server_host_ip = server_host_ip
        self.server_port = server_port
        self.global_model_status_changed_callback = global_model_version_changed_callback
        self.worker_version_global_model = worker_version_of_global_model
        self.private_key_file = private_key_file
        self.polling_wait_period = polling_wait_period

        self.server_loc = f"http://{self.server_host_ip}:{self.server_port}"
        self.worker_id = None

    def get_signed_phrase(self):
        """
        Returns the the authentication string signed using the private key of this
        worker.

        Returns
        -------
        str:
            The hex string corresponding to the signed string.
        """
        if self.private_key_file is None:
            logger.warning("Unable to sign message - no private key file provided.")
            return "No private key was provided when worker was started."
        with open(self.private_key_file, 'r') as f:
            hex_read = f.read().encode()
            private_key_read = SigningKey(hex_read, encoder=HexEncoder)
            return private_key_read.sign(WORKER_AUTHENTICATION_PHRASE).hex()

    def get_public_key_str(self):
        """
        Returns the the string version of the public key for the private key of
        the worker.

        Returns
        -------
        str:
            The hex string corresponding to the public key string.
        """
        if self.private_key_file is None:
            logger.warning("No public key file provided - server side authentication will not succeed.")
            return "No public key was provided when worker was started."

        with open(self.private_key_file+'.pub', 'r') as f:
            return f.read()

    def register_worker(self):
        """
        Returns a registration number for the worker from the server.
        Each object of this class is registered only once during its lifetime.

        Returns
        -------

        int:
            The worker id returned by the server.
        """
        if self.worker_id is None:
            data = {
                PUBLIC_KEY_STR: self.get_public_key_str(),
                SIGNED_PHRASE: self.get_signed_phrase()
            }
            self.worker_id = requests.post(
                f"{self.server_loc}/{REGISTER_WORKER_ROUTE}", json=data).content.decode('UTF-8')

            if self.worker_id == INVALID_WORKER:
                raise ValueError(
                    "Server returned {INVALID_WORKER} which means it was unable to authenticate this worker. "
                    "Please verify that the private key you started this worker with corresponds to the "
                    "public key shared with the server.")
        return self.worker_id

    def get_global_model(self):
        """
        Gets the binary string of the current global model from the server.

        Returns
        -------

        binary string:
            The current global model returned by the server.
        """
        data = {
            WORKER_ID_KEY: self.worker_id,
            LAST_WORKER_MODEL_VERSION: self.worker_version_global_model()
        }
        return pickle.loads(requests.post(f"{self.server_loc}/{RETURN_GLOBAL_MODEL_ROUTE}",
                         json=data).content)

    def send_model_update(self, model_update):
        """
        Sends the model update from the worker. Worker must register before sending
        a model update.

        Parameters
        ----------

        model_update: binary string
            The model update to send to the server.
        """
        data_dict = {
            WORKER_ID_KEY: self.worker_id,
            MODEL_UPDATE_KEY: model_update
        }
        return requests.post(
            f"{self.server_loc}/{RECEIVE_WORKER_UPDATE_ROUTE}",
            files={ID_AND_MODEL_KEY: pickle.dumps(data_dict)}
        ).content

    def run(self):
        """
        Runs the main worker loop - this calls the server_status_changed_callback if the server_status
        has changed.
        """
        try:
            while True:
                model_dict = self.get_global_model()
                if is_valid_model_dict(model_dict):
                    self.global_model_status_changed_callback(model_dict)
        except Exception as e:
            logger.warning(str(e))
            logger.info(f"Exiting DCFworker {self.worker_id} run loop.")