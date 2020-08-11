"""
Defines the core worker class for the federated learning.
Abstracts away the lower level worker logic from the federated
machine learning logic.
"""
import pickle
import time
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder

import requests
from dc_federated.backend._constants import *

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

        global_model_satus_changed_callback: function
            The callback to run if server status has changed.

        private_key_file: str
            Name of the private key to use to authenticate the worker to the server.
            If given a file with the text _constants.ALL_WORKERS_ALLOWED, then no
            authentication is performed.

        polling_wait_period: int
            The number of seconds to wait before polling the server
            for status information.
    """
    def __init__(
            self,
            server_host_ip,
            server_port,
            global_model_status_changed_callback,
            private_key_file=None,
            polling_wait_period=1):
        self.server_host_ip = server_host_ip
        self.server_port = server_port
        self.global_model_status_changed_callback = global_model_status_changed_callback
        self.private_key_file = private_key_file
        self.polling_wait_period = polling_wait_period

        self.server_loc = f"http://{self.server_host_ip}:{self.server_port}"
        self.current_global_model_status = None
        self.worker_id = None

    def get_signed_phrase(self):
        """
        Returns the the authentication string singed using the private key of this
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
        Returns the the string version of the public key for the given private key.
        worker.

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
        Each object of this class is registered only once.

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
                f"{self.server_loc}/{REGISTER_WORKER_ROUTE}",
                json=data).content.decode('UTF-8')

            if self.worker_id == INVALID_WORKER:
                raise ValueError(
                    "Server returned {INVALID_WORKER} which means it was unable to authenticate this worker. "
                    "Please ensure that the private key you started this worker corresponds to the "
                    "public key shared with the server.")
        self.current_global_model_status = self.get_global_model_status()
        return self.worker_id

    def get_global_model(self):
        """
        Gets the binary string of the current global model from the server.

        Returns
        -------

        binary string:
            The current global model returned by the server.
        """
        return requests.get(f"{self.server_loc}/{RETURN_GLOBAL_MODEL_ROUTE}").content

    def get_global_model_status(self):
        """
        Returns the status of the current global model from the server.

        Returns
        -------

        str:
            The status of the current global model.
        """
        return requests.get(f"{self.server_loc}/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}").content.decode('UTF-8')

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
                time.sleep(self.polling_wait_period)
                status = self.get_global_model_status()
                if self.current_global_model_status != status:
                    self.current_global_model_status = status
                    self.global_model_status_changed_callback()
        except Exception as e:
            logger.warning(str(e))
            logger.info(f"Exiting DCFworker {self.worker_id} run loop.")