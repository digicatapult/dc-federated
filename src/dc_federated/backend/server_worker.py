"""
Defines the core server and worker classes for the federated learning.
Abstracts away the lower level server/worker logic from the federated
machine learning logic.
"""
import time
import requests
import pickle
import logging

import bottle
from bottle import Bottle, run, request
from dc_federated.backend._constants import *

from dc_federated.utils import get_host_ip


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dc_federated.backend.server_worker')
logger.setLevel(level=logging.INFO)


class DCFServer(object):
    """
    This class abstracts away the lower level communication logic for
    the central server/node from the actual federated learning logic.
    It interacts with the central server node via the 4 callback functions
    passed in the constructor. For an example usage please refer to the
    package dc_federated.example_dcf+model.

    Parameters
    ----------

        register_worker_callback:
            This function is expected to take the id of a newly registered
            worker and should contain the application specific logic for
            dealing with a new worker joining the federated learning pool.

        return_global_model_callback: () -> bit-string
            This function is expected to return the current global model
            in some application dependent binary serialized form.

    
        query_global_model_status_callback:  () -> str
            This function is expected to return a string giving the
            application dependent current status of the global model.
        
        receive_worker_update_callback: dict -> bool
            This function should receive a worker-id and an application
            dependent binary serialized update from the worker. The
            server code ensures that the worker-id was previously
            registered.

        server_host_ip: str (default None)
            The ip-address of the host of the server. If None, then it
            uses the ip-address of the current machine.

        server_port: int (default 8080)
            The port at which the serer should listen to. If None, then it
            uses the port 8080.


    """
    def __init__(
        self, 
        register_worker_callback,
        return_global_model_callback,
        query_global_model_status_callback,
        receive_worker_update_callback,
        server_host_ip=None,
        server_port=8080,
            debug=False):

        self.server_host_ip = get_host_ip() if server_host_ip is None else server_host_ip
        self.server_port = server_port

        self.register_worker_callback = register_worker_callback
        self.return_global_model_callback = return_global_model_callback
        self.query_global_model_status_callback = query_global_model_status_callback
        self.receive_worker_update_callback = receive_worker_update_callback

        self.debug = debug

        self.worker_list = []
        self.last_worker = -1

    def register_worker(self):
        """
        Creates a new worker-id, adds it to the internal list, calls the callback function 
        for the associated server model, and returns the id to the client.
        
        Returns
        -------

        int:
            The id of the new client.
        """
        self.last_worker += 1
        self.worker_list.append(self.last_worker)
        self.register_worker_callback(self.last_worker)
        return str(self.last_worker)

    def receive_worker_update(self):
        """
        This receives the update from a worker and calls the corresponding callback function.
        Expects that the worker_id and model-update were sent using the DCFWorker.send_model_update()

        Returns
        -------

        str:
            If the update was successful then "Worker update received"
            Otherwise any exception that was raised.
        """
        try:
            data_dict = pickle.load(request.files[ID_AND_MODEL_KEY].file)
            return_value = self.receive_worker_update_callback(
                data_dict[WORKER_ID_KEY],
                data_dict[MODEL_UPDATE_KEY]
            )
            return return_value
        except Exception as e:
            logger.warning(e)
            return str(e)

    @staticmethod
    def enable_cors():
        """
        Enable the cross origin resource for the server.
        """
        bottle.response.add_header('Access-Control-Allow-Origin', '*')
        bottle.response.add_header('Access-Control-Allow-Methods',
                                   'GET, POST, PUT, OPTIONS')
        bottle.response.add_header('Access-Control-Allow-Headers',
                                   'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token')

    def start_server(self):
        """
        Sets up all the routes for the server and starts it.
        """
        application = Bottle()
        application.route(f"/{REGISTER_WORKER_ROUTE}", method='GET', callback=self.register_worker)
        application.route(f"/{RETURN_GLOBAL_MODEL_ROUTE}", method='GET', callback=self.return_global_model_callback)
        application.route(f"/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}", method='GET', callback=self.query_global_model_status_callback)
        application.route(f"/{RECEIVE_WORKER_UPDATE_ROUTE}", method='POST', callback=self.receive_worker_update)
        application.add_hook('after_request', self.enable_cors)
        
        run(application, host=self.server_host_ip, port=self.server_port, debug=self.debug)


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

        polling_wait_period: int
            The number of seconds to wait before polling the server
            for status information.
    """
    def __init__(
            self,
            server_host_ip,
            server_port,
            global_model_status_changed_callback,
            polling_wait_period=1):
        self.server_host_ip = server_host_ip
        self.server_port = server_port
        self.global_model_status_changed_callback = global_model_status_changed_callback
        self.polling_wait_period = polling_wait_period

        self.server_loc = f"http://{self.server_host_ip}:{self.server_port}"
        self.current_global_model_status = None
        self.worker_id = None

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
            self.worker_id = int(requests.get(f"{self.server_loc}/{REGISTER_WORKER_ROUTE}").content)
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
