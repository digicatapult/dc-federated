import io

import requests
import json

import bottle
from bottle import Bottle, run, request, static_file


REGISTER_WORKER_ROUTE = 'register_worker'  
RETURN_GLOBAL_MODEL_ROUTE = 'return_global_model'
QUERY_GLOBAL_MODEL_STATUS_ROUTE = 'query_global_model_status'
RECEIVE_WORKER_UPDATE_ROUTE = 'receive_worker_update'


class DCFServer(object):
    """
    This class starts a server for federated learning.
    
    Parameters
    ----------

        server_host_ip: str
            The ip-address of the host of the server.
    
        server_port: int
            The port at which the serer should listen to
    
        register_worker_callback: () -> int
            Callback for registering a client with the server.

        return_global_model_callback: () -> bit-string
            The call back function for returning the current global model.
    
        query_global_model_status_callback:  () -> str
            The last time the global model was updated
        
        receive_worker_update_callback: dict -> bool
            The call-back for receiving an update from a client.
            The client should issue a POST using a dict containing the unique client id and 
            the model id. 
    """
    def __init__(
        self, 
        server_host_ip,
        server_port,
        register_worker_callback,
        return_global_model_callback,
        query_global_model_status_callback,
        receive_worker_update_callback,
        debug=False):

        self.server_host_ip = server_host_ip
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
        for the asscociated server model, and returns the id to the client.
        
        Returns
        -------

        int:
            The id of the new client.
        """
        self.last_worker += 1
        self.worker_list.append(self.last_worker)
        self.register_worker_callback(self.last_worker)
        return self.last_worker

    def receive_worker_update(self):
        """
        This receives the update from a worker and calls the corresponding callback function.             
        """
        return json.load(io.BytesIO(self.receive_worker_update_callback(request.body)))

    def enable_cors(self):
        """
        Enable the cross origin resource for the server.
        """
        print("Enabling CORS")
        bottle.response.add_header('Access-Control-Allow-Origin', '*')
        bottle.response.add_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        bottle.response.add_header('Access-Control-Allow-Headers',
                                    'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token')

    def start_server(self):

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
    """
    def __init__(self, server_host_ip, server_port):
        self.server_host_ip = server_host_ip
        self.server_port = server_port
        self.server_loc = f"http://{self.server_host_ip}:{self.server_port}"
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
            self.worker_id = int(requests.get(f"{self.server_loc}/{REGISTER_WORKER_ROUTE}"))
        return self.worker_id
        
    def get_global_model(self):
        """
        Gets the binary string of the current global model from the server.
        
        Returns
        -------

        binary string:
            The current global model returned by the server.
        """
        return requests.get(f"{self.server_loc}/{RETURN_GLOBAL_MODEL_ROUTE}")

    def get_global_model_status(self):
        """
        Returns the status of the current global model from the server.
        
        Returns
        -------

        str:
            The status of the current global model.
        """
        return requests.get(f"{self.server_loc}/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}")

    def send_model_update(self, model_update):
        """
        Sends the model update from the worker. Worker must register before sending 
        a model update.
        
        Parameters
        ----------

        model_update: binary string
            The model update to send to the server.
        """

        return requests.post(
            f"{self.server_loc}/{RECEIVE_WORKER_UPDATE_ROUTE}",
            data={
                "worker_id": self.worker_id,
                "model_update": model_update}
            ) 
