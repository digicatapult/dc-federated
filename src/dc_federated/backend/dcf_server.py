"""
Defines the core server class for the federated learning.
Abstracts away the lower level server logic from the federated
machine learning logic.
"""
import json
import logging
import pickle
import hashlib
import time

from nacl.signing import VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError
from bottle import Bottle, run, request, response, ServerAdapter

from dc_federated.backend._constants import *
from dc_federated.utils import get_host_ip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

        key_list_file: str
            The name of the file containing the public keys for valid workers.
            The public keys are given one key per line, with each key being
            generated by the worker_key_pair_tool.py tool. If None, then
            no authentication is performed.

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
        key_list_file,
        server_host_ip=None,
        server_port=8080,
        admin_server_port=8081,
            debug=False):

        self.server_host_ip = get_host_ip() if server_host_ip is None else server_host_ip
        self.server_port = server_port
        self.admin_server_port = admin_server_port

        self.register_worker_callback = register_worker_callback
        self.return_global_model_callback = return_global_model_callback
        self.query_global_model_status_callback = query_global_model_status_callback
        self.receive_worker_update_callback = receive_worker_update_callback
        self.worker_authenticator = WorkerAuthenticator(key_list_file)

        self.debug = debug

        self.worker_list = []
        self.last_worker = -1

    def register_worker(self):
        """
        Authenticates the worker

        Returns
        -------

        int:
            The id of the new client.
        """
        worker_data = request.json
        auth_success, auth_type = \
            self.worker_authenticator.authenticate_worker(worker_data[PUBLIC_KEY_STR],
                                                          worker_data[SIGNED_PHRASE])
        if auth_success:
            logger.info(
                f"Successfully registered worker with public key: {worker_data[PUBLIC_KEY_STR]}")
            if auth_type == NO_AUTHENTICATION:
                worker_id = hashlib.sha224(str(time.time()).encode(
                    'utf-8')).hexdigest() + '_unauthenticated'
            else:
                worker_id = worker_data[PUBLIC_KEY_STR]
            if worker_id not in self.worker_list:
                self.worker_list.append(worker_id)
        else:
            logger.info(
                f"Failed to register worker with public key: {worker_data[PUBLIC_KEY_STR]}")
            worker_id = INVALID_WORKER

        self.register_worker_callback(worker_id)
        return worker_id

    def admin_list_workers(self):
        """
        List all registered workers

        Returns
        -------

        [string]:
            The id of the workers
        """
        response.content_type = 'application/json'
        return json.dumps(self.worker_list)

    def admin_register_worker(self):
        """
        TODO integrate that with pub key system
        """
        return self.register_worker()

    def admin_delete_worker(self, worker_id):
        """
        TODO integrate that with pub key system
        """
        logger.info(f"Not implemented yet: unregister worker {worker_id}")

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
            if data_dict[WORKER_ID_KEY] in self.worker_list:
                return_value = self.receive_worker_update_callback(
                    data_dict[WORKER_ID_KEY],
                    data_dict[MODEL_UPDATE_KEY]
                )
                return return_value
            else:
                logger.warning(
                    f"Unregistered worker {data_dict[WORKER_ID_KEY]} tried to send an update.")
                return UNREGISTERED_WORKER
        except Exception as e:
            logger.warning(e)
            return str(e)

    def query_global_model_status(self):
        """
        Returns the status of the global model using the provided callback. If query is not
        from a valid worker it raises an error.

        Returns
        -------

        str:
            If the update was successful then "Worker update received"
            Otherwise any exception that was raised.
        """
        try:
            query_request = request.json
            if query_request[WORKER_ID_KEY] in self.worker_list:
                return self.query_global_model_status_callback()
            else:
                return UNREGISTERED_WORKER
        except Exception as e:
            logger.warning(e)
            return str(e)

    def return_global_model(self):
        """
        Returns the global model by using the provided callback. If query is not from a valid
        worker it raises an error.

        Returns
        -------

        str:
            If the update was successful then "Worker update received"
            Otherwise any exception that was raised.
        """
        try:
            query_request = request.json
            if query_request[WORKER_ID_KEY] in self.worker_list:
                return self.return_global_model_callback()
            else:
                return UNREGISTERED_WORKER
        except Exception as e:
            logger.warning(e)
            return str(e)

    @staticmethod
    def enable_cors():
        """
        Enable the cross origin resource for the server.
        """
        response.add_header('Access-Control-Allow-Origin', '*')
        response.add_header('Access-Control-Allow-Methods',
                            'GET, POST, PUT, OPTIONS')
        response.add_header('Access-Control-Allow-Headers',
                            'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token')

    def start_server(self, server_adapter=None):
        """
        Sets up all the routes for the server and starts it.

        server_backend: bottle.ServerAdapter (default None)
            The server adapter to use. The default bottle.WSGIRefServer is used if none is given.
            WARNING: If given, this will over-ride the host-ip and port passed as parameters to this
            object.
        """
        application = Bottle()
        application.route(f"/{REGISTER_WORKER_ROUTE}",
                          method='POST', callback=self.register_worker)
        application.route(f"/{RETURN_GLOBAL_MODEL_ROUTE}",
                          method='POST', callback=self.return_global_model)
        application.route(f"/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}",
                          method='POST', callback=self.query_global_model_status)
        application.route(f"/{RECEIVE_WORKER_UPDATE_ROUTE}",
                          method='POST', callback=self.receive_worker_update)
        application.add_hook('after_request', self.enable_cors)

        if server_adapter is not None and isinstance(server_adapter, ServerAdapter):
            self.server_host_ip = server_adapter.host
            self.server_port = server_adapter.port
            run(application, server=server_adapter, debug=self.debug, quiet=True)
        else:
            run(application, host=self.server_host_ip,
                port=self.server_port, debug=self.debug, quiet=True)

    def start_admin_server(self, server_adapter=None):
        """
        Sets all the admin routes and starts the admin server
        """
        admin_app = Bottle()
        admin_app.get("/workers", callback=self.admin_list_workers)
        admin_app.post("/workers", callback=self.admin_register_worker)
        admin_app.delete("/workers/<worker_id>",
                         callback=self.admin_delete_worker)

        if server_adapter is not None and isinstance(server_adapter, ServerAdapter):
            self.admin_server_host_ip = server_adapter.host
            self.admin_server_port = server_adapter.port
            run(admin_app, server=server_adapter, debug=self.debug, quiet=True)
        else:
            run(admin_app, host='127.0.0.1',
                port=self.admin_server_port, debug=self.debug, quiet=True)


class WorkerAuthenticator(object):
    """
    Helper class for authenticating workers.

    Parameters
    ----------

    key_list_file: str
        The name of the file containing the public keys for valid workers.
        The file is a just list of the public keys, each generated by the
        worker_key_pair_tool tool. All workers are accepted if no workers
        are provided.
    """

    def __init__(self, key_list_file):
        if key_list_file is None:
            logger.warning(f"No key list file provided - "
                           f"no worker authentication will be used!!!.")
            logger.warning(f"Server is running in ****UNSAFE MODE.****")
            self.authenticate = False
            return

        with open(key_list_file, 'r') as f:
            keys = f.read().splitlines()

        # dict for efficient fetching of the public key
        self.authenticate = True
        self.keys = {key: VerifyKey(
            key.encode(), encoder=HexEncoder) for key in keys}

    def authenticate_worker(self, public_key_str, signed_message):
        """
        Authenticates a worker with the given public key against the
        given signed message.

        Parameters
        ----------

        public_key_str: str
            UFT-8 encoded version of the public key

        signed_message: str
            UTF-8 encoded signed message

        Returns
        -------

        bool:
            True if the public key matches the singed messge
            False otherwise
        """
        if not self.authenticate:
            logger.warning("Accepting worker as valid without authentication.")
            logger.warning(
                "Server was likely started without a list of valid public keys from workers.")
            return True, NO_AUTHENTICATION
        try:
            if public_key_str not in self.keys:
                return False, AUTHENTICATED
            self.keys[public_key_str].verify(
                signed_message.encode(), encoder=HexEncoder)
        except BadSignatureError:
            logger.warning(
                f"Failed to authenticate worker with public key: {public_key_str}.")
            return False, AUTHENTICATED
        else:
            logger.info(
                f"Successfully authenticated worker with public key: {public_key_str}.")
            return True, AUTHENTICATED
