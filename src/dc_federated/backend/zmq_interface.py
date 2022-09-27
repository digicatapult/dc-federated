"""
Two classes for defining the ZeroMQ interface between the server and the model.
"""

import zmq
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


# class ZMQInterfaceModel(object):
#     """
#     The model-side ZMQ interface definition for communicating with the web
#     server process. It handles ZMQ and the serialisation and exposes the same
#     API that federated algorithm expects for communicating with the DCFServer.
#     The API is split into the callback functions defined in DCFServer plus the
#     non-functional remaining server_subprocess_args.
#     """
#     def __init__(
#         self,
#         socket,
#         register_worker_callback,
#         unregister_worker_callback,
#         return_global_model_callback,
#         is_global_model_most_recent,
#         receive_worker_update_callback,
#         server_subprocess_args,
#     ) -> None:
#         self.socket = socket
#         self.register_worker_callback = register_worker_callback
#         self.unregister_worker_callback = unregister_worker_callback
#         self.return_global_model_callback = return_global_model_callback
#         self.is_global_model_most_recent = is_global_model_most_recent
#         self.receive_worker_update_callback = receive_worker_update_callback
#         self.server_subprocess_args = server_subprocess_args



#     def receive(self):
#         """
#         Wait for a multipart message on the socket. Based on the first 'part'
#         (a string key) which specifies the function the arguments are sent to
#         one of the callbacks and returns sent back. The exception is 
#         "server_args_request" where server_subprocess_args is returned.
#         """
#         message = self.socket.recv_multipart()
#         logger.debug(f"Zmq message received: {message[0]}")

#         # Server initialisation data request
#         if message[0] == b"server_args_request":
#             self.socket.send_pyobj(self.server_subprocess_args)
#         # Federated Learning API
#         elif message[0] == b"register_worker":
#             self.register_worker_callback(message[1].decode("utf-8"))
#             self.socket.send(b"1")
#         elif message[0] == b"unregister_worker":
#             self.unregister_worker_callback(message[1].decode("utf-8"))
#             self.socket.send(b"1")
#         elif message[0] == b"return_global_model":
#             global_model = self.return_global_model_callback()
#             self.socket.send_pyobj(global_model)
#         elif message[0] == b"is_global_model_most_recent":
#             most_recent = self.is_global_model_most_recent(
#                 int(message[1].decode("utf-8"))
#             )
#             self.socket.send_pyobj(most_recent)
#         elif message[0] == b"receive_worker_update":
#             status = self.receive_worker_update_callback(
#                 message[1].decode("utf-8"), message[2]
#             )
#             self.socket.send_string(status)
#         else:
#             logger.error(
#                 f'ZQM messaging interface received unrecognised message type: "{message[0]}"'
#             )


class ZMQInterfaceServer(object):
    """
    The server-side ZMQ interface definition for communicating with the model
    aggregation process. It handles ZMQ and the serialisation and exposes the same
    API that DCFServer expects for communicating with the federated algorithm.
    The API is a list of callbacks, the same as in DCFServer, with an additional
    callback for getting the non-functional arguments for running the server.
    """
    def __init__(self, port) -> None:
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://localhost:{self.port}")

    def server_args_request_send(self):
        """
        Send request for non-functional server args supplied to
        ZMQInterfaceModel.

        Returns
        -------

        dict:
            dict of non-functional args to be passed to DCFServer. keys:
            server_mode_safe, key_list_file, load_last_session_workers,
            path_to_keys_db, server_host_ip, server_port, ssl_enabled,
            ssl_keyfile, ssl_certfile, model_check_interval, debug
        """
        #  socket = self._send([b"server_args_request"])
        self.socket.send_multipart([b"server_args_request"])
        output = self.socket.recv_pyobj()
        return output

    def register_worker_send(self, worker_id):
        """
        Send request to register_worker_callback from ZMQInterfaceModel.

        Parameters
        ----------
        worker_id: str
            The id of the worker to be handled by the function.

        Returns
        -------

        str:
            '1' to confirm receipt.
        """
        self.socket.send_multipart([b"register_worker", worker_id.encode("utf-8")])
        output = self.socket.recv()
        return output

    def unregister_worker_send(self, worker_id):
        """
        Send request to unregister_worker_callback from ZMQInterfaceModel.

        Parameters
        ----------
        worker_id: str
            The id of the worker to be handled by the function.

        Returns
        -------

        str:
            '1' to confirm receipt.
        """
        self.socket.send_multipart([b"unregister_worker", worker_id.encode("utf-8")])
        output = self.socket.recv()
        return output

    def return_global_model_send(self):
        """
        Send request to return_global_model_callback from ZMQInterfaceModel.
        Returns what that function returns.
        """
        self.socket.send_multipart([b"return_global_model"])
        output = self.socket.recv_pyobj()
        return output

    def is_global_model_most_recent_send(self, model_version):
        """
        Send request to is_global_model_most_recent from ZMQInterfaceModel.
        Returns what that function returns. Which, in the case of FedAvgServer
        is a bool.

        Parameters
        ----------
        model_version: int
            The version of the model version to check.
        """
        self.socket.send_multipart(
            [b"is_global_model_most_recent", str(model_version).encode("utf-8")]
        )
        output = self.socket.recv_pyobj()
        return output

    def receive_worker_update_send(self, worker_id, model_update):
        """
        Send request to return_global_model_callback from ZMQInterfaceModel.
        Returns the string that function returns.

        Parameters
        ----------
        worker_id: str
            the worker id submitting the update.

        model_update: 
            serialised model.
        """
        self.socket.send_multipart(
            [b"receive_worker_update", worker_id.encode("utf-8"), model_update]
        )
        output = self.socket.recv_string()
        return output

    # def _new_socket(self):
    #     """
    #     Internal function for creating a new ZMQ socket.

    #     Returns
    #     ----------
    #     ZMQ Socket: 
    #         The created socket
    #     """
    #     context = zmq.Context()
    #     socket = context.socket(zmq.REQ)
    #     socket.connect(f"tcp://localhost:{self.port}")
    #     return socket

    # def _send(self, args):
    #     """
    #     Internal function for sending a multipart message on the zmq socket.

    #     Returns
    #     ----------
    #     ZMQ Socket: 
    #         The created socket
    #     """
    #     socket = self._new_socket()
    #     logger.debug(f"Sending zmq message: {args}")
    #     socket.send_multipart(args)
    #     logger.debug(f"Message sent.")
    #     return socket
