"""
Two classes for defining the ZeroMQ interface between the server and the model.
"""

import zmq
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ZMQInterfaceModel:
    def __init__(
        self,
        socket,
        register_worker_callback,
        unregister_worker_callback,
        return_global_model_callback,
        is_global_model_most_recent,
        receive_worker_update_callback,
        server_subprocess_args,
    ) -> None:
        self.socket = socket
        self.register_worker_callback = register_worker_callback
        self.unregister_worker_callback = unregister_worker_callback
        self.return_global_model_callback = return_global_model_callback
        self.is_global_model_most_recent = is_global_model_most_recent
        self.receive_worker_update_callback = receive_worker_update_callback
        self.server_subprocess_args = server_subprocess_args

    def receive(self):
        message = self.socket.recv_multipart()
        logger.debug(f"Zmq message received: {message[0]}")

        # Server initialisation data request
        if message[0] == b"server_args_request":
            self.socket.send_pyobj(self.server_subprocess_args)
        # Federated Learning API
        elif message[0] == b"register_worker":
            self.register_worker_callback(message[1].decode("utf-8"))
            self.socket.send(b"1")
        elif message[0] == b"unregister_worker":
            self.unregister_worker_callback(message[1].decode("utf-8"))
            self.socket.send(b"1")
        elif message[0] == b"return_global_model":
            global_model = self.return_global_model_callback()
            self.socket.send_pyobj(global_model)
        elif message[0] == b"is_global_model_most_recent":
            most_recent = self.is_global_model_most_recent(
                int(message[1].decode("utf-8"))
            )
            self.socket.send_pyobj(most_recent)
        elif message[0] == b"receive_worker_update":
            status = self.receive_worker_update_callback(
                message[1].decode("utf-8"), message[2]
            )
            self.socket.send_string(status)
        else:
            logger.error(
                f'ZQM messaging interface received unrecognised message type: "{message[0]}"'
            )


class ZMQInterfaceServer:
    def __init__(self, port) -> None:
        self.port = port

    def server_args_request_send(self):
        socket = self._send([b"server_args_request"])
        return socket.recv_pyobj()

    def register_worker_send(self, worker_id):
        socket = self._send([b"register_worker", worker_id.encode("utf-8")])
        return socket.recv()

    def unregister_worker_send(self, worker_id):
        socket = self._send([b"unregister_worker", worker_id.encode("utf-8")])
        return socket.recv()

    def return_global_model_send(self):
        socket = self._send([b"return_global_model"])
        return socket.recv_pyobj()

    def is_global_model_most_recent_send(self, model_version):
        socket = self._send(
            [b"is_global_model_most_recent", str(model_version).encode("utf-8")]
        )
        return socket.recv_pyobj()

    def receive_worker_update_send(self, worker_id, model_update):
        socket = self._send(
            [b"receive_worker_update", worker_id.encode("utf-8"), model_update]
        )
        return socket.recv_string()

    def _new_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{self.port}")
        return socket

    def _send(self, args):
        socket = self._new_socket()
        logger.debug(f"Sending zmq message: {args}")
        socket.send_multipart(args)
        logger.debug(f"Message sent.")
        return socket
