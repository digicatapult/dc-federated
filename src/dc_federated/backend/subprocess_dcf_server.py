"""
A script for running the DCFServer without a federated learning algorithm.
Instead the callbacks are implemented as ZeroMQ messages using the 
ZMQ Interface defined in dc_federated.backend.zmq_interface. This script is
intended to be run from the DCFServerHandler defined in dc_federated.backend.dcf_server.
"""

from dc_federated.backend.dcf_server import DCFServer
from dc_federated.backend.zmq_interface import ZMQInterfaceServer

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def run(port):
    logger.info("Starting server as a subprocess.")

    print(ZMQInterfaceServer)
    zmqi = ZMQInterfaceServer(port)
    server_subprocess_args = zmqi.server_args_request_send()

    server = DCFServer(
        register_worker_callback=zmqi.register_worker_send,
        unregister_worker_callback=zmqi.unregister_worker_send,
        return_global_model_callback=zmqi.return_global_model_send,
        is_global_model_most_recent=zmqi.is_global_model_most_recent_send,
        receive_worker_update_callback=zmqi.receive_worker_update_send,
        **server_subprocess_args,
    )
    server.start_server()


if __name__ == "__main__":
    run(port=sys.argv[1])
