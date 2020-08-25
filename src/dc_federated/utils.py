"""
Contains utility functions to be used within the demo.
"""


import socket

from bottle import Bottle, ServerAdapter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_host_ip():
    """
    Simple utility function to return the ipv4 address of the
    current machine.

    Returns
    -------

    str:
        The ipv4 address of the current machine.
    """
    return socket.gethostbyname(socket.gethostname())


class StoppableServer(ServerAdapter):
    """
    A simple server that can be stopped when it is run from a different thread.
    Meant to be used for testing.
    """

    server = None

    def run(self, handler):
        from wsgiref.simple_server import make_server, WSGIRequestHandler
        if self.quiet:
            class QuietHandler(WSGIRequestHandler):
                def log_request(*args, **kw): pass
            self.options['handler_class'] = QuietHandler
        self.server = make_server(
            self.host, self.port, handler, **self.options)
        self.server.serve_forever()

    def shutdown(self):
        logger.info("Shutting down server.")
        self.server.shutdown()
