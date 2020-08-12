"""
Contains utility functions to be used within the demo.
"""
import socket
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from wsgiref.simple_server import make_server

from bottle import WSGIRefServer

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


class StoppableServer(WSGIRefServer):
    """
    A simple server that can be stopped when it is run from a different thread.
    Meant to be used for testing.
    """
    def run(self, app): # pragma: no cover
        """
        Copied almost exactly from WSGIRefServer.run()
        """
        class FixedHandler(WSGIRequestHandler):
            def address_string(self): # Prevent reverse DNS lookups please.
                return self.client_address[0]
            def log_request(*args, **kw):
                if not self.quiet:
                    return WSGIRequestHandler.log_request(*args, **kw)

        handler_cls = self.options.get('handler_class', FixedHandler)
        server_cls  = self.options.get('server_class', WSGIServer)

        if ':' in self.host: # Fix wsgiref for IPv6 addresses.
            if getattr(server_cls, 'address_family') == socket.AF_INET:
                class server_cls(server_cls):
                    address_family = socket.AF_INET6

        # The following two lines are changed.
        self.server = make_server(self.host, self.port, app, server_cls, handler_cls)
        self.server.serve_forever()

    def shutdown(self):
        logger.info("Shutting down server.")
        self.server.shutdown()
