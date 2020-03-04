"""
Contains utility functions to be used within the demo.
"""

import socket


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
