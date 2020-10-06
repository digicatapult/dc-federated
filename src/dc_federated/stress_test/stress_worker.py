"""
Run the workers for the basic stress test.
"""

import os
import sys
import argparse
import msgpack
from datetime import datetime

from gevent import Greenlet, sleep

from dc_federated.backend import DCFWorker
from dc_federated.backend._constants import *
from dc_federated.stress_test.stress_gen_keys import STRESS_KEYS_FOLDER, STRESS_WORKER_PREFIX


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


class SimpleLPWorker(object):
    """
    Simple worker class for the stress testing. This class was created because
    for the test we need to maintain the state gm_version on the different
    greenlets (whicha are pseduo-threads).

    Parameters
    ----------

    s_host: str
        The server host

    s_port: int
        the server port

    private_key_file: str
        The file containing the private key for this worker.
    """
    def __init__(self, s_host, s_port, private_key_file):
        self.gm_version = 0
        self.update = None
        self.worker = DCFWorker(
            server_protocol='http',
            server_host_ip=s_host,
            server_port=s_port,
            global_model_version_changed_callback=self.global_model_changed_callback,
            get_worker_version_of_global_model=self.get_last_global_model_version,
            private_key_file=private_key_file
        )

    def global_model_changed_callback(self, model_dict):
        print(f'Received global model for {self.worker.worker_id}')
        self.update = model_dict[GLOBAL_MODEL]
        self.gm_version = model_dict[GLOBAL_MODEL_VERSION]

    def get_last_global_model_version(self):
        return self.gm_version


def run_stress_worker(server_host_ip, server_port):
        """
        Run the workers loop for the basic stress test. This involves
        creating a a set of workers according to the keys in STRESS_KEYS_FOLDER then:

        - registering them with the server
        - get the current global model
        - send an update
        - Get the next global model.

        Parameters
        ----------

        server_host_ip: str
            The ip-address of the host of the server.

        server_port: int
            The port at which the serer should listen to
        """
        workers = []

        for fn in os.listdir(STRESS_KEYS_FOLDER):
            if fn.startswith(STRESS_WORKER_PREFIX) and not fn.endswith('.pub'):
                workers.append(SimpleLPWorker(
                   server_host_ip, server_port,
                   os.path.join(STRESS_KEYS_FOLDER, fn))
                )

        num_workers = len(workers)
        for i, worker in enumerate(workers):
            print(f'Registering {i} th worker')
            worker.worker.register_worker()

        # get the current global model and check
        print("Requesting global model")
        for worker in workers:
            worker.global_model_changed_callback(worker.worker.get_global_model())

        done_count = 0
        def run_wg(gl_worker):
            nonlocal done_count
            logger.info(f"Starting long poll for {gl_worker.worker.worker_id}")
            gl_worker.worker.send_model_update(msgpack.packb('A model update'))
            gl_worker.global_model_changed_callback(
                gl_worker.worker.get_global_model())
            logger.info(f"Long poll for {gl_worker.worker.worker_id} finished")
            done_count += 1

        for i, worker in enumerate(workers):
            print(f"Spawning for worker {i}")
            Greenlet.spawn(run_wg, worker)
            if (i+1) % 100 == 0:
                sleep(0.5)

        start_time = datetime.now()
        # if it hasn't stopped after 100 seconds, it has failed.
        while done_count < num_workers and (datetime.now() - start_time).seconds < 100 :
            sleep(1)
            logger.info(f"{done_count} workers have received the global model update - need to get to {num_workers}...")


def get_args():
    """
    Parse the argument for the worker for the basic stress test.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Run this with the ip-address and port at which the stress_server.py was run.\n")

    p.add_argument("--server-host-ip",
                   help="The ip of the host of server",
                   type=str,
                   required=True)
    p.add_argument("--server-port",
                   help="The ip of the host of server",
                   type=int,
                   required=True)

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_stress_worker(args.server_host_ip, args.server_port)
