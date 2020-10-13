"""
Run the workers for the basic stress test.
"""

import os
import sys
import re
import argparse
import msgpack
import io
import math

import torch
import torchvision.models as models

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
        try:
            self.update = model_dict[GLOBAL_MODEL]
            self.gm_version = model_dict[GLOBAL_MODEL_VERSION]
        except Exception as e:
            print(str(e))
            print("Update received: ")
            print(model_dict)

    def get_last_global_model_version(self):
        return self.gm_version


def parse_chunk(chunk_str):
    """
    Parse a chunk string of the form "<k> of <n>" and returns k, n.
    If k > n, returns None, None

    Parameters
    ----------

    chunk_str: str
        The chunk string of the form "<k> of <n>"

    Returns
    -------

    int, int:
        k and n from valid chunk string with k<= n - None, None
        otherwise.
    """
    try:
        values = re.findall("([1-9]*)? of ([1-9]*)?", chunk_str)
        k, n = int(values[0][0]), int(values[0][1])
        if 1 <= k <= n: return k, n
    except IndexError as e:
        logger.error(e)
        return None, None


def get_worker_keys_from_chunk(chunk_str):
    """
    Gets the set of worker keys from the chunk string description.

    Parameters
    ----------

    chunk_str: str
        The chunk string of the form "<k> of <n>"

    Returns
    -------

    str list:
        List of valid worker kyes for this process.
    """
    k, n = parse_chunk(chunk_str)
    print(f"n = {n} , k = {k}")
    if n is None or k is None: return []
    files = [fn for fn in os.listdir(STRESS_KEYS_FOLDER)
             if fn.startswith(STRESS_WORKER_PREFIX) and not fn.endswith('.pub')]
    if n > len(files):
        logger.error(f"n in {chunk_str} cannot be greater than number of keys ({len(files)})")
        return []
    chunk_len = math.ceil(len(files) / n)
    return files[(k-1)*chunk_len: k*chunk_len]


def run_stress_worker(server_host_ip, server_port, num_runs, global_model_real, chunk_str):
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

        num_runs: int
            Number of runs of the sending of models etc. to perform

        global_model_real: bool
            If true, the global model returned is a bianry serialized version of
            MobileNetV2 that is used in the plantvillage example.

        chunk_str: str
            String giving the chunk of keys to use.
        """
        workers = []
        if global_model_real:
            model_data = io.BytesIO()
            torch.save(models.mobilenet_v2(pretrained=True), model_data)
            bin_model = model_data.getvalue()
        else:
            bin_model = msgpack.packb("A 'local model update'!!")

        for fn in get_worker_keys_from_chunk(chunk_str):
            workers.append(SimpleLPWorker(
               server_host_ip, server_port,
               os.path.join(STRESS_KEYS_FOLDER, fn))
            )

        num_workers = len(workers)
        for i, worker in enumerate(workers):
            # print(f'Registering {i} th worker')
            worker.worker.register_worker()

        # get the current global model and check

        for worker in workers:
            print(f"Requesting global model for {worker}")
            worker.global_model_changed_callback(worker.worker.get_global_model())

        done_count = 0
        def run_wg(gl_worker):
            nonlocal done_count
            logger.info(f"Starting long poll for {gl_worker.worker.worker_id}")
            gl_worker.worker.send_model_update(bin_model)
            gl_worker.global_model_changed_callback(
                gl_worker.worker.get_global_model())
            logger.info(f"Long poll for {gl_worker.worker.worker_id} finished")
            done_count += 1

        for run_no in range(num_runs):
            logger.info(f"********************** STARTING RUN {run_no + 1}:")
            sleep(5)
            for i, worker in enumerate(workers):
                # print(f"Spawning for worker {i}")
                Greenlet.spawn(run_wg, worker)
                if (i+1) % 100 == 0:
                    sleep(0.5)

            while done_count < num_workers:
                sleep(1)
                logger.info(f"{done_count} workers have received the global model update - need to get to {num_workers}...")
            done_count = 0


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
    p.add_argument("--num-runs",
                   help="The number of iterations of simulated FL to run.",
                   type=int,
                   required=False,
                   default=1)
    p.add_argument("--global-model-real",
                   action='store_true')
    p.add_argument("--chunk",
                   type=str,
                   required=False,
                   default="1 of 1")

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_stress_worker(
        args.server_host_ip,
        args.server_port,
        args.num_runs,
        args.global_model_real,
        args.chunk
    )
