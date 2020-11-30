"""
Check that repeated registration with the same does not kill the pool size.
"""

import os
import argparse
import msgpack
import io

import torch
import torchvision.models as models

from dc_federated.stress_test.stress_utils import get_worker_keys_from_chunk, SimpleLPWorker
from gevent import Greenlet, sleep

from dc_federated.backend import DCFWorker
from dc_federated.backend._constants import *
from dc_federated.stress_test.stress_gen_keys import STRESS_KEYS_FOLDER

import logging


logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def run_pool_exhaust_test(server_host_ip, server_port, num_runs, global_model_real):
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
    """
    workers = []
    if global_model_real:
        model_data = io.BytesIO()
        torch.save(models.mobilenet_v2(pretrained=True), model_data)
        bin_model = model_data.getvalue()
    else:
        bin_model = msgpack.packb("A 'local model update'!!")

    chunk_str = "1 of 1"
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
        print(f"Requesting global model for {worker.worker.worker_id}")
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

    try:
        for run_no in range(num_runs+1):
            logger.info(f"********************** STARTING RUN {run_no + 1}:")
            sleep(5)
            worker_lets = []
            workers_lst = workers[:-1] if run_no < num_runs else workers
            for i, worker in enumerate(workers_lst):
                logger.info(f"Spawning for worker {i}")
                worker_lets.append(Greenlet.spawn(run_wg, worker))
                if (i + 1) % 10 == 0:
                    sleep(0.5)
            if run_no < num_runs:
                for i, worker_let in enumerate(worker_lets):
                    logger.info(f"Killing worker {i}")
                    worker_let.kill()
                continue

            while done_count < num_workers:
                sleep(1)
                logger.info(
                    f"{done_count} workers have received the global model update - need to get to {num_workers}...")
            done_count = 0
    except Exception as e:
        print(e)
        exit()


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
                   default=20)
    p.add_argument("--global-model-real",
                   action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_pool_exhaust_test(
        args.server_host_ip,
        args.server_port,
        args.num_runs,
        args.global_model_real
    )
