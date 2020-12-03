"""
Run the workers for the basic stress test.
"""

import os
import argparse
import msgpack
import io

import torch
import torchvision.models as models

from dc_federated.stress_test.stress_utils import get_worker_keys_from_chunk, SimpleLPWorker
from gevent import Greenlet, sleep

from dc_federated.stress_test.stress_gen_keys import STRESS_KEYS_FOLDER

import logging


logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def run_stress_worker(server_host_ip, server_port, num_runs, worker_model_real, chunk_str):
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

    worker_model_real: bool
        If true, the global model returned is a bianry serialized version of
        MobileNetV2 that is used in the plantvillage example.

    chunk_str: str
        String giving the chunk of keys to use.
    """
    workers = []
    if worker_model_real:
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
        logger.info(f'Registering {i} th worker')
        worker.worker.register_worker()

    # get the current global model and check
    for i, worker in enumerate(workers):
        print(f"Requesting global model for {worker.worker.worker_id} (no. {i}) ")
        worker.global_model_changed_callback(worker.worker.get_global_model())

    done_count = 0

    def run_wg(gl_worker, num):
        nonlocal done_count
        logger.info(f"Starting long poll for {gl_worker.worker.worker_id} (no. {num})")
        gl_worker.global_model_changed_callback(
            gl_worker.worker.get_global_model())
        logger.info(f"Long poll for {gl_worker.worker.worker_id} finished")
        done_count += 1

    try:
        for run_no in range(num_runs):
            logger.info(f"********************** STARTING RUN {run_no + 1}:")
            sleep(5)
            for i, worker in enumerate(workers):
                response = worker.worker.send_model_update(bin_model)
                logger.info(f"Response from server sending model update: {response}")
                logger.info(f"Spawning for worker {i}")
                Greenlet.spawn(run_wg, worker, i)
                if (i+1) % 10 == 0:
                    sleep(0.5)

            while done_count < num_workers:
                sleep(1)
                logger.info(f"{done_count} workers have received the global model update - need to get to {num_workers}...")
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
                   default=1)
    p.add_argument("--worker-model-real",
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
        args.worker_model_real,
        args.chunk
    )
