"""
Run the server for the basic stress test.
"""

import sys
import os
import io
import msgpack
import argparse

from gevent import sleep
import torch
import torchvision.models as models

from dc_federated.backend import DCFServer, create_model_dict
from dc_federated.stress_test.stress_gen_keys import STRESS_KEYS_FOLDER, STRESS_WORKER_KEY_LIST_FILE

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def run_stress_server(global_model_real=False):
    """
    Runs the server for the basic stress test. This is started with a list of
    public keys and increments the model number/returns a model when it has received
    an update from each of the workers.

    Parameters
    ---------

    global_model_real: bool
        If true, the global model returned is a bianry serialized version of
        MobileNetV2 that is used in the plantvillage example.
    """
    server_model_check_interval = 1
    worker_ids = []
    worker_updates = {}
    global_model_version = 1
    updates_received_count = 0

    if global_model_real:
        model_data = io.BytesIO()
        torch.save(models.mobilenet_v2(pretrained=True), model_data)
        bin_model = model_data.getvalue()
    else:
        bin_model = msgpack.packb("A 'Global Model'!!")

    def test_register_func_cb(id):
        worker_ids.append(id)
        worker_updates[id] = None

    def test_unregister_func_cb(id):
        worker_ids.remove(id)
        del worker_updates[id]

    def test_ret_global_model_cb():
        return create_model_dict(
            bin_model,
            global_model_version)

    def is_global_model_most_recent(version):
        return version == global_model_version

    def rec_worker_update_cb(worker_id, update):
        nonlocal updates_received_count
        logger.info(f"Updates received {updates_received_count}")
        if worker_id in worker_ids and worker_updates[worker_id] is None:
            worker_updates[worker_id] = update
            updates_received_count += 1
            if updates_received_count == num_workers:
                halt_time = 10
                print(f"Sleeping for {halt_time} seconds now...")
                sleep(halt_time)
                print(f"Done sleeping ... changing global model...")
                updates_received_count = 0
                for worker_id in worker_ids:
                    worker_updates[worker_id] = None
                nonlocal global_model_version
                global_model_version += 1

            return f"Update received for worker {worker_id}."
        else:
            logger.info("Update not accepted.")
            return f"Update not accepted."

    keys_list_file = os.path.join(STRESS_KEYS_FOLDER, STRESS_WORKER_KEY_LIST_FILE)
    dcf_server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=rec_worker_update_cb,
        server_mode_safe=True,
        key_list_file=keys_list_file,
        model_check_interval=server_model_check_interval,
        load_last_session_workers=False
    )
    num_workers = len(dcf_server.worker_manager.allowed_workers)
    dcf_server.start_server()


def get_args():
    """
    Parse the argument for the worker for the basic stress test.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Start the server for basic stress test.\n")
    p.add_argument(
        "--global-model-real",
        action='store_true'
    )

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    sys.argv = sys.argv[:1]
    run_stress_server(args.global_model_real)
