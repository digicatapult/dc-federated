"""
Run the server for the basic stress test.
"""

import sys
import os
import msgpack

from gevent import sleep

from dc_federated.backend import DCFServer, create_model_dict
from dc_federated.stress_test.stress_gen_keys import STRESS_KEYS_FOLDER, STRESS_WORKER_KEY_LIST_FILE

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def run_stress_server():
    """
    Runs the server for the basic stress test. This is started with a list of
    public keys and increments the model number/returns a model when it has received
    an update from each of the workers.

    Parameters
    ---------

    stress_keys_folder: str
        The public keys of the worker for the stress test.
    """
    server_model_check_interval = 1
    worker_ids = []
    worker_updates = {}
    global_model_version = 1
    updates_received_count = 0

    def test_register_func_cb(id):
        worker_ids.append(id)
        worker_updates[id] = None

    def test_unregister_func_cb(id):
        worker_ids.remove(id)
        del worker_updates[id]

    def test_ret_global_model_cb():
        return create_model_dict(
            msgpack.packb("Dump of a string"),
            global_model_version)

    def is_global_model_most_recent(version):
        return version == global_model_version

    def rec_worker_update_cb(worker_id, update):
        print(f"update sent by: {worker_id}")
        if worker_id in worker_ids and worker_updates[worker_id] is None:
            worker_updates[worker_id] = update
            nonlocal updates_received_count
            updates_received_count += 1
            print(f"Updates received {updates_received_count}")
            if updates_received_count == num_workers:
                halt_time = 10
                print(f"Sleeping for {halt_time} seconds now...")
                sleep(halt_time)
                print(f"Done sleeping ... changing global model...")

                nonlocal global_model_version
                updates_received_count = 0
                for worker_id in worker_ids:
                    worker_updates[worker_id] = None
                global_model_version += 1

            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

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


if __name__ == '__main__':
    run_stress_server()