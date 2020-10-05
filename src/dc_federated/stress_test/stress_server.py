"""
Test long polling.
"""
import sys
import argparse
import msgpack

from gevent import sleep

from dc_federated.backend import DCFServer, create_model_dict

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def run_stress_server(stress_keys_list_file):
    """
    Runs the server for the basic stress test. This is started with a list of
    public keys and increments the model number/returns a model when it has received
    an update from each of the workers.

    Paramters
    ---------

    stress_keys_list_file: str
        The public keys of the worker for the stress test.
    """
    server_model_check_interval = 1
    worker_ids = []
    worker_updates = {}
    global_model_version = 1
    updates_received_count = 0
    num_workers = int(sys.argv[1])

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
        if worker_id in worker_ids and worker_updates[worker_id] is None:
            worker_updates[worker_id] = update
            nonlocal updates_received_count
            updates_received_count += 1
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

    dcf_server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=rec_worker_update_cb,
        server_mode_safe=True,
        key_list_file=stress_keys_list_file,
        model_check_interval=server_model_check_interval,
        load_last_session_workers=False
    )
    dcf_server.start_server()


def get_args():
    """
    Parse the argument for starting the server for the basic stress testing.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Start the server for the basic stress test.\n")

    p.add_argument("--stress-keys-file-list",
                   help="The file containing the list of worker public keys for the stress test.",
                   type=str,
                   default=None,
                   required=False)

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_stress_server(args.stress_keys_file_list)