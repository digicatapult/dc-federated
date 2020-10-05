"""
Test long polling.
"""

import os
import sys
import msgpack
from datetime import datetime

from nacl.encoding import HexEncoder
from gevent import Greenlet, sleep

from dc_federated.backend import DCFServer, DCFWorker, create_model_dict
from dc_federated.backend._constants import *
from dc_federated.backend.worker_key_pair_tool import gen_pair
from dc_federated.utils import StoppableServer, get_host_ip


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


class SimpleLPWorker(object):
    """
    Simple worker class to test the long-polling. This class was created because
    for the test we need to maintain the state gm_version on the different
    greenlets (whicha are pseduo-threads).
    """
    def __init__(self, s_host, s_port, private_key_file):
        self.gm_version = "0"
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
        self.update = model_dict[GLOBAL_MODEL]
        self.gm_version = model_dict[GLOBAL_MODEL_VERSION]

    def get_last_global_model_version(self):
        return self.gm_version


def start_server():
    # Create a set of keys to be supplied to the server
    server_model_check_interval = 1

    worker_ids = []
    worker_updates = {}
    global_model_version = 1
    updates_received_count = 0
    num_workers = int(sys.argv[1])

    def test_register_func_cb(id):
        worker_ids.append(id)

    def test_unregister_func_cb(id):
        worker_ids.remove(id)

    def test_ret_global_model_cb():
        return create_model_dict(
            msgpack.packb("Dump of a string"),
            global_model_version)

    def is_global_model_most_recent(version):
        return version == global_model_version

    def test_rec_server_update_cb(worker_id, update):
        if worker_id in worker_ids:
            worker_updates[worker_id] = update
            nonlocal updates_received_count
            updates_received_count += 1
            if updates_received_count == num_workers:
                halt_time = 10
                print(f"Sleeping for {halt_time} seconds now...")
                sleep(halt_time)
                print(f"Done sleeping changing global model...")
                nonlocal global_model_version
                global_model_version += 1
                updates_received_count = 0

            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

    keys_folder = 'stress_keys_folder'

    dcf_server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=True,
        key_list_file=os.path.join(keys_folder, 'stress_worker_public_keys.txt'),
        model_check_interval=server_model_check_interval,
        load_last_session_workers=False
    )

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)

    # def begin_server():
    #     dcf_server.start_server()
    # server_gl = Greenlet.spawn(begin_server)
    # sleep(2)

    dcf_server.start_server()


if __name__ == '__main__':
    start_server()