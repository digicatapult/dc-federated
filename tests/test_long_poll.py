"""
Test long polling.
"""

import os
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


def test_long_polling():
    # Create a set of keys to be supplied to the server
    num_workers = 100
    private_keys = []
    public_keys = []
    server_model_check_interval = 1
    halt_time = 10

    keys_folder = 'keys_folder'
    if not os.path.exists(keys_folder):
        os.mkdir(keys_folder)
    worker_key_file_prefix = 'worker_key_file'

    for n in range(num_workers):
        private_key, public_key = gen_pair(os.path.join(keys_folder, worker_key_file_prefix + f'_{n}'))
        private_keys.append(private_key)
        public_keys.append(public_key)

    worker_ids = []
    worker_updates = {}
    global_model_version = "1"

    def test_register_func_cb(id):
        worker_ids.append(id)

    def test_unregister_func_cb(id):
        worker_ids.remove(id)

    def test_ret_global_model_cb():
        return create_model_dict(
            msgpack.packb("Pickle dump of a string"),
            global_model_version)

    def is_global_model_most_recent(version):
        return version == global_model_version

    def test_rec_server_update_cb(worker_id, update):
        if worker_id in worker_ids:
            worker_updates[worker_id] = update
            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

    worker_key_file = os.path.join(keys_folder, 'worker_public_keys.txt')
    with open(worker_key_file, 'w') as f:
        for public_key in public_keys[:-1]:
            f.write(public_key.encode(encoder=HexEncoder).decode('utf-8') + os.linesep)
        f.write(public_keys[-1].encode(encoder=HexEncoder).decode('utf-8') + os.linesep)

    dcf_server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=True,
        key_list_file=worker_key_file,
        model_check_interval=server_model_check_interval,
        load_last_session_workers=False
    )

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)

    def begin_server():
        dcf_server.start_server(stoppable_server)
    server_gl = Greenlet.spawn(begin_server)
    sleep(2)

    # create the workers
    workers = [SimpleLPWorker(dcf_server.server_host_ip,
                              dcf_server.server_port,
                              os.path.join(keys_folder, worker_key_file_prefix + f"_{n}"))
               for n in range(num_workers)]

    for worker, key in zip(workers, public_keys):
        worker.worker.register_worker()

    # get the current global model and check
    for worker in workers:
        worker.global_model_changed_callback(worker.worker.get_global_model())

    for worker in workers:
        assert worker.gm_version == global_model_version

    done_count = 0

    # test that a single call to the server exits after 5 seconds.
    def run_wg(gl_worker):
        logger.info(f"Starting long poll for {gl_worker.worker.worker_id}")
        gl_worker.global_model_changed_callback(
            gl_worker.worker.get_global_model())
        logger.info(f"Long poll for {gl_worker.worker.worker_id} finished")
        nonlocal done_count
        done_count += 1

    for i, worker in enumerate(workers):
        Greenlet.spawn(run_wg, worker)
        if (i+1) % 5 == 0:
            sleep(0.5)

    logger.info(f"The test will halt for {halt_time} seconds now...")

    sleep(halt_time)
    global_model_version = "2"

    start_time = datetime.now()
    # if it hasn't stopped after 100 seconds, it has failed.
    while done_count < num_workers and (datetime.now() - start_time).seconds < 100 :
        sleep(1)
        logger.info(f"{done_count} workers have received the global model update - need to get to {num_workers}...")
    # all the calls to get the global model should have succeeded by now

    assert done_count == num_workers
    logger.info(f"All workers have received the global model update.")

    stoppable_server.shutdown()

    for f in os.listdir(keys_folder):
        os.remove(os.path.join(keys_folder, f))
