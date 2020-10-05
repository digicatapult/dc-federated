"""
Test long polling.
"""

import os
import sys
import msgpack
from datetime import datetime

from nacl.encoding import HexEncoder
from nacl.signing import SigningKey, VerifyKey

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
        self.update = model_dict[GLOBAL_MODEL]
        self.gm_version = model_dict[GLOBAL_MODEL_VERSION]

    def get_last_global_model_version(self):
        return self.gm_version


keys_folder = 'stress_keys_folder'
# read the public and private keys
private_keys = []
public_keys = []

workers = []

worker_key_file_prefix = 'worker_key_file'
for fn in os.listdir(keys_folder):
   if fn.startswith(worker_key_file_prefix) and not fn.endswith('.pub'):
       workers.append(SimpleLPWorker(
           sys.argv[1],'8080',
           os.path.join(keys_folder, fn))
       )

num_workers = len(workers)
for i, worker in enumerate(workers):
    print(f'Registering {i} th worker')
    worker.worker.register_worker()

# get the current global model and check
for worker in workers:
    worker.global_model_changed_callback(worker.worker.get_global_model())

done_count = 0

# test that a single call to the server exits after 5 seconds.
def run_wg(gl_worker):
    global done_count
    logger.info(f"Starting long poll for {gl_worker.worker.worker_id}")
    worker.worker.send_model_update(msgpack.packb('A model update'))
    gl_worker.global_model_changed_callback(
        gl_worker.worker.get_global_model())
    logger.info(f"Long poll for {gl_worker.worker.worker_id} finished")
    done_count += 1

for i, worker in enumerate(workers):
    print(f"Spawning for worker {i}")
    Greenlet.spawn(run_wg, worker)
    if (i+1) % 100 == 0:
        sleep(0.5)

# # get the current global model and check
# for i, worker in enumerate(workers):
#     if (i + 1) % 100 == 0:
#         sleep(0.5)


start_time = datetime.now()
# if it hasn't stopped after 100 seconds, it has failed.
while done_count < num_workers and (datetime.now() - start_time).seconds < 100 :
    sleep(1)
    logger.info(f"{done_count} workers have received the global model update - need to get to {num_workers}...")
# all the calls to get the global model should have succeeded by now




