import math
import os
import re

from dc_federated.backend import DCFWorker, GLOBAL_MODEL, GLOBAL_MODEL_VERSION
from dc_federated.stress_test.stress_gen_keys import STRESS_WORKER_PREFIX, STRESS_KEYS_FOLDER

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

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
    files = [(fn, int(fn[len(STRESS_WORKER_PREFIX)+1:]))
             for fn in os.listdir(STRESS_KEYS_FOLDER)
             if fn.startswith(STRESS_WORKER_PREFIX) and not fn.endswith('.pub')]
    if n > len(files):
        logger.error(f"n in {chunk_str} cannot be greater than number of keys ({len(files)})")
        return []
    chunk_len = math.ceil(len(files) / n)
    files = sorted(files, key=lambda x: x[1])
    return [fn for fn, idx in files[(k-1)*chunk_len:  k*chunk_len]]


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
        values = re.findall("([0-9]*)? of ([0-9]*)?", chunk_str)
        k, n = int(values[0][0]), int(values[0][1])
        if 1 <= k <= n: return k, n
    except IndexError as e:
        logger.error(e)
        return None, None


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