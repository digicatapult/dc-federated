"""
Test worker authentication reltated functions
"""

import os
import pickle
import zlib

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

import requests
from gevent import Greenlet, sleep

from dc_federated.backend import DCFServer, DCFWorker, create_model_dict, is_valid_model_dict
from dc_federated.backend._constants import *
from dc_federated.backend.worker_key_pair_tool import gen_pair, verify_pair
from dc_federated.utils import StoppableServer, get_host_ip


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def test_worker_key_pair_tool():
    key_file = "gen_pair_test"
    private_key, public_key = gen_pair(key_file)

    test_phrase = b"Test phrase"
    try:
        public_key.verify(private_key.sign(test_phrase))
    except BadSignatureError as bse:
        assert False

    # load the generated keys from the file
    with open(key_file, 'r') as f:
        loaded_private_key = SigningKey(f.read().encode(), encoder=HexEncoder)
    with open(key_file + '.pub', 'r') as f:
        loaded_public_key = VerifyKey(f.read().encode(), encoder=HexEncoder)

    # test the
    try:
        loaded_public_key.verify(loaded_private_key.sign(test_phrase))
    except BadSignatureError as bse:
        assert False
    assert verify_pair(key_file)

    # test that a bad signature is detected
    with open(key_file, 'w') as f:
        f.write(SigningKey.generate().encode(
            encoder=HexEncoder).decode('utf-8'))
    assert not verify_pair(key_file)

    # clean up
    os.remove(key_file)
    os.remove(key_file + '.pub')


def test_worker_authentication():
    # Create a set of keys to be supplied to the server
    num_workers = 10
    private_keys = []
    public_keys = []
    worker_key_file_prefix = 'worker_key_file'
    for n in range(num_workers):
        private_key, public_key = gen_pair(worker_key_file_prefix + f'_{n}')
        private_keys.append(private_key)
        public_keys.append(public_key)

    worker_ids = []
    worker_updates = {}
    global_model_version = "1"
    worker_global_model_version = "0"

    def test_register_func_cb(id):
        worker_ids.append(id)

    def test_ret_global_model_cb():
        return create_model_dict(
            pickle.dumps("Pickle dump of a string"),
            global_model_version)

    def is_global_model_most_recent(version):
        return version == global_model_version

    def test_rec_server_update_cb(worker_id, update):
        if worker_id in worker_ids:
            worker_updates[worker_id] = update
            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

    def test_glob_mod_chng_cb(model_dict):
        nonlocal worker_global_model_version
        worker_global_model_version = model_dict[GLOBAL_MODEL_VERSION]

    def test_get_last_glob_model_ver():
        nonlocal worker_global_model_version
        return worker_global_model_version

    worker_key_file = 'worker_public_keys.txt'
    with open(worker_key_file, 'w') as f:
        for public_key in public_keys[:-1]:
            f.write(public_key.encode(
                encoder=HexEncoder).decode('utf-8') + os.linesep)
        f.write(
            public_keys[-1].encode(encoder=HexEncoder).decode('utf-8') + os.linesep)

    dcf_server = DCFServer(
        register_worker_callback=test_register_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        key_list_file=worker_key_file
    )

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)

    def begin_server():
        dcf_server.start_server(stoppable_server)

    server_gl = Greenlet.spawn(begin_server)
    sleep(2)

    # create the workers
    workers = [
        DCFWorker(
            server_protocol='http',
            server_host_ip=dcf_server.server_host_ip,
            server_port=dcf_server.server_port,
            global_model_version_changed_callback=test_glob_mod_chng_cb,
            get_worker_version_of_global_model=test_get_last_glob_model_ver,
            private_key_file=worker_key_file_prefix + f"_{n}")
        for n in range(num_workers)]

    # test various worker actions
    for worker, key in zip(workers, public_keys):
        worker.register_worker()
        global_model_dict = worker.get_global_model()
        worker.send_model_update(b'model_update')
        assert is_valid_model_dict(global_model_dict)
        assert global_model_dict[GLOBAL_MODEL] == pickle.dumps("Pickle dump of a string")
        assert global_model_dict[GLOBAL_MODEL_VERSION] == global_model_version
        assert worker_updates[worker.worker_id] == b'model_update'
        assert worker.worker_id == key.encode(encoder=HexEncoder).decode('utf-8')

    # try to authenticate a unregistered worker
    gen_pair('bad_worker')
    bad_worker = DCFWorker(
            server_protocol='http',
            server_host_ip=dcf_server.server_host_ip,
            server_port=dcf_server.server_port,
            global_model_version_changed_callback=test_glob_mod_chng_cb,
            get_worker_version_of_global_model=test_get_last_glob_model_ver,
            private_key_file='bad_worker')
    try:
        bad_worker.register_worker()
    except ValueError:
        assert True
    else:
        assert False

    # try to send an update through the using the bad worker public key
    with open('bad_worker', 'r') as f:
        bad_worker_key = f.read()

    id_and_model_dict_good = {
        ID_AND_MODEL_KEY: zlib.compress(pickle.dumps({
            WORKER_ID_KEY: bad_worker_key,
            MODEL_UPDATE_KEY: pickle.dumps("Bad Model update!!")
        }))
    }
    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}",
        files=id_and_model_dict_good
    ).content
    assert response.decode('utf-8') == UNREGISTERED_WORKER

    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
        json={WORKER_ID_KEY: bad_worker_key}
    ).content
    assert response.decode('utf-8') == UNREGISTERED_WORKER

    # delete the files
    for n in range(num_workers):
        os.remove(worker_key_file_prefix + f'_{n}')
        os.remove(worker_key_file_prefix + f'_{n}.pub')
    os.remove(worker_key_file)
    os.remove("bad_worker")
    os.remove("bad_worker.pub")

    stoppable_server.shutdown()
    logger.info("***************** ALL TESTS PASSED *****************")


if __name__ == '__main__':
    test_worker_key_pair_tool()
    test_worker_authentication()
