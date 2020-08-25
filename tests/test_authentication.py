"""
Test worker authentication reltated functions
"""

import os
import time
from threading import Thread
import pickle
import zlib

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

from dc_federated.backend import DCFServer, DCFWorker
from dc_federated.backend._constants import *
from dc_federated.backend.worker_key_pair_tool import gen_pair, verify_pair
from dc_federated.utils import StoppableServer, get_host_ip

import requests

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
    status = 'Status is good!!'

    def test_register_func_cb(id):
        worker_ids.append(id)

    def test_unregister_func_cb(id):
        worker_ids.remove(id)

    def test_ret_global_model_cb():
        return pickle.dumps("Pickle dump of a string")

    def test_query_status_cb():
        return status

    def test_rec_server_update_cb(worker_id, update):
        if worker_id in worker_ids:
            worker_updates[worker_id] = update
            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

    def test_glob_mod_chng_cb():
        pass

    worker_key_file = 'worker_public_keys.txt'
    with open(worker_key_file, 'w') as f:
        for public_key in public_keys[:-1]:
            f.write(public_key.encode(
                encoder=HexEncoder).decode('utf-8') + os.linesep)
        f.write(
            public_keys[-1].encode(encoder=HexEncoder).decode('utf-8') + os.linesep)

    dcf_server = DCFServer(
        test_register_func_cb,
        test_unregister_func_cb,
        test_ret_global_model_cb,
        test_query_status_cb,
        test_rec_server_update_cb,
        key_list_file=worker_key_file
    )

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)

    def begin_server():
        dcf_server.start_server(stoppable_server)
    server_thread = Thread(target=begin_server)
    server_thread.start()
    time.sleep(2)

    # create the worker
    workers = [DCFWorker('http', dcf_server.server_host_ip,
                         dcf_server.server_port,
                         test_glob_mod_chng_cb,
                         worker_key_file_prefix + f"_{n}")
               for n in range(num_workers)]

    # test various worker actions
    for worker, key in zip(workers, public_keys):
        worker.register_worker()
        model_status = worker.get_global_model_status()
        global_model = worker.get_global_model()
        worker.send_model_update(b'model_update')
        assert model_status == status
        assert global_model == pickle.dumps("Pickle dump of a string")
        assert worker_updates[worker.worker_id] == b'model_update'
        assert worker.worker_id == key.encode(
            encoder=HexEncoder).decode('utf-8')

    # try to authenticate a unregistered worker
    gen_pair('bad_worker')
    bad_worker = DCFWorker('http', dcf_server.server_host_ip,
                           dcf_server.server_port,
                           test_glob_mod_chng_cb,
                           'bad_worker')
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

    routes = [RETURN_GLOBAL_MODEL_ROUTE, QUERY_GLOBAL_MODEL_STATUS_ROUTE]
    for route in routes:
        response = requests.post(
            f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{route}",
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

    logger.info("\n\n*** All Tests Passed - Testing completed successfully ***")
    stoppable_server.shutdown()


if __name__ == '__main__':
    test_worker_key_pair_tool()
    test_worker_authentication()
