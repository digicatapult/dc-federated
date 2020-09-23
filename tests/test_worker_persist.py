"""
Test persistence of workers over multiple startups.
"""

import os
import zlib
import msgpack
import json
import hashlib
from tinydb import Query

from nacl.exceptions import BadSignatureError
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey, VerifyKey

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


def test_worker_persistence():
    worker_ids = []
    added_workers = []
    worker_updates = {}

    global_model_version = "1"
    worker_global_model_version = "0"
    os.environ[ADMIN_USERNAME] = 'admin'
    os.environ[ADMIN_PASSWORD] = 'str0ng_s3cr3t'
    admin_auth = ('admin', 'str0ng_s3cr3t')

    public_keys = []
    private_keys = []
    num_workers = 6
    num_pre_load_workers = 3
    worker_key_file_prefix = 'worker_key_file'
    for n in range(num_workers):
        private_key, public_key = gen_pair(worker_key_file_prefix + f'_{n}')
        private_keys.append(private_key.encode(encoder=HexEncoder).decode('utf-8'))
        public_keys.append(public_key.encode(encoder=HexEncoder).decode('utf-8'))

    # write the pre-loaded keys to the
    worker_key_file = 'worker_public_keys.txt'
    with open(worker_key_file, 'w') as f:
        for public_key in public_keys[0:num_pre_load_workers]:
            f.write(public_key + os.linesep)

    def begin_server(server, server_adapter):
        server.start_server(server_adapter)

    def test_register_func_cb(id):
        worker_ids.append(id)

    def test_unregister_func_cb(id):
        worker_ids.remove(id)

    def test_ret_global_model_cb():
        return create_model_dict(
            msgpack.packb("Pickle dump of a string"),
            global_model_version)

    def is_global_model_most_recent(version):
        return int(version) == global_model_version

    def test_rec_server_update_cb(worker_id, update):
        if worker_id in worker_ids:
            worker_updates[worker_id] = update
            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

    def get_signed_phrase(private_key, phrase=b'test phrase'):
        return SigningKey(private_key, encoder=HexEncoder).sign(phrase).hex()

    if os.path.exists('workers_db.json'):
        os.remove('workers_db.json')

    server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=True,
        load_last_session_workers=True,
        path_to_keys_db='workers_db.json',
        key_list_file=worker_key_file)

    worker_updates = {}
    worker_ids = []
    added_workers = []
    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)
    server_gl = Greenlet.spawn(begin_server, server, stoppable_server)
    sleep(2)

    assert len(server.worker_manager.public_keys_db) == 3

    returned_ids = []
    # Phase 1: register a set of workers using the admin API and test registration
    for i in range(num_pre_load_workers, num_workers):

        admin_registered_worker = {
            PUBLIC_KEY_STR: public_keys[i],
            REGISTRATION_STATUS_KEY: True
        }
        response = requests.post(
            f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}",
            json=admin_registered_worker, auth=admin_auth)

        added_worker_dict = json.loads(response.content.decode('utf-8'))
        idx = i  - num_pre_load_workers
        assert len(worker_ids) == idx + 1
        assert worker_ids[idx] == added_worker_dict[WORKER_ID_KEY]
        added_workers.append(added_worker_dict[WORKER_ID_KEY])

    assert len(server.worker_manager.public_keys_db) == 6

    for doc in server.worker_manager.public_keys_db.all():
        assert doc[PUBLIC_KEY_STR] in public_keys

    # Phase 2: Send updates and receive global updates for the registered workers
    # This should succeed
    worker_updates = {}
    for i in range(num_pre_load_workers, num_workers):
        # send updates

        signed_phrase = get_signed_phrase(private_keys[i], hashlib.sha256(msgpack.packb("Model update!!")).digest())
        response = requests.post(
            f"http://{server.server_host_ip}:{server.server_port}/"
            f"{RECEIVE_WORKER_UPDATE_ROUTE}/{added_workers[i - num_pre_load_workers]}",
            files={ID_AND_MODEL_KEY: zlib.compress(msgpack.packb("Model update!!")),
                   SIGNED_PHRASE: signed_phrase
                   }
        ).content
        assert msgpack.unpackb(worker_updates[worker_ids[i - num_pre_load_workers]]) == "Model update!!"
        assert response.decode(
            "UTF-8") == f"Update received for worker {added_workers[i - num_pre_load_workers]}."

        # receive updates

        challenge_phrase = requests.get(f"http://{server.server_host_ip}:{server.server_port}/"
                                        f"{CHALLENGE_PHRASE_ROUTE}/{added_workers[i - num_pre_load_workers]}").content
        model_return_binary = requests.post(
            f"http://{server.server_host_ip}:{server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
            json={WORKER_ID_KEY: added_workers[i - num_pre_load_workers],
                  SIGNED_PHRASE: get_signed_phrase(private_keys[i], challenge_phrase),
                  LAST_WORKER_MODEL_VERSION: "0"}
        ).content
        model_return = msgpack.unpackb(zlib.decompress(model_return_binary))
        assert isinstance(model_return, dict)
        assert model_return[GLOBAL_MODEL_VERSION] == global_model_version
        assert msgpack.unpackb(model_return[GLOBAL_MODEL]) == "Pickle dump of a string"

    stoppable_server.shutdown()

    server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=True,
        load_last_session_workers=True,
        path_to_keys_db='workers_db.json',
        key_list_file=worker_key_file)

    assert len(server.worker_manager.public_keys_db) == 6
    assert len(server.worker_manager.allowed_workers) == 6
    for doc in server.worker_manager.public_keys_db.all():
        assert doc[PUBLIC_KEY_STR] in server.worker_manager.allowed_workers

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)
    server_gl = Greenlet.spawn(begin_server, server, stoppable_server)
    sleep(2)

    # Phase 7: Delete existing workers.
    for i in range(num_pre_load_workers):
        response = requests.delete(
            f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}"
            f"/{added_workers[i]}", auth=admin_auth)
        message_dict = json.loads(response.content.decode('utf-8'))
        assert SUCCESS_MESSAGE_KEY in message_dict
    assert len(worker_ids) == 0

    assert len(server.worker_manager.public_keys_db) == 3
    assert len(server.worker_manager.allowed_workers) == 3
    for doc in server.worker_manager.public_keys_db.all():
        assert doc[PUBLIC_KEY_STR] in server.worker_manager.allowed_workers

    stoppable_server.shutdown()

    # delete the files
    for n in range(num_workers):
        os.remove(worker_key_file_prefix + f'_{n}')
        os.remove(worker_key_file_prefix + f'_{n}.pub')
    os.remove(worker_key_file)

    os.remove('workers_db.json')
    os.remove('workers_db.json.bak')
