"""
Tests for the admin API for DCFServer.
"""

import gevent
from gevent import Greenlet, sleep
from gevent import monkey; monkey.patch_all()

from nacl.signing import SigningKey
from nacl.encoding import HexEncoder

import os
import io
import msgpack
import logging
import zlib
import requests
import json
import hashlib

from dc_federated.backend import DCFServer, DCFWorker, create_model_dict, is_valid_model_dict
from dc_federated.backend._constants import *
from dc_federated.backend.worker_key_pair_tool import gen_pair, verify_pair
from dc_federated.utils import StoppableServer, get_host_ip


def test_server_functionality():
    """
    Unit tests for the DCFServer and DCFWorker classes.
    """
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
    num_workers = 3
    worker_key_file_prefix = 'worker_key_file'
    for n in range(num_workers):
        private_key, public_key = gen_pair(worker_key_file_prefix + f'_{n}')
        private_keys.append(private_key.encode(encoder=HexEncoder))
        public_keys.append(public_key.encode(encoder=HexEncoder))

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

    dcf_server_safe = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=True,
        key_list_file=None,
        load_last_session_workers=False
    )

    dcf_server_unsafe = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=False,
        key_list_file=None,
        load_last_session_workers=False
    )

    def get_worker_key(mode, i):
        if mode == 'safe': return public_keys[i].decode('utf-8')
        else: return 'dummy_public_key'

    def get_signed_phrase(mode, i, phrase=b'test phrase'):
        if mode == 'safe':
            return SigningKey(private_keys[i], encoder=HexEncoder).sign(phrase).hex()
        else: return 'dummy_signed_phrase'

    for server, mode in zip([dcf_server_unsafe, dcf_server_safe], ['unsafe', 'safe']):
        worker_ids = []
        added_workers = []
        worker_updates = {}

        stoppable_server = StoppableServer(host=get_host_ip(), port=8080)
        server_gl = Greenlet.spawn(begin_server, server, stoppable_server)
        sleep(2)

        returned_ids = []
        # Phase 1: register a set of workers using the admin API and test registration
        for i in range(num_workers):

            admin_registered_worker = {
                PUBLIC_KEY_STR: get_worker_key(mode, i),
                REGISTRATION_STATUS_KEY: True
            }
            response = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}",
                json=admin_registered_worker, auth=admin_auth)

            added_worker_dict = json.loads(response.content.decode('utf-8'))
            assert len(worker_ids) == i + 1
            assert worker_ids[i] == added_worker_dict[WORKER_ID_KEY]
            added_workers.append(added_worker_dict[WORKER_ID_KEY])

        # Phase 2: Send updates and receive global updates for the registered workers
        # This should succeed
        worker_updates = {}
        for i in range(num_workers):
            # send updates
            signed_phrase = get_signed_phrase(mode, i, hashlib.sha256(msgpack.packb("Model update!!")).digest())
            response = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/"
                f"{RECEIVE_WORKER_UPDATE_ROUTE}/{added_workers[i]}",
                files={WORKER_MODEL_UPDATE_KEY: zlib.compress(msgpack.packb("Model update!!")),
                       SIGNED_PHRASE: signed_phrase
                       }
            ).content
            print(response)
            assert msgpack.unpackb(worker_updates[worker_ids[i]]) == "Model update!!"
            assert response.decode(
                "UTF-8") == f"Update received for worker {added_workers[i]}."

            # receive updates
            challenge_phrase = requests.get(f"http://{server.server_host_ip}:{server.server_port}/"
                                            f"{CHALLENGE_PHRASE_ROUTE}/{added_workers[i]}").content
            model_return_binary = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
                json={WORKER_ID_KEY: added_workers[i],
                      SIGNED_PHRASE: get_signed_phrase(mode, i, challenge_phrase),
                      LAST_WORKER_MODEL_VERSION: "0"}
            ).content
            model_return = msgpack.unpackb(zlib.decompress(model_return_binary))
            assert isinstance(model_return, dict)
            assert model_return[GLOBAL_MODEL_VERSION] == global_model_version
            assert msgpack.unpackb(model_return[GLOBAL_MODEL]) == "Pickle dump of a string"

        # Phase 3: Unregister workers.
        for i in range(num_workers):
            admin_registered_worker = {
                PUBLIC_KEY_STR: get_worker_key(mode, i),
                REGISTRATION_STATUS_KEY: False
            }
            response = requests.put(
                f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}"
                f"/{added_workers[i]}",
                json=admin_registered_worker, auth=admin_auth)
            unreg_worker_dict = json.loads(response.content.decode('utf-8'))
            assert not unreg_worker_dict[REGISTRATION_STATUS_KEY]
        assert len(worker_ids) == 0

        # Phase 4: Try to send updates from the unregistered workers - this should fail
        worker_updates = {}
        for i in range(num_workers):
            # send updates
            signed_phrase = get_signed_phrase(mode, i, hashlib.sha256(msgpack.packb("Model update!!")).digest())
            response = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/"
                f"{RECEIVE_WORKER_UPDATE_ROUTE}/{added_workers[i]}",
                files={WORKER_MODEL_UPDATE_KEY: zlib.compress(msgpack.packb("Model update!!")),
                       SIGNED_PHRASE: signed_phrase
                       }
            ).content
            assert added_workers[i] not in worker_updates
            assert response.decode('UTF-8') == UNREGISTERED_WORKER

            # receive updates
            model_return_binary = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
                json={WORKER_ID_KEY: added_workers[i],
                      LAST_WORKER_MODEL_VERSION: "0"}
            ).content
            assert response.decode('UTF-8') == UNREGISTERED_WORKER

        # Phase 5: Re-register existing workers.
        for i in range(num_workers):
            admin_registered_worker = {
                PUBLIC_KEY_STR: get_worker_key(mode, i),
                REGISTRATION_STATUS_KEY: True
            }
            response = requests.put(
                f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}"
                f"/{added_workers[i]}",
                json=admin_registered_worker, auth=admin_auth)
            unreg_worker_dict = json.loads(response.content.decode('utf-8'))
            assert unreg_worker_dict[REGISTRATION_STATUS_KEY]

        # Phase 6: Send updates and receive global updates for the registered workers
        # This should succeed
        worker_updates = {}
        for i in range(num_workers):
            # send updates
            signed_phrase = get_signed_phrase(mode, i, hashlib.sha256(msgpack.packb("Model update!!")).digest())
            response = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/"
                f"{RECEIVE_WORKER_UPDATE_ROUTE}/{added_workers[i]}",
                files={WORKER_MODEL_UPDATE_KEY: zlib.compress(msgpack.packb("Model update!!")),
                       SIGNED_PHRASE: signed_phrase
                       }
            ).content
            assert msgpack.unpackb(worker_updates[worker_ids[i]]) == "Model update!!"
            assert response.decode(
                "UTF-8") == f"Update received for worker {added_workers[i]}."

            # receive updates
            challenge_phrase = requests.get(f"http://{server.server_host_ip}:{server.server_port}/"
                                            f"{CHALLENGE_PHRASE_ROUTE}/{added_workers[i]}").content
            model_return_binary = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
                json={WORKER_ID_KEY: added_workers[i],
                      SIGNED_PHRASE: get_signed_phrase(mode, i, challenge_phrase),
                      LAST_WORKER_MODEL_VERSION: "0"}
            ).content
            model_return = msgpack.unpackb(zlib.decompress(model_return_binary))
            assert isinstance(model_return, dict)
            assert model_return[GLOBAL_MODEL_VERSION] == global_model_version
            assert msgpack.unpackb(model_return[GLOBAL_MODEL]) == "Pickle dump of a string"

        # Phase 7: Delete existing workers.
        for i in range(num_workers):
            response = requests.delete(
                f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}"
                f"/{added_workers[i]}", auth=admin_auth)
            message_dict = json.loads(response.content.decode('utf-8'))
            assert SUCCESS_MESSAGE_KEY in message_dict
        assert len(worker_ids) == 0

        # Phase 8: Try to send updates to the deleted workers - this should fail
        worker_updates = {}
        for i in range(num_workers):
            # send updates
            signed_phrase = get_signed_phrase(mode, i, hashlib.sha256(msgpack.packb("Model update!!")).digest())
            response = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/"
                f"{RECEIVE_WORKER_UPDATE_ROUTE}/{added_workers[i]}",
                files={WORKER_MODEL_UPDATE_KEY: zlib.compress(msgpack.packb("Model update!!")),
                       SIGNED_PHRASE: signed_phrase
                       }
            ).content
            assert added_workers[i] not in worker_updates
            assert response.decode('UTF-8') == INVALID_WORKER

            # receive updates
            challenge_phrase = requests.get(f"http://{server.server_host_ip}:{server.server_port}/"
                                            f"{CHALLENGE_PHRASE_ROUTE}/{added_workers[i]}").content
            model_return_binary = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
                json={WORKER_ID_KEY: added_workers[i],
                      SIGNED_PHRASE: get_signed_phrase(mode, i, challenge_phrase),
                      LAST_WORKER_MODEL_VERSION: "0"}
            ).content
            assert response.decode('UTF-8') == INVALID_WORKER

        # Phase 9: Try to register non-existent workers using the public API
        # - this should fail in the safe mode and succeed in the unsafe mode.
        for i in range(num_workers):
            registration_data = {
                PUBLIC_KEY_STR: get_worker_key(mode, i),
                SIGNED_PHRASE: get_signed_phrase(mode, i)
            }
            response = requests.post(
                f"http://{server.server_host_ip}:{server.server_port}/{REGISTER_WORKER_ROUTE}",
                json=registration_data)
            if mode == 'safe':
                assert response.content.decode('utf-8') == INVALID_WORKER
            else:
                assert 'unauthenticated' in response.content.decode('utf-8')

        # Phase 10 - for the safe mode try registering with the public and admin API
        # with invalid public keys - these should both fail
        if mode == 'safe':
            for i in range(num_workers):
                registration_data = {
                    PUBLIC_KEY_STR: "dummy public key",
                    SIGNED_PHRASE: get_signed_phrase(mode, i)
                }
                response = requests.post(
                    f"http://{server.server_host_ip}:{server.server_port}/{REGISTER_WORKER_ROUTE}",
                    json=registration_data)
                assert response.content.decode('utf-8') == INVALID_WORKER

                registration_data = {
                    PUBLIC_KEY_STR: get_worker_key(mode, i),
                    SIGNED_PHRASE: "dummy signed phrase key"
                }
                response = requests.post(
                    f"http://{server.server_host_ip}:{server.server_port}/{REGISTER_WORKER_ROUTE}",
                    json=registration_data)
                assert response.content.decode('utf-8') == INVALID_WORKER

                admin_registered_worker = {
                    PUBLIC_KEY_STR: "dummy public key",
                    REGISTRATION_STATUS_KEY: True
                }
                response = requests.post(
                    f"http://{server.server_host_ip}:{server.server_port}/{WORKERS_ROUTE}",
                    json=admin_registered_worker, auth=admin_auth)
                message = json.loads(response.content.decode('utf-8'))
                assert ERROR_MESSAGE_KEY in message
                assert message[ERROR_MESSAGE_KEY] == \
                       "Unable to validate public key for dummy public key " \
                       "- worker not added."

        stoppable_server.shutdown()
