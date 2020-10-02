"""
Tests for the DCFWorker and DCFServer class.
"""

import gevent
from gevent import Greenlet, sleep
from gevent import monkey; monkey.patch_all()

import os
import msgpack
import zlib
import requests
import json

from dc_federated.backend import DCFServer, DCFWorker, create_model_dict, is_valid_model_dict
from dc_federated.backend._constants import *
from dc_federated.utils import StoppableServer, get_host_ip


def test_server_functionality():
    """
    Unit tests for the DCFServer and DCFWorker classes.
    """
    worker_ids = []
    worker_updates = {}
    global_model_version = "1"
    worker_global_model_version = "0"
    os.environ[ADMIN_USERNAME] = 'admin'
    os.environ[ADMIN_PASSWORD] = 'str0ng_s3cr3t'
    admin_auth = ('admin', 'str0ng_s3cr3t')

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)

    def begin_server():
        dcf_server.start_server(stoppable_server)

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

    def test_glob_mod_chng_cb(model_dict):
        nonlocal worker_global_model_version
        worker_global_model_version = model_dict[GLOBAL_MODEL_VERSION]

    def test_get_last_glob_model_ver():
        nonlocal worker_global_model_version
        return worker_global_model_version

    # try to create a server with incorrect server mode, key file combination - should raise ValueError

    try:
        dcf_server = DCFServer(
            register_worker_callback=test_register_func_cb,
            unregister_worker_callback=test_unregister_func_cb,
            return_global_model_callback=test_ret_global_model_cb,
            is_global_model_most_recent=is_global_model_most_recent,
            receive_worker_update_callback=test_rec_server_update_cb,
            server_mode_safe=False,
            key_list_file="some_file_name.txt",
            load_last_session_workers=False
        )
    except ValueError as ve:
        error_str = "Server started in unsafe mode but list of public keys provided. " \
                    "Either explicitly start server in safe mode or do not " \
                    "supply a public key list."
        assert str(ve) == error_str
    else:
        assert False

    # now create the actual server instance to use
    dcf_server = DCFServer(
        register_worker_callback=test_register_func_cb,
        unregister_worker_callback=test_unregister_func_cb,
        return_global_model_callback=test_ret_global_model_cb,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=test_rec_server_update_cb,
        server_mode_safe=False,
        key_list_file=None
    )
    server_gl = Greenlet.spawn(begin_server)
    sleep(2)

    # register a set of workers
    data = {
        PUBLIC_KEY_STR: "dummy public key",
        SIGNED_PHRASE: "dummy signed phrase"
    }
    for _ in range(3):
        requests.post(
            f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{REGISTER_WORKER_ROUTE}", json=data)

    assert len(worker_ids) == 3
    assert len(set(worker_ids)) == 3
    assert worker_ids[0].__class__ == worker_ids[1].__class__ == worker_ids[2].__class__

    response = requests.get(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{WORKERS_ROUTE}",
        auth=admin_auth).content

    workers_list = json.loads(response)
    assert all([worker[WORKER_ID_KEY] in worker_ids for worker in workers_list])

    requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{WORKERS_ROUTE}",
        json={}, auth=admin_auth)
    assert len(worker_ids) == 3

    admin_registered_worker = {
        PUBLIC_KEY_STR: "new_public_key",
        REGISTRATION_STATUS_KEY: True
    }
    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{WORKERS_ROUTE}",
        json=admin_registered_worker, auth=admin_auth)

    added_worker_dict = json.loads(response.content.decode('utf-8'))

    assert len(worker_ids) == 4
    assert worker_ids[3] != admin_registered_worker[PUBLIC_KEY_STR]
    assert worker_ids[3] == added_worker_dict[WORKER_ID_KEY]

    requests.delete(
        f"http://{dcf_server.server_host_ip}:"
        f"{dcf_server.server_port}/{WORKERS_ROUTE}/{added_worker_dict[WORKER_ID_KEY]}", auth=admin_auth)
    assert len(worker_ids) == 3

    # test getting the global model
    model_return_binary = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
        json={WORKER_ID_KEY: worker_ids[0],
              SIGNED_PHRASE: "",
              LAST_WORKER_MODEL_VERSION: "0"}
    ).content
    model_return = msgpack.unpackb(zlib.decompress(model_return_binary))
    assert isinstance(model_return, dict)
    assert model_return[GLOBAL_MODEL_VERSION] == global_model_version
    assert msgpack.unpackb(model_return[GLOBAL_MODEL]) == "Pickle dump of a string"

    # test sending the model update
    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}/{worker_ids[1]}",
        files={WORKER_MODEL_UPDATE_KEY: zlib.compress(msgpack.packb("Model update!!")),
               SIGNED_PHRASE: ""
               }
    ).content

    assert msgpack.unpackb(worker_updates[worker_ids[1]]) == "Model update!!"
    assert response.decode(
        "UTF-8") == f"Update received for worker {worker_ids[1]}."

    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}/3",
        files={WORKER_MODEL_UPDATE_KEY: zlib.compress(msgpack.packb("Model update for unregistered worker!!")),
               SIGNED_PHRASE: ""}).content

    assert 3 not in worker_updates
    assert response.decode('UTF-8') == INVALID_WORKER

    # *********** #
    # now test a DCFWorker on the same server.
    dcf_worker = DCFWorker(
        server_protocol='http',
        server_host_ip=dcf_server.server_host_ip,
        server_port=dcf_server.server_port,
        global_model_version_changed_callback=test_glob_mod_chng_cb,
        get_worker_version_of_global_model=test_get_last_glob_model_ver,
        private_key_file=None)

    # test worker registration
    dcf_worker.register_worker()
    assert dcf_worker.worker_id == worker_ids[3]

    # test getting the global model update
    global_model_dict = dcf_worker.get_global_model()
    assert is_valid_model_dict(global_model_dict)
    assert global_model_dict[GLOBAL_MODEL_VERSION] == global_model_version
    assert msgpack.unpackb(global_model_dict[GLOBAL_MODEL]) == "Pickle dump of a string"

    # test sending the model update
    response = dcf_worker.send_model_update(
        msgpack.packb("DCFWorker model update"))
    assert msgpack.unpackb(worker_updates[worker_ids[3]]) == "DCFWorker model update"
    assert response.decode(
        "UTF-8") == f"Update received for worker {worker_ids[3]}."

    stoppable_server.shutdown()
