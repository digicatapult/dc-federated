import os
import io
import zlib
import time
import pickle
import requests

from threading import Thread

from dc_federated.backend import DCFServer, DCFWorker
from dc_federated.backend._constants import *
from dc_federated.utils import StoppableServer, get_host_ip


def test_server_functionality():
    """
    Unit tests for the DCFServer and DCFWorker classes.
    """
    worker_ids = []
    worker_updates = {}
    status = 'Status is good!!'
    os.environ['ADMIN_USERNAME'] = 'admin'
    os.environ['ADMIN_PASSWORD'] = 'str0ng_s3cr3t'

    stoppable_server = StoppableServer(host=get_host_ip(), port=8080)

    def begin_server():
        dcf_server.start_server(stoppable_server)

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

    dcf_server = DCFServer(
        test_register_func_cb,
        test_unregister_func_cb,
        test_ret_global_model_cb,
        test_query_status_cb,
        test_rec_server_update_cb,
        None
    )
    server_thread = Thread(target=begin_server)
    server_thread.start()

    time.sleep(2)

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

    adminAuth = ('admin', 'str0ng_s3cr3t')

    response = requests.get(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/workers", auth=adminAuth).content
    workers_list = json.loads(response)

    assert all([worker["worker_id"] in worker_ids for worker in workers_list])

    requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/workers", json={}, auth=adminAuth)
    assert len(worker_ids) == 3

    admin_registered_worker = {
        PUBLIC_KEY_STR: "new_public_key",
        "active": True
    }
    requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/workers", json=admin_registered_worker, auth=adminAuth)
    assert len(worker_ids) == 4
    assert worker_ids[3] == admin_registered_worker[PUBLIC_KEY_STR]

    requests.delete(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/workers/new_public_key", auth=adminAuth)
    assert len(worker_ids) == 3

    # test the model status
    server_status = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}",
        json={WORKER_ID_KEY: worker_ids[0]}
    ).content.decode('UTF-8')
    print(server_status)

    assert server_status == "Status is good!!"

    status = 'Status is bad!!'
    server_status = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}",
        json={WORKER_ID_KEY: worker_ids[0]}
    ).content.decode('UTF-8')
    assert server_status == 'Status is bad!!'

    # test getting the global model
    model_binary = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}",
        json={WORKER_ID_KEY: worker_ids[0]}
    ).content
    assert pickle.load(io.BytesIO(zlib.decompress(
        model_binary))) == "Pickle dump of a string"

    # test sending the model update
    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}/{worker_ids[1]}",
        files={ID_AND_MODEL_KEY: zlib.compress(pickle.dumps("Model update!!"))}
    ).content

    assert pickle.load(io.BytesIO(
        worker_updates[worker_ids[1]])) == "Model update!!"
    assert response.decode(
        "UTF-8") == f"Update received for worker {worker_ids[1]}."

    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}/3",
        files={ID_AND_MODEL_KEY: zlib.compress(
            pickle.dumps("Model update for unregistered worker!!"))}
    ).content

    assert 3 not in worker_updates
    assert response.decode('UTF-8') == UNREGISTERED_WORKER

    # *********** #
    # now test a DCFWorker on the same server.
    dcf_worker = DCFWorker('http', dcf_server.server_host_ip,
                           dcf_server.server_port, test_glob_mod_chng_cb, None)

    # test worker registration
    dcf_worker.register_worker()
    assert dcf_worker.worker_id == worker_ids[3]

    # test getting the model status
    status = dcf_worker.get_global_model_status()
    assert status == "Status is bad!!"
    status = "Status is good!!"
    status = dcf_worker.get_global_model_status()
    assert status == "Status is good!!"

    # test getting the global model update
    global_model = dcf_worker.get_global_model()
    assert pickle.load(io.BytesIO(global_model)) == "Pickle dump of a string"

    # test sending the model update
    response = dcf_worker.send_model_update(
        pickle.dumps("DCFWorker model update"))
    assert pickle.load(io.BytesIO(
        worker_updates[worker_ids[3]])) == "DCFWorker model update"
    assert response.decode(
        "UTF-8") == f"Update received for worker {worker_ids[3]}."

    stoppable_server.shutdown()
