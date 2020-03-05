"""
Tests for the DCFWorker and DCFServer class. As of now I am not sure what a good
way is to programmatically kill a server thread - so you have to kill the program
by pressing Ctrl+C.
"""
import io
from threading import Thread
import pickle


import requests
import time

from dc_fl_demo.dc_fed_sw import DCFServer, DCFWorker
from dc_fl_demo.dc_fed_sw._constants import *


def test_server_functionality():
    worker_ids = []
    worker_updates = {}
    status = 'Status is good!!'

    def test_register_func_cb(id):
        worker_ids.append(id)

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

    dcf_server = DCFServer(
        test_register_func_cb,
        test_ret_global_model_cb,
        test_query_status_cb,
        test_rec_server_update_cb,
    )
    server_thread = Thread(target=dcf_server.start_server)
    server_thread.start()
    time.sleep(2)

    # register a set of workers
    for i in range(3):
        requests.get(f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{REGISTER_WORKER_ROUTE}")

    assert worker_ids ==[0, 1, 2]

    # test the model status
    server_status = requests.get(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}"
    ).content.decode('UTF-8')
    assert server_status == "Status is good!!"

    status = 'Status is bad!!'
    server_status = requests.get(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{QUERY_GLOBAL_MODEL_STATUS_ROUTE}"
    ).content.decode('UTF-8')
    assert server_status == 'Status is bad!!'

    # test getting the global model
    model_binary = requests.get(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RETURN_GLOBAL_MODEL_ROUTE}").content
    assert pickle.load(io.BytesIO(model_binary)) == "Pickle dump of a string"

    # test sending the model update
    id_and_model_dict_good = {
        ID_AND_MODEL_KEY: pickle.dumps({
            WORKER_ID_KEY: 1,
            MODEL_UPDATE_KEY: pickle.dumps("Model update!!")
        })
    }
    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}",
        files=id_and_model_dict_good
    ).content
    assert pickle.load(io.BytesIO(worker_updates[1])) == "Model update!!"
    assert response.decode("UTF-8") == "Update received for worker 1."

    # test sending a model update for an unregistered worker
    id_and_model_dict_bad = {
        ID_AND_MODEL_KEY: pickle.dumps({
            WORKER_ID_KEY: 3,
            MODEL_UPDATE_KEY: pickle.dumps("Model update for unregistered worker!!")
        })
    }
    response = requests.post(
        f"http://{dcf_server.server_host_ip}:{dcf_server.server_port}/{RECEIVE_WORKER_UPDATE_ROUTE}",
        files=id_and_model_dict_bad
    ).content

    assert 3 not in worker_updates
    assert response.decode('UTF-8') == "Unregistered worker 3 tried to send an update."

    # *********** #
    # now test a DCFWorker on the same server.
    dcf_worker = DCFWorker(dcf_server.server_host_ip, dcf_server.server_port)

    # test worker registration
    dcf_worker.register_worker()
    assert dcf_worker.worker_id == 3
    assert 3 in worker_ids

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
    response = dcf_worker.send_model_update(pickle.dumps("DCFWorker model update"))
    assert pickle.load(io.BytesIO(worker_updates[3])) == "DCFWorker model update"
    assert response.decode("UTF-8") == "Update received for worker 3."

    # TODO: figure out how to kill the server thread and
    # TODO: eliminate this awfulness!
    print("\n\n*** Testing completed successfully - exit by pressing Ctrl+C ***")


if __name__ == '__main__':
    test_server_functionality()
