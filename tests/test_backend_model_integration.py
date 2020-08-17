"""
This will start and run the example local and global model and test that
they communicate as expected.
"""

import os
import logging
import threading

from multiprocessing import Process
import time
import torch

from dc_federated.examples.example_dcf_model import ExampleGlobalModel, ExampleLocalModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def test_example():
    """
    This test will start a server using an instance of ExampleGlobalModel in a
    process, wait 3 seconds, and run a loop of an instance of ExampleLocalModel
    in a different process. After that it will test by looking at the model
    parameters, written to disk by the two objects, that their parameters are
    identical or same as required by the logic.
    """
    egm = ExampleGlobalModel()
    server_process = Process(target=egm.start)
    server_process.start()

    time.sleep(3)

    elm = ExampleLocalModel()

    th = threading.Thread(target=elm.start)
    th.start()

    time.sleep(3)

    # TODO: the sleeps above to let the server/workers
    # finish is a hack - try to do something smarter.

    server_process.terminate()

    # check that the global and local model parameters are equal
    logger.info("Checking tensors are equal")
    egm_worker_update_name = f"egm_worker_update_{elm.worker.worker_id}.torch"
    elm_worker_update_name = f"elm_worker_update_{elm.worker.worker_id}.torch"

    # load the saved models
    with open("egm_global_model.torch", 'rb') as f:
        egm_global_model = torch.load(f)
    with open(egm_worker_update_name, 'rb') as f:
        egm_local_model = torch.load(f)
    with open("elm_global_model.torch", 'rb') as f:
        elm_global_model = torch.load(f)
    with open(elm_worker_update_name, 'rb') as f:
        elm_local_model = torch.load(f)

    # check for equality and non-equality
    for param_egm, param_elm in zip(egm_global_model.parameters(),
                                    elm_global_model.parameters()):
        assert torch.all(torch.eq(param_egm.data, param_elm.data))

    for param_egm, param_elm in zip(egm_local_model.parameters(),
                                    elm_local_model.parameters()):
        assert torch.all(torch.eq(param_egm.data, param_elm.data))

    for param_egm, param_elm in zip(egm_local_model.parameters(),
                                    egm_global_model.parameters()):
        assert not torch.all(torch.eq(param_egm.data, param_elm.data))

    logger.info(
        "All tensors are equal and the local and global models are different")
    logger.info("***************** ALL TESTS PASSED *****************")
    logger.info(
        "******* Ignore WARNINGs related to worker shutting down *******")
    logger.info("Cleaning up.")
    os.remove('egm_global_model.torch')
    os.remove(egm_worker_update_name)
    os.remove('elm_global_model.torch')
    os.remove(elm_worker_update_name)


if __name__ == '__main__':
    test_example()
