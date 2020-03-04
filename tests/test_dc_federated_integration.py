"""
This will start and run the example local and global model and test that
they communicate as expected.
"""

import os

from multiprocessing import Process
import time
import torch

from dc_fl_demo.example_dcf_model import ExampleGlobalModel, ExampleLocalModel


def test_example():
    """
    This test will start a server using an instance of ExampleGlobalModel in a
    process, wait 3 seconds, and run a loop of an instance of ExampleLocalModel
    in a different process. After that it will test by looking at the model
    parameters, written to disk by the two objects, that they parameters are
    identical or same as required by the logic.
    """

    egm = ExampleGlobalModel()
    server_process = Process(target=egm.start)
    server_process.start()

    time.sleep(3)

    elm = ExampleLocalModel()
    worker_process = Process(target=elm.run_model)
    worker_process.start()

    time.sleep(3)
    # TODO: this sleep to let the server/workers
    # finish is a hack - try to do something smarter

    server_process.kill()
    worker_process.kill()

    # check that the global and local model parameters are equal
    print("Checking tensors are equal")

    # load the saved models
    with open("egm_global_model.torch", 'rb') as f:
        egm_global_model = torch.load(f)
    with open("egm_worker_update_0.torch", 'rb') as f:
        egm_local_model = torch.load(f)
    with open("elm_global_model.torch", 'rb') as f:
        elm_global_model = torch.load(f)
    with open("elm_worker_update_0.torch", 'rb') as f:
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

    print("All tensors are equal and the local and global models are different - tests passed.")
    print("Cleaning up.")
    os.remove('egm_global_model.torch')
    os.remove('egm_worker_update_0.torch')
    os.remove('elm_global_model.torch')
    os.remove('elm_worker_update_0.torch')


if __name__ == '__main__':
    test_example()
