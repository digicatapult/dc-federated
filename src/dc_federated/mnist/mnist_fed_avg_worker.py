import sys
import argparse

from dc_federated.fed_avg.fed_avg_worker import FedAvgWorker
from dc_federated.mnist.mnist_fed_model import MNISTModelTrainer, MNISTSubSet


def get_args():
    """
    Parse the argument for running the example local model for the distributed
    federated test.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Run this with the parameter provided by running the mnist_fed_avg_server\n")

    p.add_argument("--server-host-ip",
                   help="The ip of the host of server",
                   type=str,
                   required=True)
    p.add_argument("--server-port",
                   help="The ip of the host of server",
                   type=str,
                   required=True)

    p.add_argument("--digit-class",
                   help="The digit set this worker should focus on - allowed values are 0, 2 and 3.",
                   type=int,
                   required=True)


    return p.parse_args()


def run():
    """
    This should be run to start a local model runner loop to test the backend + model
    integration in a distributed setting. Once the global model server has been started
    in a another machine using python federated_global_model.py, run this runner in
    other remote machines using the parameters provided by the global model server.
    """
    digit_classes = [[0, 1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]

    args = get_args()

    data_transform = MNISTSubSet.default_data_transform()
    mnist_ds_train = MNISTSubSet.default_mnist_ds(is_train=True, data_transform=data_transform)
    mnist_ds_test = MNISTSubSet.default_mnist_ds(is_train=False, data_transform=data_transform)

    local_model_trainer = MNISTModelTrainer(
        train_loader=MNISTSubSet(
            mnist_ds_train,
            digits=digit_classes[args.digit_class],
            transform=data_transform
        ).get_data_loader(),
        test_loader=MNISTSubSet(
            mnist_ds_test,
            digits=digit_classes[args.digit_class],
            transform=data_transform
        ).get_data_loader()
    )

    fed_avg_worker = FedAvgWorker(local_model_trainer, args.server_host_ip, args.server_port)
    fed_avg_worker.run_worker_loop()


if __name__ == '__main__':
    run()
