"""
Simple runner to start FedAvgServer server for the MNIST dataset.
"""
import argparse
import sys

from dc_federated.algorithms.fed_avg.fed_avg_server import FedAvgServer
from dc_federated.examples.mnist.mnist_fed_model import MNISTModelTrainer, MNISTSubSet


def get_args():
    """
    Parse the argument for running the example local model for the distributed
    federated test.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Parameters for running the mnist_fed_avg_server\n")

    p.add_argument("--key-list-file",
                   help="The list of public keys for each worker to be authenticated.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--ssl-enabled", dest="ssl_enabled",
                   default=False, action="store_true")
    p.add_argument("--ssl-keyfile",
                   help="The path to the SSL key file.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--ssl-certfile",
                   help="The path to the SSL Certificate.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--server-host-ip",
                   help="The hostname or ip address of the server.",
                   type=str,
                   required=False,
                   default=None)

    args, rest = p.parse_known_args()

    # We remove the known args because gunicorn also uses its own ArgumentParser that would conflict with this
    sys.argv = sys.argv[:1] + rest

    return args


def run():
    """
    This should be run to start the global server to test the backend + model integration
    in a distributed setting. Once the server has started, run the corresponding local-model
    runner(s) in local_model.py in other devices. Run `python federated_local_model.py -h' for instructions
    on how to do so.
    """
    args = get_args()

    global_model_trainer = MNISTModelTrainer(
        train_loader=MNISTSubSet.default_dataset(
            is_train=True).get_data_loader(),
        test_loader=MNISTSubSet.default_dataset(is_train=False).get_data_loader())
    fed_avg_server = FedAvgServer(global_model_trainer=global_model_trainer,
                                  key_list_file=args.key_list_file,
                                  update_lim=3,
                                  server_host_ip=args.server_host_ip,
                                  ssl_enabled=args.ssl_enabled,
                                  ssl_keyfile=args.ssl_keyfile,
                                  ssl_certfile=args.ssl_certfile)
    print("\n************")
    print("Starting an Federated Average Server at"
          f"\n\tserver-host-ip: {fed_avg_server.server.server_host_ip} \n\tserver-port: {fed_avg_server.server.server_port}")
    print("\n************\n")
    fed_avg_server.start()


if __name__ == '__main__':
    run()
