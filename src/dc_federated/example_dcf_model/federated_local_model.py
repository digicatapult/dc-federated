import sys
import argparse

from dc_federated.example_dcf_model import ExampleLocalModel


def get_args():
    """
    Parse the argument for running the example local model for the distributed
    federated test.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Run this with the parameter provided by running python federated_global_model.py\n")

    p.add_argument("--server-host-ip",
                   help="The ip of the host of server",
                   type=str,
                   required=True)
    p.add_argument("--server-port",
                   help="The ip of the host of server",
                   type=str,
                   required=True)

    return p.parse_args()


def run():
    """
    This should be run to start a local model runner loop to test the backend + model
    integration in a distributed setting. Once the global model server has been started
    in a another machine using python federated_global_model.py, run this runner in
    other remote machines using the parameters provided by the global model server.
    """
    args = get_args()
    egm = ExampleLocalModel(args.server_host_ip, args.server_port)
    egm.run_model()


if __name__ == '__main__':
    run()
