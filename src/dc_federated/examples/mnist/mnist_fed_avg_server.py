"""
Simple runner to start FedAvgServer server for the MNIST dataset.
"""


from dc_federated.algorithms.fed_avg.fed_avg_server import FedAvgServer
from dc_federated.examples.mnist import MNISTModelTrainer, MNISTSubSet


def run():
    """
    This should be run to start the global server to test the backend + model integration
    in a distributed setting. Once the server has started, run the corresponding local-model
    runner(s) in local_model.py in other devices. Run `python federated_local_model.py -h' for instructions
    on how to do so.
    """
    global_model_trainer = MNISTModelTrainer(
        train_loader=MNISTSubSet.default_dataset(is_train=True).get_data_loader(),
        test_loader = MNISTSubSet.default_dataset(is_train=False).get_data_loader())

    fed_avg_server = FedAvgServer(global_model_trainer=global_model_trainer, update_lim=3)
    print("\n************")
    print("Starting an Federated Average Server at"
          f"\n\tserver-host-ip: {fed_avg_server.server.server_host_ip} \n\tserver-port: {fed_avg_server.server.server_port}")
    print("\n************\n")
    fed_avg_server.start()

if __name__ == '__main__':
    run()
