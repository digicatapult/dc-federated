"""
Simple runner to start FedAvgServer server for the PlantVillage dataset.
"""


from dc_federated.fed_avg.fed_avg_server import FedAvgServer
from dc_federated.plantvillage.plant_fed_model import MobileNetv2Trainer, PlantVillageSubSet


def run():
    """
    This should be run to start the global server to test the backend + model integration
    in a distributed setting. Once the server has started, run the corresponding local-model
    runner(s) in local_model.py in other devices. Run `python federated_local_model.py -h' for instructions
    on how to do so.
    """
    train_data = '~/code/PlantVillageData/dataset/processed/train1'
    valid_data = '~/code/PlantVillageData/dataset/processed/val'
    global_model_trainer = MobileNetv2Trainer(
        train_loader= PlantVillageSubSet.default_dataset(True, train_data, (224,224)).get_data_loader(),
        test_loader = PlantVillageSubSet.default_dataset(False, valid_data, (224,224)).get_data_loader())

    fed_avg_server = FedAvgServer(global_model_trainer=global_model_trainer, update_lim=4)
    print("\n************ FEDERATED LEARNING EXPERIMENT ************")
    print("Starting a Federated Average Server at"
          f"\n\tserver-host-ip: {fed_avg_server.server.server_host_ip} \n\tserver-port: {fed_avg_server.server.server_port}")
    print("\n************\n")
    fed_avg_server.start()

if __name__ == '__main__':
    run()
