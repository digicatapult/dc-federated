"""
Simple runner to start FedAvgServer server for the PlantVillage dataset.
"""
import argparse
import yaml

from dc_federated.fed_avg.fed_avg_server import FedAvgServer
from dc_federated.plantvillage.plant_fed_model import MobileNetV2Trainer, PlantVillageSubSet


def get_args():
    """
    Parse the arguments for running the FedAvg server for PlantVillage dataset.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Start the FedAvg server for the PlantVillage with "
                    "the train and validation data-folders provided.\n")

    p.add_argument("--train-data-path",
                   help="The path to the train data (created by the <insert-name> script).",
                   type=str,
                   required=False)

    p.add_argument("--validation-data-path",
                   help="The path to the validation data (created by the <insert-name> script).",
                   type=str,
                   required=False)

    return p.parse_args()


def run():
    """
    Main run function to start a FedAvg server for the PlantVillage dataset.
    """
    args = get_args()

    cfg = open("PlantVillage_cfg.yaml", 'r')
    cfg_dict = yaml.load(cfg)
    if args.train_data_path is None:
        args.train_data_path = cfg_dict['output_dataset']['path']
    if args.validation_data_path is None:
        args.validation_data_path = cfg_dict['output_dataset']['val_path']

    global_model_trainer = MobileNetV2Trainer(
        train_loader=PlantVillageSubSet.default_dataset(
            True, args.train_data_path, (224,224)).get_data_loader(),
        test_loader=PlantVillageSubSet.default_dataset(
            False, args.validation_data_path, (224,224)).get_data_loader()
    )

    fed_avg_server = FedAvgServer(global_model_trainer=global_model_trainer, update_lim=4)
    print("\n******** FEDERATED LEARNING EXPERIMENT ********")
    print("Starting an Federated Average server for PlantVillage at"
          f"\n\tserver-host-ip: {fed_avg_server.server.server_host_ip} \n\tserver-port: {fed_avg_server.server.server_port}")
    print("\n***********************************************\n")
    fed_avg_server.start()


if __name__ == '__main__':
    run()
