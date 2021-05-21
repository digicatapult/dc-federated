"""
Simple runner to start FedAvgServer server for the PlantVillage dataset.
"""
import argparse
import yaml

from dc_federated.algorithms.fed_avg.fed_avg_server import FedAvgServer
from dc_federated.examples.plantvillage.plant_fed_model import MobileNetV2Trainer, PlantVillageSubSet


def get_args():
    """
    Parse the arguments for running the FedAvg server for PlantVillage dataset.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Start the FedAvg server for the PlantVillage with "
                    "the train and validation data-folders provided.\n")

    p.add_argument("--train-data-path",
                   help="The path to the train data.",
                   type=str,
                   required=False)
    p.add_argument("--validation-data-path",
                   help="The path to the validation data.",
                   type=str,
                   required=False)
    p.add_argument("--checkpoint-path",
                   help="The path to save the global model checkpoint.",
                   type=str,
                   required=False)
    p.add_argument("--update-lim",
                   help="The number of desired workers updates ber iteration.",
                   type=str,
                   required=False)
    p.add_argument("--server-host-ip",
                   help="The hostname or ip address of the server.",
                   type=str,
                   required=False,
                   default=None)
    p.add_argument("--server-port",
                   help="The port at which the server listens.",
                   type=int,
                   required=False,
                   default=8080)


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
    if args.checkpoint_path is None:
        args.checkpoint_path = cfg_dict['checkpoint_path']
    if args.update_lim is None:
        args.update_lim = cfg_dict['update_lim']

    global_model_trainer = MobileNetV2Trainer(
        train_loader=PlantVillageSubSet.default_dataset(
            True, args.train_data_path, (224,224)).get_data_loader(),
        test_loader=PlantVillageSubSet.default_dataset(
            False, args.validation_data_path, (224,224)).get_data_loader(),
        global_model=True,
        checkpoints=args.checkpoint_path
    )

    fed_avg_server = FedAvgServer(global_model_trainer=global_model_trainer,
                                  server_host_ip=args.server_host_ip,
                                  server_port=args.server_port,
                                  key_list_file=None,
                                  update_lim=args.update_lim)
    print("\n******** FEDERATED LEARNING EXPERIMENT ********")
    print("Starting an Federated Average server for PlantVillage at"
          f"\n\tserver-host-ip: {fed_avg_server.server.server_host_ip} \n\tserver-port: {fed_avg_server.server.server_port}")
    print("\n***********************************************\n")
    fed_avg_server.start()


if __name__ == '__main__':
    run()
