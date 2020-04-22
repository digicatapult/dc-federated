"""
Simple runner to start FedAvgWorker for the PlantVillage dataset.
"""

import sys
import argparse
import yaml

from dc_federated.fed_avg.fed_avg_worker import FedAvgWorker
from dc_federated.plantvillage.plant_fed_model import MobileNetV2Trainer, PlantVillageSubSet


def get_args():
    """
    Parse the arguments for running the FedAvg worker for the PlantVillage dataset.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Start a FedAvg worker for the PlangVillage dataset.\n"
                    "Run this with the parameter provided by running the plant_fed_avg_server.py\n")

    p.add_argument("--server-host-ip",
                   help="The ip of the host of server",
                   type=str,
                   required=True)
    p.add_argument("--server-port",
                   help="The ip of the host of server",
                   type=str,
                   required=True)

    p.add_argument("--worker-id",
                   help="The id of the worker",
                   type=str,
                   required=True)

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
    Main run function to start a FedAvg worker for the PlantVillage dataset.
    """
    args = get_args()
    cfg = open("PlantVillage_cfg.yaml", 'r')
    cfg_dict = yaml.load(cfg)
    if args.train_data_path is None:
        args.train_data_path = cfg_dict['output_dataset']['train_path']+str(args.worker_id)
    if args.validation_data_path is None:
        args.validation_data_path = cfg_dict['output_dataset']['val_path']

    train_data_transform = PlantVillageSubSet.default_input_transform(True, (224,224))
    test_data_transform = PlantVillageSubSet.default_input_transform(False, (224,224))
    plant_ds_train = PlantVillageSubSet.default_plant_ds(
        root=args.train_data_path, transform=train_data_transform)
    plant_ds_test = PlantVillageSubSet.default_plant_ds(
        root=args.validation_data_path, transform=test_data_transform)

    local_model_trainer = MobileNetV2Trainer(
        train_loader=PlantVillageSubSet(
            plant_ds_train,
            transform=train_data_transform
        ).get_data_loader(),
        test_loader=PlantVillageSubSet(
            plant_ds_test,
            transform=test_data_transform
        ).get_data_loader(),
        batches_per_iter = cfg_dict['batches_per_iter'],
        num_classes = cfg_dict['num_classes']
    )

    print("\n******** FEDERATED LEARNING EXPERIMENT ********")
    print(f"\n\tStarting Federated Average Worker: {args.worker_id}")
    print("\n***********************************************\n")

    fed_avg_worker = FedAvgWorker(local_model_trainer, args.server_host_ip, args.server_port)
    fed_avg_worker.run_worker_loop()


if __name__ == '__main__':
    run()
