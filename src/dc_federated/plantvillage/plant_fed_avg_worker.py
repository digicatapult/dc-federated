"""
Simple runner to start FedAvgWorker for the PlantVillage dataset.
"""

import sys
import argparse

from dc_federated.fed_avg.fed_avg_worker import FedAvgWorker
from dc_federated.plantvillage.plant_fed_model import MobileNetv2Trainer, PlantVillageSubSet


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

    p.add_argument("--worker-id",
                   help="The number of the worker",
                   type=str,
                   required=True)

    return p.parse_args()


def run():
    """
    This should be run to start a FedAvgWorker. Run this script with the --help option
    to see what the options are.
    """

    args = get_args()


    train_data = '~/code/PlantVillageData/dataset/processed/train'+str(args.worker_id)
    valid_data = '~/code/PlantVillageData/dataset/processed/val'

    train_data_transform = PlantVillageSubSet.default_input_transform(True, (224,224))
    test_data_transform = PlantVillageSubSet.default_input_transform(False, (224,224))
    plant_ds_train = PlantVillageSubSet.default_plant_ds(root = train_data, transform=train_data_transform)
    plant_ds_test = PlantVillageSubSet.default_plant_ds(root = valid_data, transform=test_data_transform)

    local_model_trainer = MobileNetv2Trainer(
        train_loader=PlantVillageSubSet(
            plant_ds_train,
            transform=train_data_transform
        ).get_data_loader(),
        test_loader=PlantVillageSubSet(
            plant_ds_test,
            transform=test_data_transform
        ).get_data_loader()
    )

    print("\n************ FEDERATED LEARNING EXPERIMENT ************")
    print(f"\n\tStarting Federated Average Worker: {args.worker_id}")
    print("\n************\n")
    fed_avg_server.start()

    fed_avg_worker = FedAvgWorker(local_model_trainer, args.server_host_ip, args.server_port)
    fed_avg_worker.run_worker_loop()


if __name__ == '__main__':
    run()
