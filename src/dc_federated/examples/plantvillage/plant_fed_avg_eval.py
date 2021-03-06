"""
Simple runner for evaluating or running inference with a model trained on the
PlantVillage dataset using FedAvg.
"""

import argparse
import yaml
import os

from dc_federated.examples.plantvillage.plant_fed_model import MobileNetV2Eval, PlantVillageSubSet


def get_args():
    """
    Parse the arguments for evaluating or running inference with a model trained on the PlantVillage dataset.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Start inference with model trained on the PlangVillage dataset.\n")

    p.add_argument("--test-data-path",
                   help="The path to the test dataset or a single image.",
                   type=str,
                   required=False)

    return p.parse_args()


def run():
    """
    Main run function to evaluate a model trained on the PlantVillage dataset.
    The test can be ran over a test set or a single image.
    """
    args = get_args()
    cfg = open("PlantVillage_cfg.yaml", 'r')
    cfg_dict = yaml.load(cfg)
    single_image_pred  = False
    if args.test_data_path is None:
        args.test_data_path = cfg_dict['output_dataset']['test_path']
    if os.path.isfile(args.test_data_path):
        single_image_pred  = True
        cat = os.path.basename(os.path.dirname(args.test_data_path))

    test_data_transform = PlantVillageSubSet.default_input_transform(False, (224,224))
    plant_ds_test = PlantVillageSubSet.default_plant_ds(
        root=args.test_data_path, transform=test_data_transform)

    model_evaluator = MobileNetV2Eval(
        model=cfg_dict['checkpoint_path'],
        test_loader=PlantVillageSubSet(
            plant_ds_test,
            transform=test_data_transform
        ).get_data_loader(),
        batches_per_iter = cfg_dict['batches_per_iter'],
        num_classes = cfg_dict['num_classes']
    )

    print("\n******** FEDERATED LEARNING MODEL EVAL ********")
    print(f"\n\tInitiating model")
    print("\n***********************************************\n")

    model_evaluator.load_model(cfg_dict['checkpoint_path'])
    if single_image_pred:
        model_evaluator.predict(args.test_data_path, cat, plant_ds_test.class_to_idx)
    else:
        model_evaluator.test()


if __name__ == '__main__':
    run()
