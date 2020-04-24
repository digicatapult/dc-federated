"""
Simple runner to prepare subsets of the PlantVillage dataset for running FedAvg workers.
Create train, test and validation sets for training a classifier for leaf diseases.

Create as many subsets as needed for the training and validation data, with predefined
distributions for the training sets between the various workers.
Each training set folder, corresponding to a worker, will be identified by an integer
starting from 0.


The output datasets are organised as follow:
    Original dataset:   'PlantVillageData/dataset/'+data_type, i.e. "color"
    Split dataset:      'dataset/processed/'
    Split train sets:   'dataset/processed/train0/
                        'dataset/processed/train1/
                        'dataset/processed/train.../
    Split val set:      'dataset/processed/val/
    Split test set:     'dataset/processed/test/
"""

import numpy as np
import os
import shutil
from pathlib import Path
import json
from os import listdir
import yaml


def distributions_list(distributions, categories): ####### Make a class #########
    """
    Map the categories to the fraction of the dataset for each worker subset.

    Parameters
    ----------
    distributions: list of lists
        The list of the distributions for each categories for each worker.

    categories: list of str
        The list of all categories present in the dataset folder.

    Returns
    -------

    distribs: list of tuples
        The list of categories and subset fraction pairs for each worker.
    """
    distribs = []
    for farm_distrib in distributions:
        distribs.append(dict(zip(categories, farm_distrib)))
    return distribs


def create_directories(base_dir, train_dir, val_dir, test_dir, distribs):
    """
    Create the directories to store the training subsets and validation
    and test sets.

    Parameters
    ----------
    base_dir: str
        The path of the original dataset.

    train_dir, val_dir, test_dir: str
        The target paths for train, validation and test subsets.

    distribs: list of tuples
        The list of categories and subset fraction pairs for each worker.
    """
    # Remove existing datasets and create the new dataset directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    # Create the train, val and test directories
    for i in range(len(distribs)):
        Path(train_dir+str(i)+'/').mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)


def copy_image(directory, category, image, source):
    """
    Copy an image file

    Parameters
    ----------
    directory: str
        The path of the target subset.

    category: str
        The category of the image file.

    image: str
        The image filename.

    source: str
        The path of the image source directory.
    """
    path = os.path.join(directory,category)
    if not os.path.exists(os.path.join(path,image)):
        shutil.copyfile(os.path.join(source,image), os.path.join(path,image))


def copy_subset(list_img, bound_inf, bound_sup, category, directory, source_directory):
    """
    Create the target directory if it does not exist.
    Copy the images of a category for a subset.

    Parameters
    ----------
    list_img: list
        The category image list.

    bound_inf: int
        Start index of the images of the subset in the category image list.

    bound_sup: int
        End index of the images of the subset in the category image list.

    category: str
        The name of the images category.

    directory: str
        The path of the image target directory.

    source_directory: str
        The path of the image source directory.
    """
    path = os.path.join(directory,category)
    if not os.path.exists(path):
        os.mkdir(path)
    for image in list_img[bound_inf:bound_sup]:
        if image.endswith(".jpg") or image.endswith(".JPG"):
            copy_image(directory, category, image, source_directory)


def create_subsets(data_dir, test_dir, val_dir, train_dir, categories, distribs):
    """
    Create the different subsets for original data for the PlantVillage dataset.

    Parameters
    ----------
    data_dir: str
        The path of the original dataset.

    train_dir, val_dir, test_dir: str
        The target paths for train, validation and test subsets.

    categories: list of str
        The list of categories of interest.

    distribs: list of tuples
        The list of categories and subset fraction pairs for each worker.
    """
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(data_dir)

        train_images = [0] * len(categories)

        for plant_disease_folder in categories:
            plant_disease_folder_list = listdir(os.path.join(data_dir,plant_disease_folder))
            print("[INFO] Processing {} with {} images".format(plant_disease_folder, len(plant_disease_folder_list)))
            for image in plant_disease_folder_list:
                # remove .DS_Store from list
                if image == ".DS_Store" :
                    plant_disease_folder_list.remove(image)

            source_directory = os.path.join(data_dir,plant_disease_folder)
            test_images = len(plant_disease_folder_list)//10
            valid_images = len(plant_disease_folder_list)//6

            # Copy test image samples
            copy_subset(plant_disease_folder_list, 0, test_images, plant_disease_folder,
                        test_dir, source_directory)

            # Copy validation image samples
            copy_subset(plant_disease_folder_list, -valid_images, -1, plant_disease_folder,
                        val_dir, source_directory)

            # Preparation of the split train sets
            train_images = min(len(plant_disease_folder_list)-test_images-valid_images, 1200)
            start_idx = test_images
            i = 0
            for farms in distribs:
                train_subset = round(train_images * farms[plant_disease_folder])
                # Copy train image samples
                copy_subset(plant_disease_folder_list, start_idx, start_idx+train_subset,
                            plant_disease_folder, train_dir+str(i), source_directory)
                start_idx += train_subset
                i += 1


        print("[INFO] Image loading completed")
    except Exception as e:
        print("Error : {}">format(e))


def run():
    cfg = open("PlantVillage_cfg.yaml", 'r')
    cfg_dict = yaml.load(cfg)

    data_dir = cfg_dict['orig_dataset']['path']
    base_dir = cfg_dict['output_dataset']['path']
    train_dir = base_dir+'train'
    val_dir = os.path.join(base_dir,'val')
    test_dir = os.path.join(base_dir,'test')


    # Select categories to remove from the analysis
    categories = cfg_dict['included_categories']
    print('Dataset categories: \n{}\n\nNumber of categories: {}'.format(categories, len(categories)))

    # Define distribution for various training subsets
    distributions = cfg_dict['distributions']
    distribs = distributions_list(distributions, categories)

    # Create the datasets
    create_directories(base_dir, train_dir, val_dir, test_dir, distribs)
    create_subsets(data_dir, test_dir, val_dir, train_dir, categories, distribs)


if __name__ == '__main__':
    run()
