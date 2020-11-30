"""
Simple runner to prepare subsets of the PlantVillage dataset for running FedAvg workers.
Create train, test and validation sets for training a classifier for leaf diseases.

Create as many subsets as needed for the training and validation data, with predefined
distributions for the training sets between the various workers.
Each training set folder, corresponding to a worker, will be identified by an integer,
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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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


def select_img(plant_disease_folder_list, max_size):
    """
    Return shuffled sublist of images to split between train, val and test sets.

    Parameters
    ----------
    plant_disease_folder_list: list
        The whole list of images for a category

    max_size:
        The maximal number of images retained for a category.

    Returns
    -------
    plant_disease_folder_list: list
        A shuffled sublist of images to split between train, val and test sets.

    img_number: into
        Number of images to splits between train, val and test sets.
    """
    np.random.shuffle(plant_disease_folder_list)
    img_number = min(len(plant_disease_folder_list), max_size)
    plant_disease_folder_list = plant_disease_folder_list[:img_number]

    return plant_disease_folder_list, img_number


def create_subsets(data_dir, test_dir, val_dir, train_dir, categories, distribs,
                    test_split, val_split, max_size):
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

    test_split, val_split: float
        Define the proportion of images going in the test and validation sets.

    max_size:
        The maximal number of images retained for a category.
    """
    try:
        logger.info(f" Loading images ...")
        root_dir = listdir(data_dir)

        train_images = [0] * len(categories)

        for plant_disease_folder in categories:
            source_directory = os.path.join(data_dir,plant_disease_folder)
            plant_disease_folder_list = listdir(source_directory)
            logger.info(f" Processing {plant_disease_folder} with {len(plant_disease_folder_list)} images")
            for image in plant_disease_folder_list:
                # remove .DS_Store from list
                if image == ".DS_Store" :
                    plant_disease_folder_list.remove(image)

            # Return shuffled sublist of images to split between train, val and test sets.
            plant_disease_folder_list, img_number = select_img(plant_disease_folder_list, max_size)

            test_images = int(img_number*test_split)
            valid_images = int(img_number*val_split)

            # Copy test image samples
            copy_subset(plant_disease_folder_list, 0, test_images, plant_disease_folder,
                        test_dir, source_directory)

            # Copy validation image samples
            copy_subset(plant_disease_folder_list, test_images, test_images+valid_images, plant_disease_folder,
                        val_dir, source_directory)

            # Preparation of the split train sets
            train_images = img_number-test_images-valid_images
            start_idx = test_images+valid_images
            i = 0
            for farms in distribs:
                train_subset = round(train_images * farms[plant_disease_folder])
                # Copy train image samples
                copy_subset(plant_disease_folder_list, start_idx, start_idx+train_subset,
                            plant_disease_folder, train_dir+str(i), source_directory)
                start_idx += train_subset
                i += 1

        logger.info(f" Image loading completed")
    except Exception as e:
        logger.info(f" Error : {format(e)}")


def run():
    cfg = open("PlantVillage_cfg.yaml", 'r')
    cfg_dict = yaml.load(cfg)

    data_dir = cfg_dict['orig_dataset']['path']
    base_dir = cfg_dict['output_dataset']['path']
    train_dir = base_dir+'train'
    val_dir = os.path.join(base_dir,'val')
    test_dir = os.path.join(base_dir,'test')

    categories = cfg_dict['included_categories']
    logger.info(f" \nDataset categories: \n{categories}\nNumber of categories: {len(categories)}\n")

    # Define distribution for various training subsets
    distributions = cfg_dict['distributions']
    distribs = distributions_list(distributions, categories)

    # Create the datasets
    test_split = cfg_dict['test_split']
    val_split = cfg_dict['val_split']
    max_size = cfg_dict['max_size']
    np.random.seed(42)
    create_directories(base_dir, train_dir, val_dir, test_dir, distribs)
    create_subsets(data_dir, test_dir, val_dir, train_dir, categories, distribs,
                        test_split, val_split, max_size)


if __name__ == '__main__':
    run()
