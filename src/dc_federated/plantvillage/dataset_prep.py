"""
Simple runner to prepare subsets of the PlantVillage dataset for running FedAvg workers.
Create train, test and validation sets for training a classifier for leaf diseases.

Create as many subsets as there needed for the training and validation data, with predefined distributions for the training sets between the various workers.

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


def clean_listdir(categories):
    """
    Remove .DS_Store from a list of directories entries.
    """
    for cat in categories :
        # remove .DS_Store from list
        if cat == ".DS_Store" :
            categories.remove(cat)
    return categories


def remove_cat(excluded_categories, categories):
    """
    Remove excluded categories from the categories list.
    """
    for cat in excluded_categories:
        categories.remove(cat)
    return categories


def distributions_list(distributions, categories): ####### Make a class #########
    """
    Remove .DS_Store from a list of directories entries.
    """
    distribs = []
    for farm_distrib in distributions:
        distribs.append(dict(zip(categories, farm_distrib)))
    return distribs


def create_directories(BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, distribs):
    """
    Create the directories to store the training subsets and validation
    and test sets.
    """
    # Remove existing datasets and create the new dataset directory
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    path = Path(BASE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    # Create the train, val and test directories
    for i in range(len(distribs)):
        Path(TRAIN_DIR+str(i)+'/').mkdir(parents=True, exist_ok=True)
    Path(VAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True)


def copy_image(directory, category, image, source):
    """
    Copy an image file
    """
    path = directory+"/"+category
    if not os.path.exists(path+"/"+image):
        shutil.copyfile(source+'/'+image, path+"/"+image)


def copy_subset(list_img, bound_inf, bound_sup, category, directory, source_directory):
    """
    Create the target directory if it does not exist.
    Copy the images of a category for a subset.
    """
    path = directory+"/"+category
    if not os.path.exists(path):
        os.mkdir(path)
    for image in list_img[bound_inf:bound_sup]:
        if image.endswith(".jpg") == True or image.endswith(".JPG") == True:
            copy_image(directory, category, image, source_directory)


def create_subsets(DATA_DIR, TEST_DIR, VAL_DIR, TRAIN_DIR, categories, distribs):
    """
    Create the different subsets for original data for the PlantVillage dataset.
    """
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(DATA_DIR)

        train_images = [0] * len(categories)

        for plant_disease_folder in categories:
            plant_disease_folder_list = listdir("{}/{}".format(DATA_DIR,plant_disease_folder))
            print("[INFO] Processing {} with {} images".format(plant_disease_folder, len(plant_disease_folder_list)))
            for image in plant_disease_folder_list:
                # remove .DS_Store from list
                if image == ".DS_Store" :
                    plant_disease_folder_list.remove(image)

            source_directory = DATA_DIR+'/'+plant_disease_folder
            test_images = len(plant_disease_folder_list)//10
            valid_images = len(plant_disease_folder_list)//6

            # Copy test image samples
            copy_subset(plant_disease_folder_list, 0, test_images, plant_disease_folder,
                        TEST_DIR, source_directory)

            # Copy validation image samples
            copy_subset(plant_disease_folder_list, -valid_images, -1, plant_disease_folder,
                        VAL_DIR, source_directory)

            # Preparation of the split train sets
            train_images = min(len(plant_disease_folder_list)-test_images-valid_images, 1200)
            start_idx = test_images
            i = 0
            for farms in distribs:
                train_subset = round(train_images * farms[plant_disease_folder])
                # Copy train image samples
                copy_subset(plant_disease_folder_list, start_idx, start_idx+train_subset,
                            plant_disease_folder, TRAIN_DIR+str(i)+'/', source_directory)
                start_idx += train_subset
                i += 1


        print("[INFO] Image loading completed")
    except Exception as e:
        print("Error : {}">format(e))


if __name__ == '__main__':
    cfg = open("PlantVillage_cfg.yaml", 'r')
    cfg_dict = yaml.load(cfg)

    DATA_DIR = cfg_dict['orig_dataset']['path']
    BASE_DIR = cfg_dict['output_dataset']['path']
    TRAIN_DIR = BASE_DIR+'train'
    VAL_DIR = BASE_DIR+'val/'
    TEST_DIR = BASE_DIR+'test/'

    # List the dataset categories
    categories = clean_listdir(listdir(DATA_DIR))
    print('Dataset categories: \n{}\n\nNumber of categories: {}'.format(categories, len(categories)))

    # Select categories to remove from the analysis
    excluded_categories = cfg_dict['excluded_categories']

    categories = remove_cat(excluded_categories, categories)
    print('Dataset categories: \n{}\n\nNumber of categories: {}'.format(categories, len(categories)))

    # Define distribution for various training subsets
    distributions = cfg_dict['distributions']
    distribs = distributions_list(distributions, categories)

    # Create the datasets
    create_directories(BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, distribs)
    create_subsets(DATA_DIR, TEST_DIR, VAL_DIR, TRAIN_DIR, categories, distribs)
