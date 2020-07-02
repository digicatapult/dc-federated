## Federated Learning with PlantVillage dataset

This document will show you how to run the PlantVillage dataset federated learning example included in the repo in a fully distributed fashion. In particular, it will show you:
- how to start the server
- how to start a desired number different workers in one or more machines
- run the FedAvg algorithm to solve the PlantVillage machine vision problem.

### The Problem

The PlantVillage dataset consists of images of diseased leaves for various kinds of crops and the diseases that each image corresponds to. To use this data for federated learning case, we choose a subset of the whole set and split the data into 4 datasets with different distributions (which can be specified via a configuration file) each corresponding to a different worker. As in the standard federated learning setting the goal is to learn a model at the server that uses all the data in the workers.


## Running the Example

We assume that you are running on Linux or Mac OS. You will need to translate the instructions for Windows, which should be quite similar. Follow the steps below to run the example.

### Setting up

Clone and install the repo in the machines where you want to deploy the servers and the workers (as described in `docs/getting_started.md`). *You can skip this step if you are running everything on the same machine.* Open a terminal for the server, and one terminal for each of the three workers.



### Preparing the datasets

In the following we assume you have cloned the repo in the folder `dc_fed_lib`. The original PlantVillage dataset should be stored in the folder `dc_fed_lib`, and can be obtained by running:
```bash
(venv_dc_federated)> git clone https://github.com/spMohanty/PlantVillage-Dataset
```

First navigate to `dc_fed_lib/src/dc_federated/examples/plantvillage` then use the configuration file `PlantVillage_cfg.yaml` to define how to split the dataset in multiple subsets and with customized distributions:
* target directory for the train, validation and test data,
* PlantVillage dataset categories to use as a list,
* PlantVillage dataset distributions per worker,
* number of subsets.

Apply the parameters from `PlantVillage_cfg.yaml` to create data subsets using:
```bash
(venv_dc_federated)> python dataset_prep.py
```

### Running the demo
The demo uses the pytorch implementation of the  [MobileNetv2](https://arxiv.org/abs/1801.04381) as base model. Use the following steps to run the server and the worker.

#### Server

In the terminal for the server, `cd` into < plant-village folder >  and start the server using:
```bash
(venv_dc_federated)> python plant_fed_avg_server.py
```
Use the configuration file `PlantVillage_cfg.yaml` to define datasets location, where to save model checkpoints and the expected number of update per iteration. Optionally, the training and validation data can be supplied as argument using:
```bash
(venv_dc_federated)> python plant_fed_avg_server.py --train-data-path <train-data-path> --validation-data-path <validation-data-path> --checkpoint-path <checkpoint-path> --update-lim <v>
```
where `v` is the number of updates per iteration. 

#### Workers

By default, the projects expect 4 workers. `cd` into the folder containg the stuff. Start as many workers as required using:
```bash
(venv_dc_federated)> python plant_fed_avg_worker.py --server-host-ip 192.124.1.177 --server-port 8080 --worker-id <v>
```
where `v` is between `0` to `3`. Use the configuration file `PlantVillage_cfg.yaml` to define datasets location and the hyperparameters for training the models. Optionally, the training and validation data can be supplied as argument using:
```bash
(venv_dc_federated)> python plant_fed_avg_worker.py --server-host-ip 192.124.1.177 --server-port 8080 --worker-id [worker id] --train-data-path [path] --validation-data-path [path]
```
